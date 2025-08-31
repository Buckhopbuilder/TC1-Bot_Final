#!/usr/bin/env python3
import argparse, os, numpy as np, pandas as pd
from pathlib import Path
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

def load_trades_auto():
    for p in ["reports/trades/trades_enriched.csv","reports/trades/trades.csv"]:
        if os.path.exists(p):
            return pd.read_csv(p), p
    raise SystemExit("No trades file found. Expected reports/trades/trades_enriched.csv or reports/trades/trades.csv")

def ensure_outcome(tr):
    cols = {c.lower(): c for c in tr.columns}
    for name in ("outcome","result","label"):
        if name in cols:
            s = tr[cols[name]].astype(str).str.lower().str.strip()
            return s.replace({"1":"win","0":"loss","true":"win","false":"loss"})
    if "y_true" in cols:
        return pd.to_numeric(tr[cols["y_true"]], errors="coerce").map({1:"win",0:"loss"})
    need = {"side","exit_price","tp_price","sl_price"}
    if need.issubset(cols.keys()):
        side = tr[cols["side"]].astype(str).str.lower()
        ex   = pd.to_numeric(tr[cols["exit_price"]], errors="coerce")
        tp   = pd.to_numeric(tr[cols["tp_price"]],  errors="coerce")
        sl   = pd.to_numeric(tr[cols["sl_price"]],  errors="coerce")
        eps = 1e-10
        out = pd.Series("unknown", index=tr.index, dtype=object)
        mL = side.str.startswith("l"); mS = side.str.startswith("s")
        out[mL & (ex >= tp - eps)] = "win"
        out[mL & (ex <= sl + eps)] = "loss"
        out[mS & (ex <= tp + eps)] = "win"
        out[mS & (ex >= sl - eps)] = "loss"
        return out
    return pd.Series(index=tr.index, dtype=object)

def ensure_fill_time(tr):
    cols = {c.lower(): c for c in tr.columns}
    if "fill_time" in cols:
        return tr[cols["fill_time"]]
    for k in ("fill_time_iso","fill_dt","fill_datetime"):
        if k in cols:
            dt = pd.to_datetime(tr[cols[k]], utc=True, errors="coerce")
            return (dt.view("int64") // 10**6).astype("int64")
    raise SystemExit("Could not find fill_time/fill_time_iso/fill_dt in trades to build timeline.")

def pick_col(df, names, required=False, err_msg=""):
    for n in names:
        if n in df.columns:
            return df[n]
    for n in names:
        for suf in ("_y","_x"):
            nn = f"{n}{suf}"
            if nn in df.columns:
                return df[nn]
    if required:
        raise SystemExit(err_msg or f"Missing required column among: {names}")
    return None

def main():
    ap = argparse.ArgumentParser(description="Walk-forward LightGBM on features_plus (robust outcome/timestamps; hard label blocklist).")
    ap.add_argument("--features", default="reports/features/features_at_entry_clean.csv")
    ap.add_argument("--trades", default=None)
    ap.add_argument("--out_preds", default="reports/ml/lgbm_walkforward_preds.csv")
    ap.add_argument("--out_summary", default="reports/ml/lgbm_walkforward_summary.csv")
    ap.add_argument("--out_importance", default="reports/ml/lgbm_feature_importance.csv")
    args = ap.parse_args()

    fe = pd.read_csv(args.features)

    if args.trades and os.path.exists(args.trades):
        tr = pd.read_csv(args.trades); tr_path = args.trades
    else:
        tr, tr_path = load_trades_auto()

    tr = tr.copy()
    if "outcome" not in tr.columns:
        tr["outcome"] = ensure_outcome(tr)
    if "fill_time" not in tr.columns:
        tr["fill_time"] = ensure_fill_time(tr)

    keep = [c for c in ["trade_id","outcome","fill_time"] if c in tr.columns]
    tr_m = tr[keep].copy()

    df = fe.merge(tr_m, on="trade_id", how="left")

    outcome_series = pick_col(df, names=["outcome","result","label","y_true"], required=True,
                              err_msg="Failed to create outcome labels — check trades columns.")
    ytxt = outcome_series.astype(str).str.lower().str.strip().replace(
        {"1":"win","0":"loss","true":"win","false":"loss"}
    )
    df["y"] = (ytxt=="win").astype(int)

    fill_time_series = pick_col(df, names=["fill_time","fill_time_ms"])
    if fill_time_series is not None and pd.api.types.is_numeric_dtype(fill_time_series):
        df["fill_dt"] = pd.to_datetime(fill_time_series, unit="ms", utc=True)
    else:
        fill_dt_series = pick_col(df, names=["fill_dt","fill_time_iso"]) or pick_col(fe, names=["fill_dt","fill_time_iso"])
        if fill_dt_series is None:
            raise SystemExit("Could not find fill_time/fill_dt after merge.")
        df["fill_dt"] = pd.to_datetime(fill_dt_series, utc=True, errors="coerce")
    df["year"] = df["fill_dt"].dt.year

    # ---- Feature matrix (HARD BLOCKLIST) ----
    DROP_META = {"trade_id","symbol","side","fill_time","fill_dt","outcome","RR_target"}
    BAD_LABELS = {"y","outcome","result","label","RR_target"}  # never allow as features
    X_cols = [c for c in df.columns
              if c not in DROP_META and c not in BAD_LABELS and pd.api.types.is_numeric_dtype(df[c])]

    df = df.dropna(subset=["y"])
    preds, rows, importances = [], [], []

    years = sorted(df["year"].unique())
    for y in years:
        train = df[df["year"] < y].copy()
        test  = df[df["year"] == y].copy()
        if len(train) < 1000 or len(test) < 100:
            continue

        clf = LGBMClassifier(
            n_estimators=700,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            reg_alpha=0.1,
            min_child_samples=100,
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(train[X_cols], train["y"], eval_set=[(test[X_cols], test["y"])], eval_metric="auc")

        p = clf.predict_proba(test[X_cols])[:,1]
        test_idx = test.index
        preds.append(pd.DataFrame({"trade_id": df.loc[test_idx,"trade_id"].values,
                                   "year": y, "y_prob": p}, index=None))

        auc = roc_auc_score(df.loc[test_idx,"y"], p)
        rows.append(dict(year=y, trades=len(test), auroc=auc))
        importances.append(pd.DataFrame({
            "feature": X_cols,
            "gain": clf.booster_.feature_importance(importance_type="gain")
        }))

    Path("reports/ml").mkdir(parents=True, exist_ok=True)
    if preds:
        pd.concat(preds, ignore_index=True).to_csv(args.out_preds, index=False)
    if rows:
        pd.DataFrame(rows).to_csv(args.out_summary, index=False)
    if importances:
        (pd.concat(importances)
           .groupby("feature")["gain"].median()
           .sort_values(ascending=False)
           .reset_index()
           .to_csv(args.out_importance, index=False))

    print(f"[DONE] preds → {args.out_preds}")
    print(f"[DONE] summary → {args.out_summary}")
    print(f"[DONE] feature importance → {args.out_importance}")
    print(f"(merged trades from: {tr_path})")

if __name__ == "__main__":
    main()
