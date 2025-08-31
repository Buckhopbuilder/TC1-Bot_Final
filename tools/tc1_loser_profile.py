#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

def coerce_trade_id(s):
    # robust normalization so features/trades always join
    try:
        # many IDs are small ints; some are strings
        return pd.to_numeric(s, errors="ignore")
    except Exception:
        return s

def find_or_build_outcome(tr):
    """
    Return a Series of 'win'/'loss' per trade_id.
    Priority:
      1) existing columns: outcome/result/label (win/loss), or y_true (1/0)
      2) reconstruct via side + exit_price vs tp_price/sl_price
    """
    cols = {c.lower(): c for c in tr.columns}
    # 1) direct text labels
    for name in ("outcome", "result", "label"):
        if name in cols:
            s = tr[cols[name]].astype(str).str.lower().str.strip()
            # normalize to win/loss
            s = s.replace({"1":"win","0":"loss","true":"win","false":"loss"})
            s = s.where(s.isin(["win","loss"]), np.nan)
            if s.notna().any():
                return s
    # 2) y_true as 1/0
    if "y_true" in cols:
        yt = pd.to_numeric(tr[cols["y_true"]], errors="coerce")
        if yt.notna().any():
            return yt.map({1:"win", 0:"loss"})
    # 3) reconstruct from prices
    needed = ["side","exit_price","tp_price","sl_price"]
    if all(n in cols for n in needed):
        side = tr[cols["side"]].astype(str).str.lower()
        ex   = pd.to_numeric(tr[cols["exit_price"]], errors="coerce")
        tp   = pd.to_numeric(tr[cols["tp_price"]],  errors="coerce")
        sl   = pd.to_numeric(tr[cols["sl_price"]],  errors="coerce")
        # decide win/loss by which level exit hit (best-effort)
        win_long  = (side.str.startswith("l")) & (ex >= tp - 1e-12)
        loss_long = (side.str.startswith("l")) & (ex <= sl + 1e-12)
        win_short = (side.str.startswith("s")) & (ex <= tp + 1e-12)
        loss_short= (side.str.startswith("s")) & (ex >= sl - 1e-12)
        out = pd.Series(index=tr.index, dtype="object")
        out[win_long | win_short] = "win"
        out[loss_long| loss_short]= "loss"
        return out
    # otherwise None
    return pd.Series(index=tr.index, dtype="object")

def main():
    ap = argparse.ArgumentParser(description="Profile winners vs losers for numeric and categorical features.")
    ap.add_argument("--trades",   default="reports/trades/trades.csv")
    ap.add_argument("--features", default="reports/features/features_at_entry_plus.csv")
    ap.add_argument("--outdir",   default="reports/eda/loser_profile_plus")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Load
    tr = pd.read_csv(args.trades)
    fe = pd.read_csv(args.features)

    # Normalize join key
    for df in (tr, fe):
        if "trade_id" not in df.columns:
            raise SystemExit("Missing 'trade_id' in one of the inputs.")
        df["trade_id"] = coerce_trade_id(df["trade_id"])

    # Build/locate outcome
    outcome = find_or_build_outcome(tr)
    if outcome.isna().all():
        # last resort: try reconstruct if we have minimal columns
        raise SystemExit("Could not locate or reconstruct outcome from trades.csv. "
                         "Please ensure one of: 'outcome'/'result'/'label', 'y_true', "
                         "or columns side+exit_price+tp_price+sl_price exist.")
    tr["_outcome_norm"] = outcome.fillna("unknown")

    # Merge
    df = fe.merge(tr[["trade_id","_outcome_norm"]], on="trade_id", how="left")
    if df["_outcome_norm"].isna().all():
        raise SystemExit("Merge produced no outcomes. Check that trade_id types match between files.")

    # Binary labels
    y = (df["_outcome_norm"].astype(str)=="win").astype(int)

    # Split numeric vs categorical (ignore time columns)
    num_cols = df.select_dtypes(include=[np.number]).columns.drop(["trade_id"], errors="ignore")
    drop_cat = {"trade_id","_outcome_norm","fill_dt"}
    cat_cols = [c for c in df.columns if c not in num_cols and c not in drop_cat]

    rows_num = []
    for c in num_cols:
        if c in ("RR_target",): continue
        x = df[c]
        if x.isna().all(): continue
        mu_win, mu_loss = x[y==1].mean(), x[y==0].mean()
        d = (mu_win - mu_loss) / (x.std(ddof=0) + 1e-9)
        try:
            auc = roc_auc_score(y, x.fillna(x.median()))
        except Exception:
            auc = np.nan
        rows_num.append(dict(feature=c, mean_win=mu_win, mean_loss=mu_loss, cohens_d=d, auroc=auc))

    rows_cat = []
    for c in cat_cols:
        vc = df.groupby([c,"_outcome_norm"]).size().unstack(fill_value=0)
        if not {"win","loss"}.issubset(vc.columns): continue
        wr = vc["win"] / (vc["win"]+vc["loss"]).replace(0,np.nan)
        lift = (wr.max() - wr.min()) if wr.notna().any() else 0.0
        rows_cat.append(dict(feature=c, levels=int(len(vc)), winrates=wr.to_dict(), lift=float(lift)))

    pd.DataFrame(rows_num).sort_values("cohens_d", key=lambda x: x.abs(), ascending=False)\
        .to_csv(outdir/"numeric_features_ranked.csv", index=False)
    pd.DataFrame(rows_cat).sort_values("lift", ascending=False)\
        .to_csv(outdir/"categorical_features_ranked.csv", index=False)

    # Quick summary + diagnostics
    with open(outdir/"summary.txt","w") as f:
        # outcome diagnostics
        wins = int((df["_outcome_norm"]=="win").sum()); losses = int((df["_outcome_norm"]=="loss").sum())
        unknowns = int((df["_outcome_norm"]=="unknown").sum())
        f.write(f"OUTCOME DIAGNOSTICS: wins={wins} losses={losses} unknown={unknowns}\n")
        f.write("Outcome source: ")
        if "outcome" in tr.columns: f.write("trades.outcome\n")
        elif "result" in tr.columns: f.write("trades.result\n")
        elif "label" in tr.columns: f.write("trades.label\n")
        elif "y_true" in tr.columns: f.write("trades.y_true\n")
        elif set(["side","exit_price","tp_price","sl_price"]).issubset(tr.columns): f.write("reconstructed from prices\n")
        else: f.write("mixed/partial\n")

        f.write("\n=== Numeric features (top by effect size) ===\n")
        for r in sorted(rows_num, key=lambda r: abs(r["cohens_d"]), reverse=True)[:20]:
            f.write(f"{r['feature']:28s} d={r['cohens_d']:.3f} auroc={r['auroc']:.3f}\n")
        f.write("\n=== Categorical features (top by lift) ===\n")
        for r in sorted(rows_cat, key=lambda r: r["lift"], reverse=True)[:20]:
            f.write(f"{r['feature']:28s} lift={r['lift']:.3f} levels={r['levels']}\n")

    print(f"[DONE] Wrote profiling results → {outdir}")

if __name__ == "__main__":
    main()
