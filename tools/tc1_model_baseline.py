#!/usr/bin/env python3
import os, argparse
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

RANDOM_STATE = 42

DROP_NUMERIC = {
    "trade_id","A_time","B_time","fill_time","exit_time","exit_price",
    "bars_in_trade","ambiguous",
    "A_price","B_price","entry_price","sl_price","tp_price","y"
}

def load_data(trades_fp, feats_fp):
    tr = pd.read_csv(trades_fp)
    fe = pd.read_csv(feats_fp)
    tr = tr[tr["outcome"].isin(["win","loss"])].copy()
    tr["y"] = (tr["outcome"]=="win").astype(int)
    # drop duplicate cols from features before merge
    drop_dupes = ["symbol","side","leg_id","entry_price","sl_price","tp_price"]
    fe = fe.drop(columns=[c for c in drop_dupes if c in fe.columns])
    # merge on trade_id only
    df = tr.merge(fe, on="trade_id", how="inner")
    df["fill_dt"] = pd.to_datetime(df["fill_time"], unit="ms", utc=True)
    df = df.sort_values("fill_dt").reset_index(drop=True)
    return df

def make_Xy(df):
    num = df.select_dtypes(include=[np.number]).copy()
    keep_cols = [c for c in num.columns if c not in DROP_NUMERIC]
    X = num[keep_cols].copy()
    y = df["y"].astype(int).values
    rrt = df["RR_target"].astype(float).values
    return X, y, rrt, keep_cols

def time_split(df, test_frac=0.2):
    n = len(df)
    split_idx = int(np.floor(n * (1.0 - test_frac)))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

def summarize_top(pred, label="logreg"):
    # Baseline
    base_wr = pred["y_true"].mean()
    base_exp = np.mean(np.where(pred["y_true"]==1, pred["RR_target"], -1))

    # Deciles
    dec = pred.copy()
    dec["prob_decile"] = pd.qcut(dec["y_prob"], 10, labels=False, duplicates="drop")
    dec_grp = dec.groupby("prob_decile", group_keys=False).apply(lambda g: pd.Series({
        "n": len(g),
        "winrate": g["y_true"].mean(),
        "expectancy_R_obs": np.mean(np.where(g["y_true"]==1, g["RR_target"], -1)),
        "expected_R_pred": g["expected_R_pred"].mean(),
        "prob_mean": g["y_prob"].mean()
    })).reset_index().sort_values("prob_decile", ascending=False)

    # Top decile row
    top = dec_grp.iloc[0]
    lift = (top["winrate"] / base_wr) if base_wr > 0 else np.nan

    # Threshold sweep quick picks
    def policy(q):
        thr = np.quantile(pred["y_prob"], q)
        sel = pred[pred["y_prob"] >= thr]
        if len(sel)==0:
            return q, 0, np.nan, np.nan
        wr = sel["y_true"].mean()
        exp_obs = np.mean(np.where(sel["y_true"]==1, sel["RR_target"], -1))
        return q, len(sel), wr, exp_obs

    q20 = policy(0.80)  # top 20%
    q30 = policy(0.70)  # top 30%
    q40 = policy(0.60)  # top 40%

    print(f"\n=== Triple-barrel reality check [{label}] ===")
    print(f"#1 LIFT — Top decile vs baseline:")
    print(f"    Baseline win-rate: {base_wr:.4f} | Top decile: {top['winrate']:.4f} | Lift: {lift:.2f}x")
    print(f"#2 EDGE — Expectancy (R):")
    print(f"    Baseline: {base_exp:.4f} | Top decile: {top['expectancy_R_obs']:.4f}")
    print(f"    Policy top 20%: n={q20[1]} wr={q20[2]:.4f} expR={q20[3]:.4f}")
    print(f"    Policy top 30%: n={q30[1]} wr={q30[2]:.4f} expR={q30[3]:.4f}")
    print(f"    Policy top 40%: n={q40[1]} wr={q40[2]:.4f} expR={q40[3]:.4f}")
    print(f"#3 CALIBRATION — (mean predicted p) should trend with (actual winrate) by decile:")
    print(dec_grp[["prob_decile","n","prob_mean","winrate"]].head(5).to_string(index=False))

    return dec_grp

def fit_eval_model(name, pipe, Xtr, ytr, Xte, yte, rrt_te, test_meta, outdir):
    pipe.fit(Xtr, ytr)
    prob = pipe.predict_proba(Xte)[:,1]

    # metrics
    auroc = roc_auc_score(yte, prob)
    aupr  = average_precision_score(yte, prob)
    brier = brier_score_loss(yte, prob)
    try:
        ll = log_loss(yte, np.c_[1-prob, prob])
    except Exception:
        ll = np.nan

    # expected R per trade (predicted) on test
    exp_R_pred = prob * rrt_te + (1 - prob) * (-1)

    # Predictions frame for outputs
    pred = test_meta.copy()
    pred["y_true"] = yte
    pred["y_prob"] = prob
    pred["expected_R_pred"] = exp_R_pred

    # Deciles table
    dec_grp = pred.copy()
    dec_grp["prob_decile"] = pd.qcut(dec_grp["y_prob"], 10, labels=False, duplicates="drop")
    dec_grp = dec_grp.groupby("prob_decile", group_keys=False).apply(lambda g: pd.Series({
        "n": len(g),
        "winrate": g["y_true"].mean(),
        "expectancy_R_obs": np.mean(np.where(g["y_true"]==1, g["RR_target"], -1)),
        "expected_R_pred": g["expected_R_pred"].mean(),
        "prob_mean": g["y_prob"].mean()
    })).reset_index().sort_values("prob_decile", ascending=False)

    # Threshold sweep (top-% policy)
    qs = np.arange(0.50, 0.96, 0.05)
    rows = []
    for q in qs:
        thr = np.quantile(prob, q)
        sel = pred[pred["y_prob"] >= thr]
        if len(sel)==0:
            rows.append({"quantile": q, "n": 0, "winrate": np.nan, "expectancy_R_obs": np.nan,
                         "expected_R_pred": np.nan, "threshold": thr})
            continue
        wr = sel["y_true"].mean()
        expR_obs = np.mean(np.where(sel["y_true"]==1, sel["RR_target"], -1))
        rows.append({"quantile": q, "n": len(sel), "winrate": wr,
                     "expectancy_R_obs": expR_obs,
                     "expected_R_pred": sel["expected_R_pred"].mean(),
                     "threshold": thr})
    sweep = pd.DataFrame(rows).sort_values("quantile", ascending=False)

    # Save artifacts
    os.makedirs(outdir, exist_ok=True)
    pred.to_csv(os.path.join(outdir, f"predictions_{name}.csv"), index=False)
    dec_grp.to_csv(os.path.join(outdir, f"deciles_{name}.csv"), index=False)
    sweep.to_csv(os.path.join(outdir, f"threshold_sweep_{name}.csv"), index=False)

    # Summary file
    with open(os.path.join(outdir, f"summary_{name}.txt"), "w") as f:
        f.write(f"Model: {name}\n")
        f.write(f"Test size: {len(yte)}\n")
        f.write(f"AUROC: {auroc:.4f}\nAverage Precision: {aupr:.4f}\nBrier: {brier:.4f}\nLogLoss: {ll:.4f}\n")
        f.write("\nDeciles (top=highest prob):\n")
        f.write(dec_grp.to_string(index=False))
        f.write("\n\nThreshold sweep (quantile, trade only >= threshold):\n")
        f.write(sweep.to_string(index=False))

    print(f"[{name}] AUROC={auroc:.4f}  AUPR={aupr:.4f}  Brier={brier:.4f}  LogLoss={ll:.4f}")

    # Console: triple-barrel reality check
    summarize_top(pred, label=name)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", default="reports/trades/trades.csv")
    ap.add_argument("--features", default="reports/features/features_at_entry.csv")
    ap.add_argument("--outdir", default="reports/ml")
    ap.add_argument("--test_frac", type=float, default=0.2)
    args = ap.parse_args()

    df = load_data(args.trades, args.features)
    X, y, rrt, cols = make_Xy(df)

    # Time split
    df_train, df_test = time_split(df, test_frac=args.test_frac)
    Xtr, ytr, rrt_tr, _ = make_Xy(df_train)
    Xte, yte, rrt_te, _ = make_Xy(df_test)

    # Meta for predictions (test set)
    meta_cols = ["trade_id","symbol","side","fill_time","RR_target"]
    test_meta = df_test[meta_cols].copy()

    models = {
        "logreg": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(max_iter=1000, solver="lbfgs", class_weight="balanced"))
        ]),
        "rf": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=None, n_jobs=-1, random_state=RANDOM_STATE,
                class_weight="balanced_subsample", min_samples_leaf=3))
        ])
    }
    for name, pipe in models.items():
        fit_eval_model(name, pipe, Xtr, ytr, Xte, yte, rrt_te, test_meta, args.outdir)

if __name__ == "__main__":
    main()
