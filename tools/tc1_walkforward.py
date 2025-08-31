#!/usr/bin/env python3
import os, argparse
import pandas as pd, numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

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
    # drop duplicate cols from features
    drop_dupes = ["symbol","side","leg_id","entry_price","sl_price","tp_price"]
    fe = fe.drop(columns=[c for c in drop_dupes if c in fe.columns])
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

def run_fold(train_df, test_df):
    Xtr, ytr, _, _ = make_Xy(train_df)
    Xte, yte, rrt_te, _ = make_Xy(test_df)
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler(with_mean=True, with_std=True)),
        ("clf", RandomForestClassifier(
            n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE,
            class_weight="balanced_subsample", min_samples_leaf=3))
    ])
    pipe.fit(Xtr, ytr)
    prob = pipe.predict_proba(Xte)[:,1]
    auroc = roc_auc_score(yte, prob) if len(np.unique(yte))>1 else np.nan
    # expectancy (observed)
    expR = np.mean(np.where(yte==1, rrt_te, -1))
    wr = yte.mean()
    return pd.DataFrame({
        "trade_id": test_df["trade_id"],
        "fill_dt": test_df["fill_dt"],
        "symbol": test_df["symbol"],
        "y_true": yte,
        "y_prob": prob,
        "RR_target": rrt_te
    }), dict(trades=len(yte), winrate=wr, expectancy=expR, auroc=auroc)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", default="reports/trades/trades.csv")
    ap.add_argument("--features", default="reports/features/features_at_entry.csv")
    ap.add_argument("--out_preds", default="reports/ml/walkforward_preds.csv")
    ap.add_argument("--out_summary", default="reports/ml/walkforward_summary.csv")
    ap.add_argument("--years", nargs="*", type=int, help="Optional list of test years (e.g., 2021 2022 2023)")
    args = ap.parse_args()

    df = load_data(args.trades, args.features)
    df["year"] = df["fill_dt"].dt.year

    preds_all = []
    rows = []

    test_years = args.years if args.years else sorted(df["year"].unique())[1:]  # skip first year
    for y in test_years:
        train_df = df[df["year"] < y]
        test_df  = df[df["year"] == y]
        if len(train_df)<500 or len(test_df)<200: 
            continue
        preds, stats = run_fold(train_df, test_df)
        preds["fold"] = y
        preds_all.append(preds)
        rows.append({"year": y, **stats})

    if preds_all:
        allp = pd.concat(preds_all, ignore_index=True)
        allp.to_csv(args.out_preds, index=False)
        pd.DataFrame(rows).to_csv(args.out_summary, index=False)
        print("=== Walk-forward summary ===")
        print(pd.DataFrame(rows).round(4))
        print(f"\nSaved predictions → {args.out_preds}")
        print(f"Saved summary     → {args.out_summary}")
    else:
        print("No folds run; check data/year coverage.")

if __name__ == "__main__":
    main()
