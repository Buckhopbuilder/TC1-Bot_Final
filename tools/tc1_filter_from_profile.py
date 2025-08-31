#!/usr/bin/env python3
import argparse, os, numpy as np, pandas as pd
from pathlib import Path

def load_trades_auto():
    cand = [
        "reports/trades/trades_enriched.csv",
        "reports/trades/trades.csv",
    ]
    for p in cand:
        if os.path.exists(p):
            tr = pd.read_csv(p)
            # ensure timing cols
            if "fill_dt" not in tr.columns and "fill_time" in tr.columns:
                tr["fill_dt"] = pd.to_datetime(tr["fill_time"], unit="ms", utc=True)
            return tr, p
    raise SystemExit("No trades file found. Expected one of: reports/trades/trades_enriched.csv or reports/trades/trades.csv")

def main():
    ap = argparse.ArgumentParser(description="Filter a policy using top-|d| features with quantile thresholds; always enrich with trades.")
    ap.add_argument("--policy", required=True, help="CSV of candidate trades (must include trade_id).")
    ap.add_argument("--features", default="reports/features/features_at_entry_plus.csv")
    ap.add_argument("--profile",  default="reports/eda/loser_profile_plus/numeric_features_ranked.csv")
    ap.add_argument("--top_k", type=int, default=4)
    ap.add_argument("--quantiles", nargs="+", type=float, default=[0.6,0.7,0.8])
    ap.add_argument("--out_prefix", default="reports/ml/filters/auto")
    args = ap.parse_args()

    Path(Path(args.out_prefix).parent).mkdir(parents=True, exist_ok=True)

    pol  = pd.read_csv(args.policy)
    fe   = pd.read_csv(args.features)
    prof = pd.read_csv(args.profile).dropna(subset=["cohens_d"])
    prof = prof.reindex(prof["cohens_d"].abs().sort_values(ascending=False).index)[:args.top_k]

    # auto-load trades (enriched preferred)
    tr, tr_path = load_trades_auto()
    keep = [c for c in ["trade_id","symbol","side","fill_time","fill_dt","entry_price","sl_price","tp_price","RR_target","outcome"] if c in tr.columns]
    tr = tr[keep].copy()

    # build working frame
    df = pol.merge(fe, on="trade_id", how="left").merge(tr, on="trade_id", how="left")

    # Apply quantile rules (direction by sign of d)
    for q in args.quantiles:
        mask = pd.Series(True, index=df.index)
        rules = []
        for _, r in prof.iterrows():
            f = r["feature"]; d = float(r["cohens_d"])
            if f not in df.columns: continue
            x = pd.to_numeric(df[f], errors="coerce")
            if x.notna().sum() == 0: continue
            thr = np.nanquantile(x, q)
            if np.isnan(thr): continue
            if d >= 0:
                keep_mask = x >= thr
                rules.append(f"{f} >= q{int(q*100)} ({thr:.5g})")
            else:
                keep_mask = x <= thr
                rules.append(f"{f} <= q{int(q*100)} ({thr:.5g})")
            mask &= keep_mask

        out = df[mask].copy()

        # safe columns to write: start with whatever was in the original policy
        safe = [c for c in pol.columns if c in out.columns]
        # add common helpful fields if present
        for extra in ["symbol","side","fill_time","fill_dt","entry_price","sl_price","tp_price","RR_target","y_prob","outcome"]:
            if extra in out.columns and extra not in safe:
                safe.append(extra)
        if not safe:
            safe = ["trade_id"]

        out_file = f"{args.out_prefix}_top{args.top_k}_q{int(q*100)}.csv"
        out[safe].to_csv(out_file, index=False)
        kept = len(out)
        print(f"[DONE] {out_file}  kept={kept}  (rules: { ' | '.join(rules) if rules else 'none'})")

if __name__ == "__main__":
    main()
