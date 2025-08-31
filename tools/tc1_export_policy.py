#!/usr/bin/env python3
import argparse, pandas as pd
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="Export filtered TC1 trades by model probability.")
    ap.add_argument("--pred", default="reports/ml/predictions_rf.csv", help="Predictions CSV from model")
    ap.add_argument("--trades", default="reports/trades/trades.csv", help="Raw trades CSV")
    ap.add_argument("--out", default="reports/ml/policy_trades.csv", help="Output CSV path")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--top_frac", type=float, help="Top fraction by prob (e.g., 0.30 for top 30%)")
    group.add_argument("--prob_thr", type=float, help="Absolute prob threshold (e.g., 0.28)")
    ap.add_argument("--min_prob", type=float, default=0.0, help="Optional floor on probability")
    args = ap.parse_args()

    pred = pd.read_csv(args.pred)
    tr = pd.read_csv(args.trades)

    # Choose threshold
    if args.top_frac is not None:
        thr = pred["y_prob"].quantile(1.0 - args.top_frac)
        policy_desc = f"top_frac={args.top_frac:.2f}"
    else:
        thr = float(args.prob_thr)
        policy_desc = f"prob_thr={thr:.4f}"

    sel = pred[pred["y_prob"] >= thr].copy()
    if args.min_prob > 0:
        sel = sel[sel["y_prob"] >= args.min_prob]

    # Rank by probability (1 = best)
    sel = sel.sort_values("y_prob", ascending=False).reset_index(drop=True)
    sel["policy_rank"] = sel.index + 1
    sel["policy_threshold"] = thr
    sel["policy_rule"] = policy_desc

    # Pull only extra fields from trades to avoid column overlap
    tr_keep = [
        "trade_id","A_time","B_time","exit_time",
        "A_price","B_price","entry_price","sl_price","tp_price",
        "outcome"  # << include realized outcome
    ]
    tr_small = tr[tr_keep].copy()

    # Merge safely
    df = sel.merge(tr_small, on="trade_id", how="left")

    # Pretty timestamps
    for col in ["A_time","B_time","fill_time","exit_time"]:
        if col in df.columns:
            df[col + "_iso"] = pd.to_datetime(df[col], unit="ms", utc=True)

    # Final column order
    cols = [
        "trade_id","symbol","side",
        "A_time_iso","B_time_iso","fill_time_iso","exit_time_iso",
        "A_price","B_price","entry_price","sl_price","tp_price",
        "RR_target","outcome",          # realized
        "y_prob","y_true",              # model preds & label (if present)
        "policy_rank","policy_threshold","policy_rule"
    ]
    cols = [c for c in cols if c in df.columns]
    out = df[cols]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"[DONE] Exported {len(out)} trades ≥ threshold={thr:.4f} → {args.out}")

if __name__ == "__main__":
    main()
