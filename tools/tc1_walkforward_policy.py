#!/usr/bin/env python3
import argparse, pandas as pd, numpy as np
from pathlib import Path

def summarize(df):
    n = len(df)
    if n == 0:
        return dict(trades=0, winrate=np.nan, expectancy=np.nan)
    wr = df["y_true"].mean()
    expR = np.mean(np.where(df["y_true"]==1, df["RR_target"], -1))
    return dict(trades=int(n), winrate=float(wr), expectancy=float(expR))

def main():
    ap = argparse.ArgumentParser(description="Apply a policy per year on walk-forward predictions.")
    ap.add_argument("--preds", default="reports/ml/walkforward_preds.csv")
    ap.add_argument("--out_summary", default="reports/ml/walkforward_policy_summary.csv")
    ap.add_argument("--out_trades", default="reports/ml/walkforward_policy_trades.csv")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--top_frac", type=float, help="Top fraction by prob per year (e.g., 0.30 for top 30%)")
    group.add_argument("--prob_thr", type=float, help="Absolute probability threshold (e.g., 0.28)")
    args = ap.parse_args()

    df = pd.read_csv(args.preds)
    # ensure timestamps & year
    if "fill_dt" in df.columns:
        df["fill_dt"] = pd.to_datetime(df["fill_dt"], utc=True)
    else:
        raise ValueError("walkforward_preds.csv must contain fill_dt")
    df["year"] = df["fill_dt"].dt.year

    rows = []
    outs = []
    for y, g in df.groupby("year"):
        if args.top_frac is not None:
            thr = g["y_prob"].quantile(1.0 - args.top_frac)
        else:
            thr = float(args.prob_thr)

        sel = g[g["y_prob"] >= thr].copy().sort_values("y_prob", ascending=False)
        stats = summarize(sel)
        rows.append({"year": int(y), "threshold": float(thr), **stats})

        if len(sel):
            sel["policy_year"] = int(y)
            sel["policy_threshold"] = thr
            sel["policy_rule"] = f"{'top_frac' if args.top_frac is not None else 'prob_thr'}=" + \
                                 (f"{args.top_frac:.2f}" if args.top_frac is not None else f"{thr:.4f}")
            outs.append(sel)

    # overall line (across all selected trades)
    if outs:
        all_sel = pd.concat(outs, ignore_index=True)
        all_stats = summarize(all_sel)
        rows.append({"year": "OVERALL", "threshold": np.nan, **all_stats})
        # save filtered trades
        Path(args.out_trades).parent.mkdir(parents=True, exist_ok=True)
        all_sel.to_csv(args.out_trades, index=False)
        print(f"[DONE] Saved filtered trades → {args.out_trades}")
    else:
        all_sel = pd.DataFrame()

    # save summary
    summ = pd.DataFrame(rows)
    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)
    summ.to_csv(args.out_summary, index=False)
    print("\n=== Walk-forward policy summary ===")
    if not summ.empty:
        print(summ.to_string(index=False))
    print(f"\n[DONE] Saved summary → {args.out_summary}")

if __name__ == "__main__":
    main()
