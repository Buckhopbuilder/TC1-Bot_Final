#!/usr/bin/env python3
import argparse, re, pandas as pd
from pathlib import Path

LEAKY_EXACT = {
    "y","y_true","outcome","result","label",
    "RR_target","exit_price","exit_time","exit_time_iso","exit_dt","exit_bar",
    "bars_in_trade","ambiguous","tp_hit","sl_hit","tp_reached","sl_reached",
    "entry_price","sl_price","tp_price",
    "fill_time","fill_dt","fill_time_iso",
}
LEAKY_PREFIXES = [r"^exit_", r"^tp_", r"^sl_", r"^label_", r"^target_", r"^pnl_", r"^reward_"]
LEAKY_SUBSTR = ["outcome","result","label","reward","pnl","takeprofit","stoploss"]
META_EXACT = {"symbol","side","leg_id"}

def is_leaky(c):
    c = c.strip()
    if c in LEAKY_EXACT or c in META_EXACT: return True
    if any(re.search(p, c) for p in LEAKY_PREFIXES): return True
    if any(s in c for s in LEAKY_SUBSTR): return True
    return False

def main():
    ap = argparse.ArgumentParser(description="Remove label/forward-looking leakage from features CSV.")
    ap.add_argument("--in_csv",  default="reports/features/features_at_entry_plus.csv")
    ap.add_argument("--out_csv", default="reports/features/features_at_entry_clean.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    if "trade_id" not in df.columns:
        raise SystemExit("Input features must include 'trade_id'.")

    drop_cols = [c for c in df.columns if c != "trade_id" and is_leaky(c)]
    kept_cols = [c for c in df.columns if c not in drop_cols]

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df[kept_cols].to_csv(args.out_csv, index=False)

    print(f"[CLEAN] Read:  {args.in_csv}")
    print(f"[CLEAN] Kept:  {len(kept_cols)} columns (incl. trade_id)")
    print(f"[CLEAN] Dropped {len(drop_cols)} columns:")
    for c in sorted(drop_cols): print(f"  - {c}")
    print(f"[DONE] Clean features → {args.out_csv}")

if __name__ == "__main__":
    main()
