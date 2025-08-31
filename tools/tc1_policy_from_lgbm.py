#!/usr/bin/env python3
import argparse, numpy as np, pandas as pd
from pathlib import Path

def in_block_local(dt_utc, tz_name, start_h, start_m, dur_minutes):
    """
    dt_utc: pandas Series of tz-aware UTC datetimes
    Returns a boolean mask for times within [start, start+dur) in the given local tz.
    """
    local = dt_utc.dt.tz_convert(tz_name)
    day = local.dt.floor("D")
    start = day + pd.to_timedelta(start_h, unit="h") + pd.to_timedelta(start_m, unit="m")
    end = start + pd.to_timedelta(dur_minutes, unit="m")
    return (local >= start) & (local < end)

def main():
    ap = argparse.ArgumentParser(description="Build a policy from LGBM preds with session exclusions and per-day cap.")
    ap.add_argument("--preds", default="reports/ml/lgbm_walkforward_preds.csv")
    ap.add_argument("--trades", default="reports/trades/trades_enriched.csv")
    ap.add_argument("--out", default="reports/ml/lgbm_policy_topN.csv")
    ap.add_argument("--min_prob", type=float, default=0.0, help="Optional minimum probability threshold.")
    ap.add_argument("--per_day", type=int, default=4, help="Max trades per UTC day.")
    ap.add_argument("--include_sessions", action="store_true", help="If set, DO NOT exclude London/NY openings.")
    args = ap.parse_args()

    Path(Path(args.out).parent).mkdir(parents=True, exist_ok=True)

    P = pd.read_csv(args.preds)              # trade_id, year, y_prob
    T = pd.read_csv(args.trades)             # enriched: trade_id, symbol, side, fill_time, etc.
    if "fill_dt" not in T.columns and "fill_time" in T.columns:
        T["fill_dt"] = pd.to_datetime(T["fill_time"], unit="ms", utc=True)

    df = P.merge(T, on="trade_id", how="left")
    if "fill_dt" not in df.columns:
        raise SystemExit("No fill_dt/fill_time available after merge.")

    df["fill_dt"] = pd.to_datetime(df["fill_dt"], utc=True)

    # Optional prob floor
    if args.min_prob > 0:
        df = df[df["y_prob"] >= args.min_prob].copy()

    # Session exclusions (unless include_sessions)
    if not args.include_sessions:
        mask_lon = in_block_local(df["fill_dt"], "Europe/London", 8, 0, 120)     # 08:00–10:00 London
        mask_ny  = in_block_local(df["fill_dt"], "America/New_York", 9, 30, 180) # 09:30–12:30 New York
        df = df[~(mask_lon | mask_ny)].copy()

    # Cap per UTC day by highest probability
    df["utc_day"] = df["fill_dt"].dt.floor("D")
    df = df.sort_values(["utc_day","y_prob"], ascending=[True, False])
    df["rank_day"] = df.groupby("utc_day").cumcount() + 1
    df = df[df["rank_day"] <= args.per_day].copy()

    # Choose safe columns for backtester
    keep = [c for c in ["trade_id","symbol","side","fill_time","fill_dt","entry_price","sl_price","tp_price","y_prob"] if c in df.columns]
    if "trade_id" not in keep: keep = ["trade_id"] + keep
    df[keep].to_csv(args.out, index=False)
    print(f"[DONE] Policy → {args.out}  | trades={len(df)}  | min_prob={args.min_prob}  | per_day={args.per_day}  | sessions={'kept' if args.include_sessions else 'excluded'}")

if __name__ == "__main__":
    main()
