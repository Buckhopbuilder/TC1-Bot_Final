#!/usr/bin/env python3
import argparse, pandas as pd, numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--group", default="symbol_date", choices=["date","symbol_date"],
                    help="Grouping for sessions: 'date' (UTC date) or 'symbol_date' (per symbol per UTC date)")
    ap.add_argument("--min_prob", type=float, default=None, help="Optional y_prob threshold (e.g., 0.5)")
    args = ap.parse_args()

    df = pd.read_csv(args.policy_csv)

    # Parse session timestamp (prefer 'fill_dt', else 'fill_time' ms)
    if "fill_dt" in df.columns:
        ts = pd.to_datetime(df["fill_dt"], utc=True, errors="coerce")
    else:
        ts = pd.to_datetime(df["fill_time"], unit="ms", utc=True, errors="coerce")
    session_date = ts.dt.date

    if args.group == "date":
        session_key = session_date.astype(str)
    else:
        # symbol + date (default)
        session_key = df["symbol"].astype(str) + "|" + session_date.astype(str)

    df["_session"] = session_key

    # Optional: threshold by y_prob first
    if args.min_prob is not None and "y_prob" in df.columns:
        df = df[df["y_prob"] >= args.min_prob].copy()

    # For each session, keep top 4 by y_prob
    if "y_prob" not in df.columns:
        raise SystemExit("ERROR: y_prob column not found in policy CSV.")
    df = df.sort_values(["_session","y_prob","fill_time"], ascending=[True, False, True])
    top4 = df.groupby("_session", as_index=False, group_keys=False).head(4).copy()

    # Output
    top4.drop(columns=["_session"], errors="ignore").to_csv(args.out_csv, index=False)

    # Stats
    sessions = top4["_session"].nunique() if "_session" in top4.columns else np.nan
    print(f"[OK] Wrote {args.out_csv}")
    print(f" Total sessions kept: {sessions}")
    print(f" Total trades kept:   {len(top4)}")

if __name__ == "__main__":
    main()
