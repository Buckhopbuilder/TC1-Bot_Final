#!/usr/bin/env python3
import argparse, pandas as pd, numpy as np
from pathlib import Path

def parse_hhmm(s):
    h,m = map(int, s.split(":")); return h, m

def main():
    ap = argparse.ArgumentParser(description="Filter out trades whose fill time falls in specified UTC windows.")
    ap.add_argument("--inp", required=True, help="Input trades CSV (e.g., policy_trades.csv)")
    ap.add_argument("--out", required=True, help="Output filtered CSV")
    # Default: London 08:00–10:00, New York 13:00–16:00 (UTC)
    ap.add_argument("--windows", nargs="+", default=["08:00-10:00","13:00-16:00"],
                    help='List of HH:MM-HH:MM UTC windows to EXCLUDE')
    ap.add_argument("--weekday_only", action="store_true", help="Also drop weekends (Sat/Sun)")
    args = ap.parse_args()

    df = pd.read_csv(args.inp)

    # Determine fill timestamp column (accept fill_dt, fill_time, fill_time_iso)
    if "fill_dt" in df.columns:
        ts = pd.to_datetime(df["fill_dt"], utc=True)
    elif "fill_time" in df.columns:
        ts = pd.to_datetime(df["fill_time"], unit="ms", utc=True)
    elif "fill_time_iso" in df.columns:
        ts = pd.to_datetime(df["fill_time_iso"], utc=True)
    else:
        raise ValueError("No fill_dt, fill_time, or fill_time_iso column found.")

    df["_fill_dt"] = ts
    df["_dow"] = df["_fill_dt"].dt.dayofweek  # 0=Mon..6=Sun
    minutes = df["_fill_dt"].dt.hour * 60 + df["_fill_dt"].dt.minute

    # Build exclusion mask
    excl = np.zeros(len(df), dtype=bool)
    for w in args.windows:
        try:
            s,e = w.split("-")
            sh, sm = parse_hhmm(s); eh, em = parse_hhmm(e)
            start_min, end_min = sh*60 + sm, eh*60 + em
            excl |= (minutes >= start_min) & (minutes < end_min)
        except Exception as ex:
            raise ValueError(f"Bad window format '{w}'. Use HH:MM-HH:MM.") from ex

    if args.weekday_only:
        excl |= df["_dow"] >= 5  # Sat/Sun

    removed = int(excl.sum()); kept = len(df) - removed
    out = df.loc[~excl].drop(columns=[c for c in ["_fill_dt","_dow"] if c in df.columns])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"[DONE] Kept {kept} trades, removed {removed} due to windows={args.windows}"
          + (", plus weekends" if args.weekday_only else "")
          + f" → {args.out}")
if __name__ == "__main__":
    main()
