#!/usr/bin/env python3
import argparse, heapq, pandas as pd
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="Limit max concurrent open trades using fill_time (ms) and exit_time (ms).")
    ap.add_argument("--policy", required=True, help="CSV with candidate trades (must include trade_id, fill_time).")
    ap.add_argument("--trades", default="reports/trades/trades_enriched.csv", help="Trades file with exit_time.")
    ap.add_argument("--out", required=True, help="Output CSV with concurrency-limited trades.")
    ap.add_argument("--max_concurrent", type=int, default=2)
    args = ap.parse_args()

    Path(Path(args.out).parent).mkdir(parents=True, exist_ok=True)

    pol = pd.read_csv(args.policy)
    tr  = pd.read_csv(args.trades)

    # sanity checks
    for need in ["trade_id","fill_time"]:
        if need not in pol.columns:
            raise SystemExit(f"Policy missing required column: {need}")
    if "exit_time" not in tr.columns:
        raise SystemExit("Trades file missing required column: exit_time")

    # merge exit_time on trade_id
    keep_cols = ["trade_id","exit_time"]
    df = pol.merge(tr[keep_cols], on="trade_id", how="left", validate="one_to_one")

    if df["exit_time"].isna().any():
        missing = int(df["exit_time"].isna().sum())
        print(f"[WARN] {missing} trades missing exit_time after merge; dropping them.")
        df = df.dropna(subset=["exit_time"]).copy()

    # ensure integer ms
    df["fill_time"] = pd.to_numeric(df["fill_time"], errors="coerce").astype("int64")
    df["exit_time"] = pd.to_numeric(df["exit_time"], errors="coerce").astype("int64")

    # sort: earliest first; if equal fill_time, prefer higher y_prob
    sort_by = ["fill_time"] + (["y_prob"] if "y_prob" in df.columns else [])
    ascending = [True] + ([False] if "y_prob" in df.columns else [])
    df = df.sort_values(sort_by, ascending=ascending).reset_index(drop=True)

    # simulate concurrency
    active = []  # min-heap of exit_time
    kept_idx = []
    max_seen = 0

    for i, row in df.iterrows():
        t_in  = int(row["fill_time"])
        t_out = int(row["exit_time"])
        while active and active[0] <= t_in:
            heapq.heappop(active)
        if len(active) < args.max_concurrent:
            kept_idx.append(i)
            heapq.heappush(active, t_out)
            max_seen = max(max_seen, len(active))
        else:
            # skip this trade due to concurrency cap
            pass

    out = df.loc[kept_idx].copy()

    # preserve original policy columns in the same order, then useful extras
    cols = [c for c in pol.columns if c in out.columns]
    for extra in ["exit_time"]:
        if extra in out.columns and extra not in cols:
            cols.append(extra)

    out[cols].to_csv(args.out, index=False)

    print(f"[DONE] Concurrency-limited policy → {args.out}")
    print(f"  max_concurrent = {args.max_concurrent}")
    print(f"  kept {len(out)} / {len(df)} trades ({len(out)/max(1,len(df)):.1%})")
    print(f"  max_concurrency_seen = {max_seen}")

if __name__ == "__main__":
    main()
