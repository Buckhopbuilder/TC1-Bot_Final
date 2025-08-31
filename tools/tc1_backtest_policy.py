#!/usr/bin/env python3
import argparse, pandas as pd, numpy as np
from pathlib import Path

def drawdown_stats(equity):
    highwater = equity.cummax()
    dd = equity - highwater
    max_dd = dd.min()
    end_idx = dd.idxmin()
    start_idx = (equity[:end_idx]).idxmax() if end_idx is not None else None
    return float(max_dd), start_idx, end_idx

def main():
    ap = argparse.ArgumentParser(description="Backtest a filtered TC1 policy selection.")
    ap.add_argument("--policy", required=True, help="Policy trades CSV")
    ap.add_argument("--trades", default="reports/trades/trades.csv", help="Raw trades (only used if policy lacks needed cols)")
    ap.add_argument("--out", default="reports/ml/policy_equity.csv")
    ap.add_argument("--cost_R", type=float, default=0.02, help="Round-trip cost per trade in R units")
    args = ap.parse_args()

    pol = pd.read_csv(args.policy)

    # Auto-detect: if policy already has outcome & RR_target, don't merge; else merge with suffix safety.
    if ("outcome" in pol.columns) and ("RR_target" in pol.columns):
        df = pol.copy()
    else:
        tr = pd.read_csv(args.trades)[["trade_id","outcome","RR_target","fill_time","symbol","side"]]
        df = pol.merge(tr, on="trade_id", how="left", suffixes=("", "_tr"))
        # Coalesce suffixed columns if base missing
        for col in ["RR_target","outcome","fill_time","symbol","side"]:
            if col not in df.columns and f"{col}_tr" in df.columns:
                df[col] = df[f"{col}_tr"]
        # Clean up _tr columns
        df = df.drop(columns=[c for c in df.columns if c.endswith("_tr")], errors="ignore")

    # Timestamps normalization
    if "fill_time" in df.columns and df["fill_time"].notna().any():
        df["fill_dt"] = pd.to_datetime(df["fill_time"], unit="ms", utc=True, errors="coerce")
    elif "fill_time_iso" in df.columns and df["fill_time_iso"].notna().any():
        df["fill_dt"] = pd.to_datetime(df["fill_time_iso"], utc=True, errors="coerce")
    elif "fill_dt" in df.columns:
        df["fill_dt"] = pd.to_datetime(df["fill_dt"], utc=True, errors="coerce")
    else:
        raise ValueError("No fill_time/fill_time_iso/fill_dt available to build timeline.")

    # Final sanity: must have outcome & RR_target
    missing = [c for c in ["outcome","RR_target"] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns after merge: {missing}. "
                       f"Check that your --policy or --trades files contain them for trade_id joins.")

    df = df.dropna(subset=["outcome","RR_target"]).sort_values("fill_dt").reset_index(drop=True)

    # Compute R
    df["R_gross"] = np.where(df["outcome"]=="win", df["RR_target"], -1.0)
    df["R_net"]   = df["R_gross"] - args.cost_R
    df["equity_R"] = df["R_net"].cumsum()

    # Overall stats
    n = len(df)
    wins = (df["outcome"]=="win").sum()
    losses = (df["outcome"]=="loss").sum()
    wr = wins / n if n else 0.0
    expR = df["R_net"].mean() if n else 0.0
    mdd, dd_s, dd_e = drawdown_stats(df["equity_R"])

    print("=== Overall Backtest ===")
    print(f"Trades={n}  Wins={wins}  Losses={losses}  WR={wr:.4f}")
    print(f"Avg R / trade (after costs): {expR:.4f}")
    print(f"Max Drawdown (R): {mdd:.2f}")
    if dd_s is not None and dd_e is not None:
        print(f"DD window: {df.loc[dd_s,'fill_dt']} → {df.loc[dd_e,'fill_dt']}")

    # Per-symbol (silencing the pandas deprecation by excluding group key)
    if "symbol" in df.columns:
        def sym_stats(g):
            n=len(g); wr=(g['outcome']=='win').mean()
            expR=g['R_net'].mean()
            mdd,_,_=drawdown_stats(g['R_net'].cumsum())
            return pd.Series(dict(trades=n, winrate=wr, avg_R=expR, max_dd=mdd))
        ps = df.groupby("symbol", group_keys=False).apply(sym_stats).sort_values("avg_R", ascending=False)
        print("\n=== Per-symbol ===")
        print(ps.round(4))

    # Save equity curve
    out = df[["trade_id","symbol","fill_dt","R_net","equity_R"]].copy() if "symbol" in df.columns \
          else df[["trade_id","fill_dt","R_net","equity_R"]].copy()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\n[DONE] Equity curve → {args.out}")

if __name__ == "__main__":
    main()
