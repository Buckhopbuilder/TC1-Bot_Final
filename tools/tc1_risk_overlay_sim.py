#!/usr/bin/env python3
import argparse, pandas as pd
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="Event-driven risk overlay: size at ENTRY, realize PnL at EXIT.")
    ap.add_argument("--policy", required=True)
    ap.add_argument("--trades", default="reports/trades/trades_enriched.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--start_balance", type=float, default=100.0)
    ap.add_argument("--pct_risk", type=float, default=0.02)
    ap.add_argument("--fixed_add", type=float, default=15.0)
    ap.add_argument("--cost_R", type=float, default=0.02)
    args = ap.parse_args()

    pol = pd.read_csv(args.policy)
    tr  = pd.read_csv(args.trades)

    # Build direct map of exit_time
    if "exit_time" not in tr.columns:
        raise SystemExit(f"Trades file missing exit_time. Found: {list(tr.columns)}")
    exit_map = tr.set_index("trade_id")["exit_time"].to_dict()

    # Merge outcome & RR_target (but not exit_time)
    df = pol.merge(tr[["trade_id","outcome","RR_target"]], on="trade_id", how="left", validate="one_to_one")

    # Attach exit_time explicitly from map
    df["exit_time"] = df["trade_id"].map(exit_map)

    # Convert to int ms
    df["fill_time"] = pd.to_numeric(df["fill_time"], errors="coerce").astype("Int64")
    df["exit_time"] = pd.to_numeric(df["exit_time"], errors="coerce").astype("Int64")

    # Drop unusable
    df = df.dropna(subset=["fill_time","exit_time","outcome","RR_target"]).copy()

    # Build event list (exits before entries at same t)
    entries = df[["trade_id","fill_time"]].rename(columns={"fill_time":"t"}).assign(kind="entry")
    exits   = df[["trade_id","exit_time"]].rename(columns={"exit_time":"t"}).assign(kind="exit")
    events  = pd.concat([entries, exits], ignore_index=True)
    events["t"] = pd.to_numeric(events["t"], errors="coerce").astype("int64")
    events["order"] = events["kind"].map({"exit":0,"entry":1})
    events = events.sort_values(["t","order"]).reset_index(drop=True)

    # Precompute R
    won = df["outcome"].astype(str).str.lower().isin({"win","1","true"})
    df["R_gross"] = pd.to_numeric(df["RR_target"], errors="coerce").where(won, -1.0)
    df["R_net"]   = df["R_gross"] - float(args.cost_R)

    # Simulate
    bal = float(args.start_balance)
    alloc = {}
    closed = []

    for _, ev in events.iterrows():
        tid = ev["trade_id"]
        if ev["kind"] == "exit" and tid in alloc:
            rnet = float(df.loc[df.trade_id==tid,"R_net"].iloc[0])
            risk_used = alloc.pop(tid)
            pnl = rnet * risk_used
            bal += pnl
            closed.append({
                "trade_id": tid,
                "fill_time": int(df.loc[df.trade_id==tid,"fill_time"].iloc[0]),
                "exit_time": int(df.loc[df.trade_id==tid,"exit_time"].iloc[0]),
                "RR_target": float(df.loc[df.trade_id==tid,"RR_target"].iloc[0]),
                "outcome":   df.loc[df.trade_id==tid,"outcome"].iloc[0],
                "pnl_R":     rnet,
                "risk_used": risk_used,
                "pnl_USD":   pnl,
                "balance_USD": bal,
            })
        elif ev["kind"] == "entry":
            risk = bal * float(args.pct_risk) + float(args.fixed_add)
            alloc[tid] = risk

    out = pd.DataFrame(closed)
    if not out.empty:
        out["when"] = out["exit_time"]
        out["when_dt"] = pd.to_datetime(out["when"], unit="ms", utc=True)
    Path(Path(args.out).parent).mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    tot=len(out); wins=int((out["pnl_R"]>0).sum()); losses=tot-wins
    avgR=out["pnl_R"].mean() if tot else 0
    print(f"[DONE] Risk overlay → {args.out}")
    print(f"  trades={tot} wins={wins} losses={losses} mean_R={avgR:.3f}")
    if tot: print(f"  final balance=${out['balance_USD'].iloc[-1]:,.2f}")

if __name__ == "__main__":
    main()
