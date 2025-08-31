#!/usr/bin/env python3
import argparse, pandas as pd
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="Risk overlay: convert R to $ with compounding balance.")
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

    # Build exit_time mapping directly
    if "exit_time" not in tr.columns:
        raise SystemExit("Trades file must contain exit_time (ms). Found: " + str(tr.columns))
    exit_map = tr.set_index("trade_id")["exit_time"].to_dict()

    # Merge only outcome, RR_target
    df = pol.merge(tr[["trade_id","outcome","RR_target"]], on="trade_id", how="left", validate="one_to_one")

    # Attach exit_time explicitly from map
    df["exit_time"] = df["trade_id"].map(exit_map)

    # Convert times
    fill_ms = pd.to_numeric(df["fill_time"], errors="coerce").astype("Int64")
    exit_ms = pd.to_numeric(df["exit_time"], errors="coerce").astype("Int64")

    # Use exit_time if present, else fallback
    df["when"] = exit_ms.fillna(fill_ms)

    # Drop unusable
    df = df.dropna(subset=["when","outcome","RR_target"]).copy()
    df = df.sort_values("when").reset_index(drop=True)

    # Net R
    won = df["outcome"].astype(str).str.lower().isin({"win","1","true"})
    df["R_gross"] = pd.to_numeric(df["RR_target"], errors="coerce").where(won, -1.0)
    df["R_net"]   = df["R_gross"] - float(args.cost_R)

    # Simulate compounding
    bal = float(args.start_balance)
    balances, risks, pnlR, pnlUSD = [], [], [], []
    for r in df["R_net"]:
        risk = bal * float(args.pct_risk) + float(args.fixed_add)
        pnl  = float(r) * risk
        bal += pnl
        balances.append(bal); risks.append(risk); pnlR.append(float(r)); pnlUSD.append(pnl)

    df["risk_used"]   = risks
    df["pnl_R"]       = pnlR
    df["pnl_USD"]     = pnlUSD
    df["balance_USD"] = balances
    df["when_dt"]     = pd.to_datetime(df["when"].astype("int64"), unit="ms", utc=True)

    keep = [c for c in [
        "trade_id","symbol","side","fill_time","exit_time","when","when_dt","y_prob",
        "RR_target","outcome","pnl_R","pnl_USD","risk_used","balance_USD"
    ] if c in df.columns]
    Path(Path(args.out).parent).mkdir(parents=True, exist_ok=True)
    df[keep].to_csv(args.out, index=False)

    used = df["exit_time"].notna().sum()
    print(f"[INFO] Used exit times for {used}/{len(df)} trades; others fell back to fill_time.")
    print(f"[DONE] Risk overlay → {args.out}")
    tot=len(df); wins=int((df['pnl_R']>0).sum()); losses=tot-wins; avgR=df['pnl_R'].mean() if tot else 0.0
    print(f"  start=${args.start_balance:,.2f}  sizing={args.pct_risk*100:.2f}% + ${args.fixed_add:.2f}  cost_R={args.cost_R}")
    print(f"  trades={tot}  wins={wins}  losses={losses}  mean_R={avgR:.3f}")
    if tot: print(f"  final balance = ${df['balance_USD'].iloc[-1]:,.2f}")

if __name__ == "__main__":
    main()
