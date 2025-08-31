#!/usr/bin/env python3
import argparse, pandas as pd
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="Entry-sized risk overlay with per-trade and total-risk caps.")
    ap.add_argument("--policy", required=True)
    ap.add_argument("--trades", default="reports/trades/trades_enriched.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--start_balance", type=float, default=100.0)
    ap.add_argument("--pct_risk", type=float, default=0.02)   # 2% of balance
    ap.add_argument("--fixed_add", type=float, default=15.0)  # +$15
    ap.add_argument("--cap_per_trade", type=float, default=None, help="Max $ risk per trade (e.g., 5 for $5).")
    ap.add_argument("--cap_total_open", type=float, default=None, help="Max total $ risk across open trades.")
    ap.add_argument("--cost_R", type=float, default=0.02)
    args = ap.parse_args()

    pol = pd.read_csv(args.policy); tr = pd.read_csv(args.trades)
    if "exit_time" not in tr.columns: raise SystemExit("trades must have exit_time (ms)")
    exit_map = tr.set_index("trade_id")["exit_time"].to_dict()
    df = pol.merge(tr[["trade_id","outcome","RR_target"]], on="trade_id", how="left", validate="one_to_one")
    df["exit_time"] = df["trade_id"].map(exit_map)
    df["fill_time"] = pd.to_numeric(df["fill_time"], errors="coerce").astype("Int64")
    df["exit_time"] = pd.to_numeric(df["exit_time"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["fill_time","exit_time","outcome","RR_target"]).copy()

    entries = df[["trade_id","fill_time"]].rename(columns={"fill_time":"t"}).assign(kind="entry")
    exits   = df[["trade_id","exit_time"]].rename(columns={"exit_time":"t"}).assign(kind="exit")
    ev = pd.concat([entries, exits], ignore_index=True)
    ev["t"] = pd.to_numeric(ev["t"], errors="coerce").astype("int64")
    ev["order"] = ev["kind"].map({"exit":0,"entry":1})
    ev = ev.sort_values(["t","order"]).reset_index(drop=True)

    df = df.set_index("trade_id", drop=False)
    won = df["outcome"].astype(str).str.lower().isin({"win","1","true"})
    df["R_gross"] = pd.to_numeric(df["RR_target"], errors="coerce").where(won, -1.0)
    df["R_net"]   = df["R_gross"] - float(args.cost_R)

    bal = float(args.start_balance)
    open_risk = 0.0
    alloc = {}
    closed = []

    def size_now():
        r = bal * float(args.pct_risk) + float(args.fixed_add)
        if args.cap_per_trade is not None:
            r = min(r, float(args.cap_per_trade))
        return max(0.0, r)

    for _, e in ev.iterrows():
        tid = e["trade_id"]
        if e["kind"] == "exit":
            if tid in alloc:
                rnet = float(df.at[tid,"R_net"])
                risk_used = alloc.pop(tid)
                open_risk -= risk_used
                pnl = rnet * risk_used
                bal += pnl
                rec = {
                    "trade_id": tid,
                    "fill_time": int(df.at[tid,"fill_time"]),
                    "exit_time": int(df.at[tid,"exit_time"]),
                    "RR_target": float(df.at[tid,"RR_target"]),
                    "outcome":   df.at[tid,"outcome"],
                    "pnl_R":     rnet,
                    "risk_used": risk_used,
                    "pnl_USD":   pnl,
                    "balance_USD": bal,
                }
                if "symbol" in df.columns: rec["symbol"] = df.at[tid,"symbol"]
                if "side"   in df.columns: rec["side"]   = df.at[tid,"side"]
                if "y_prob" in df.columns: rec["y_prob"] = df.at[tid,"y_prob"]
                closed.append(rec)
        else:
            risk = size_now()
            # respect total-open-risk cap
            if args.cap_total_open is not None and open_risk + risk > float(args.cap_total_open):
                continue
            alloc[tid] = risk
            open_risk += risk

    out = pd.DataFrame(closed)
    if not out.empty:
        out["when"] = out["exit_time"]
        out["when_dt"] = pd.to_datetime(out["when"], unit="ms", utc=True)
        cols = [c for c in ["trade_id","symbol","side","fill_time","exit_time","when","when_dt","y_prob",
                            "RR_target","outcome","pnl_R","pnl_USD","risk_used","balance_USD"] if c in out.columns]
        out = out[cols].sort_values("when").reset_index(drop=True)

    Path(Path(args.out).parent).mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    tot=len(out); wins=int((out["pnl_R"]>0).sum()) if tot else 0; losses=tot-wins
    avgR=out["pnl_R"].mean() if tot else 0.0
    print(f"[DONE] Risk overlay (capped) → {args.out}")
    print(f"  start=${args.start_balance:,.2f}  sizing={args.pct_risk*100:.2f}% + ${args.fixed_add:.2f}  cost_R={args.cost_R}")
    if args.cap_per_trade is not None: print(f"  cap_per_trade=${args.cap_per_trade:.2f}")
    if args.cap_total_open is not None: print(f"  cap_total_open=${args.cap_total_open:.2f}")
    print(f"  trades={tot}  wins={wins}  losses={losses}  mean_R={avgR:.3f}")
    if tot: print(f"  final balance=${out['balance_USD'].iloc[-1]:,.2f}")

if __name__ == "__main__":
    main()
