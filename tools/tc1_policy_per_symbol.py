#!/usr/bin/env python3
import argparse, numpy as np, pandas as pd
from pathlib import Path

def pick_threshold(pred_sym, min_trades=400, grid=None):
    if grid is None: grid = np.linspace(0.15, 0.45, 7)  # 15%..45%
    best = None
    for frac in grid:
        thr = pred_sym["y_prob"].quantile(1.0 - frac)
        sel = pred_sym[pred_sym["y_prob"] >= thr].copy()
        n = len(sel)
        if n < min_trades: continue
        wr = sel["y_true"].mean() if "y_true" in sel else np.nan
        expR = np.mean(np.where(sel["y_true"]==1, sel["RR_target"], -1)) if "y_true" in sel else np.nan
        # simple score: expectancy; you can try Sharpe-ish: expR / std(R)
        if best is None or (not np.isnan(expR) and expR > best["expR"]):
            best = {"frac": float(frac), "thr": float(thr), "n": int(n), "wr": float(wr), "expR": float(expR)}
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="reports/ml/predictions_rf.csv")
    ap.add_argument("--trades", default="reports/trades/trades.csv")
    ap.add_argument("--out_summary", default="reports/ml/per_symbol_thresholds.csv")
    ap.add_argument("--out_trades", default="reports/ml/policy_trades_per_symbol.csv")
    args = ap.parse_args()

    pred = pd.read_csv(args.pred)
    tr   = pd.read_csv(args.trades)[["trade_id","A_time","B_time","exit_time","A_price","B_price","entry_price","sl_price","tp_price","outcome"]]
    df = pred.merge(tr, on="trade_id", how="left")

    rows = []
    outs = []
    for sym, g in df.groupby("symbol"):
        best = pick_threshold(g)
        if not best:
            rows.append({"symbol": sym, "note": "no candidate"}); continue
        thr = best["thr"]; frac = best["frac"]
        sel = g[g["y_prob"] >= thr].copy().sort_values("y_prob", ascending=False)
        sel["policy_symbol"] = sym
        sel["policy_frac"] = frac
        outs.append(sel)
        rows.append({"symbol": sym, "top_frac": frac, "threshold": thr, "trades": best["n"], "winrate": best["wr"], "expectancy_R_obs": best["expR"]})

    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_summary, index=False)

    if outs:
        out = pd.concat(outs, ignore_index=True)
        # timestamps
        for col in ["A_time","B_time","fill_time","exit_time"]:
            if col in out.columns:
                out[col+"_iso"] = pd.to_datetime(out[col], unit="ms", utc=True)
        keep = ["trade_id","symbol","side","A_time_iso","B_time_iso","fill_time_iso","exit_time_iso",
                "A_price","B_price","entry_price","sl_price","tp_price","RR_target","outcome","y_prob",
                "policy_symbol","policy_frac"]
        out[keep].to_csv(args.out_trades, index=False)
    print(f"[DONE] summary -> {args.out_summary}\n[DONE] trades  -> {args.out_trades}")

if __name__ == "__main__":
    main()
