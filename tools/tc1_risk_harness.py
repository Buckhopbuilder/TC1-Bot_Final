#!/usr/bin/env python3
"""
TC1 Risk Harness (robust columns)
- Works with either raw trades_enriched.csv (fill_time) OR the prebuilt trades_for_overlay.csv (timestamp)
- No policy merge required when you pass the already-filtered CSV
"""
import argparse, os, json
import pandas as pd

try:
    from tools.tc1_risk_engine import RiskConfig, RiskEngine, load_config, run_simulation
    _ENGINE_SOURCE = "tools.tc1_risk_engine"
except Exception:
    from risk_overlay.TC1_RiskOverlay_Module.tools.tc1_risk_simulator import RiskConfig, RiskEngine, load_config, run_simulation
    _ENGINE_SOURCE = "risk_overlay.TC1_RiskOverlay_Module.tools.tc1_risk_simulator"

def _norm_side(x):
    s = str(x).strip().lower()
    if s in ("long","buy","b","1","+1"): return 1
    if s in ("short","sell","s","-1"):  return -1
    try: return 1 if float(x) >= 0 else -1
    except: return 0

def _pick(cols_map, *names):
    for n in names:
        if n in cols_map: return cols_map[n]
    return None

def build_trades_df(trades_csv):
    df = pd.read_csv(trades_csv)
    cols = {c.lower().strip(): c for c in df.columns}

    c_symbol = _pick(cols, "symbol","sym","ticker")
    c_side   = _pick(cols, "side","direction","position")
    c_time   = _pick(cols, "timestamp","fill_time","entry_time","open_time","time_open")
    c_ep     = _pick(cols, "entry_price","entry","open","price_entry")
    c_sl     = _pick(cols, "sl_price","stop","sl","stop_price")
    c_xt     = _pick(cols, "exit_time","close_time","time_close")
    c_xp     = _pick(cols, "exit_price","exit","close","price_exit")

    need = [c_symbol,c_side,c_time,c_ep,c_sl,c_xt,c_xp]
    missing = [n for n,v in zip(
        ["symbol","side","timestamp/fill_time","entry_price","sl_price","exit_time","exit_price"], need) if v is None]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nFound: {list(df.columns)}")

    out = pd.DataFrame({
        "symbol": df[c_symbol].astype(str).str.upper(),
        "side": df[c_side].apply(_norm_side),
        "timestamp": df[c_time],
        "entry_price": df[c_ep].astype(float),
        "sl_price": df[c_sl].astype(float),
        "exit_time": df[c_xt],
        "exit_price": df[c_xp].astype(float),
    })
    out = out[out["side"].isin([+1,-1])].reset_index(drop=True)
    return out

def summarize(eq_df, closed_df, initial_balance):
    if eq_df is None or len(eq_df)==0:
        return {"initial_balance": initial_balance, "end_balance": initial_balance, "pnl":0.0, "num_trades": int(len(closed_df))}
    end_balance = float(eq_df["balance"].iloc[-1])
    return {"initial_balance": initial_balance, "end_balance": end_balance, "pnl": end_balance-initial_balance, "num_trades": int(len(closed_df))}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades_csv", required=True)
    ap.add_argument("--risk_config", required=True)
    ap.add_argument("--out_dir", default="reports/ml/risk_overlay")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = load_config(args.risk_config)
    trades_df = build_trades_df(args.trades_csv)

    (open_df, eq_df), closed_df = run_simulation(cfg, trades_df)

    open_path   = os.path.join(args.out_dir,"open_allocations.csv")
    eq_path     = os.path.join(args.out_dir,"equity_curve.csv")
    closed_path = os.path.join(args.out_dir,"realized_pnl.csv")
    summary_path= os.path.join(args.out_dir,"summary.json")

    open_df.to_csv(open_path, index=False)
    eq_df.to_csv(eq_path, index=False)
    closed_df.to_csv(closed_path, index=False)

    summary = summarize(eq_df, closed_df, cfg.initial_balance)
    with open(summary_path,"w") as f:
        json.dump(summary,f,indent=2)

    print(f"[OK] Wrote {open_path}, {closed_path}, {eq_path}, {summary_path} (engine: {_ENGINE_SOURCE})")

if __name__=="__main__":
    main()
