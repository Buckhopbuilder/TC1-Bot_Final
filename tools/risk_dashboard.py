#!/usr/bin/env python3
"""
TC1 Risk Dashboard
- Prints engine snapshot (balance, open positions, recent equity)
- Prints safety rails config + daily counters (same as --show-policy)
"""
from tools.risk_integration import init_risk, _load_policy_state, _CFG_POLICY_PATH
import pandas as pd, json, os, sys

def main():
    e = init_risk()
    print("=== ENGINE SNAPSHOT ===")
    print(f"Balance: {e.balance:.2f}")
    print(f"Open positions: {len(e.open_positions)}  |  Total open risk: {e.total_open_risk:.2f}")
    for tid,p in e.open_positions.items():
        print(f"  - {tid}: {p.symbol} side={p.side} qty={p.qty} entry={p.entry_price} SL={p.sl_price} risk={p.open_risk:.2f}")

    if e.equity_points:
        eq = pd.DataFrame(e.equity_points, columns=["time","balance"]).tail(10)
        print("\nLast equity points:")
        print(eq.to_string(index=False))

    print("\n=== SAFETY RAILS ===")
    print("Config:", _CFG_POLICY_PATH)
    try:
        cfg = json.load(open(_CFG_POLICY_PATH))
    except Exception:
        cfg = {}
    print(json.dumps(cfg, indent=2))

    print("\nDaily counters (UTC):")
    st = _load_policy_state()
    print(json.dumps(st, indent=2) if st else "(no state yet)")

if __name__ == "__main__":
    main()
