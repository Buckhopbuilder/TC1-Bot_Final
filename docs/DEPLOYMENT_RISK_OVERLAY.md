# TC1 Risk Overlay & Sizing – Deployment Guide

This module enforces hybrid sizing, risk caps, and concurrency for both backtests and live trading. It’s designed to produce **identical results** in sim/backtest and live.

---

## Layout & Key Files

```
configs/
  risk_overlay.json     # sizing & fees (engine)
  risk_policy.json      # safety rails (equity floor, daily loss cap, max trades/day)

runtime/
  risk_state.json       # persisted engine state (balance, open positions, equity points)
  risk_policy_state.json# per-UTC-day counters (entries, pnl)

tools/
  tc1_risk_engine.py    # engine (try_open/close); vendored, stable imports
  tc1_risk_harness.py   # standalone simulator runner for CSVs
  risk_integration.py   # live integration + persistence + safety rails + CLI
  risk_dashboard.py     # quick snapshot of engine + safety rails
  tc1_risk_live_smoketest.py  # minimal end-to-end sanity test
```

---

## Quick Start (Live Integration)

1) **Initialize the engine once at startup**
```python
from tools.risk_integration import init_risk, risk_try_open, risk_close
risk = init_risk()   # loads configs + restores state from runtime/
```

2) **On entry signal (before placing order)**
```python
alloc = risk_try_open(
    trade_id=f"{symbol}_{int(entry_ts_ms)}",
    symbol=symbol,
    is_long=is_long,                 # True for long, False for short
    entry_ts_ms=int(entry_ts_ms),    # ms epoch
    entry_price=float(entry_price),
    stop_price=float(stop_price),    # required for risk sizing
)
if alloc["skipped_reason"]:
    logger.info(f"RISK SKIP {symbol}: {alloc['skipped_reason']}")
    return

qty = alloc["qty"]
# place order with `qty` at the exchange
```

3) **On exit fill (after order closes)**
```python
closed = risk_close(
    trade_id=trade_id,
    exit_ts_ms=int(exit_ts_ms),
    exit_price=float(exit_price),
)
logger.info(f"Closed {trade_id}: R={closed['R_multiple']:.3f}, PnL={closed['pnl_after_fees']:.2f}, Bal={closed['balance_after']:.2f}")
```

> State is automatically persisted to `runtime/`. If the process restarts, it resumes with the same balance and open positions.

---

## Configs

### `configs/risk_overlay.json` (sizing & fees)
- `initial_balance`: starting equity for sim/live.
- `flat_risk_until`: balance threshold where we switch to percent risk.
- `flat_risk_amount`: $ risk per trade until threshold.
- `percent_risk`: fraction of balance risk per trade after threshold (e.g., `0.005` = 0.5%).
- Caps: `per_trade_risk_cap`, `total_open_risk_cap`, `max_concurrent_trades`.
- Fees/market rules: `fee_bps_per_side`, `default_qty_step`, `default_min_qty`, `default_min_notional`, `symbol_overrides`.

### `configs/risk_policy.json` (safety rails)
```json
{
  "equity_floor": 1000.0,
  "daily_loss_cap": 50.0,
  "max_trades_per_day": 10
}
```
- **Equity floor**: block new entries if `balance ≤ floor`.
- **Daily loss cap**: block new entries once loss today ≥ cap.
- **Max trades/day**: block entries after N opens in the same UTC day.

> Daily counters auto-roll over at **UTC midnight**. Manual controls below.

---

## Monitoring & CLI

### Dashboard
```bash
python -m tools.risk_dashboard
```
Shows: balance, open risk, open positions, recent equity, safety rails config + per-day counters.

### Integration CLI
```bash
# Show policy config + state
python -m tools.risk_integration --show-policy

# Reset counters
python -m tools.risk_integration --reset-policy         # all days
python -m tools.risk_integration --reset-policy=today   # today's bucket only
```

---

## Backtest / Sim Harness

Use the harness to run the same engine on CSV inputs and verify parity:

```bash
python -m tools.tc1_risk_harness   --trades_csv reports/ml/_overlay_check/trades_for_overlay.csv   --risk_config configs/risk_overlay.json   --out_dir reports/ml/risk_overlay
```

Produces:
- `open_allocations.csv` (allocations & skip reasons)
- `realized_pnl.csv` (R, PnL, balance_after)
- `equity_curve.csv`
- `summary.json`

> Tip: Build `trades_for_overlay.csv` by joining the selected policy set (e.g., `conc2`) with `trades_enriched` and renaming **entry time** to `timestamp`, keeping:
> `symbol, side, timestamp, entry_price, sl_price, exit_time, exit_price`.

---

## Sanity / Smoke Tests

1) **Live smoke**
```bash
python -m tools.tc1_risk_live_smoketest
# Expect alloc + close printouts and runtime/risk_state.json updated
```

2) **Dashboard**
```bash
python -m tools.risk_dashboard
```

3) **Safety rails demo**
```bash
python -m tools.risk_integration --show-policy
python -m tools.risk_integration --reset-policy=today
```

---

## Persistence & Resets

- Engine state lives in `runtime/risk_state.json`.  
- Safety counters live in `runtime/risk_policy_state.json` and **auto-roll** at UTC midnight.  
- To seed a starting balance, write a minimal JSON:
```json
{"balance": 2500.0, "total_open_risk": 0.0, "open_positions": {}, "equity_points": []}
```

---

## Troubleshooting

- **“ModuleNotFoundError: tools …”**  
  Run from repo root and ensure `tools/__init__.py` exists.  
  Use `python -m tools.<module>`.

- **Skipped entries**  
  Check `open_allocations.csv` → `skipped_reason` column (caps, concurrency, safety rails).

- **Backtest/live mismatch**  
  Ensure same **trade set** and **columns**; confirm fees/rounding and exits. For apples-to-apples, run fixed-$ no-fee mode first, then re-enable live settings.

---

## Deployment Checklist

- [ ] `tools/risk_integration.py` imported and `init_risk()` called at startup  
- [ ] Wrap entries with `risk_try_open`, exits with `risk_close`  
- [ ] Confirm `configs/risk_overlay.json` and `configs/risk_policy.json` values  
- [ ] Confirm `runtime/` is writable and persisted across restarts  
- [ ] Dashboard/CLI available to ops team
