# TC1 Risk Overlay & Sizing Rules (Standalone Simulator)

## What this does
- Hybrid per-trade risk: flat `$` risk until `balance ≥ flat_risk_until`, then `% of balance`.
- Per-trade risk cap, total open-risk cap, max concurrent trades.
- Exchange-aware quantity rounding (floor to `qty_step`) + min qty/notional checks.
- Deterministic and identical logic for backtest & live (import the same engine).
- Outputs:
  - `out_csv`: per-trade allocation results (risk $, qty, skipped reason if any)
  - `equity_csv`: equity curve points (after each close)
  - `closed_csv` (optional): realized PnL details after each close

## Files
- `tools/tc1_risk_simulator.py` – the engine + CLI
- `configs/risk_overlay_example.json` – example configuration
- `tools/sample_trades.csv` – minimal example input

## Run (Linux/Mac)
```bash
cd TC1_RiskOverlay_Module
python tools/tc1_risk_simulator.py \
  --config configs/risk_overlay_example.json \
  --trades_csv tools/sample_trades.csv \
  --out_csv out/open_allocations.csv \
  --equity_csv out/equity_curve.csv \
  --closed_csv out/realized_pnl.csv
```

## Run (Windows CMD)
```cmd
cd /d C:\where\you\unzipped\TC1_RiskOverlay_Module
python tools\tc1_risk_simulator.py ^
  --config configs\risk_overlay_example.json ^
  --trades_csv tools\sample_trades.csv ^
  --out_csv out\open_allocations.csv ^
  --equity_csv out\equity_curve.csv ^
  --closed_csv out\realized_pnl.csv
```

## Integrate into live bot
```python
# tools/tc1_live_runner.py (excerpt)
from tools.tc1_risk_simulator import RiskConfig, RiskEngine

cfg = RiskConfig(...load your JSON...)
engine = RiskEngine(cfg)

# when a new TC1 entry signal is confirmed (with known stop price):
alloc = engine.try_open(
    trade_id=f"{symbol}_{entry_ts_ms}",
    symbol=symbol,
    side=+1 if is_long else -1,
    entry_price=entry_price,
    sl_price=stop_price,
    entry_time=pd.to_datetime(entry_ts_ms, unit="ms", utc=True),
)
if alloc["skipped_reason"]:
    # do nothing
    pass
else:
    qty = alloc["qty"]
    # place order at exchange with qty; store trade_id -> qty, entry, stop

# when the position exits:
closed = engine.close(trade_id, exit_price, pd.to_datetime(exit_ts_ms, unit="ms", utc=True))
balance = closed["balance_after"]
```

## Notes
- If `sizing_policy = "skip"`, the engine will **not** scale a trade to fit remaining risk capacity; it will skip instead.
- Fees are simple bps per side. Adjust to your taker/maker reality.
- If an exchange min (qty or notional) cannot be satisfied **within risk caps**, the trade is skipped (no over-risking to meet mins).
- For inverse/coin-margined contracts, extend `SymbolRules` to support contract value conversions.