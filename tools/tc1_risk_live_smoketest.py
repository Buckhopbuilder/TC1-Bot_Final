from tools.risk_integration import init_risk, risk_try_open, risk_close

risk = init_risk()

symbol = "BTCUSDT"
entry_ts_ms = 1724880000000
exit_ts_ms  = entry_ts_ms + 3_600_000
trade_id = f"{symbol}_{entry_ts_ms}"

alloc = risk_try_open(
    trade_id=trade_id,
    symbol=symbol,
    is_long=True,
    entry_ts_ms=entry_ts_ms,
    entry_price=65000.0,
    stop_price=64200.0,
)
print("alloc:", {k: alloc.get(k) for k in ("skipped_reason","qty","risk_allocated","open_risk_after","open_concurrent_after")})

if not alloc["skipped_reason"]:
    closed = risk_close(
        trade_id=trade_id,
        exit_ts_ms=exit_ts_ms,
        exit_price=65400.0,
    )
    print("closed:", {k: closed.get(k) for k in ("pnl_after_fees","R_multiple","balance_after")})
