#!/usr/bin/env python3
"""
Risk integration for TC1 live runner.
- Loads RiskEngine from configs/risk_overlay.json
- Safety rails (configs/risk_policy.json):
    * Equity floor: pause new entries if balance ≤ X
    * Daily loss cap: pause entries after loss_today ≥ Y (UTC)
    * Max trades/day: pause entries after N entries (UTC)
- Persists engine state to runtime/risk_state.json
- Persists safety-day state to runtime/risk_policy_state.json
- Auto-rollover of daily counters at UTC midnight
- CLI:
    python -m tools.risk_integration --reset-policy [all|today]
    python -m tools.risk_integration --show-policy
"""
from __future__ import annotations
import os, json, argparse, sys
from typing import Optional, Dict, Any, Tuple
import pandas as pd

from tools.tc1_risk_engine import RiskConfig, RiskEngine, load_config

_ROOT = os.path.dirname(os.path.dirname(__file__))
_CFG_RISK_PATH   = os.path.join(_ROOT, "configs", "risk_overlay.json")
_CFG_POLICY_PATH = os.path.join(_ROOT, "configs", "risk_policy.json")

_STATE_ENGINE = os.path.join(_ROOT, "runtime", "risk_state.json")
_STATE_POLICY = os.path.join(_ROOT, "runtime", "risk_policy_state.json")

_engine: Optional[RiskEngine] = None
_policy_cfg: Dict[str, Any] = {}

def _ensure_dirs():
    os.makedirs(os.path.join(_ROOT, "runtime"), exist_ok=True)

def _load_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def _save_json(path: str, obj: Any):
    _ensure_dirs()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def _policy_defaults() -> Dict[str, Any]:
    return {"equity_floor": None, "daily_loss_cap": None, "max_trades_per_day": None}

def _load_policy_cfg():
    global _policy_cfg
    raw = _load_json(_CFG_POLICY_PATH, {})
    base = _policy_defaults()
    base.update(raw or {})
    _policy_cfg = base

def _serialize_engine(engine: RiskEngine) -> Dict[str, Any]:
    opens = {}
    for k, p in engine.open_positions.items():
        opens[k] = {
            "trade_id": p.trade_id,
            "symbol": p.symbol,
            "side": p.side,
            "qty": p.qty,
            "entry_price": p.entry_price,
            "sl_price": p.sl_price,
            "open_risk": p.open_risk,
            "notional": p.notional,
            "entry_fee": p.entry_fee,
            "entry_time": None if pd.isna(p.entry_time) else pd.Timestamp(p.entry_time).isoformat(),
        }
    return {
        "balance": engine.balance,
        "total_open_risk": engine.total_open_risk,
        "open_positions": opens,
        "equity_points": [
            (pd.Timestamp(t).isoformat() if pd.notna(t) else None, float(b))
            for (t, b) in (engine.equity_points or [])
        ],
    }

def _restore_engine_state(engine: RiskEngine, state: Dict[str, Any]) -> None:
    engine.balance = float(state.get("balance", engine.balance))
    engine.total_open_risk = float(state.get("total_open_risk", 0.0))
    engine.open_positions.clear()
    OpenPosition = getattr(engine, "OpenPosition", engine.__class__.__dict__.get("OpenPosition"))
    for k, v in (state.get("open_positions") or {}).items():
        engine.open_positions[k] = OpenPosition(
            trade_id=v["trade_id"],
            symbol=v["symbol"],
            side=int(v["side"]),
            qty=float(v["qty"]),
            entry_price=float(v["entry_price"]),
            sl_price=float(v["sl_price"]),
            open_risk=float(v["open_risk"]),
            notional=float(v["notional"]),
            entry_fee=float(v["entry_fee"]),
            entry_time=pd.to_datetime(v["entry_time"], utc=True) if v.get("entry_time") else pd.NaT,
        )
    engine.equity_points = [
        (pd.to_datetime(t, utc=True) if t else pd.NaT, float(b))
        for (t, b) in (state.get("equity_points") or [])
    ]

def _save_engine(engine: RiskEngine):
    _save_json(_STATE_ENGINE, _serialize_engine(engine))

def _load_engine_state() -> Optional[Dict[str, Any]]:
    return _load_json(_STATE_ENGINE, None)

def _today_key(ts_ms: Optional[int] = None) -> str:
    if ts_ms is None:
        return pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    return pd.to_datetime(int(ts_ms), unit="ms", utc=True).strftime("%Y-%m-%d")

def _load_policy_state() -> Dict[str, Any]:
    # structure: {"YYYY-MM-DD": {"entries": int, "pnl_cum": float}, "_meta": {"last_seen_day": "YYYY-MM-DD"}}
    st = _load_json(_STATE_POLICY, {})
    if "_meta" not in st:
        st["_meta"] = {"last_seen_day": _today_key()}
    return st

def _save_policy_state(st: Dict[str, Any]):
    _save_json(_STATE_POLICY, st)

def _rollover_policy_state(st: Dict[str, Any], current_day: str, keep_days: int = 7) -> Dict[str, Any]:
    # If day changed since last_seen_day, ensure current_day bucket exists; prune old days beyond keep_days
    last = st.get("_meta", {}).get("last_seen_day")
    if last != current_day:
        st.setdefault(current_day, {"entries": 0, "pnl_cum": 0.0})
        # prune old days (keep last N + meta)
        days = sorted([k for k in st.keys() if k != "_meta"])
        to_keep = set(days[-keep_days:])
        for k in list(st.keys()):
            if k not in to_keep and k != "_meta":
                del st[k]
        st["_meta"]["last_seen_day"] = current_day
    return st

def init_risk() -> RiskEngine:
    """Initialize (or return) the global RiskEngine with persisted state restored."""
    global _engine
    if _engine is not None:
        return _engine
    _load_policy_cfg()
    cfg = load_config(_CFG_RISK_PATH)
    _engine = RiskEngine(cfg)
    st = _load_engine_state()
    if st:
        try:
            _restore_engine_state(_engine, st)
        except Exception:
            pass
    return _engine

def _guard_can_open(engine: RiskEngine, entry_ts_ms: int) -> Tuple[bool, str, Dict[str, Any]]:
    """Return (ok, reason, snapshot) based on safety rails; auto-rollover daily counters."""
    _load_policy_cfg()  # reload in case JSON changed on disk
    pst = _load_policy_state()
    day = _today_key(entry_ts_ms)
    pst = _rollover_policy_state(pst, day)
    day_st = pst.get(day, {"entries": 0, "pnl_cum": 0.0})

    # Equity floor
    eq_floor = _policy_cfg.get("equity_floor")
    if isinstance(eq_floor, (int, float)) and eq_floor is not None:
        if engine.balance <= float(eq_floor):
            _save_policy_state(pst)  # persist rollover/meta
            return (False, "safety_equity_floor", {"day": day, **day_st})

    # Daily loss cap
    cap = _policy_cfg.get("daily_loss_cap")
    if isinstance(cap, (int, float)) and cap is not None:
        loss_today = max(0.0, -float(day_st.get("pnl_cum", 0.0)))
        if loss_today >= float(cap):
            _save_policy_state(pst)
            return (False, "safety_daily_loss_cap", {"day": day, "loss_today": loss_today, **day_st})

    # Max trades/day
    max_tr = _policy_cfg.get("max_trades_per_day")
    if isinstance(max_tr, int) and max_tr is not None:
        if int(day_st.get("entries", 0)) >= int(max_tr):
            _save_policy_state(pst)
            return (False, "safety_max_trades_per_day", {"day": day, **day_st})

    # Save any rollover/meta changes
    _save_policy_state(pst)
    return (True, "", {"day": day, **day_st})

def risk_try_open(*, trade_id: str, symbol: str, is_long: bool, entry_ts_ms: int,
                  entry_price: float, stop_price: float) -> Dict[str, Any]:
    """
    Call this before placing an order. Returns alloc dict:
      - if alloc['skipped_reason'] is non-empty: DO NOT PLACE THE ORDER
      - else: use alloc['qty'] for the order size
    """
    eng = init_risk()
    ok, reason, snap = _guard_can_open(eng, entry_ts_ms)
    if not ok:
        return {
            "trade_id": trade_id, "symbol": symbol, "side": (+1 if is_long else -1),
            "entry_price": float(entry_price), "sl_price": float(stop_price),
            "entry_time": pd.to_datetime(int(entry_ts_ms), unit="ms", utc=True),
            "risk_mode": None, "risk_target": None, "risk_allocated": 0.0,
            "qty": 0.0, "notional": 0.0, "entry_fee": 0.0,
            "open_concurrent_after": len(eng.open_positions),
            "open_risk_after": eng.total_open_risk,
            "skipped_reason": reason,
            "safety_snapshot": snap
        }

    alloc = eng.try_open(
        trade_id=trade_id,
        symbol=symbol,
        side=+1 if is_long else -1,
        entry_price=float(entry_price),
        sl_price=float(stop_price),
        entry_time=pd.to_datetime(int(entry_ts_ms), unit="ms", utc=True),
    )
    if not alloc.get("skipped_reason"):
        # increment entries_today
        pst = _load_policy_state()
        day = _today_key(entry_ts_ms)
        pst = _rollover_policy_state(pst, day)
        d = pst.get(day, {"entries": 0, "pnl_cum": 0.0})
        d["entries"] = int(d.get("entries", 0)) + 1
        pst[day] = d
        _save_policy_state(pst)
        _save_engine(eng)
    return alloc

def risk_close(*, trade_id: str, exit_ts_ms: int, exit_price: float) -> Dict[str, Any]:
    """
    Call this after your exit fill is confirmed. Returns closed dict with pnl, R, balance_after, etc.
    """
    eng = init_risk()
    closed = eng.close(
        trade_id=trade_id,
        exit_price=float(exit_price),
        exit_time=pd.to_datetime(int(exit_ts_ms), unit="ms", utc=True),
    )
    if closed.get("closed"):
        pst = _load_policy_state()
        day = _today_key(exit_ts_ms)
        pst = _rollover_policy_state(pst, day)
        d = pst.get(day, {"entries": 0, "pnl_cum": 0.0})
        d["pnl_cum"] = float(d.get("pnl_cum", 0.0)) + float(closed.get("pnl_after_fees", 0.0))
        pst[day] = d
        _save_policy_state(pst)
        _save_engine(eng)
    return closed

# ---- CLI utilities ----
def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset-policy", nargs="?", const="all", choices=["all","today"],
                    help="Reset policy counters (all days or just today's bucket).")
    ap.add_argument("--show-policy", action="store_true", help="Print policy config and state.")
    args = ap.parse_args()

    if args.reset_policy:
        mode = args.reset_policy
        st = _load_policy_state()
        if mode == "today":
            day = _today_key()
            if day in st: del st[day]
        else:
            st = {"_meta": {"last_seen_day": _today_key()}}
        _save_policy_state(st)
        print(f"[OK] reset-policy {mode}")
        return

    if args.show_policy:
        _load_policy_cfg()
        print("[policy cfg]:", json.dumps(_policy_cfg, indent=2))
        st = _load_policy_state()
        print("[policy state]:", json.dumps(st, indent=2))
        return

    # default: init and print a small status
    e = init_risk()
    print("[risk] balance:", e.balance, "open_positions:", len(e.open_positions))
    _load_policy_cfg()
    print("[policy cfg]:", json.dumps(_policy_cfg, indent=2))
    print("[policy state]:", json.dumps(_load_policy_state(), indent=2))

if __name__ == "__main__":
    _cli()
