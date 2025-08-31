#!/usr/bin/env python3
"""
TC1 Risk Overlay & Sizing Rules
- Hybrid sizing: flat $ risk until balance ≥ threshold, then % risk of balance
- Per-trade risk cap and total open-risk cap
- Max concurrent trades
- Same logic usable in backtest and live sim (deterministic, exchange rounding-aware)
- Outputs: per-trade risk $, qty, realized PnL, updated balance, equity curve

Assumptions:
- Linear USDT-margined contracts (PnL in USDT). For symbol-specific rules
  (qty step, min notional/qty), provide overrides in the config.
- Input trades CSV contains executed entries and exits (times & prices).

CSV expected columns (case-insensitive; flexible names supported):
- symbol
- side            (one of: long, short, buy, sell, 1, -1)
- timestamp       (entry time; unix ms or ISO8601)  | fallback: entry_time
- entry_price     (fallback: entry)
- sl_price        (fallback: stop, sl)
- exit_time       (unix ms or ISO8601)              | optional for live but required to compute realized PnL here
- exit_price      (fallback: exit)

Optional columns are preserved in output if present (e.g., tp_price, outcome, R_multiple, etc.).

Determinism & consistency:
- Quantity rounding uses floor to the exchange step (never exceed risk caps because of rounding up).
- If exchange min_notional/min_qty cannot be met *within allowed risk*, the trade is skipped.
- Concurrency & total-open-risk are enforced on entry only.
- Fees are charged per side: entry_fee = fee_bps_per_side * notional / 10_000 (and same for exit).

CLI:
  python tools/tc1_risk_simulator.py ^
      --config configs/risk_overlay_example.json ^
      --trades_csv path\to\trades.csv ^
      --out_csv out\sim_results.csv ^
      --equity_csv out\equity_curve.csv

Author: TC1 Project
"""
from __future__ import annotations

import dataclasses as dc
from dataclasses import dataclass
from typing import Dict, Optional, List, Any, Tuple
from datetime import datetime, timezone
import pandas as pd
import math
import json


def _to_dt(x):
    """Best-effort parse to pandas.Timestamp (UTC)."""
    if pd.isna(x):
        return pd.NaT
    try:
        # Try numeric epoch (ms or s)
        xv = float(x)
        if xv > 1e12:  # ms
            return pd.to_datetime(int(xv), unit="ms", utc=True)
        else:          # s
            return pd.to_datetime(int(xv), unit="s", utc=True)
    except Exception:
        try:
            return pd.to_datetime(x, utc=True)
        except Exception:
            return pd.NaT


def _norm_side(x) -> int:
    """Return +1 for long/buy, -1 for short/sell."""
    if isinstance(x, (int, float)):
        return 1 if float(x) >= 0 else -1
    s = str(x).strip().lower()
    if s in ("long", "buy", "b", "1", "+1"):
        return 1
    if s in ("short", "sell", "s", "-1"):
        return -1
    raise ValueError(f"Unrecognized side: {x}")


def _get_first(df_row, *cands, default=None):
    for c in cands:
        if c in df_row and pd.notna(df_row[c]):
            return df_row[c]
    return default


@dataclass
class SymbolRules:
    qty_step: float = 0.001          # e.g., BTC 0.001
    min_qty: float = 0.0             # exchange minimum quantity
    min_notional: float = 0.0        # exchange minimum notional in USDT

    def round_qty(self, qty: float) -> float:
        if self.qty_step <= 0:
            return qty
        # floor to step so we never exceed risk due to rounding
        steps = math.floor(qty / self.qty_step)
        return max(0.0, steps * self.qty_step)

    def meets_mins(self, qty: float, entry_price: float) -> bool:
        if qty <= 0:
            return False
        if self.min_qty and qty < self.min_qty:
            return False
        notional = qty * float(entry_price)
        if self.min_notional and notional < self.min_notional:
            return False
        return True


@dataclass
class RiskConfig:
    initial_balance: float
    flat_risk_until: float
    flat_risk_amount: float
    percent_risk: float                  # e.g., 0.005 for 0.5%
    per_trade_risk_cap: float
    total_open_risk_cap: float           # absolute USDT cap across all open trades
    max_concurrent_trades: int
    fee_bps_per_side: float              # e.g., 5.0 = 0.05% per side
    sizing_policy: str = "scale_to_fit"  # "scale_to_fit" or "skip"
    default_qty_step: float = 0.001
    default_min_qty: float = 0.0
    default_min_notional: float = 0.0
    symbol_overrides: Dict[str, Dict[str, float]] = None

    def rules_for(self, symbol: str) -> SymbolRules:
        sym = symbol.upper()
        o = (self.symbol_overrides or {}).get(sym, {})
        return SymbolRules(
            qty_step=o.get("qty_step", self.default_qty_step),
            min_qty=o.get("min_qty", self.default_min_qty),
            min_notional=o.get("min_notional", self.default_min_notional),
        )


@dc.dataclass
class OpenPosition:
    trade_id: str
    symbol: str
    side: int            # +1 long, -1 short
    qty: float
    entry_price: float
    sl_price: float
    open_risk: float     # qty * |entry - sl|
    notional: float
    entry_fee: float
    entry_time: pd.Timestamp


class RiskEngine:
    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg
        self.balance = float(cfg.initial_balance)
        self.open_positions: Dict[str, OpenPosition] = {}
        self.total_open_risk: float = 0.0
        self.equity_points: List[Tuple[pd.Timestamp, float]] = []  # (time, balance after close)

    # ---------- sizing helpers ----------
    def risk_target(self) -> float:
        """Compute per-trade risk target given current balance."""
        if self.balance < self.cfg.flat_risk_until:
            r = self.cfg.flat_risk_amount
        else:
            r = self.cfg.percent_risk * self.balance
        return min(r, self.cfg.per_trade_risk_cap)

    def remaining_risk_capacity(self) -> float:
        return max(0.0, self.cfg.total_open_risk_cap - self.total_open_risk)

    def _fees_for_notional(self, notional: float) -> float:
        return float(self.cfg.fee_bps_per_side) * notional / 10_000.0

    # ---------- core API ----------
    def try_open(self, trade_id: str, symbol: str, side: int, entry_price: float, sl_price: float,
                 entry_time: pd.Timestamp) -> Dict[str, Any]:
        """Attempt to open a new position. Returns a dict with allocation results and possible skip reason."""
        res = {
            "trade_id": trade_id, "symbol": symbol, "side": side,
            "entry_time": entry_time, "entry_price": entry_price, "sl_price": sl_price,
            "risk_mode": None, "risk_target": None, "risk_allocated": 0.0,
            "qty": 0.0, "notional": 0.0, "entry_fee": 0.0,
            "open_concurrent_after": len(self.open_positions),
            "open_risk_after": self.total_open_risk,
            "skipped_reason": ""
        }

        # Enforce max concurrent trades
        if len(self.open_positions) >= self.cfg.max_concurrent_trades:
            res["skipped_reason"] = "max_concurrent_reached"
            return res

        # Compute per-trade risk target
        r_target = self.risk_target()
        res["risk_mode"] = "flat" if self.balance < self.cfg.flat_risk_until else "percent"
        res["risk_target"] = r_target

        # Enforce total-open-risk cap
        remaining = self.remaining_risk_capacity()
        if remaining <= 0:
            res["skipped_reason"] = "open_risk_cap_reached"
            return res

        r = min(r_target, remaining)
        if r <= 0:
            res["skipped_reason"] = "no_risk_capacity"
            return res

        # If scaling is disabled and r < r_target → skip
        if self.cfg.sizing_policy != "scale_to_fit" and r < r_target:
            res["skipped_reason"] = "would_exceed_open_risk_cap"
            return res

        # Compute qty from risk and stop distance (linear USDT)
        risk_per_unit = abs(float(entry_price) - float(sl_price))
        if risk_per_unit <= 0:
            res["skipped_reason"] = "invalid_stop_distance"
            return res

        rules = self.cfg.rules_for(symbol)
        qty_raw = r / risk_per_unit
        qty = rules.round_qty(qty_raw)

        # If qty rounds to zero, try to skip or (optionally) bump? We skip to keep risk discipline.
        if qty <= 0:
            res["skipped_reason"] = "qty_rounded_to_zero"
            return res

        # Check exchange mins within allowed risk (no topping up beyond risk)
        if not rules.meets_mins(qty, entry_price):
            res["skipped_reason"] = "exchange_mins_violate_risk"
            return res

        notional = qty * float(entry_price)
        entry_fee = self._fees_for_notional(notional)

        # Effective allocated risk after rounding (<= r)
        open_risk = qty * risk_per_unit

        # Register position
        pos = OpenPosition(
            trade_id=trade_id, symbol=symbol, side=side, qty=qty,
            entry_price=float(entry_price), sl_price=float(sl_price),
            open_risk=open_risk, notional=notional, entry_fee=entry_fee,
            entry_time=entry_time,
        )
        self.open_positions[trade_id] = pos
        self.total_open_risk += open_risk

        res.update({
            "risk_allocated": open_risk,
            "qty": qty,
            "notional": notional,
            "entry_fee": entry_fee,
            "open_concurrent_after": len(self.open_positions),
            "open_risk_after": self.total_open_risk,
        })
        return res

    def close(self, trade_id: str, exit_price: float, exit_time: pd.Timestamp) -> Dict[str, Any]:
        """Close an existing position and realize PnL."""
        if trade_id not in self.open_positions:
            # Unknown trade_id; return a no-op (useful for robustness in batch sims)
            return {
                "trade_id": trade_id, "exit_time": exit_time, "exit_price": exit_price,
                "pnl": 0.0, "pnl_after_fees": 0.0, "R_multiple": 0.0,
                "balance_after": self.balance, "closed": False, "reason": "not_open"
            }

        pos = self.open_positions.pop(trade_id)
        self.total_open_risk -= pos.open_risk
        # Signed PnL
        price_diff = (float(exit_price) - pos.entry_price) * pos.side
        gross_pnl = pos.qty * price_diff
        exit_fee = self._fees_for_notional(pos.qty * float(exit_price))
        pnl_after_fees = gross_pnl - pos.entry_fee - exit_fee

        # R (using allocated open_risk, not theoretical r_target)
        R = pnl_after_fees / pos.open_risk if pos.open_risk > 0 else 0.0

        # Update balance
        self.balance += pnl_after_fees
        self.equity_points.append((exit_time, self.balance))

        return {
            "trade_id": trade_id, "symbol": pos.symbol, "side": pos.side,
            "entry_time": pos.entry_time, "exit_time": exit_time,
            "entry_price": pos.entry_price, "exit_price": float(exit_price),
            "qty": pos.qty, "entry_fee": pos.entry_fee, "exit_fee": exit_fee,
            "risk_allocated": pos.open_risk, "pnl": gross_pnl,
            "pnl_after_fees": pnl_after_fees, "R_multiple": R,
            "balance_after": self.balance, "closed": True, "reason": "ok"
        }


# --------- Batch simulator (reads a CSV and produces results) ---------
def run_simulation(cfg: RiskConfig, trades_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Normalize columns
    df = trades_df.copy()
    cols = {c.lower().strip(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    c_symbol = pick("symbol")
    c_side = pick("side")
    c_entry_t = pick("timestamp", "entry_time")
    c_entry = pick("entry_price", "entry")
    c_sl    = pick("sl_price", "stop", "sl")
    c_exit_t = pick("exit_time")
    c_exit   = pick("exit_price", "exit")
    if not all([c_symbol, c_side, c_entry_t, c_entry, c_sl]):
        raise ValueError("Missing required columns. Need at least: symbol, side, timestamp/entry_time, entry_price/entry, sl_price/stop/sl")

    # Create a stable trade_id
    def mk_id(i, row):
        t = _to_dt(row[c_entry_t])
        return f"{row[c_symbol]}_{int(t.value//10**6) if pd.notna(t) else i}"

    df["_trade_id"] = [mk_id(i, r) for i, r in df.iterrows()]
    df["_side"] = df[c_side].apply(_norm_side)
    df["_entry_time"] = df[c_entry_t].map(_to_dt)
    if c_exit_t:
        df["_exit_time"] = df[c_exit_t].map(_to_dt)
    if c_exit:
        df["_exit_price"] = df[c_exit].astype(float)

    # Sort by entry time, then index to keep determinism
    df = df.sort_values(by=["_entry_time", "_trade_id"]).reset_index(drop=True)

    engine = RiskEngine(cfg)
    open_records = []
    close_records = []

    # Attempt to open all trades in order
    for _, row in df.iterrows():
        trade_id = row["_trade_id"]
        rec = engine.try_open(
            trade_id=trade_id,
            symbol=str(row[c_symbol]).upper(),
            side=int(row["_side"]),
            entry_price=float(row[c_entry]),
            sl_price=float(row[c_sl]),
            entry_time=row["_entry_time"],
        )
        open_records.append(rec)

        # If we have an exit for this trade, close it now (simple 1:1 enter->exit model)
        has_exit = (c_exit_t in row and pd.notna(row[c_exit_t])) and (c_exit in row and pd.notna(row[c_exit]))
        if has_exit and rec.get("skipped_reason", "") == "":
            close_rec = engine.close(trade_id, float(row[c_exit]), row["_exit_time"])
            # Carry through any optional columns
            for keep_col in ("tp_price", "outcome", "R_multiple"):
                if keep_col in df.columns:
                    close_rec[keep_col] = row.get(keep_col)
            close_records.append(close_rec)
        elif has_exit:
            # Trade had an exit but was skipped on entry -> mark as skipped close (no effect)
            close_records.append({
                "trade_id": trade_id, "symbol": str(row[c_symbol]).upper(), "side": int(row["_side"]),
                "entry_time": row["_entry_time"], "exit_time": row["_exit_time"],
                "entry_price": float(row[c_entry]), "exit_price": float(row[c_exit]),
                "qty": 0.0, "entry_fee": 0.0, "exit_fee": 0.0,
                "risk_allocated": 0.0, "pnl": 0.0, "pnl_after_fees": 0.0,
                "R_multiple": 0.0, "balance_after": engine.balance, "closed": False, "reason": "skipped_on_entry"
            })

    open_df = pd.DataFrame(open_records)
    close_df = pd.DataFrame(close_records)

    # Equity curve (only after closes)
    eq = pd.DataFrame(engine.equity_points, columns=["time", "balance"]).sort_values("time")
    return (open_df, eq if len(eq) else pd.DataFrame(columns=["time", "balance"])), close_df


def load_config(path: str) -> RiskConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return RiskConfig(
        initial_balance=float(raw["initial_balance"]),
        flat_risk_until=float(raw["flat_risk_until"]),
        flat_risk_amount=float(raw["flat_risk_amount"]),
        percent_risk=float(raw["percent_risk"]),
        per_trade_risk_cap=float(raw["per_trade_risk_cap"]),
        total_open_risk_cap=float(raw["total_open_risk_cap"]),
        max_concurrent_trades=int(raw["max_concurrent_trades"]),
        fee_bps_per_side=float(raw["fee_bps_per_side"]),
        sizing_policy=str(raw.get("sizing_policy", "scale_to_fit")),
        default_qty_step=float(raw.get("default_qty_step", 0.001)),
        default_min_qty=float(raw.get("default_min_qty", 0.0)),
        default_min_notional=float(raw.get("default_min_notional", 0.0)),
        symbol_overrides=raw.get("symbol_overrides", {})
    )


if __name__ == "__main__":
    import argparse, sys, os
    parser = argparse.ArgumentParser(description="TC1 Risk Overlay Simulator (standalone)")
    parser.add_argument("--config", required=True, help="Path to risk_overlay config JSON")
    parser.add_argument("--trades_csv", required=True, help="CSV of trades (entries+exits)")
    parser.add_argument("--out_csv", required=True, help="Output CSV (per-trade allocation & entry results)")
    parser.add_argument("--equity_csv", required=True, help="Output CSV (equity curve after each close)")
    parser.add_argument("--closed_csv", required=False, help="Optional CSV of realized PnL after closes")
    args = parser.parse_args()

    cfg = load_config(args.config)
    trades = pd.read_csv(args.trades_csv)

    (open_df, eq_df), closed_df = run_simulation(cfg, trades)

    # Ensure output folders
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.equity_csv), exist_ok=True)
    if args.closed_csv:
        os.makedirs(os.path.dirname(args.closed_csv), exist_ok=True)

    open_df.to_csv(args.out_csv, index=False)
    eq_df.to_csv(args.equity_csv, index=False)
    if args.closed_csv:
        closed_df.to_csv(args.closed_csv, index=False)

    print(f"[OK] Wrote:\n - {args.out_csv}\n - {args.equity_csv}")
    if args.closed_csv:
        print(f" - {args.closed_csv}")
