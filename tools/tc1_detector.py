#!/usr/bin/env python3
"""
TC1 Detector & Simulator (Thread A)

- Reads merged Binance Futures 5m CSVs with columns:
    time,open,high,low,close,volume
- Detects swing pivots (2-left / 2-right) to form legs (A -> B):
    * Long leg:  A = swing low,  B = subsequent swing high
    * Short leg: A = swing high, B = subsequent swing low
- Filters legs by size: 1.2% <= |B-A|/A*100 <= 6.0%
- Computes TC1 levels (fib anchored as agreed):
    Long:  entry = A + 0.382*(B-A),  stop = A + 0.170*(B-A),  tp = B + 0.272*(B-A)
    Short: entry = A - 0.382*(A-B),  stop = A - 0.170*(A-B),  tp = B - 0.272*(A-B)
- Entry trigger (default): wick-touch
    Long  fills when low <= entry
    Short fills when high >= entry
- Post-entry outcome (conservative default):
    Within each bar, assume SL-first if both SL and TP touched
    -> outcome ∈ {"win","loss"} and ambiguous ∈ {0,1}
- Overlapping trades are allowed (each leg is independent).
- Outputs CSV at --out with columns:

    trade_id,symbol,side,leg_id,
    A_time,A_price,B_time,B_price,leg_bars,leg_pct,
    entry_price,sl_price,tp_price,
    fill_time,exit_time,exit_price,outcome,ambiguous,
    bars_in_trade,RR_target

Notes:
- We start searching for entry **after B is confirmed**, i.e. from index (i_B + pivot_right).
- Time values are epoch milliseconds from source data.
- No fees/funding; no look-ahead beyond pivot confirmation.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import pandas as pd
import numpy as np


# -----------------------------
# Config defaults (can override via CLI)
# -----------------------------
DEFAULT_MIN_LEG_PCT = 1.2
DEFAULT_MAX_LEG_PCT = 6.0
DEFAULT_PIVOT_LEFT = 2
DEFAULT_PIVOT_RIGHT = 2
DEFAULT_TRIGGER = "wick"   # "wick" or "close" (we log only; core logic uses "wick" as baseline)
AMBIGUITY_MODE = "sl_first"  # "sl_first" | "tp_first" | "exclude" (we output 'ambiguous' flag regardless)


@dataclass
class Leg:
    side: str            # "long" or "short"
    idx_A: int
    A_price: float
    time_A: int
    idx_B: int
    B_price: float
    time_B: int
    leg_bars: int
    leg_pct: float       # 100 * |B-A|/A


@dataclass
class Trade:
    trade_id: int
    symbol: str
    side: str
    leg_id: str
    A_time: int
    A_price: float
    B_time: int
    B_price: float
    leg_bars: int
    leg_pct: float
    entry_price: float
    sl_price: float
    tp_price: float
    fill_time: int
    exit_time: int
    exit_price: float
    outcome: str         # "win" | "loss"
    ambiguous: int       # 0 | 1
    bars_in_trade: int
    RR_target: float


def load_ohlcv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Ensure expected columns exist and types are sane
    expected = ["time","open","high","low","close","volume"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    # Coerce types
    for c in expected:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=expected)
    df = df.sort_values("time").reset_index(drop=True)
    return df


def find_pivots(df: pd.DataFrame, left: int, right: int):
    """
    Return arrays of pivot types and indices.
    pivot_types[i] ∈ {'H','L', None}
    Definition (strict):
      - Pivot High at i if high[i] is strictly greater than highs in [i-left, i+right]
        and strictly greater than high[i-1] and high[i+1]
      - Pivot Low  at i if low[i]  is strictly lower  than lows  in [i-left, i+right]
        and strictly lower than low[i-1]  and low[i+1]
    Edges (i < left or i > n-1-right) cannot be pivots.
    """
    high = df["high"].values
    low  = df["low"].values
    n = len(df)
    types = np.array([None]*n, dtype=object)

    for i in range(left, n-right):
        win_hi = high[i-left:i+right+1]
        win_lo = low[i-left:i+right+1]
        h = high[i]
        l = low[i]

        # ensure neighbors exist for strict comparison
        if i-1 < 0 or i+1 >= n:
            continue

        is_H = (h == np.max(win_hi)) and (h > high[i-1]) and (h > high[i+1]) \
               and np.sum(win_hi == h) == 1  # unique peak
        is_L = (l == np.min(win_lo)) and (l < low[i-1])  and (l < low[i+1])  \
               and np.sum(win_lo == l) == 1  # unique trough

        if is_H:
            types[i] = 'H'
        elif is_L:
            types[i] = 'L'
    return types


def build_legs(df: pd.DataFrame, piv_types, left: int, right: int,
               min_leg_pct: float, max_leg_pct: float):
    """
    Build legs from consecutive opposite-type pivots:
      L -> H => long leg
      H -> L => short leg
    """
    t = df["time"].values
    price_for = {'H': df["high"].values, 'L': df["low"].values}

    # Collect ordered pivots (idx, type)
    pivots = [(i, piv_types[i]) for i in range(len(piv_types)) if piv_types[i] in ('H','L')]

    legs = []
    for j in range(len(pivots) - 1):
        i0, t0 = pivots[j]
        i1, t1 = pivots[j+1]
        if t0 == t1:
            continue
        # Ensure "subsequent" (i1 > i0)
        if i1 <= i0:
            continue

        if t0 == 'L' and t1 == 'H':
            A = price_for['L'][i0]
            B = price_for['H'][i1]
            side = "long"
        elif t0 == 'H' and t1 == 'L':
            A = price_for['H'][i0]
            B = price_for['L'][i1]
            side = "short"
        else:
            continue

        # Leg size filter
        leg_pct = abs(B - A) / A * 100.0
        if leg_pct < min_leg_pct or leg_pct > max_leg_pct:
            continue

        legs.append(Leg(
            side=side,
            idx_A=i0,
            A_price=float(A),
            time_A=int(t[i0]),
            idx_B=i1,
            B_price=float(B),
            time_B=int(t[i1]),
            leg_bars=int(i1 - i0),
            leg_pct=float(leg_pct)
        ))
    return legs


def tc1_levels(leg: Leg):
    """Return entry, sl, tp based on agreed fib rules."""
    A = leg.A_price
    B = leg.B_price
    R = (B - A) if leg.side == "long" else (A - B)

    if leg.side == "long":
        entry = A + 0.382 * R
        sl    = A + 0.170 * R
        tp    = B + 0.272 * R
    else:
        entry = A - 0.382 * R
        sl    = A - 0.170 * R
        tp    = B - 0.272 * R
    return float(entry), float(sl), float(tp)


def simulate_trade(df: pd.DataFrame, leg: Leg, pivot_right: int,
                   trigger: str = "wick",
                   ambiguity_mode: str = "sl_first") -> Trade | None:
    """
    Simulate one TC1 trade for a given leg:
      - Search for entry starting at idx = leg.idx_B + pivot_right
      - Wick trigger (default):
          Long:  low <= entry
          Short: high >= entry
      - After fill, walk forward until SL or TP is hit.
        If both touched same bar => ambiguous=1 and apply ambiguity_mode.
    """
    time = df["time"].values
    high = df["high"].values
    low  = df["low"].values

    entry, sl, tp = tc1_levels(leg)

    # Search for fill after B confirmation
    start = leg.idx_B + pivot_right
    if start >= len(df):
        return None

    fill_idx = None
    # Trigger selection (we keep "wick" as baseline; allow "close" if needed)
    if trigger == "wick":
        if leg.side == "long":
            for i in range(start, len(df)):
                if low[i] <= entry:
                    fill_idx = i
                    break
        else:  # short
            for i in range(start, len(df)):
                if high[i] >= entry:
                    fill_idx = i
                    break
    elif trigger == "close":
        close = df["close"].values
        if leg.side == "long":
            for i in range(start, len(df)):
                if close[i] <= entry:  # close crosses below/at entry
                    fill_idx = i
                    break
        else:
            for i in range(start, len(df)):
                if close[i] >= entry:
                    fill_idx = i
                    break
    else:
        raise ValueError(f"Unknown trigger: {trigger}")

    if fill_idx is None:
        return None  # never filled

    # After fill: walk forward bar by bar
    exit_idx = None
    outcome = None
    ambiguous = 0

    for i in range(fill_idx, len(df)):
        hi = high[i]
        lo = low[i]

        hit_sl = (lo <= sl) if leg.side == "long" else (hi >= sl)
        hit_tp = (hi >= tp) if leg.side == "long" else (lo <= tp)

        if hit_sl and hit_tp:
            ambiguous = 1
            if ambiguity_mode == "sl_first":
                outcome = "loss"
                exit_idx = i
                exit_price = sl
            elif ambiguity_mode == "tp_first":
                outcome = "win"
                exit_idx = i
                exit_price = tp
            elif ambiguity_mode == "exclude":
                return None  # drop ambiguous trades
            else:
                raise ValueError("Invalid ambiguity_mode")
            break

        if hit_sl:
            outcome = "loss"
            exit_idx = i
            exit_price = sl
            break
        if hit_tp:
            outcome = "win"
            exit_idx = i
            exit_price = tp
            break

    if exit_idx is None:
        # trade remains open at end of data -> drop from stats (or keep partial)
        return None

    RR_target = ((tp - entry) / (entry - sl)) if leg.side == "long" else ((entry - tp) / (sl - entry))
    leg_id = f"{leg.side[0].upper()}_{leg.idx_A}_{leg.idx_B}"

    return Trade(
        trade_id=-1,  # filled later
        symbol="",    # filled later
        side=leg.side,
        leg_id=leg_id,
        A_time=leg.time_A,
        A_price=leg.A_price,
        B_time=leg.time_B,
        B_price=leg.B_price,
        leg_bars=leg.leg_bars,
        leg_pct=leg.leg_pct,
        entry_price=entry,
        sl_price=sl,
        tp_price=tp,
        fill_time=int(time[fill_idx]),
        exit_time=int(time[exit_idx]),
        exit_price=float(exit_price),
        outcome=outcome,
        ambiguous=int(ambiguous),
        bars_in_trade=int(exit_idx - fill_idx + 1),
        RR_target=float(RR_target)
    )


def run_for_symbol(symbol: str, data_dir: Path, out_rows: list,
                   min_leg_pct: float, max_leg_pct: float,
                   pivot_left: int, pivot_right: int,
                   trigger: str, ambiguity_mode: str):
    csv_path = data_dir / f"{symbol}_5m.csv"
    if not csv_path.exists():
        print(f"[WARN] Missing merged CSV for {symbol}: {csv_path}")
        return

    df = load_ohlcv(csv_path)
    piv_types = find_pivots(df, left=pivot_left, right=pivot_right)
    legs = build_legs(df, piv_types, left=pivot_left, right=pivot_right,
                      min_leg_pct=min_leg_pct, max_leg_pct=max_leg_pct)

    trade_id_seq = 0
    for leg in legs:
        tr = simulate_trade(df, leg, pivot_right=pivot_right,
                            trigger=trigger, ambiguity_mode=ambiguity_mode)
        if tr is None:
            continue
        trade_id_seq += 1
        out_rows.append({
            "trade_id": trade_id_seq,
            "symbol": symbol,
            "side": tr.side,
            "leg_id": tr.leg_id,
            "A_time": tr.A_time,
            "A_price": tr.A_price,
            "B_time": tr.B_time,
            "B_price": tr.B_price,
            "leg_bars": tr.leg_bars,
            "leg_pct": tr.leg_pct,
            "entry_price": tr.entry_price,
            "sl_price": tr.sl_price,
            "tp_price": tr.tp_price,
            "fill_time": tr.fill_time,
            "exit_time": tr.exit_time,
            "exit_price": tr.exit_price,
            "outcome": tr.outcome,
            "ambiguous": tr.ambiguous,
            "bars_in_trade": tr.bars_in_trade,
            "RR_target": tr.RR_target,
            "trigger_type": trigger,
        })

    print(f"[OK] {symbol}: legs={len(legs)} trades={trade_id_seq}")


def main():
    ap = argparse.ArgumentParser(description="TC1 Detector & Simulator")
    ap.add_argument("--data", type=str, default="data/futures_merged_5m",
                    help="Directory containing <SYMBOL>_5m.csv")
    ap.add_argument("--out", type=str, default="reports/trades/trades.csv",
                    help="Output CSV path")
    ap.add_argument("--symbols", nargs="+", required=True,
                    help="Symbols, e.g. SOLUSDT ETHUSDT AVAXUSDT MANAUSDT")
    ap.add_argument("--min_leg_pct", type=float, default=DEFAULT_MIN_LEG_PCT)
    ap.add_argument("--max_leg_pct", type=float, default=DEFAULT_MAX_LEG_PCT)
    ap.add_argument("--pivot_left", type=int, default=DEFAULT_PIVOT_LEFT)
    ap.add_argument("--pivot_right", type=int, default=DEFAULT_PIVOT_RIGHT)
    ap.add_argument("--trigger", type=str, default=DEFAULT_TRIGGER, choices=["wick","close"])
    ap.add_argument("--ambiguity_mode", type=str, default=AMBIGUITY_MODE,
                    choices=["sl_first","tp_first","exclude"])
    args = ap.parse_args()

    data_dir = Path(args.data)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for sym in args.symbols:
        run_for_symbol(sym, data_dir, rows,
                       min_leg_pct=args.min_leg_pct, max_leg_pct=args.max_leg_pct,
                       pivot_left=args.pivot_left, pivot_right=args.pivot_right,
                       trigger=args.trigger, ambiguity_mode=args.ambiguity_mode)

    if not rows:
        print("[WARN] No trades generated. Check inputs and thresholds.")
        # still write an empty file with headers for reproducibility
        cols = ["trade_id","symbol","side","leg_id",
                "A_time","A_price","B_time","B_price","leg_bars","leg_pct",
                "entry_price","sl_price","tp_price",
                "fill_time","exit_time","exit_price","outcome","ambiguous",
                "bars_in_trade","RR_target","trigger_type"]
        pd.DataFrame(columns=cols).to_csv(out_path, index=False)
        return

    # Normalize trade_id to be unique across all symbols (running index)
    df_out = pd.DataFrame(rows).reset_index(drop=True)
    df_out["trade_id"] = np.arange(1, len(df_out)+1)
    df_out.to_csv(out_path, index=False)
    print(f"[DONE] Wrote {len(df_out)} trades → {out_path}")


if __name__ == "__main__":
    main()
