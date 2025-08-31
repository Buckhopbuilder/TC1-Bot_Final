#!/usr/bin/env python3
import argparse, math
from pathlib import Path
import numpy as np, pandas as pd

# --- Utils ---
def to_ts_ms(s):
    try:
        s = pd.to_datetime(s, utc=True)
        return (s.view("int64") // 10**6)
    except Exception:
        return np.nan

def load_ohlc(root_dir, symbol):
    """
    Load per-symbol 5m OHLC. Tries a few common schemas.
    Must return columns: ts (ms), open, high, low, close
    """
    fp = Path(root_dir) / f"{symbol}.csv"
    if not fp.exists():
        raise FileNotFoundError(f"OHLC file not found for {symbol}: {fp}")
    df = pd.read_csv(fp)
    # pick a timestamp column
    ts_cols = [c for c in df.columns if c.lower() in ("open_time","timestamp","time","ts","date","datetime")]
    if not ts_cols:
        raise ValueError(f"No time column in {fp}")
    tcol = ts_cols[0]
    # build ts in ms
    if np.issubdtype(df[tcol].dtype, np.number) and df[tcol].max() > 10**12:
        ts = df[tcol].astype("int64")
    else:
        ts = pd.to_datetime(df[tcol], utc=True, errors="coerce").view("int64") // 10**6
    # map price columns
    def find(name):
        for c in df.columns:
            cl = c.lower()
            if cl == name: return c
        for c in df.columns:
            cl = c.lower()
            if name in cl: return c
        return None
    oc = find("open"); hc = find("high"); lc = find("low"); cc = find("close")
    if any(c is None for c in (oc,hc,lc,cc)):
        raise ValueError(f"Missing OHLC columns in {fp}. Found: {df.columns.tolist()}")
    out = pd.DataFrame({
        "ts": ts.astype("int64"),
        "open": pd.to_numeric(df[oc], errors="coerce"),
        "high": pd.to_numeric(df[hc], errors="coerce"),
        "low" : pd.to_numeric(df[lc], errors="coerce"),
        "close": pd.to_numeric(df[cc], errors="coerce"),
    }).dropna().sort_values("ts").reset_index(drop=True)
    return out

def first_touch_outcome(ohlc, start_ts, entry, sl, tp, max_bars=None):
    """
    Simulate from start_ts forward.
    Step 1: wait for price to touch entry (wick fill).
    Step 2: after filled, whichever of SL/TP is touched first decides outcome.
    Returns: ("no_fill"|"win"|"loss", bars_wait, bars_in_trade)
    """
    # find index at/after start_ts
    i = int(np.searchsorted(ohlc["ts"].values, start_ts, side="left"))
    n = len(ohlc)
    waited = 0
    # Wait for fill
    while i < n:
        if max_bars is not None and waited >= max_bars:
            return "no_fill", waited, 0
        hi = ohlc.at[i,"high"]; lo = ohlc.at[i,"low"]
        if lo <= entry <= hi:
            # filled at this bar
            break
        i += 1; waited += 1
    if i >= n:
        return "no_fill", waited, 0
    # After fill, check who hits first
    filled_idx = i
    j = i
    in_trade_bars = 0
    while j < n:
        if max_bars is not None and in_trade_bars >= max_bars:
            # Treat as no decision (skip)
            return "no_fill", waited, in_trade_bars
        hi = ohlc.at[j,"high"]; lo = ohlc.at[j,"low"]
        # For longs, SL < entry < TP. For shorts, SL > entry > TP.
        # Use touch rule (wick)
        hit_tp = (hi >= tp) if tp >= entry else (lo <= tp)
        hit_sl = (lo <= sl) if sl <= entry else (hi >= sl)
        if hit_tp and hit_sl:
            # If both in same bar, assume worst case (SL first) to be conservative
            return "loss", waited, in_trade_bars+1
        if hit_tp: return "win", waited, in_trade_bars+1
        if hit_sl: return "loss", waited, in_trade_bars+1
        j += 1; in_trade_bars += 1
    return "no_fill", waited, in_trade_bars

def adjust_box(side, entry, sl, tp, frac):
    """Shift the entire box toward SL by frac of (entry - sl) (long) or (sl - entry) (short)."""
    if side == "long":
        d = frac * (entry - sl)
        new_entry = entry - d
        shift = new_entry - entry
        return new_entry, sl + shift, tp + shift
    else:  # short
        d = frac * (sl - entry)
        new_entry = entry + d
        shift = new_entry - entry
        return new_entry, sl + shift, tp + shift

def rr_from(entry, sl, tp, side):
    if side == "long":
        r = abs(entry - sl)
        return (tp - entry) / r if r>0 else np.nan
    else:
        r = abs(sl - entry)
        return (entry - tp) / r if r>0 else np.nan

# --- Main ---
def main():
    ap = argparse.ArgumentParser(description="Quick FVG shifted-entry experiment.")
    ap.add_argument("--trades", default="reports/trades/trades.csv")
    ap.add_argument("--features", default="reports/features/features_at_entry.csv")
    ap.add_argument("--ohlc_dir", default="data/futures_merged_5m", help="Folder with per-symbol 5m OHLC CSVs")
    ap.add_argument("--shift_fracs", nargs="+", type=float, default=[0.05,0.10,0.15], help="Fractions to shift toward SL (e.g., 0.10=10%)")
    ap.add_argument("--max_bars_wait", type=int, default=288, help="Cap bars to wait for adjusted entry (~1 day on 5m)")
    ap.add_argument("--max_bars_in_trade", type=int, default=288, help="Cap bars after fill to detect TP/SL (~1 day)")
    ap.add_argument("--out_dir", default="reports/experiments/fvg_shift")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    tr = pd.read_csv(args.trades)
    fe = pd.read_csv(args.features)
    # keep the essentials
    keep = ["trade_id","symbol","side","fill_time","entry_price","sl_price","tp_price","outcome","RR_target"]
    tr = tr[[c for c in keep if c in tr.columns]].copy()

    # merge FVG flags
    fcols = ["trade_id","fvg_bull","fvg_bear"]
    for c in fcols:
        if c not in fe.columns: 
            fe[c] = np.nan
    df = tr.merge(fe[fcols], on="trade_id", how="left")

    # ensure time in ms
    if "fill_time" not in df.columns:
        raise ValueError("trades.csv must have fill_time (ms since epoch) to simulate forward.")
    df["fill_time"] = pd.to_numeric(df["fill_time"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["fill_time","symbol","side","entry_price","sl_price","tp_price"]).reset_index(drop=True)

    # subset: FVG present matching side
    def fvg_ok(row):
        if row["side"] == "long":
            return row.get("fvg_bull", 0) == 1
        else:
            return row.get("fvg_bear", 0) == 1
    df = df[df.apply(fvg_ok, axis=1)].reset_index(drop=True)

    if df.empty:
        print("[INFO] No FVG-flagged trades found. Nothing to test.")
        return

    # Load OHLC grouped by symbol once
    symbols = sorted(df["symbol"].unique().tolist())
    ohlc_map = {}
    for s in symbols:
        try:
            ohlc_map[s] = load_ohlc(args.ohlc_dir, s)
        except Exception as e:
            print(f"[WARN] Skipping {s} due to OHLC load error: {e}")
    df = df[df["symbol"].isin(ohlc_map.keys())].reset_index(drop=True)

    # Baseline stats on this subset
    base_wr = (df["outcome"]=="win").mean() if len(df) else np.nan
    base_expR = np.mean(np.where(df["outcome"]=="win", df["RR_target"], -1.0)) if "RR_target" in df.columns else np.nan

    rows = []
    detail_rows = []

    for frac in args.shift_fracs:
        wins = losses = nofills = 0
        rs = []
        for _, r in df.iterrows():
            sym = r["symbol"]; side = r["side"]
            entry, sl, tp = float(r["entry_price"]), float(r["sl_price"]), float(r["tp_price"])
            # sanity side
            sd = "long" if side.lower().startswith("l") else "short"

            # adjust
            new_entry, new_sl, new_tp = adjust_box(sd, entry, sl, tp, frac)
            rr = rr_from(new_entry, new_sl, new_tp, sd)

            # simulate
            ohlc = ohlc_map[sym]
            status, wait_bars, in_bars = first_touch_outcome(
                ohlc, int(r["fill_time"]), new_entry, new_sl, new_tp,
                max_bars=args.max_bars_wait
            )
            if status == "no_fill":
                nofills += 1
                outcome = "no_fill"
                rnet = 0.0  # not counted in expectancy; we'll compute over filled only
            elif status == "win":
                wins += 1
                outcome = "win"
                rnet = rr
            else:
                losses += 1
                outcome = "loss"
                rnet = -1.0

            detail_rows.append({
                "trade_id": r["trade_id"], "symbol": sym, "side": sd, "shift_frac": frac,
                "new_entry": new_entry, "new_sl": new_sl, "new_tp": new_tp,
                "rr": rr, "sim_outcome": outcome,
                "wait_bars": wait_bars, "bars_in_trade": in_bars
            })
            if outcome != "no_fill":
                rs.append(rnet)

        filled = wins + losses
        wr = (wins/filled) if filled else np.nan
        expR = (np.mean(rs) if rs else np.nan)
        rows.append({
            "shift_frac": frac, "subset_trades": len(df),
            "filled": filled, "no_fills": nofills,
            "winrate_filled": wr, "expectancy_R_filled": expR,
            "baseline_winrate": base_wr, "baseline_expectancy_R": base_expR
        })

    # Save outputs
    out_dir = Path(args.out_dir)
    summary = pd.DataFrame(rows)
    detail = pd.DataFrame(detail_rows)
    summary.to_csv(out_dir/"fvg_shift_summary.csv", index=False)
    detail.to_csv(out_dir/"fvg_shift_details.csv", index=False)

    print("=== FVG Shift Quick Test ===")
    print(summary.round(4).to_string(index=False))
    print(f"\n[DONE] Summary → {out_dir/'fvg_shift_summary.csv'}")
    print(f"[DONE] Details → {out_dir/'fvg_shift_details.csv'}")

if __name__ == "__main__":
    main()
