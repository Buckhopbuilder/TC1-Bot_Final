#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np, pandas as pd

def load_ohlc(root_dir, symbol):
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

    def find(name):
        for c in df.columns:
            if c.lower() == name: return c
        for c in df.columns:
            if name in c.lower(): return c
        return None

    oc = find("open"); hc = find("high"); lc = find("low"); cc = find("close")
    out = pd.DataFrame({
        "ts": ts.astype("int64"),
        "open": pd.to_numeric(df[oc], errors="coerce"),
        "high": pd.to_numeric(df[hc], errors="coerce"),
        "low" : pd.to_numeric(df[lc], errors="coerce"),
        "close": pd.to_numeric(df[cc], errors="coerce"),
    }).dropna().sort_values("ts").reset_index(drop=True)
    return out

def detect_fvgs(ohlc_lb):
    """
    ICT 3-candle FVGs over a lookback slice.
    Bullish FVG when high[i] < low[i+2] -> gap band [high[i], low[i+2]]
    Bearish FVG when low[i] > high[i+2] -> gap band [high[i+2], low[i]]
    Returns DataFrame with columns: kind ('bull'|'bear'), lo, hi, ts_i (gap 'center' time = ts[i+1])
    """
    h = ohlc_lb["high"].values
    l = ohlc_lb["low"].values
    ts = ohlc_lb["ts"].values
    rows=[]
    # need i, i+1, i+2
    for i in range(0, len(ohlc_lb)-2):
        # bullish gap
        if h[i] < l[i+2]:
            rows.append(dict(kind="bull", lo=h[i], hi=l[i+2], ts_i=ts[i+1]))
        # bearish gap
        if l[i] > h[i+2]:
            rows.append(dict(kind="bear", lo=h[i+2], hi=l[i], ts_i=ts[i+1]))
    return pd.DataFrame(rows)

def bands_overlap_price_interval(lo1, hi1, lo2, hi2):
    return (max(lo1, lo2) <= min(hi1, hi2))

def first_touch_outcome(ohlc, start_ts, entry, sl, tp, max_bars_wait=288, max_bars_in_trade=288):
    # wait for entry touch
    i = int(np.searchsorted(ohlc["ts"].values, start_ts, side="left"))
    n = len(ohlc); waited=0
    while i < n:
        if max_bars_wait is not None and waited >= max_bars_wait:
            return "no_fill", waited, 0
        hi = ohlc.at[i,"high"]; lo = ohlc.at[i,"low"]
        if lo <= entry <= hi:
            break
        i += 1; waited += 1
    if i >= n: return "no_fill", waited, 0
    # after fill
    j=i; inbars=0
    while j < n:
        if max_bars_in_trade is not None and inbars >= max_bars_in_trade:
            return "no_fill", waited, inbars
        hi = ohlc.at[j,"high"]; lo = ohlc.at[j,"low"]
        hit_tp = (hi >= tp) if tp >= entry else (lo <= tp)
        hit_sl = (lo <= sl) if sl <= entry else (hi >= sl)
        if hit_tp and hit_sl: return "loss", waited, inbars+1  # conservative
        if hit_tp: return "win", waited, inbars+1
        if hit_sl: return "loss", waited, inbars+1
        j += 1; inbars += 1
    return "no_fill", waited, inbars

def adjust_box(side, entry, sl, tp, frac):
    if side == "long":
        d = frac * (entry - sl)
        new_entry = entry - d
    else:
        d = frac * (sl - entry)
        new_entry = entry + d
    shift = new_entry - entry
    return new_entry, sl + shift, tp + shift

def rr_from(entry, sl, tp, side):
    if side == "long":
        r = abs(entry - sl); return (tp - entry)/r if r>0 else np.nan
    else:
        r = abs(sl - entry); return (entry - tp)/r if r>0 else np.nan

def main():
    ap = argparse.ArgumentParser(description="FVG shifted-entry experiment with on-the-fly FVG detection.")
    ap.add_argument("--trades", default="reports/trades/trades.csv")
    ap.add_argument("--ohlc_dir", default="data/futures_merged_5m")
    ap.add_argument("--lookback_bars", type=int, default=300, help="Bars to scan before entry for FVGs (5m bars)")
    ap.add_argument("--shift_fracs", nargs="+", type=float, default=[0.05,0.10,0.15])
    ap.add_argument("--max_bars_wait", type=int, default=288)
    ap.add_argument("--max_bars_in_trade", type=int, default=288)
    ap.add_argument("--out_dir", default="reports/experiments/fvg_shift_detect")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    tr = pd.read_csv(args.trades)
    need = ["trade_id","symbol","side","fill_time","entry_price","sl_price","tp_price","outcome","RR_target"]
    tr = tr[[c for c in need if c in tr.columns]].dropna().reset_index(drop=True)
    tr["fill_time"] = pd.to_numeric(tr["fill_time"], errors="coerce").astype("Int64")
    tr = tr.dropna(subset=["fill_time"]).reset_index(drop=True)

    # load OHLC per symbol
    symbols = sorted(tr["symbol"].unique().tolist())
    ohlc_map = {}
    for s in symbols:
        try:
            ohlc_map[s] = load_ohlc(args.ohlc_dir, s)
        except Exception as e:
            print(f"[WARN] Skipping {s}: {e}")

    # subset to symbols we have OHLC for
    tr = tr[tr["symbol"].isin(ohlc_map.keys())].reset_index(drop=True)

    fvg_counts = {}
    qualified = []

    for _, r in tr.iterrows():
        sym = r["symbol"]; side = r["side"].lower()
        o = ohlc_map[sym]
        # get lookback slice ending at (or just before) fill_time
        idx = int(np.searchsorted(o["ts"].values, int(r["fill_time"]), side="left"))
        lb0 = max(0, idx - args.lookback_bars)
        lb = o.iloc[lb0:idx].copy()
        if len(lb) < 3: continue
        fvgs = detect_fvgs(lb)
        if fvgs.empty: 
            fvg_counts[sym] = fvg_counts.get(sym, 0)
            continue

        # price interval between entry and SL
        entry = float(r["entry_price"]); sl = float(r["sl_price"])
        lo = min(entry, sl); hi = max(entry, sl)

        # choose matching FVG kind (long cares about bullish; short cares about bearish)
        want = "bull" if side.startswith("l") else "bear"
        fvgs_want = fvgs[fvgs["kind"]==want]
        # any FVG band overlapping [lo,hi]?
        ok = False
        for _,g in fvgs_want.iterrows():
            if bands_overlap_price_interval(g["lo"], g["hi"], lo, hi):
                ok = True; break
        if ok:
            qualified.append(r)

        fvg_counts[sym] = fvg_counts.get(sym, 0) + len(fvgs_want)

    if not qualified:
        print("[INFO] No trades with detected FVGs overlapping entry↔SL. Nothing to test.")
        # Also print detector stats:
        if fvg_counts:
            total = sum(fvg_counts.values())
            print(f"[INFO] Detected {total} matching FVGs across symbols: {fvg_counts}")
        return

    df = pd.DataFrame(qualified).reset_index(drop=True)

    # Baseline on this subset
    base_wr = (df["outcome"]=="win").mean() if len(df) else np.nan
    base_expR = np.mean(np.where(df["outcome"]=="win", df["RR_target"], -1.0)) if "RR_target" in df.columns else np.nan

    rows = []; detail_rows=[]
    for frac in args.shift_fracs:
        wins=losses=nofills=0
        rs=[]
        for _, r in df.iterrows():
            sym = r["symbol"]; sd = "long" if r["side"].lower().startswith("l") else "short"
            o = ohlc_map[sym]
            entry, sl, tp = float(r["entry_price"]), float(r["sl_price"]), float(r["tp_price"])
            new_entry, new_sl, new_tp = adjust_box(sd, entry, sl, tp, frac)
            rr = rr_from(new_entry, new_sl, new_tp, sd)

            status, wait_bars, in_bars = first_touch_outcome(
                o, int(r["fill_time"]), new_entry, new_sl, new_tp,
                max_bars_wait=args.max_bars_wait, max_bars_in_trade=args.max_bars_in_trade
            )
            if status == "no_fill":
                nofills += 1
                outcome="no_fill"; rnet=0.0
            elif status == "win":
                wins += 1
                outcome="win"; rnet=rr
            else:
                losses += 1
                outcome="loss"; rnet=-1.0

            detail_rows.append(dict(
                trade_id=r["trade_id"], symbol=sym, side=sd, shift_frac=frac,
                new_entry=new_entry, new_sl=new_sl, new_tp=new_tp, rr=rr,
                sim_outcome=outcome, wait_bars=wait_bars, bars_in_trade=in_bars
            ))
            if outcome != "no_fill": rs.append(rnet)

        filled = wins + losses
        wr = (wins/filled) if filled else np.nan
        expR = (np.mean(rs) if rs else np.nan)
        rows.append(dict(
            shift_frac=frac, subset_trades=len(df), filled=filled, no_fills=nofills,
            winrate_filled=wr, expectancy_R_filled=expR,
            baseline_winrate=base_wr, baseline_expectancy_R=base_expR
        ))

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows); detail = pd.DataFrame(detail_rows)
    summary.to_csv(out_dir/"fvg_shift_summary.csv", index=False)
    detail.to_csv(out_dir/"fvg_shift_details.csv", index=False)

    print("=== FVG Shift (auto-detect) ===")
    print(summary.round(4).to_string(index=False))
    print(f"\n[DONE] Summary → {out_dir/'fvg_shift_summary.csv'}")
    print(f"[DONE] Details → {out_dir/'fvg_shift_details.csv'}")
    print(f"[INFO] Matching FVG counts per symbol: {fvg_counts}")

if __name__ == "__main__":
    main()
