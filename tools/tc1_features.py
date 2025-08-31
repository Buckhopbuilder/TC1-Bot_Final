#!/usr/bin/env python3
"""
TC1 Feature Extractor (Thread B)
- Inputs:
    data/futures_merged_5m/<SYMBOL>_5m.csv
    reports/trades/trades.csv  (from Thread A)
- Outputs:
    reports/features/features_at_leg.csv
    reports/features/features_at_entry.csv
- Defaults: 5m features only; add HTFs with --htf 15m 1h 4h
- All features are computed 'as of' the bar (no look-ahead).
- Shorts get side-normalized versions for direction-aware features.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# ---------------------------
# Helpers: Indicators (vectorized)
# ---------------------------
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def sma(s, n):
    return s.rolling(n, min_periods=n).mean()

def rsi(close, n=14):
    delta = close.diff()
    up = (delta.clip(lower=0)).rolling(n, min_periods=n).mean()
    down = (-delta.clip(upper=0)).rolling(n, min_periods=n).mean()
    rs = up / (down.replace(0,np.nan))
    out = 100 - (100/(1+rs))
    return out

def macd(close, fast=12, slow=26, signal=9):
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    sig = ema(macd_line, signal)
    hist = macd_line - sig
    return macd_line, sig, hist

def atr(df, n=14):
    high = df["High"]; low = df["Low"]; close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()

def stoch(df, k=14, d=3, smooth=3):
    low_min = df["Low"].rolling(k, min_periods=k).min()
    high_max = df["High"].rolling(k, min_periods=k).max()
    pctK = 100 * (df["Close"] - low_min) / (high_max - low_min)
    pctK = pctK.rolling(smooth, min_periods=smooth).mean()
    pctD = pctK.rolling(d, min_periods=d).mean()
    return pctK, pctD

def bollinger(close, n=20, mult=2):
    ma = sma(close, n)
    sd = close.rolling(n, min_periods=n).std()
    upper = ma + mult * sd
    lower = ma - mult * sd
    width = (upper - lower) / ma
    pos = (close - ma) / (upper - lower)
    return ma, upper, lower, width, pos

def keltner(df, n=20, atr_mult=1.5):
    mid = ema(df["Close"], n)
    rng = atr(df, n)
    upper = mid + atr_mult * rng
    lower = mid - atr_mult * rng
    width = (upper - lower) / mid
    return mid, upper, lower, width

def donchian(df, n=20):
    upper = df["High"].rolling(n, min_periods=n).max()
    lower = df["Low"].rolling(n, min_periods=n).min()
    mid = (upper + lower)/2
    pos = (df["Close"] - lower) / (upper - lower)
    return upper, lower, mid, pos

def roc(close, n=10):
    return close.pct_change(n)

def momentum(close, n=10):
    return close.diff(n)

def obv(df):
    vol = df["Volume"].fillna(0)
    sign = np.sign(df["Close"].diff().fillna(0))
    return (sign * vol).cumsum()

def slope(series, n=5):
    # simple difference slope over n bars (per bar)
    return (series - series.shift(n)) / n

def wick_body_ratios(df):
    o,h,l,c = df["Open"], df["High"], df["Low"], df["Close"]
    body = (c - o).abs()
    top = h - np.maximum(c,o)
    bot = np.minimum(c,o) - l
    with np.errstate(divide='ignore', invalid='ignore'):
        top_r = np.where(body!=0, top/body, np.nan)
        bot_r = np.where(body!=0, bot/body, np.nan)
        body_r= np.where((h-l)!=0, body/(h-l), np.nan)
    return pd.Series(top_r, index=df.index), pd.Series(bot_r, index=df.index), pd.Series(body_r, index=df.index)

def fvg_flags(df):
    # 3-candle FVG (ICT-style)
    # Bullish FVG at i if Low[i] > High[i-2]
    # Bearish FVG at i if High[i] < Low[i-2]
    hi = df["High"]; lo = df["Low"]
    bull = (lo > hi.shift(2))
    bear = (hi < lo.shift(2))
    # Gap size (positive where true)
    bull_sz = (lo - hi.shift(2)).where(bull, other=0)
    bear_sz = (lo.shift(2) - hi).where(bear, other=0)
    return bull.fillna(False), bear.fillna(False), bull_sz.fillna(0), bear_sz.fillna(0)

def price_to_ma_distance(close, ma):
    return (close - ma) / close

def align_time_index(df):
    # Expect ms epoch in 'time' col; set UTC index for joins & resampling
    d = df.copy()
    d["time"] = pd.to_datetime(d["time"], unit="ms", utc=True)
    d = d.set_index("time")
    d = d.rename(columns={
        "open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"
    })
    return d[["Open","High","Low","Close","Volume"]].sort_index()

def resample_htf(df5, rule):
    # Use OHLCV aggregation and drop current partial bar (strictly <= t)
    agg = {
        "Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"
    }
    htf = df5.resample(rule, label="right", closed="right").agg(agg).dropna()
    return htf

# ---------------------------
# Feature builders
# ---------------------------
def compute_feature_frame(df):
    # 5m base features
    out = pd.DataFrame(index=df.index)
    # MAs & slopes
    for n in (20,50,200):
        out[f"ema{n}"] = ema(df["Close"], n)
        out[f"ema{n}_slope"] = slope(out[f"ema{n}"], 5)
        out[f"sma{n}"] = sma(df["Close"], n)
        out[f"sma{n}_slope"] = slope(out[f"sma{n}"], 5)
    out["ema_align_up"] = ((out["ema20"]>out["ema50"]) & (out["ema50"]>out["ema200"])).astype(int)
    out["ema_align_down"] = ((out["ema20"]<out["ema50"]) & (out["ema50"]<out["ema200"])).astype(int)
    out["close_to_ema20"] = price_to_ma_distance(df["Close"], out["ema20"])
    out["close_to_ema50"] = price_to_ma_distance(df["Close"], out["ema50"])
    out["close_to_ema200"] = price_to_ma_distance(df["Close"], out["ema200"])

    # Volatility
    out["atr14"] = atr(df, 14)
    bb_ma, bb_up, bb_lo, bb_w, bb_pos = bollinger(df["Close"], 20, 2)
    out["bb_width"] = bb_w
    out["bb_pos"] = bb_pos
    kc_mid, kc_up, kc_lo, kc_w = keltner(df, 20, 1.5)
    out["kc_width"] = kc_w
    out["squeeze_on"] = ((bb_up < kc_up) & (bb_lo > kc_lo)).astype(int)
    d_up, d_lo, d_mid, d_pos = donchian(df, 20)
    out["donch_pos"] = d_pos

    # Momentum/Oscillators
    out["rsi14"] = rsi(df["Close"], 14)
    m_line, m_sig, m_hist = macd(df["Close"], 12, 26, 9)
    out["macd_line"] = m_line
    out["macd_sig"] = m_sig
    out["macd_hist"] = m_hist
    k, d = stoch(df, 14, 3, 3)
    out["stochK"] = k
    out["stochD"] = d
    out["roc10"] = roc(df["Close"], 10)
    out["roc20"] = roc(df["Close"], 20)
    out["mom10"] = momentum(df["Close"], 10)
    out["mom20"] = momentum(df["Close"], 20)

    # Volume
    out["vol"] = df["Volume"]
    out["vol_ma20"] = sma(df["Volume"], 20)
    v_mean = df["Volume"].rolling(100).mean()
    v_std  = df["Volume"].rolling(100).std()
    out["vol_z100"] = (df["Volume"] - v_mean) / v_std
    out["obv"] = obv(df)
    out["obv_slope"] = slope(out["obv"], 10)

    # Candle anatomy
    top_r, bot_r, body_r = wick_body_ratios(df)
    out["wick_top_ratio"] = top_r
    out["wick_bot_ratio"] = bot_r
    out["body_ratio"] = body_r

    # FVGs
    bull, bear, bull_sz, bear_sz = fvg_flags(df)
    out["fvg_bull"] = bull.astype(int)
    out["fvg_bear"] = bear.astype(int)
    out["fvg_bull_sz"] = bull_sz
    out["fvg_bear_sz"] = bear_sz

    # Time features
    idx = out.index
    out["hour_utc"] = idx.hour
    out["dow"] = idx.dayofweek
    out["session"] = pd.cut(out["hour_utc"],
                            bins=[-1,7,12,20,24],
                            labels=["Asia","EU_open","US","Asia2"]).astype(str)
    return out

def merge_htf_features(df5, feats5, htf_dfs, tag):
    # Align HTF features by last closed bar at/before each 5m index
    htf = htf_dfs.copy()
    # Build HTF feature frame
    f = compute_feature_frame(htf)
    # Forward-fill to 5m timestamps (right-closed bars)
    f = f.reindex(df5.index.union(f.index)).sort_index().ffill().loc[df5.index]
    # Prefix columns
    f = f.add_prefix(f"{tag}_")
    return pd.concat([feats5, f], axis=1)

def corridor_flags(df, entry, sl, tp):
    # Presence of any FVG within price corridor between (entry↔SL) and (entry↔TP) up to 'now'
    lo1, hi1 = (min(entry, sl), max(entry, sl))
    lo2, hi2 = (min(entry, tp), max(entry, tp))
    # We’ll just report whether the *current bar* is an FVG and whether its gap size is inside corridor
    # (more advanced scan over lookback window can be added later)
    bull = (df["Low"] > df["High"].shift(2))
    bear = (df["High"] < df["Low"].shift(2))
    bull_sz = (df["Low"] - df["High"].shift(2)).where(bull, other=0)
    bear_sz = (df["Low"].shift(2) - df["High"]).where(bear, other=0)
    # Flags if the gap sits within corridor ranges (approximate)
    f1 = ((bull_sz>0) & (df["Low"].between(lo1, hi1))).astype(int)
    f2 = ((bear_sz>0) & (df["High"].between(lo2, hi2))).astype(int)
    return f1, f2

# ---------------------------
# Main pipeline
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="TC1 Feature Extraction")
    ap.add_argument("--data", default="data/futures_merged_5m", help="Folder with <SYMBOL>_5m.csv")
    ap.add_argument("--trades", default="reports/trades/trades.csv", help="trades.csv from detector")
    ap.add_argument("--out_leg", default="reports/features/features_at_leg.csv")
    ap.add_argument("--out_entry", default="reports/features/features_at_entry.csv")
    ap.add_argument("--htf", nargs="*", default=[], help="Add HTFs: any of 15m 1h 4h")
    args = ap.parse_args()

    data_dir = Path(args.data)
    trades = pd.read_csv(args.trades)
    symbols = sorted(trades["symbol"].unique())

    # Output collectors
    leg_rows = []
    entry_rows = []

    for sym in symbols:
        # Load 5m prices
        p = data_dir / f"{sym}_5m.csv"
        if not p.exists(): 
            print(f"[WARN] missing {p}, skipping.")
            continue
        df5 = pd.read_csv(p)
        df5 = align_time_index(df5)
        feats5 = compute_feature_frame(df5)

        # Optional HTFs
        if args.htf:
            for rule in args.htf:
                if rule not in ("15m","1h","4h"):
                    print(f"[WARN] unknown HTF '{rule}', skipping.")
                    continue
                htf_df = resample_htf(df5, dict(**{"15m":"15T","1h":"1H","4h":"4H"})[rule])
                feats5 = merge_htf_features(df5, feats5, htf_df, tag=rule)

        # Per-symbol trades
        t_sym = trades[trades["symbol"]==sym].copy()
        # Convert key times to index for joins
        t_sym["A_time"] = pd.to_datetime(t_sym["A_time"], unit="ms", utc=True)
        t_sym["B_time"] = pd.to_datetime(t_sym["B_time"], unit="ms", utc=True)
        t_sym["fill_time"] = pd.to_datetime(t_sym["fill_time"], unit="ms", utc=True)

        # ----- Leg-level features (at B confirmation) -----
        # Snapshot 5m/HTF features at B_time (exact index match exists since trades come from same 5m)
        fB = feats5.reindex(t_sym["B_time"]).reset_index(drop=True)
        leg_tbl = pd.concat([t_sym[["leg_id","symbol","side","A_price","B_price","leg_bars","leg_pct"]].reset_index(drop=True),
                             fB.add_prefix("").reset_index(drop=True)], axis=1)
        # directional normalization (side-aware)
        # Example: macd_hist_norm = macd_hist * (+1 for long, -1 for short)
        sign = np.where(leg_tbl["side"]=="long", 1.0, -1.0)
        for col in ["ema20_slope","ema50_slope","ema200_slope","macd_hist","roc10","roc20","mom10","mom20"]:
            if col in leg_tbl:
                leg_tbl[col+"_norm"] = leg_tbl[col] * sign
        leg_rows.append(leg_tbl)

        # ----- Entry-level features (at fill_time) -----
        fE = feats5.reindex(t_sym["fill_time"]).reset_index(drop=True)
        entry_tbl = pd.concat([
            t_sym[["trade_id","leg_id","symbol","side","entry_price","sl_price","tp_price"]].reset_index(drop=True),
            fE.add_prefix("").reset_index(drop=True)
        ], axis=1)

        # FVG-in-corridor flags (computed at entry bar)
        if not entry_tbl.empty:
            # align a df slice at fill_time rows
            idxE = t_sym["fill_time"].values
            dfE = df5.reindex(t_sym["fill_time"]).reset_index(drop=True)
            f1_list, f2_list = [], []
            for r, de in zip(entry_tbl.itertuples(index=False), dfE.itertuples(index=False)):
                # just placeholders if any NaNs
                if any(pd.isna([r.entry_price, r.sl_price, r.tp_price, de.High, de.Low])):
                    f1_list.append(0); f2_list.append(0); continue
                f1, f2 = corridor_flags(
                    pd.DataFrame({"High":[de.High],"Low":[de.Low]}),
                    r.entry_price, r.sl_price, r.tp_price
                )
                f1_list.append(int(f1.iloc[-1])); f2_list.append(int(f2.iloc[-1]))
            entry_tbl["fvg_in_entry_sl_corridor"] = f1_list
            entry_tbl["fvg_in_entry_tp_corridor"] = f2_list

        # side-normalized features at entry
        signE = np.where(entry_tbl["side"]=="long", 1.0, -1.0)
        for col in ["ema20_slope","ema50_slope","ema200_slope","macd_hist","roc10","roc20","mom10","mom20"]:
            if col in entry_tbl:
                entry_tbl[col+"_norm"] = entry_tbl[col] * signE

        entry_rows.append(entry_tbl)

    # Write outputs
    out_leg = pd.concat(leg_rows, ignore_index=True) if leg_rows else pd.DataFrame()
    out_entry = pd.concat(entry_rows, ignore_index=True) if entry_rows else pd.DataFrame()

    out_leg.to_csv(args.out_leg, index=False)
    out_entry.to_csv(args.out_entry, index=False)
    print(f"[DONE] Wrote {len(out_leg)} leg rows → {args.out_leg}")
    print(f"[DONE] Wrote {len(out_entry)} entry rows → {args.out_entry}")

if __name__ == "__main__":
    main()
