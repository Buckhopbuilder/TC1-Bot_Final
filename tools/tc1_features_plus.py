#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np, pandas as pd

# ---------- small helpers ----------
def ensure_utc(s):
    """Return a tz-aware UTC datetime Series from a datetime-like Series."""
    s = pd.to_datetime(s, utc=True, errors="coerce")
    return s

# ---------- indicator helpers (unchanged logic) ----------
def ema(s, n): return s.ewm(span=n, adjust=False, min_periods=n).mean()
def rsi(close, n=14):
    d = close.diff(); up = d.clip(lower=0.0); down = -d.clip(upper=0.0)
    return 100 - 100/(1 + ema(up,n)/ema(down,n))
def true_range(h,l,c_prev): return pd.concat([h-l,(h-c_prev).abs(),(l-c_prev).abs()],axis=1).max(axis=1)
def atr(h,l,c,n=14): return true_range(h,l,c.shift(1)).ewm(span=n,adjust=False,min_periods=n).mean()
def macd(close,fast=12,slow=26,signal=9):
    m = ema(close,fast)-ema(close,slow); sig = ema(m,signal); return m,sig,m-sig
def adx(h,l,c,n=14):
    up=h.diff(); dn=-l.diff()
    plus=((up>dn)&(up>0)).astype(float)*up.clip(lower=0); minus=((dn>up)&(dn>0)).astype(float)*dn.clip(lower=0)
    atrn = true_range(h,l,c.shift(1)).ewm(span=n,adjust=False,min_periods=n).mean()
    pdi=100*(ema(plus,n)/atrn); mdi=100*(ema(minus,n)/atrn)
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    return ema(dx,n)
def bollinger_width(close,n=20,k=2.0):
    ma=close.rolling(n,min_periods=n).mean(); sd=close.rolling(n,min_periods=n).std()
    return (ma+k*sd - (ma-k*sd))/close
def keltner_width(close,h,l,n=20,m=2.0):
    mid=ema(close,n); atrn=atr(h,l,close,n); return ((mid+m*atrn)-(mid-m*atrn))/close
def wick_features(o,h,l,c):
    body=(c-o).abs(); upper=h-np.maximum(o,c); lower=np.minimum(o,c)-l; rng=(h-l).replace(0,np.nan)
    body_ratio=(upper+lower)/body.replace(0,np.nan); return body_ratio.fillna(0.0),(upper/rng).fillna(0.0),(lower/rng).fillna(0.0)
def vwap_intraday(dt,close,vol):
    if vol is None or vol.isna().all(): return pd.Series(index=close.index,dtype=float)
    day = ensure_utc(dt).dt.floor('D'); tpv=close*vol
    return (tpv.groupby(day).cumsum())/(vol.groupby(day).cumsum()).replace(0,np.nan)
def round_number_proximity(close):
    frac = (close % 1.0)
    dist = pd.concat([(frac-0.00).abs(),(frac-0.25).abs(),(frac-0.50).abs(),(frac-0.75).abs(),(1.0-frac).abs()],axis=1).min(axis=1)
    return (dist/close)*1e4, (frac.where(frac<=0.5,1.0-frac)/close)*1e4
def daily_context(dt,h,l,c):
    day = ensure_utc(dt).dt.floor('D'); d_high=h.groupby(day).transform('max'); d_low=l.groupby(day).transform('min')
    rng=(d_high-d_low).replace(0,np.nan); pos=((c-d_low)/rng).clip(0,1).fillna(0.5)
    return pos, ((c-d_low)/c)*1e4, ((d_high-c)/c)*1e4

# ---------- loader & resampler ----------
def load_ohlc(root_dir, symbol):
    root = Path(root_dir)
    candidates = [root/f"{symbol}.csv", root/f"{symbol}-5m.csv", root/f"{symbol}_5m.csv", root/f"{symbol}-5T.csv"]
    fp=None
    for c in candidates:
        if c.exists(): fp=c; break
    if fp is None:
        matches = sorted(root.glob(f"{symbol}*"))
        if matches: fp = matches[0]
    if fp is None or not fp.exists():
        raise FileNotFoundError(f"OHLC not found for {symbol} under {root_dir}")
    df = pd.read_csv(fp)

    # timestamp
    tcol=None
    for c in df.columns:
        if c.lower() in ("open_time","timestamp","time","ts","date","datetime"):
            tcol=c; break
    if tcol is None: raise ValueError(f"No timestamp column in {fp}")
    if np.issubdtype(df[tcol].dtype, np.number) and df[tcol].max() > 10**12:
        ts = df[tcol].astype("int64")
        dt = ensure_utc(pd.to_datetime(ts, unit="ms"))
    else:
        dt = ensure_utc(df[tcol])
    # prices
    def find(name):
        for c in df.columns:
            if c.lower()==name: return c
        for c in df.columns:
            if name in c.lower(): return c
        return None
    oc, hc, lc, cc = map(find, ["open","high","low","close"])
    vc = find("volume")
    out = pd.DataFrame({
        "dt": ensure_utc(dt),
        "open": pd.to_numeric(df[oc], errors="coerce"),
        "high": pd.to_numeric(df[hc], errors="coerce"),
        "low" : pd.to_numeric(df[lc], errors="coerce"),
        "close": pd.to_numeric(df[cc], errors="coerce"),
    }).dropna().sort_values("dt").reset_index(drop=True)
    out["ts"] = (out["dt"].view("int64") // 10**6).astype("int64")
    out["volume"] = pd.to_numeric(df[vc], errors="coerce").fillna(0.0) if vc else np.nan
    return out

def resample_ohlc(df5, minutes):
    # keep tz-aware UTC
    dt = ensure_utc(df5["dt"])
    g = dt.dt.floor(f"{minutes}T")
    agg = df5.groupby(g).agg(
        open=("open","first"),
        high=("high","max"),
        low=("low","min"),
        close=("close","last"),
        volume=("volume","sum")
    ).dropna().reset_index()
    agg = agg.rename(columns={agg.columns[0]:"dt"})
    agg["dt"] = ensure_utc(agg["dt"])
    agg["ts"] = (agg["dt"].view("int64") // 10**6).astype("int64")
    return agg

# ---------- per-frame features ----------
def compute_frame_feats(df, label):
    out = pd.DataFrame(index=df.index)
    close, high, low, open_, vol, dt = df["close"], df["high"], df["low"], df["open"], df["volume"], ensure_utc(df["dt"])
    out[f"rsi_{label}"] = rsi(close,14)
    out[f"rsi_slope_{label}"] = out[f"rsi_{label}"] - out[f"rsi_{label}"].shift(5)
    out[f"ema20_{label}"] = ema(close,20); out[f"ema50_{label}"] = ema(close,50); out[f"ema200_{label}"] = ema(close,200)
    out[f"ema20_slope_{label}"] = out[f"ema20_{label}"] - out[f"ema20_{label}"].shift(5)
    out[f"ema50_slope_{label}"] = out[f"ema50_{label}"] - out[f"ema50_{label}"].shift(5)
    out[f"ema200_slope_{label}"] = out[f"ema200_{label}"] - out[f"ema200_{label}"].shift(5)
    out[f"atrp_{label}"] = (atr(high,low,close,14) / close) * 100.0
    out[f"bb_width_{label}"] = bollinger_width(close,20,2.0)
    out[f"kc_width_{label}"] = keltner_width(close,high,low,20,2.0)
    vwap = vwap_intraday(dt, close, vol); out[f"vwap_dist_bp_{label}"] = ((close - vwap) / close) * 1e4
    out[f"adx14_{label}"] = adx(high,low,close,14)
    _,_,hist = macd(close,12,26,9); out[f"macd_hist_{label}"] = hist; out[f"macd_slope_{label}"] = hist - hist.shift(3)
    br, uwr, lwr = wick_features(open_,high,low,close)
    out[f"body_ratio_{label}"] = br; out[f"upper_wick_ratio_{label}"] = uwr; out[f"lower_wick_ratio_{label}"] = lwr
    pos,dlo,dhi = daily_context(dt,high,low,close)
    out[f"day_pos_{label}"] = pos; out[f"dist_dlow_bp_{label}"] = dlo; out[f"dist_dhigh_bp_{label}"] = dhi
    rn_q_bp, rn_i_bp = round_number_proximity(close)
    out[f"rn_quarter_bp_{label}"] = rn_q_bp; out[f"rn_integer_bp_{label}"] = rn_i_bp
    out[f"align_up_{label}"]   = ((out[f"ema20_{label}"]>out[f"ema50_{label}"]) & (out[f"ema50_{label}"]>out[f"ema200_{label}"])).astype(int)
    out[f"align_down_{label}"] = ((out[f"ema20_{label}"]<out[f"ema50_{label}"]) & (out[f"ema50_{label}"]<out[f"ema200_{label}"])).astype(int)
    out["dt"] = ensure_utc(df["dt"]); out["ts"] = df["ts"].values
    return out

def asof_join(left, right, on="dt", suffix=""):
    L = left.sort_values(on).copy(); R = right.sort_values(on).copy()
    L[on] = ensure_utc(L[on]); R[on] = ensure_utc(R[on])
    return pd.merge_asof(L, R, on=on, direction="backward", suffixes=("", suffix))

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Build expanded features at entry with M5/M15/H1 & trend/RSI/volatility/VWAP context.")
    ap.add_argument("--trades", default="reports/trades/trades.csv")
    ap.add_argument("--ohlc_dir", default="data/futures_merged_5m")
    ap.add_argument("--out", default="reports/features/features_at_entry_plus.csv")
    ap.add_argument("--symbols", nargs="*", default=None)
    args = ap.parse_args()

    trades = pd.read_csv(args.trades)
    need = ["trade_id","symbol","fill_time","side","entry_price","sl_price","tp_price","outcome","RR_target"]
    trades = trades[[c for c in need if c in trades.columns]].dropna(subset=["symbol","fill_time"]).copy()
    trades["fill_dt"] = ensure_utc(pd.to_datetime(trades["fill_time"], unit="ms", errors="coerce"))

    symbols = sorted(trades["symbol"].unique().tolist()) if args.symbols is None else args.symbols
    rows = []
    for sym in symbols:
        t_sym = trades[trades["symbol"]==sym].copy()
        if t_sym.empty: continue
        try:
            px5 = load_ohlc(args.ohlc_dir, sym)
        except Exception as e:
            print(f"[WARN] skip {sym}: {e}")
            continue

        feat5 = compute_frame_feats(px5, "5m")
        px15 = resample_ohlc(px5, 15);   feat15 = compute_frame_feats(px15, "15m")
        px60 = resample_ohlc(px5, 60);   feat60 = compute_frame_feats(px60, "60m")

        base = feat5[["dt","ts"]].copy()
        m = base.copy()
        m = asof_join(m, feat5.drop(columns=["ts"]), on="dt")
        m = asof_join(m, feat15.drop(columns=["ts"]), on="dt", suffix="_15m")
        m = asof_join(m, feat60.drop(columns=["ts"]), on="dt", suffix="_60m")

        for tf in ["5m","15m","60m"]:
            if f"align_up_{tf}" not in m.columns: m[f"align_up_{tf}"]=0
            if f"align_down_{tf}" not in m.columns: m[f"align_down_{tf}"]=0
        m["align_up_count"] = m[["align_up_5m","align_up_15m","align_up_60m"]].sum(axis=1)
        m["align_down_count"] = m[["align_down_5m","align_down_15m","align_down_60m"]].sum(axis=1)

        t_sym = t_sym.sort_values("fill_dt"); m = m.sort_values("dt")
        feat_at_entry = asof_join(t_sym, m, left_on="fill_dt", right_on="dt") if False else pd.merge_asof(
            t_sym, m, left_on="fill_dt", right_on="dt", direction="backward"
        )
        rows.append(feat_at_entry.assign(symbol=sym))
        print(f"[OK] {sym}: trades={len(t_sym)}  last_ts={m['dt'].max()}")

    if not rows:
        raise SystemExit("No features built (no symbols processed).")
    out = pd.concat(rows, ignore_index=True)

    core = ["trade_id","symbol","side","fill_time","fill_dt","entry_price","sl_price","tp_price","outcome","RR_target"]
    cols = core + [c for c in out.columns if c not in core]
    out = out[cols]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"[DONE] Wrote features → {args.out}")
    print("Columns:", len(out.columns))

if __name__ == "__main__":
    main()
