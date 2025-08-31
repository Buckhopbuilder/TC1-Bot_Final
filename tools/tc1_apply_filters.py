#!/usr/bin/env python3
import argparse, re
from pathlib import Path
import numpy as np, pandas as pd

def to_fill_dt(df):
    if "fill_time" in df.columns and df["fill_time"].notna().any():
        return pd.to_datetime(df["fill_time"], unit="ms", utc=True, errors="coerce")
    if "fill_time_iso" in df.columns and df["fill_time_iso"].notna().any():
        return pd.to_datetime(df["fill_time_iso"], utc=True, errors="coerce")
    if "fill_dt" in df.columns and df["fill_dt"].notna().any():
        return pd.to_datetime(df["fill_dt"], utc=True, errors="coerce")
    raise ValueError("No fill_time/fill_time_iso/fill_dt column available.")

def add_session_meta(df):
    ts = df["fill_dt"]
    df["_minute"] = ts.dt.hour * 60 + ts.dt.minute
    df["_dow"] = ts.dt.dayofweek
    return df

def apply_session_exclusions(df, windows, weekday_only):
    excl = np.zeros(len(df), dtype=bool)
    for w in windows:
        s,e = w.split("-")
        sh,sm = map(int, s.split(":")); eh,em = map(int, e.split(":"))
        st = sh*60 + sm; en = eh*60 + em
        excl |= (df["_minute"] >= st) & (df["_minute"] < en)
    if weekday_only: excl |= (df["_dow"] >= 5)
    df["_rej_session"] = excl
    return df

def summarize(y_true, rr):
    if len(y_true)==0: return dict(trades=0, winrate=np.nan, expectancy=np.nan)
    wr = float(np.mean(y_true))
    expR = float(np.mean(np.where(y_true==1, rr, -1.0)))
    return dict(trades=len(y_true), winrate=wr, expectancy=expR)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--features", default="reports/features/features_at_entry.csv")
    ap.add_argument("--out_trades", default="reports/ml/policy_trades_filtered.csv")
    ap.add_argument("--out_equity", default="reports/ml/policy_equity_filtered.csv")
    ap.add_argument("--out_summary", default="reports/ml/policy_filters_summary.txt")
    ap.add_argument("--max_body_ratio", type=float, default=0.50)
    ap.add_argument("--reject_fvg", choices=["none","bull","bear","both"], default="both")
    ap.add_argument("--width_quantile", type=float, default=0.75)
    ap.add_argument("--width_patterns", nargs="+", default=["bb.*width","kc.*width"])
    ap.add_argument("--windows", nargs="+", default=["08:00-10:00","13:00-16:00"])
    ap.add_argument("--weekday_only", action="store_true")
    ap.add_argument("--min_prob", type=float, default=None)
    args = ap.parse_args()

    pol = pd.read_csv(args.policy)
    feats = pd.read_csv(args.features)
    feats = feats.drop(columns=[c for c in ["symbol","side","RR_target"] if c in feats.columns], errors="ignore")
    df = pol.merge(feats, on="trade_id", how="left")

    # ✅ Always ensure these exist
    if "y_true" not in df.columns and "outcome" in df.columns:
        df["y_true"] = (df["outcome"]=="win").astype(int)
    if "RR_target" not in df.columns and "RR_target" in pol.columns:
        df["RR_target"] = pol["RR_target"]

    df["fill_dt"] = to_fill_dt(df)
    df = add_session_meta(df)

    # === Ruleouts ===
    reasons = pd.DataFrame(index=df.index)
    reasons["_rej_body"] = (df["body_ratio"] > args.max_body_ratio) if "body_ratio" in df.columns else False
    rej_fvg = np.zeros(len(df), dtype=bool)
    if args.reject_fvg in ("bull","both") and "fvg_bull" in df.columns: rej_fvg |= (df["fvg_bull"]==1)
    if args.reject_fvg in ("bear","both") and "fvg_bear" in df.columns: rej_fvg |= (df["fvg_bear"]==1)
    reasons["_rej_fvg"] = rej_fvg
    width_cols = []
    for pat in args.width_patterns:
        rx = re.compile(pat, flags=re.IGNORECASE)
        width_cols += [c for c in df.columns if rx.search(c)]
    width_cols = sorted(set(width_cols))
    if width_cols:
        too_wide = np.zeros(len(df), dtype=bool)
        for c in width_cols:
            s = pd.to_numeric(df[c], errors="coerce")
            q = np.nanquantile(s, args.width_quantile) if s.notna().any() else np.nan
            too_wide |= (s > q)
        reasons["_rej_width"] = too_wide
    else: reasons["_rej_width"] = False
    df = apply_session_exclusions(df, args.windows, args.weekday_only)
    if args.min_prob is not None and "y_prob" in df.columns:
        reasons["_rej_prob"] = df["y_prob"] < args.min_prob
    else: reasons["_rej_prob"] = False

    for col in reasons.columns: df[col] = reasons[col]
    df["_keep"] = ~reasons.any(axis=1)

    # Stats
    before = summarize(df["y_true"].values, df["RR_target"].values)
    kept = df[df["_keep"]].copy()
    after = summarize(kept["y_true"].values, kept["RR_target"].values)

    # Equity
    if "outcome" in kept.columns:
        kept = kept.sort_values("fill_dt").reset_index(drop=True)
        kept["R_gross"] = np.where(kept["outcome"]=="win", kept["RR_target"], -1.0)
        kept["R_net"] = kept["R_gross"]
        kept["equity_R"] = kept["R_net"].cumsum()

    # ✅ Always include RR_target/outcome/y_true
    base_cols = ["trade_id","symbol","side","fill_time","fill_time_iso","fill_dt",
                 "y_prob","y_true","outcome","RR_target","body_ratio"]
    out_cols = [c for c in base_cols if c in kept.columns] + width_cols
    out_cols += [c for c in ["_rej_body","_rej_fvg","_rej_width","_rej_session","_rej_prob","_keep"] if c in kept.columns]
    Path(args.out_trades).parent.mkdir(parents=True, exist_ok=True)
    kept[out_cols].to_csv(args.out_trades, index=False)

    if "equity_R" in kept.columns:
        eq_cols = ["trade_id","symbol","fill_dt","R_net","equity_R"]
        kept[eq_cols].to_csv(args.out_equity, index=False)

    # Summary (with trades/day)
    days = int(df["fill_dt"].dt.normalize().nunique())
    kept_days = int(kept["fill_dt"].dt.normalize().nunique()) if len(kept) else 0
    tpd = kept.shape[0]/kept_days if kept_days else 0.0
    with open(args.out_summary,"w") as f:
        f.write(f"BEFORE: {before}\nAFTER : {after} (~{tpd:.2f} trades/day)\n")

    print(f"[DONE] Trades  → {args.out_trades}")
    print(f"[DONE] Equity  → {args.out_equity}")
    print(f"[DONE] Summary → {args.out_summary}")
    print(f"AFTER trades/day ≈ {tpd:.2f}")

if __name__ == "__main__":
    main()
