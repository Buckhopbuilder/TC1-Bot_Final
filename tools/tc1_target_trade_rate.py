#!/usr/bin/env python3
import argparse, numpy as np, pandas as pd
from pathlib import Path

def to_utc_ts(df):
    if "fill_dt" in df.columns: return pd.to_datetime(df["fill_dt"], utc=True)
    if "fill_time" in df.columns: return pd.to_datetime(df["fill_time"], unit="ms", utc=True)
    if "fill_time_iso" in df.columns: return pd.to_datetime(df["fill_time_iso"], utc=True)
    raise ValueError("No fill_dt, fill_time, or fill_time_iso in predictions file.")

def apply_session_filter(df, windows, weekday_only):
    ts = df["fill_dt"]; mins = ts.dt.hour*60 + ts.dt.minute; dow = ts.dt.dayofweek
    mask = np.zeros(len(df), dtype=bool)
    for w in windows:
        s,e = w.split("-"); sh,sm = map(int,s.split(":")); eh,em = map(int,e.split(":"))
        st = sh*60+sm; en = eh*60+em
        mask |= (mins >= st) & (mins < en)
    if weekday_only: mask |= (dow >= 5)
    return df.loc[~mask].copy()

def summarize(y_true, rr):
    n = len(y_true)
    if n == 0: return dict(trades=0, winrate=np.nan, expectancy=np.nan)
    wr = float(np.mean(y_true))
    expR = float(np.mean(np.where(y_true==1, rr, -1)))
    return dict(trades=int(n), winrate=wr, expectancy=expR)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="reports/ml/walkforward_preds.csv")
    ap.add_argument("--trades", default="reports/trades/trades.csv")
    ap.add_argument("--target_per_day", type=float, default=6.0)
    ap.add_argument("--windows", nargs="+", default=["08:00-10:00","13:00-16:00"])
    ap.add_argument("--weekday_only", action="store_true")
    ap.add_argument("--out_summary", default="reports/ml/target6_summary.csv")
    ap.add_argument("--out_trades", default="reports/ml/policy_trades_target6.csv")
    args = ap.parse_args()

    preds = pd.read_csv(args.preds)
    preds["fill_dt"] = to_utc_ts(preds)
    preds["year"] = preds["fill_dt"].dt.year

    # only bring what we truly need from trades
    tr_cols = ["trade_id","outcome","fill_time"]
    tr = pd.read_csv(args.trades)[tr_cols]

    preds_f = apply_session_filter(preds, args.windows, args.weekday_only)

    rows, outs = [], []
    for y, g in preds_f.groupby("year"):
        days = g["fill_dt"].dt.normalize().nunique()
        target_total = args.target_per_day * float(days)

        fracs = np.linspace(0.05, 0.30, 6)
        probs = np.linspace(0.20, 0.40, 11)
        cands = []
        for f in fracs:
            thr = g["y_prob"].quantile(1.0 - f)
            cands.append(("top_frac", float(f), float(thr), int((g["y_prob"]>=thr).sum())))
        for p in probs:
            cands.append(("prob_thr", float(p), float(p), int((g["y_prob"]>=p).sum())))
        cands = [c for c in cands if c[3] > 0]
        if not cands:
            rows.append({"year": int(y), "rule": "none", "param": np.nan, "threshold": np.nan,
                         "trades": 0, "winrate": np.nan, "expectancy": np.nan,
                         "days": int(days), "target_total": int(target_total)})
            continue

        mode, param, thr, _ = min(cands, key=lambda c: abs(c[3]-target_total))
        sel = g[g["y_prob"] >= thr].copy().sort_values("y_prob", ascending=False)
        sel = sel.merge(tr, on="trade_id", how="left")

        if "y_true" not in sel.columns:
            sel["y_true"] = (sel["outcome"]=="win").astype(int)
        if "fill_time" not in sel.columns or sel["fill_time"].isna().all():
            sel["fill_time"] = (sel["fill_dt"].astype("int64") // 10**6)

        stats = summarize(sel["y_true"].values, sel["RR_target"].values)
        rows.append({"year": int(y), "rule": mode, "param": param, "threshold": thr,
                     **stats, "days": int(days), "target_total": int(target_total)})

        sel["policy_year"] = int(y)
        sel["policy_rule"] = f"{mode}={param:.2f}"
        sel["policy_threshold"] = thr
        outs.append(sel)

    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_summary, index=False)
    print(f"[DONE] Saved summary → {args.out_summary}")

    if outs:
        out = pd.concat(outs, ignore_index=True)
        out["fill_time_iso"] = pd.to_datetime(out["fill_time"], unit="ms", utc=True)
        # keep only columns that actually exist
        keep = ["trade_id","symbol","side","fill_time","fill_time_iso",
                "RR_target","y_prob","y_true","outcome",
                "policy_year","policy_rule","policy_threshold"]
        keep = [c for c in keep if c in out.columns]
        out = out[keep]
        Path(args.out_trades).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.out_trades, index=False)
        print(f"[DONE] Saved trades → {args.out_trades} (rows={len(out)})")

if __name__ == "__main__":
    main()
