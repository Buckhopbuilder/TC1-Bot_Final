#!/usr/bin/env python3
import argparse, numpy as np, pandas as pd
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="Add outcome (if missing) and RR_target to trades.csv")
    ap.add_argument("--in_csv",  default="reports/trades/trades.csv")
    ap.add_argument("--out_csv", default="reports/trades/trades_enriched.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)

    # normalize columns (lowercase check)
    cols = {c.lower(): c for c in df.columns}

    # --- Compute RR_target if missing ---
    def rr_row(r):
        try:
            side = str(r[cols.get("side","side")]).lower()
            entry = float(r[cols.get("entry_price","entry_price")])
            sl    = float(r[cols.get("sl_price","sl_price")])
            tp    = float(r[cols.get("tp_price","tp_price")])
        except Exception:
            return np.nan
        if side.startswith("l"):
            denom = abs(entry - sl)
            return (tp - entry) / denom if denom > 0 else np.nan
        else:
            denom = abs(sl - entry)
            return (entry - tp) / denom if denom > 0 else np.nan

    if "RR_target" not in df.columns:
        df["RR_target"] = df.apply(rr_row, axis=1)

    # --- Build/normalize outcome if missing ---
    if "outcome" not in df.columns:
        # try reconstruct from side + exit vs tp/sl
        if all(k in cols for k in ["side","exit_price","tp_price","sl_price"]):
            side = df[cols["side"]].astype(str).str.lower()
            ex   = pd.to_numeric(df[cols["exit_price"]], errors="coerce")
            tp   = pd.to_numeric(df[cols["tp_price"]],  errors="coerce")
            sl   = pd.to_numeric(df[cols["sl_price"]],  errors="coerce")
            eps  = 1e-10

            win = pd.Series(False, index=df.index)
            loss= pd.Series(False, index=df.index)

            # longs
            maskL = side.str.startswith("l")
            win[ maskL] = ex[maskL] >= (tp[maskL] - eps)
            loss[maskL] = ex[maskL] <= (sl[maskL] + eps)
            # shorts
            maskS = side.str.startswith("s")
            win[ maskS] = ex[maskS] <= (tp[maskS] + eps)
            loss[maskS] = ex[maskS] >= (sl[maskS] - eps)

            out = pd.Series("unknown", index=df.index)
            out[win]  = "win"
            out[loss] = "loss"
            df["outcome"] = out
        else:
            # create an 'outcome' but mark as unknown if not reconstructable
            df["outcome"] = "unknown"
    else:
        # normalize casing
        df["outcome"] = df["outcome"].astype(str).str.lower().replace({"1":"win","0":"loss","true":"win","false":"loss"})

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    wins = (df["outcome"]=="win").sum()
    losses = (df["outcome"]=="loss").sum()
    unk = (df["outcome"]=="unknown").sum()
    print(f"[DONE] Enriched trades → {args.out_csv} | wins={wins} losses={losses} unknown={unk} | RR_target non-null={(df['RR_target'].notna().sum())}")
if __name__ == "__main__":
    main()
