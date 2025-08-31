#!/usr/bin/env python3
import argparse, numpy as np, pandas as pd

def find_outcome(df):
    cand = [c for c in df.columns if c.lower() in ("outcome","result","label","y_true")]
    if cand:
        c = cand[0]
        s = df[c].astype(str).str.lower().str.strip().replace({"1":"win","0":"loss","true":"win","false":"loss"})
        return s
    # try reconstruct
    have = set([c.lower() for c in df.columns])
    need = {"side","exit_price","tp_price","sl_price"}
    if need.issubset(have):
        side = df[[c for c in df.columns if c.lower()=="side"][0]].astype(str).str.lower()
        ex   = pd.to_numeric(df[[c for c in df.columns if c.lower()=="exit_price"][0]], errors="coerce")
        tp   = pd.to_numeric(df[[c for c in df.columns if c.lower()=="tp_price"][0]],  errors="coerce")
        sl   = pd.to_numeric(df[[c for c in df.columns if c.lower()=="sl_price"][0]],  errors="coerce")
        eps=1e-10
        out = pd.Series("unknown", index=df.index)
        # long
        mL = side.str.startswith("l")
        out[mL & (ex >= tp - eps)] = "win"
        out[mL & (ex <= sl + eps)] = "loss"
        # short
        mS = side.str.startswith("s")
        out[mS & (ex <= tp + eps)] = "win"
        out[mS & (ex >= sl - eps)] = "loss"
        return out
    return pd.Series("unknown", index=df.index)

def main():
    ap = argparse.ArgumentParser(description="Suggest simple rules from loser_profile outputs.")
    ap.add_argument("--trades", default="reports/trades/trades_enriched.csv")
    ap.add_argument("--features", default="reports/features/features_at_entry_plus.csv")
    ap.add_argument("--profile_csv", default="reports/eda/loser_profile_plus/numeric_features_ranked.csv")
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--quantiles", nargs="+", type=float, default=[0.2,0.3,0.4,0.5,0.6,0.7,0.8])
    args = ap.parse_args()

    prof = pd.read_csv(args.profile_csv)
    fe   = pd.read_csv(args.features)
    tr   = pd.read_csv(args.trades)

    # keep columns safely
    keep = ["trade_id","symbol","fill_time","side","entry_price","sl_price","tp_price"]
    if "outcome" in tr.columns: keep.append("outcome")
    if "RR_target" in tr.columns: keep.append("RR_target")
    tr = tr[[c for c in keep if c in tr.columns]]

    df = fe.merge(tr, on="trade_id", how="left")

    # ensure outcome
    if "outcome" not in df.columns:
        df["outcome"] = find_outcome(df)
    df["outcome"] = df["outcome"].fillna("unknown").astype(str).str.lower()

    y = (df["outcome"]=="win").astype(int).values

    # pick top_k by |d|
    prof = prof.dropna(subset=["cohens_d"])
    top = prof.reindex(prof["cohens_d"].abs().sort_values(ascending=False).index)[:args.top_k]
    print("=== Top features by |Cohen's d| ===")
    print(top[["feature","cohens_d","auroc"]].to_string(index=False))

    print("\n=== Candidate one-liner filters (per feature, thresholds by quantile) ===")
    for _, row in top.iterrows():
        f = row["feature"]; d = row["cohens_d"]
        if f not in df.columns: continue
        x = pd.to_numeric(df[f], errors="coerce")
        if x.isna().all(): continue
        best = None
        for q in args.quantiles:
            thr = np.nanquantile(x, q)
            if np.isnan(thr): continue
            mask = (x >= thr) if d >= 0 else (x <= thr)
            kept = df[mask & df["outcome"].isin(["win","loss"])]
            if kept.empty: continue
            wr = (kept["outcome"]=="win").mean()
            if "RR_target" in kept.columns and kept["RR_target"].notna().any():
                expR = np.where(kept["outcome"]=="win", kept["RR_target"], -1.0).mean()
            else:
                expR = wr - (1-wr)
            n = len(kept)
            score = expR
            desc = f"{f} {'>=' if d>=0 else '<='} {thr:.5g}"
            if (best is None) or (score > best[0]):
                best = (score, wr, n, desc)
        if best:
            print(f"- {best[3]}  → n={best[2]}  wr={best[1]:.3f}  expR={best[0]:.3f}")

if __name__ == "__main__":
    main()
