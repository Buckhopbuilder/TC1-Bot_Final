import os, warnings
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np

TRADES = "reports/trades/trades.csv"
FEAT_E = "reports/features/features_at_entry.csv"
OUTDIR = "reports/eda"

def load():
    tr = pd.read_csv(TRADES)
    fe = pd.read_csv(FEAT_E)

    # Keep only completed trades, build label
    tr = tr[tr["outcome"].isin(["win","loss"])].copy()
    tr["y"] = (tr["outcome"]=="win").astype(int)

    # Drop duplicate price columns from features (they exist in trades already)
    for c in ["entry_price","sl_price","tp_price"]:
        if c in fe.columns:
            fe = fe.drop(columns=c)

    # Merge on trade_id only (unique), keep all trades that have features
    df = tr.merge(fe, on="trade_id", how="inner")
    return tr, fe, df

def per_symbol_stats(tr):
    def stats(g):
        w = (g.outcome=="win").sum(); l = (g.outcome=="loss").sum()
        n = w + l
        wr = w/n if n else 0.0
        exp = (g.apply(lambda r: r.RR_target if r.outcome=="win" else -1, axis=1)).mean()
        return pd.Series(dict(trades=n,wins=w,losses=l,winrate=wr,expectancy_R=exp))
    return tr.groupby("symbol").apply(stats).sort_values("winrate", ascending=False)

def correlations(df):
    # Numeric-only features
    num = df.select_dtypes(include=[np.number]).copy()

    # Drop identifiers / leakage-ish fields
    drop = {
        "trade_id","A_time","B_time","fill_time","exit_time","exit_price",
        "A_price","B_price","bars_in_trade","RR_target","ambiguous","y"
    }
    keep_cols = [c for c in num.columns if c not in drop]
    X = num[keep_cols]

    # Point-biserial correlation == Pearson corr with binary y
    cor = X.corrwith(df["y"]).dropna().sort_values(ascending=False)
    return cor

def mutual_info(df):
    try:
        from sklearn.feature_selection import mutual_info_classif
    except Exception:
        return pd.Series(dtype=float)
    num = df.select_dtypes(include=[np.number]).copy()
    drop = {
        "trade_id","A_time","B_time","fill_time","exit_time","exit_price",
        "A_price","B_price","bars_in_trade","RR_target","ambiguous","y"
    }
    X = num[[c for c in num.columns if c not in drop]].fillna(0.0)
    y = df["y"].values
    # Standardize for stability
    X = (X - X.mean())/X.std(ddof=0)
    mi = mutual_info_classif(X.fillna(0.0), y, discrete_features=False, random_state=0)
    return pd.Series(mi, index=X.columns).sort_values(ascending=False)

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    tr, fe, df = load()

    # 1) Per-symbol stats
    ps = per_symbol_stats(tr)
    ps.to_csv(f"{OUTDIR}/per_symbol_stats.csv")

    # 2) Correlations
    cor = correlations(df)
    cor.to_csv(f"{OUTDIR}/point_biserial_correlations.csv", header=["point_biserial_corr"])

    # 3) Mutual information (optional)
    try:
        mi = mutual_info(df)
        if not mi.empty:
            mi.to_csv(f"{OUTDIR}/mutual_info.csv", header=["mutual_info"])
    except Exception:
        mi = pd.Series(dtype=float)

    # Console summary
    print("== Per-symbol W/L + Expectancy ==")
    print(ps.round(4))
    print("\n== Top 15 positive correlations (entry features) ==")
    print(cor.head(15).round(4))
    print("\n== Top 15 negative correlations (entry features) ==")
    print(cor.tail(15).round(4))

    if not mi.empty:
        print("\n== Top 15 mutual information (entry features) ==")
        print(mi.head(15).round(4))
        print("\n== Bottom 15 mutual information (entry features) ==")
        print(mi.tail(15).round(4))

    print(f"\nSaved: {OUTDIR}/per_symbol_stats.csv, {OUTDIR}/point_biserial_correlations.csv" + (", mutual_info.csv" if not mi.empty else ""))
if __name__ == "__main__":
    main()
