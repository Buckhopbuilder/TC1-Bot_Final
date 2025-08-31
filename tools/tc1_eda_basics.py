import pandas as pd, numpy as np, sys
fp = sys.argv[1] if len(sys.argv)>1 else "reports/trades/trades.csv"
df = pd.read_csv(fp)
# Basic W/L per symbol
def sym_stats(g):
    wins = (g.outcome=="win").sum()
    losses = (g.outcome=="loss").sum()
    n = wins + losses
    wr = wins / n if n else 0.0
    # Expectancy using per-trade RR_target (already in file)
    # win R = RR_target, loss R = -1
    exp = (g.apply(lambda r: r.RR_target if r.outcome=="win" else (-1 if r.outcome=="loss" else 0), axis=1)).mean()
    return pd.Series(dict(trades=n,wins=wins,losses=losses,winrate=wr,expectancy_R=exp))
print("\n=== Per-symbol W/L + expectancy ===")
print(df.groupby("symbol").apply(sym_stats).sort_values("winrate", ascending=False).round(4))
print("\n=== Overall ===")
print(sym_stats(df).round(4))
