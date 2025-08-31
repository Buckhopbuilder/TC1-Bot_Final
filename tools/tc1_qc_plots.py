import pandas as pd
from pathlib import Path
import argparse
import mplfinance as mpf

def load_prices(data_dir, symbol):
    p = Path(data_dir)/f"{symbol}_5m.csv"
    df = pd.read_csv(p)
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df = df.set_index("time")
    # mplfinance expects capitalized column names
    df = df.rename(columns={
        "open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"
    })
    return df[["Open","High","Low","Close","Volume"]]

def plot_trade(df_px, trade, outdir):
    # window around the trade (1 bar padding on both sides)
    t0 = pd.to_datetime(min(trade["A_time"], trade["fill_time"]) - 5*60*1000, unit="ms", utc=True)
    t1 = pd.to_datetime(trade["exit_time"] + 5*60*1000, unit="ms", utc=True)
    win = df_px.loc[t0:t1].copy()
    if len(win) < 5:
        return

    entry = float(trade["entry_price"])
    sl    = float(trade["sl_price"])
    tp    = float(trade["tp_price"])

    hlines = dict(
        hlines=[entry, sl, tp],
        colors=['#888888', '#cc3333', '#2a912a'],
        linewidths=[1.2, 1.2, 1.2],
        linestyle='-',
        alpha=0.95
    )

    title = f"{trade['symbol']} {trade['side']}  leg={trade['leg_id']}  outcome={trade['outcome']}"
    outfile = outdir / f"{trade['symbol']}_{int(trade['trade_id'])}_fib.png"
    mpf.plot(
        win, type="candle", volume=False, style="yahoo",
        hlines=hlines, title=title, savefig=str(outfile)
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/futures_merged_5m")
    ap.add_argument("--trades", default="reports/trades/trades.csv")
    ap.add_argument("--outdir", default="reports/qc")
    ap.add_argument("--per_symbol", type=int, default=10)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    trades = pd.read_csv(args.trades)

    for sym, g in trades.groupby("symbol"):
        df_px = load_prices(args.data, sym)
        sample = g.sample(min(args.per_symbol, len(g)), random_state=42)
        for _, tr in sample.iterrows():
            plot_trade(df_px, tr, outdir)

    print(f"Saved plots to {outdir}")

if __name__ == "__main__":
    main()
