TC1 Bot – Automated Futures Trading (v0.1.0)
Overview

TC1 Bot is an automated crypto futures trading engine built around the TC1 strategy.
It strictly follows fixed Fibonacci geometry and includes multi-timeframe confluences, filters, and backtesting tools.

TC1 Strategy Rules (Non-Negotiable)
Fib Level	Purpose	Rule
0	Swing Low / High	Anchor start of move
1	Swing High / Low	Anchor end of move
0.382	Entry	Bot enters trade here
0.17	Stop Loss	Hard stop beyond invalidation
1.272	Take Profit	Fixed target

⚠️ Important:

0.618 is NEVER used in TC1.

Entry is always at the 0.382 retracement.

Short = flip anchors: 0 = high, 1 = low.

Project Structure
TC1_Bot/
├── tools/          # Core detection, backtesting & trade scripts
├── configs/        # JSON/YAML configs for strategy and filters
├── reports/        # Backtest outputs (ignored in git)
├── data/           # Binance klines & trade history (ignored in git)
├── requirements.txt # Exact Python dependencies
└── .gitignore      # Prevents committing CSVs, data, reports, zips

Installation
git clone https://github.com/Buckhopbuilder/TC1-Bot_Final.git
cd TC1-Bot_Final
pip install -r requirements.txt

Usage
Backtesting
python tools/tc1_backtester.py --config configs/defaults_futures.json


Requires pre-downloaded Binance futures data in data/.

Outputs results into reports/.

Live Paper Trading (future phase)
python tools/tc1_live_runner.py --config configs/live_default.json

Latest Release

Grab the latest clean, code-only release here:
🔗 Releases

Includes:

Tools + configs

requirements.txt

SHA256 checksum for verification

Contributing

Always confirm TC1 fib anchors before modifying detection code.

Never commit data, reports, or CSVs.

Use branches for experiments:

git checkout -b feature/new-filter

License

Private repository.
Do not redistribute without explicit permission.
