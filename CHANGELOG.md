# TC1 Bot – Changelog

All notable changes to this project will be documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),  
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [v0.1.0] - 2025-08-31
### Added
- **Initial clean release** of the TC1 Bot codebase.
- Full `tools/` and `configs/` directories committed.
- Added `.gitignore` to exclude:
  - `data/` folder  
  - `reports/` folder  
  - All `.csv` / `.zip` files  
  - `__pycache__/`
- Added `requirements.txt` to lock Python dependencies for reproducibility.
- Added **README.md** documenting:
  - Project structure
  - TC1 Fibonacci strategy rules
  - Usage instructions for backtesting and future live trading.

### Fixed
- Removed old, experimental scripts and unrelated artifacts.
- Ensured no data, reports, or large binary files are included in the repo.

### Notes
- **TC1 geometry locked in**:
    - Entry → **0.382**
    - Stop Loss → **0.17**
    - Take Profit → **1.272**
    - **0.618 is NEVER used.**
- This release is the clean baseline for all future development.
- Binance futures data (`data/`) and reports (`reports/`) remain local-only.

---

## Upcoming
- Add automated backtest runners with summary reporting.
- Integrate TC1 multi-timeframe confluences for EMA alignment.
- Add CI/CD safeguards to prevent committing large CSVs/zips.
- Future release tags will track incremental feature additions and performance upgrades.

---

[v0.1.0]: https://github.com/Buckhopbuilder/TC1-Bot_Final/releases/tag/v0.1.0
