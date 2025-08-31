## TC1 Bot – PR Checklist

**Geometry (must stay fixed)**
- [ ] Entry at **0.382**
- [ ] Stop at **0.17**
- [ ] TP at **1.272**
- [ ] **0.618 is NOT used** anywhere
- [ ] Fib anchors correct:
  - Long: 0 = swing low, 1 = swing high
  - Short: 0 = swing high, 1 = swing low

**No data leaks**
- [ ] No `data/`, `reports/`, `*.csv`, `*.zip` added
- [ ] `.gitignore` unchanged or stricter

**Sanity**
- [ ] `requirements.txt` still installs
- [ ] Brief test/backtest notes included
- [ ] README/CHANGELOG updated if behavior changed
