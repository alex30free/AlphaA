# 🇸🇪 Swedish Quality Momentum Screener

Automated weekly screener for Swedish stocks (Nasdaq Stockholm) using the **Alpha Architect Quality Momentum** methodology — updated every Friday evening after market close.

## Live Screener

Enable **GitHub Pages** (Settings → Pages → Branch: `main`, folder: `/root`) to view the live screener at:
```
https://<your-username>.github.io/<repo-name>/
```

---

## Methodology — 5-Step Funnel

| Step | What happens |
|------|-------------|
| 1 | **Universe**: 275 Swedish stocks from `stocks.csv` (sourced from Börsdata) |
| 2 | **Outlier removal**: Remove top 5% by 1-month return (lottery stocks) + bottom 10% by market beta |
| 3 | **12_2 Momentum rank**: Rank by 12-month return *excluding* last month (t-12 → t-1). Keep top 50% |
| 4 | **FIP quality filter**: From momentum top 50%, keep top 50 by Frog-in-the-Pan score (smoothest momentum) |
| 5 | **Quality gate**: F-Score ≥ 5 shown as soft flag. Re-rank by 12_2 momentum → publish top 20 |

---

## Key Signals Explained

### 12_2 Momentum
```
12_2 = Price(t-1) / Price(t-12) - 1
```
Skip the most recent month to avoid short-term reversal. Standard academic convention (Jegadeesh & Titman, 1993).

### Frog-in-the-Pan (FIP)
```
FIP = sign(12m_return) × (%days_negative − %days_positive)
```
Lower FIP = more positive trading days = smoother, more consistent upward trend.
Stocks with gradual drift outperform "big jump" momentum (Alpha Architect research).

### Piotroski F-Score (from Börsdata CSV)
9-signal financial health rating. Scores 7–9 = strong quality. Used as soft filter / conviction booster.

---

## Automation

| Step | What happens |
|------|-------------|
| 1 | GitHub Actions triggers every **Friday at 17:30 UTC** |
| 2 | `screener.py` fetches 14 months of daily prices for 275 stocks via Yahoo Finance |
| 3 | Runs the 5-step funnel: outlier filter → momentum rank → FIP → top 20 |
| 4 | Saves `screener_data.json` + updates `prev_ranks.json` for rank comparison |
| 5 | Commits & pushes → GitHub Pages serves the updated `index.html` automatically |

---

## Files

```
├── index.html               # Screener webpage
├── screener.py              # Python screener — runs the 5-step funnel
├── stocks.csv               # 275-stock Börsdata universe (updated quarterly)
├── screener_data.json       # Auto-generated: top 20 results (committed by Actions)
├── prev_ranks.json          # Auto-generated: last week's ranks for comparison
└── .github/
    └── workflows/
        └── update_screener.yml   # GitHub Actions — weekly automation
```

---

## Setup

1. **Fork / clone** this repository
2. Go to **Settings → Pages** → Source: Deploy from branch `main`, folder `/` (root)
3. The workflow runs automatically every Friday — or trigger manually via **Actions → Run workflow**

### Run locally
```bash
pip install yfinance pandas numpy
python screener.py
```

---

## Dependencies

- `yfinance` — live price data from Yahoo Finance
- `pandas` — data manipulation
- `numpy` — numerical calculations

---

## Disclaimer

For informational purposes only. Not financial advice. Always conduct your own research before investing.

---

*Based on: Alpha Architect — "The Quantitative Momentum Investing Philosophy" (Vogel & Gray, 2016)*  
*Jegadeesh & Titman (1993) · Asness, Moskowitz & Pedersen (2013) · Piotroski (2000)*
