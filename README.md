# Swedish Sector Trends

Part of [StockLab](https://alex30free.github.io/stocklab/).
Ten equal-weight sector indices built from Swedish stocks since 2006, each with its
40-week moving average, plus a short summary of the current situation.

- `index.html` – the page (GitHub Pages)
- `data.json` – weekly data, rebuilt every Friday by `.github/workflows/sector-weekly.yml`
- `scripts/sector_update.py` – downloads prices from Yahoo Finance and builds the indices.
  Edit the `SECTORS` list at the top to change sectors or stocks.

For informational purposes only — not financial advice.
