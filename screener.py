#!/usr/bin/env python3
"""
Alpha Architect Quality Momentum Screener — Swedish Stocks
===========================================================
Based on: "The Quantitative Momentum Investing Philosophy" (Alpha Architect)
Universe: 275 Swedish stocks (Nasdaq Stockholm) from Börsdata CSV

METHODOLOGY (5 steps):
  1. Universe: 275 stocks from stocks.csv (Large + Mid + Small Cap)
  2. Outlier removal: drop top 5% by 1-month return (lottery stocks),
                      drop bottom 10% by market beta (weak trend stocks)
  3. Momentum rank: 12_2 momentum = 12-month return EXCLUDING last month
                    (t-12 to t-1, standard academic convention)
  4. Quality filter: keep top 50% by momentum, then apply Frog-in-the-Pan (FIP)
                     FIP = sign(12m return) × (%negative days − %positive days)
                     Lower FIP = smoother, more consistent momentum = preferred
                     Keep top 50 by FIP score from momentum top-50%
  5. Quality gate: F-Score ≥ 5 (from Börsdata CSV — Piotroski quality check)
                   Stocks failing F-Score gate are flagged but not hard-excluded
                   (soft filter — shown in output but marked)

OUTPUT:
  screener_data.json — loaded by index.html
  prev_ranks.json    — rank history for "vs last week" column

HOW TO RUN:
  pip install yfinance pandas
  python screener.py

AUTOMATION:
  GitHub Actions runs this every Friday at 17:30 UTC (18:30 CET)
"""

import csv
import json
import math
import os
import time
import datetime
import warnings
warnings.filterwarnings("ignore")

try:
    import yfinance as yf
    import pandas as pd
    import numpy as np
except ImportError:
    print("ERROR: pip install yfinance pandas numpy")
    raise

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

CSV_FILE        = "stocks.csv"          # Börsdata export (ships with repo)
OUTPUT_JSON     = "screener_data.json"
PREV_RANKS_FILE = "prev_ranks.json"
TOP_N           = 20

# Outlier filters
TOP_PCT_1M_REMOVE   = 0.05   # Remove top 5% by 1-month return (lottery stocks)
BOT_PCT_BETA_REMOVE = 0.10   # Remove bottom 10% by market beta

# Momentum funnel
MOM_TOP_PCT   = 0.50   # Keep top 50% by 12_2 momentum score
FIP_TOP_N     = 50     # From momentum top 50%, keep top 50 by FIP quality

# Quality gate (soft — flags stocks, doesn't hard-exclude)
FSCORE_MIN    = 5      # Piotroski F-Score minimum for "quality" flag

# Calendar lookback
DAYS_HISTORY  = 420    # Fetch ~14 months of daily price data
DAYS_12M      = 252    # Trading days in 12 months
DAYS_1M       = 21     # Trading days in 1 month

# Market index for beta calculation
INDEX_TICKER  = "^OMXSPI"   # OMX Stockholm All-Share Index

# ─────────────────────────────────────────────────────────────────────────────
# LOAD UNIVERSE FROM CSV
# ─────────────────────────────────────────────────────────────────────────────

def load_universe(csv_path: str) -> list[dict]:
    """
    Load the 275-stock universe from the Börsdata CSV export.
    Converts Börsdata tickers to Yahoo Finance format (spaces → hyphens, + .ST).
    """
    stocks = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            raw_ticker = row["Info - Ticker"].strip()
            if not raw_ticker:
                continue

            # Convert to Yahoo Finance format: "VOLV B" → "VOLV-B.ST"
            yf_ticker = raw_ticker.replace(" ", "-") + ".ST"

            # Parse F-Score (may be empty for some stocks)
            fs_raw = row.get("F-Score - Poäng", "").strip()
            fscore = int(fs_raw) if fs_raw.isdigit() else None

            # Parse market cap (MSEK, Swedish decimal format)
            mc_raw = row.get("Börsvärde - Senaste", "").strip().replace(".", "").replace(",", ".")
            try:
                market_cap_msek = float(mc_raw)
            except ValueError:
                market_cap_msek = None

            stocks.append({
                "borsdata_id":    row.get("Börsdata ID", "").strip(),
                "name":           row.get("Bolagsnamn", "").strip(),
                "ticker_borsdata":raw_ticker,
                "ticker":         yf_ticker,
                "fscore":         fscore,
                "market_cap_msek":market_cap_msek,
                "sector":         row.get("Info - Sektor", "").strip(),
                "industry":       row.get("Info - Bransch", "").strip(),
            })

    return stocks


# ─────────────────────────────────────────────────────────────────────────────
# PRICE DATA FETCHING
# ─────────────────────────────────────────────────────────────────────────────

def fetch_index_returns(start_date: str, end_date: str) -> pd.Series | None:
    """Fetch daily returns for the OMX Stockholm index (for beta calculation)."""
    try:
        idx = yf.Ticker(INDEX_TICKER)
        hist = idx.history(start=start_date, end=end_date, auto_adjust=True)
        if hist.empty or len(hist) < 50:
            return None
        return hist["Close"].pct_change().dropna()
    except Exception as e:
        print(f"  ⚠ Could not fetch index {INDEX_TICKER}: {e}")
        return None


def fetch_prices(ticker: str, start_date: str, end_date: str) -> pd.Series | None:
    """Fetch daily close prices for a single ticker."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date, auto_adjust=True)
        if hist.empty or len(hist) < 60:
            return None
        return hist["Close"].dropna()
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# METRIC CALCULATIONS
# ─────────────────────────────────────────────────────────────────────────────

def calc_12_2_momentum(prices: pd.Series) -> float | None:
    """
    12_2 Momentum: price return from 12 months ago to 1 month ago.
    Standard Alpha Architect convention — skips the most recent month
    to avoid short-term reversal contamination.
    """
    n = len(prices)
    if n < DAYS_12M + 5:
        return None
    p_12m_ago = float(prices.iloc[max(0, n - DAYS_12M - 1)])
    p_1m_ago  = float(prices.iloc[max(0, n - DAYS_1M - 1)])
    if p_12m_ago <= 0:
        return None
    return (p_1m_ago / p_12m_ago - 1.0) * 100.0


def calc_1m_return(prices: pd.Series) -> float | None:
    """Most recent 1-month return (used for outlier filter)."""
    n = len(prices)
    if n < DAYS_1M + 2:
        return None
    p_now   = float(prices.iloc[-1])
    p_1m    = float(prices.iloc[max(0, n - DAYS_1M - 1)])
    if p_1m <= 0:
        return None
    return (p_now / p_1m - 1.0) * 100.0


def calc_beta(stock_returns: pd.Series, index_returns: pd.Series) -> float | None:
    """
    Market beta over the available overlapping period.
    Beta = Cov(stock, index) / Var(index)
    """
    try:
        aligned = pd.concat([stock_returns, index_returns], axis=1, join="inner")
        aligned.columns = ["stock", "index"]
        if len(aligned) < 60:
            return None
        cov = aligned["stock"].cov(aligned["index"])
        var = aligned["index"].var()
        if var == 0:
            return None
        return cov / var
    except Exception:
        return None


def calc_fip_score(prices: pd.Series) -> float | None:
    """
    Frog-in-the-Pan (FIP) Score — Alpha Architect quality filter.

    FIP measures the CONSISTENCY of momentum:
      FIP = sign(12m_return) × (%days_negative − %days_positive)

    Interpretation:
      For stocks with POSITIVE 12m momentum (which is what we want):
        sign = +1
        FIP = %neg_days − %pos_days
        → More positive days = lower (more negative) FIP = BETTER quality momentum
        → We PREFER stocks with LOWEST FIP (most smooth, consistent upward drift)

    The FIP filter eliminates "big jump" momentum — stocks that gained
    most of their return in a few big days (lottery-like, prone to reversal)
    versus stocks with gradual, consistent upward drift (more durable).

    Calculated over the 12-month lookback window.
    """
    n = len(prices)
    if n < DAYS_12M + 5:
        return None

    # Use the 12-month window (same as momentum calculation)
    window = prices.iloc[max(0, n - DAYS_12M - 1):-DAYS_1M] if n > DAYS_12M + DAYS_1M else prices.iloc[-DAYS_12M:]
    if len(window) < 50:
        return None

    daily_returns = window.pct_change().dropna()
    if len(daily_returns) < 50:
        return None

    total_days  = len(daily_returns)
    pos_days    = (daily_returns > 0).sum()
    neg_days    = (daily_returns < 0).sum()

    pct_pos = pos_days / total_days
    pct_neg = neg_days / total_days

    # 12-month return sign (from the same window)
    p_start = float(window.iloc[0])
    p_end   = float(window.iloc[-1])
    if p_start <= 0:
        return None

    sign_12m = 1 if p_end >= p_start else -1

    fip = sign_12m * (pct_neg - pct_pos)
    return round(fip, 6)


def calc_rsl(prices: pd.Series, period: int = 130) -> float | None:
    """RSL = current price / 130-day SMA. Used as trend filter display."""
    n = len(prices)
    if n < period:
        return None
    current = float(prices.iloc[-1])
    sma = float(prices.iloc[-period:].mean())
    if sma <= 0:
        return None
    return round(current / sma, 4)


# ─────────────────────────────────────────────────────────────────────────────
# PREV RANKS
# ─────────────────────────────────────────────────────────────────────────────

def load_prev_ranks() -> dict:
    if os.path.exists(PREV_RANKS_FILE):
        with open(PREV_RANKS_FILE) as f:
            return json.load(f)
    return {}


def save_prev_ranks(top20: list) -> None:
    ranks = {r["ticker"]: r["rank"] for r in top20}
    with open(PREV_RANKS_FILE, "w") as f:
        json.dump(ranks, f)


# ─────────────────────────────────────────────────────────────────────────────
# SECTOR TRANSLATION (Swedish → English)
# ─────────────────────────────────────────────────────────────────────────────

SECTOR_MAP = {
    "Material":           "Materials",
    "Informationsteknik": "Technology",
    "Hälsovård":          "Healthcare",
    "Kraftförsörjning":   "Utilities",
    "Telekommunikation":  "Telecom",
    "Dagligvaror":        "Cons. Staples",
    "Industri":           "Industrials",
    "Energi":             "Energy",
    "Sällanköpsvaror":    "Cons. Discret.",
    "Finans":             "Financials",
    "Fastigheter":        "Real Estate",
}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SCREENER
# ─────────────────────────────────────────────────────────────────────────────

def main():
    now        = datetime.datetime.now(datetime.timezone.utc)
    end_date   = datetime.datetime.today()
    start_date = end_date - datetime.timedelta(days=DAYS_HISTORY)
    start_str  = start_date.strftime("%Y-%m-%d")
    end_str    = end_date.strftime("%Y-%m-%d")

    # ── Load universe ─────────────────────────────────────────────────────────
    universe = load_universe(CSV_FILE)
    total    = len(universe)

    print("=" * 70)
    print("  Alpha Architect Quality Momentum — OMX Stockholm")
    print(f"  Universe: {total} stocks  |  Börsdata CSV + Yahoo Finance prices")
    print(f"  Running: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)
    print()

    # ── Fetch market index for beta ───────────────────────────────────────────
    print(f"  Fetching index returns ({INDEX_TICKER})…")
    index_returns = fetch_index_returns(start_str, end_str)
    if index_returns is None:
        print("  ⚠ Could not fetch index — beta filter will be skipped")
    else:
        print(f"  ✓ Index: {len(index_returns)} trading days\n")
    time.sleep(0.5)

    # ── Fetch prices + compute metrics ───────────────────────────────────────
    records = []
    skipped = []

    for i, stock in enumerate(universe):
        ticker = stock["ticker"]
        name   = stock["name"]
        print(f"[{i+1:>3}/{total}] {ticker:<22}", end="", flush=True)

        prices = fetch_prices(ticker, start_str, end_str)
        if prices is None or len(prices) < DAYS_12M + 5:
            days = len(prices) if prices is not None else 0
            reason = f"Insufficient history ({days} days, need {DAYS_12M+5})"
            skipped.append({"name": name, "ticker": ticker, "reason": reason, "days": days})
            print(f"✗  {reason}")
            time.sleep(0.3)
            continue

        # Core metrics
        mom_12_2 = calc_12_2_momentum(prices)
        mom_1m   = calc_1m_return(prices)
        fip      = calc_fip_score(prices)
        rsl      = calc_rsl(prices)

        if mom_12_2 is None:
            skipped.append({"name": name, "ticker": ticker, "reason": "Cannot compute 12_2 momentum", "days": len(prices)})
            print("✗  Cannot compute 12_2 momentum")
            time.sleep(0.3)
            continue

        # Beta (optional — skip if index unavailable)
        beta = None
        if index_returns is not None:
            stock_ret = prices.pct_change().dropna()
            beta = calc_beta(stock_ret, index_returns)

        current_price = float(prices.iloc[-1])

        records.append({
            "name":            name,
            "ticker":          ticker,
            "ticker_display":  stock["ticker_borsdata"],
            "sector":          SECTOR_MAP.get(stock["sector"], stock["sector"]),
            "fscore":          stock["fscore"],
            "market_cap_msek": stock["market_cap_msek"],
            "price":           round(current_price, 2),
            "mom_12_2":        round(mom_12_2, 2),
            "mom_1m":          round(mom_1m, 2) if mom_1m is not None else None,
            "fip":             fip,
            "rsl":             rsl,
            "beta":            round(beta, 3) if beta is not None else None,
        })

        beta_str   = f"β={beta:.2f}" if beta is not None else "β=N/A"
        fip_str    = f"FIP={fip:.3f}" if fip is not None else "FIP=N/A"
        fscore_str = f"F={stock['fscore']}" if stock["fscore"] is not None else "F=N/A"
        print(f"✓  12_2={mom_12_2:+7.1f}%  1M={mom_1m or 0:+6.1f}%  {fip_str}  RSL={rsl or 0:.3f}  {beta_str}  {fscore_str}")

        time.sleep(0.3)

    print(f"\n{'─'*70}")
    print(f"  Valid: {len(records)}   Skipped: {len(skipped)}")
    print(f"{'─'*70}\n")

    if len(records) < TOP_N:
        print(f"⚠  Only {len(records)} valid stocks — need at least {TOP_N}. Aborting.")
        return

    df = pd.DataFrame(records)
    universe_count = len(df)

    # ── STEP 2: Outlier removal ───────────────────────────────────────────────
    # A) Remove top 5% by 1-month return (lottery/speculative stocks)
    df_with_1m = df[df["mom_1m"].notna()].copy()
    threshold_1m = df_with_1m["mom_1m"].quantile(1 - TOP_PCT_1M_REMOVE)
    lottery_removed = df_with_1m[df_with_1m["mom_1m"] > threshold_1m]
    df = df[~df["ticker"].isin(lottery_removed["ticker"])].copy()
    print(f"→ Step 2a — Removed top {int(TOP_PCT_1M_REMOVE*100)}% by 1M return (lottery): "
          f"{len(lottery_removed)} removed, {len(df)} remain")
    print(f"  (1M threshold: >{threshold_1m:.1f}%)")

    # B) Remove bottom 10% by beta (weakest trend / near-zero momentum stocks)
    if index_returns is not None:
        df_with_beta = df[df["beta"].notna()].copy()
        if len(df_with_beta) > 20:
            threshold_beta = df_with_beta["beta"].quantile(BOT_PCT_BETA_REMOVE)
            low_beta_removed = df_with_beta[df_with_beta["beta"] < threshold_beta]
            df = df[~df["ticker"].isin(low_beta_removed["ticker"])].copy()
            print(f"→ Step 2b — Removed bottom {int(BOT_PCT_BETA_REMOVE*100)}% by beta: "
                  f"{len(low_beta_removed)} removed, {len(df)} remain")
            print(f"  (Beta threshold: <{threshold_beta:.3f})")
        else:
            print("→ Step 2b — Beta filter skipped (too few stocks with beta)")
    else:
        print("→ Step 2b — Beta filter skipped (no index data)")

    # ── STEP 3: Momentum rank (12_2) ─────────────────────────────────────────
    df = df.sort_values("mom_12_2", ascending=False).reset_index(drop=True)
    mom_cutoff_idx = max(1, int(len(df) * MOM_TOP_PCT))
    df_top_mom = df.iloc[:mom_cutoff_idx].copy()
    print(f"\n→ Step 3  — Top {int(MOM_TOP_PCT*100)}% by 12_2 momentum: "
          f"{len(df_top_mom)} stocks (cutoff: >{df_top_mom['mom_12_2'].iloc[-1]:.1f}%)")

    # ── STEP 4: FIP quality filter ────────────────────────────────────────────
    df_with_fip = df_top_mom[df_top_mom["fip"].notna()].copy()
    df_no_fip   = df_top_mom[df_top_mom["fip"].isna()].copy()

    # Sort by FIP ascending (lowest FIP = smoothest momentum = best quality)
    df_with_fip = df_with_fip.sort_values("fip", ascending=True).reset_index(drop=True)

    # Take top FIP_TOP_N (most consistent momentum)
    df_fip_passed = df_with_fip.iloc[:FIP_TOP_N].copy()

    print(f"→ Step 4  — FIP quality filter: top {FIP_TOP_N} by consistency")
    print(f"  ({len(df_no_fip)} stocks had no FIP data — excluded from FIP ranking)")

    # ── STEP 5: Final sort by 12_2 momentum, take top 20 ─────────────────────
    # After FIP filter, re-rank by 12_2 momentum for the final top 20
    df_final = df_fip_passed.sort_values("mom_12_2", ascending=False).reset_index(drop=True)
    df_final["rank"] = df_final.index + 1

    top20 = df_final.head(TOP_N).copy()

    # ── Attach previous ranks + F-Score quality flag ──────────────────────────
    prev_ranks = load_prev_ranks()
    top20_list = []

    for _, row in top20.iterrows():
        fscore_ok = (row["fscore"] is not None and row["fscore"] >= FSCORE_MIN)
        top20_list.append({
            "name":            row["name"],
            "ticker":          row["ticker"],
            "ticker_display":  row["ticker_display"],
            "sector":          row["sector"],
            "rank":            int(row["rank"]),
            "prev_rank":       prev_ranks.get(row["ticker"], None),
            "price":           row["price"],
            "mom_12_2":        row["mom_12_2"],
            "mom_1m":          row["mom_1m"],
            "fip":             row["fip"],
            "rsl":             row["rsl"],
            "beta":            row["beta"],
            "fscore":          row["fscore"],
            "fscore_ok":       fscore_ok,
            "market_cap_msek": row["market_cap_msek"],
        })

    save_prev_ranks(top20_list)

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  TOP {TOP_N} — ALPHA ARCHITECT QUALITY MOMENTUM")
    print(f"{'='*70}")
    for r in top20_list:
        prev = f"(prev #{r['prev_rank']})" if r["prev_rank"] else "(new)"
        fs   = f"F={r['fscore']}" if r["fscore"] is not None else "F=N/A"
        fip  = f"FIP={r['fip']:.3f}" if r["fip"] is not None else "FIP=N/A"
        print(f"  #{r['rank']:>2}  {r['ticker_display']:<18}  "
              f"12_2={r['mom_12_2']:+7.1f}%  {fip}  RSL={r['rsl'] or 0:.3f}  {fs}  {prev}")

    # ── Build funnel stats for JSON ───────────────────────────────────────────
    funnel = {
        "universe":           universe_count,
        "after_data_filter":  len(records),
        "after_lottery_rm":   universe_count - len(lottery_removed),
        "after_momentum_top": len(df_top_mom),
        "after_fip":          len(df_fip_passed),
        "final_top20":        TOP_N,
    }

    # ── Write JSON ────────────────────────────────────────────────────────────
    output = {
        "updated":          now.strftime("%Y-%m-%d %H:%M UTC"),
        "date":             now.strftime("%Y-%m-%d"),
        "total_attempted":  total,
        "total_valid":      len(records),
        "skipped_count":    len(skipped),
        "funnel":           funnel,
        "top20":            top20_list,
        "skipped":          skipped,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✅  Saved → {OUTPUT_JSON}")
    print(f"    Universe: {total} stocks → {len(records)} valid → Top {TOP_N} selected")
    print(f"    Updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n")


if __name__ == "__main__":
    main()
