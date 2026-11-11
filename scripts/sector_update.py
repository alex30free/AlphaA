#!/usr/bin/env python3
"""
Weekly update for the StockLab sector trends page.

For every stock in SECTORS it downloads daily prices from Yahoo Finance (since 2006),
turns them into weekly closes (last trading day of each week), and builds one
equal-weight index per sector: each week the index moves by the average weekly
price change of the sector's stocks (stocks listed later join from their first week).

Prices are Yahoo's closing prices, which Yahoo already adjusts for share splits and
spin-offs (e.g. Atlas Copco -> Epiroc 2018, SCA -> Essity 2017). Dividends are not
added back: it is a price index, like a chart of closing prices. Yahoo's separate
"adjusted close" is not used, because it is unreliable for some Swedish stocks.

Only the Python standard library is used. Writes data.json next to index.html.
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "data.json"))
START = date(2006, 1, 6)            # first Friday of the series
UA = "Mozilla/5.0 (compatible; StockLab sector trends; github.com/alex30free)"

# Edit this list to change sectors or members. Tickers are Yahoo Finance symbols.
SECTORS = [
    ("bank", "Banks", [("NDA-SE.ST", "Nordea"), ("SEB-A.ST", "SEB A"), ("SHB-A.ST", "Handelsbanken A"), ("SWED-A.ST", "Swedbank A")]),
    ("gaming", "Gaming", [("BETS-B.ST", "Betsson B"), ("EVO.ST", "Evolution"), ("KAMBI.ST", "Kambi")]),
    ("telecom", "Telecom operators", [("TELIA.ST", "Telia"), ("TEL2-B.ST", "Tele2 B")]),
    ("construction", "Construction", [("NCC-B.ST", "NCC B"), ("JM.ST", "JM"), ("PEAB-B.ST", "Peab B"), ("SKA-B.ST", "Skanska B")]),
    ("realestate", "Real estate", [("FABG.ST", "Fabege"), ("HUFV-A.ST", "Hufvudstaden A"), ("SBB-B.ST", "SBB B"),
                                    ("WALL-B.ST", "Wallenstam B"), ("WIHL.ST", "Wihlborgs"), ("SAGA-B.ST", "Sagax B")]),
    ("staples", "Consumer staples", [("AXFO.ST", "Axfood"), ("HM-B.ST", "H&M B")]),
    ("forest", "Forestry", [("BILL.ST", "Billerud"), ("HOLM-B.ST", "Holmen B"), ("SCA-B.ST", "SCA B"), ("STE-R.ST", "Stora Enso R")]),
    ("durables", "Consumer durables", [("ELUX-B.ST", "Electrolux B"), ("HUSQ-B.ST", "Husqvarna B"), ("NIBE-B.ST", "Nibe B")]),
    ("materials", "Commodities", [("BOL.ST", "Boliden"), ("IPCO.ST", "International Petroleum"), ("LUMI.ST", "Lundin Mining"),
                                   ("LUG.ST", "Lundin Gold"), ("SSAB-A.ST", "SSAB A")]),
    ("industry", "Industrials", [("ABB.ST", "ABB"), ("ALIV-SDB.ST", "Autoliv SDB"), ("ASSA-B.ST", "Assa Abloy B"),
                                  ("ATCO-A.ST", "Atlas Copco A"), ("SAND.ST", "Sandvik"), ("SKF-B.ST", "SKF B"),
                                  ("TREL-B.ST", "Trelleborg B"), ("VOLV-B.ST", "Volvo B"), ("EPI-A.ST", "Epiroc A")]),
]


def fetch_daily(symbol):
    """Daily (date, close, adjclose) from Yahoo's chart API."""
    p1 = int(datetime(2005, 12, 1, tzinfo=timezone.utc).timestamp())
    p2 = int(time.time()) + 86400
    last_err = None
    for host in ("query1", "query2"):
        url = ("https://%s.finance.yahoo.com/v8/finance/chart/%s?period1=%d&period2=%d&interval=1d"
               "&events=div%%2Csplit&includeAdjustedClose=true" % (host, urllib.parse.quote(symbol), p1, p2))
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                js = json.loads(resp.read().decode("utf-8"))
            res = js["chart"]["result"][0]
            off = res.get("meta", {}).get("gmtoffset", 3600)
            ts = res.get("timestamp") or []
            close = res["indicators"]["quote"][0]["close"]
            adj = (res["indicators"].get("adjclose") or [{}])[0].get("adjclose") or close
            rows = []
            for t, c, a in zip(ts, close, adj):
                if c is None or a is None or c <= 0 or a <= 0:
                    continue
                d = datetime.fromtimestamp(t + off, timezone.utc).date()
                if rows and rows[-1][0] == d:
                    rows[-1] = (d, c, a)
                else:
                    rows.append((d, c, a))
            if len(rows) < 20:
                raise ValueError("only %d rows" % len(rows))
            return rows
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(2)
    raise RuntimeError(last_err)


def price_index(rows):
    """Daily closes (split- and spin-off-adjusted by Yahoo)."""
    return [(d, c) for d, c, _ in rows], []


def fridays(until):
    d, out = START, []
    while d <= until:
        out.append(d)
        d += timedelta(days=7)
    return out


def weekly(daily, weeks):
    """Close of the last trading day on or before each Friday (None before listing)."""
    vals, j, last = [], 0, None
    for f in weeks:
        while j < len(daily) and daily[j][0] <= f:
            last = daily[j][1]
            j += 1
        vals.append(last)
    return vals


def r(v):
    return None if v is None else round(v, 3 if v < 10 else 2)


def load_previous():
    try:
        with open(OUT, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def main():
    prev = load_previous()
    prev_stocks = prev.get("stocks", {})
    prev_weeks = prev.get("weeks", {})
    daily = {}
    events = {}
    for _, _, members in SECTORS:
        for sym, name in members:
            if sym in daily:
                continue
            try:
                d, ev = price_index(fetch_daily(sym))
                daily[sym] = d
                events[sym] = ev
                print("OK   %-12s %5d days %s -> %s" % (sym, len(d), d[0][0], d[-1][0]))
            except Exception as exc:  # noqa: BLE001
                print("WARN %-12s %s" % (sym, exc))
            time.sleep(1)
    if not daily:
        print("ERROR nothing downloaded; data.json left unchanged")
        return 1

    last_day = max(v[-1][0] for v in daily.values())
    end = last_day + timedelta(days=(4 - last_day.weekday()) % 7)   # Friday of the latest week
    weeks = fridays(end)

    stocks = {}
    for _, _, members in SECTORS:
        for sym, name in members:
            if sym in daily:
                stocks[sym] = {"name": name, "close": [r(v) for v in weekly(daily[sym], weeks)]}
            elif sym in prev_stocks and prev_weeks.get("start") == START.isoformat():
                old = prev_stocks[sym]["close"]
                stocks[sym] = {"name": name, "close": (old + [old[-1]] * len(weeks))[:len(weeks)], "stale": True}
                print("WARN %-12s kept previous data" % sym)

    sectors, index = [], {}
    for sid, sname, members in SECTORS:
        syms = [s for s, _ in members if s in stocks]
        idx, level = [], None
        for k in range(len(weeks)):
            if level is None:
                if any(stocks[s]["close"][k] for s in syms):
                    level = 100.0
                idx.append(r(level) if level else None)
                continue
            rets = []
            for s in syms:
                a, b = stocks[s]["close"][k - 1], stocks[s]["close"][k]
                if a and b:
                    rets.append(b / a - 1)
            if rets:
                level *= 1 + sum(rets) / len(rets)
            idx.append(r(level))
        index[sid] = idx
        sectors.append({"id": sid, "name": sname, "members": [s for s, _ in members],
                        "missing": [s for s, _ in members if s not in stocks]})

    payload = {
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_through": last_day.isoformat(),
        "source": "Yahoo Finance daily closing prices (adjusted by Yahoo for splits and spin-offs); weekly = last close of each week",
        "method": "Equal-weight sector index: each week moves by the average weekly price change of its stocks",
        "weeks": {"start": START.isoformat(), "count": len(weeks), "step_days": 7},
        "sectors": sectors,
        "index": index,
        "stocks": stocks,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    print("Wrote %s: %d sectors, %d stocks, %d weeks, data through %s" % (
        OUT, len(sectors), len(stocks), len(weeks), last_day))
    return 0


if __name__ == "__main__":
    sys.exit(main())
