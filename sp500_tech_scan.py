#!/usr/bin/env python3
"""Technischer Scan aller aktuellen S&P-500-Aktien.

Berechnet je Aktie: aktueller Kurs, RSI 14, SMA 50, SMA 200, EMA 50, ADX 14
und schreibt das Ergebnis als JSON (plus Ausgabe auf der Konsole).

Aufruf:
    python sp500_tech_scan.py [--out DATEI.json] [--sort SPALTE] [--quiet]
    python sp500_tech_scan.py --version

Grundlage ist ad_line.py (Vorlage): Konstituenten von Wikipedia, Kurse in einem
einzigen yfinance-Batch-Download. Anders als dort werden hier dividenden- und
split-bereinigte Kurse (auto_adjust=True) genutzt, damit Splits die gleitenden
Durchschnitte und Indikatoren nicht verfälschen.

Indikatoren (Standarddefinitionen):
- RSI 14:  Wilder-Glättung (RMA, alpha = 1/14) der Auf-/Abwärtsbewegungen.
- SMA 50/200: einfacher gleitender Durchschnitt der Schlusskurse.
- EMA 50:  exponentieller Durchschnitt, alpha = 2/(50+1).
- ADX 14:  Wilder: +DM/-DM und True Range per RMA geglättet, DX daraus,
           ADX = RMA(DX, 14).
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

SKRIPT_VERSION = "1.0.0"

if sys.argv[1:2] == ["--version"]:
    print(f"{os.path.basename(__file__)} {SKRIPT_VERSION}")
    raise SystemExit(0)

# Fremdpakete liegen in der gemeinsamen vendor-Umgebung unter .../Chats/vendor.
VENDOR_DIR = "/media/veracrypt10/Daten/ClaudeCode/Chats/vendor"
if os.path.isdir(VENDOR_DIR) and VENDOR_DIR not in sys.path:
    sys.path.insert(0, VENDOR_DIR)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402
import yfinance as yf  # noqa: E402
from bs4 import BeautifulSoup  # noqa: E402

log = logging.getLogger("sp500_tech_scan")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(BASE_DIR, "data.json")

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
WIKI_HEADERS = {"User-Agent": "SP500-Tech-Scan/1.0 (private Marktanalyse)"}
MIN_CONSTITUENTS = 400

# SMA 200 braucht 200 Handelstage, ADX/RSI zusätzlich Einschwingzeit.
# 2 Jahre (~500 Handelstage) reichen dafür sicher.
DOWNLOAD_PERIOD = "2y"

RSI_LEN = 14
ADX_LEN = 14

COLUMNS = ["symbol", "date", "close", "rsi14", "sma50", "sma200", "ema50", "adx14"]


def fetch_constituents() -> list[str]:
    """Liest die S&P-500-Liste von Wikipedia (Tabelle id="constituents").

    Bewusst mit BeautifulSoup statt pd.read_html: in der vendor-Umgebung ist
    weder lxml noch html5lib installiert."""
    log.info("Lade Konstituenten von %s", WIKI_URL)
    resp = requests.get(WIKI_URL, headers=WIKI_HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", id="constituents") or soup.find("table", class_="wikitable")
    if table is None:
        raise RuntimeError("Konstituenten-Tabelle auf Wikipedia nicht gefunden")

    tickers = set()
    for row in table.find_all("tr")[1:]:
        cell = row.find("td")
        if cell is None:
            continue
        sym = cell.get_text(strip=True)
        if sym:
            tickers.add(sym.replace(".", "-"))  # BRK.B -> BRK-B
    if len(tickers) < MIN_CONSTITUENTS:
        raise RuntimeError(f"Nur {len(tickers)} Ticker gefunden — Seitenaufbau geändert?")
    log.info("%d Ticker gefunden", len(tickers))
    return sorted(tickers)


def download_ohlc(tickers: list[str]) -> pd.DataFrame:
    """Ein Batch-Download (High/Low/Close) für alle Ticker."""
    log.info("Lade Kurse für %d Ticker (Zeitraum %s, ein Batch-Aufruf)", len(tickers), DOWNLOAD_PERIOD)
    kwargs = dict(period=DOWNLOAD_PERIOD, auto_adjust=True, progress=False, group_by="column")
    try:
        data = yf.download(tickers, threads=True, **kwargs)
    except RuntimeError as exc:
        if "can't start new thread" not in str(exc):
            raise
        log.warning("Keine Threads verfügbar (%s) — lade sequenziell.", exc)
        data = yf.download(tickers, threads=False, **kwargs)
    if data is None or data.empty:
        raise RuntimeError("yfinance lieferte keine Daten")
    return data


def rma(series: pd.Series, length: int) -> pd.Series:
    """Wilder-Glättung (Running Moving Average)."""
    return series.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def calc_rsi(close: pd.Series, length: int = RSI_LEN) -> pd.Series:
    delta = close.diff()
    gain = rma(delta.clip(lower=0), length)
    loss = rma(-delta.clip(upper=0), length)
    rs = gain / loss
    rsi = 100 - 100 / (1 + rs)
    # Keine Verluste im Fenster -> RSI 100
    return rsi.where(loss != 0, 100.0)


def calc_adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = ADX_LEN) -> pd.Series:
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=close.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=close.index)
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)

    atr = rma(tr, length)
    plus_di = 100 * rma(plus_dm, length) / atr
    minus_di = 100 * rma(minus_dm, length) / atr
    di_sum = plus_di + minus_di
    dx = 100 * (plus_di - minus_di).abs() / di_sum.replace(0, np.nan)
    return rma(dx, length)


def analyse_ticker(sym: str, high: pd.Series, low: pd.Series, close: pd.Series) -> dict | None:
    df = pd.DataFrame({"high": high, "low": low, "close": close}).dropna()
    if df.empty:
        return None
    c = df["close"]

    def last(s: pd.Series):
        v = s.iloc[-1]
        return None if pd.isna(v) else round(float(v), 2)

    return {
        "symbol": sym,
        "date": df.index[-1].strftime("%Y-%m-%d"),
        "close": last(c),
        "rsi14": last(calc_rsi(c)),
        "sma50": last(c.rolling(50).mean()),
        "sma200": last(c.rolling(200).mean()),
        "ema50": last(c.ewm(span=50, adjust=False, min_periods=50).mean()),
        "adx14": last(calc_adx(df["high"], df["low"], c)),
    }


def run(out_path: str, sort_col: str, quiet: bool) -> pd.DataFrame:
    tickers = fetch_constituents()
    data = download_ohlc(tickers)
    highs, lows, closes = data["High"], data["Low"], data["Close"]

    rows, dead = [], []
    for sym in tickers:
        if sym not in closes.columns:
            dead.append(sym)
            continue
        res = analyse_ticker(sym, highs[sym], lows[sym], closes[sym])
        if res is None:
            dead.append(sym)
        else:
            rows.append(res)
    if dead:
        log.warning("%d Ticker ohne Daten: %s", len(dead), ", ".join(dead))
    if not rows:
        raise RuntimeError("Für keinen Ticker konnten Werte berechnet werden")

    result = pd.DataFrame(rows, columns=COLUMNS)
    ascending = sort_col == "symbol"
    result = result.sort_values(sort_col, ascending=ascending, na_position="last").reset_index(drop=True)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    payload = {
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "index": "S&P 500",
        "count": len(result),
        "missing": dead,
        "stocks": json.loads(result.to_json(orient="records")),
    }
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    os.replace(tmp_path, out_path)
    log.info("%d Aktien geschrieben -> %s", len(result), out_path)

    if not quiet:
        with pd.option_context("display.max_rows", None, "display.width", 200):
            print(result.to_string(index=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Technischer Scan (RSI, SMA, EMA, ADX) aller S&P-500-Aktien")
    parser.add_argument("--out", default=DEFAULT_OUT, metavar="DATEI",
                        help="Ziel-JSON (Vorgabe: data.json neben dem Skript)")
    parser.add_argument("--sort", default="symbol", choices=COLUMNS,
                        help="Sortierspalte (symbol aufsteigend, alle anderen absteigend)")
    parser.add_argument("--quiet", action="store_true", help="Tabelle nicht auf der Konsole ausgeben")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        run(args.out, args.sort, args.quiet)
    except Exception as exc:
        log.error("Abbruch: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
