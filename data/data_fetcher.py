"""
Data fetcher for G3B.SI, its top holdings, and cross-asset macro proxies.
Caches OHLCV and earnings calendars under data/cache/ to survive Yahoo rate
limits and session restarts.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

G3B = "G3B.SI"

# Top-10 G3B holdings from the AMOVA STI factsheet (07 Aug 2026).
# Weights are relative to the fund and normalised to 1.0 when computing baskets.
TOP_HOLDINGS = {
    "D05.SI": 0.2923,  # DBS Group Holdings
    "O39.SI": 0.1848,  # OCBC Bank
    "U11.SI": 0.0987,  # UOB
    "Z74.SI": 0.0607,  # Singtel
    "S68.SI": 0.0377,  # Singapore Exchange
    "BN4.SI": 0.0359,  # Keppel
    "C38U.SI": 0.0294,  # CapitaLand Integrated Commercial Trust
    "S63.SI": 0.0292,  # ST Engineering
    "J36.SI": 0.0286,  # Jardine Matheson
    "C6L.SI": 0.0223,  # Singapore Airlines
}

BANKS = ["D05.SI", "O39.SI", "U11.SI"]
BANK_WEIGHTS = {k: v for k, v in TOP_HOLDINGS.items() if k in BANKS}

SECTORS = {
    "financials": ["D05.SI", "O39.SI", "U11.SI", "S68.SI"],
    "real_estate": ["C38U.SI"],
    "industrials": ["BN4.SI", "S63.SI", "J36.SI", "C6L.SI"],
    "telco": ["Z74.SI"],
}

MACRO = [
    "^STI",  # Straits Times Index
    "SPY",  # US large cap
    "QQQ",  # US tech / growth
    "DX-Y.NYB",  # DXY (US dollar index)
    "USDSGD=X",  # SGD FX
    "^VIX",  # Volatility index
    "HG=F",  # Copper
    "GC=F",  # Gold
    "CL=F",  # Crude oil
]

UNIVERSE = [G3B] + list(TOP_HOLDINGS.keys()) + MACRO

CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def _cache_path(ticker: str, period: str) -> Path:
    safe = ticker.replace("^", "_").replace("=", "_").replace("/", "_")
    return CACHE_DIR / f"{safe}_{period}.csv"


def _earnings_cache_path(ticker: str) -> Path:
    safe = ticker.replace("^", "_").replace("=", "_").replace("/", "_")
    return CACHE_DIR / f"{safe}_earnings.csv"


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def fetch_ohlcv(
    ticker: str,
    period: str = "5y",
    interval: str = "1d",
    auto_adjust: bool = True,
    max_retries: int = 3,
    use_cache: bool = True,
) -> Optional[pd.DataFrame]:
    """Fetch OHLCV with caching and retry for rate limits."""
    cache = _cache_path(ticker, period)
    if use_cache and cache.exists():
        try:
            df = pd.read_csv(cache, index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            logger.info("Loaded %s from cache: %d bars", ticker, len(df))
            return df
        except Exception:
            logger.warning("Cache read failed for %s", ticker)

    for attempt in range(max_retries):
        try:
            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                auto_adjust=auto_adjust,
                progress=False,
                threads=False,
            )
            df = _flatten(df)
            if df.empty:
                raise ValueError(f"Empty data for {ticker}")
            df = df.rename(columns=str.title)[
                ["Open", "High", "Low", "Close", "Volume"]
            ]
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df[~df.index.duplicated(keep="last")].sort_index()
            if use_cache:
                df.to_csv(cache)
            logger.info(
                "Fetched %s: %d bars (%s -> %s)",
                ticker,
                len(df),
                df.index.min().date(),
                df.index.max().date(),
            )
            return df
        except Exception as e:
            wait = 4 * (attempt + 1)
            logger.warning(
                "%s attempt %d failed: %s. Waiting %ds",
                ticker,
                attempt + 1,
                type(e).__name__,
                wait,
            )
            time.sleep(wait)
    logger.error("Failed to fetch %s after %d retries", ticker, max_retries)
    return None


def fetch_earnings_calendar(
    ticker: str, max_age_days: int = 7
) -> Optional[pd.DataFrame]:
    """Fetch and cache yfinance earnings_dates for a ticker, if available."""
    cache = _earnings_cache_path(ticker)
    if cache.exists():
        try:
            mtime = datetime.fromtimestamp(cache.stat().st_mtime)
            if (datetime.now() - mtime).days <= max_age_days:
                df = pd.read_csv(cache, index_col=0, parse_dates=True)
                if not df.empty:
                    return df
        except Exception:
            logger.warning("Earnings cache read failed for %s", ticker)

    try:
        tk = yf.Ticker(ticker)
        ed = tk.earnings_dates
        if ed is None or ed.empty:
            return None
        ed = ed.copy()
        if ed.index.tz is not None:
            ed.index = ed.index.tz_localize(None)
        # earnings_dates index is the earnings date with timestamp; normalise to date.
        ed.index = pd.to_datetime(ed.index).normalize()
        ed = ed[~ed.index.duplicated(keep="last")].sort_index()
        try:
            ed.to_csv(cache)
        except Exception:
            pass
        return ed
    except Exception as e:
        logger.warning("Could not fetch earnings for %s: %s", ticker, e)
    return None


def fetch_universe(
    tickers: Optional[List[str]] = None,
    period: str = "5y",
    required: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Fetch multiple tickers with polite delays and robust fallback."""
    tickers = tickers or UNIVERSE
    required = required or [G3B]
    out: Dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            df = fetch_ohlcv(t, period=period)
            if df is not None:
                out[t] = df
        except Exception as e:
            logger.error("Could not fetch %s: %s", t, e)
        time.sleep(0.6)
    missing = [t for t in required if t not in out]
    if missing:
        logger.warning("Missing required tickers: %s", missing)
    return out


def fetch_earnings_for_holdings(
    holdings: Optional[Dict[str, float]] = None,
) -> Dict[str, pd.DataFrame]:
    """Fetch earnings calendars for the configured top holdings."""
    holdings = holdings or TOP_HOLDINGS
    out: Dict[str, pd.DataFrame] = {}
    for t in holdings:
        try:
            ed = fetch_earnings_calendar(t)
            if ed is not None and not ed.empty:
                out[t] = ed
        except Exception as e:
            logger.warning("Earnings fetch failed for %s: %s", t, e)
        time.sleep(0.5)
    return out


def _weighted_return_from_closes(
    close: pd.DataFrame, weights: Dict[str, float]
) -> pd.Series:
    """Return a weighted 1-day return series from a close-price DataFrame."""
    cols = [c for c in close.columns if c in weights]
    if not cols:
        return pd.Series(index=close.index, dtype=float)
    total = sum(weights[c] for c in cols)
    w = {c: weights[c] / total for c in cols}
    weighted = sum(close[c] * w[c] for c in cols)
    return weighted.pct_change()


def make_weighted_return(
    data: Dict[str, pd.DataFrame],
    weights: Dict[str, float],
    index: pd.DatetimeIndex,
) -> pd.Series:
    """Build a weighted 1-day return series aligned to the supplied index."""
    close = pd.DataFrame({t: df["Close"] for t, df in data.items() if t in weights})
    close = close.reindex(index).ffill()
    return _weighted_return_from_closes(close, weights).reindex(index)


def make_bank_weighted_return(
    data: Dict[str, pd.DataFrame], index: Optional[pd.DatetimeIndex] = None
) -> pd.Series:
    """Weighted 1-day return of the top-3 bank holdings."""
    idx = index
    if idx is None:
        idx = (
            pd.DatetimeIndex(pd.concat([df["Close"] for df in data.values()]).index)
            .unique()
            .sort_values()
        )
    return make_weighted_return(data, BANK_WEIGHTS, idx)


def make_top10_weighted_return(
    data: Dict[str, pd.DataFrame], index: Optional[pd.DatetimeIndex] = None
) -> pd.Series:
    """Weighted 1-day return of the top-10 G3B holdings."""
    idx = index
    if idx is None:
        idx = (
            pd.DatetimeIndex(pd.concat([df["Close"] for df in data.values()]).index)
            .unique()
            .sort_values()
        )
    return make_weighted_return(data, TOP_HOLDINGS, idx)


def make_sector_returns(
    data: Dict[str, pd.DataFrame], index: Optional[pd.DatetimeIndex] = None
) -> pd.DataFrame:
    """Daily returns for Financials, Real Estate, Industrials, Telco baskets."""
    idx = index
    if idx is None:
        idx = (
            pd.DatetimeIndex(pd.concat([df["Close"] for df in data.values()]).index)
            .unique()
            .sort_values()
        )
    out = pd.DataFrame(index=idx)
    close = pd.DataFrame({t: df["Close"] for t, df in data.items()})
    close = close.reindex(idx).ffill()
    for sector, tickers in SECTORS.items():
        weights = {t: TOP_HOLDINGS.get(t, 0.0) for t in tickers}
        total = sum(weights.values())
        if total > 0:
            w = {t: v / total for t, v in weights.items()}
            weighted = sum(close[t] * w[t] for t in w if t in close.columns)
            out[f"{sector}_ret1"] = weighted.pct_change()
    return out


def make_proxy_g3b(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create a synthetic G3B proxy from the top-10 holdings."""
    bank_dfs = {t: df for t, df in data.items() if t in BANKS}
    if not bank_dfs:
        raise ValueError("No bank data for proxy")
    close = pd.DataFrame(
        {t: df["Close"] for t, df in data.items() if t in TOP_HOLDINGS}
    )
    close = close.dropna()
    total = sum(TOP_HOLDINGS.values())
    weights = {t: TOP_HOLDINGS[t] / total for t in close.columns if t in TOP_HOLDINGS}
    proxy_close = sum(close[t] * weights[t] for t in weights)

    proxy = pd.DataFrame(index=proxy_close.index)
    proxy["Close"] = proxy_close
    proxy["Open"] = proxy_close.shift(1).fillna(proxy_close)
    proxy["High"] = proxy[["Open", "Close"]].max(axis=1) * 1.002
    proxy["Low"] = proxy[["Open", "Close"]].min(axis=1) * 0.998
    proxy["Volume"] = sum(
        data[t]["Volume"].reindex(proxy.index).fillna(0) * weights[t]
        for t in weights
        if t in data
    )
    return proxy


def load_g3b_data(
    period: str = "5y",
    include_earnings: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Load G3B.SI (or proxy), all supporting market data, and earnings calendars."""
    logger.info("Loading G3B universe...")
    data = fetch_universe(period=period)
    if G3B in data:
        g3b = data[G3B]
    else:
        logger.warning("G3B.SI unavailable; building top-10 proxy")
        g3b = make_proxy_g3b(data)

    earnings = fetch_earnings_for_holdings() if include_earnings else {}
    return g3b, data, earnings


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    g3b, data, earnings = load_g3b_data()
    print("G3B bars:", len(g3b))
    print("Loaded tickers:", list(data.keys()))
    print("Earnings calendars:", list(earnings.keys()))
