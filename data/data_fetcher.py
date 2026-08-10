"""
Data fetcher for G3B.SI, its dominant bank holdings, and cross-asset macro proxies.
Caches CSVs under data/cache/ to survive Yahoo rate limits and session restarts.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Core universe based on actual G3B concentration (factsheet 10 Aug 2026)
G3B = "G3B.SI"
BANKS = ["D05.SI", "O39.SI", "U11.SI"]  # DBS 29.23%, OCBC 18.48%, UOB 9.87%
BANK_WEIGHTS = {"D05.SI": 0.2923, "O39.SI": 0.1848, "U11.SI": 0.0987}
UNIVERSE = [G3B] + BANKS + [
    "^STI",    # Straits Times Index proxy
    "SPY",     # US large cap
    "QQQ",     # US tech / growth
    "DX-Y.NYB", # DXY (US Dollar index)
    "USDSGD=X", # SGD FX
    "^VIX",    # Volatility index
    "HG=F",    # Copper
    "GC=F",    # Gold
    "CL=F",    # Crude oil
]

CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def _cache_path(ticker: str, period: str) -> Path:
    safe = ticker.replace("^", "_").replace("=", "_").replace("/", "_")
    return CACHE_DIR / f"{safe}_{period}.csv"


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
        time.sleep(1.0)
    missing = [t for t in required if t not in out]
    if missing:
        logger.warning("Missing required tickers: %s", missing)
    return out


def make_bank_weighted_return(bank_dfs: Dict[str, pd.DataFrame]) -> pd.Series:
    """Weighted 1-day return of the top-3 bank holdings."""
    total = sum(BANK_WEIGHTS.values())
    weights = {k: v / total for k, v in BANK_WEIGHTS.items()}
    close = pd.DataFrame({t: df["Close"] for t, df in bank_dfs.items() if t in weights})
    close = close.dropna()
    weighted = sum(close[t] * w for t, w in weights.items())
    return weighted.pct_change()


def make_proxy_g3b(bank_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Create a synthetic G3B proxy from the top 3 banks using factsheet weights.
    Used only as a fallback when G3B.SI itself is unavailable.
    """
    if not bank_dfs:
        raise ValueError("No bank data for proxy")
    total = sum(BANK_WEIGHTS.values())
    weights = {k: v / total for k, v in BANK_WEIGHTS.items()}
    closes = pd.DataFrame({t: df["Close"] for t, df in bank_dfs.items() if t in weights})
    closes = closes.dropna()
    proxy_close = sum(closes[t] * w for t, w in weights.items())

    proxy = pd.DataFrame(index=proxy_close.index)
    proxy["Close"] = proxy_close
    proxy["Open"] = proxy_close.shift(1).fillna(proxy_close)
    proxy["High"] = proxy[["Open", "Close"]].max(axis=1) * 1.002
    proxy["Low"] = proxy[["Open", "Close"]].min(axis=1) * 0.998
    proxy["Volume"] = sum(
        bank_dfs[t]["Volume"].reindex(proxy.index).fillna(0) * w
        for t, w in weights.items()
    )
    return proxy


def load_g3b_data(period: str = "5y") -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Load G3B.SI (or bank proxy fallback) and all supporting macro series.
    Returns (g3b_df, macro_dict).
    """
    logger.info("Loading G3B universe...")
    data = fetch_universe(period=period)
    if G3B in data:
        g3b = data[G3B]
    else:
        logger.warning("G3B.SI unavailable; building bank proxy")
        bank_dfs = {t: df for t, df in data.items() if t in BANKS}
        g3b = make_proxy_g3b(bank_dfs)
    return g3b, data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    g3b, data = load_g3b_data()
    print("G3B bars:", len(g3b))
    print("Loaded tickers:", list(data.keys()))
