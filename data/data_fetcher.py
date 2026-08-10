"""
Data fetcher for G3B and its dominant holdings (DBS, OCBC, UOB).
Handles rate limits gracefully and supports both single ticker and multi-asset pulls.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Core universe based on actual G3B concentration (screenshot 10 Aug 2026)
G3B = "G3B.SI"
BANKS = ["D05.SI", "O39.SI", "U11.SI"]  # DBS, OCBC, UOB ≈ 57.6%
UNIVERSE = [G3B] + BANKS


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def fetch_ohlcv(
    ticker: str = G3B,
    period: str = "2y",
    interval: str = "1d",
    auto_adjust: bool = True,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Fetch OHLCV with simple retry for rate limits."""
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
            df = df.rename(columns=str.title)[["Open", "High", "Low", "Close", "Volume"]]
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df[~df.index.duplicated(keep="last")].sort_index()
            logger.info("Fetched %s: %d bars (%s → %s)", ticker, len(df), df.index.min().date(), df.index.max().date())
            return df
        except Exception as e:
            wait = 5 * (attempt + 1)
            logger.warning("%s attempt %d failed: %s. Waiting %ds", ticker, attempt + 1, type(e).__name__, wait)
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch {ticker} after {max_retries} retries")


def fetch_universe(
    tickers: Optional[List[str]] = None,
    period: str = "2y",
) -> Dict[str, pd.DataFrame]:
    """Fetch multiple tickers with polite delays."""
    tickers = tickers or UNIVERSE
    out: Dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            out[t] = fetch_ohlcv(t, period=period)
            time.sleep(1.5)
        except Exception as e:
            logger.error("Could not fetch %s: %s", t, e)
    return out


def make_proxy_g3b(bank_dfs: Dict[str, pd.DataFrame], weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """
    Create a synthetic G3B proxy from the top 3 banks using approximate weights
    from the 07 Aug 2026 factsheet (DBS 29.23%, OCBC 18.48%, UOB 9.87%).
    Useful when G3B.SI itself is rate-limited.
    """
    if weights is None:
        weights = {"D05.SI": 0.2923, "O39.SI": 0.1848, "U11.SI": 0.0987}
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}

    closes = pd.DataFrame({t: df["Close"] for t, df in bank_dfs.items() if t in weights})
    closes = closes.dropna()
    proxy_close = sum(closes[t] * w for t, w in weights.items())

    proxy = pd.DataFrame(index=proxy_close.index)
    proxy["Close"] = proxy_close
    proxy["Open"] = proxy_close.shift(1).fillna(proxy_close)
    proxy["High"] = proxy[["Open", "Close"]].max(axis=1) * 1.002
    proxy["Low"] = proxy[["Open", "Close"]].min(axis=1) * 0.998
    proxy["Volume"] = sum(bank_dfs[t]["Volume"].reindex(proxy.index).fillna(0) * w for t, w in weights.items())
    return proxy


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data = fetch_universe(period="1y")
    print({k: len(v) for k, v in data.items()})
