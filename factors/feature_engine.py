"""
Multi-horizon feature engineering for the SGX G3B predictor.

The pipeline builds a common feature matrix from G3B price/volume, macro
proxies, top-10 holding baskets, sector baskets, and earnings calendars.  It
also constructs forward-return labels for horizons 1, 2, 3, 5, 10, 20 and 60
days.  All forward-looking quantities are placed in ``label_*`` columns and
removed at training time so the model cannot peek.
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import ta

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

HORIZONS = [1, 2, 3, 5, 10, 20, 60]

BUCKET_LABELS = ["Strong Down", "Weak Down", "Flat", "Weak Up", "Strong Up"]


def _bucketize(ret: pd.Series, thresholds: Optional[List[float]] = None) -> pd.Series:
    """Map signed returns to five ordered buckets."""
    if thresholds is None:
        # Adaptive thresholds using the in-sample standard deviation.
        std = float(ret.std()) if len(ret) > 0 else 0.01
        thresholds = [-0.75 * std, -0.15 * std, 0.15 * std, 0.75 * std]
    t = sorted(thresholds)
    out = pd.Series(index=ret.index, dtype="object")
    out[ret <= t[0]] = BUCKET_LABELS[0]
    out[(ret > t[0]) & (ret <= t[1])] = BUCKET_LABELS[1]
    out[(ret > t[1]) & (ret < t[2])] = BUCKET_LABELS[2]
    out[(ret >= t[2]) & (ret < t[3])] = BUCKET_LABELS[3]
    out[ret >= t[3]] = BUCKET_LABELS[4]
    return out


def _add_g3b_price_features(
    df: pd.DataFrame,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
) -> pd.DataFrame:
    """Technical features derived from G3B OHLCV."""
    for h in [1, 2, 3, 5, 10, 20, 60]:
        df[f"ret_{h}d"] = close.pct_change(h)
        df[f"logret_{h}d"] = np.log(close / close.shift(h))

    df["intraday_range"] = (high - low) / close
    df["body"] = (close - df.get("open", close)) / close
    df["gap"] = (df.get("open", close) - close.shift(1)) / close.shift(1)

    for window in [5, 10, 20, 50]:
        sma = close.rolling(window).mean()
        df[f"dist_sma{window}"] = close / sma - 1.0

    # RSI and MACD
    df["rsi_14"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    macd = ta.trend.MACD(close)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    # Bollinger %b
    bb = ta.volatility.BollingerBands(close)
    df["bb_pband"] = bb.bollinger_pband()
    df["bb_width"] = bb.bollinger_wband() / close

    # ATR and ADX
    atr = ta.volatility.AverageTrueRange(high, low, close, window=14)
    df["atr_14"] = atr.average_true_range() / close
    adx = ta.trend.ADXIndicator(high, low, close, window=14)
    df["adx_14"] = adx.adx()
    df["di_plus_14"] = adx.adx_pos()
    df["di_minus_14"] = adx.adx_neg()

    # Stochastic
    stoch = ta.momentum.StochasticOscillator(
        high, low, close, window=14, smooth_window=3
    )
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # Volume
    vol_sma20 = volume.rolling(20).mean()
    df["volume_ratio"] = volume / vol_sma20
    df["volume_sma5"] = volume.rolling(5).mean() / vol_sma20
    df["dollar_volume"] = close * volume

    # Price position relative to recent high/low
    high20 = high.rolling(20).max()
    low20 = low.rolling(20).min()
    df["dist_high20"] = close / high20 - 1.0
    df["dist_low20"] = close / low20 - 1.0
    high60 = high.rolling(60).max()
    low60 = low.rolling(60).min()
    df["dist_high60"] = close / high60 - 1.0
    df["dist_low60"] = close / low60 - 1.0

    # Realised volatility and skew
    ret1 = close.pct_change()
    df["realised_vol_20d"] = ret1.rolling(20).std() * np.sqrt(252)
    df["realised_vol_60d"] = ret1.rolling(60).std() * np.sqrt(252)
    df["ret_skew_20d"] = ret1.rolling(20).skew()
    df["ret_kurt_20d"] = ret1.rolling(20).kurt()

    return df


def _add_macro_features(
    df: pd.DataFrame, data: Dict[str, pd.DataFrame], index: pd.DatetimeIndex
) -> pd.DataFrame:
    """Macro and FX 1d/5d/20d returns plus levels."""
    close_map: Dict[str, pd.Series] = {}
    for ticker in [
        "^STI",
        "SPY",
        "QQQ",
        "DX-Y.NYB",
        "USDSGD=X",
        "^VIX",
        "HG=F",
        "GC=F",
        "CL=F",
    ]:
        if ticker in data:
            close_map[ticker] = data[ticker]["Close"].reindex(index).ffill()
        else:
            logger.warning("Missing macro ticker: %s", ticker)
            close_map[ticker] = pd.Series(index=index, dtype=float)

    for ticker, s in close_map.items():
        prefix = (
            ticker.replace("^", "").replace("=", "").replace("-", "_").replace(".", "_")
        )
        ret = s.pct_change()
        for h in [1, 5, 20]:
            df[f"{prefix}_ret{h}d"] = ret.rolling(h).sum() if h > 1 else ret
        # Momentum: close / sma - 1
        for w in [20, 60]:
            df[f"{prefix}_sma_dist{w}"] = s / s.rolling(w).mean() - 1.0
        if ticker == "^VIX":
            df[f"{prefix}_level"] = s

    # Cross-asset spreads
    df["g3b_vs_sti_1d"] = df["ret_1d"] - close_map["^STI"].pct_change()
    df["bank_vs_g3b_1d"] = df.get("bank_w", pd.Series(index=index)) - df["ret_1d"]
    df["top10_vs_g3b_1d"] = df.get("top10_w", pd.Series(index=index)) - df["ret_1d"]

    # Risk-on / risk-off composites
    df["vix_level"] = close_map["^VIX"]
    df["dxy_level"] = close_map["DX-Y.NYB"]
    df["usdsgd_level"] = close_map["USDSGD=X"]

    return df


def _add_basket_features(
    df: pd.DataFrame, data: Dict[str, pd.DataFrame], index: pd.DatetimeIndex
) -> pd.DataFrame:
    """Weighted top-10 and sector basket returns."""
    from data.data_fetcher import (
        TOP_HOLDINGS,
        make_weighted_return,
        make_sector_returns,
    )

    top10_ret = make_weighted_return(data, TOP_HOLDINGS, index)
    df["top10_w"] = top10_ret

    bank_ret = make_weighted_return(
        data, {t: TOP_HOLDINGS[t] for t in ["D05.SI", "O39.SI", "U11.SI"]}, index
    )
    df["bank_w"] = bank_ret

    sector_rets = make_sector_returns(data, index)
    for col in sector_rets.columns:
        df[col] = sector_rets[col]

    # Add 5d/20d sector momentum
    for col in [c for c in sector_rets.columns if c.endswith("_ret1")]:
        for h in [5, 20]:
            df[f"{col.replace('_ret1', '')}_ret{h}d"] = (
                sector_rets[col].rolling(h).sum()
            )

    # Dispersion: top10 vs individual holdings, financials vs reits vs industrials
    if "financials_ret1" in df and "real_estate_ret1" in df:
        df["fin_vs_reit_1d"] = df["financials_ret1"] - df["real_estate_ret1"]
    if "financials_ret1" in df and "industrials_ret1" in df:
        df["fin_vs_indust_1d"] = df["financials_ret1"] - df["industrials_ret1"]
    if "telco_ret1" in df and "industrials_ret1" in df:
        df["telco_vs_indust_1d"] = df["telco_ret1"] - df["industrials_ret1"]

    return df


def _add_earnings_features(
    df: pd.DataFrame,
    earnings: Dict[str, pd.DataFrame],
    index: pd.DatetimeIndex,
    holdings: Dict[str, float],
) -> pd.DataFrame:
    """Calendar features based on upcoming/past earnings for top holdings."""
    if not earnings:
        df["earnings_next_days_w"] = np.nan
        df["earnings_count_5d"] = 0
        df["earnings_count_10d"] = 0
        df["earnings_count_20d"] = 0
        df["earnings_last_surprise_w"] = np.nan
        df["earnings_last_days_w"] = np.nan
        df["earnings_max_weight_5d"] = 0.0
        return df

    next_days = []
    count5 = np.zeros(len(index))
    count10 = np.zeros(len(index))
    count20 = np.zeros(len(index))
    last_surprise = []
    last_days = []
    max_weight_5d = np.zeros(len(index))

    for i, t in enumerate(index):
        weights_total = 0.0
        weighted_next = 0.0
        weighted_surprise = 0.0
        weighted_days = 0.0
        local_max_weight = 0.0
        for ticker, weight in holdings.items():
            if ticker not in earnings:
                continue
            ed = earnings[ticker]
            future = ed[ed.index > t]
            if not future.empty:
                next_dt = future.index[0]
                days_to = (next_dt - t).days
                weighted_next += weight * days_to
                weights_total += weight
                if days_to <= 5:
                    count5[i] += weight
                    local_max_weight = max(local_max_weight, weight)
                if days_to <= 10:
                    count10[i] += weight
                if days_to <= 20:
                    count20[i] += weight
            past = ed[ed.index <= t]
            if not past.empty:
                last = past.iloc[-1]
                surprise = last.get("Surprise(%)")
                if pd.notna(surprise):
                    weighted_surprise += weight * float(surprise)
                    weighted_days += weight * (t - past.index[-1]).days
        if weights_total > 0:
            next_days.append(weighted_next / weights_total)
            last_surprise.append(weighted_surprise / weights_total)
            last_days.append(weighted_days / weights_total)
        else:
            next_days.append(np.nan)
            last_surprise.append(np.nan)
            last_days.append(np.nan)
        max_weight_5d[i] = local_max_weight

    df["earnings_next_days_w"] = pd.Series(next_days, index=index)
    df["earnings_count_5d"] = count5
    df["earnings_count_10d"] = count10
    df["earnings_count_20d"] = count20
    df["earnings_last_surprise_w"] = pd.Series(last_surprise, index=index)
    df["earnings_last_days_w"] = pd.Series(last_days, index=index)
    df["earnings_max_weight_5d"] = max_weight_5d
    df["earnings_flag_5d"] = (max_weight_5d > 0).astype(int)
    return df


def _add_calendar_features(df: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Calendar/time features."""
    df["dayofweek"] = index.dayofweek
    df["month"] = index.month
    df["quarter"] = index.quarter
    df["is_month_end"] = index.is_month_end.astype(int)
    df["is_quarter_end"] = index.is_quarter_end.astype(int)
    return df


def build_features(
    g3b: pd.DataFrame,
    data: Dict[str, pd.DataFrame],
    earnings: Optional[Dict[str, pd.DataFrame]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    label_horizons: Optional[List[int]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build the full feature matrix and return (features_df, feature_column_names).

    The returned DataFrame contains ``label_*`` columns (forward returns and
    buckets) which should be stripped before training/inference.
    """
    index = g3b.index
    close = g3b["Close"]
    high = g3b["High"]
    low = g3b["Low"]
    volume = g3b["Volume"]
    open_ = g3b["Open"]

    df = pd.DataFrame(index=index)
    df["open"] = open_
    df["high"] = high
    df["low"] = low
    df["close"] = close
    df["volume"] = volume

    df = _add_g3b_price_features(df, close, high, low, volume)
    df = _add_basket_features(df, data, index)
    df = _add_macro_features(df, data, index)

    from data.data_fetcher import TOP_HOLDINGS

    df = _add_earnings_features(df, earnings or {}, index, TOP_HOLDINGS)

    df = _add_calendar_features(df, index)

    # Forward labels
    horizons = label_horizons or HORIZONS
    for h in horizons:
        df[f"label_ret_{h}d"] = close.shift(-h) / close - 1.0
        df[f"label_dir_{h}d"] = (df[f"label_ret_{h}d"] > 0).astype(int)

    # Bucket labels for next-day only (bucketised adaptively)
    ret1 = df["label_ret_1d"]
    df["label_bucket_1d"] = _bucketize(ret1)

    # Mask requested date range
    if start is not None:
        df = df[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df[df.index <= pd.Timestamp(end)]

    # Feature columns = everything except label_ and raw open/high/low/close/volume
    exclude = [c for c in df.columns if c.startswith("label_")] + [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    feature_cols = [c for c in df.columns if c not in exclude]

    # Consolidate the DataFrame to remove fragmentation caused by repeated insertions.
    return df.copy(), feature_cols


def impute_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Forward-fill then median-fill remaining NaNs in feature columns."""
    df = df.copy()
    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()
            median = df[col].median()
            if pd.isna(median):
                median = 0.0
            df[col] = df[col].fillna(median)
    return df
