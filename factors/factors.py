"""
Regime-aware factor scoring engine optimised for G3B's real structure.
G3B is ~58% three Singapore banks → bank relative strength and concentration
factors are first-class citizens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, EMAIndicator, MACD, SMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator


@dataclass
class FactorScore:
    name: str
    score: float          # ~ -2.0 … +2.0
    raw_value: float
    description: str
    weight: float = 1.0


def _clip(x: float, lo: float = -2.0, hi: float = 2.0) -> float:
    return float(np.clip(x, lo, hi))


def score_momentum(df: pd.DataFrame) -> FactorScore:
    ret5 = df["Close"].pct_change(5).iloc[-1]
    ret10 = df["Close"].pct_change(10).iloc[-1]
    rsi = RSIIndicator(df["Close"], 14).rsi().iloc[-1]
    mom = 0.0
    if not np.isnan(ret5):
        mom += np.tanh(ret5 * 40) * 1.2
    if not np.isnan(ret10):
        mom += np.tanh(ret10 * 25) * 0.8
    if not np.isnan(rsi):
        if rsi > 72:
            mom -= 0.7
        elif rsi < 28:
            mom += 0.5
        else:
            mom += (rsi - 50) / 50 * 0.35
    return FactorScore("momentum", _clip(mom), ret5 if not np.isnan(ret5) else 0.0,
                       f"5d={ret5:.2%} RSI={rsi:.1f}" if not np.isnan(rsi) else "momentum", 1.3)


def score_mean_reversion(df: pd.DataFrame) -> FactorScore:
    close = df["Close"].iloc[-1]
    sma20 = SMAIndicator(df["Close"], 20).sma_indicator().iloc[-1]
    sma50 = SMAIndicator(df["Close"], 50).sma_indicator().iloc[-1]
    pct_b = BollingerBands(df["Close"], 20, 2).bollinger_pband().iloc[-1]
    dist20 = (close - sma20) / sma20 if sma20 else 0.0
    dist50 = (close - sma50) / sma50 if sma50 else 0.0
    mr = -np.tanh(dist20 * 30) * 1.0 - np.tanh(dist50 * 20) * 0.6
    if not np.isnan(pct_b):
        mr += (0.5 - pct_b) * 2.0
    return FactorScore("mean_reversion", _clip(mr), pct_b if not np.isnan(pct_b) else dist20,
                       f"%B={pct_b:.2f} dist20={dist20:.2%}" if not np.isnan(pct_b) else "mr", 1.0)


def score_trend(df: pd.DataFrame) -> FactorScore:
    hist = MACD(df["Close"]).macd_diff().iloc[-1]
    ema12 = EMAIndicator(df["Close"], 12).ema_indicator().iloc[-1]
    ema26 = EMAIndicator(df["Close"], 26).ema_indicator().iloc[-1]
    adx = ADXIndicator(df["High"], df["Low"], df["Close"], 14).adx().iloc[-1]
    trend = 0.0
    if not np.isnan(hist):
        trend += np.tanh(hist / (df["Close"].iloc[-1] * 0.002 + 1e-9)) * 1.3
    if not np.isnan(ema12) and not np.isnan(ema26):
        trend += 0.5 if ema12 > ema26 else -0.5
    if not np.isnan(adx):
        trend *= (0.7 + 0.3 * min(adx / 40.0, 1.5))
    return FactorScore("trend", _clip(trend), hist if not np.isnan(hist) else 0.0,
                       f"ADX={adx:.1f}" if not np.isnan(adx) else "trend", 1.2)


def score_volatility_regime(df: pd.DataFrame) -> FactorScore:
    atr = AverageTrueRange(df["High"], df["Low"], df["Close"], 14).average_true_range()
    atr_pct_series = atr / df["Close"]
    atr_pct = float(atr_pct_series.iloc[-1])
    atr_ma = float(atr_pct_series.rolling(50).mean().iloc[-1]) if len(atr_pct_series) > 50 else atr_pct
    rel = atr_pct / atr_ma if atr_ma and not np.isnan(atr_ma) and atr_ma > 0 else 1.0
    score = 0.4 if rel < 0.75 else (-0.8 if rel > 1.4 else 0.0)
    ret = df["Close"].pct_change()
    vol5 = ret.rolling(5).std().iloc[-1]
    vol20 = ret.rolling(20).std().iloc[-1]
    if not np.isnan(vol5) and not np.isnan(vol20) and vol20 > 0 and vol5 < vol20 * 0.7:
        score += 0.3
    return FactorScore("volatility_regime", _clip(score), float(rel), f"relATR={rel:.2f}", 0.7)


def score_volume(df: pd.DataFrame) -> FactorScore:
    vol_sma = df["Volume"].rolling(20).mean().iloc[-1]
    rel_vol = df["Volume"].iloc[-1] / vol_sma if vol_sma else 1.0
    obv_chg = OnBalanceVolumeIndicator(df["Close"], df["Volume"]).on_balance_volume().pct_change(5).iloc[-1]
    score = 0.0
    if rel_vol > 1.4:
        ret1 = df["Close"].pct_change(1).iloc[-1]
        if not np.isnan(ret1):
            score += np.sign(ret1) * 0.6
    elif rel_vol < 0.6:
        score -= 0.2
    if not np.isnan(obv_chg):
        score += np.tanh(obv_chg * 5) * 0.8
    return FactorScore("volume", _clip(score), rel_vol, f"relVol={rel_vol:.2f}", 0.9)


def score_candle(df: pd.DataFrame) -> FactorScore:
    o, h, l, c = df["Open"].iloc[-1], df["High"].iloc[-1], df["Low"].iloc[-1], df["Close"].iloc[-1]
    rng = max(h - l, 1e-8)
    body_pct = (c - o) / rng
    upper = (h - max(o, c)) / rng
    lower = (min(o, c) - l) / rng
    score = 0.0
    if body_pct > 0.6 and upper < 0.2:
        score += 0.7
    elif body_pct < -0.6 and lower < 0.2:
        score -= 0.7
    if lower > 0.5 and body_pct > -0.2:
        score += 0.4
    if upper > 0.5 and body_pct < 0.2:
        score -= 0.4
    return FactorScore("candle", _clip(score), body_pct, f"body%={body_pct:.2f}", 0.6)


def score_bank_relative_strength(
    g3b: pd.DataFrame,
    banks: Dict[str, pd.DataFrame],
) -> FactorScore:
    if not banks:
        return FactorScore("bank_relative_strength", 0.0, 0.0, "no bank data", 1.4)
    g_ret = g3b["Close"].pct_change(5).iloc[-1]
    bank_rets = []
    for t, df in banks.items():
        if len(df) > 5:
            bank_rets.append(df["Close"].pct_change(5).iloc[-1])
    if not bank_rets or np.isnan(g_ret):
        return FactorScore("bank_relative_strength", 0.0, 0.0, "insufficient data", 1.4)
    avg_bank = np.nanmean(bank_rets)
    spread = avg_bank - g_ret
    score = _clip(np.tanh(spread * 50) * 1.5)
    return FactorScore(
        "bank_relative_strength",
        score,
        spread,
        f"banks5d={avg_bank:.2%} g3b5d={g_ret:.2%}",
        1.5,
    )


def score_bank_momentum(banks: Dict[str, pd.DataFrame]) -> FactorScore:
    moms = []
    for df in banks.values():
        if len(df) > 10:
            r = df["Close"].pct_change(5).iloc[-1]
            if not np.isnan(r):
                moms.append(r)
    if not moms:
        return FactorScore("bank_momentum", 0.0, 0.0, "no data", 1.3)
    avg = float(np.mean(moms))
    score = _clip(np.tanh(avg * 45) * 1.4)
    return FactorScore("bank_momentum", score, avg, f"avg5d={avg:.2%}", 1.3)


def detect_regime(df: pd.DataFrame) -> str:
    adx = ADXIndicator(df["High"], df["Low"], df["Close"], 14).adx().iloc[-1]
    ret20 = df["Close"].pct_change(20).iloc[-1]
    if not np.isnan(adx) and adx > 28 and not np.isnan(ret20) and ret20 > 0.04:
        return "strong_uptrend"
    if not np.isnan(adx) and adx > 28 and not np.isnan(ret20) and ret20 < -0.04:
        return "strong_downtrend"
    if not np.isnan(adx) and adx < 18:
        return "range"
    return "neutral"


def compute_all_factors(
    g3b: pd.DataFrame,
    banks: Optional[Dict[str, pd.DataFrame]] = None,
) -> List[FactorScore]:
    if len(g3b) < 60:
        raise ValueError("Need ≥60 bars")
    banks = banks or {}
    factors = [
        score_momentum(g3b),
        score_mean_reversion(g3b),
        score_trend(g3b),
        score_volatility_regime(g3b),
        score_volume(g3b),
        score_candle(g3b),
        score_bank_relative_strength(g3b, banks),
        score_bank_momentum(banks),
    ]
    regime = detect_regime(g3b)
    for f in factors:
        if regime == "strong_uptrend":
            if f.name in ("momentum", "trend", "bank_momentum", "bank_relative_strength"):
                f.weight *= 1.35
            if f.name == "mean_reversion":
                f.weight *= 0.6
        elif regime == "strong_downtrend":
            if f.name in ("momentum", "trend"):
                f.weight *= 1.2
            if f.name == "mean_reversion":
                f.weight *= 0.7
        elif regime == "range":
            if f.name == "mean_reversion":
                f.weight *= 1.4
            if f.name in ("momentum", "trend"):
                f.weight *= 0.7
    return factors


def composite_score(factors: List[FactorScore], method: str = "weighted") -> float:
    if method == "equal":
        return float(np.mean([f.score for f in factors]))
    if method == "majority":
        signs = [np.sign(f.score) for f in factors if abs(f.score) > 0.15]
        return float(np.mean(signs)) if signs else 0.0
    total_w = sum(f.weight for f in factors) or 1.0
    return float(sum(f.score * f.weight for f in factors) / total_w)


def factors_to_dict(factors: List[FactorScore]) -> Dict[str, float]:
    d = {f"factor_{f.name}": f.score for f in factors}
    d["factor_composite"] = composite_score(factors)
    return d


if __name__ == "__main__":
    from data.data_fetcher import fetch_universe, make_proxy_g3b
    data = fetch_universe(["D05.SI", "O39.SI", "U11.SI"], period="1y")
    proxy = make_proxy_g3b(data)
    facs = compute_all_factors(proxy, data)
    print("Regime:", detect_regime(proxy))
    for f in facs:
        print(f"{f.name:25s} {f.score:+.2f}  w={f.weight:.2f}  | {f.description}")
    print("Composite:", round(composite_score(facs), 3))
