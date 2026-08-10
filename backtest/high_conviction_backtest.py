"""
High-conviction backtest for G3B next-day direction.

Evaluates the empirical rule overlay over a target date range and reports
win rate, average return, max drawdown, profit factor, and number of trades.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from data.data_fetcher import load_g3b_data, make_bank_weighted_return
from factors.high_conviction import backtest_high_conviction, build_features, evaluate_high_conviction

logger = logging.getLogger(__name__)


@dataclass
class HighConvictionBacktestResult:
    start: str
    end: str
    n_trades: int
    directional_accuracy: float
    hit_rate_after_cost: float
    avg_return_per_trade: float
    sharpe: float
    profit_factor: float
    max_drawdown: float
    total_return: float
    long_accuracy: float | None
    short_accuracy: float | None
    equity_curve: pd.Series
    trades: List[dict]


def run_backtest(
    start: str = "2024-01-01",
    end: str = "2026-08-08",
    cost_bps: float = 4.0,
) -> HighConvictionBacktestResult:
    """Fetch data and run the high-conviction rule backtest."""
    logger.info("Running high-conviction backtest %s -> %s", start, end)
    g3b, data = load_g3b_data(period="5y")
    if g3b is None or g3b.empty:
        raise RuntimeError("Could not load G3B data")

    bank_dfs = {t: df for t, df in data.items() if t in ["D05.SI", "O39.SI", "U11.SI"]}
    bank_weight = make_bank_weighted_return(bank_dfs)

    # Also need macro dict with proper ticker keys for build_features
    macro = {t: df for t, df in data.items() if t not in ["G3B.SI", "D05.SI", "O39.SI", "U11.SI"]}

    result = backtest_high_conviction(g3b, macro, bank_weight, start, end, cost_bps=cost_bps)
    return HighConvictionBacktestResult(
        start=result["start"],
        end=result["end"],
        n_trades=result["n_trades"],
        directional_accuracy=result["directional_accuracy"],
        hit_rate_after_cost=result["hit_rate_after_cost"],
        avg_return_per_trade=result["avg_return_per_trade"],
        sharpe=result["sharpe"],
        profit_factor=result["profit_factor"],
        max_drawdown=result["max_drawdown"],
        total_return=result["total_return"],
        long_accuracy=result["long_accuracy"],
        short_accuracy=result["short_accuracy"],
        equity_curve=result["equity_curve"],
        trades=result["trades"],
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    res = run_backtest()
    print("\n" + "=" * 60)
    print("  SGX G3B HIGH-CONVICTION BACKTEST")
    print(f"  {res.start} -> {res.end}")
    print("=" * 60)
    print(f"  Directional Accuracy        : {res.directional_accuracy:6.1%}")
    print(f"  Number of high-conviction trades: {res.n_trades}")
    print(f"  Hit rate after 4 bps cost   : {res.hit_rate_after_cost:6.1%}")
    print(f"  Avg return / trade          : {res.avg_return_per_trade:+.3%}")
    print(f"  Sharpe (annualised)         : {res.sharpe:6.2f}")
    print(f"  Profit Factor               : {res.profit_factor:6.2f}")
    print(f"  Maximum Drawdown            : {res.max_drawdown:6.1%}")
    print(f"  Total Return                : {res.total_return:+.2%}")
    if res.long_accuracy:
        print(f"  Long-only accuracy          : {res.long_accuracy:6.1%}")
    if res.short_accuracy:
        print(f"  Short-only accuracy         : {res.short_accuracy:6.1%}")
    print("=" * 60)
