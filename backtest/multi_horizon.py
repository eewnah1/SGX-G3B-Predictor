"""
Multi-horizon OOS backtest for the G3B predictor.

Usage:
    PYTHONPATH=. python backtest/multi_horizon.py
"""

from __future__ import annotations

import logging
from typing import Any, Dict


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _fmt_pct(x: float) -> str:
    return f"{x:>6.1%}"


def _fmt_num(x: float) -> str:
    return f"{x:>7.3f}"


def print_backtest_report(metrics: Dict[str, Any]) -> None:
    print("\n" + "=" * 70)
    print(f"  SGX G3B Multi-Horizon Backtest {metrics['start']} -> {metrics['end']}")
    print("=" * 70)
    header = (
        f"{'H':>3} | {'N':>4} | {'DirAcc':>7} | "
        f"{'HC N':>5} | {'HCAcc':>6} | {'HC AvgR':>8} | {'All AvgR':>8}"
    )
    print(header)
    print("-" * 70)
    for h, m in metrics["horizons"].items():
        da = _fmt_pct(m["directional_accuracy"])
        hc = _fmt_pct(m["high_conviction_accuracy"])
        hr = _fmt_pct(m["high_conviction_avg_return"])
        ar = _fmt_pct(m["avg_return_per_trade"])
        hc_n = m["high_conviction_n"]
        print(
            f"{h:>3} | {m['n']:>4} | {da} | {hc_n:>5} | {hc} | {hr} | {ar}"
        )
    print("=" * 70)


def run_multi_horizon_backtest(
    start: str = "2024-01-01",
    end: str = "2026-08-08",
    cost_bps: float = 4.0,
) -> Dict[str, Any]:
    from factors.predictor import G3BPredictor

    logger.info("Running multi-horizon backtest %s -> %s", start, end)
    predictor = G3BPredictor()
    predictor.load_data()
    predictor.build_features()
    predictor.fit()
    return predictor.backtest(start=start, end=end, cost_bps=cost_bps)


if __name__ == "__main__":
    metrics = run_multi_horizon_backtest()
    print_backtest_report(metrics)
