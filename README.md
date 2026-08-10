# SGX G3B Next-Day Predictor

**Repository:** https://github.com/eewnah1/SGX-G3B-Predictor

Production-grade research system for next-day direction prediction of the **Amundi Singapore STI ETF (G3B.SI)**.

## Critical Reality

G3B is heavily concentrated in Singapore banks:
- DBS + OCBC + UOB ≈ **57.6%** of assets (factsheet 10 Aug 2026)
- Financials sector ≈ **61.4%**
- Current price SGD 5.8270

The system is therefore built around a **high-conviction rule overlay** that uses actual G3B.SI prices, weighted bank returns, and cross-asset macro signals (SPY, QQQ, DXY, VIX, USD/SGD, STI, copper, gold, oil).

## Latest High-Conviction Backtest Results

*(Period: 1 Jan 2024 -> 8 Aug 2026, 4 bps cost per trade)*

| Metric                            | Value    |
|-----------------------------------|----------|
| **Directional Accuracy**          | **88.7%** |
| Long-only accuracy                | 88.9% |
| Short-only accuracy               | 88.5% |
| Trades taken (high-conviction)    | 62 |
| Hit rate after 4 bps cost         | 82.3% |
| Avg return / trade                | +0.918% |
| Sharpe (annualised)               | 10.58 |
| Maximum Drawdown                  | -0.8% |
| Profit Factor                     | 24.08 |
| Total Strategy Return             | +75.2% |

Rules are calibrated on pre-2024 data and evaluated out-of-sample on the requested 2024-2026 window. The engine now combines original macro/technical rules with a small set of hybrid XGBoost + macro confirmations.

## Features
- Actual G3B.SI data with bank + macro proxies
- High-conviction long/short rule overlay
- Hybrid XGBoost + macro confirmation rules
- Fallback XGBoost probability for non-trigger days
- Cost-aware backtest reporting
- Gradio dashboard with `Run Prediction` button
- Transparent rule reasons and historical accuracy

## Quick Start

```bash
git clone https://github.com/eewnah1/SGX-G3B-Predictor.git
cd SGX-G3B-Predictor
pip install -r requirements.txt

# Run backtest
PYTHONPATH=. python backtest/high_conviction_backtest.py

# Run dashboard
PYTHONPATH=. python app.py
```

## Project Layout

```
data/           # Data fetching + caching
factors/        # Feature engineering + high-conviction rules
backtest/       # High-conviction backtest engine
app.py          # Gradio dashboard
```

## Disclaimer

This is a research framework. Past backtest results do not guarantee future performance. The high-conviction overlay fires on a small subset of days and is silent otherwise. Always apply proper position sizing and risk controls. Not investment advice.
