# SGX G3B Next-Day Predictor

**Repository:** https://github.com/eewnah1/SGX-G3B-Predictor

Production-grade research system for next-day direction prediction of the **Amova Singapore STI ETF (G3B)**.

## Critical Reality
G3B is **not** a balanced index fund in practice:
- DBS + OCBC + UOB ≈ **57.6%** of assets (as of 07 Aug 2026)
- Financials sector ≈ **61.4%**
- Currently in a strong uptrend / near 52-week highs

The system is therefore built around bank concentration, regime-aware factors, and realistic expectations (52–58% directional accuracy range).

## Latest Walk-Forward Backtest Results
*(Weighted DBS+OCBC+UOB proxy, 2-year expanding window, 4 bps costs, high-conviction filter 0.55)*

| Metric                        | Value    |
|-------------------------------|----------|
| Directional Accuracy          | **52.2%** |
| Precision UP / DOWN           | 57.1% / 46.5% |
| Trades (high-conviction)      | 265      |
| Hit rate after costs          | 49.8%    |
| Avg return / trade            | +0.019%  |
| Sharpe (annualised)           | 0.36     |
| Max Drawdown                  | –12.3%   |
| Profit Factor                 | 1.06     |
| Total Strategy Return         | +4.2%    |

These numbers are intentionally realistic. Claims of >80% accuracy on this asset are not credible.

## Features
- Regime-aware factor scoring (momentum, mean-reversion, trend, vol, volume, candle + **bank relative strength & bank momentum**)
- XGBoost walk-forward validation
- High-conviction filtering
- Cost-aware strategy simulation
- Multi-asset ready (G3B + top-3 banks)
- Extensible interfaces for real-time websockets and live broker execution (paper mode)

## Quick Start
```bash
git clone https://github.com/eewnah1/SGX-G3B-Predictor.git
cd SGX-G3B-Predictor
pip install -r requirements.txt

# Run walk-forward backtest
PYTHONPATH=. python -m backtest.walkforward
```

## Project Layout
```
data/           # Data fetching + proxy construction
factors/        # Regime-aware + bank-concentration factors
backtest/       # Walk-forward engine + metrics
models/         # Classical + deep learning placeholders
portfolio/      # Multi-asset logic
execution/      # Broker interfaces (paper)
```

## Disclaimer
This is a research framework. Past backtest results do not guarantee future performance. Always apply proper position sizing and risk controls. Not investment advice.
