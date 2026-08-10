"""
Walk-forward backtest for G3B (bank-weighted proxy).
Produces realistic metrics with costs and high-conviction filtering.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score
from xgboost import XGBClassifier

from factors.factors import compute_all_factors, factors_to_dict, detect_regime
from data.data_fetcher import fetch_universe, make_proxy_g3b

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    accuracy: float
    precision_up: float
    precision_down: float
    n_trades: int
    hit_rate_after_cost: float
    avg_return_per_trade: float
    sharpe: float
    max_drawdown: float
    profit_factor: float
    total_return: float
    equity_curve: pd.Series
    predictions: pd.DataFrame
    regime_breakdown: Dict[str, float]


def build_features_fast(price: pd.DataFrame, banks: Dict[str, pd.DataFrame], step: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
    min_hist = 70
    rows, targets, dates = [], [], []
    indices = list(range(min_hist, len(price) - 1, step))
    logger.info("Building features on %d points...", len(indices))

    for i in indices:
        window = price.iloc[: i + 1]
        bank_w = {t: df.iloc[: min(i + 1, len(df))] for t, df in banks.items()}
        try:
            facs = compute_all_factors(window, bank_w)
            feat = factors_to_dict(facs)
            close = window["Close"]
            feat["ret_1"] = float(close.pct_change(1).iloc[-1] or 0)
            feat["ret_5"] = float(close.pct_change(5).iloc[-1] or 0)
            feat["vol_10"] = float(close.pct_change().rolling(10).std().iloc[-1] or 0)
            rows.append(feat)
            next_ret = price["Close"].iloc[i + 1] / price["Close"].iloc[i] - 1.0
            targets.append(1 if next_ret > 0 else 0)
            dates.append(price.index[i])
        except Exception as e:
            logger.debug("Skip %s: %s", price.index[i], e)
            continue

    X = pd.DataFrame(rows, index=pd.DatetimeIndex(dates))
    y = pd.Series(targets, index=X.index, name="target")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X, y


def walk_forward_backtest(
    price: pd.DataFrame,
    banks: Dict[str, pd.DataFrame],
    n_splits: int = 4,
    train_min: int = 120,
    confidence: float = 0.55,
    cost_bps: float = 4.0,
) -> BacktestResult:
    X, y = build_features_fast(price, banks, step=1)
    logger.info("Feature matrix: %d samples x %d features", len(X), X.shape[1])
    if len(X) < train_min + 30:
        raise RuntimeError(f"Not enough samples ({len(X)}) for walk-forward")

    n = len(X)
    fold_size = max((n - train_min) // n_splits, 15)
    preds, probas, actuals, dts, next_rets = [], [], [], [], []

    for fold in range(n_splits):
        tr_end = train_min + fold * fold_size
        te_end = min(tr_end + fold_size, n)
        if tr_end >= n - 5:
            break
        Xtr, ytr = X.iloc[:tr_end], y.iloc[:tr_end]
        Xte, yte = X.iloc[tr_end:te_end], y.iloc[tr_end:te_end]

        model = XGBClassifier(
            n_estimators=120, max_depth=3, learning_rate=0.06,
            subsample=0.85, colsample_bytree=0.8, min_child_weight=4,
            reg_alpha=0.15, reg_lambda=1.0, random_state=42, n_jobs=2, verbosity=0,
        )
        model.fit(Xtr, ytr)
        proba = model.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.5).astype(int)

        preds.extend(pred)
        probas.extend(proba)
        actuals.extend(yte.values)
        dts.extend(Xte.index.tolist())

        for dt in Xte.index:
            loc = price.index.get_loc(dt)
            r = (price["Close"].iloc[loc + 1] / price["Close"].iloc[loc] - 1) if loc + 1 < len(price) else 0.0
            next_rets.append(r)

    pred_df = pd.DataFrame({
        "actual": actuals, "pred": preds, "proba": probas, "next_ret": next_rets
    }, index=pd.DatetimeIndex(dts))

    signal = np.where(pred_df["proba"] >= confidence, 1,
                      np.where(pred_df["proba"] <= 1 - confidence, -1, 0))
    pred_df["signal"] = signal
    cost = cost_bps / 10000.0
    strat = pred_df["signal"] * pred_df["next_ret"] - np.abs(pred_df["signal"]) * cost
    pred_df["strategy_ret"] = strat

    acc = accuracy_score(pred_df["actual"], pred_df["pred"])
    prec_up = precision_score(pred_df["actual"], pred_df["pred"], pos_label=1, zero_division=0)
    prec_dn = precision_score(pred_df["actual"], pred_df["pred"], pos_label=0, zero_division=0)

    traded = pred_df[pred_df["signal"] != 0]
    n_trades = len(traded)
    if n_trades > 0:
        hit = float((traded["strategy_ret"] > 0).mean())
        avg = float(traded["strategy_ret"].mean())
        std = float(traded["strategy_ret"].std())
        sharpe = (avg / std * np.sqrt(252)) if std > 1e-9 else 0.0
        wins = traded.loc[traded["strategy_ret"] > 0, "strategy_ret"].sum()
        losses = -traded.loc[traded["strategy_ret"] < 0, "strategy_ret"].sum()
        pf = float(wins / losses) if losses > 0 else 999.0
    else:
        hit = avg = sharpe = pf = 0.0

    equity = (1 + strat.fillna(0)).cumprod()
    dd = float(((equity - equity.cummax()) / equity.cummax()).min())
    total = float(equity.iloc[-1] - 1) if len(equity) else 0.0

    regime_rets: Dict[str, List[float]] = {}
    for dt in pred_df.index:
        try:
            loc = price.index.get_loc(dt)
            reg = detect_regime(price.iloc[: loc + 1])
            regime_rets.setdefault(reg, []).append(float(pred_df.loc[dt, "strategy_ret"]))
        except Exception:
            pass
    regime_breakdown = {k: float(np.mean(v)) for k, v in regime_rets.items() if v}

    return BacktestResult(
        accuracy=float(acc), precision_up=float(prec_up), precision_down=float(prec_dn),
        n_trades=n_trades, hit_rate_after_cost=hit, avg_return_per_trade=avg,
        sharpe=sharpe, max_drawdown=dd, profit_factor=pf, total_return=total,
        equity_curve=equity, predictions=pred_df, regime_breakdown=regime_breakdown,
    )


def run_backtest(period: str = "2y", confidence: float = 0.55) -> BacktestResult:
    logger.info("Fetching bank universe (proxy for G3B)...")
    data = fetch_universe(["D05.SI", "O39.SI", "U11.SI"], period=period)
    if len(data) < 2:
        raise RuntimeError("Insufficient bank data")
    price = make_proxy_g3b(data)
    logger.info("Proxy series length: %d", len(price))
    return walk_forward_backtest(price, data, confidence=confidence)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    res = run_backtest(period="2y", confidence=0.55)

    print("\n" + "=" * 60)
    print("  G3B NEXT-DAY PREDICTOR – WALK-FORWARD BACKTEST")
    print("  (Weighted DBS+OCBC+UOB proxy ≈ 57.6% of real G3B)")
    print("=" * 60)
    print(f"  Directional Accuracy        : {res.accuracy:6.1%}")
    print(f"  Precision UP / DOWN         : {res.precision_up:5.1%} / {res.precision_down:5.1%}")
    print(f"  Trades taken (high-conviction): {res.n_trades:4d}")
    print(f"  Hit rate after 4 bps cost   : {res.hit_rate_after_cost:6.1%}")
    print(f"  Avg return per trade        : {res.avg_return_per_trade:7.3%}")
    print(f"  Sharpe ratio (annualised)   : {res.sharpe:6.2f}")
    print(f"  Maximum Drawdown            : {res.max_drawdown:6.1%}")
    print(f"  Profit Factor               : {res.profit_factor:6.2f}")
    print(f"  Total Strategy Return       : {res.total_return:6.1%}")
    print("-" * 60)
    print("  Avg strategy return by regime:")
    for k, v in sorted(res.regime_breakdown.items()):
        print(f"    {k:20s} : {v:+.3%}")
    print("=" * 60)
    print("Note: Results use a bank-weighted proxy because G3B.SI is")
    print("currently rate-limited. Economic exposure is highly similar.")
