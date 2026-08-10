"""
Unified G3B next-day + multi-horizon predictor.

Combines:
- a high-conviction next-day rule overlay (factors/high_conviction.py),
- a per-horizon high-conviction signal engine (factors/multi_horizon_signal.py),
- a gradient-boosted + LSTM multi-horizon ensemble for full bucket probabilities
  (models/ensemble.py).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data.data_fetcher import load_g3b_data, make_bank_weighted_return
from factors import high_conviction
from factors.feature_engine import build_features, impute_features
from factors.multi_horizon_signal import SignalEngine
from models.ensemble import MultiHorizonEnsemble

logger = logging.getLogger(__name__)

HORIZONS = [1, 2, 3, 5, 10, 20, 60]


@dataclass
class HorizonPrediction:
    """Prediction container for one horizon."""

    horizon: int
    bucket: str
    bucket_probs: Dict[str, float] = field(default_factory=dict)
    direction: str = "NEUTRAL"
    confidence: float = 0.0
    expected_return_pct: float = 0.0
    model_proba_up: float = 0.5
    signal_type: str = "ensemble"
    reasons: List[str] = field(default_factory=list)
    top_features: List[Tuple[str, float]] = field(default_factory=list)


class G3BPredictor:
    """End-to-end predictor for G3B.SI."""

    def __init__(self, train_end: str = "2023-12-31", period: str = "5y"):
        self.train_end = pd.Timestamp(train_end)
        self.period = period
        self.g3b: Optional[pd.DataFrame] = None
        self.data: Dict[str, pd.DataFrame] = {}
        self.earnings: Dict[str, pd.DataFrame] = {}
        self.feats: Optional[pd.DataFrame] = None
        self.feature_cols: List[str] = []
        self.ensemble: Optional[MultiHorizonEnsemble] = None
        self.signal_engine = SignalEngine()
        self.high_conviction_features: Optional[pd.DataFrame] = None
        self.latest_date: Optional[pd.Timestamp] = None

    def load_data(self) -> None:
        logger.info("Predictor loading G3B data...")
        self.g3b, self.data, self.earnings = load_g3b_data(period=self.period)
        if self.g3b is None or self.g3b.empty:
            raise RuntimeError("Could not load G3B data")

    def build_features(self) -> None:
        if self.g3b is None:
            raise RuntimeError("Load data first")
        self.feats, self.feature_cols = build_features(
            self.g3b, self.data, self.earnings, start=None, end=None
        )
        self.feats = impute_features(self.feats, self.feature_cols)
        self.feats = self.feats.replace([np.inf, -np.inf], np.nan)

        # Build the legacy high-conviction feature set for next-day overlay.
        bank_dfs = {
            t: df for t, df in self.data.items() if t in ["D05.SI", "O39.SI", "U11.SI"]
        }
        bank_weight = make_bank_weighted_return(bank_dfs)
        macro = {
            t: df
            for t, df in self.data.items()
            if t not in ["G3B.SI", "D05.SI", "O39.SI", "U11.SI"]
        }
        self.high_conviction_features = high_conviction.build_features(
            self.g3b, macro, bank_weight
        )

    def fit(self) -> None:
        if self.feats is None:
            self.build_features()

        logger.info("Training multi-horizon ensemble...")
        self.ensemble = MultiHorizonEnsemble()
        self.ensemble.fit(
            self.feats, self.feature_cols, train_end=str(self.train_end.date())
        )

        logger.info("Training multi-horizon signal engine...")
        self.signal_engine.fit(
            self.feats, self.feature_cols, train_end=str(self.train_end.date())
        )

        self.latest_date = self.feats.index[-1]

    def _next_day_high_conviction(
        self, date: Optional[pd.Timestamp] = None
    ) -> Optional[high_conviction.HighConvictionSignal]:
        """Evaluate the legacy high-conviction next-day rule set."""
        if self.high_conviction_features is None:
            return None
        if date is None:
            date = self.high_conviction_features.index[-1]
        if date not in self.high_conviction_features.index:
            return None

        # XGB proba is computed live on all prior rows (no lookahead).
        xgb_proba = high_conviction.compute_xgb_proba(
            self.high_conviction_features,
            self.g3b["Close"],
            train_cutoff=None,
        )
        row = self.high_conviction_features.loc[date].copy()
        row["xgb_proba"] = xgb_proba.loc[date]
        return high_conviction.evaluate_high_conviction(row)

    def _bucket_from_return(self, ret: float, horizon: int) -> str:
        std = (
            float(self.feats[f"label_ret_{horizon}d"].std())
            if self.feats is not None
            else 0.01
        )
        std = max(std, 0.001)
        t = sorted([-0.75 * std, -0.15 * std, 0.15 * std, 0.75 * std])
        if ret <= t[0]:
            return "Strong Down"
        if ret <= t[1]:
            return "Weak Down"
        if ret < t[2]:
            return "Flat"
        if ret < t[3]:
            return "Weak Up"
        return "Strong Up"

    def predict(self, date: Optional[pd.Timestamp] = None) -> Dict[str, Any]:
        if self.ensemble is None or self.feats is None:
            self.fit()
        if date is None:
            date = self.feats.index[-1]
        if date not in self.feats.index:
            raise ValueError(f"Date {date} not in feature matrix")

        row = self.feats.loc[date]

        # Full ensemble bucket probabilities for all horizons.
        ensemble_preds = self.ensemble.predict(self.feats, latest_index=date)

        # Next-day overlay (h=1).
        hc1 = self._next_day_high_conviction(date)

        horizons: Dict[int, HorizonPrediction] = {}
        for h in HORIZONS:
            ens = ensemble_preds.get(h, {})
            if h == 1 and hc1 is not None:
                # High-conviction next-day signal from legacy overlay.
                direction = "UP" if hc1.direction == "LONG" else "DOWN"
                bucket = self._bucket_from_return(
                    hc1.avg_next_day_return_pct / 100.0, h
                )
                confidence = hc1.confidence
                reasons = list(hc1.reasons)
                reasons.append(f"Triggered rule: {hc1.rule_name}")
                horizons[h] = HorizonPrediction(
                    horizon=h,
                    bucket=bucket,
                    bucket_probs=ens.get("bucket_probs", {}),
                    direction=direction,
                    confidence=confidence,
                    expected_return_pct=hc1.avg_next_day_return_pct,
                    model_proba_up=1.0 if hc1.direction == "LONG" else 0.0,
                    signal_type="high_conviction_rule",
                    reasons=reasons,
                    top_features=ens.get("top_features", []),
                )
            else:
                sig = self.signal_engine.predict(row, h)
                if sig.signal_type != "ensemble":
                    horizons[h] = HorizonPrediction(
                        horizon=h,
                        bucket=sig.bucket,
                        bucket_probs=ens.get("bucket_probs", {}),
                        direction="UP" if sig.direction == "LONG" else "DOWN",
                        confidence=sig.confidence,
                        expected_return_pct=sig.expected_return_pct,
                        model_proba_up=sig.model_proba_up,
                        signal_type=sig.signal_type,
                        reasons=sig.reasons,
                        top_features=sig.top_features or ens.get("top_features", []),
                    )
                else:
                    horizons[h] = HorizonPrediction(
                        horizon=h,
                        bucket=ens.get("bucket", "Flat"),
                        bucket_probs=ens.get("bucket_probs", {}),
                        direction=ens.get("direction", "NEUTRAL"),
                        confidence=ens.get("confidence", 0.0),
                        expected_return_pct=0.0,
                        model_proba_up=sig.model_proba_up,
                        signal_type="ensemble",
                        reasons=sig.reasons,
                        top_features=ens.get("top_features", []),
                    )

        return {
            "as_of": str(date.date()),
            "last_close": float(self.g3b["Close"].loc[date]),
            "latest_date": str(self.g3b.index[-1].date()),
            "horizons": horizons,
            "data_health": self._data_health(),
            "earnings": self._earnings_summary(date),
            "top_features_global": self._global_top_features(),
        }

    def _data_health(self) -> Dict[str, Any]:
        if self.g3b is None:
            return {"status": "error", "missing": []}
        latest = self.g3b.index[-1]
        missing = [t for t, df in self.data.items() if df is None or df.empty]
        return {
            "status": "ok" if not missing else "degraded",
            "latest_date": str(latest.date()),
            "missing_tickers": missing,
            "n_rows": len(self.g3b),
        }

    def _earnings_summary(self, date: pd.Timestamp) -> List[Dict[str, Any]]:
        """Upcoming earnings within 20 days for top holdings."""
        if not self.earnings:
            return []
        out = []
        for ticker, ed in self.earnings.items():
            future = ed[ed.index > date]
            if future.empty:
                continue
            next_dt = future.index[0]
            days = (next_dt - date).days
            if days <= 20:
                out.append(
                    {
                        "ticker": ticker,
                        "date": str(next_dt.date()),
                        "days_to": int(days),
                        "surprise_pct": float(future.iloc[0].get("Surprise(%)", np.nan))
                        if "Surprise(%)" in future.columns
                        else None,
                    }
                )
        return sorted(out, key=lambda x: x["days_to"])

    def _global_top_features(self) -> List[Tuple[str, float]]:
        if self.ensemble is None:
            return []
        # Average bucket model importances across horizons.
        importances: Dict[str, float] = {}
        counts = 0
        for h in [1, 2, 3, 5, 10, 20, 60]:
            model = self.ensemble.bucket_models.get(h)
            if model is None:
                continue
            for name, val in model.importances.items():
                importances[name] = importances.get(name, 0.0) + val
            counts += 1
        if counts == 0:
            return []
        avg = {k: v / counts for k, v in importances.items()}
        return sorted(avg.items(), key=lambda x: x[1], reverse=True)[:10]

    def backtest(
        self,
        start: str = "2024-01-01",
        end: str = "2026-08-08",
        cost_bps: float = 4.0,
    ) -> Dict[str, Any]:
        """Run an OOS multi-horizon backtest and return per-horizon metrics."""
        if self.ensemble is None or self.feats is None:
            self.fit()

        oos = self.feats[
            (self.feats.index >= pd.Timestamp(start))
            & (self.feats.index <= pd.Timestamp(end))
        ].copy()
        if oos.empty:
            raise ValueError(f"No OOS rows between {start} and {end}")

        # Pre-compute ensemble predictions for every OOS row.
        all_ens = self.ensemble.predict_proba_all(self.feats)

        # Pre-compute next-day high-conviction xgb_proba OOS to avoid lookahead.
        if self.high_conviction_features is not None:
            hc_xgb = high_conviction.compute_xgb_proba(
                self.high_conviction_features,
                self.g3b["Close"],
                train_cutoff=start,
            )
        else:
            hc_xgb = pd.Series(np.nan, index=self.feats.index)

        cost = cost_bps / 10000.0
        per_horizon: Dict[int, Dict[str, Any]] = {}

        for h in HORIZONS:
            trades = []
            for dt in oos.index:
                row = self.feats.loc[dt]
                actual_ret = oos.loc[dt, f"label_ret_{h}d"]
                if pd.isna(actual_ret):
                    continue

                # Direction/bucket from signal engine or ensemble.
                if h == 1:
                    if dt in self.high_conviction_features.index and pd.notna(
                        hc_xgb.loc[dt]
                    ):
                        hc_row = self.high_conviction_features.loc[dt].copy()
                        hc_row["xgb_proba"] = hc_xgb.loc[dt]
                        sig = high_conviction.evaluate_high_conviction(hc_row)
                    else:
                        sig = None
                    if sig is not None:
                        direction = sig.direction
                        confidence = sig.confidence
                        signal_type = "high_conviction_rule"
                        bucket = self._bucket_from_return(
                            sig.avg_next_day_return_pct / 100.0, h
                        )
                    else:
                        ens = all_ens[h]
                        p_up = (
                            ens["direction_probs"]
                            .get("UP", pd.Series(0.5, index=oos.index))
                            .loc[dt]
                        )
                        direction = "LONG" if p_up >= 0.5 else "SHORT"
                        confidence = max(p_up, 1 - p_up)
                        signal_type = "ensemble"
                        bucket = ens["predicted_bucket"].loc[dt]
                else:
                    sig = self.signal_engine.predict(row, h)
                    if sig.signal_type != "ensemble":
                        direction = sig.direction
                        confidence = sig.confidence
                        signal_type = sig.signal_type
                        bucket = sig.bucket
                    else:
                        ens = all_ens[h]
                        p_up = (
                            ens["direction_probs"]
                            .get("UP", pd.Series(0.5, index=oos.index))
                            .loc[dt]
                        )
                        direction = "LONG" if p_up >= 0.5 else "SHORT"
                        confidence = max(p_up, 1 - p_up)
                        signal_type = "ensemble"
                        bucket = ens["predicted_bucket"].loc[dt]

                pos = 1.0 if direction == "LONG" else -1.0
                net_ret = pos * actual_ret - cost
                hit = (actual_ret > 0) if direction == "LONG" else (actual_ret < 0)
                trades.append(
                    {
                        "date": dt,
                        "direction": direction,
                        "signal_type": signal_type,
                        "bucket": bucket,
                        "predicted_direction": "UP" if direction == "LONG" else "DOWN",
                        "actual_return": actual_ret,
                        "net_return": net_ret,
                        "hit": hit,
                        "confidence": float(confidence),
                    }
                )

            if not trades:
                per_horizon[h] = self._empty_metrics(h)
                continue

            rets = pd.Series([t["net_return"] for t in trades])
            equity = (1 + rets).cumprod()
            dd = float((equity / equity.cummax() - 1).min())
            std = float(rets.std()) or 1e-9
            sharpe = float(rets.mean() / std * np.sqrt(252 / h)) if h > 0 else 0.0

            # High-conviction subset metrics.
            hc_trades = [t for t in trades if t["signal_type"] != "ensemble"]
            hc_acc = (
                sum(t["hit"] for t in hc_trades) / len(hc_trades) if hc_trades else 0.0
            )
            hc_avg = (
                float(pd.Series([t["net_return"] for t in hc_trades]).mean())
                if hc_trades
                else 0.0
            )

            per_horizon[h] = {
                "horizon": h,
                "n": len(trades),
                "directional_accuracy": sum(t["hit"] for t in trades) / len(trades),
                "bucket_accuracy": 0.0,  # computed separately if needed
                "high_conviction_n": len(hc_trades),
                "high_conviction_accuracy": hc_acc,
                "high_conviction_avg_return": hc_avg,
                "avg_return_per_trade": float(rets.mean()),
                "total_return": float(equity.iloc[-1] - 1),
                "max_drawdown": dd,
                "sharpe": sharpe,
                "trades": trades,
            }

        return {
            "start": start,
            "end": end,
            "cost_bps": cost_bps,
            "generated_at": datetime.utcnow().isoformat(),
            "horizons": per_horizon,
        }

    def _empty_metrics(self, horizon: int) -> Dict[str, Any]:
        return {
            "horizon": horizon,
            "n": 0,
            "directional_accuracy": 0.0,
            "bucket_accuracy": 0.0,
            "high_conviction_n": 0,
            "high_conviction_accuracy": 0.0,
            "high_conviction_avg_return": 0.0,
            "avg_return_per_trade": 0.0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "trades": [],
        }
