"""
Multi-horizon ML ensemble for G3B.

Trains gradient-boosted (XGBoost / LightGBM / HistGradient) and Random Forest
classifiers for each horizon.  A separate LSTM is trained for the next-day
bucket.  Probabilities are averaged across all available models and optionally
gated by a high-conviction rule overlay.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

from models.deep_learning import LSTMBucketClassifier

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent / "saved_models"
MODELS_DIR.mkdir(exist_ok=True)

HORIZONS = [1, 2, 3, 5, 10, 20, 60]
BUCKET_LABELS = ["Strong Down", "Weak Down", "Flat", "Weak Up", "Strong Up"]


def _try_classifier(kind: str, n_classes: int):
    """Instantiate an available tree classifier based on class count."""
    if n_classes < 2:
        return None
    if kind == "xgb":
        try:
            from xgboost import XGBClassifier

            if n_classes == 2:
                return XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.85,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=42,
                    n_jobs=2,
                    verbosity=0,
                )
            return XGBClassifier(
                objective="multi:softprob",
                eval_metric="mlogloss",
                num_class=n_classes,
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=2,
                verbosity=0,
            )
        except Exception:
            return None
    if kind == "lgb":
        try:
            from lightgbm import LGBMClassifier

            return LGBMClassifier(
                objective="binary" if n_classes == 2 else "multiclass",
                n_estimators=100,
                learning_rate=0.05,
                num_leaves=31,
                verbosity=-1,
                n_jobs=2,
                random_state=42,
            )
        except Exception:
            return None
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=2,
        )
    return None


def _make_tree_ensemble(n_classes: int):
    """Return a list of (name, unfitted classifier) tuples."""
    out: List[Tuple[str, Any]] = []
    for kind in ["xgb", "lgb"]:
        clf = _try_classifier(kind, n_classes)
        if clf is not None:
            out.append((kind, clf))
    return out


class PerHorizonModel:
    """Container for the classifiers and bucket thresholds of a single horizon."""

    def __init__(self, horizon: int, task: str):
        self.horizon = horizon
        self.task = task  # "bucket" or "direction"
        self.classes: List[str] = []
        self.label_to_int: Dict[str, int] = {}
        self.int_to_label: Dict[int, str] = {}
        self.models: Dict[str, Any] = {}
        self.feature_names: List[str] = []
        self.thresholds: Optional[List[float]] = None
        self.importances: Dict[str, float] = {}

    def fit(
        self, X: pd.DataFrame, y: pd.Series, feature_names: List[str]
    ) -> "PerHorizonModel":
        labels = sorted(y.dropna().unique().tolist(), key=lambda x: _bucket_order(x))
        self.classes = [str(label) for label in labels]
        self.label_to_int = {label: i for i, label in enumerate(self.classes)}
        self.int_to_label = {i: label for i, label in enumerate(self.classes)}
        self.feature_names = list(feature_names)

        y_int = y.map(self.label_to_int)
        valid = y_int.notna() & X.notna().all(axis=1)
        Xv = X.loc[valid].values
        yv = y_int.loc[valid].values.astype(int)
        if len(yv) < 50 or len(np.unique(yv)) < 2:
            logger.warning("Horizon %d: insufficient data", self.horizon)
            return self

        for name, clf in _make_tree_ensemble(len(self.classes)):
            try:
                clf.fit(Xv, yv)
                self.models[name] = clf
            except Exception as exc:
                logger.warning(
                    "%s failed for h=%d: %s", name, self.horizon, exc, exc_info=True
                )

        self.importances = self._compute_importances(Xv, yv)
        return self

    def predict_proba(self, x: pd.Series | np.ndarray) -> Dict[str, float]:
        if not self.models:
            return (
                {c: 1.0 / len(self.classes) for c in self.classes}
                if self.classes
                else {}
            )
        xarr = np.asarray(x, dtype=float).reshape(1, -1)
        probs = np.zeros(len(self.classes))
        for clf in self.models.values():
            try:
                p = clf.predict_proba(xarr)[0]
                classes = list(clf.classes_)
                if len(p) == len(probs):
                    probs += p
                elif len(classes) == len(p):
                    # Classifiers trained on integer labels return numeric classes.
                    for i, cls in enumerate(classes):
                        idx = int(cls)
                        if 0 <= idx < len(probs):
                            probs[idx] += p[i]
                else:
                    for i, cls in enumerate(classes):
                        if cls in self.label_to_int:
                            probs[self.label_to_int[cls]] += p[i]
            except Exception as exc:
                logger.debug("predict_proba failed: %s", exc)
        total = probs.sum()
        if total > 0:
            probs = probs / total
        return {self.int_to_label[i]: float(probs[i]) for i in range(len(self.classes))}

    def predict_proba_df(self, X: pd.DataFrame) -> pd.DataFrame:
        """Batch probability predictions for all rows of X."""
        if not self.models:
            n = X.shape[0]
            base = 1.0 / len(self.classes) if self.classes else 0.0
            return pd.DataFrame({c: [base] * n for c in self.classes}, index=X.index)
        probs = np.zeros((len(X), len(self.classes)))
        for clf in self.models.values():
            try:
                p = clf.predict_proba(X.values)
                classes = list(clf.classes_)
                if len(classes) == p.shape[1]:
                    for i, cls in enumerate(classes):
                        idx = None
                        if isinstance(cls, (int, np.integer, float, np.floating)):
                            idx = int(cls)
                        elif isinstance(cls, str) and cls.isdigit():
                            idx = int(cls)
                        else:
                            idx = self.label_to_int.get(str(cls))
                        if idx is not None and 0 <= idx < len(self.classes):
                            probs[:, idx] += p[:, i]
                else:
                    probs[:, : p.shape[1]] += p
            except Exception as exc:
                logger.debug("predict_proba_df failed: %s", exc)
        total = probs.sum(axis=1, keepdims=True)
        probs = np.where(total > 0, probs / total, 1.0 / len(self.classes))
        return pd.DataFrame(probs, index=X.index, columns=self.classes)

    def _compute_importances(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        imp = np.zeros(len(self.feature_names))
        n = 0
        for clf in self.models.values():
            try:
                if hasattr(clf, "feature_importances_"):
                    imp += np.asarray(clf.feature_importances_)
                    n += 1
            except Exception:
                pass
        if n == 0 and X is not None and y is not None:
            try:
                for i in range(X.shape[1]):
                    with np.errstate(divide="ignore", invalid="ignore"):
                        c = np.abs(np.corrcoef(X[:, i], y)[0, 1])
                    imp[i] = 0.0 if np.isnan(c) else c
                n = 1
            except Exception:
                pass
        if n > 0:
            imp = imp / n
        total = imp.sum() or 1.0
        return {
            self.feature_names[i]: float(imp[i] / total)
            for i in range(len(self.feature_names))
        }

    def top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        return sorted(self.importances.items(), key=lambda kv: kv[1], reverse=True)[:n]

    def save(self, path: Path) -> None:
        with path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> Optional["PerHorizonModel"]:
        if not path.exists():
            return None
        try:
            with path.open("rb") as f:
                return pickle.load(f)
        except Exception as exc:
            logger.warning("Failed to load model %s: %s", path, exc)
            return None


def _bucket_order(label: Any) -> int:
    """Stable ordering for bucket labels."""
    if label in BUCKET_LABELS:
        return BUCKET_LABELS.index(str(label))
    if str(label).lower() == "up":
        return 1
    if str(label).lower() == "down":
        return -1
    return 0


class MultiHorizonEnsemble:
    """Train and predict multi-horizon bucket/direction for G3B."""

    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = models_dir or MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.horizon_models: Dict[int, PerHorizonModel] = {}
        self.direction_models: Dict[int, PerHorizonModel] = {}
        self.bucket_models: Dict[int, PerHorizonModel] = {}
        self.lstm_next_day: Optional[LSTMBucketClassifier] = None
        self.feature_names: List[str] = []
        self.imputer = SimpleImputer(strategy="median")
        self.fitted = False
        self.bucket_thresholds: Dict[int, List[float]] = {}

    def _bucket_labels(self, ret: pd.Series) -> pd.Series:
        """Adaptive 5-bucket labels based on the return standard deviation."""
        std = float(ret.std()) if len(ret) > 0 else 0.01
        thresholds = [-0.75 * std, -0.15 * std, 0.15 * std, 0.75 * std]
        t = sorted(thresholds)
        self.bucket_thresholds[ret.name] = t
        out = pd.Series(index=ret.index, dtype="object")
        out[ret <= t[0]] = BUCKET_LABELS[0]
        out[(ret > t[0]) & (ret <= t[1])] = BUCKET_LABELS[1]
        out[(ret > t[1]) & (ret < t[2])] = BUCKET_LABELS[2]
        out[(ret >= t[2]) & (ret < t[3])] = BUCKET_LABELS[3]
        out[ret >= t[3]] = BUCKET_LABELS[4]
        return out

    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        train_end: str = "2023-12-31",
    ) -> "MultiHorizonEnsemble":
        """Train on all rows <= train_end; keep rows > train_end untouched."""
        logger.info("Training multi-horizon ensemble up to %s", train_end)
        train_df = df[df.index <= pd.Timestamp(train_end)].copy()
        self.feature_names = list(feature_cols)

        # Impute on the full train window (training set is past-only, so no lookahead).
        self.imputer.fit(train_df[self.feature_names].values)
        X_train = pd.DataFrame(
            self.imputer.transform(train_df[self.feature_names].values),
            index=train_df.index,
            columns=self.feature_names,
        )

        # Next-day bucket LSTM (exclude the last training row to avoid lookahead)
        y_bucket_1 = self._bucket_labels(train_df["label_ret_1d"].iloc[:-1])
        X_lstm = X_train.iloc[:-1]
        self.lstm_next_day = LSTMBucketClassifier(
            seq_len=10, hidden_size=32, num_layers=2, dropout=0.25, epochs=50
        )
        self.lstm_next_day.fit(X_lstm, y_bucket_1)

        for h in HORIZONS:
            # Exclude last h rows so the forward-return label is known by train_end.
            X_h = X_train.iloc[:-h] if h > 0 else X_train
            ret_h = train_df[f"label_ret_{h}d"].iloc[:-h]

            # Bucket model
            y_bucket = self._bucket_labels(ret_h)
            bucket = PerHorizonModel(h, "bucket")
            bucket.fit(X_h, y_bucket, self.feature_names)
            self.bucket_models[h] = bucket

            # Direction model (UP / DOWN)
            y_dir = (ret_h > 0).map({True: "UP", False: "DOWN"})
            direction = PerHorizonModel(h, "direction")
            direction.fit(X_h, y_dir, self.feature_names)
            self.direction_models[h] = direction

            # Primary combined model (kept for compatibility)
            self.horizon_models[h] = bucket

        self.fitted = True
        return self

    def predict(
        self,
        df: pd.DataFrame,
        latest_index: Optional[pd.Timestamp] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """Return per-horizon predictions for the supplied feature DataFrame."""
        if not self.fitted:
            raise RuntimeError("Ensemble has not been fitted")
        if latest_index is None:
            latest_index = df.index[-1]

        X_full = pd.DataFrame(
            self.imputer.transform(df[self.feature_names].values),
            index=df.index,
            columns=self.feature_names,
        )
        x = X_full.loc[latest_index]

        result: Dict[int, Dict[str, Any]] = {}
        for h in HORIZONS:
            bucket_probs = self.bucket_models[h].predict_proba(x)
            dir_probs = self.direction_models[h].predict_proba(x)

            # Combine bucket probabilities with direction model by redistributing mass:
            # scale weak/strong up by P(UP), weak/strong down by P(DOWN).
            p_up = dir_probs.get("UP", 0.5)
            p_down = dir_probs.get("DOWN", 0.5)
            adjusted = self._adjust_bucket_probs(bucket_probs, p_up, p_down)

            predicted_bucket = max(adjusted, key=adjusted.get)
            predicted_direction = "UP" if p_up >= p_down else "DOWN"
            confidence = max(adjusted.values())

            result[h] = {
                "bucket": predicted_bucket,
                "bucket_probs": adjusted,
                "direction": predicted_direction,
                "direction_probs": dir_probs,
                "confidence": confidence,
                "top_features": self.bucket_models[h].top_features(5),
            }

        # Add LSTM next-day bucket probability overlay.
        if self.lstm_next_day is not None:
            lstm_probs = self.lstm_next_day.predict_proba(X_full, latest_index)
            if lstm_probs:
                result[1]["lstm_probs"] = lstm_probs
                # Blend LSTM into next-day bucket probabilities (50/50 with trees).
                blended = self._blend_probs(
                    result[1]["bucket_probs"], lstm_probs, weight=0.5
                )
                result[1]["bucket_probs"] = blended
                result[1]["bucket"] = max(blended, key=blended.get)
                result[1]["confidence"] = max(blended.values())

        return result

    def predict_proba_all(self, df: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
        """Batch predictions for every row in df (used by backtests)."""
        if not self.fitted:
            raise RuntimeError("Ensemble has not been fitted")
        X_full = pd.DataFrame(
            self.imputer.transform(df[self.feature_names].values),
            index=df.index,
            columns=self.feature_names,
        )
        result: Dict[int, Dict[str, Any]] = {}
        for h in HORIZONS:
            bucket_df = self.bucket_models[h].predict_proba_df(X_full)
            dir_df = self.direction_models[h].predict_proba_df(X_full)
            adj = self._adjust_bucket_probs_df(bucket_df, dir_df)
            predicted_bucket = adj.idxmax(axis=1)
            confidence = adj.max(axis=1)
            p_up = dir_df.get("UP", pd.Series(0.5, index=dir_df.index))
            p_down = dir_df.get("DOWN", pd.Series(0.5, index=dir_df.index))
            predicted_direction = (p_up >= p_down).map({True: "UP", False: "DOWN"})
            result[h] = {
                "bucket_probs": adj,
                "direction_probs": dir_df,
                "predicted_bucket": predicted_bucket,
                "predicted_direction": predicted_direction,
                "confidence": confidence,
            }
        return result

    def _adjust_bucket_probs(
        self, bucket_probs: Dict[str, float], p_up: float, p_down: float
    ) -> Dict[str, float]:
        """Rescale bucket probabilities to respect a separate direction forecast."""
        base = {k: v for k, v in bucket_probs.items()}
        up_mass = base.get("Weak Up", 0.0) + base.get("Strong Up", 0.0)
        down_mass = base.get("Weak Down", 0.0) + base.get("Strong Down", 0.0)
        flat_mass = base.get("Flat", 0.0)
        if up_mass + down_mass + flat_mass <= 0:
            return base
        # New up/down totals, preserving split within each side and flat unchanged.
        new_up = p_up * (up_mass + down_mass + flat_mass * 0.5)
        new_down = p_down * (up_mass + down_mass + flat_mass * 0.5)
        flat_new = flat_mass * (1 - p_up - p_down)
        flat_new = max(flat_new, 0.0)

        def split(old_a: float, old_b: float, new_total: float):
            total = old_a + old_b
            if total <= 0:
                return new_total / 2.0, new_total / 2.0
            return new_total * old_a / total, new_total * old_b / total

        weak_up, strong_up = split(
            base.get("Weak Up", 0.0), base.get("Strong Up", 0.0), new_up
        )
        weak_down, strong_down = split(
            base.get("Weak Down", 0.0), base.get("Strong Down", 0.0), new_down
        )
        adjusted = {
            "Strong Down": strong_down,
            "Weak Down": weak_down,
            "Flat": flat_new,
            "Weak Up": weak_up,
            "Strong Up": strong_up,
        }
        total = sum(adjusted.values()) or 1.0
        return {k: v / total for k, v in adjusted.items()}

    def _adjust_bucket_probs_df(
        self, bucket_df: pd.DataFrame, dir_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Vectorised version of _adjust_bucket_probs."""
        base = bucket_df.copy()
        up_mass = base.get("Weak Up", 0) + base.get("Strong Up", 0)
        down_mass = base.get("Weak Down", 0) + base.get("Strong Down", 0)
        flat_mass = base.get("Flat", 0)
        p_up = dir_df.get("UP", pd.Series(0.5, index=base.index))
        p_down = dir_df.get("DOWN", pd.Series(0.5, index=base.index))
        total_mass = up_mass + down_mass + flat_mass
        new_up = p_up * (total_mass - flat_mass * 0.5)
        new_down = p_down * (total_mass - flat_mass * 0.5)
        flat_new = (flat_mass * (1.0 - p_up - p_down)).clip(lower=0.0)

        def split(old_a, old_b, new_total):
            total = old_a + old_b
            new_a = np.where(total > 0, new_total * old_a / total, new_total * 0.5)
            new_b = np.where(total > 0, new_total * old_b / total, new_total * 0.5)
            return new_a, new_b

        weak_up, strong_up = split(base["Weak Up"], base["Strong Up"], new_up)
        weak_down, strong_down = split(base["Weak Down"], base["Strong Down"], new_down)
        adjusted = pd.DataFrame(
            {
                "Strong Down": strong_down,
                "Weak Down": weak_down,
                "Flat": flat_new,
                "Weak Up": weak_up,
                "Strong Up": strong_up,
            },
            index=base.index,
        )
        row_total = adjusted.sum(axis=1).replace(0, np.nan)
        adjusted = adjusted.div(row_total, axis=0).fillna(0.2)
        return adjusted

    def _blend_probs(
        self, a: Dict[str, float], b: Dict[str, float], weight: float = 0.5
    ) -> Dict[str, float]:
        keys = set(a.keys()) | set(b.keys())
        out = {}
        for k in keys:
            out[k] = weight * a.get(k, 0.0) + (1 - weight) * b.get(k, 0.0)
        total = sum(out.values()) or 1.0
        return {k: v / total for k, v in out.items()}

    def save(self, path: Optional[Path] = None) -> None:
        path = path or self.models_dir / "ensemble.pkl"
        payload = {
            "bucket_models": self.bucket_models,
            "direction_models": self.direction_models,
            "feature_names": self.feature_names,
            "imputer": self.imputer,
            "fitted": self.fitted,
            "bucket_thresholds": self.bucket_thresholds,
        }
        with path.open("wb") as f:
            pickle.dump(payload, f)
        # LSTM saved separately because of torch state
        if self.lstm_next_day is not None:
            self.lstm_next_day.save(self.models_dir / "lstm_next_day.pkl")

    def load(self, path: Optional[Path] = None) -> "MultiHorizonEnsemble":
        path = path or self.models_dir / "ensemble.pkl"
        if not path.exists():
            return self
        try:
            with path.open("rb") as f:
                payload = pickle.load(f)
            self.bucket_models = payload["bucket_models"]
            self.direction_models = payload["direction_models"]
            self.horizon_models = self.bucket_models
            self.feature_names = payload["feature_names"]
            self.imputer = payload["imputer"]
            self.fitted = payload.get("fitted", True)
            self.bucket_thresholds = payload.get("bucket_thresholds", {})
        except Exception as exc:
            logger.warning("Failed to load ensemble: %s", exc)
        self.lstm_next_day = LSTMBucketClassifier.load(
            self.models_dir / "lstm_next_day.pkl"
        )
        return self
