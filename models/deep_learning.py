"""
Small PyTorch LSTM for next-day bucket/direction classification.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class LSTMBucketModel(nn.Module):
    """LSTM classifier for time-series feature sequences."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 5,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class LSTMBucketClassifier:
    """Scikit-learn-style wrapper around the LSTM with standard scaling."""

    def __init__(
        self,
        seq_len: int = 20,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.25,
        epochs: int = 120,
        batch_size: int = 32,
        lr: float = 1e-3,
        patience: int = 15,
    ):
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.model: Optional[LSTMBucketModel] = None
        self.scaler = StandardScaler()
        self.classes_: List[str] = []
        self.label_to_int_: Dict[str, int] = {}
        self.int_to_label_: Dict[int, str] = {}
        self.n_features: int = 0

    def _build_sequences(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build non-overlapping-ish sequences for each valid index."""
        valid_idx = y.dropna().index.intersection(X.index)
        Xs, ys, idxs = [], [], []
        values = X.values.astype(float)
        for i, dt in enumerate(valid_idx):
            pos = X.index.get_loc(dt)
            if pos < self.seq_len - 1:
                continue
            seq = values[pos - self.seq_len + 1 : pos + 1]
            if np.isnan(seq).any():
                continue
            Xs.append(seq)
            ys.append(self.label_to_int_[y.loc[dt]])
            idxs.append(dt)
        return np.asarray(Xs), np.asarray(ys), np.asarray(idxs)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LSTMBucketClassifier":
        labels = sorted(y.dropna().unique().tolist())
        if not labels:
            logger.warning("LSTM: no labels to fit")
            return self
        self.classes_ = [str(label) for label in labels]
        self.label_to_int_ = {label: i for i, label in enumerate(self.classes_)}
        self.int_to_label_ = {i: label for i, label in enumerate(self.classes_)}

        # Scale on the full feature matrix (fit is past-only, so no lookahead).
        self.scaler.fit(X.values)
        X_scaled = pd.DataFrame(
            self.scaler.transform(X.values), index=X.index, columns=X.columns
        )

        X_seq, y_seq, _ = self._build_sequences(X_scaled, y)
        if len(X_seq) < 50 or len(np.unique(y_seq)) < 2:
            logger.warning("LSTM: not enough sequences or classes")
            return self

        # Time-series split for validation
        split = int(len(X_seq) * 0.85)
        X_train, X_val = X_seq[:split], X_seq[split:]
        y_train, y_val = y_seq[:split], y_seq[split:]

        self.n_features = X_train.shape[2]
        self.model = LSTMBucketModel(
            input_size=self.n_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_classes=len(self.classes_),
            dropout=self.dropout,
        )

        class_weights = self._class_weights(y_train)
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32)
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        Xt = torch.tensor(X_train, dtype=torch.float32)
        yt = torch.tensor(y_train, dtype=torch.long)
        Xv = torch.tensor(X_val, dtype=torch.float32)
        yv = torch.tensor(y_val, dtype=torch.long)

        best_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), self.batch_size):
                batch_idx = perm[i : i + self.batch_size]
                xb = Xt[batch_idx]
                yb = yt[batch_idx]
                optimizer.zero_grad()
                out = self.model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_out = self.model(Xv)
                val_loss = float(nn.functional.cross_entropy(val_out, yv).item())

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {
                    k: v.cpu().numpy() for k, v in self.model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(
                {k: torch.tensor(v) for k, v in best_state.items()}
            )
        return self

    def _class_weights(self, y: np.ndarray) -> np.ndarray:
        counts = np.bincount(y, minlength=len(self.classes_))
        weights = 1.0 / (counts + 1.0)
        return weights / weights.sum() * len(weights)

    def predict_proba(
        self, X: pd.DataFrame, latest_index: Optional[pd.Timestamp] = None
    ) -> Dict[str, float]:
        if self.model is None:
            return {}
        X_scaled = pd.DataFrame(
            self.scaler.transform(X.values), index=X.index, columns=X.columns
        )
        if latest_index is None:
            latest_index = X_scaled.index[-1]
        pos = X_scaled.index.get_loc(latest_index)
        if pos < self.seq_len - 1:
            return {}
        seq = X_scaled.values[pos - self.seq_len + 1 : pos + 1].astype(float)
        if np.isnan(seq).any():
            return {}
        self.model.eval()
        with torch.no_grad():
            tensor = torch.tensor(seq[np.newaxis, ...], dtype=torch.float32)
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).numpy()[0]
        return {
            self.int_to_label_[i]: float(probs[i]) for i in range(len(self.classes_))
        }

    def predict(
        self, X: pd.DataFrame, latest_index: Optional[pd.Timestamp] = None
    ) -> Optional[str]:
        probs = self.predict_proba(X, latest_index)
        if not probs:
            return None
        return max(probs, key=probs.get)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(
                {
                    "scaler": self.scaler,
                    "classes": self.classes_,
                    "label_to_int": self.label_to_int_,
                    "int_to_label": self.int_to_label_,
                    "n_features": self.n_features,
                    "seq_len": self.seq_len,
                    "state_dict": self.model.state_dict() if self.model else None,
                    "params": {
                        "hidden_size": self.hidden_size,
                        "num_layers": self.num_layers,
                        "dropout": self.dropout,
                    },
                },
                f,
            )

    @classmethod
    def load(cls, path: Path) -> Optional["LSTMBucketClassifier"]:
        if not path.exists():
            return None
        try:
            with path.open("rb") as f:
                payload = pickle.load(f)
            obj = cls(
                seq_len=payload["seq_len"],
                hidden_size=payload["params"]["hidden_size"],
                num_layers=payload["params"]["num_layers"],
                dropout=payload["params"]["dropout"],
            )
            obj.scaler = payload["scaler"]
            obj.classes_ = payload["classes"]
            obj.label_to_int_ = payload["label_to_int"]
            obj.int_to_label_ = payload["int_to_label"]
            obj.n_features = payload["n_features"]
            if payload["state_dict"] is not None:
                obj.model = LSTMBucketModel(
                    input_size=obj.n_features,
                    hidden_size=obj.hidden_size,
                    num_layers=obj.num_layers,
                    num_classes=len(obj.classes_),
                    dropout=obj.dropout,
                )
                obj.model.load_state_dict(payload["state_dict"])
                obj.model.eval()
            return obj
        except Exception as exc:
            logger.warning("Failed to load LSTM model: %s", exc)
            return None
