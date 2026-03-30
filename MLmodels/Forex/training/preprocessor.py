"""
TradeAI — Training Preprocessor
Handles normalization, sequence creation, and train/val/test splitting.
Scaler is saved/loaded per (pair, timeframe) so inference uses the same transform.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


def _load_cfg() -> dict:
    import yaml
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


class Preprocessor:
    """
    Preprocessing pipeline shared between training, inference, and backtesting.

    Responsibilities
    ----------------
    - Fit/transform feature scaling (MinMaxScaler)
    - Create sliding windows (sequences) for LSTM input
    - Chronological train/val/test split
    - Persist and restore scaler per (pair, timeframe)
    """

    def __init__(
        self,
        models_root: str,
        cfg: Optional[dict] = None,
    ):
        self.cfg = cfg or _load_cfg()
        self.models_root = models_root
        train_cfg = self.cfg.get("training", {})
        model_cfg = self.cfg.get("models", {})
        self.seq_length: int = train_cfg.get("sequence_length", 60)
        self.scaler_filename: str = model_cfg.get("scaler_filename", "scaler.pkl")
        self.scaler: MinMaxScaler = MinMaxScaler()
        self._fitted: bool = False

    # ──────────────────────────────────────────────────────────────────────
    # Scaling
    # ──────────────────────────────────────────────────────────────────────

    def fit_transform(self, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
        """Fit scaler on df[feature_cols] and return scaled array."""
        data = df[feature_cols].values
        scaled = self.scaler.fit_transform(data)
        self._fitted = True
        logger.debug("Scaler fitted on %d rows × %d features.", *scaled.shape)
        return scaled

    def transform(self, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
        """Transform using an already-fitted scaler."""
        if not self._fitted:
            raise RuntimeError("Scaler not fitted. Call fit_transform first or load a saved scaler.")
        return self.scaler.transform(df[feature_cols].values)

    # ──────────────────────────────────────────────────────────────────────
    # Sequence Creation
    # ──────────────────────────────────────────────────────────────────────

    def create_sequences(
        self,
        scaled: np.ndarray,
        labels_direction: np.ndarray,
        labels_price: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Build sliding-window sequences for LSTM input.

        Parameters
        ----------
        scaled           : 2D array (n_rows, n_features) — scaled feature matrix
        labels_direction : 1D array of binary direction targets
        labels_price     : 1D array of regression price targets (optional)

        Returns
        -------
        X               : (n_samples, seq_length, n_features)
        y_direction     : (n_samples,) — binary
        y_price         : (n_samples,) or None
        """
        X, y_dir, y_price = [], [], []
        n = len(scaled)
        for i in range(n - self.seq_length):
            X.append(scaled[i : i + self.seq_length])
            y_dir.append(labels_direction[i + self.seq_length])
            if labels_price is not None:
                y_price.append(labels_price[i + self.seq_length])

        X = np.array(X, dtype=np.float32)
        y_dir = np.array(y_dir, dtype=np.float32)
        y_price_arr = np.array(y_price, dtype=np.float32) if y_price else None

        logger.debug("Sequences: X=%s y_dir=%s", X.shape, y_dir.shape)
        return X, y_dir, y_price_arr

    # ──────────────────────────────────────────────────────────────────────
    # Train / Val / Test Split
    # ──────────────────────────────────────────────────────────────────────

    def walk_forward_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: Optional[float] = None,
        val_ratio: Optional[float] = None,
    ) -> Tuple[Tuple, Tuple, Tuple]:
        """
        Chronological split (no shuffling) for time-series data.

        Returns
        -------
        (X_train, y_train), (X_val, y_val), (X_test, y_test)
        """
        train_r = train_ratio or self.cfg["training"]["train_ratio"]
        val_r = val_ratio or self.cfg["training"]["val_ratio"]

        n = len(X)
        train_end = int(n * train_r)
        val_end = int(n * (train_r + val_r))

        return (
            (X[:train_end], y[:train_end]),
            (X[train_end:val_end], y[train_end:val_end]),
            (X[val_end:], y[val_end:]),
        )

    # ──────────────────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────────────────

    def save_scaler(self, symbol: str, timeframe: str) -> str:
        """Save the fitted scaler to the model directory."""
        path = self._scaler_path(symbol, timeframe)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.scaler, path)
        logger.info("Scaler saved to %s", path)
        return path

    def load_scaler(self, symbol: str, timeframe: str) -> "Preprocessor":
        """Load a saved scaler. Returns self for chaining."""
        path = self._scaler_path(symbol, timeframe)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Scaler not found at {path}")
        self.scaler = joblib.load(path)
        self._fitted = True
        logger.info("Scaler loaded from %s", path)
        return self

    def _scaler_path(self, symbol: str, timeframe: str) -> str:
        pair_tag = symbol.replace("/", "")
        return os.path.join(
            self.models_root, pair_tag, timeframe, self.scaler_filename
        )
