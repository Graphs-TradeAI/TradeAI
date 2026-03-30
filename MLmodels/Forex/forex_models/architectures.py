"""
TradeAI — Model Architectures
LSTM and hybrid deep learning models for forex prediction.
"""

from __future__ import annotations

import logging
from typing import Optional

import tensorflow as tf
from tensorflow.keras import layers, models

logger = logging.getLogger(__name__)


def _load_cfg() -> dict:
    import os, yaml
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────────────────────────────────
# Classification Model (Direction: BUY / SELL)
# ──────────────────────────────────────────────────────────────────────────

def build_classification_lstm(
    input_shape: tuple,
    cfg: Optional[dict] = None,
) -> tf.keras.Model:
    """
    Multi-layer LSTM for binary direction classification.
    Output: sigmoid probability of upward movement.
    """
    cfg = cfg or _load_cfg()
    lr = cfg.get("training", {}).get("learning_rate", 0.001)

    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.LSTM(256, return_sequences=True),
            layers.Dropout(0.3),
            layers.LSTM(128, return_sequences=True),
            layers.Dropout(0.25),
            layers.LSTM(64),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="sigmoid"),  # Direction probability
        ],
        name="classification_lstm",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    logger.debug("Built classification LSTM: input=%s params=%d", input_shape, model.count_params())
    return model


# ──────────────────────────────────────────────────────────────────────────
# Regression Model (Next Close Price)
# ──────────────────────────────────────────────────────────────────────────

def build_regression_lstm(
    input_shape: tuple,
    cfg: Optional[dict] = None,
) -> tf.keras.Model:
    """
    LSTM for regression — predicts the next close price.
    Output: single linear unit (raw price).
    """
    cfg = cfg or _load_cfg()
    lr = cfg.get("training", {}).get("learning_rate", 0.001)

    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.LSTM(128, return_sequences=True),
            layers.Dropout(0.3),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.25),
            layers.LSTM(32),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="linear"),  # Predicted price
        ],
        name="regression_lstm",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=["mae"],
    )
    return model


# ──────────────────────────────────────────────────────────────────────────
# Hybrid Model (Dual-Head: Direction + Price)
# ──────────────────────────────────────────────────────────────────────────

def build_hybrid_lstm(
    input_shape: tuple,
    cfg: Optional[dict] = None,
) -> tf.keras.Model:
    """
    Dual-head hybrid LSTM.

    Shared encoder → two output heads:
    - direction_output: sigmoid (BUY=1 / SELL=0 probability)
    - price_output:     linear  (predicted next close price)

    This is the primary production model used for inference.
    """
    cfg = cfg or _load_cfg()
    lr = cfg.get("training", {}).get("learning_rate", 0.001)

    # ── Shared Encoder ──────────────────────────────────────────────────
    inputs = layers.Input(shape=input_shape, name="ohlcv_sequence")

    x = layers.LSTM(256, return_sequences=True, name="lstm_1")(inputs)
    x = layers.Dropout(0.3, name="drop_1")(x)
    x = layers.LSTM(128, return_sequences=True, name="lstm_2")(x)
    x = layers.Dropout(0.25, name="drop_2")(x)
    x = layers.LSTM(64, name="lstm_3")(x)
    shared = layers.Dense(64, activation="relu", name="shared_dense")(x)

    # ── Direction Head ───────────────────────────────────────────────────
    dir_branch = layers.Dense(32, activation="relu", name="dir_hidden")(shared)
    dir_branch = layers.Dropout(0.2, name="dir_drop")(dir_branch)
    direction_output = layers.Dense(1, activation="sigmoid", name="direction_output")(dir_branch)

    # ── Price Head ────────────────────────────────────────────────────────
    price_branch = layers.Dense(32, activation="relu", name="price_hidden")(shared)
    price_branch = layers.Dropout(0.2, name="price_drop")(price_branch)
    price_output = layers.Dense(1, activation="linear", name="price_output")(price_branch)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=[direction_output, price_output],
        name="hybrid_lstm",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss={
            "direction_output": "binary_crossentropy",
            "price_output": "mse",
        },
        loss_weights={
            "direction_output": 0.6,
            "price_output": 0.4,
        },
        metrics={
            "direction_output": ["accuracy"],
            "price_output": ["mae"],
        },
    )

    logger.debug(
        "Built hybrid LSTM: input=%s total_params=%d",
        input_shape,
        model.count_params(),
    )
    return model


# ──────────────────────────────────────────────────────────────────────────
# Legacy compatibility: GlobalLSTMModel (used by old code paths)
# ──────────────────────────────────────────────────────────────────────────

class GlobalLSTMModel:
    """Backward-compatible wrapper — delegates to build_classification_lstm."""

    @staticmethod
    def build(input_shape: tuple, lr: float = 0.001) -> tf.keras.Model:
        import yaml, os
        cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
        try:
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
        except Exception:
            cfg = {"training": {"learning_rate": lr}}
        cfg["training"]["learning_rate"] = lr
        return build_classification_lstm(input_shape, cfg=cfg)


# ──────────────────────────────────────────────────────────────────────────
# XGBoost Hybrid (Volatile regime fallback)
# ──────────────────────────────────────────────────────────────────────────

class XGBoostHybrid:
    """XGBoost classifier for high-volatility regimes."""

    def __init__(self, params: Optional[dict] = None):
        import xgboost as xgb
        self.params = params or {
            "objective": "binary:logistic",
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 200,
            "eval_metric": "logloss",
            "use_label_encoder": False,
        }
        self.model = xgb.XGBClassifier(**self.params)

    def train(self, X_train, y_train, X_val=None, y_val=None):
        if X_val is not None:
            self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            self.model.fit(X_train, y_train)

    def predict_proba(self, X) -> float:
        return self.model.predict_proba(X)[0][1]

    def save(self, path: str) -> None:
        self.model.save_model(path)

    def load(self, path: str) -> None:
        import xgboost as xgb
        self.model = xgb.XGBClassifier()
        self.model.load_model(path)


class ModelFactory:
    """Utility for scaler persistence (backward compat)."""

    @staticmethod
    def save_scaler(scaler, path: str) -> None:
        import joblib
        joblib.dump(scaler, path)

    @staticmethod
    def load_scaler(path: str):
        import joblib
        return joblib.load(path)
