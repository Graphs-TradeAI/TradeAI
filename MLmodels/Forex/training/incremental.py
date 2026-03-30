"""
TradeAI — Incremental Retraining
Updates an existing model on new data without a full retrain.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping

from ..data_layer.loader import DataLoader
from ..feature_engineering.indicators import build_features, get_feature_columns
from .evaluator import Evaluator
from .preprocessor import Preprocessor

logger = logging.getLogger(__name__)


def _load_cfg() -> dict:
    import yaml
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


class IncrementalTrainer:
    """
    Update an existing trained model with new data only.

    Strategy
    --------
    1. Load the saved model + scaler for (symbol, timeframe)
    2. Apply the existing scaler transform to new data (no re-fitting)
    3. Fine-tune for a few epochs at a lower learning rate
    4. Save updated model + metrics

    This avoids full-dataset retraining while letting the model
    adapt to recent market conditions.
    """

    def __init__(
        self,
        api_key: str,
        models_root: Optional[str] = None,
        cfg: Optional[dict] = None,
    ):
        self.cfg = cfg or _load_cfg()
        self.api_key = api_key
        model_cfg = self.cfg.get("models", {})

        if models_root is None:
            project_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            )
            models_root = os.path.join(
                project_root, model_cfg.get("root", "MLmodels/Forex/forex_models")
            )
        self.models_root = models_root
        self.loader = DataLoader(api_key, self.cfg)
        self.evaluator = Evaluator()

    # ──────────────────────────────────────────────────────────────────────

    def retrain(
        self,
        symbol: str = "EUR/USD",
        timeframe: str = "1h",
        lookback_days: int = 90,
        epochs: int = 10,
        learning_rate: float = 1e-4,
        new_df: Optional[pd.DataFrame] = None,
    ) -> dict:
        """
        Incrementally retrain model for (symbol, timeframe).

        Parameters
        ----------
        symbol        : Currency pair
        timeframe     : Timeframe
        lookback_days : Days of recent data to use (if new_df not provided)
        epochs        : Max fine-tuning epochs
        learning_rate : Lower than initial training LR
        new_df        : Pre-fetched DataFrame (optional, skips API call)

        Returns
        -------
        dict : Updated evaluation metrics
        """
        pair_tag = symbol.replace("/", "")
        model_dir = os.path.join(self.models_root, pair_tag, timeframe)
        model_path = os.path.join(
            model_dir, self.cfg["models"].get("model_filename", "model.keras")
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No trained model found for {symbol} {timeframe} at {model_path}. "
                "Run ForexTrainer.train() first."
            )

        logger.info("Incremental retrain: %s %s", symbol, timeframe)

        # 1. Load existing model
        model = tf.keras.models.load_model(model_path)

        # 2. Load existing scaler
        preprocessor = Preprocessor(self.models_root, self.cfg)
        preprocessor.load_scaler(symbol, timeframe)

        # 3. Get new data
        if new_df is None:
            new_df = self.loader.load(
                symbol, timeframe,
                lookback_days=lookback_days,
                force_refresh=True,   # Always get fresh data
            )

        # 4. Feature engineering (using same pipeline)
        df_feat = build_features(new_df, symbol=symbol, timeframe=timeframe, cfg=self.cfg)
        feature_cols = get_feature_columns(df_feat)

        # 5. Transform (no re-fit — preserve scaler from original training)
        scaled = preprocessor.transform(df_feat, feature_cols)
        y_dir = df_feat["target_direction"].values

        X, y_dir_seq, _ = preprocessor.create_sequences(scaled, y_dir)

        if len(X) < 50:
            raise ValueError(
                f"Insufficient new data ({len(X)} sequences) for incremental retrain."
            )

        split = int(len(X) * 0.85)
        X_train, y_train = X[:split], y_dir_seq[:split]
        X_val, y_val = X[split:], y_dir_seq[split:]

        # 6. Recompile at lower LR + fine-tune
        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
            ],
            verbose=1,
        )

        # 7. Save updated model
        model.save(model_path)
        logger.info("Updated model saved to %s", model_path)

        # 8. Evaluate
        y_prob = model.predict(X_val, verbose=0).flatten()
        annualise = Evaluator.annualise_factor_for_timeframe(timeframe)
        metrics = self.evaluator.evaluate_classification(
            y_val, y_prob, annualise_factor=annualise
        )
        metrics.update({"symbol": symbol, "timeframe": timeframe, "mode": "incremental"})

        # Persist metrics
        metrics_path = os.path.join(
            model_dir, self.cfg["models"].get("metrics_filename", "metrics.json")
        )
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(
            "Incremental retrain done — Dir Acc: %.3f | Sharpe: %.3f",
            metrics["directional_accuracy"], metrics["sharpe_ratio"],
        )
        return metrics


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from decouple import config as env

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Incrementally retrain a forex model")
    parser.add_argument("--symbol", type=str, default="EUR/USD")
    parser.add_argument("--timeframe", type=str, default="1h")
    parser.add_argument("--lookback", type=int, default=90, help="Days of recent data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    trainer = IncrementalTrainer(api_key=env("TWELVE_DATA_API_KEY"))
    result = trainer.retrain(
        symbol=args.symbol,
        timeframe=args.timeframe,
        lookback_days=args.lookback,
        epochs=args.epochs,
        learning_rate=args.lr,
    )
    print(json.dumps(result, indent=2))
