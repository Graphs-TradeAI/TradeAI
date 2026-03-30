"""
TradeAI — Modular Training Pipeline
Orchestrates: data load → feature engineering → preprocessing → train → evaluate → save
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from ..data_layer.loader import DataLoader
from ..feature_engineering.indicators import build_features, get_feature_columns
from ..forex_models.architectures import build_hybrid_lstm
from .evaluator import Evaluator
from .preprocessor import Preprocessor

logger = logging.getLogger(__name__)


def _load_cfg() -> dict:
    import yaml
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


class ForexTrainer:
    """
    Modular training orchestrator for per-(pair, timeframe) LSTM models.

    Usage
    -----
    trainer = ForexTrainer(api_key="...", models_root="...")
    metrics = trainer.train(symbol="EUR/USD", timeframe="1h")
    """

    def __init__(
        self,
        api_key: str,
        models_root: Optional[str] = None,
        cfg: Optional[dict] = None,
        use_cache: bool = True,
    ):
        self.cfg = cfg or _load_cfg()
        self.api_key = api_key
        train_cfg = self.cfg.get("training", {})
        model_cfg = self.cfg.get("models", {})

        # Resolve models root
        if models_root is None:
            project_root = self._project_root()
            models_root = os.path.join(project_root, model_cfg.get("root", "MLmodels/Forex/forex_models"))
        self.models_root = models_root

        # Hyper-params from config
        self.seq_length: int = train_cfg.get("sequence_length", 60)
        self.epochs: int = train_cfg.get("epochs", 50)
        self.batch_size: int = train_cfg.get("batch_size", 64)
        self.patience: int = train_cfg.get("patience", 7)
        self.output_size: int = train_cfg.get("output_size", 5000)

        # Sub-components
        self.loader = DataLoader(api_key, self.cfg, use_cache=use_cache)
        self.preprocessor = Preprocessor(models_root, self.cfg)
        self.evaluator = Evaluator()

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def train(
        self,
        symbol: str = "EUR/USD",
        timeframe: str = "1h",
        lookback_days: int = 365,
    ) -> Dict:
        """
        Full training pipeline for a single (symbol, timeframe).

        Returns
        -------
        dict : evaluation metrics + model save path
        """
        logger.info("═" * 60)
        logger.info("Training: %s  %s", symbol, timeframe)
        logger.info("═" * 60)

        # 1. Load data
        df_raw = self.loader.load(symbol, timeframe, lookback_days=lookback_days)
        logger.info("Loaded %d rows of raw OHLCV data", len(df_raw))

        # 2. Feature engineering
        df_feat = build_features(df_raw, symbol=symbol, timeframe=timeframe, cfg=self.cfg)
        feature_cols = get_feature_columns(df_feat)
        logger.info("Feature set: %d cols", len(feature_cols))

        # 3. Preprocessing
        scaled = self.preprocessor.fit_transform(df_feat, feature_cols)
        y_dir = df_feat["target_direction"].values
        y_price = df_feat["target_price"].values

        X, y_dir_seq, y_price_seq = self.preprocessor.create_sequences(
            scaled, y_dir, y_price
        )
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = \
            self.preprocessor.walk_forward_split(X, y_dir_seq)

        _, _, (_, y_price_test) = self.preprocessor.walk_forward_split(X, y_price_seq)

        logger.info(
            "Split — train: %d | val: %d | test: %d", len(X_train), len(X_val), len(X_test)
        )

        # 4. Build model
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = build_hybrid_lstm(input_shape, cfg=self.cfg)

        # 5. Train
        save_dir = self._model_dir(symbol, timeframe)
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, self.cfg["models"].get("model_filename", "model.keras"))

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=self._callbacks(model_path),
            verbose=1,
        )

        # Reload best checkpoint
        model = tf.keras.models.load_model(model_path)

        # 6. Evaluate
        y_prob_test = model.predict(X_test, verbose=0).flatten()

        # Get test-set prices for Sharpe
        test_start = len(X_train) + len(X_val) + self.seq_length
        test_prices = df_feat["close"].values[test_start: test_start + len(X_test)]

        annualise = Evaluator.annualise_factor_for_timeframe(timeframe)
        metrics = self.evaluator.evaluate_classification(
            y_test, y_prob_test,
            prices=test_prices,
            annualise_factor=annualise,
        )
        metrics["model_path"] = model_path
        metrics["symbol"] = symbol
        metrics["timeframe"] = timeframe
        metrics["n_features"] = len(feature_cols)
        metrics["n_train"] = len(X_train)

        # 7. Save scaler + metrics
        self.preprocessor.save_scaler(symbol, timeframe)
        self._save_metrics(metrics, symbol, timeframe)

        logger.info(
            "Done — Dir Acc: %.3f | Win Rate: %.3f | Sharpe: %.3f",
            metrics["directional_accuracy"],
            metrics["win_rate"],
            metrics["sharpe_ratio"],
        )
        return metrics

    def train_all(
        self,
        pairs: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        lookback_days: int = 365,
    ) -> List[Dict]:
        """Train models for all (pair, timeframe) combinations."""
        pairs = pairs or self.cfg.get("pairs", [])
        timeframes = timeframes or self.cfg.get("timeframes", [])
        results = []
        for symbol in pairs:
            for tf in timeframes:
                try:
                    result = self.train(symbol, tf, lookback_days)
                    results.append(result)
                except Exception as exc:
                    logger.error("Failed %s %s: %s", symbol, tf, exc, exc_info=True)
                    results.append({"symbol": symbol, "timeframe": tf, "error": str(exc)})
        return results

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    def _callbacks(self, model_path: str) -> list:
        train_cfg = self.cfg.get("training", {})
        return [
            ModelCheckpoint(
                model_path,
                monitor="val_loss",
                save_best_only=True,
                verbose=0,
            ),
            EarlyStopping(
                monitor="val_loss",
                patience=train_cfg.get("patience", 7),
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1,
            ),
        ]

    def _model_dir(self, symbol: str, timeframe: str) -> str:
        pair_tag = symbol.replace("/", "")
        return os.path.join(self.models_root, pair_tag, timeframe)

    def _save_metrics(self, metrics: dict, symbol: str, timeframe: str) -> None:
        path = os.path.join(
            self._model_dir(symbol, timeframe),
            self.cfg["models"].get("metrics_filename", "metrics.json"),
        )
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Metrics saved to %s", path)

    @staticmethod
    def _project_root() -> str:
        # MLmodels/Forex/training/ → MLmodels/Forex/ → MLmodels/ → project root
        return os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        )


# ──────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from decouple import config as env

    parser = argparse.ArgumentParser(description="Train a ForexTrader model")
    parser.add_argument("--symbol", type=str, default="EUR/USD")
    parser.add_argument("--timeframe", type=str, default="1h")
    parser.add_argument("--lookback", type=int, default=365)
    parser.add_argument("--all", action="store_true", help="Train all pairs & timeframes")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    api_key = env("TWELVE_DATA_API_KEY")
    trainer = ForexTrainer(api_key=api_key)

    if args.all:
        results = trainer.train_all(lookback_days=args.lookback)
        for r in results:
            print(json.dumps(r, indent=2))
    else:
        result = trainer.train(args.symbol, args.timeframe, args.lookback)
        print(json.dumps(result, indent=2))
