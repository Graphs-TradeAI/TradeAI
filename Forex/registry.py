from __future__ import annotations
import json
import logging
import os
from typing import Optional, Tuple

import joblib

logger = logging.getLogger(__name__)


def _load_cfg() -> dict:
    import yaml

    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")

    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


class ModelNotFoundError(Exception):
    pass


class ModelRegistry:
    """
    Model layout:

    forex/models/
        EURUSD/
            1h/
                model.keras
                scaler.pkl
                metrics.json
        global_base.keras
        global_scaler.pkl
    """

    def __init__(self, models_root: Optional[str] = None, cfg: Optional[dict] = None):
        self.cfg = cfg or _load_cfg()
        model_cfg = self.cfg.get("models", {})

        # ─────────────────────────────────────────────
        # FIX: enforce forex/models as default root
        # ─────────────────────────────────────────────
        if models_root is None:
            project_root = self._project_root()
            models_root = os.path.join(project_root, "Forex", "Models")

        self.models_root = models_root

        self.model_filename = model_cfg.get("model_filename", "model.keras")
        self.scaler_filename = model_cfg.get("scaler_filename", "scaler.pkl")
        self.global_model = model_cfg.get("global_base_filename", "global_base.keras")
        self.global_scaler = model_cfg.get("global_scaler_filename", "global_scaler.pkl")

    # ─────────────────────────────────────────────
    # MODEL LOADING
    # ─────────────────────────────────────────────

    def load_model(self, symbol: str, timeframe: str) -> Tuple[object, str]:
        import tensorflow as tf
        pair_path = self._pair_model_path(symbol, timeframe)
        global_path = os.path.join(self.models_root, self.global_model)

        if os.path.exists(pair_path):
            logger.info("Loading model: %s", pair_path)
            return tf.keras.models.load_model(pair_path), f"pair ({symbol} {timeframe})"

        if os.path.exists(global_path):
            logger.warning("Falling back to global model")
            return tf.keras.models.load_model(global_path), "global"

        raise ModelNotFoundError(f"Missing model for {symbol} {timeframe}")

    # ─────────────────────────────────────────────

    def load_scaler(self, symbol: str, timeframe: str):
        pair = self._pair_scaler_path(symbol, timeframe)
        global_scaler = os.path.join(self.models_root, self.global_scaler)

        if os.path.exists(pair):
            return joblib.load(pair)

        if os.path.exists(global_scaler):
            logger.warning("Falling back to global scaler")
            return joblib.load(global_scaler)

        raise ModelNotFoundError(f"Missing scaler for {symbol} {timeframe}")

    # ─────────────────────────────────────────────

    def load_metrics(self, symbol: str, timeframe: str) -> Optional[dict]:
        path = os.path.join(
            self._pair_dir(symbol, timeframe),
            self.cfg["models"].get("metrics_filename", "metrics.json"),
        )

        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)

        return None

    def save_metrics(self, symbol: str, timeframe: str, metrics: dict) -> str:
        path = os.path.join(
            self._pair_dir(symbol, timeframe),
            self.cfg["models"].get("metrics_filename", "metrics.json"),
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Saved metrics: %s", path)
        return path

    def model_exists(self, symbol: str, timeframe: str) -> bool:
        return os.path.exists(self._pair_model_path(symbol, timeframe))

    # ─────────────────────────────────────────────
    # LIST MODELS
    # ─────────────────────────────────────────────

    def list_available(self) -> list[dict]:
        results = []

        for symbol in self.cfg.get("pairs", []):
            for tf in self.cfg.get("timeframes", []):
                if self.model_exists(symbol, tf):
                    metrics = self.load_metrics(symbol, tf) or {}
                    results.append({
                        "symbol": symbol,
                        "timeframe": tf,
                        "path": self._pair_model_path(symbol, tf),
                        "directional_accuracy": metrics.get("directional_accuracy"),
                        "sharpe_ratio": metrics.get("sharpe_ratio"),
                        "mae": metrics.get("mae"),
                    })

        return results

    def select_best_available(self) -> Optional[dict]:
        models = self.list_available()
        if not models:
            return None
        # Prefer sharpe, then directional accuracy, then lower MAE.
        def _score_key(item: dict):
            sharpe = item.get("sharpe_ratio")
            da = item.get("directional_accuracy")
            mae = item.get("mae")
            return (
                float(sharpe) if sharpe is not None else float("-inf"),
                float(da) if da is not None else float("-inf"),
                -float(mae) if mae is not None else float("-inf"),
            )

        return sorted(models, key=_score_key, reverse=True)[0]

    # ─────────────────────────────────────────────
    # ── PATH HELPERS ──────────────────────────────────────────────────────────

    def _pair_dir(self, symbol: str, timeframe: str) -> str:
        """Standardized directory: Models/EURUSD/1h/"""
        # Normalize symbol: EUR/USD -> EURUSD
        pair_tag = symbol.replace("/", "").upper()
        # Normalize timeframe: 1H -> 1h
        tf_tag = timeframe.lower()
        return os.path.join(self.models_root, pair_tag, tf_tag)

    def _pair_model_path(self, symbol: str, timeframe: str) -> str:
        return os.path.join(self._pair_dir(symbol, timeframe), self.model_filename)

    def _pair_scaler_path(self, symbol: str, timeframe: str) -> str:
        return os.path.join(self._pair_dir(symbol, timeframe), self.scaler_filename)

    @staticmethod
    def _project_root() -> str:
        """Find the TradeAI project root."""
        # Current file: /home/.../TradeAI/Forex/registry.py
        current_file = os.path.abspath(__file__)
        # dirname: /home/.../TradeAI/Forex/
        # parent dirname: /home/.../TradeAI/
        return os.path.dirname(os.path.dirname(current_file))
