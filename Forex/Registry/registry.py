"""
TradeAI — Model Registry
Dynamic model/scaler loading per (symbol, timeframe) with fallback handling.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional, Tuple

import joblib
import tensorflow as tf

logger = logging.getLogger(__name__)


def _load_cfg() -> dict:
    import yaml
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


class ModelNotFoundError(Exception):
    """Raised when no trained model exists for a (symbol, timeframe) pair."""
    pass


class ModelRegistry:
    """
    Dynamic model registry for forex prediction models.

    Resolution order for model_path:
      1. forexmodels/{PAIR}/{timeframe}/model.keras  (finetuned pair-specific)
      2. forexmodels/global_base.keras               (global fallback)
      → ModelNotFoundError if neither exists

    Resolution order for scaler:
      1. forexmodels/{PAIR}/{timeframe}/scaler.pkl
      2. forexmodels/global_scaler.pkl
      → ModelNotFoundError if neither exists
    """

    def __init__(self, models_root: Optional[str] = None, cfg: Optional[dict] = None):
        self.cfg = cfg or _load_cfg()
        model_cfg = self.cfg.get("models", {})

        if models_root is None:
            project_root = self._project_root()
            models_root = os.path.join(
                project_root, model_cfg.get("root", "MLmodels/Forex/forex_models")
            )
        self.models_root = models_root

        self.model_filename: str = model_cfg.get("model_filename", "model.keras")
        self.scaler_filename: str = model_cfg.get("scaler_filename", "scaler.pkl")
        self.global_model: str = model_cfg.get("global_base_filename", "global_base.keras")
        self.global_scaler: str = model_cfg.get("global_scaler_filename", "global_scaler.pkl")

    # ── Model Loading ──────────────────────────────────────────────────────

    def load_model(self, symbol: str, timeframe: str) -> Tuple[tf.keras.Model, str]:
        """
        Load the best available model for (symbol, timeframe).

        Returns
        -------
        (model, source_label) where source_label indicates which model was loaded.

        Raises
        ------
        ModelNotFoundError if no model found at any path.
        """
        pair_path = self._pair_model_path(symbol, timeframe)
        global_path = os.path.join(self.models_root, self.global_model)

        if os.path.exists(pair_path):
            logger.info("Loading pair-specific model: %s", pair_path)
            model = tf.keras.models.load_model(pair_path)
            return model, f"pair-specific ({symbol} {timeframe})"

        if os.path.exists(global_path):
            logger.warning(
                "No model for %s %s. Falling back to global base model.", symbol, timeframe
            )
            model = tf.keras.models.load_model(global_path)
            return model, "global-base (fallback)"

        raise ModelNotFoundError(
            f"No trained model found for {symbol} {timeframe}.\n"
            f"  Looked for: {pair_path}\n"
            f"  Fallback:   {global_path}\n"
            "Run: python -m MLmodels.Forex.training.trainer --symbol '{symbol}' --timeframe '{timeframe}'"
        )

    def load_scaler(self, symbol: str, timeframe: str):
        """
        Load the scaler for (symbol, timeframe).

        Raises
        ------
        ModelNotFoundError if no scaler found.
        """
        pair_scaler = self._pair_scaler_path(symbol, timeframe)
        global_scaler = os.path.join(self.models_root, self.global_scaler)

        if os.path.exists(pair_scaler):
            logger.debug("Loading pair scaler: %s", pair_scaler)
            return joblib.load(pair_scaler)

        if os.path.exists(global_scaler):
            logger.warning(
                "No scaler for %s %s. Falling back to global scaler.", symbol, timeframe
            )
            return joblib.load(global_scaler)

        raise ModelNotFoundError(
            f"No scaler found for {symbol} {timeframe}.\n"
            f"  Looked for: {pair_scaler}\n"
            f"  Fallback:   {global_scaler}"
        )

    def load_metrics(self, symbol: str, timeframe: str) -> Optional[dict]:
        """Load saved evaluation metrics if available."""
        path = os.path.join(
            self._pair_dir(symbol, timeframe),
            self.cfg["models"].get("metrics_filename", "metrics.json"),
        )
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return None

    def model_exists(self, symbol: str, timeframe: str) -> bool:
        """Return True if a pair-specific model exists."""
        return os.path.exists(self._pair_model_path(symbol, timeframe))

    def list_available(self) -> list[dict]:
        """List all trained (pair, timeframe) models in the registry."""
        available = []
        supported_pairs = self.cfg.get("pairs", [])
        supported_tfs = self.cfg.get("timeframes", [])
        for symbol in supported_pairs:
            for tf in supported_tfs:
                if self.model_exists(symbol, tf):
                    metrics = self.load_metrics(symbol, tf) or {}
                    available.append({
                        "symbol": symbol,
                        "timeframe": tf,
                        "path": self._pair_model_path(symbol, tf),
                        "directional_accuracy": metrics.get("directional_accuracy"),
                        "sharpe_ratio": metrics.get("sharpe_ratio"),
                    })
        return available

    # ── Path helpers ──────────────────────────────────────────────────────

    def _pair_dir(self, symbol: str, timeframe: str) -> str:
        pair_tag = symbol.replace("/", "")
        return os.path.join(self.models_root, pair_tag, timeframe)

    def _pair_model_path(self, symbol: str, timeframe: str) -> str:
        return os.path.join(self._pair_dir(symbol, timeframe), self.model_filename)

    def _pair_scaler_path(self, symbol: str, timeframe: str) -> str:
        return os.path.join(self._pair_dir(symbol, timeframe), self.scaler_filename)

    @staticmethod
    def _project_root() -> str:
        # models/registry.py → models/ → Forex/ → MLmodels/ → project root
        return os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        )
