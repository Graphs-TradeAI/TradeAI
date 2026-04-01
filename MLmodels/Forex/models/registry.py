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
    pass


class ModelRegistry:
    def __init__(self, models_root: Optional[str] = None, cfg: Optional[dict] = None):
        self.cfg = cfg or _load_cfg()
        model_cfg = self.cfg.get("models", {})

        if models_root is None:
            project_root = self._project_root()
            models_root = os.path.join(
                project_root, model_cfg.get("root", "MLmodels/Forex/forex_models")
            )

        self.models_root = models_root

        self.model_filename = model_cfg.get("model_filename", "model.keras")
        self.scaler_filename = model_cfg.get("scaler_filename", "scaler.pkl")
        self.metadata_filename = model_cfg.get("metadata_filename", "metadata.pkl")
        self.metrics_filename = model_cfg.get("metrics_filename", "metrics.json")

        self.global_model = model_cfg.get("global_base_filename", "global_base.keras")
        self.global_scaler = model_cfg.get("global_scaler_filename", "global_scaler.pkl")

        # In-memory cache
        self._model_cache = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_bundle(self, symbol: str, timeframe: str):
        """
        Load model + scaler + metadata as a single bundle.
        """
        cache_key = f"{symbol}_{timeframe}"

        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        model, source = self.load_model(symbol, timeframe)
        scaler = self.load_scaler(symbol, timeframe)
        metadata = self.load_metadata(symbol, timeframe)
        metrics = self.load_metrics(symbol, timeframe)

        bundle = {
            "model": model,
            "scaler": scaler,
            "metadata": metadata,
            "metrics": metrics,
            "source": source,
        }

        self._model_cache[cache_key] = bundle
        return bundle

    def load_model(self, symbol: str, timeframe: str) -> Tuple[tf.keras.Model, str]:
        pair_path = self._pair_model_path(symbol, timeframe)
        global_path = os.path.join(self.models_root, self.global_model)

        if os.path.exists(pair_path):
            model = self._safe_load_model(pair_path)
            return model, "pair-specific"

        if os.path.exists(global_path):
            logger.warning(f"Fallback to global model for {symbol} {timeframe}")
            model = self._safe_load_model(global_path)
            return model, "global"

        raise ModelNotFoundError(f"No model found for {symbol} {timeframe}")

    def load_scaler(self, symbol: str, timeframe: str):
        pair_path = self._pair_scaler_path(symbol, timeframe)
        global_path = os.path.join(self.models_root, self.global_scaler)

        if os.path.exists(pair_path):
            return joblib.load(pair_path)

        if os.path.exists(global_path):
            logger.warning(f"Fallback to global scaler for {symbol} {timeframe}")
            return joblib.load(global_path)

        raise ModelNotFoundError("No scaler found")

    def load_metadata(self, symbol: str, timeframe: str) -> Optional[dict]:
        path = os.path.join(self._pair_dir(symbol, timeframe), self.metadata_filename)
        if os.path.exists(path):
            return joblib.load(path)
        return None

    def load_metrics(self, symbol: str, timeframe: str) -> Optional[dict]:
        path = os.path.join(self._pair_dir(symbol, timeframe), self.metrics_filename)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_features(self, df_columns, metadata):
        """
        Ensure inference features match training features.
        """
        expected = set(metadata.get("features", []))
        actual = set(df_columns)

        missing = expected - actual
        extra = actual - expected

        if missing:
            raise ValueError(f"Missing features: {missing}")

        if extra:
            logger.warning(f"Extra features ignored: {extra}")

    def validate_model_quality(self, metrics: dict):
        """
        Reject weak models.
        """
        if not metrics:
            return True

        min_acc = self.cfg.get("models", {}).get("min_directional_accuracy", 0.52)
        min_sharpe = self.cfg.get("models", {}).get("min_sharpe_ratio", 0.1)

        acc = metrics.get("directional_accuracy", 0)
        sharpe = metrics.get("sharpe_ratio", 0)

        if acc < min_acc or sharpe < min_sharpe:
            logger.warning(
                f"Model rejected: acc={acc:.3f}, sharpe={sharpe:.3f}"
            )
            return False

        return True

    # ------------------------------------------------------------------
    # Safe Loading
    # ------------------------------------------------------------------

    def _safe_load_model(self, path):
        try:
            return tf.keras.models.load_model(path, compile=False)
        except Exception as e:
            logger.error(f"Failed to load model: {path}")
            raise e

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    def _pair_dir(self, symbol: str, timeframe: str):
        return os.path.join(self.models_root, symbol.replace("/", ""), timeframe)

    def _pair_model_path(self, symbol: str, timeframe: str):
        return os.path.join(self._pair_dir(symbol, timeframe), self.model_filename)

    def _pair_scaler_path(self, symbol: str, timeframe: str):
        return os.path.join(self._pair_dir(symbol, timeframe), self.scaler_filename)

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------

    @staticmethod
    def _project_root():
        return os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        )