
from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

from loader import DataLoader
from processing import build_forex_feature_set
from indicators import get_indicator_snapshot
from registry import ModelNotFoundError, ModelRegistry

logger = logging.getLogger(__name__)


def _load_cfg() -> dict:
    import yaml
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


class ForexPredictor:
    """
    End-to-end inference for a (symbol, timeframe) pair.

    Pipeline
    --------
    1. Fetch recent OHLCV via DataLoader
    2. Build features via feature_engineering
    3. Scale with saved per-pair scaler
    4. Create last sequence window
    5. Run model prediction
    6. Return structured PredictionResult dict
    """

    def __init__(
        self,
        api_key: str,
        models_root: Optional[str] = None,
        cfg: Optional[dict] = None,
    ):
        self.cfg = cfg or _load_cfg()
        self.api_key = api_key
        train_cfg = self.cfg.get("training", {})
        self.seq_length: int = train_cfg.get("sequence_length", 60)

        self.registry = ModelRegistry(models_root, self.cfg)
        self.loader = DataLoader(api_key, self.cfg, use_cache=False)

    # ──────────────────────────────────────────────────────────────────────

    def predict(
        self,
        symbol: str = "AUD/USD",
        timeframe: str = "1h",
        output_size: int = 500,
    ) -> dict:
        """
        Run a full prediction for (symbol, timeframe).

        Returns
        -------
        dict:
            symbol, timeframe, signal, confidence, predicted_price,
            current_price, regime, risk_level, indicators, timestamp
        """
        logger.info("Predicting %s %s", symbol, timeframe)

        # 1. Fetch recent data
        df_raw = self.loader.load_for_inference(symbol, timeframe, output_size=output_size)
        if len(df_raw) < self.seq_length + 60:
            raise ValueError(
                f"Insufficient data: got {len(df_raw)} rows, need > {self.seq_length + 60}"
            )

        # 2. Feature engineering (USE TRAINING FEATURE PIPELINE)
        df_feat = build_forex_feature_set(df_raw, symbol=symbol, timeframe=timeframe)
        # Feature columns: mirror trainer (exclude timestamp, target_price, target_return)
        feature_cols = [c for c in df_feat.columns if c not in ("timestamp", "target_price", "target_return")]

        # 3. Load model (training pipeline used no scaling)
        try:
            model, source = self.registry.load_model(symbol, timeframe)
        except ModelNotFoundError as exc:
            logger.error("Model not found: %s", exc)
            raise

        # 4. Build last sequence directly from engineered features (matches trainer)
        feature_matrix = df_feat[feature_cols].values
        if len(feature_matrix) < self.seq_length:
            raise ValueError(f"Not enough rows ({len(feature_matrix)}) for seq_length={self.seq_length}")

        last_seq = feature_matrix[-self.seq_length:]
        X = last_seq[np.newaxis, ...]

        # 5. Model inference
        raw_output = model.predict(X, verbose=0)

        # Handle both single-head and dual-head (hybrid) models
        if isinstance(raw_output, list):
            # Hybrid: [direction_prob, predicted_price]
            dir_prob = float(raw_output[0][0][0])
            pred_price = float(raw_output[1][0][0])
        else:
            # Classification-only
            dir_prob = float(raw_output[0][0])
            pred_price = float(df_feat["close"].iloc[-1])  # Fallback to current

        # 6. Derive signal and confidence
        signal = "BUY" if dir_prob > 0.5 else "SELL"
        # Confidence: how far the probability is from 0.5, scaled to [0,1]
        confidence = round(abs(dir_prob - 0.5) * 2, 4)

        # Snap signal to HOLD for low-confidence predictions
        if confidence < self.cfg.get("risk", {}).get("min_confidence_threshold", 0.55) - 0.5:
            signal = "HOLD"

        # 7. Risk level from confidence + regime
        last_row = df_feat.iloc[-1]
        regime_idx = int(last_row.get("regime", 0))
        regime_map = {0: "range", 1: "bull_trend", 2: "bear_trend", 3: "volatile"}
        regime = regime_map.get(regime_idx, "range")
        risk_level = self._compute_risk_level(confidence, regime)

        # 8. Indicator snapshot
        indicators = get_indicator_snapshot(df_feat)

        current_price = float(df_feat["close"].iloc[-1])
        timestamp = str(df_feat["timestamp"].iloc[-1])

        result = {
            "symbol": symbol,
            "timeframe": timeframe,
            "signal": signal,
            "confidence": confidence,
            "direction_probability": round(dir_prob, 4),
            "predicted_price": round(pred_price, 5),
            "current_price": round(current_price, 5),
            "regime": regime,
            "risk_level": risk_level,
            "indicators": indicators,
            "model_source": source,
            "timestamp": timestamp,
        }

        logger.info(
            "Prediction: %s %s → %s (conf=%.2f, price=%.5f)",
            symbol, timeframe, signal, confidence, pred_price,
        )
        return result

    def predict_multi_timeframe(
        self,
        symbol: str,
        timeframes: Optional[list] = None,
    ) -> dict:
        """
        Predict across multiple timeframes for the same symbol.
        Useful for multi-timeframe conflict detection in risk management.
        """
        timeframes = timeframes or self.cfg.get("timeframes", [])
        results = {}
        for tf in timeframes:
            try:
                results[tf] = self.predict(symbol, tf)
            except Exception as exc:
                logger.warning("Failed prediction for %s %s: %s", symbol, tf, exc)
                results[tf] = {"error": str(exc)}
        return results

    @staticmethod
    def _compute_risk_level(confidence: float, regime: str) -> str:
        """Compute risk level from confidence score and market regime."""
        if regime == "volatile":
            return "HIGH"
        if confidence >= 0.70:
            return "LOW"
        if confidence >= 0.50:
            return "MEDIUM"
        return "HIGH"
