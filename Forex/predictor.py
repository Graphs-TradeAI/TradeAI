
from __future__ import annotations

import logging
import os
import hashlib
from typing import Optional

import numpy as np

from Forex.loader import DataLoader
from Forex.indicators import build_features, get_feature_columns, get_indicator_snapshot
from Forex.registry import ModelNotFoundError, ModelRegistry

logger = logging.getLogger(__name__)


def _load_cfg() -> dict:
    import yaml
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


class ForexPredictor:

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
        symbol: str,
        timeframe: str,
        output_size: int = 500,
    ) -> dict:
      
        logger.info("Predicting %s %s", symbol, timeframe)

        # 1. Fetch recent data
        df_raw = self.loader.load_for_inference(symbol, timeframe, output_size=output_size)
        if len(df_raw) < self.seq_length + 60:
            raise ValueError(
                f"Insufficient data: got {len(df_raw)} rows, need > {self.seq_length + 60}"
            )

        # 2. Feature engineering (USE TRAINING FEATURE PIPELINE)
        df_feat = build_features(df_raw, symbol=symbol, timeframe=timeframe)
        # Feature columns: strict training contract when available.
        metrics = self.registry.load_metrics(symbol, timeframe) or {}
        expected_feature_cols = metrics.get("feature_columns")
        if isinstance(expected_feature_cols, list) and expected_feature_cols:
            missing = [c for c in expected_feature_cols if c not in df_feat.columns]
            if missing:
                raise ValueError(
                    f"Missing required training features for {symbol} {timeframe}: {missing}"
                )
            feature_cols = expected_feature_cols
        else:
            feature_cols = get_feature_columns(df_feat)

        try:
            model, source = self.registry.load_model(symbol, timeframe)
        except ModelNotFoundError as exc:
            logger.error("Model not found: %s", exc)
            raise

        # Optional feature-contract validation against training metadata.
        expected_n_features = metrics.get("n_features")
        if expected_n_features is not None and int(expected_n_features) != len(feature_cols):
            raise ValueError(
                f"Feature count mismatch for {symbol} {timeframe}: "
                f"inference={len(feature_cols)} expected={expected_n_features}"
            )

        expected_hash = metrics.get("feature_columns_hash")
        if expected_hash:
            signature = "|".join(feature_cols)
            current_hash = hashlib.sha256(signature.encode("utf-8")).hexdigest()
            if current_hash != expected_hash:
                raise ValueError(
                    f"Feature contract hash mismatch for {symbol} {timeframe}. "
                    "Retrain model or align feature pipeline."
                )

        # 4. Build last sequence directly from engineered features (matches trainer)
        feature_matrix = df_feat[feature_cols].values
        if len(feature_matrix) < self.seq_length:
            raise ValueError(f"Not enough rows ({len(feature_matrix)}) for seq_length={self.seq_length}")

        last_seq = feature_matrix[-self.seq_length:]
        X = last_seq[np.newaxis, ...]

        # 5. Model inference (supports legacy single-output and new multi-output models)
        raw_output = model.predict(X, verbose=0)

        pred_return, direction_probability = self._parse_model_output(raw_output)
        signal = "BUY" if pred_return > 0 else "SELL"
        if direction_probability is not None:
            signal = "BUY" if direction_probability >= 0.5 else "SELL"

        # Confidence score derived from magnitude of predicted return relative to typical volatility
        # Normalizing by ATR pct to get a sense of relative strength
        atr_pct = float(df_feat["atr_pct"].iloc[-1])
        reg_confidence = min(max(abs(pred_return) / (atr_pct + 1e-9), 0.0), 1.0)
        if direction_probability is not None:
            cls_confidence = min(max(abs(direction_probability - 0.5) * 2.0, 0.0), 1.0)
            confidence = round(0.6 * cls_confidence + 0.4 * reg_confidence, 4)
        else:
            confidence = reg_confidence
        
        # Predicted price = current price * exp(pred_return)
        current_price = float(df_feat["close"].iloc[-1])
        pred_price = current_price * np.exp(pred_return)

        # Snap signal to HOLD for low-confidence predictions
        risk_cfg = self.cfg.get("risk", {})
        hold_threshold = risk_cfg.get(
            "hold_confidence_threshold",
            risk_cfg.get("min_confidence_threshold", 0.55),
        )
        if confidence < hold_threshold:
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
            "confidence": round(confidence, 4),
            "direction_probability": (
                round(float(direction_probability), 4)
                if direction_probability is not None
                else None
            ),
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

    @staticmethod
    def _parse_model_output(raw_output) -> tuple[float, Optional[float]]:
        """
        Parse keras predict output.
        Returns:
            (predicted_log_return, direction_probability_or_none)
        """
        # New multi-output models may return dict keyed by output names.
        if isinstance(raw_output, dict):
            reg = raw_output.get("return_head")
            cls = raw_output.get("direction_head")
            pred_return = float(np.ravel(reg)[0]) if reg is not None else 0.0
            direction_probability = float(np.ravel(cls)[0]) if cls is not None else None
            if direction_probability is not None:
                direction_probability = min(max(direction_probability, 0.0), 1.0)
            return pred_return, direction_probability

        # Multi-output list/tuple: [return_head, direction_head]
        if isinstance(raw_output, (list, tuple)):
            if len(raw_output) == 0:
                raise ValueError("Empty model prediction output")
            pred_return = float(np.ravel(raw_output[0])[0])
            direction_probability = None
            if len(raw_output) >= 2 and raw_output[1] is not None:
                direction_probability = float(np.ravel(raw_output[1])[0])
                direction_probability = min(max(direction_probability, 0.0), 1.0)
            return pred_return, direction_probability

        # Legacy single-output model.
        pred_return = float(np.ravel(raw_output)[0])
        return pred_return, None

    def predict_multi_timeframe(
        self,
        symbol: str,
        timeframes: Optional[list] = None,
        output_size: int = 500,
    ) -> dict:
        """
        Predict across multiple timeframes for the same symbol.
        Useful for multi-timeframe conflict detection in risk management.
        """
        timeframes = timeframes or self.cfg.get("timeframes", [])
        results = {}
        for tf in timeframes:
            try:
                results[tf] = self.predict(
                    symbol=symbol,
                    timeframe=tf,
                    output_size=output_size,
                )
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
