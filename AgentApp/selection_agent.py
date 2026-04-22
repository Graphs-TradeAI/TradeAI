"""
TradeAI — Model Selection Agent (AgentApp)
Delegates to ModelRegistry + ForexPredictor for dynamic model selection.
"""

from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


class ModelSelectionAgent:
    """
    Agent responsible for selecting and running the correct model
    for a given (symbol, timeframe) pair.

    Now delegates entirely to ForexPredictor + ModelRegistry.
    Regime detection is embedded inside the feature engineering pipeline.
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self._predictor = None

    @property
    def predictor(self):
        if self._predictor is None:
            from Forex.predictor import ForexPredictor
            self._predictor = ForexPredictor(api_key=self.api_key)
        return self._predictor

    def analyze(self, symbol: str = "EUR/USD", timeframe: str = "15min") -> dict:
        """
        Run model selection and inference for (symbol, timeframe).

        Returns a dict compatible with the old selection_agent contract:
            symbol, timeframe, regime, direction, confidence,
            probability, explanation, current_price, timestamp
        """
        try:
            pred = self.predictor.predict(symbol=symbol, timeframe=timeframe)

            # Map to legacy output format for backward compat
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "regime": pred.get("regime", "range"),
                "direction": pred.get("signal", "HOLD"),
                "confidence": pred.get("confidence", 0.0),
                "probability": pred.get("direction_probability", 0.5),
                "explanation": f"Model: {pred.get('model_source', 'unknown')} | "
                               f"Regime: {pred.get('regime', 'range')} | "
                               f"Confidence: {pred.get('confidence', 0):.2%}",
                "current_price": pred.get("current_price", 0.0),
                "sentiment_score": 0.0,
                "timestamp": pred.get("timestamp", ""),
                # Extended fields
                "predicted_price": pred.get("predicted_price"),
                "risk_level": pred.get("risk_level"),
                "indicators": pred.get("indicators", {}),
                "model_source": pred.get("model_source", ""),
            }

        except Exception as exc:
            logger.error("ModelSelectionAgent.analyze error: %s", exc)
            return {"error": str(exc)}

    def list_available_models(self) -> list:
        """List all trained models available in the registry."""
        try:
            from Forex.registry import ModelRegistry
            registry = ModelRegistry()
            models = registry.list_available()
            best = registry.select_best_available()
            if best:
                for item in models:
                    item["is_recommended"] = (
                        item["symbol"] == best["symbol"] and item["timeframe"] == best["timeframe"]
                    )
            return models
        except Exception as exc:
            logger.error("Could not list models: %s", exc)
            return []

    def select_best_model(self) -> dict:
        """Return the current best trained model based on saved metrics."""
        try:
            from Forex.registry import ModelRegistry
            registry = ModelRegistry()
            best = registry.select_best_available()
            return best or {}
        except Exception as exc:
            logger.error("Could not select best model: %s", exc)
            return {}
