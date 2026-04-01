
from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger(__name__)

# Ensure MLmodels is importable
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


class LLMService:


    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self._analyst = None
        self._rag_service = None

    @property
    def analyst(self):
        """Lazy-load ForexAnalyst to avoid import overhead at startup."""
        if self._analyst is None:
            from Forex.analyst import ForexAnalyst
            self._analyst = ForexAnalyst(api_key=self.api_key)
        return self._analyst

    @property
    def rag(self):
        """Lazy-load RAGService."""
        if self._rag_service is None:
            try:
                from .rag_service import RAGService
                self._rag_service = RAGService()
            except Exception as exc:
                logger.error("Failed to initialize RAGService: %s", exc)
                self._rag_service = None
        return self._rag_service

    # ── Intent Parsing (unchanged signature) ─────────────────────────────

    def parse_intent(self, user_prompt: str) -> dict:
        """
        Extract symbol and timeframe from natural language query.

        Returns
        -------
        dict: {"symbol": "EUR/USD", "timeframe": "1h"}
        """
        try:
            return self.analyst.parse_intent(user_prompt)
        except Exception as exc:
            logger.error("Intent parsing error: %s", exc)
            return {"symbol": "AUD/USD", "timeframe": "1h"}


    def generate_response(self, user_prompt: str, prediction_data: dict) -> str:
        """
        Generate a plain-text trading analysis explanation with RAG context.
        Used by the existing /api/chat/ endpoint.

        Returns
        -------
        str: Bullet-point trading analysis
        """
        context = ""
        if self.rag:
            try:
                context = self.rag.get_context(user_prompt)
            except Exception as exc:
                logger.error("RAG context retrieval failed: %s", exc)

        try:
            # Combine prediction data and RAG context for the analyst
            if context:
                prediction_data["rag_context"] = context
            return self.analyst.generate_text_response(user_prompt, prediction_data)
        except Exception as exc:
            logger.error("generate_response error: %s", exc)
            return f"Analysis unavailable: {exc}"

    # ── Structured insight (new /api/predict/ endpoint) ───────────────────

    def generate_insight(self, prediction: dict, risk: dict, user_prompt: str = None) -> dict:
        """
        Generate a structured JSON trading insight.
        Used by the new /api/predict/ endpoint.

        Returns
        -------
        dict: signal, confidence, predicted_price, risk_level,
              reasoning, indicators_used, stop_loss, take_profit, position_size
        """
        try:
            return self.analyst.generate_insight(prediction, risk, user_prompt)
        except Exception as exc:
            logger.error("generate_insight error: %s", exc)
            return {
                "signal": prediction.get("signal", "HOLD"),
                "confidence": prediction.get("confidence", 0.0),
                "predicted_price": prediction.get("predicted_price", 0.0),
                "risk_level": prediction.get("risk_level", "MEDIUM"),
                "reasoning": f"LLM analysis temporarily unavailable: {exc}",
                "indicators_used": ["RSI", "MACD", "EMA"],
                "stop_loss": risk.get("stop_loss"),
                "take_profit": risk.get("take_profit"),
                "position_size": risk.get("position_size"),
                "should_trade": risk.get("should_trade", False),
            }
