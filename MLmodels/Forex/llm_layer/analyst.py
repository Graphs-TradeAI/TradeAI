"""
TradeAI — LLM Analyst (LangChain + Gemini)
Converts model predictions into structured, human-readable trading insights.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.schema.output_parser import StrOutputParser

logger = logging.getLogger(__name__)


def _load_cfg() -> dict:
    import yaml
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


# ── Output Schema ──────────────────────────────────────────────────────────

INSIGHT_SCHEMAS = [
    ResponseSchema(name="signal", description="Trading signal: BUY, SELL, or HOLD"),
    ResponseSchema(name="confidence", description="Confidence score between 0.0 and 1.0"),
    ResponseSchema(name="predicted_price", description="Predicted next close price as a float"),
    ResponseSchema(name="risk_level", description="Risk level: LOW, MEDIUM, or HIGH"),
    ResponseSchema(name="reasoning", description="Detailed natural language reasoning for the signal"),
    ResponseSchema(name="indicators_used", description="JSON array of indicator names that influenced the decision"),
]


class ForexAnalyst:
    """
    LangChain-powered LLM layer for forex prediction explainability.

    Uses Gemini via LangChain's ChatGoogleGenerativeAI to:
    - Generate structured JSON trading insights
    - Parse user intent (symbol + timeframe extraction)
    - Provide natural language reasoning
    """

    def __init__(self, api_key: Optional[str] = None, cfg: Optional[dict] = None):
        self.cfg = cfg or _load_cfg()
        llm_cfg = self.cfg.get("llm", {})

        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required. Set it in .env or pass explicitly.")

        self.llm = ChatGoogleGenerativeAI(
            model=llm_cfg.get("model", "gemini-2.0-flash"),
            google_api_key=api_key,
            temperature=llm_cfg.get("temperature", 0.3),
            max_output_tokens=llm_cfg.get("max_output_tokens", 1024),
        )

        self._output_parser = StructuredOutputParser.from_response_schemas(INSIGHT_SCHEMAS)
        self._str_parser = StrOutputParser()

    # ──────────────────────────────────────────────────────────────────────
    # Core: Structured Insight Generation
    # ──────────────────────────────────────────────────────────────────────

    def generate_insight(
        self,
        prediction: dict,
        risk: dict,
        user_prompt: Optional[str] = None,
    ) -> dict:
        """
        Generate a structured trading insight from model prediction + risk data.

        Parameters
        ----------
        prediction : Output from ForexPredictor.predict()
        risk       : Output from RiskEngine.assess()
        user_prompt: Optional user question to contextualize the explanation

        Returns
        -------
        dict with: signal, confidence, predicted_price, risk_level,
                   reasoning, indicators_used, stop_loss, take_profit, position_size
        """
        format_instructions = self._output_parser.get_format_instructions()

        system_prompt = SystemMessagePromptTemplate.from_template(
            """You are an expert Forex trading analyst AI.
Your role is to interpret machine learning model outputs and provide clear, professional trading insights.

Rules:
- Base your analysis ONLY on the provided data — do not invent values.
- Use conditional language ("suggests", "indicates", "appears") — never absolute.
- If indicators conflict, explicitly acknowledge the conflict.
- Do NOT provide financial advice; this is informational only.
- Keep reasoning concise (3-5 bullet points max).

{format_instructions}"""
        )

        human_prompt = HumanMessagePromptTemplate.from_template(
            """Analyze the following Forex prediction and generate a structured insight.

=== Model Prediction ===
Symbol: {symbol}
Timeframe: {timeframe}
Signal: {signal}
Confidence: {confidence:.1%}
Direction Probability: {direction_probability:.3f}
Predicted Price: {predicted_price}
Current Price: {current_price}
Market Regime: {regime}
Model Source: {model_source}

=== Technical Indicators ===
RSI: {rsi}
MACD: {macd} (Signal: {macd_signal}, Histogram: {macd_diff})
EMA 7: {ema_7} | EMA 20: {ema_20} | EMA 50: {ema_50}
ADX: {adx} (Trend Strength)
ATR: {atr} ({atr_pct:.4%} of price)
Bollinger Position: {bb_position:.3f} (-1=below lower, +1=above upper)
Stochastic K: {stoch_k} | D: {stoch_d}

=== Risk Parameters ===
Stop Loss: {stop_loss}
Take Profit: {take_profit}
Risk/Reward: 1:{risk_reward_ratio}
Position Size: {position_size} units
Risk Level: {risk_level}
Should Trade: {should_trade}
Filter Reason: {filter_reason}

{user_context}

Generate a structured JSON response following the format instructions above."""
        )

        prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
        chain = prompt | self.llm | self._str_parser

        indicators = prediction.get("indicators", {})

        try:
            raw_response = chain.invoke({
                "format_instructions": format_instructions,
                "symbol": prediction.get("symbol", ""),
                "timeframe": prediction.get("timeframe", ""),
                "signal": prediction.get("signal", "HOLD"),
                "confidence": prediction.get("confidence", 0.0),
                "direction_probability": prediction.get("direction_probability", 0.5),
                "predicted_price": prediction.get("predicted_price", 0.0),
                "current_price": prediction.get("current_price", 0.0),
                "regime": prediction.get("regime", "unknown"),
                "model_source": prediction.get("model_source", "unknown"),
                # Indicators
                "rsi": indicators.get("rsi", "N/A"),
                "macd": indicators.get("macd", 0),
                "macd_signal": indicators.get("macd_signal", 0),
                "macd_diff": indicators.get("macd_diff", 0),
                "ema_7": indicators.get("ema_7", 0),
                "ema_20": indicators.get("ema_20", 0),
                "ema_50": indicators.get("ema_50", 0),
                "adx": indicators.get("adx", "N/A"),
                "atr": indicators.get("atr", 0),
                "atr_pct": indicators.get("atr_pct", 0),
                "bb_position": indicators.get("bb_position", 0),
                "stoch_k": indicators.get("stoch_k", "N/A"),
                "stoch_d": indicators.get("stoch_d", "N/A"),
                # Risk
                "stop_loss": risk.get("stop_loss", "N/A"),
                "take_profit": risk.get("take_profit", "N/A"),
                "risk_reward_ratio": risk.get("risk_reward_ratio", 2.0),
                "position_size": risk.get("position_size", 0),
                "risk_level": risk.get("risk_level", "MEDIUM"),
                "should_trade": risk.get("should_trade", False),
                "filter_reason": risk.get("filter_reason", ""),
                "user_context": f"User Question: {user_prompt}" if user_prompt else "",
            })

            # Parse structured output
            parsed = self._output_parser.parse(raw_response)

        except Exception as exc:
            logger.error("Structured output parsing failed: %s — using fallback.", exc)
            parsed = self._fallback_insight(prediction, risk)

        # Merge risk params into output
        parsed["stop_loss"] = risk.get("stop_loss")
        parsed["take_profit"] = risk.get("take_profit")
        parsed["position_size"] = risk.get("position_size")
        parsed["should_trade"] = risk.get("should_trade")

        # Ensure types
        parsed["confidence"] = float(parsed.get("confidence", prediction.get("confidence", 0)))
        parsed["predicted_price"] = float(parsed.get("predicted_price", prediction.get("predicted_price", 0)))

        return parsed

    # ──────────────────────────────────────────────────────────────────────
    # Intent Parsing
    # ──────────────────────────────────────────────────────────────────────

    def parse_intent(self, user_prompt: str) -> dict:
        """
        Extract symbol and timeframe from natural language user input.

        Returns
        -------
        dict: {"symbol": "EUR/USD", "timeframe": "1h"}
        """
        supported_pairs = self.cfg.get("pairs", [])
        supported_tfs = self.cfg.get("timeframes", [])

        system = SystemMessagePromptTemplate.from_template(
            """You are a financial intent parser. Extract the Forex currency pair and timeframe from the user's query.

Supported pairs: {pairs}
Supported timeframes: {timeframes}

Return ONLY a valid JSON object with exactly two keys: "symbol" and "timeframe".
Do NOT include markdown or code blocks.
Normalize pair to uppercase with slash (e.g., "EUR/USD").
Normalize timeframe to match supported list (e.g., "15 minute" → "15min", "hourly" → "1h", "daily" → "1day").
If information is missing, use defaults: symbol="EUR/USD", timeframe="1h"."""
        )

        human = HumanMessagePromptTemplate.from_template("User Query: {query}")
        prompt = ChatPromptTemplate.from_messages([system, human])
        chain = prompt | self.llm | self._str_parser

        try:
            raw = chain.invoke({
                "pairs": ", ".join(supported_pairs),
                "timeframes": ", ".join(supported_tfs),
                "query": user_prompt,
            })
            # Strip markdown code blocks if present
            raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            result = json.loads(raw)

            # Validate against supported lists
            if result.get("symbol") not in supported_pairs:
                result["symbol"] = "EUR/USD"
            if result.get("timeframe") not in supported_tfs:
                result["timeframe"] = "1h"

            return result

        except Exception as exc:
            logger.warning("Intent parsing failed: %s — using defaults.", exc)
            return {"symbol": "EUR/USD", "timeframe": "1h"}

    # ──────────────────────────────────────────────────────────────────────
    # Legacy compatibility: generate a plain-text response (like old llm_service.py)
    # ──────────────────────────────────────────────────────────────────────

    def generate_text_response(self, user_prompt: str, prediction_data: dict) -> str:
        """
        Generate a plain-text explanation (backward compat with old api_chat view).
        """
        system = SystemMessagePromptTemplate.from_template(
            """You are an expert Forex trading signal explainer.
Use ONLY the provided market data. Be professional and concise.

Format:
- Use bullet points starting with ">"
- No markdown symbols (*, +, #)
- Include: Recommendation, Entry Price, TP, SL, and brief indicator reasoning."""
        )
        human = HumanMessagePromptTemplate.from_template(
            """User Question: {question}

Market Data:
Symbol: {symbol} | Timeframe: {timeframe}
Signal: {signal} | Confidence: {confidence}
Current Price: {current_price} | Predicted: {predicted_close}
TP: {tp} | SL: {sl}
Trend: {trend} | Momentum: {momentum} | Volatility: {volatility}"""
        )
        prompt = ChatPromptTemplate.from_messages([system, human])
        chain = prompt | self.llm | self._str_parser

        try:
            return chain.invoke({
                "question": user_prompt,
                "symbol": prediction_data.get("symbol", ""),
                "timeframe": prediction_data.get("timeframe", ""),
                "signal": prediction_data.get("signal", ""),
                "confidence": prediction_data.get("confidence", ""),
                "current_price": prediction_data.get("current_price", ""),
                "predicted_close": prediction_data.get("predicted_close", prediction_data.get("predicted_price", "")),
                "tp": prediction_data.get("tp", prediction_data.get("take_profit", "")),
                "sl": prediction_data.get("sl", prediction_data.get("stop_loss", "")),
                "trend": prediction_data.get("trend", prediction_data.get("regime", "")),
                "momentum": prediction_data.get("momentum", ""),
                "volatility": prediction_data.get("volatility", prediction_data.get("risk_level", "")),
            })
        except Exception as exc:
            logger.error("Text generation failed: %s", exc)
            return f"Analysis unavailable: {exc}"

    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _fallback_insight(prediction: dict, risk: dict) -> dict:
        """Deterministic fallback when LLM parsing fails."""
        signal = prediction.get("signal", "HOLD")
        indicators = prediction.get("indicators", {})
        rsi = indicators.get("rsi", 50)
        regime = prediction.get("regime", "range")

        indicators_used = ["RSI", "MACD", "EMA"]
        if rsi > 70:
            reasoning = f"RSI at {rsi:.1f} indicates overbought conditions. Signal {signal} based on trend direction."
        elif rsi < 30:
            reasoning = f"RSI at {rsi:.1f} indicates oversold conditions. Signal {signal} based on expected recovery."
        else:
            reasoning = (
                f"Market is in {regime} regime. EMA alignment and MACD direction support the {signal} signal."
            )

        return {
            "signal": signal,
            "confidence": prediction.get("confidence", 0.0),
            "predicted_price": prediction.get("predicted_price", 0.0),
            "risk_level": prediction.get("risk_level", "MEDIUM"),
            "reasoning": reasoning,
            "indicators_used": indicators_used,
        }
