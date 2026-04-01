import json
import logging
import os
from typing import Optional, List
from pydantic import BaseModel, Field, ValidationError
from groq import Groq

logger = logging.getLogger(__name__)


def _load_cfg() -> dict:
    import yaml
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


# ── Pydantic Schemas ────────────────────────────────────────────────────────

class IntentSchema(BaseModel):
    symbol: str = Field(description="The Forex currency pair, e.g., 'EUR/USD'")
    timeframe: str = Field(description="The timeframe, e.g., '1h', '15min', '1day'")


class InsightSchema(BaseModel):
    signal: str = Field(description="Trading signal: BUY, SELL, or HOLD")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")
    predicted_price: float = Field(description="Predicted next close price as a float")
    risk_level: str = Field(description="Risk level: LOW, MEDIUM, or HIGH")
    reasoning: str = Field(description="Detailed natural language reasoning for the signal (bullet points preferred)")
    indicators_used: List[str] = Field(description="List of indicator names that influenced the decision")


class ForexAnalyst:
    def __init__(self, api_key: Optional[str] = None, cfg: Optional[dict] = None):
        self.cfg = cfg or _load_cfg()
        llm_cfg = self.cfg.get("llm", {})

        api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not api_key or "your_actual_key" in api_key:
            logger.warning("GROQ_API_KEY is missing or invalid. Set it in .env.")

        self.client = Groq(api_key=api_key)
        self.model_name = llm_cfg.get("model", "llama-3.3-70b-versatile")
        self.temperature = llm_cfg.get("temperature", 0.3)
        self.max_tokens = llm_cfg.get("max_output_tokens", 1024)

    def generate_insight(
        self,
        prediction: dict,
        risk: dict,
        user_prompt: Optional[str] = None,
    ) -> dict:
        """
        Generate a structured trading insight from model prediction + risk data.
        """
        indicators = prediction.get("indicators", {})
        
        prompt = f"""
Analyze the following Forex prediction and technical data. 
Generate a professional trading insight.

=== Model Prediction ===
Symbol: {prediction.get('symbol', 'N/A')}
Timeframe: {prediction.get('timeframe', 'N/A')}
Signal: {prediction.get('signal', 'HOLD')}
Confidence: {prediction.get('confidence', 0.0):.1%}
Predicted Price: {prediction.get('predicted_price', 0.0)}
Current Price: {prediction.get('current_price', 0.0)}
Market Regime: {prediction.get('regime', 'unknown')}

=== Technical Indicators ===
RSI: {indicators.get('rsi', 'N/A')}
MACD: {indicators.get('macd', 0)} (Signal: {indicators.get('macd_signal', 0)})
EMA 7: {indicators.get('ema_7', 0)} | EMA 20: {indicators.get('ema_20', 0)} | EMA 50: {indicators.get('ema_50', 0)}
ATR: {indicators.get('atr', 0)}
ADX: {indicators.get('adx', 'N/A')}

=== Risk Parameters ===
Stop Loss: {risk.get('stop_loss', 'N/A')}
Take Profit: {risk.get('take_profit', 'N/A')}
Position Size: {risk.get('position_size', 0)} units
Risk Level: {risk.get('risk_level', 'MEDIUM')}
Should Trade: {risk.get('should_trade', False)}

User Context: {user_prompt if user_prompt else 'None'}

RESPONSE REQUIREMENT:
Return ONLY a valid JSON object matching this structure:
{{
  "signal": "BUY/SELL/HOLD",
  "confidence": 0.0-1.0,
  "predicted_price": 0.0,
  "risk_level": "LOW/MEDIUM/HIGH",
  "reasoning": "bullet points explaining the trade",
  "indicators_used": ["RSI", "MACD", etc]
}}
"""

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert Forex trading analyst AI. You must respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}
            )
            parsed = json.loads(completion.choices[0].message.content)
            # Validate with Pydantic
            InsightSchema(**parsed)
        except Exception as exc:
            logger.error("Groq insight generation failed: %s — using fallback.", exc)
            parsed = self._fallback_insight(prediction, risk)

        # Merge risk params into output
        parsed["stop_loss"] = risk.get("stop_loss")
        parsed["take_profit"] = risk.get("take_profit")
        parsed["position_size"] = risk.get("position_size")
        parsed["should_trade"] = risk.get("should_trade")

        return parsed

    def parse_intent(self, user_prompt: str) -> dict:
        """
        Extract symbol and timeframe from natural language user input.
        """
        supported_pairs = self.cfg.get("pairs", [])
        supported_tfs = self.cfg.get("timeframes", [])

        system_instruction = f"""You are a financial intent parser. Extract the Forex currency pair and timeframe from the user's query.
Supported pairs: {', '.join(supported_pairs)}
Supported timeframes: {', '.join(supported_tfs)}
Normalize pair to uppercase (e.g., "EUR/USD").
Normalize timeframe (e.g., "1h", "15min").
Return ONLY JSON: {{"symbol": "...", "timeframe": "..."}}"""

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Return only a JSON object."},
                    {"role": "user", "content": f"User Query: {user_prompt}"}
                ],
                response_format={"type": "json_object"}
            )
            result = json.loads(completion.choices[0].message.content)

            # Validate against supported lists
            if result.get("symbol") not in supported_pairs:
                result["symbol"] = "AUD/USD"
            if result.get("timeframe") not in supported_tfs:
                result["timeframe"] = "1h"

            return result
        except Exception as exc:
            logger.warning("Groq intent parsing failed: %s — using defaults.", exc)
            return {"symbol": "AUD/USD", "timeframe": "1h"}

    def generate_text_response(self, user_prompt: str, prediction_data: dict) -> str:
        """
        Generate a plain-text explanation.
        """
        rag_context = prediction_data.get('rag_context', '')
        
        prompt = f"""
Explain this Forex trading signal to the user in a professional way.
User Question: {user_prompt}

=== Market Data ===
Signal: {prediction_data.get('signal', '')} for {prediction_data.get('symbol', '')}
Confidence: {prediction_data.get('confidence', '')}
Target: {prediction_data.get('tp', '')} | Stop: {prediction_data.get('sl', '')}
Current Price: {prediction_data.get('current_price', '')}
"""
        if rag_context:
            prompt += f"\n=== Knowledge Base Context (RAG) ===\n{rag_context}\n"
            
        prompt += "\nINSTRUCTION: If the user asks a general Forex question, use the Knowledge Base Context to provide an accurate answer. For trading signals, explain the technical setup using the Market Data."
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a professional Forex signal explainer. Use bullet points starting with '>'. No markdown headers."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature
            )
            return completion.choices[0].message.content.strip()
        except Exception as exc:
            logger.error("Groq text generation failed: %s", exc)
            return f"Analysis unavailable: {exc}"

    @staticmethod
    def _fallback_insight(prediction: dict, risk: dict) -> dict:
        """Deterministic fallback when LLM generation fails."""
        signal = prediction.get("signal", "HOLD")
        indicators = prediction.get("indicators", {})
        rsi = indicators.get("rsi", 50)
        regime = prediction.get("regime", "range")

        indicators_used = ["RSI", "MACD", "EMA"]
        if isinstance(rsi, (int, float)):
            if rsi > 70:
                reasoning = f"> RSI at {rsi:.1f} indicates overbought conditions.\n> Signal {signal} based on trend direction."
            elif rsi < 30:
                reasoning = f"> RSI at {rsi:.1f} indicates oversold conditions.\n> Signal {signal} based on expected recovery."
            else:
                reasoning = f"> Market is in {regime} regime.\n> EMA alignment and MACD direction support the {signal} signal."
        else:
            reasoning = f"> Directional signal is {signal}.\n> Based on machine learning model confidence of {prediction.get('confidence', 0.0):.1%}."

        return {
            "signal": signal,
            "confidence": prediction.get("confidence", 0.0),
            "predicted_price": prediction.get("predicted_price", 0.0),
            "risk_level": prediction.get("risk_level", "MEDIUM"),
            "reasoning": reasoning,
            "indicators_used": indicators_used,
        }
