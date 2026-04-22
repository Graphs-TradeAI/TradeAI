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
    reasoning: str = Field(description="Concise reasoning for the signal (max 3-5 high-impact bullet points)")
    indicators_used: List[str] = Field(description="List of primarily responsible indicator names (RSI, MACD, etc)")


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
Analyze the Forex prediction and technical indicators. Provide a CONCISE, professional insight.
Focus reasoning ONLY on the most significant data points.

=== Prediction Data ===
Symbol: {prediction.get('symbol', 'N/A')} | TF: {prediction.get('timeframe', 'N/A')}
Signal: {prediction.get('signal', 'HOLD')} | Confidence: {prediction.get('confidence', 0.0):.1%}
Target: {risk.get('take_profit', 'N/A')} | Stop: {risk.get('stop_loss', 'N/A')}
Regime: {prediction.get('regime', 'unknown')}

=== Indicators ===
RSI: {indicators.get('rsi', 'N/A')} | Stochastic (K/D): {indicators.get('stoch_k', 'N/A')}/{indicators.get('stoch_d', 'N/A')}
MACD Gap: {indicators.get('macd_diff', 0):.6f}
EMA Alignment: EMA12({indicators.get('ema_12', 0)}) vs EMA26({indicators.get('ema_26', 0)}) vs EMA200({indicators.get('ema_200', 0)})
Volatility: ATR({indicators.get('atr', 0)}) | Bollinger Bands Pos: {indicators.get('bb_position', 0)}
Trend: CCI({indicators.get('cci', 0)}) | ADX({indicators.get('adx', 'N/A')})
Ichimoku: Conv({indicators.get('ichi_conv', 0)}) | Base({indicators.get('ichi_base', 0)})

RESPONSE REQUIREMENT:
Return ONLY a valid JSON object. Reasoning MUST be 4-6 concise bullet points starting with '>'.
Ensure you explicitly map and discuss the impact of at least 4 different indicators per response (e.g. RSI, MACD, EMAs, Ichimoku, etc).
Structure:
{{
  "signal": "BUY/SELL/HOLD",
  "confidence": 0.0-1.0,
  "predicted_price": 0.0,
  "risk_level": "LOW/MEDIUM/HIGH",
  "reasoning": "> bullet point\n> bullet point",
  "indicators_used": ["Indicator1", "Indicator2"]
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
        Generate an ultra-concise technical explanation focusing silently on significant data.
        """
        prompt = f"""
Analyze the Forex setup and provide an ultra-concise breakdown.
Focus SILENTLY only on the most significant high-impact indicators and model confidence. 
Do NOT mention or explain why other indicators are being ignored. No fluff. No meta-commentary.

User Question: {user_prompt}

=== Setup ===
Asset: {prediction_data.get('symbol', 'N/A')} | TF: {prediction_data.get('timeframe', 'N/A')}
Signal: {prediction_data.get('signal', 'HOLD')} | Confidence: {prediction_data.get('confidence', '')}
Indicators: {prediction_data.get('indicators', {})}
"""
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Expert technical analyst. Provide 3-5 ultra-concise bullet points starting with '>'. Direct and professional."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
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
        macd = indicators.get("macd_diff", 0)
        ema_12 = indicators.get("ema_12", 0)
        ema_200 = indicators.get("ema_200", 0)
        cci = indicators.get("cci", 0)
        regime = prediction.get("regime", "range")

        indicators_used = ["RSI", "MACD", "EMA", "CCI"]
        reasoning_lines = []
        
        if isinstance(rsi, (int, float)):
            if rsi > 70:
                reasoning_lines.append(f"> RSI at {rsi:.1f} indicates overbought conditions.")
            elif rsi < 30:
                reasoning_lines.append(f"> RSI at {rsi:.1f} indicates oversold conditions.")
            else:
                reasoning_lines.append(f"> RSI at {rsi:.1f} suggests neutral momentum.")
                
        if isinstance(macd, (int, float)):
            trend = "bullish" if macd > 0 else "bearish"
            reasoning_lines.append(f"> MACD gap ({macd:.5f}) confirms {trend} momentum.")
            
        if isinstance(ema_12, (int, float)) and isinstance(ema_200, (int, float)):
            alignment = "bullish" if ema_12 > ema_200 else "bearish"
            reasoning_lines.append(f"> EMA12 vs EMA200 indicates {alignment} tracking, aligning with {signal}.")
            
        if isinstance(cci, (int, float)):
            reasoning_lines.append(f"> CCI is positioned at {cci:.1f}, confirming the {regime} classification.")

        reasoning = "\n".join(reasoning_lines)

        return {
            "signal": signal,
            "confidence": prediction.get("confidence", 0.0),
            "predicted_price": prediction.get("predicted_price", 0.0),
            "risk_level": prediction.get("risk_level", "MEDIUM"),
            "reasoning": reasoning,
            "indicators_used": indicators_used,
        }
