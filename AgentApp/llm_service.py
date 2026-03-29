import os
import json
from groq import Groq

class LLMService:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is not set. Please provide it.")
        self.client = Groq(api_key=self.api_key)

    def parse_intent(self, user_prompt):
        """
        Extracts symbol and timeframe from user prompt.
        Returns a dictionary: {"symbol": "EUR/USD", "timeframe": "30min"}
        """
        system_prompt = """
        You are a financial intent parser. Extract the Forex currency pair and timeframe from the user's query.
        Supported timeframes: 1min, 5min, 15min, 30min, 1h, 4h, 1day.
        Supported pairs: EUR/USD, GBP/USD, USD/CHF, (and others, normalize to standard format XXX/YYY).
        
        Return ONLY a JSON object with keys "symbol" and "timeframe".
        If information is missing, use defaults: symbol="EUR/USD", timeframe="30min".
        Example: "Should I buy euro dollar on 15m?" -> {"symbol": "EUR/USD", "timeframe": "15min"}
        """
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(chat_completion.choices[0].message.content)
            return result
        except Exception as e:
            print(f"Error parsing intent: {e}")
            return {"symbol": "EUR/USD", "timeframe": "30min"} # Fallback

    def generate_response(self, user_prompt, prediction_data):
        system_prompt = """
        You are an expert Forex trading signal explainer.
        Use ONLY the provided market analysis data to respond.
        Be professional, concise, and brief.

        Output Requirements:
        - Recommendation: BUY or SELL
        - Entry Price: current_price
        - Take Profit (TP): provided TP value
        - Stop Loss (SL): provided SL value

        Explanation Rules:
        - Explain the reasoning briefly by comparing predicted_close vs current_price.
        - Explain how each provided indicator supports or conflicts with the signal.
        - You MUST only use the provided indicator interpretations.
        - Do NOT invent indicators, values, or market conditions.
        - Do NOT predict price movement beyond the given signal.
        - Do NOT use absolute language (e.g., “will”, “guaranteed”).
        - Do NOT provide financial advice; this is informational only.
        - If indicators conflict, explicitly mention the conflict.
        - If indicators support the signal expilictly mention the support.

        Indicators Provided:
        - Signal
        - Trend
        - Trend Strength
        - Momentum
        - Stochastic
        - Volatility

        Formatting Rules:
        - Use plain text bullet points ONLY.
        - Each bullet must start with a greater than sign (>).
        - Do NOT use markdown symbols such as *, +, or •.
        """
        
        user_content = f"""
        User Question: {user_prompt}
        
        Market Analysis:
        Symbol: {prediction_data['symbol']}
        Timeframe: {prediction_data['timeframe']}
        Current Price: {prediction_data['current_price']}
        Predicted Close: {prediction_data['predicted_close']}
        Signal: {prediction_data['signal']}
        Take Profit: {prediction_data['tp']}
        Stop Loss: {prediction_data['sl']}
        Trend: {prediction_data['trend']}
        Trend Strength: {prediction_data['trend_strength']}
        Momentum: {prediction_data['momentum']}
        Stochastic: {prediction_data['stochastic']}
        Volatility: {prediction_data['volatility']}
        
        """
        
        chat_completion = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
        )
        
        return chat_completion.choices[0].message.content
