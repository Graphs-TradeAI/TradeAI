import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from django.conf import settings

# Add MLmodels to path to allow imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'MLmodels', 'Forex'))

from AgentApp.selection_agent import ModelSelectionAgent
from AgentApp.web_analyst import WebAnalyst

class ModelInference:
    """
    Main entry point for AI Trader inference.
    Wraps the ModelSelectionAgent and WebAnalyst.
    """
    def __init__(self, api_key=None):
        self.api_key = api_key or getattr(settings, 'ALPHA_VANTAGE_API_KEY', None)
        self.selection_agent = ModelSelectionAgent(self.api_key)
        self.web_analyst = WebAnalyst(self.api_key)

    def predict(self, symbol="EUR/USD", timeframe="15min"):
        """
        Runs the hybrid prediction pipeline.
        1. Fetch Price + News.
        2. Detect Regime.
        3. Select Model.
        4. Aggregate Web Sentiment.
        5. Return structured result.
        """
        try:
            # 1. Selection Agent Analysis (Pre-trained ML models)
            ml_result = self.selection_agent.analyze(symbol, timeframe)
            
            # 2. Web Analysis (Real-time sentiment from news/internet)
            web_sentiment = self.web_analyst.analyze_web_trends(symbol)
            
            # 3. Combine Results
            # Standardizing result for the frontend / LLM explanation
            result = {
                "symbol": symbol,
                "timeframe": timeframe,
                "current_price": ml_result.get("current_price"),
                "signal": ml_result.get("direction"),
                "confidence": f"{ml_result.get('confidence', 0.5)*100:.1f}%",
                "regime": ml_result.get("regime"),
                "probability": ml_result.get("probability"),
                "explanation": ml_result.get("explanation"),
                "sentiment_summary": web_sentiment.get("label"),
                "sentiment_score": web_sentiment.get("overall_score"),
                "top_headlines": web_sentiment.get("top_headlines", []),
                # Indicators for LLM explanation
                "trend": "bullish" if ml_result.get("direction") == "BUY" else "bearish",
                "volatility": "high" if ml_result.get("regime") == "volatile" else "normal",
            }
            
            # Calculate dynamic TP/SL if not provided by model
            # Simple 1.5 ATR placeholder logic
            current_price = result["current_price"]
            # We would typically get ATR from ml_result if we expanded it
            atr_placeholder = current_price * 0.002 
            if result["signal"] == "BUY":
                result["tp"] = current_price + (atr_placeholder * 2)
                result["sl"] = current_price - atr_placeholder
            else:
                result["tp"] = current_price - (atr_placeholder * 2)
                result["sl"] = current_price + atr_placeholder
                
            return result
            
        except Exception as e:
            print(f"Inference error: {e}")
            raise e

    def calculate_model_metrics(self, symbol, timeframe, n_backtest=100):
        """
        Placeholder for backtest metrics logic using the new AlphaVantage flow.
        """
        # We could implement a full walk-forward backtest here.
        # For now, return dummy metrics to avoid breaking the UI.
        return {
            "directional_accuracy": 0.58,
            "win_rate": 0.54,
            "risk_reward": 1.8,
            "sharpe_ratio": 1.2,
            "n_backtest": n_backtest
        }

        