import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from MLmodels.Forex.Data.alphavantage import AlphaVantageClient
from MLmodels.Forex.Data.processing import build_forex_feature_set
from MLmodels.Forex.forex_models.architectures import XGBoostHybrid

class ModelSelectionAgent:
    """
    Agent responsible for:
    1. Detecting market regime (Trend, Range, Volatile).
    2. Selecting the best model for the current regime.
    3. Returning a unified prediction and explanation.
    """
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.client = AlphaVantageClient(api_key)
        self.base_path = "/home/job/Desktop/projects/TradeAI/MLmodels/Forex"
        self.models_dir = os.path.join(self.base_path, "forex_models")

    def _detect_regime(self, df_row):
        """
        Logic to determine market state.
        Returns: 'trend', 'range', or 'volatile'
        """
        regime_idx = df_row["regime"]
        if regime_idx == 3:
            return "volatile"
        elif regime_idx in [1, 2]:
            return "trending"
        else:
            return "range"

    def _get_lstm_prediction(self, symbol, timeframe, last_sequence, scaler):
        # Path for finetuned model or global base
        pair_tag = symbol.replace("/", "")
        ft_model_path = os.path.join(self.models_dir, f"{pair_tag}/{timeframe}/finetuned_model.keras")
        global_model_path = os.path.join(self.models_dir, "global_base.keras")
        
        model_path = ft_model_path if os.path.exists(ft_model_path) else global_model_path
        
        if not os.path.exists(model_path):
            return 0.5, "No LSTM model found"
        
        model = tf.keras.models.load_model(model_path)
        # Prepare input
        input_data = last_sequence.reshape(1, last_sequence.shape[0], last_sequence.shape[1])
        prob = model.predict(input_data)[0][0]
        return prob, f"Used LSTM ({'Specialized' if model_path == ft_model_path else 'Global'})"

    def _get_xgboost_prediction(self, symbol, timeframe, last_features):
        pair_tag = symbol.replace("/", "")
        xgb_path = os.path.join(self.models_dir, f"{pair_tag}/{timeframe}/xgboost_model.json")
        
        if not os.path.exists(xgb_path):
            return 0.5, "No XGBoost model found"
        
        xgb_model = XGBoostHybrid()
        xgb_model.load(xgb_path)
        # last_features needs to be a 2D array [1, num_features]
        prob = xgb_model.model.predict_proba(last_features.reshape(1, -1))[0][1]
        return prob, "Used XGBoost (Volatility Optimized)"

    def analyze(self, symbol="EUR/USD", timeframe="15min"):
        # 1. Fetch Data
        from_sym, to_sym = symbol.split("/")
        df = self.client.get_forex_history(from_sym, to_sym, interval=timeframe)
        
        # 2. Fetch News for sentiment
        news_df = self.client.get_news_sentiment(tickers=f"FOREX:{from_sym}")
        
        # 3. Process Features
        df_features = build_forex_feature_set(df, symbol=symbol, timeframe=timeframe, news_df=news_df)
        
        # 4. Regime Detection
        last_row = df_features.iloc[-1]
        regime = self._detect_regime(last_row)
        
        # 5. Model Selection & Prediction
        # Load global scaler
        scaler_path = os.path.join(self.models_dir, "global_scaler.pkl")
        if not os.path.exists(scaler_path):
            return {"error": "Global scaler not found. Please train model first."}
        
        scaler = joblib.load(scaler_path)
        feature_cols = [c for c in df_features.columns if c not in ["timestamp", "target_direction", "target_return_5"]]
        
        # Scale only features
        scaled_data = scaler.transform(df_features[feature_cols])
        seq_length = 60
        last_sequence = scaled_data[-seq_length:]
        last_features = scaled_data[-1]

        explanation = ""
        prob = 0.5
        
        if regime == "volatile":
            prob, msg = self._get_xgboost_prediction(symbol, timeframe, last_features)
            explanation = msg
        else:
            prob, msg = self._get_lstm_prediction(symbol, timeframe, last_sequence, scaler)
            explanation = msg

        direction = "BUY" if prob > 0.5 else "SELL"
        confidence = abs(prob - 0.5) * 2 # 0.0 to 1.0

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "regime": regime,
            "direction": direction,
            "confidence": float(confidence),
            "probability": float(prob),
            "explanation": explanation,
            "current_price": float(last_row["close"]),
            "sentiment_score": float(last_row["overall_sentiment_score"]),
            "timestamp": str(last_row["timestamp"])
        }
