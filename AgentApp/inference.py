import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler



# Add MLmodels to path to allow imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'MLmodels', 'Forex'))


from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import RSIIndicator, StochasticOscillator
from Data.twelvedata import TwelveDataClient

class ModelInference:
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = TwelveDataClient(api_key)
        self.models_dir = os.path.join(BASE_DIR, 'MLmodels', 'Forex', 'forex_models')

    def get_model_path(self, symbol, timeframe):
        # Remove slash from symbol if present (e.g. EUR/USD -> EURUSD)
        clean_symbol = symbol.replace('/', '')
        return os.path.join(self.models_dir, clean_symbol, timeframe, 'model.keras')

    def _build_features_inference(self, df):
        """
        Re-implementation of build_forex_feature_set but keeps the last row
        by filling future_close with current close.
        """
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        # --- Price Features ---
        df["return"] = df["close"].pct_change()
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["volatility_10"] = df["return"].rolling(10).std()
        df["volatility_20"] = df["return"].rolling(20).std()




           # --- Momentum Indicators ---
        df["rsi_14"] = RSIIndicator(df["close"], window=14).rsi()
        df["stoch_k"] = StochasticOscillator(df["high"], df["low"], df["close"], window=14).stoch()
        df["stoch_d"] = StochasticOscillator(df["high"], df["low"], df["close"], window=14).stoch_signal()

        # --- Trend Indicators ---
        df["ema_7"] = EMAIndicator(df["close"], window=7).ema_indicator()
        df["ema_20"] = EMAIndicator(df["close"], window=20).ema_indicator()
        df["ema_50"] = EMAIndicator(df["close"], window=50).ema_indicator()
        df["adx_14"] = ADXIndicator(df["high"], df["low"], df["close"], window=14).adx()

        # --- Volatility Indicators ---
        
        bb = BollingerBands(df["close"], window=20, window_dev=2)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_middle"] = bb.bollinger_mavg()
        df["bb_lower"] = bb.bollinger_lband()
        df["atr_14"] = AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()

        # --- MACD ---
        macd = MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()

        # --- Target Variable ---
        df["future_close"] = df["close"].shift(-1)
        
        # FILL NaN future_close with current close for the last row
        # This is the critical fix to allow inference on the latest candle
        df.loc[df.index[-1], "future_close"] = df.loc[df.index[-1], "close"]

        # Drop other NaNs (caused by rolling windows)
        df = df.dropna().reset_index(drop=True)
        
        return df

    def predict(self, symbol, timeframe):
        model_path = self.get_model_path(symbol, timeframe)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found for {symbol} {timeframe} at {model_path}")

        # Load model
        model = tf.keras.models.load_model(model_path)

        # Fetch data
        df = self.client.get_forex_history(
            symbol=symbol,
            interval=timeframe,
            output_size=5000
        )

        if df.empty:
            raise ValueError("No data fetched from API")

        # Build features using custom inference method
        df_features = self._build_features_inference(df)
        
        seq_length = 50
        if len(df_features) < seq_length:
             raise ValueError(f"Not enough data points. Need at least {seq_length}")

        # Feature columns: All except timestamp. 
        # Note: We INCLUDE future_close because the model expects it.
        feature_cols = [c for c in df_features.columns if c not in ["timestamp","future_close"]]
        
        scaler = StandardScaler()
        scaler.fit(df_features[feature_cols])
        
        # Prepare the last sequence
        last_sequence = df_features[feature_cols].iloc[-seq_length:].values
        last_sequence_scaled = scaler.transform(last_sequence)
        
        # Reshape for LSTM
        input_data = last_sequence_scaled.reshape(1, seq_length, len(feature_cols))
        
        # Predict
        prediction = model.predict(input_data)
        predicted_close = prediction[0][0]
        
        current_close = df_features['close'].iloc[-1]
        ema_7 = df_features['ema_7'].iloc[-1]
        ema_20 = df_features['ema_20'].iloc[-1]
        ema_50 = df_features['ema_50'].iloc[-1]
        adx = df_features['adx_14'].iloc[-1]
        rsi = df_features['rsi_14'].iloc[-1]
        stoch_k = df_features['stoch_k'].iloc[-1]
        stoch_d = df_features['stoch_d'].iloc[-1]
        volatility_10 = df_features['volatility_10'].iloc[-1]
        volatility_20 = df_features['volatility_20'].iloc[-1]
        
        # Determine signal
        signal = "BUY" if predicted_close > current_close else "SELL"
        if ema_7 > ema_20 > ema_50:
            trend = "bullish"
        elif ema_7 < ema_20 < ema_50:
            trend = "bearish"
        else:
            trend = "consolidation"
        
        if volatility_10 > volatility_20:
            volatility = "high"
        else:
            volatility = "low"

        if adx > 40:
            trend_strength = "very strong"
        elif adx > 25:
            trend_strength = "strong"
        elif adx > 20:
            trend_strength = "building"
        else:
            trend_strength = "weak"

        if rsi >= 65:
            momentum = "strong bullish"
        elif rsi > 55:
            momentum = "bullish"
        elif rsi <= 35:
            momentum = "strong bearish"
        elif rsi < 45:
            momentum = "bearish"
        else:
            momentum = "neutral"
        
        if stoch_k > stoch_d and stoch_k < 80:
            stochastic = "bullish continuation"
        elif stoch_k < stoch_d and stoch_k > 20:
            stochastic = "bearish continuation"
        elif stoch_k > 80:
            stochastic = "overbought"
        elif stoch_k < 20:
            stochastic = "oversold"
        else:
            stochastic = "neutral"


        

        if 'atr_14' in df_features.columns:
            atr = df_features['atr_14'].iloc[-1]
            sl_dist = 1.5 * atr
            tp_dist = 2.0 * sl_dist
        else:
            sl_dist = current_close * 0.001
            tp_dist = current_close * 0.002
            
        if signal == "BUY":
            sl = current_close - sl_dist
            tp = current_close + tp_dist
        else:
            sl = current_close + sl_dist
            tp = current_close - tp_dist
            
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "current_price": float(current_close),
            "predicted_close": float(predicted_close),
            "signal": signal,
            "tp": float(tp),
            "sl": float(sl),
            "confidence": "High",
            "trend": trend,
            "trend_strength": trend_strength,
            "momentum": momentum,
            "stochastic": stochastic,
            "volatility": volatility,
        }
