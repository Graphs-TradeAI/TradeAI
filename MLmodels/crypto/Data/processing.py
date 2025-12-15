import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange

def build_crypto_feature_set(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure timestamp is datetime and sorted
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # --- Price Features ---
    df["return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["volatility_10"] = df["return"].rolling(10).std()
    df["volatility_20"] = df["return"].rolling(20).std()

   

    # --- Trend Indicators ---
    df["ema_7"] = EMAIndicator(df["close"], window=7).ema_indicator()
    df["ema_20"] = EMAIndicator(df["close"], window=20).ema_indicator()
    df["ema_50"] = EMAIndicator(df["close"], window=50).ema_indicator()
    df["adx_14"] = ADXIndicator(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=14
    ).adx()

    # --- Momentum Indicators ---
    df["rsi_14"] = RSIIndicator(df["close"], window=14).rsi()

    stoch = StochasticOscillator(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=14
    )
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # --- Volatility Indicators ---
    bb = BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_middle"] = bb.bollinger_mavg()
    df["bb_lower"] = bb.bollinger_lband()

    df["atr_14"] = AverageTrueRange(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=14
    ).average_true_range()

    # --- MACD ---
    macd = MACD(
        close=df["close"],
        window_slow=26,
        window_fast=12,
        window_sign=9
    )
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    # --- Target Variable ---
    df["future_close"] = df["close"].shift(-1)
   

    # Clean NaNs
    df = df.dropna().reset_index(drop=True)

    return df
