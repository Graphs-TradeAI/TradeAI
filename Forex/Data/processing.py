import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange

def build_forex_feature_set(df: pd.DataFrame, symbol="EUR/USD", timeframe="5min", news_df=None) -> pd.DataFrame:
    """
    Enhanced feature engineering for Forex.
    - symbol: Currency pair string.
    - timeframe: Timeframe string.
    - news_df: DataFrame with sentiment scores.
    """
    df = df.copy()

    # Ensure timestamp is datetime and sorted
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # --- Metadata Features ---
    # Convert timeframe and symbol to categorical numeric representations
    tf_map = {"1min": 0, "5min": 1, "15min": 2, "30min": 3, "1h": 4, "4h": 5, "1day": 6}
    df["tf_idx"] = tf_map.get(timeframe, 1) # Default to 5min
    
    # --- Price Features ---
    df["return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    
    # --- Volatility ---
    df["volatility_10"] = df["log_return"].rolling(10).std()
    df["volatility_20"] = df["log_return"].rolling(20).std()
    df["atr_14"] = AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()

    # --- Trend Indicators ---
    df["ema_7"] = EMAIndicator(df["close"], window=7).ema_indicator()
    df["ema_20"] = EMAIndicator(df["close"], window=20).ema_indicator()
    df["ema_50"] = EMAIndicator(df["close"], window=50).ema_indicator()
    
    adx = ADXIndicator(df["high"], df["low"], df["close"], window=14)
    df["adx_14"] = adx.adx()
    df["adx_pos"] = adx.adx_pos()
    df["adx_neg"] = adx.adx_neg()

    # --- Momentum Indicators ---
    df["rsi_14"] = RSIIndicator(df["close"], window=14).rsi()
    stoch = StochasticOscillator(df["high"], df["low"], df["close"], window=14)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # --- Volatility Channels ---
    bb = BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_middle"] = bb.bollinger_mavg()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]

    # --- MACD ---
    macd = MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    # --- Market Regime Detection (Simple Logic) ---
    # 0: Range, 1: Bull Trend, 2: Bear Trend, 3: High Volatility
    df["regime"] = 0
    # Bull Trend: ADX > 25 and Close > EMA_50
    df.loc[(df["adx_14"] > 25) & (df["close"] > df["ema_50"]), "regime"] = 1
    # Bear Trend: ADX > 25 and Close < EMA_50
    df.loc[(df["adx_14"] > 25) & (df["close"] < df["ema_50"]), "regime"] = 2
    # High Volatility: ATR > 1.5 * ATR_SMA(20)
    atr_sma = df["atr_14"].rolling(20).mean()
    df.loc[df["atr_14"] > 1.5 * atr_sma, "regime"] = 3

    # --- Sentiment Integration ---
    if news_df is not None and not news_df.empty:
        news_df["timestamp"] = pd.to_datetime(news_df["timestamp"])
        news_df = news_df.sort_values("timestamp")
        
        # Merge sentiment score on nearest timestamp (backward)
        # Using merge_asof requires both to be sorted
        df = pd.merge_asof(
            df.sort_values("timestamp"), 
            news_df[["timestamp", "overall_sentiment_score"]].sort_values("timestamp"),
            on="timestamp",
            direction="backward"
        )
        df["overall_sentiment_score"] = df["overall_sentiment_score"].fillna(0)
    else:
        df["overall_sentiment_score"] = 0

    # --- Target Variable (Regression) ---
    # Next close price
    df["target_price"] = df["close"].shift(-1)

    # Clean NaNs
    df = df.dropna().reset_index(drop=True)

    return df