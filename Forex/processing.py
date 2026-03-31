import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange


def build_forex_feature_set(
    df: pd.DataFrame,
    symbol: str = "AUD/USD",
    timeframe: str = "30min",
    horizon: int = 1
) -> pd.DataFrame:

    df = df.copy()

    # ----------------------------
    # Time handling
    # ----------------------------
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    tf_map = {
        "1min": 0, "5min": 1, "15min": 2,
        "30min": 3, "1h": 4, "4h": 5, "1day": 6
    }
    df["tf_idx"] = tf_map.get(timeframe, 1)

    # ----------------------------
    # Returns (core stable signal)
    # ----------------------------
    df["log_return"] = np.log(df["close"]).diff()
    df["return"] = df["close"].pct_change()

    # normalized price movement (removes scale dependency)
    df["hl_range"] = (df["high"] - df["low"]) / df["close"]
    df["oc_range"] = (df["close"] - df["open"]) / df["open"]

    # ----------------------------
    # Volatility (normalized)
    # ----------------------------
    df["vol_10"] = df["log_return"].rolling(10).std()
    df["vol_20"] = df["log_return"].rolling(20).std()

    atr = AverageTrueRange(df["high"], df["low"], df["close"], window=14)
    df["atr_14"] = atr.average_true_range()
    df["atr_pct"] = df["atr_14"] / df["close"]  # IMPORTANT normalization

    # ----------------------------
    # Trend features
    # ----------------------------
    df["ema_7"] = EMAIndicator(df["close"], window=7).ema_indicator()
    df["ema_20"] = EMAIndicator(df["close"], window=20).ema_indicator()
    df["ema_50"] = EMAIndicator(df["close"], window=50).ema_indicator()

    # normalized EMA distances (important fix)
    df["ema_7_dist"] = (df["close"] - df["ema_7"]) / df["close"]
    df["ema_20_dist"] = (df["close"] - df["ema_20"]) / df["close"]
    df["ema_50_dist"] = (df["close"] - df["ema_50"]) / df["close"]

    adx = ADXIndicator(df["high"], df["low"], df["close"], window=14)
    df["adx"] = adx.adx() / 100.0  # normalized
    df["adx_pos"] = adx.adx_pos()
    df["adx_neg"] = adx.adx_neg()

    # ----------------------------
    # Momentum
    # ----------------------------
    df["rsi"] = RSIIndicator(df["close"], window=14).rsi() / 100.0

    stoch = StochasticOscillator(df["high"], df["low"], df["close"], window=14)
    df["stoch_k"] = stoch.stoch() / 100.0
    df["stoch_d"] = stoch.stoch_signal() / 100.0

    # ----------------------------
    # Volatility bands (normalized)
    # ----------------------------
    bb = BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_width"] = (
        (bb.bollinger_hband() - bb.bollinger_lband())
        / bb.bollinger_mavg()
    )

    df["bb_pos"] = (
        (df["close"] - bb.bollinger_lband())
        / (bb.bollinger_hband() - bb.bollinger_lband())
    )

    # ----------------------------
    # MACD (already scale-free but still normalize)
    # ----------------------------
    macd = MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    df["macd_norm"] = df["macd_diff"] / df["close"]

    # ----------------------------
    # Market regime (cleaned logic)
    # ----------------------------
    df["regime"] = 0

    df.loc[(df["adx"] > 0.25) & (df["ema_50_dist"] > 0), "regime"] = 1
    df.loc[(df["adx"] > 0.25) & (df["ema_50_dist"] < 0), "regime"] = 2

    atr_mean = df["atr_pct"].rolling(20).mean()
    df.loc[df["atr_pct"] > 1.5 * atr_mean, "regime"] = 3

    # ----------------------------
    # Target (REGRESSION)
    # ----------------------------
    df["target_price"] = df["close"].shift(-horizon)

    # better alternative (recommended for ML stability):
    df["target_return"] = df["log_return"].shift(-horizon)

    # ----------------------------
    # Final cleanup
    # ----------------------------
    df = df.dropna().reset_index(drop=True)

    return df