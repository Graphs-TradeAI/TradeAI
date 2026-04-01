"""
TradeAI — Feature Engineering
Refactored from Data/processing.py — config-driven, fully modular.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import ADXIndicator, EMAIndicator, MACD, CCIIndicator, IchimokuIndicator
from ta.volatility import AverageTrueRange, BollingerBands

logger = logging.getLogger(__name__)


def _load_cfg() -> dict:
    import yaml
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def build_features(
    df: pd.DataFrame,
    symbol: str = "EUR/USD",
    timeframe: str = "1h",
    news_df: Optional[pd.DataFrame] = None,
    cfg: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Full feature engineering pipeline for a single DataFrame.

    Parameters
    ----------
    df        : Raw OHLCV DataFrame (columns: timestamp, open, high, low, close, volume)
    symbol    : Currency pair label e.g. "EUR/USD"
    timeframe : Timeframe string e.g. "1h"
    news_df   : Optional sentiment DataFrame with columns [timestamp, overall_sentiment_score]
    cfg       : Config dict (loaded from config.yaml if None)

    Returns
    -------
    pd.DataFrame with all engineered features + target columns.
    NaN rows dropped at end.
    """
    cfg = cfg or _load_cfg()
    feat_cfg = cfg.get("features", {})
    tf_index = cfg.get("timeframe_index", {})

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Validate minimum rows
    min_rows = 220 # Increased because EMA 200 needs 200 rows
    if len(df) < min_rows:
        raise ValueError(
            f"Insufficient data ({len(df)} rows) — need at least {min_rows} for feature engineering."
        )

    # ── Metadata ──────────────────────────────────────────────────────────
    df["tf_idx"] = tf_index.get(timeframe, 1)

    # ── Price Features ────────────────────────────────────────────────────
    df["return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["hl_spread"] = (df["high"] - df["low"]) / df["close"]
    df["co_spread"] = (df["close"] - df["open"]) / df["open"]

    # ── Volatility ────────────────────────────────────────────────────────
    for window in feat_cfg.get("volatility_windows", [10, 20]):
        df[f"volatility_{window}"] = df["log_return"].rolling(window).std()

    atr_window = feat_cfg.get("atr_window", 14)
    df["atr"] = AverageTrueRange(
        df["high"], df["low"], df["close"], window=atr_window
    ).average_true_range()
    # ATR as percentage of price (normalised)
    df["atr_pct"] = df["atr"] / df["close"]
    # ATR relative to rolling mean (regime signal)
    df["atr_ratio"] = df["atr"] / df["atr"].rolling(20).mean()

    # ── Trend Indicators ──────────────────────────────────────────────────
    ema_windows = feat_cfg.get("ema_windows", [12, 26, 50, 200])
    for w in ema_windows:
        df[f"ema_{w}"] = EMAIndicator(df["close"], window=w).ema_indicator()

    # EMA cross ratios (price relative to EMA)
    df["price_ema12_ratio"] = df["close"] / df.get("ema_12", df["close"]) - 1
    df["price_ema26_ratio"] = df["close"] / df.get("ema_26", df["close"]) - 1
    df["ema12_ema26_ratio"] = df.get("ema_12", df["close"]) / df.get("ema_26", df["close"]) - 1
    df["ema26_ema50_ratio"] = df.get("ema_26", df["close"]) / df.get("ema_50", df["close"]) - 1
    df["ema50_ema200_ratio"] = df.get("ema_50", df["close"]) / df.get("ema_200", df["close"]) - 1

    # ── Ichimoku ──────────────────────────────────────────────────────────
    ichi = IchimokuIndicator(
        high=df["high"], low=df["low"], 
        window1=feat_cfg.get("ichimoku_conv", 9),
        window2=feat_cfg.get("ichimoku_base", 26), 
        window3=feat_cfg.get("ichimoku_span", 52)
    )
    df["ichi_conv"] = ichi.ichimoku_conversion_line()
    df["ichi_base"] = ichi.ichimoku_base_line()
    df["ichi_spana"] = ichi.ichimoku_a()
    df["ichi_spanb"] = ichi.ichimoku_b()

    adx_window = feat_cfg.get("adx_window", 14)
    adx = ADXIndicator(df["high"], df["low"], df["close"], window=adx_window)
    df["adx"] = adx.adx()
    df["adx_pos"] = adx.adx_pos()   # +DI
    df["adx_neg"] = adx.adx_neg()   # -DI
    df["adx_di_ratio"] = df["adx_pos"] / (df["adx_neg"] + 1e-9)

    # ── Momentum Indicators ───────────────────────────────────────────────
    rsi_window = feat_cfg.get("rsi_window", 14)
    df["rsi"] = RSIIndicator(df["close"], window=rsi_window).rsi()
    # Normalised RSI: 0-1 range
    df["rsi_norm"] = df["rsi"] / 100.0

    stoch = StochasticOscillator(
        df["high"], df["low"], df["close"], window=feat_cfg.get("stoch_window", 14)
    )
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()
    df["stoch_diff"] = df["stoch_k"] - df["stoch_d"]

    # ── CCI ───────────────────────────────────────────────────────────────
    cci_window = feat_cfg.get("cci_window", 20)
    df["cci"] = CCIIndicator(df["high"], df["low"], df["close"], window=cci_window).cci()
    df["cci_norm"] = df["cci"] / 100.0

    # ── MACD ─────────────────────────────────────────────────────────────
    macd = MACD(
        df["close"],
        window_slow=feat_cfg.get("macd_slow", 26),
        window_fast=feat_cfg.get("macd_fast", 12),
        window_sign=feat_cfg.get("macd_signal", 9),
    )
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()
    # Normalise by price
    df["macd_norm"] = df["macd"] / df["close"]

    # ── Bollinger Bands ───────────────────────────────────────────────────
    bb_window = feat_cfg.get("bb_window", 20)
    bb = BollingerBands(df["close"], window=bb_window, window_dev=feat_cfg.get("bb_dev", 2))
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_middle"] = bb.bollinger_mavg()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
    # BB position: -1 (below lower) to +1 (above upper)
    df["bb_position"] = (df["close"] - df["bb_middle"]) / (
        (df["bb_upper"] - df["bb_lower"]) / 2 + 1e-9
    )

    # ── Market Regime ─────────────────────────────────────────────────────
    # 0=Range, 1=Bull Trend, 2=Bear Trend, 3=High Volatility
    df["regime"] = 0
    df.loc[(df["adx"] > 25) & (df["close"] > df["ema_50"]), "regime"] = 1
    df.loc[(df["adx"] > 25) & (df["close"] < df["ema_50"]), "regime"] = 2
    atr_ma = df["atr"].rolling(20).mean()
    df.loc[df["atr"] > 1.5 * atr_ma, "regime"] = 3

    # ── Volume (if available) ─────────────────────────────────────────────
    if "volume" in df.columns and df["volume"].sum() > 0:
        df["volume_ma"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / (df["volume_ma"] + 1e-9)
    else:
        df["volume_ratio"] = 1.0

    # ── News/Sentiment Integration ────────────────────────────────────────
    if news_df is not None and not news_df.empty:
        news_df = news_df.copy()
        news_df["timestamp"] = pd.to_datetime(news_df["timestamp"])
        news_df = news_df.sort_values("timestamp")
        df = pd.merge_asof(
            df.sort_values("timestamp"),
            news_df[["timestamp", "overall_sentiment_score"]].sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )
        df["sentiment"] = df["overall_sentiment_score"].fillna(0.0)
    else:
        df["sentiment"] = 0.0

    # ── Target Variables ──────────────────────────────────────────────────
    # Classification target: 1 if next close > current close
    df["target_direction"] = np.where(df["close"].shift(-1) > df["close"], 1, 0)

    # Regression target: predicted next close price
    df["target_price"] = df["close"].shift(-1)
    
    # Regression target (STABLE): predicted next log return
    df["target_return"] = df["log_return"].shift(-1)

    # 5-candle forward return (for winrate / Sharpe use)
    df["target_return_5"] = (df["close"].shift(-5) - df["close"]) / df["close"]

    # ── Clean ─────────────────────────────────────────────────────────────
    df = df.dropna().reset_index(drop=True)
    logger.debug("Feature set: %d rows × %d cols for %s %s", len(df), len(df.columns), symbol, timeframe)

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return feature columns (exclude timestamp and target columns)."""
    exclude = {"timestamp", "target_direction", "target_price", "target_return", "target_return_5"}
    return [c for c in df.columns if c not in exclude]


def get_indicator_snapshot(df: pd.DataFrame) -> dict:
    """
    Extract human-readable indicator values from the last row of a feature DataFrame.
    Used by the inference and LLM layers.
    """
    last = df.iloc[-1]
    return {
        "rsi": round(float(last.get("rsi", 50)), 2),
        "macd": round(float(last.get("macd", 0)), 6),
        "macd_signal": round(float(last.get("macd_signal", 0)), 6),
        "macd_diff": round(float(last.get("macd_diff", 0)), 6),
        "ema_12": round(float(last.get("ema_12", 0)), 5),
        "ema_26": round(float(last.get("ema_26", 0)), 5),
        "ema_50": round(float(last.get("ema_50", 0)), 5),
        "ema_200": round(float(last.get("ema_200", 0)), 5),
        "cci": round(float(last.get("cci", 0)), 2),
        "ichi_conv": round(float(last.get("ichi_conv", 0)), 5),
        "ichi_base": round(float(last.get("ichi_base", 0)), 5),
        "atr": round(float(last.get("atr", 0)), 6),
        "atr_pct": round(float(last.get("atr_pct", 0)), 6),
        "adx": round(float(last.get("adx", 0)), 2),
        "bb_width": round(float(last.get("bb_width", 0)), 6),
        "bb_position": round(float(last.get("bb_position", 0)), 4),
        "stoch_k": round(float(last.get("stoch_k", 50)), 2),
        "stoch_d": round(float(last.get("stoch_d", 50)), 2),
        "regime": int(last.get("regime", 0)),
        "volume_ratio": round(float(last.get("volume_ratio", 1.0)), 4),
    }
