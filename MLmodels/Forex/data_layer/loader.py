"""
TradeAI Data Layer — Higher-level data loading with caching.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from .client import TwelveDataClient

logger = logging.getLogger(__name__)


def _load_cfg() -> dict:
    import yaml
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


class DataLoader:
    """
    High-level data loading utility with optional CSV caching.

    Caching avoids repeated API calls during iterative training/backtesting.
    Cache files are stored in:
        <project_root>/data_cache/<PAIR>/<timeframe>.csv
    """

    def __init__(
        self,
        api_key: str,
        cfg: Optional[dict] = None,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
    ):
        self.cfg = cfg or _load_cfg()
        self.client = TwelveDataClient(api_key, self.cfg)
        self.use_cache = use_cache

        # Cache directory: project_root/data_cache/
        if cache_dir is None:
            # Navigate up from MLmodels/Forex/data_layer/ → project root
            project_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            )
            cache_dir = os.path.join(project_root, "data_cache")
        self.cache_dir = cache_dir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(
        self,
        symbol: str,
        timeframe: str,
        lookback_days: int = 365,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Load OHLCV data for a pair/timeframe.

        Parameters
        ----------
        symbol       : e.g. "EUR/USD"
        timeframe    : e.g. "1h"
        lookback_days: Days of history to fetch (used when start/end not given)
        start_date   : ISO "YYYY-MM-DD" — overrides lookback_days
        end_date     : ISO "YYYY-MM-DD" — defaults to today
        force_refresh: Ignore cache and re-fetch from API

        Returns
        -------
        pd.DataFrame sorted ascending by timestamp
        """
        if not start_date:
            end_dt = datetime.utcnow()
            start_dt = end_dt - timedelta(days=lookback_days)
            start_date = start_dt.strftime("%Y-%m-%d")
            end_date = end_date or end_dt.strftime("%Y-%m-%d")

        cache_path = self._cache_path(symbol, timeframe)

        if self.use_cache and not force_refresh and os.path.exists(cache_path):
            logger.info("Loading cached data from %s", cache_path)
            df = pd.read_csv(cache_path, parse_dates=["timestamp"])
            # Check if cache covers the requested range
            if not df.empty:
                cached_start = df["timestamp"].min()
                cached_end = df["timestamp"].max()
                requested_start = pd.to_datetime(start_date)
                requested_end = pd.to_datetime(end_date)
                if cached_start <= requested_start and cached_end >= requested_end:
                    df_slice = df[
                        (df["timestamp"] >= requested_start)
                        & (df["timestamp"] <= requested_end)
                    ].reset_index(drop=True)
                    logger.info(
                        "Cache hit: %d rows for %s %s", len(df_slice), symbol, timeframe
                    )
                    return df_slice
                logger.info("Cache miss (range mismatch), re-fetching from API...")

        # Fetch from API
        logger.info(
            "Fetching %s %s from %s to %s via API", symbol, timeframe, start_date, end_date
        )
        df = self.client.get_forex_history(
            symbol=symbol,
            interval=timeframe,
            start_date=start_date,
            end_date=end_date,
        )

        if df.empty:
            raise ValueError(
                f"No data returned for {symbol} {timeframe} [{start_date}→{end_date}]"
            )

        # Save to cache
        if self.use_cache:
            self._save_cache(df, cache_path)

        return df

    def load_for_inference(
        self,
        symbol: str,
        timeframe: str,
        output_size: int = 500,
    ) -> pd.DataFrame:
        """
        Lightweight fetch for real-time inference (no date range, no caching).
        Returns the most recent `output_size` candles.
        """
        logger.info(
            "Fetching %d recent candles for %s %s", output_size, symbol, timeframe
        )
        return self.client.get_forex_history(
            symbol=symbol,
            interval=timeframe,
            output_size=output_size,
        )

    def validate_pair_timeframe(self, symbol: str, timeframe: str) -> None:
        """Raise ValueError if pair or timeframe is not in config."""
        supported_pairs = self.cfg.get("pairs", [])
        supported_tfs = self.cfg.get("timeframes", [])
        if symbol not in supported_pairs:
            raise ValueError(
                f"Unsupported pair '{symbol}'. Supported: {supported_pairs}"
            )
        if timeframe not in supported_tfs:
            raise ValueError(
                f"Unsupported timeframe '{timeframe}'. Supported: {supported_tfs}"
            )

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_path(self, symbol: str, timeframe: str) -> str:
        pair_tag = symbol.replace("/", "")
        cache_dir = os.path.join(self.cache_dir, pair_tag)
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"{timeframe}.csv")

    @staticmethod
    def _save_cache(df: pd.DataFrame, path: str) -> None:
        df.to_csv(path, index=False)
        logger.info("Saved %d rows to cache: %s", len(df), path)
