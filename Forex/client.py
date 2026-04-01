from __future__ import annotations
import logging
import time
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import requests
from dotenv import load_dotenv
load_dotenv()
import os
TWELVE_DATA_API_KEY=os.getenv("TWELVE_DATA_API_KEY")

logger = logging.getLogger(__name__)


def _load_cfg() -> dict:
    import os, yaml

    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")

    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

class TwelveDataClient:
    BASE_URL = "https://api.twelvedata.com"

    def __init__(self, api_key: str, cfg: Optional[dict] = None):
        self.api_key = api_key
        cfg = cfg or _load_cfg()
        td_cfg = cfg.get("twelvedata", {})
        self.max_output_size: int = td_cfg.get("max_output_size", 5000)
        self.retry_attempts: int = td_cfg.get("retry_attempts", 3)
        self.retry_backoff_base: int = td_cfg.get("retry_backoff_base", 2)
        self.request_timeout: int = td_cfg.get("request_timeout", 30)


    def get_forex_history(
        self,
        symbol: str = "EUR/USD",
        interval: str = "30min",
        output_size: int = 5000,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        
        if start_date and end_date:
            return self._fetch_paginated(symbol, interval, start_date, end_date)
        return self._fetch_single(symbol, interval, output_size)


    def _fetch_single(
        self, symbol: str, interval: str, output_size: int
    ) -> pd.DataFrame:
        params = {
            "symbol": symbol,
            "interval": interval,
            "apikey": self.api_key,
            "outputsize": min(output_size, self.max_output_size),
            "format": "JSON",
        }
        raw = self._get_with_retry("/time_series", params)
        return self._parse_response(raw, symbol, interval)

    def _fetch_paginated(
        self,
        symbol: str,
        interval: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
    
        logger.info(
            "Paginating %s %s from %s to %s", symbol, interval, start_date, end_date
        )
        all_frames: list[pd.DataFrame] = []
        current_end = end_date

        while True:
            params = {
                "symbol": symbol,
                "interval": interval,
                "apikey": self.api_key,
                "outputsize": self.max_output_size,
                "end_date": current_end,
                "format": "JSON",
            }
            raw = self._get_with_retry("/time_series", params)
            df = self._parse_response(raw, symbol, interval)

            if df.empty:
                logger.warning("Empty page returned — stopping pagination.")
                break

            all_frames.append(df)

            # If oldest candle is at or before start_date, we are done
            oldest = df["timestamp"].min()
            start_dt = pd.to_datetime(start_date)
            if oldest <= start_dt:
                break

            # Shift window: new end = oldest candle - 1 unit
            current_end = (oldest - timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S")
            logger.debug("Next page end_date: %s", current_end)

        if not all_frames:
            raise RuntimeError(
                f"No data returned for {symbol} {interval} between {start_date} and {end_date}"
            )

        combined = (
            pd.concat(all_frames, ignore_index=True)
            .drop_duplicates(subset=["timestamp"])
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        # Trim to requested range
        combined = combined[
            (combined["timestamp"] >= pd.to_datetime(start_date))
            & (combined["timestamp"] <= pd.to_datetime(end_date))
        ].reset_index(drop=True)

        logger.info("Paginated result: %d rows for %s %s", len(combined), symbol, interval)
        return combined

    def _get_with_retry(self, endpoint: str, params: dict) -> dict:
        """HTTP GET with exponential backoff retry."""
        url = f"{self.BASE_URL}{endpoint}"
        last_exc: Exception = RuntimeError("Unknown error")

        for attempt in range(1, self.retry_attempts + 1):
            try:
                logger.debug("GET %s params=%s (attempt %d)", url, params, attempt)
                resp = requests.get(url, params=params, timeout=self.request_timeout)
                resp.raise_for_status()
                data = resp.json()

                if "values" not in data:
                    # API-level error (e.g., bad symbol, rate limit)
                    code = data.get("code", "?")
                    msg = data.get("message", str(data))
                    if code == 429:
                        wait = self.retry_backoff_base ** attempt
                        logger.warning("Rate limited — retrying in %ds", wait)
                        time.sleep(wait)
                        continue
                    raise ValueError(f"TwelveData API error [{code}]: {msg}")

                return data

            except (requests.RequestException, ValueError) as exc:
                last_exc = exc
                if attempt < self.retry_attempts:
                    wait = self.retry_backoff_base ** attempt
                    logger.warning(
                        "Request failed (attempt %d/%d): %s — retrying in %ds",
                        attempt, self.retry_attempts, exc, wait,
                    )
                    time.sleep(wait)
                else:
                    logger.error("All %d attempts failed for %s", self.retry_attempts, url)

        raise last_exc

    @staticmethod
    def _parse_response(data: dict, symbol: str, interval: str) -> pd.DataFrame:
        """Convert raw API JSON into a clean typed DataFrame."""
        rows = data.get("values", [])
        if not rows:
            logger.warning("No values in response for %s %s", symbol, interval)
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close"]
            )

        df = pd.DataFrame(rows)

        # Rename datetime → timestamp
        if "datetime" in df.columns:
            df.rename(columns={"datetime": "timestamp"}, inplace=True)

        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Numeric conversion
        numeric_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
        df[numeric_cols] = df[numeric_cols].astype(float)

   
        df = (
            df[["timestamp", "open", "high", "low", "close"]]
            .sort_values("timestamp")
            .dropna()
            .reset_index(drop=True)
        )

        logger.debug("Parsed %d rows for %s %s", len(df), symbol, interval)
        return df

''''client=TwelveDataClient(TWELVE_DATA_API_KEY)
df=client.get_forex_history()

print(df.tail())'''