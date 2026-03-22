import requests
import pandas as pd
from datetime import datetime, timedelta

class TwelveDataClient:
    BASE_URL = "https://api.twelvedata.com"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def get_forex_history(self, symbol="AUD/USD", interval="15min", output_size=5000):

        url = f"{self.BASE_URL}/time_series"
        params = {
            "symbol": symbol,
            "interval": interval,
            "apikey": self.api_key,
            "outputsize": output_size,
        }

        response = requests.get(url, params=params)
        data = response.json()

        if "values" not in data:
            raise Exception(f"API error: {data}")

        df = pd.DataFrame(data["values"])
        df.rename(columns={
            "datetime": "timestamp", 
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
           
        }, inplace=True)

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        df[["open", "high", "low", "close"]] = df[
            ["open", "high", "low", "close"]
        ].astype(float)

        return df
