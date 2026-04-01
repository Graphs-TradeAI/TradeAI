import requests
import pandas as pd
from datetime import datetime, timedelta
import time


class TwelveDataClient:
    BASE_URL = "https://api.twelvedata.com"

    def __init__(self, api_key: str):
        self.api_key = api_key


    def fetch_chunk(self, symbol, interval, output_size=5000, end_date=None):

        url = f"{self.BASE_URL}/time_series"

        params = {
            "symbol": symbol,
            "interval": interval,
            "apikey": self.api_key,
            "outputsize": output_size,
        }

        if end_date:
            params["end_date"] = end_date

        response = requests.get(url, params=params)
        data = response.json()

        if "values" not in data:
            raise Exception(f"API error: {data}")

        df = pd.DataFrame(data["values"])

        df = df.rename(columns={
            "datetime": "timestamp",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
        })

        df["timestamp"] = pd.to_datetime(df["timestamp"])

        df[["open", "high", "low", "close"]] = df[
            ["open", "high", "low", "close"]
        ].astype(float)

        return df

    def get_forex_history(
        self,
        symbol="AUD/USD",
        interval="1day",
        batches=5,         
        output_size=5000
    ):

        all_data = []
        end_date = None

        for i in range(batches):

            df = self.fetch_chunk(
                symbol=symbol,
                interval=interval,
                output_size=output_size,
                end_date=end_date
            )

            if df.empty:
                break

            # sort just in case
            df = df.sort_values("timestamp")

            all_data.append(df)

            # move backward in time
            oldest_time = df["timestamp"].min()

            # shift slightly to avoid overlap
            end_date = (oldest_time - timedelta(minutes=1)).strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            print(f"Batch {i+1}: fetched {len(df)} candles, next end_date = {end_date}")

            time.sleep(0.5)  # avoid rate limits

        final_df = pd.concat(all_data)

        # remove duplicates (important for safety)
        final_df = final_df.drop_duplicates(subset=["timestamp"])

        final_df = final_df.sort_values("timestamp").reset_index(drop=True)

        return final_df




