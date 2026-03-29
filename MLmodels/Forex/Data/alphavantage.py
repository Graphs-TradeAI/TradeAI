import requests
import pandas as pd
import time
import os

class AlphaVantageClient:
    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("ALPHA_VANTAGE_API_KEY")
        if not self.api_key:
            # Fallback for demo or testing, though full functionality needs a key
            self.api_key = "demo"

    def _call_api(self, params):
        params["apikey"] = self.api_key
        response = requests.get(self.BASE_URL, params=params)
        
        if response.status_code != 200:
            raise Exception(f"API request failed with status {response.status_code}")
        
        data = response.json()
        
        # Alpha Vantage rate limiting or error messages
        if "Note" in data:
            print(f"Alpha Vantage Note: {data['Note']}")
            # Simple retry logic for rate limiting
            if "standard API call frequency" in data["Note"]:
                time.sleep(60)
                return self._call_api(params)
        
        if "Error Message" in data:
            raise Exception(f"Alpha Vantage Error: {data['Error Message']}")
            
        return data

    def get_forex_history(self, from_symbol="EUR", to_symbol="USD", interval="5min", outputsize="full"):
        """
        Fetches intraday forex data.
        Intervals: 1min, 5min, 15min, 30min, 60min
        """
        params = {
            "function": "FX_INTRADAY",
            "from_symbol": from_symbol,
            "to_symbol": to_symbol,
            "interval": interval,
            "outputsize": outputsize
        }
        
        data = self._call_api(params)
        
        key = f"Time Series FX ({interval})"
        if key not in data:
            # Try Daily if Intraday fails or if interval is 'daily'
            if interval == "daily":
                params["function"] = "FX_DAILY"
                data = self._call_api(params)
                key = "Time Series FX (Daily)"
            else:
                raise Exception(f"Could not find '{key}' in API response. Response keys: {list(data.keys())}")

        df = pd.DataFrame.from_dict(data[key], orient="index")
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        
        # Rename columns to standard names
        df.rename(columns={
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close"
        }, inplace=True)
        
        df = df.astype(float)
        df.reset_index(inplace=True)
        df.rename(columns={"index": "timestamp"}, inplace=True)
        
        return df

    def get_news_sentiment(self, tickers=None, topics=None, limit=200):
        """
        Fetches news sentiment from Alpha Vantage.
        tickers: e.g. "FOREX:EUR", "CRYPTO:BTC"
        topics: e.g. "economy_macro", "finance"
        """
        params = {
            "function": "NEWS_SENTIMENT",
            "limit": limit
        }
        if tickers:
            params["tickers"] = tickers
        if topics:
            params["topics"] = topics
            
        data = self._call_api(params)
        
        if "feed" not in data:
            return pd.DataFrame()
            
        feed = data["feed"]
        news_data = []
        for item in feed:
            news_data.append({
                "timestamp": pd.to_datetime(item["time_published"]),
                "title": item["title"],
                "summary": item["summary"],
                "overall_sentiment_score": item["overall_sentiment_score"],
                "overall_sentiment_label": item["overall_sentiment_label"],
                "url": item["url"]
            })
            
        return pd.DataFrame(news_data)
