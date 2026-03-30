import requests
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class WebAnalyst:
    """
    Agent for aggregating and summarizing sentiment from news and APIs.
    Uses Twelve Data for price data and provides basic technical sentiment.
    """
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.base_url = "https://api.twelvedata.com"
        self.analyzer = SentimentIntensityAnalyzer()

    def _fetch_price_data(self, symbol="EUR/USD", interval="1h"):
        """Fetch recent price data for analysis."""
        url = f"{self.base_url}/time_series"
        params = {
            "symbol": symbol,
            "interval": interval,
            "apikey": self.api_key,
            "outputsize": 100
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if "values" not in data:
            return None
        
        df = pd.DataFrame(data["values"])
        df.rename(columns={"datetime": "timestamp"}, inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
        
        return df

    def _calculate_technical_sentiment(self, df):
        """
        Calculate sentiment based on technical indicators.
        Returns a score from -1 (bearish) to 1 (bullish).
        """
        if df is None or len(df) < 50:
            return 0.0
        
        sentiment = 0.0
        weights = []
        
        # Simple Moving Averages
        sma_20 = df["close"].rolling(20).mean()
        sma_50 = df["close"].rolling(50).mean()
        
        # Current price vs SMAs
        current_price = df["close"].iloc[-1]
        if sma_20.iloc[-1] > sma_50.iloc[-1]:
            sentiment += 0.3
        else:
            sentiment -= 0.3
        
        # Price vs 20 SMA
        if current_price > sma_20.iloc[-1]:
            sentiment += 0.2
        else:
            sentiment -= 0.2
        
        # Recent trend (last 10 candles)
        recent_change = (df["close"].iloc[-1] - df["close"].iloc[-10]) / df["close"].iloc[-10]
        if recent_change > 0.02:
            sentiment += 0.3
        elif recent_change < -0.02:
            sentiment -= 0.3
        else:
            sentiment += recent_change * 10  # Scale smaller changes
        
        # RSI analysis
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        if current_rsi > 70:
            sentiment -= 0.2  # Overbought - bearish
        elif current_rsi < 30:
            sentiment += 0.2  # Oversold - bullish
        else:
            sentiment += (50 - current_rsi) / 100  # Neutral zone
        
        # Normalize to -1 to 1 range
        return max(-1.0, min(1.0, sentiment))

    def get_market_sentiment(self, symbol="EUR/USD"):
        """
        Get market sentiment for a given symbol.
        Returns a dictionary with overall score, label, and basic analysis.
        """
        df = self._fetch_price_data(symbol=symbol, interval="1h")
        
        if df is None or len(df) < 50:
            return {"overall_score": 0.0, "label": "Neutral", "top_headlines": []}
        
        # Calculate technical sentiment
        tech_sentiment = self._calculate_technical_sentiment(df)
        
        # Get recent price movement for headlines
        recent_high = df["high"].iloc[-1]
        recent_low = df["low"].iloc[-1]
        recent_close = df["close"].iloc[-1]
        
        # Generate synthetic headlines based on technical analysis
        top_headlines = []
        if tech_sentiment > 0.2:
            top_headlines.append({
                "title": f"{symbol} showing bullish momentum - Price action positive",
                "overall_sentiment_score": tech_sentiment
            })
        elif tech_sentiment < -0.2:
            top_headlines.append({
                "title": f"{symbol} under bearish pressure - Technical weakness",
                "overall_sentiment_score": tech_sentiment
            })
        else:
            top_headlines.append({
                "title": f"{symbol} consolidating - Sideways market conditions",
                "overall_sentiment_score": tech_sentiment
            })
        
        label = "Neutral"
        if tech_sentiment > 0.15:
            label = "Bullish"
        elif tech_sentiment < -0.15:
            label = "Bearish"
        
        return {
            "overall_score": float(tech_sentiment),
            "label": label,
            "top_headlines": top_headlines,
            "count": 1,
            "price_data": {
                "current": float(recent_close),
                "high": float(recent_high),
                "low": float(recent_low)
            }
        }

    def analyze_web_trends(self, symbol="EUR/USD"):
        """
        Main entry point for web trend analysis.
        Returns sentiment analysis based on technical indicators.
        """
        return self.get_market_sentiment(symbol)
