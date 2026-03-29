import pandas as pd
from MLmodels.Forex.Data.alphavantage import AlphaVantageClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class WebAnalyst:
    """
    Agent for aggregating and summarizing sentiment from news and APIs.
    """
    def __init__(self, api_key=None):
        self.client = AlphaVantageClient(api_key)
        self.analyzer = SentimentIntensityAnalyzer()

    def get_market_sentiment(self, symbol="EUR/USD"):
        from_sym, to_sym = symbol.split("/")
        # Alpha Vantage News
        news_df = self.client.get_news_sentiment(tickers=f"FOREX:{from_sym}")
        
        if news_df.empty:
            return {"overall_score": 0.0, "label": "Neutral", "top_headlines": []}
            
        # Top 5 news headlines
        top_headlines = news_df[["title", "overall_sentiment_score"]].head(5).to_dict("records")
        avg_score = news_df["overall_sentiment_score"].astype(float).mean()
        
        label = "Neutral"
        if avg_score > 0.15: label = "Bullish"
        elif avg_score < -0.15: label = "Bearish"
        
        return {
            "overall_score": float(avg_score),
            "label": label,
            "top_headlines": top_headlines,
            "count": len(news_df)
        }

    def analyze_web_trends(self, symbol="EUR/USD"):
        """
        Placeholder for future web scraping if API is insufficient.
        """
        # For now, we rely on Alpha Vantage as it is specialized for Forex
        return self.get_market_sentiment(symbol)
