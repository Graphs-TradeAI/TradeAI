import sys
import os
import pandas as pd

# Add project root to path
sys.path.append("/home/job/Desktop/projects/TradeAI")

from MLmodels.Forex.Data.alphavantage import AlphaVantageClient
from MLmodels.Forex.Data.processing import build_forex_feature_set
from AgentApp.selection_agent import ModelSelectionAgent

def test_pipeline():
    print("Testing AlphaVantage Data Fetching...")
    client = AlphaVantageClient() # Uses 'demo' key by default
    try:
        df = client.get_forex_history("EUR", "USD", interval="5min", outputsize="compact")
        print(f"Fetched {len(df)} rows of EUR/USD data.")
        
        print("Testing Feature Engineering...")
        df_features = build_forex_feature_set(df, symbol="EUR/USD", timeframe="5min")
        print(f"Engineered {len(df_features)} rows with {len(df_features.columns)} features.")
        print(f"Sample features: {list(df_features.columns[:10])}")
        
        print("Testing Selection Agent Structure...")
        agent = ModelSelectionAgent()
        # We can't run full analysis without models/scalers, but we can check if it initializes
        print("Selection Agent initialized successfully.")
        
        return True
    except Exception as e:
        print(f"Pipeline Test Failed: {e}")
        return False

if __name__ == "__main__":
    test_pipeline()
