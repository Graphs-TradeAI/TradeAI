import sys
import os
import pandas as pd

# Add project root to path
sys.path.append("/home/job/Desktop/projects/TradeAI")

from MLmodels.Forex.Data.twelvedata import TwelveDataClient
from MLmodels.Forex.Data.processing import build_forex_feature_set
from AgentApp.selection_agent import ModelSelectionAgent

def test_pipeline():
    print("Testing Twelve Data API Connection...")
    
    # Get API key from environment or use the one from .env
    api_key = os.environ.get("TWELVE_DATA_API_KEY", "fb941e0ebad44b4caa431760fcc5bef3")
    client = TwelveDataClient(api_key)
    
    try:
        # Test fetching EUR/USD data
        print("Fetching EUR/USD 5min data...")
        df = client.get_forex_history(symbol="EUR/USD", interval="5min", output_size=100)
        print(f"✅ Fetched {len(df)} rows of EUR/USD data.")
        print(f"   Columns: {list(df.columns)}")
        
        print("\nTesting Feature Engineering...")
        df_features = build_forex_feature_set(df, symbol="EUR/USD", timeframe="5min")
        print(f"✅ Engineered {len(df_features)} rows with {len(df_features.columns)} features.")
        print(f"   Sample features: {list(df_features.columns[:10])}")
        
        print("\nTesting Model Selection Agent...")
        agent = ModelSelectionAgent(api_key=api_key)
        print("✅ Selection Agent initialized successfully.")
        
        print("\nTesting Web Analyst...")
        from AgentApp.web_analyst import WebAnalyst
        analyst = WebAnalyst(api_key=api_key)
        sentiment = analyst.get_market_sentiment(symbol="EUR/USD")
        print(f"✅ Web Analyst sentiment: {sentiment.get('label', 'Unknown')}")
        
        print("\n" + "="*50)
        print("All pipeline tests passed! ✅")
        print("="*50)
        
        return True
    except Exception as e:
        print(f"❌ Pipeline Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_inference():
    """Test the full model inference pipeline."""
    print("\n" + "="*50)
    print("Testing Model Inference Pipeline...")
    print("="*50)
    
    api_key = os.environ.get("TWELVE_DATA_API_KEY", "fb941e0ebad44b4caa431760fcc5bef3")
    
    try:
        from AgentApp.inference import ModelInference
        inference = ModelInference(api_key=api_key)
        
        # Test prediction (this requires trained models)
        # For now, just verify initialization
        print("✅ ModelInference initialized successfully.")
        
        return True
    except Exception as e:
        print(f"❌ Inference Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_pipeline()
    if success:
        test_model_inference()
