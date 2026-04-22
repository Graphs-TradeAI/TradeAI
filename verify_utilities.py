
import os
import sys
import logging
from decouple import config

# Add current directory to path so we can import Forex
sys.path.append(os.getcwd())

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def verify_ml_pipeline():
    try:
        from Forex.predictor import ForexPredictor
        from Forex.risk_engine import RiskEngine
        from Forex.engine import BacktestEngine
        
        api_key = config("TWELVE_DATA_API_KEY")
        symbol = "AUD/USD"
        timeframe = "1h"
        account_balance = 10000.0
        
        logger.info(f"--- Starting Utility Verification for {symbol} ({timeframe}) ---")
        
        # 1. Predictor Test
        logger.info("Initializing ForexPredictor...")
        predictor = ForexPredictor(api_key=api_key)
        
        logger.info(f"Running prediction for {symbol}...")
        
        # DEBUG: Check features
        df_raw = predictor.loader.load_for_inference(symbol, timeframe, output_size=500)
        from Forex.indicators import build_features, get_feature_columns
        df_feat = build_features(df_raw, symbol=symbol, timeframe=timeframe)
        feature_cols = get_feature_columns(df_feat)
        logger.info(f"Feature count: {len(feature_cols)}")
        logger.info(f"Features: {feature_cols}")
        
        prediction = predictor.predict(symbol=symbol, timeframe=timeframe)
        
        logger.info("Prediction successful:")
        logger.info(f"  Signal: {prediction['signal']}")
        logger.info(f"  Confidence: {prediction['confidence']}")
        logger.info(f"  Current Price: {prediction['current_price']}")
        logger.info(f"  Predicted Price: {prediction['predicted_price']}")
        logger.info(f"  Regime: {prediction['regime']}")
        
        # 2. Risk Engine Test
        logger.info("Initializing RiskEngine...")
        risk_engine = RiskEngine()
        
        logger.info(f"Assessing risk for {symbol} with balance ${account_balance}...")
        risk = risk_engine.assess(prediction, account_balance=account_balance)
        
        logger.info("Risk calculation successful:")
        logger.info(f"  Stop Loss: {risk['stop_loss']}")
        logger.info(f"  Take Profit: {risk['take_profit']}")
        logger.info(f"  Risk Reward: {risk['risk_reward_ratio']:.2f}")
        logger.info(f"  Position Size: {risk['position_size']}")
        logger.info(f"  Should Trade: {risk['should_trade']}")
        
        # 3. Backtest Engine Test
        logger.info("Initializing BacktestEngine...")
        backtest_engine = BacktestEngine(api_key=api_key)
        
        logger.info(f"Running short backtest (100 bars) for {symbol}...")
        # Note: BacktestEngine.run might take some time depending on data fetching
        backtest_result = backtest_engine.run(
            symbol=symbol, 
            timeframe=timeframe, 
            account_balance=account_balance,
            lookback_days=30 # Short lookback for quick verification
        )
        
        report = backtest_result.get("report", {})
        logger.info("Backtest successful:")
        logger.info(f"  Total Trades: {len(backtest_result.get('trade_log', []))}")
        logger.info(f"  Win Rate: {report.get('win_rate', 0)*100:.1f}%")
        logger.info(f"  Final Balance: ${backtest_result.get('equity_curve', [])[-1]:.2f}")
        
        logger.info("--- All Utilities Verified Successfully ---")
        return True
        
    except Exception as e:
        logger.error(f"Verification Failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = verify_ml_pipeline()
    sys.exit(0 if success else 1)
