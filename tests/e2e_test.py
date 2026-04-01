import os
import sys
import logging
from dotenv import load_dotenv

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
# Also add Forex to path
FOREX_DIR = os.path.join(BASE_DIR, "Forex")
if FOREX_DIR not in sys.path:
    sys.path.insert(0, FOREX_DIR)

from Forex.modeltrain import ForexTrainer
from AgentApp.inference import ModelInference
from AgentApp.llm_service import LLMService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("E2E_TEST")

def run_e2e_test():
    load_dotenv()
    
    symbol = "EUR/USD"
    timeframe = "1h"
    
    logger.info(f"=== Starting E2E Test for {symbol} {timeframe} ===")
    
    # 1. Training
    logger.info("--- Phase 1: Training ---")
    trainer = ForexTrainer(base_path=os.path.join(BASE_DIR, "Forex", "Models"))
    try:
        # Train for only 2 epochs for quick verification
        trainer.train_model_for_pair(symbol=symbol, timeframe=timeframe, epochs=2, batch_size=32)
        logger.info("Training completed successfully.")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return

    # 2. Inference
    logger.info("--- Phase 2: Inference ---")
    inference = ModelInference()
    try:
        result = inference.predict(symbol=symbol, timeframe=timeframe)
        logger.info(f"Inference result: {result['signal']} at {result['confidence_pct']} confidence.")
        logger.debug(f"Full result: {result}")
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return

    # 3. LLM Insight
    logger.info("--- Phase 3: LLM Insight ---")
    llm = LLMService()
    try:
        insight = llm.generate_insight(
            prediction=result,
            risk={
                "stop_loss": result["stop_loss"],
                "take_profit": result["take_profit"],
                "position_size": result["position_size"],
                "risk_level": result["risk_level"],
                "should_trade": result["should_trade"]
            },
            user_prompt=f"Should I trade {symbol} right now?"
        )
        logger.info("LLM Insight generated successfully.")
        logger.info(f"Signal: {insight['signal']}")
        logger.info(f"Reasoning: {insight['reasoning']}")
    except Exception as e:
        logger.error(f"LLM Insight generation failed: {e}")
        return

    logger.info("=== E2E Test Passed Successfully ===")

if __name__ == "__main__":
    run_e2e_test()
