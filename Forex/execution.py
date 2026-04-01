import logging

logger = logging.getLogger(__name__)

class ExecutionEngine:
    """
    Formats the finalized trade decision for the UI and LLM layers.
    Only called AFTER risk validation.
    """

    def __init__(self, debug=True):
        self.debug = debug

    def prepare_trade_packet(self, symbol, timeframe, prediction, risk_data):
        """
        prediction: dict with signal, confidence
        risk_data: dict with sl, tp, size, atr
        """
        
        signal = prediction.get("signal", "HOLD")
        confidence = prediction.get("confidence", 0.0)
        
        packet = {
            "symbol": symbol,
            "timeframe": timeframe,
            "signal": signal,
            "confidence": round(confidence, 4),
            "current_price": prediction.get("current_price"),
            "stop_loss": risk_data.get("sl"),
            "take_profit": risk_data.get("tp"),
            "position_size": risk_data.get("size"),
            "atr": risk_data.get("atr"),
            "timestamp": prediction.get("timestamp"),
            "status": "APPROVED" if signal != "HOLD" else "FILTERED"
        }
        
        if self.debug:
            logger.info(f"Execution Packet: {symbol} {signal} @ {packet['current_price']}")
            
        return packet
