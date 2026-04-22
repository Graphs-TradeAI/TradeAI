import os
import pytest

from AgentApp.inference import ModelInference


def test_unit_inference_uses_audusd_defaults(monkeypatch):
    """Unit test: mock predictor and risk engine to validate output shape and AUD/USD default."""

    # Dummy predictor returning expected fields
    class DummyPredictor:
        def predict(self, symbol, timeframe, output_size=500):
            assert symbol == "AUD/USD"
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "signal": "BUY",
                "confidence": 0.75,
                "direction_probability": 0.85,
                "predicted_price": 0.68818,
                "current_price": 0.68750,
                "regime": "range",
                "risk_level": "MEDIUM",
                "indicators": {},
                "model_source": "unit-test",
                "timestamp": "now",
            }

    class DummyRisk:
        def assess(self, pred, **kwargs):
            return {
                "stop_loss": 0.686,
                "take_profit": 0.690,
                "position_size": 1000,
                "sl_distance": 0.0015,
                "risk_reward_ratio": 2.0,
                "amount_at_risk": 10.0,
                "should_trade": True,
                "filter_reason": "",
            }

    inf = ModelInference(api_key=None)
    # Inject mocks
    inf._predictor = DummyPredictor()
    inf._risk_engine = DummyRisk()

    out = inf.predict()

    assert out["symbol"] == "AUD/USD"
    assert out["timeframe"] == "1h"
    assert out["signal"] in {"BUY", "SELL", "HOLD"}
    assert isinstance(out["confidence"], float)
    assert "stop_loss" in out and "take_profit" in out
    assert out["position_size"] == 1000


@pytest.mark.integration
def test_integration_inference_audusd_if_keys_present():
    """Integration test: runs real inference only if TWELVE_DATA_API_KEY exists."""
    td_key = os.getenv("TWELVE_DATA_API_KEY")
    if not td_key:
        pytest.skip("TWELVE_DATA_API_KEY not set; skipping integration test")

    inf = ModelInference(api_key=td_key)
    res = inf.predict(symbol="AUD/USD", timeframe="1h")

    # Basic sanity checks
    assert res["symbol"] == "AUD/USD"
    assert "signal" in res
    assert "current_price" in res
    assert "predicted_price" in res
