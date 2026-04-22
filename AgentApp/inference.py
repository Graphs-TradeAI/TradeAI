
from __future__ import annotations

import logging
from typing import Optional

from django.conf import settings

logger = logging.getLogger(__name__)


class ModelInference:


    def __init__(self, api_key: str = None):
        self.api_key = api_key or getattr(settings, "TWELVE_DATA_API_KEY", None)
        self._predictor = None
        self._risk_engine = None

    # ── Lazy-loaded sub-components ─────────────────────────────────────────

    @property
    def predictor(self):
        if self._predictor is None:
            from Forex.predictor import ForexPredictor
            self._predictor = ForexPredictor(api_key=self.api_key)
        return self._predictor

    @property
    def risk_engine(self):
        if self._risk_engine is None:
            from Forex.risk_engine import RiskEngine
            self._risk_engine = RiskEngine()
        return self._risk_engine

    # ── Public API ─────────────────────────────────────────────────────────

    def predict(
        self,
        symbol: str = "AUD/USD",
        timeframe: str = "1h",
        account_balance: float = 10_000.0,
        daily_trade_count: int = 0,
        current_drawdown: float = 0.0,
        output_size: int = 500,
    ) -> dict:
      
        # 1. Model prediction
        pred = self.predictor.predict(
            symbol=symbol,
            timeframe=timeframe,
            output_size=output_size,
        )

        # 2. Risk assessment
        risk = self.risk_engine.assess(
            pred,
            account_balance=account_balance,
            daily_trade_count=daily_trade_count,
            current_drawdown=current_drawdown,
        )

        # 3. Merge into unified response dict
        indicators = pred.get("indicators", {})
        result = {
        
            "symbol": symbol,
            "timeframe": timeframe,
            "signal": pred["signal"],
            "confidence": pred["confidence"],
            "confidence_pct": f"{pred['confidence'] * 100:.1f}%",
            "direction_probability": pred.get("direction_probability"),
            "predicted_price": pred.get("predicted_price"),
            "predicted_close": pred.get("predicted_price"),  # LLM compat
            "current_price": pred["current_price"],
            "regime": pred.get("regime", "range"),
            "risk_level": pred.get("risk_level", "MEDIUM"),
            "model_source": pred.get("model_source", ""),
            "timestamp": pred.get("timestamp", ""),
        
            "stop_loss": risk["stop_loss"],
            "take_profit": risk["take_profit"],
            "sl": risk["stop_loss"],   # short aliases
            "tp": risk["take_profit"],
            "position_size": risk["position_size"],
            "sl_distance": risk["sl_distance"],
            "risk_reward_ratio": risk["risk_reward_ratio"],
            "amount_at_risk": risk["amount_at_risk"],
            "should_trade": risk["should_trade"],
            "filter_reason": risk.get("filter_reason", ""),
            # Indicators
            "indicators": indicators,
            # LLM-layer compatibility fields (old api_chat)
            "trend": (
                "bullish" if pred["signal"] == "BUY"
                else "bearish" if pred["signal"] == "SELL"
                else "neutral"
            ),
            "trend_strength": "strong" if pred["confidence"] > 0.65 else "moderate",
            "momentum": "positive" if indicators.get("macd_diff", 0) > 0 else "negative",
            "stochastic": f"K={indicators.get('stoch_k', 50):.1f}",
            "volatility": pred.get("risk_level", "MEDIUM").lower(),
        }
        return result

    def run_backtest(
        self,
        symbol: str = "AUD/USD",
        timeframe: str = "1h",
        account_balance: float = 10_000.0,
        lookback_days: int = 365,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> dict:
        """Run a full backtest via BacktestEngine."""
        from Forex.engine import BacktestEngine
        engine = BacktestEngine(api_key=self.api_key)
        return engine.run(
            symbol=symbol,
            timeframe=timeframe,
            account_balance=account_balance,
            lookback_days=lookback_days,
            start_date=start_date,
            end_date=end_date,
        )

    def calculate_model_metrics(
        self,
        symbol: str,
        timeframe: str,
        n_backtest: int = 100,
    ) -> dict:
        """
        Load saved metrics from the model registry.
        Falls back to a lightweight backtest if no saved metrics exist.
        """
        try:
            from Forex.registry import ModelRegistry
            registry = ModelRegistry()
            metrics = registry.load_metrics(symbol, timeframe)
            if metrics:
                return self._normalize_metrics(metrics, n_backtest=n_backtest)
        except Exception as exc:
            logger.warning("Could not load saved metrics: %s", exc)

        try:
            result = self.run_backtest(
                symbol=symbol,
                timeframe=timeframe,
                account_balance=10_000.0,
                lookback_days=30,
                start_date=None,
                end_date=None,
            )
            report = result["report"]
            
            gross_profit = report.get("gross_profit", 0)
            gross_loss = report.get("gross_loss", 1)
            n_wins = report.get("n_wins", 1)
            n_losses = report.get("n_losses", 1)
            
            avg_win = gross_profit / max(n_wins, 1)
            avg_loss = gross_loss / max(n_losses, 1)
            risk_reward = avg_win / max(avg_loss, 1e-9)
            
            return {
                "directional_accuracy": report.get("win_rate", 0.5),
                "f1_score": report.get("win_rate", 0.5), # Approx representation
                "win_rate": report.get("win_rate", 0.0),
                "risk_reward": risk_reward,
                "rrr_ratio": risk_reward,
                "expectancy": report.get("avg_pnl", 0.0),
                "sharpe_ratio": report.get("sharpe_ratio", 0.0),
                "profit_factor": report.get("profit_factor", 0.0),
                "max_drawdown": report.get("max_drawdown_pct", 0.0) / 100.0,
                "n_backtest": report.get("n_trades", n_backtest),
            }
        except Exception as e:
            logger.error("Real-time backtest generation failed: %s", e)
            # Conservative fallback when data/model is unavailable.
            return {
                "directional_accuracy": 0.0,
                "f1_score": 0.0,
                "win_rate": 0.0,
                "risk_reward": 0.0,
                "rrr_ratio": 0.0,
                "expectancy": 0.0,
                "sharpe_ratio": 0.0,
                "profit_factor": 0.0,
                "max_drawdown": 0.0,
                "n_backtest": n_backtest or 100,
                "error": str(e),
            }

    @staticmethod
    def _normalize_metrics(metrics: dict, n_backtest: int = 100) -> dict:
        """Normalize mixed metric schemas into one stable API contract."""
        max_dd = metrics.get("max_drawdown")
        if max_dd is None and metrics.get("max_drawdown_pct") is not None:
            max_dd = float(metrics["max_drawdown_pct"]) / 100.0

        directional_accuracy = float(metrics.get("directional_accuracy", 0.0) or 0.0)
        win_rate = float(metrics.get("win_rate", directional_accuracy) or 0.0)
        sharpe_ratio = float(metrics.get("sharpe_ratio", 0.0) or 0.0)
        expectancy = float(metrics.get("expectancy", metrics.get("avg_pnl", 0.0)) or 0.0)
        risk_reward = float(metrics.get("risk_reward", metrics.get("rrr_ratio", 0.0)) or 0.0)

        return {
            "directional_accuracy": directional_accuracy,
            "f1_score": float(metrics.get("f1_score", directional_accuracy) or directional_accuracy),
            "win_rate": win_rate,
            "risk_reward": risk_reward,
            "rrr_ratio": risk_reward,
            "expectancy": expectancy,
            "sharpe_ratio": sharpe_ratio,
            "profit_factor": float(metrics.get("profit_factor", 0.0) or 0.0),
            "max_drawdown": float(max_dd or 0.0),
            "n_backtest": int(metrics.get("n_backtest", metrics.get("n_test", n_backtest)) or n_backtest),
        }
