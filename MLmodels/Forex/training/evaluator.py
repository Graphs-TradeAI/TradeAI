"""
TradeAI — Training Evaluator
Computes all required metrics from model predictions.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Computes comprehensive evaluation metrics for forex model predictions.

    Metrics
    -------
    - MAE, MSE  (price regression)
    - Directional Accuracy, Win Rate  (direction classification)
    - Sharpe Ratio  (annualised, based on simulated returns)
    - Max Drawdown
    """

    def evaluate_classification(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        prices: Optional[np.ndarray] = None,
        threshold: float = 0.5,
        annualise_factor: float = 252,
    ) -> Dict:
        """
        Evaluate a direction-classification model.

        Parameters
        ----------
        y_true           : Ground-truth binary labels (0/1)
        y_prob           : Predicted probability of upward movement [0,1]
        prices           : Actual close prices aligned with y_true (for Sharpe/drawdown)
        threshold        : Decision threshold for class assignment
        annualise_factor : Trading periods per year (252 for daily, 8760 for 1h etc.)

        Returns
        -------
        dict of metric names → values
        """
        y_pred = (y_prob >= threshold).astype(int)

        # ── Direction Accuracy ────────────────────────────────────────────
        dir_acc = float(np.mean(y_pred == y_true))

        # ── Win Rate ─────────────────────────────────────────────────────
        # Among predicted UP signals, how many were correct
        up_mask = y_pred == 1
        win_rate = float(np.mean(y_true[up_mask])) if up_mask.sum() > 0 else 0.0

        # ── Sharpe Ratio (signal-based) ───────────────────────────────────
        sharpe = 0.0
        max_dd = 0.0
        if prices is not None and len(prices) > 1:
            # Simulate returns: go long on predicted BUY signals, short on SELL
            signals = np.where(y_pred == 1, 1.0, -1.0)
            price_returns = np.diff(prices) / prices[:-1]
            # Align lengths
            min_len = min(len(signals), len(price_returns))
            strat_returns = signals[:min_len] * price_returns[:min_len]
            sharpe = self._sharpe_ratio(strat_returns, annualise_factor)
            max_dd = self._max_drawdown(strat_returns)

        metrics = {
            "directional_accuracy": round(dir_acc, 4),
            "win_rate": round(win_rate, 4),
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown": round(max_dd, 4),
            "total_predictions": int(len(y_true)),
            "buy_signals": int(up_mask.sum()),
        }

        logger.info(
            "Classification metrics — Dir Acc: %.3f | Win Rate: %.3f | Sharpe: %.3f | MaxDD: %.3f",
            dir_acc, win_rate, sharpe, max_dd,
        )
        return metrics

    def evaluate_regression(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        current_prices: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Evaluate a price-regression model.

        Parameters
        ----------
        y_true         : Actual next-close prices
        y_pred         : Predicted next-close prices
        current_prices : Current close prices (for computing returns-based Sharpe)
        """
        mae = float(mean_absolute_error(y_true, y_pred))
        mse = float(mean_squared_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))

        # Directional accuracy from regression output
        if current_prices is not None and len(current_prices) == len(y_true):
            true_dir = (y_true > current_prices).astype(int)
            pred_dir = (y_pred > current_prices).astype(int)
            dir_acc = float(np.mean(true_dir == pred_dir))
        else:
            dir_acc = 0.0

        metrics = {
            "mae": round(mae, 6),
            "mse": round(mse, 8),
            "rmse": round(rmse, 6),
            "directional_accuracy": round(dir_acc, 4),
        }
        logger.info("Regression metrics — MAE: %.6f | RMSE: %.6f | Dir Acc: %.3f", mae, rmse, dir_acc)
        return metrics

    def evaluate_full(
        self,
        y_true_dir: np.ndarray,
        y_prob: np.ndarray,
        y_true_price: Optional[np.ndarray] = None,
        y_pred_price: Optional[np.ndarray] = None,
        current_prices: Optional[np.ndarray] = None,
        annualise_factor: float = 252,
    ) -> Dict:
        """Combined classification + regression metrics."""
        metrics = self.evaluate_classification(
            y_true_dir, y_prob,
            prices=current_prices,
            annualise_factor=annualise_factor,
        )
        if y_true_price is not None and y_pred_price is not None:
            reg_metrics = self.evaluate_regression(y_true_price, y_pred_price, current_prices)
            metrics.update(reg_metrics)
        return metrics

    # ──────────────────────────────────────────────────────────────────────
    # Static helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _sharpe_ratio(returns: np.ndarray, annualise_factor: float = 252) -> float:
        """Annualised Sharpe ratio (assumes risk-free rate = 0)."""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        return float(np.mean(returns) / np.std(returns) * np.sqrt(annualise_factor))

    @staticmethod
    def _max_drawdown(returns: np.ndarray) -> float:
        """Maximum peak-to-trough drawdown from a returns array."""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return float(np.min(drawdown))

    @staticmethod
    def annualise_factor_for_timeframe(timeframe: str) -> float:
        """Return typical annualisation factor for a given timeframe."""
        mapping = {
            "1min": 525_600,
            "5min": 105_120,
            "15min": 35_040,
            "30min": 17_520,
            "1h": 8_760,
            "4h": 2_190,
            "1day": 252,
        }
        return float(mapping.get(timeframe, 252))
