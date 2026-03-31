"""
TradeAI — Risk Management Engine
Position sizing, SL/TP calculation, trade filters, and constraint checks.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _load_cfg() -> dict:
    import yaml
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


class RiskEngine:
    """
    Dedicated risk management layer.

    Responsibilities
    ----------------
    - Position sizing (% risk per trade)
    - Stop loss & take profit calculation (ATR-based)
    - Trade filter checks (volatility, confidence, multi-TF conflict)
    - Portfolio constraint checks (max trades/day, drawdown, exposure)
    """

    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = cfg or _load_cfg()
        risk_cfg = self.cfg.get("risk", {})
        self.risk_per_trade: float = risk_cfg.get("risk_per_trade", 0.01)
        self.rr_ratio: float = risk_cfg.get("default_rr_ratio", 2.0)
        self.atr_sl_multiplier: float = risk_cfg.get("atr_sl_multiplier", 1.5)
        self.atr_low_vol_threshold: float = risk_cfg.get("atr_low_volatility_threshold", 0.0005)
        self.min_confidence: float = risk_cfg.get("min_confidence_threshold", 0.55)
        self.max_trades_per_day: int = risk_cfg.get("max_trades_per_day", 5)
        self.max_drawdown_limit: float = risk_cfg.get("max_drawdown_limit", 0.15)
        self.max_exposure_per_pair: float = risk_cfg.get("max_exposure_per_pair", 0.20)

    # ── Position Sizing ────────────────────────────────────────────────────

    def calculate_position_size(
        self,
        account_balance: float,
        stop_loss_distance: float,
        risk_pct: Optional[float] = None,
    ) -> float:
        """
        Risk-adjusted position sizing.

        Formula: position_size = (balance × risk_pct) / stop_loss_distance

        Parameters
        ----------
        account_balance   : Total account balance in quote currency
        stop_loss_distance: Distance from entry to stop loss (in price units)
        risk_pct          : Fraction of balance to risk (default from config)

        Returns
        -------
        float: Units/lots to trade (rounded to 2 decimal places)
        """
        if stop_loss_distance <= 0:
            logger.warning("stop_loss_distance must be > 0; returning 0.")
            return 0.0
        rp = risk_pct or self.risk_per_trade
        amount_at_risk = account_balance * rp
        position_size = amount_at_risk / stop_loss_distance
        return round(position_size, 2)

    # ── Stop Loss & Take Profit ──────────────────────────────────────────

    def calculate_sl_tp(
        self,
        signal: str,
        current_price: float,
        atr: float,
        rr_ratio: Optional[float] = None,
        swing_levels: Optional[Tuple[float, float]] = None,
    ) -> Tuple[float, float]:
        """
        Calculate ATR-based Stop Loss and Take Profit.

        Parameters
        ----------
        signal       : "BUY" or "SELL"
        current_price: Current market price
        atr          : Average True Range value
        rr_ratio     : Reward-to-risk ratio (default from config)
        swing_levels : Optional (swing_low, swing_high) to override ATR-based SL

        Returns
        -------
        (stop_loss, take_profit) prices
        """
        rr = rr_ratio or self.rr_ratio
        sl_distance = atr * self.atr_sl_multiplier

        if swing_levels:
            swing_low, swing_high = swing_levels
            if signal == "BUY":
                sl = swing_low
                sl_distance = max(current_price - swing_low, sl_distance)
            else:
                sl = swing_high
                sl_distance = max(swing_high - current_price, sl_distance)
        else:
            sl_distance = max(sl_distance, current_price * 0.001)  # Min 0.1% SL

        if signal == "BUY":
            stop_loss = round(current_price - sl_distance, 5)
            take_profit = round(current_price + sl_distance * rr, 5)
        else:  # SELL
            stop_loss = round(current_price + sl_distance, 5)
            take_profit = round(current_price - sl_distance * rr, 5)

        return stop_loss, take_profit

    # ── Trade Filters ──────────────────────────────────────────────────────

    def should_trade(
        self,
        signal: str,
        confidence: float,
        atr: float,
        multi_tf_signals: Optional[Dict[str, str]] = None,
    ) -> Tuple[bool, str]:
        """
        Evaluate whether a trade should be taken.

        Parameters
        ----------
        signal          : "BUY"/"SELL"/"HOLD"
        confidence      : Model confidence score [0, 1]
        atr             : Current ATR value
        multi_tf_signals: Dict of {timeframe: signal} from other timeframes

        Returns
        -------
        (should_trade: bool, reason: str)
        """
        if signal == "HOLD":
            return False, "Signal is HOLD"

        # Confidence filter
        if confidence < self.min_confidence:
            return False, f"Confidence {confidence:.2f} < threshold {self.min_confidence}"

        # Low-volatility filter
        if atr < self.atr_low_vol_threshold:
            return False, f"ATR {atr:.6f} below low-volatility threshold {self.atr_low_vol_threshold}"

        # Multi-timeframe conflict filter
        if multi_tf_signals:
            signals_list = list(multi_tf_signals.values())
            contradicting = sum(1 for s in signals_list if s != signal and s != "HOLD")
            if contradicting > len(signals_list) // 2:
                return False, f"Multi-timeframe conflict: {contradicting}/{len(signals_list)} TFs oppose signal"

        return True, "All filters passed"

    # ── Portfolio Constraints ──────────────────────────────────────────────

    def check_constraints(
        self,
        daily_trade_count: int,
        current_drawdown: float,
        current_exposure: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        Check portfolio-level risk constraints.

        Parameters
        ----------
        daily_trade_count: Number of trades taken today
        current_drawdown : Current portfolio drawdown as fraction (0.10 = 10%)
        current_exposure : Current exposure for this pair as fraction of portfolio

        Returns
        -------
        (constraint_ok: bool, reason: str)
        """
        if daily_trade_count >= self.max_trades_per_day:
            return False, f"Daily trade limit reached ({daily_trade_count}/{self.max_trades_per_day})"

        if current_drawdown >= self.max_drawdown_limit:
            return False, (
                f"Max drawdown limit breached: {current_drawdown:.1%} >= {self.max_drawdown_limit:.1%}"
            )

        if current_exposure is not None and current_exposure >= self.max_exposure_per_pair:
            return False, (
                f"Pair exposure limit reached: {current_exposure:.1%} >= {self.max_exposure_per_pair:.1%}"
            )

        return True, "Constraints OK"

    # ── Full Risk Assessment ───────────────────────────────────────────────

    def assess(
        self,
        prediction: dict,
        account_balance: float = 10_000.0,
        daily_trade_count: int = 0,
        current_drawdown: float = 0.0,
    ) -> dict:
        """
        Compute a complete risk assessment from a prediction dict.

        Parameters
        ----------
        prediction      : Output from ForexPredictor.predict()
        account_balance : Current account equity
        daily_trade_count: Trades taken today
        current_drawdown : Current portfolio drawdown fraction

        Returns
        -------
        dict with: stop_loss, take_profit, position_size, risk_level,
                   should_trade, filter_reason, constraint_reason
        """
        signal = prediction.get("signal", "HOLD")
        confidence = prediction.get("confidence", 0.0)
        current_price = prediction.get("current_price", 0.0)
        indicators = prediction.get("indicators", {})
        atr = indicators.get("atr", current_price * 0.002)

        # SL / TP
        stop_loss, take_profit = self.calculate_sl_tp(signal, current_price, atr)
        sl_distance = abs(current_price - stop_loss)

        # Position size
        position_size = self.calculate_position_size(account_balance, sl_distance)

        # Risk level
        risk_level = prediction.get("risk_level", "MEDIUM")

        # Trade filters
        tradeable, filter_reason = self.should_trade(signal, confidence, atr)

        # Portfolio constraints
        constraint_ok, constraint_reason = self.check_constraints(
            daily_trade_count, current_drawdown
        )

        return {
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "sl_distance": round(sl_distance, 5),
            "position_size": position_size,
            "risk_level": risk_level,
            "risk_reward_ratio": round(self.rr_ratio, 2),
            "amount_at_risk": round(account_balance * self.risk_per_trade, 2),
            "should_trade": tradeable and constraint_ok,
            "filter_reason": filter_reason if not tradeable else constraint_reason,
            "atr_used": round(atr, 6),
        }
