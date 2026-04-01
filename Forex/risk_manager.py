import numpy as np


class RiskManager:
    """
    Handles position sizing, stop loss, and take profit logic
    for forex trading based on ATR and account risk.
    """

    def __init__(self, account_balance=1000, risk_per_trade=0.01):
        self.account_balance = account_balance
        self.risk_per_trade = risk_per_trade

    def position_size(self, entry_price, stop_loss_price):
        """
        Calculates lot size based on risk percentage.
        """
        risk_amount = self.account_balance * self.risk_per_trade

        pip_risk = abs(entry_price - stop_loss_price)

        if pip_risk == 0:
            return 0

        size = risk_amount / pip_risk
        return round(size, 4)

    def compute_sl_tp(self, entry_price, atr, direction, sl_mult=1.5, tp_mult=3.0):
        """
        ATR-based stop loss and take profit.
        direction: 1 = BUY, -1 = SELL
        """
        sl_distance = atr * sl_mult
        tp_distance = atr * tp_mult

        if direction == 1:
            sl = entry_price - sl_distance
            tp = entry_price + tp_distance
        else:
            sl = entry_price + sl_distance
            tp = entry_price - tp_distance

        return sl, tp

    def validate_trade(self, confidence, min_threshold=0.6):
        """
        Filters weak signals.
        """
        return abs(confidence) >= min_threshold