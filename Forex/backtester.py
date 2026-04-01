import numpy as np
import pandas as pd


class Backtester:
    """
    Simple sequential backtesting engine for LSTM-based forex model.
    """

    def __init__(self, initial_balance=1000):
        self.initial_balance = initial_balance
        self.balance = initial_balance

        self.trades = []

    def run(self, df, risk_manager, signal_col="target_rr"):
        """
        df must contain:
        - open, high, low, close
        - atr_pct or atr_14
        - prediction signal column
        """

        for i in range(len(df) - 1):

            row = df.iloc[i]

            signal = row[signal_col]
            atr = row["atr_14"]
            price = row["close"]

            # ----------------------------
            # 1. Generate direction
            # ----------------------------
            if signal > 0.5:
                direction = 1
            elif signal < -0.5:
                direction = -1
            else:
                continue

            if not risk_manager.validate_trade(signal):
                continue

            # ----------------------------
            # 2. SL / TP
            # ----------------------------
            sl, tp = risk_manager.compute_sl_tp(
                entry_price=price,
                atr=atr,
                direction=direction
            )

            size = risk_manager.position_size(price, sl)

            next_rows = df.iloc[i+1:i+20]  # forward simulation window

            exit_price = None
            exit_reason = None

            # ----------------------------
            # 3. Simulate trade
            # ----------------------------
            for _, future in next_rows.iterrows():

                high = future["high"]
                low = future["low"]

                if direction == 1:
                    if low <= sl:
                        exit_price = sl
                        exit_reason = "SL"
                        break
                    if high >= tp:
                        exit_price = tp
                        exit_reason = "TP"
                        break

                else:
                    if high >= sl:
                        exit_price = sl
                        exit_reason = "SL"
                        break
                    if low <= tp:
                        exit_price = tp
                        exit_reason = "TP"
                        break

            if exit_price is None:
                exit_price = next_rows.iloc[-1]["close"]
                exit_reason = "TIME_EXIT"

            pnl = (exit_price - price) * size * direction

            self.balance += pnl

            self.trades.append({
                "entry": price,
                "exit": exit_price,
                "direction": direction,
                "size": size,
                "pnl": pnl,
                "reason": exit_reason,
                "balance": self.balance
            })

        return self.generate_report()

    def generate_report(self):
        df = pd.DataFrame(self.trades)

        if len(df) == 0:
            return {
                "error": "No trades executed"
            }

        return {
            "final_balance": self.balance,
            "return_pct": ((self.balance - self.initial_balance) / self.initial_balance) * 100,
            "win_rate": len(df[df["pnl"] > 0]) / len(df),
            "total_trades": len(df),
            "avg_pnl": df["pnl"].mean(),
            "max_drawdown": self.compute_drawdown(df),
            "trades": df
        }

    def compute_drawdown(self, df):
        equity = df["balance"].values
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        return drawdown.min()