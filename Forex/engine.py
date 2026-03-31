

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .loader import DataLoader
from .indicators import (
    build_features,
    get_feature_columns,
    get_indicator_snapshot,
)
from .registry import ModelNotFoundError, ModelRegistry
from .risk_engine import RiskEngine
from .evaluator import Evaluator
from .preprocessor import Preprocessor

logger = logging.getLogger(__name__)


def _load_cfg() -> dict:
    import os, yaml

    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")

    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


class BacktestEngine:


    def __init__(
        self,
        api_key: str,
        models_root: Optional[str] = None,
        cfg: Optional[dict] = None,
    ):
        self.cfg = cfg or _load_cfg()
        self.api_key = api_key
        bt_cfg = self.cfg.get("backtesting", {})
        train_cfg = self.cfg.get("training", {})

        self.seq_length: int = train_cfg.get("sequence_length", 60)
        self.slippage: float = bt_cfg.get("slippage_pips", 0.5) / 10_000  # Convert pips → price
        self.output_dir: str = bt_cfg.get("output_dir", "backtesting_results")

        self.registry = ModelRegistry(models_root, self.cfg)
        self.loader = DataLoader(api_key, self.cfg, use_cache=True)
        self.risk_engine = RiskEngine(self.cfg)
        self.evaluator = Evaluator()



    def run(
        self,
        symbol: str = "AUD/USD",
        timeframe: str = "1h",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        account_balance: float = 10_000.0,
        lookback_days: int = 365,
        step_size: int = 1,   # Evaluate every N candles (1 = every candle)
    ) -> Dict:

        logger.info("Backtest: %s %s | balance=%.2f", symbol, timeframe, account_balance)

        # 1. Load full historical data
        df_raw = self.loader.load(
            symbol, timeframe,
            lookback_days=lookback_days,
            start_date=start_date,
            end_date=end_date,
        )
        logger.info("Loaded %d rows for backtest", len(df_raw))

        # 2. Build full feature set (no look-ahead because we slice per candle)
        df_feat = build_features(df_raw, symbol=symbol, timeframe=timeframe, cfg=self.cfg)
        feature_cols = get_feature_columns(df_feat)

        # 3. Load model + scaler
        try:
            model, source = self.registry.load_model(symbol, timeframe)
            scaler = self.registry.load_scaler(symbol, timeframe)
        except ModelNotFoundError as exc:
            raise RuntimeError(f"Cannot run backtest: {exc}")

        preprocessor = Preprocessor(self.registry.models_root, self.cfg)
        preprocessor.scaler = scaler
        preprocessor._fitted = True

        # Scale the entire dataset once (no look-ahead in scaling is acceptable here
        # since we already fit the scaler during training on training data only)
        scaled = preprocessor.transform(df_feat, feature_cols)

        # 4. Walk-forward simulation
        equity = account_balance
        equity_curve: List[float] = [equity]
        trade_log: List[dict] = []
        daily_trades: Dict[str, int] = {}

        n = len(df_feat)
        min_idx = self.seq_length  # Need at least seq_length rows before first prediction

        for i in range(min_idx, n - 1, step_size):
            # Slice sequence up to current candle (no future data)
            seq = scaled[i - self.seq_length : i]
            X = seq[np.newaxis, ...]

            # Generate prediction
            raw = model.predict(X, verbose=0)
            if isinstance(raw, list):
                dir_prob = float(raw[0][0][0])
            else:
                dir_prob = float(raw[0][0])

            signal = "BUY" if dir_prob > 0.5 else "SELL"
            confidence = abs(dir_prob - 0.5) * 2

            current_row = df_feat.iloc[i]
            entry_price = float(current_row["close"]) + (
                self.slippage if signal == "BUY" else -self.slippage
            )
            atr = float(current_row.get("atr", entry_price * 0.002))

            # Risk management filters
            prediction_dict = {
                "signal": signal,
                "confidence": confidence,
                "current_price": entry_price,
                "risk_level": "MEDIUM",
                "indicators": {"atr": atr},
            }
            trade_date = str(current_row["timestamp"])[:10]
            daily_count = daily_trades.get(trade_date, 0)
            current_dd = self._current_drawdown(equity_curve)

            risk = self.risk_engine.assess(prediction_dict, equity, daily_count, current_dd)

            if not risk["should_trade"]:
                continue

            stop_loss = risk["stop_loss"]
            take_profit = risk["take_profit"]
            position_size = risk["position_size"]

            # Simulate trade outcome (look at future candles)
            outcome, exit_price, exit_idx = self._simulate_trade(
                df_feat, i + 1, signal, entry_price, stop_loss, take_profit
            )

            # Calculate PnL
            if signal == "BUY":
                pnl = (exit_price - entry_price) * position_size
            else:
                pnl = (entry_price - exit_price) * position_size

            equity += pnl
            equity = max(equity, 0.0)  # No margin calls below 0
            equity_curve.append(round(equity, 2))

            daily_trades[trade_date] = daily_count + 1

            trade_log.append({
                "entry_time": str(current_row["timestamp"]),
                "exit_time": str(df_feat.iloc[min(exit_idx, n - 1)]["timestamp"]),
                "symbol": symbol,
                "timeframe": timeframe,
                "signal": signal,
                "entry_price": round(entry_price, 5),
                "exit_price": round(exit_price, 5),
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "position_size": position_size,
                "outcome": outcome,
                "pnl": round(pnl, 4),
                "equity": round(equity, 2),
                "confidence": round(confidence, 4),
            })

        # 5. Compute report metrics
        report = self._compute_report(trade_log, equity_curve, account_balance, symbol, timeframe)
        report["model_source"] = source

        # 6. Save results
        self._save_results(report, trade_log, equity_curve, symbol, timeframe)

        logger.info(
            "Backtest done — %d trades | Return: %.2f%% | Sharpe: %.3f | MaxDD: %.2f%%",
            report["n_trades"],
            report["total_return_pct"],
            report["sharpe_ratio"],
            report["max_drawdown_pct"],
        )
        return {
            "report": report,
            "trade_log": trade_log,
            "equity_curve": equity_curve,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Internal simulation
    # ──────────────────────────────────────────────────────────────────────

    def _simulate_trade(
        self,
        df: pd.DataFrame,
        start_idx: int,
        signal: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        max_hold: int = 50,
    ):
     
        n = len(df)
        for j in range(start_idx, min(start_idx + max_hold, n)):
            row = df.iloc[j]
            high = float(row["high"])
            low = float(row["low"])

            if signal == "BUY":
                if low <= stop_loss:
                    return "LOSS", stop_loss, j
                if high >= take_profit:
                    return "WIN", take_profit, j
            else:  # SELL
                if high >= stop_loss:
                    return "LOSS", stop_loss, j
                if low <= take_profit:
                    return "WIN", take_profit, j

        # Timeout: exit at last close
        close = float(df.iloc[min(start_idx + max_hold - 1, n - 1)]["close"])
        return "TIMEOUT", close, min(start_idx + max_hold - 1, n - 1)

  
    def _compute_report(
        self,
        trade_log: list,
        equity_curve: list,
        initial_balance: float,
        symbol: str,
        timeframe: str,
    ) -> dict:
        if not trade_log:
            return {
                "symbol": symbol, "timeframe": timeframe,
                "n_trades": 0, "total_return_pct": 0,
                "win_rate": 0, "sharpe_ratio": 0,
                "max_drawdown_pct": 0, "profit_factor": 0,
                "initial_balance": initial_balance, "final_balance": initial_balance,
            }

        pnls = np.array([t["pnl"] for t in trade_log])
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]

        win_rate = len(wins) / len(pnls)
        final_balance = equity_curve[-1]
        total_return = (final_balance - initial_balance) / initial_balance * 100

        # Sharpe from trade PnL
        period_returns = pnls / initial_balance
        sharpe = float(
            np.mean(period_returns) / (np.std(period_returns) + 1e-9)
            * np.sqrt(252)
        )

        # Max drawdown
        equity_arr = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_arr)
        dd = (equity_arr - running_max) / (running_max + 1e-9)
        max_dd_pct = float(np.min(dd)) * 100

        # Profit factor
        gross_profit = float(wins.sum()) if len(wins) > 0 else 0
        gross_loss = float(abs(losses.sum())) if len(losses) > 0 else 1
        profit_factor = gross_profit / max(gross_loss, 1e-9)

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "n_trades": len(trade_log),
            "n_wins": len(wins),
            "n_losses": len(losses),
            "win_rate": round(win_rate, 4),
            "total_return_pct": round(total_return, 4),
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown_pct": round(max_dd_pct, 4),
            "profit_factor": round(profit_factor, 4),
            "gross_profit": round(gross_profit, 4),
            "gross_loss": round(gross_loss, 4),
            "initial_balance": initial_balance,
            "final_balance": round(final_balance, 2),
            "avg_pnl": round(float(np.mean(pnls)), 4),
        }

    @staticmethod
    def _current_drawdown(equity_curve: list) -> float:
        if not equity_curve:
            return 0.0
        arr = np.array(equity_curve)
        peak = np.max(arr)
        current = arr[-1]
        return float((peak - current) / (peak + 1e-9))

    def _save_results(
        self,
        report: dict,
        trade_log: list,
        equity_curve: list,
        symbol: str,
        timeframe: str,
    ) -> None:
        pair_tag = symbol.replace("/", "")
        out_dir = os.path.join(self.output_dir, pair_tag, timeframe)
        os.makedirs(out_dir, exist_ok=True)

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(out_dir, f"report_{ts}.json")
        trades_path = os.path.join(out_dir, f"trades_{ts}.json")

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        with open(trades_path, "w") as f:
            json.dump({"trade_log": trade_log, "equity_curve": equity_curve}, f, indent=2)

        logger.info("Backtest results saved to %s", out_dir)


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from decouple import config as env

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Run a forex backtest")
    parser.add_argument("--symbol", type=str, default="EUR/USD")
    parser.add_argument("--timeframe", type=str, default="1h")
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--balance", type=float, default=10000.0)
    parser.add_argument("--lookback", type=int, default=365)
    args = parser.parse_args()

    engine = BacktestEngine(api_key=env("TWELVE_DATA_API_KEY"))
    result = engine.run(
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=args.start,
        end_date=args.end,
        account_balance=args.balance,
        lookback_days=args.lookback,
    )
    print(json.dumps(result["report"], indent=2))
