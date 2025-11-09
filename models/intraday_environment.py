"""
SuperTrader.AI Intraday Trading Environment

Top-level, import-safe environment definitions:
- IntradayMarket: primary environment used by training
- IntradayTradingEnv: alias for IntradayMarket
- create_intraday_env: factory function
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import logging
from datetime import time

logger = logging.getLogger(__name__)


class IntradayMarket:
    """Simple intraday market simulator with a minimal API used across the repo."""

    def __init__(
        self,
        data: pd.DataFrame,
        initial_capital: float = 500_000,
        lot_size: int = 50,
        max_lots: int = 5,
        lookback: int = 30,
    ) -> None:
        if data is None or len(data) == 0:
            raise ValueError("data must be a non-empty DataFrame")

        # Ensure the index is a DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            data = data.copy()
            data.index = pd.to_datetime(data.index)

        self.raw_data = data.sort_index()
        self.initial_capital = float(initial_capital)
        self.lot_size = int(lot_size)
        self.max_lots = int(max_lots)
        self.lookback = int(lookback)

        # Partition into trading days by date
        self.trading_days: List[pd.DataFrame] = [df.copy() for _, df in self.raw_data.groupby(self.raw_data.index.date)]
        if len(self.trading_days) == 0:
            raise ValueError("No trading days found in data")

        # Episode state
        self.current_day_idx: int = 0
        self.current_day_data: pd.DataFrame = self.trading_days[0]
        self.current_step: int = 0
        self.position: int = 0  # -1 short, 0 flat, +1 long
        self.entry_price: float = 0.0
        self.total_pnl: float = 0.0
        self.trades: List[Dict[str, Any]] = []

        # Derived
        self.feature_cols = self._infer_feature_columns(self.current_day_data)

        # Initialize current day
        self._set_day(self.current_day_idx)

    # ----------------- Helpers -----------------
    def _infer_feature_columns(self, df: pd.DataFrame) -> List[str]:
        prefer = ["open", "high", "low", "close", "volume"]
        cols = [c for c in prefer if c in df.columns]
        if len(cols) == 0:
            numeric = df.select_dtypes(include=[np.number]).columns.tolist()
            cols = numeric[: min(10, len(numeric))]
        return cols

    def _set_day(self, day_idx: int) -> None:
        self.current_day_idx = int(day_idx)
        self.current_day_data = self.trading_days[self.current_day_idx].reset_index()
        self.current_day_data.set_index(self.current_day_data.columns[0], inplace=True)
        self.current_step = max(self.lookback, 0)
        self.position = 0
        self.entry_price = 0.0
        self.total_pnl = 0.0
        self.trades = []
        self.feature_cols = self._infer_feature_columns(self.current_day_data)

    # ----------------- Time helpers -----------------
    def _minutes_until_close(self, current_time: Optional[time]) -> int:
        """Return minutes until standard close (15:15) from a time object."""
        if current_time is None:
            return 180
        close = time(15, 15)
        now_minutes = current_time.hour * 60 + current_time.minute
        close_minutes = close.hour * 60 + close.minute
        return max(0, close_minutes - now_minutes)

    # Back-compat alias used in some modules
    _get_minutes_to_close = _minutes_until_close

    # ----------------- Environment API -----------------
    def reset(self):
        if self.current_day_idx >= len(self.trading_days):
            self.current_day_idx = 0
        self._set_day(self.current_day_idx)

        n = len(self.current_day_data)
        if n >= self.lookback:
            start = self.current_step - self.lookback
            window = self.current_day_data.iloc[start : self.current_step]
            state = window[self.feature_cols].values.astype(np.float32)
        else:
            state = np.zeros((self.lookback, max(1, len(self.feature_cols))), dtype=np.float32)
        return state

    def step(self, action: int):
        if self.current_step >= len(self.current_day_data):
            return self.reset(), 0.0, True, {"reason": "end_of_data"}

        mapping = {0: -1, 1: 0, 2: 1}
        target_pos = mapping.get(int(action), 0)

        price = float(self.current_day_data['close'].iloc[self.current_step]) if 'close' in self.current_day_data.columns else float(self.current_day_data.iloc[self.current_step, 0])
        reward = 0.0
        info: Dict[str, Any] = {
            'position': self.position,
            'price': price,
            'total_pnl': self.total_pnl,
        }

        if target_pos != self.position:
            if self.position != 0:
                pnl = (price - self.entry_price) * self.position * self.lot_size
                self.total_pnl += pnl
                reward += pnl / max(1.0, self.initial_capital)
                info['trade_pnl'] = float(pnl)

            old_pos = self.position
            self.position = target_pos
            self.entry_price = price if self.position != 0 else 0.0

            self.trades.append({
                'step': int(self.current_step),
                'old_pos': int(old_pos),
                'new_pos': int(self.position),
                'price': price,
            })
            info['position_change'] = f"{old_pos}->{self.position}"

        self.current_step += 1
        done = self.current_step >= len(self.current_day_data)

        if done and self.position != 0:
            last_price = float(self.current_day_data['close'].iloc[-1]) if 'close' in self.current_day_data.columns else float(self.current_day_data.iloc[-1, 0])
            final_pnl = (last_price - self.entry_price) * self.position * self.lot_size
            self.total_pnl += final_pnl
            reward += final_pnl / max(1.0, self.initial_capital)
            info['final_exit_pnl'] = float(final_pnl)
            self.position = 0
            self.entry_price = 0.0

        if not done:
            start = max(0, self.current_step - self.lookback)
            window = self.current_day_data.iloc[start : self.current_step]
            next_state = window[self.feature_cols].values.astype(np.float32)
            if next_state.shape[0] < self.lookback:
                pad = np.zeros((self.lookback - next_state.shape[0], next_state.shape[1]), dtype=np.float32)
                next_state = np.vstack([pad, next_state])
        else:
            next_state = np.zeros((self.lookback, max(1, len(self.feature_cols))), dtype=np.float32)

        info.update({'total_pnl': float(self.total_pnl), 'num_trades': len(self.trades)})
        return next_state, float(reward), bool(done), info

    def get_episode_metrics(self) -> Dict[str, Any]:
        pnl = float(self.total_pnl)
        num_trades = len(self.trades)
        pnl_pct = (pnl / self.initial_capital) * 100.0
        returns = np.array([pnl / max(1.0, self.initial_capital)])
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0.0
        max_dd = 0.0
        return {
            'total_pnl': pnl,
            'pnl_pct': pnl_pct,
            'num_trades': num_trades,
            'max_drawdown': max_dd,
            'sharpe_ratio': sharpe,
            'win_rate': 1.0 if pnl > 0 else 0.0,
        }

    def get_episode_summary(self) -> Dict[str, Any]:
        """Back-compat wrapper for training.py"""
        return self.get_episode_metrics()


# Backward compatible alias
class IntradayTradingEnv(IntradayMarket):
    pass


def create_intraday_env(data: pd.DataFrame, config: Optional[dict] = None) -> IntradayMarket:
    config = config or {}
    return IntradayMarket(
        data=data,
        initial_capital=config.get('initial_capital', 500_000),
        lot_size=config.get('lot_size', 50),
        max_lots=config.get('max_lots', 5),
        lookback=config.get('lookback', 30),
    )


__all__ = [
    'IntradayMarket',
    'IntradayTradingEnv',
    'create_intraday_env',
]

# Additional alias used in some modules
TradingEnvironment = IntradayTradingEnv
