"""
SuperTrader.AI Intraday Trading Environment

Provides trading environments for intraday reinforcement learning with:
- IntradayTradingEnv: Main environment for training/evaluation
- IntradayMarket: Legacy alias for compatibility 
- create_intraday_env: Factory function for environment creation

Features:
- Position tracking (Short=-1, Hold=0, Long=1)
- Transaction cost modeling
- Real market constraints (lot sizes, STT)
- Intraday risk management
"""


from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
from datetime import datetime, time
import warnings

logger = logging.getLogger(__name__)


class IntradayTradingEnv:
    """
    Intraday trading environment for reinforcement learning.
    
    Action Space:
        0: SHORT - Take short position (sell)
        1: HOLD/FLAT - No position  
        2: LONG - Take long position (buy)
    
    State Space:
        (lookback, n_features) array representing market state
    
    Reward:
        Normalized P&L based on position changes and market movements
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        initial_capital: float = 500000,
        lot_size: int = 50,
        max_lots: int = 3,
        target_volatility: float = 0.12,
        transaction_cost_bp: float = 10,
        stt_sell_pct: float = 0.025,
        lookback: int = 30,
        start_time: str = "09:15",
        end_time: str = "15:15",
        **kwargs
    ):
        """
        Initialize intraday trading environment.
        
        Args:
            data: OHLCV price data with datetime index
            initial_capital: Starting capital
            lot_size: Shares per lot
            max_lots: Maximum position size in lots
            target_volatility: Target annualized volatility
            transaction_cost_bp: Transaction cost in basis points
            stt_sell_pct: Securities Transaction Tax on sells (%)
            lookback: Number of historical bars for state
            start_time: Trading start time (HH:MM)
            end_time: Trading end time (HH:MM)
        """
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.lot_size = lot_size
        self.max_lots = max_lots
        self.target_volatility = target_volatility
        self.transaction_cost_bp = transaction_cost_bp / 10000  # Convert bp to decimal
        self.stt_sell_pct = stt_sell_pct / 100  # Convert % to decimal
        self.lookback = lookback
        self.start_time = start_time
        self.end_time = end_time
        
        # Ensure required columns exist
        self._validate_data()
        
        # Extract trading days
        self.trading_days = sorted(self.data['date'].unique()) if 'date' in self.data.columns else [self.data.index[0].date()]
        
        # State variables
        self.reset()
        
        logger.info(f"IntradayTradingEnv initialized - {len(self.trading_days)} trading days, {len(self.data)} total bars")
    
    def _validate_data(self):
        """Ensure data has required columns and proper format"""
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        
        if missing_cols:
            logger.warning(f"Missing columns {missing_cols}, creating defaults")
            
            # Fill missing OHLC with close price if available
            if 'close' in self.data.columns:
                for col in ['open', 'high', 'low']:
                    if col not in self.data.columns:
                        self.data[col] = self.data['close']
            else:
                # Use first numeric column as proxy
                numeric_cols = self.data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    price_col = numeric_cols[0]
                    for col in required_cols[:-1]:  # Skip volume
                        self.data[col] = self.data[price_col]
                else:
                    # Last resort: create synthetic data
                    for col in required_cols[:-1]:
                        self.data[col] = 22000
            
            # Default volume
            if 'volume' not in self.data.columns:
                self.data['volume'] = 1000
        
        # Add date column if missing
        if 'date' not in self.data.columns:
            if hasattr(self.data.index, 'date'):
                self.data['date'] = self.data.index.date
            else:
                self.data['date'] = pd.to_datetime(self.data.index).date
    
    def reset(self, day_index: Optional[int] = None) -> np.ndarray:
        """
        Reset environment for new episode.
        
        Args:
            day_index: Specific trading day to start from (random if None)
            
        Returns:
            Initial state observation
        """
        # Select trading day
        if day_index is None:
            self.current_day_idx = np.random.randint(0, len(self.trading_days))
        else:
            self.current_day_idx = min(day_index, len(self.trading_days) - 1)
        
        current_day = self.trading_days[self.current_day_idx]
        
        # Get day's data
        if 'date' in self.data.columns:
            day_data = self.data[self.data['date'] == current_day].copy()
        else:
            # Fallback: use all data as single day
            day_data = self.data.copy()
        
        if len(day_data) == 0:
            logger.warning(f"No data for day {current_day}, using first available day")
            day_data = self.data.iloc[:100].copy()  # Take first 100 bars as fallback
        
        self.day_data = day_data.reset_index(drop=True)
        
        # Initialize state
        self.step_idx = self.lookback  # Start after lookback period
        self.position = 0  # 0=flat, -1=short, 1=long
        self.entry_price = 0.0
        self.cash = self.initial_capital
        self.portfolio_value = self.initial_capital
        self.trades = []
        self.done = False
        
        logger.debug(f"Reset to day {current_day}, {len(self.day_data)} bars available")
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """
        Get current market state for agent.
        
        Returns:
            State array of shape (lookback, n_features)
        """
        if self.step_idx < self.lookback:
            # Pad with zeros for early steps
            available_data = self.day_data.iloc[:self.step_idx]
            padding_needed = self.lookback - len(available_data)
            
            if len(available_data) == 0:
                # No data yet, return zero state
                return np.zeros((self.lookback, 5), dtype=np.float32)
            
            # Use numeric columns for features
            feature_cols = available_data.select_dtypes(include=[np.number]).columns[:5]
            if len(feature_cols) < 5:
                feature_cols = list(feature_cols) + ['close'] * (5 - len(feature_cols))
            
            features = available_data[feature_cols].values
            padding = np.zeros((padding_needed, features.shape[1]), dtype=np.float32)
            state = np.vstack([padding, features])
        else:
            # Normal case: get last N bars
            window_data = self.day_data.iloc[self.step_idx - self.lookback:self.step_idx]
            feature_cols = window_data.select_dtypes(include=[np.number]).columns[:5]
            if len(feature_cols) < 5:
                feature_cols = ['open', 'high', 'low', 'close', 'volume']
            
            state = window_data[feature_cols].values
        
        # Normalize state (simple percentage change)
        if state.shape[0] > 1:
            normalized_state = np.zeros_like(state)
            normalized_state[0] = 0  # First row stays zero
            for i in range(1, state.shape[0]):
                normalized_state[i] = (state[i] - state[i-1]) / (state[i-1] + 1e-8)
        else:
            normalized_state = np.zeros_like(state)
        
        return normalized_state.astype(np.float32)
    
    def _get_feature_count(self) -> int:
        """Get number of features in state vector"""
        return 5  # OHLCV features
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute trading action and return results.
        
        Args:
            action: 0=SHORT, 1=HOLD, 2=LONG
            
        Returns:
            (next_state, reward, done, info)
        """
        if self.done:
            return self._get_state(), 0.0, True, {'error': 'Episode already done'}
        
        """Intraday market environment for SuperTrader.AI.

        This module provides a compact but compatible implementation of the
        environment API expected across the repo:

        - IntradayMarket: main environment used by `models/training.py`
        - IntradayTradingEnv: alias for IntradayMarket (used by scripts/train.py)
        - create_intraday_env(data, config): convenience factory

        The goal is to provide a deterministic, well-documented simulator with the
        minimal interface required by training/evaluation code. This implementation
        is intentionally lightweight and focuses on correctness and compatibility.
        """
        class IntradayMarket:
            """Simple intraday market simulator.

            Key attributes expected by the rest of the codebase:
            - trading_days: list of DataFrame, one per trading day
            - current_day_idx: index of the current trading day
            - current_day_data: DataFrame of the current day
            - current_step: integer bar index within current_day_data
            - initial_capital: starting capital

            Methods:
            - reset() -> initial_state
            - step(action) -> (next_state, reward, done, info)
            - get_episode_metrics() -> dict
            - _minutes_until_close(current_time) and _get_minutes_to_close(time)
            """

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
                    try:
                        data.index = pd.to_datetime(data.index)
                    except Exception:
                        raise ValueError("data must have a datetime-like index")

                self.raw_data = data.sort_index()
                self.initial_capital = float(initial_capital)
                self.lot_size = int(lot_size)
                self.max_lots = int(max_lots)
                self.lookback = int(lookback)

                # Partition into trading days by date
                self.trading_days: List[pd.DataFrame] = []
                for d, df in self.raw_data.groupby(self.raw_data.index.date):
                    self.trading_days.append(df.copy())

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
                # Prefer common OHLCV + engineered columns; fall back to numeric columns
                prefer = ["open", "high", "low", "close", "volume"]
                cols = [c for c in prefer if c in df.columns]
                if len(cols) == 0:
                    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
                    cols = numeric[: min(10, len(numeric))]
                return cols

            def _set_day(self, day_idx: int) -> None:
                self.current_day_idx = int(day_idx)
                self.current_day_data = self.trading_days[self.current_day_idx].reset_index()
                # Ensure index column is datetime and set as index for convenience
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

            # Alternate name used in other modules
            _get_minutes_to_close = _minutes_until_close

            # ----------------- Environment API -----------------
            def reset(self):
                """Start (or restart) the current trading day and return initial state.

                Returns a state array shaped (lookback, n_features) filled with the most
                recent lookback bars (or zeros if not enough history).
                """
                # Clamp day index
                if self.current_day_idx >= len(self.trading_days):
                    self.current_day_idx = 0
                self._set_day(self.current_day_idx)

                # Prepare initial state (most recent lookback bars if available)
                n = len(self.current_day_data)
                if n >= self.lookback:
                    start = self.current_step - self.lookback
                    window = self.current_day_data.iloc[start : self.current_step]
                    state = window[self.feature_cols].values.astype(np.float32)
                else:
                    state = np.zeros((self.lookback, max(1, len(self.feature_cols))), dtype=np.float32)

                return state

            def step(self, action: int):
                """Execute action (0=short,1=hold,2=long) and return next_state, reward, done, info."""
                if self.current_step >= len(self.current_day_data):
                    # End of day
                    return self.reset(), 0.0, True, {"reason": "end_of_data"}

                # Map actions
                mapping = {0: -1, 1: 0, 2: 1}
                target_pos = mapping.get(int(action), 0)

                price = float(self.current_day_data['close'].iloc[self.current_step]) if 'close' in self.current_day_data.columns else float(self.current_day_data.iloc[self.current_step, 0])
                reward = 0.0
                info: Dict[str, Any] = {
                    'position': self.position,
                    'price': price,
                    'total_pnl': self.total_pnl,
                }

                # Position change
                if target_pos != self.position:
                    # If exiting an existing position calculate realized P&L
                    if self.position != 0:
                        pnl = (price - self.entry_price) * self.position * self.lot_size
                        self.total_pnl += pnl
                        reward += pnl / max(1.0, self.initial_capital)
                        info['trade_pnl'] = float(pnl)

                    # Update position
                    old_pos = self.position
                    self.position = target_pos
                    if self.position != 0:
                        self.entry_price = price
                    else:
                        self.entry_price = 0.0

                    self.trades.append({
                        'step': int(self.current_step),
                        'old_pos': int(old_pos),
                        'new_pos': int(self.position),
                        'price': price,
                    })
                    info['position_change'] = f"{old_pos}->{self.position}"

                # Move forward one bar
                self.current_step += 1

                # Done when past the last bar
                done = self.current_step >= len(self.current_day_data)

                # If done, force close any open position and account P&L
                if done and self.position != 0:
                    # Use last available price for final exit
                    last_price = float(self.current_day_data['close'].iloc[-1]) if 'close' in self.current_day_data.columns else float(self.current_day_data.iloc[-1, 0])
                    final_pnl = (last_price - self.entry_price) * self.position * self.lot_size
                    self.total_pnl += final_pnl
                    reward += final_pnl / max(1.0, self.initial_capital)
                    info['final_exit_pnl'] = float(final_pnl)
                    # reset position
                    self.position = 0
                    self.entry_price = 0.0

                # Prepare next_state
                if not done:
                    start = max(0, self.current_step - self.lookback)
                    window = self.current_day_data.iloc[start : self.current_step]
                    next_state = window[self.feature_cols].values.astype(np.float32)
                    # Ensure shape (lookback, n_features)
                    if next_state.shape[0] < self.lookback:
                        pad = np.zeros((self.lookback - next_state.shape[0], next_state.shape[1]), dtype=np.float32)
                        next_state = np.vstack([pad, next_state])
                else:
                    next_state = np.zeros((self.lookback, max(1, len(self.feature_cols))), dtype=np.float32)

                info.update({'total_pnl': float(self.total_pnl), 'num_trades': len(self.trades)})

                return next_state, float(reward), bool(done), info

            # ----------------- Episode metrics -----------------
            def get_episode_metrics(self) -> Dict[str, Any]:
                pnl = float(self.total_pnl)
                num_trades = len(self.trades)
                pnl_pct = (pnl / self.initial_capital) * 100.0

                # Simple sharpe estimate based on single-day pnl (placeholder)
                returns = np.array([pnl / max(1.0, self.initial_capital)])
                sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0.0

                # Max drawdown placeholder (not tracking intra-day equity here)
                max_dd = 0.0

                return {
                    'total_pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'num_trades': num_trades,
                    'max_drawdown': max_dd,
                    'sharpe_ratio': sharpe,
                    'win_rate': 1.0 if pnl > 0 else 0.0,
                }


        # Backwards-compatible names expected elsewhere in the codebase
        class IntradayTradingEnv(IntradayMarket):
            """Alias kept for scripts that import `IntradayTradingEnv`."""


        def create_intraday_env(data: pd.DataFrame, config: Optional[dict] = None) -> IntradayMarket:
            """Factory helper to create an IntradayMarket from data and a config dict."""
            config = config or {}
            return IntradayMarket(
                data=data,
                initial_capital=config.get('initial_capital', 500_000),
                lot_size=config.get('lot_size', 50),
                max_lots=config.get('max_lots', 5),
                lookback=config.get('lookback', 30),
            )


        # Convenience export for older code that expects IntradayMarket as a top-level name
        __all__ = [
            'IntradayMarket',
            'IntradayTradingEnv',
            'create_intraday_env',
        ]
TradingEnvironment = IntradayTradingEnv
