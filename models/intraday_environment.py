"""
Intraday Trading Environment - PRODUCTION
Gym-like environment for RL training with intraday-specific reward function
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any
import logging
from datetime import datetime, time as dt_time

logger = logging.getLogger(__name__)


class IntradayTradingEnv:
    """
    Trading environment for intraday F&O trading
    - Episode = Single trading day (9:15 AM - 3:15 PM)
    - Mandatory square-off by 3:10 PM
    - Intraday-specific reward function with time penalties
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        initial_capital: float = 500000,
        lot_size: int = 50,
        target_volatility: float = 0.12,
        max_lots: int = 5,
        transaction_cost_bp: float = 2.0,
        stt_sell_pct: float = 0.0125,
        time_penalty_weight: float = 0.1,
        flat_by_close_bonus: float = 0.5,
        mis_mode: bool = True
    ):
        """
        Initialize environment
        
        Args:
            data: DataFrame with OHLCV + indicators for multiple days
            initial_capital: Starting capital in rupees
            lot_size: Lot size for the instrument
            target_volatility: Target volatility for position sizing (12% for intraday)
            max_lots: Maximum lots per position
            transaction_cost_bp: Transaction costs in basis points
            stt_sell_pct: STT on sell side (%)
            time_penalty_weight: Weight for time-based penalty
            flat_by_close_bonus: Bonus for exiting before 3:10 PM
            mis_mode: Use MIS (intraday) margin
        """
        self.data = data
        self.initial_capital = initial_capital
        self.lot_size = lot_size
        self.target_volatility = target_volatility
        self.max_lots = max_lots
        self.transaction_cost_bp = transaction_cost_bp
        self.stt_sell_pct = stt_sell_pct
        self.time_penalty_weight = time_penalty_weight
        self.flat_by_close_bonus = flat_by_close_bonus
        self.mis_mode = mis_mode
        
        # Group data by trading days
        self.data['date'] = pd.to_datetime(self.data.index).date
        self.trading_days = self.data['date'].unique()
        
        # Episode management
        self.current_day_idx = 0
        self.current_step = 0
        self.current_day_data = None
        
        # Portfolio state
        self.capital = initial_capital
        self.position = 0  # Current position in lots (signed: +ve = long, -ve = short)
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        
        # Episode metrics
        self.episode_pnl = []
        self.episode_positions = []
        self.episode_actions = []
        self.num_trades = 0
        
        logger.info(f"IntradayTradingEnv initialized - {len(self.trading_days)} trading days, Capital: ₹{initial_capital:,.0f}")
    
    def reset(self) -> np.ndarray:
        """
        Reset environment to start of new episode (trading day)
        
        Returns:
            initial_state: First state of the episode
        """
        # Get next trading day
        if self.current_day_idx >= len(self.trading_days):
            self.current_day_idx = 0  # Loop back for training
        
        current_date = self.trading_days[self.current_day_idx]
        self.current_day_data = self.data[self.data['date'] == current_date].copy()
        
        # Reset state
        self.current_step = 0
        self.position = 0
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        
        # Episode metrics
        self.episode_pnl = []
        self.episode_positions = []
        self.episode_actions = []
        self.num_trades = 0
        
        logger.debug(f"Reset to day {self.current_day_idx} - {current_date} ({len(self.current_day_data)} bars)")
        
        # Return initial state (will be built by state builder)
        return self._get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take action in environment
        
        Args:
            action: Action index {0, 1, 2} → {-1, 0, 1} (Short, Hold, Long)
        
        Returns:
            next_state: Next state
            reward: Reward for this step
            done: Episode done flag
            info: Additional info
        """
        # Convert action index to position target
        action_map = {0: -1, 1: 0, 2: 1}  # Short, Hold, Long
        target_position = action_map[action] * self.max_lots
        
        # Get current bar data
        if self.current_step >= len(self.current_day_data):
            # Episode done
            return self._get_state(), 0.0, True, {'reason': 'end_of_day'}
        
        current_bar = self.current_day_data.iloc[self.current_step]
        current_price = current_bar['close']
        current_time = current_bar.name.time() if hasattr(current_bar.name, 'time') else dt_time(12, 0)
        
        # Calculate time features
        minutes_to_close = self._get_minutes_to_close(current_time)
        
        # Force exit if close to market close (3:10 PM)
        if minutes_to_close < 5:
            target_position = 0  # Force flat
            logger.debug(f"Force flat - {minutes_to_close:.1f} mins to close")
        
        # Execute trade if position changes
        position_change = target_position - self.position
        
        if position_change != 0:
            # Calculate transaction costs
            trade_value = abs(position_change) * self.lot_size * current_price
            transaction_cost = self._calculate_transaction_cost(
                trade_value=trade_value,
                is_sell=(position_change < 0 if self.position >= 0 else position_change > 0)
            )
            
            # Update realized P&L with transaction cost
            self.realized_pnl -= transaction_cost
            
            # Update position
            old_position = self.position
            self.position = target_position
            
            # Track entry price
            if self.position != 0 and old_position == 0:
                self.entry_price = current_price
            elif self.position == 0:
                self.entry_price = 0.0
            
            self.num_trades += 1
            logger.debug(f"Trade: {old_position} → {self.position} lots @ ₹{current_price:.2f}, Cost: ₹{transaction_cost:.2f}")
        
        # Calculate unrealized P&L
        if self.position != 0:
            self.unrealized_pnl = (current_price - self.entry_price) * self.position * self.lot_size
        else:
            self.unrealized_pnl = 0.0
        
        # Calculate reward
        reward = self._calculate_reward(
            price_change=current_bar['close'] - self.current_day_data.iloc[self.current_step - 1]['close'] if self.current_step > 0 else 0,
            position_change=position_change,
            current_price=current_price,
            minutes_to_close=minutes_to_close
        )
        
        # Store metrics
        self.episode_pnl.append(self.realized_pnl + self.unrealized_pnl)
        self.episode_positions.append(self.position)
        self.episode_actions.append(action)
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode done
        done = self.current_step >= len(self.current_day_data)
        
        # Add flat-by-close bonus
        if done and self.position == 0:
            reward += self.flat_by_close_bonus
            logger.debug(f"Flat-by-close bonus: +{self.flat_by_close_bonus}")
        
        # Get next state
        next_state = self._get_state()
        
        info = {
            'position': self.position,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'total_pnl': self.realized_pnl + self.unrealized_pnl,
            'num_trades': self.num_trades,
            'minutes_to_close': minutes_to_close
        }
        
        return next_state, reward, done, info
    
    def _calculate_reward(
        self,
        price_change: float,
        position_change: float,
        current_price: float,
        minutes_to_close: float
    ) -> float:
        """
        Calculate intraday-specific reward
        
        Reward = position_scaled_return - transaction_costs - time_penalty + flat_bonus
        """
        # Per-minute return scaled by position
        if self.position != 0:
            position_return = (price_change / current_price) * self.position * self.lot_size * current_price
        else:
            position_return = 0.0
        
        # Volatility scaling (normalize by target volatility)
        # estimated_vol = self.current_day_data['close'].pct_change().std() if self.current_step > 10 else 0.15
        # vol_scalar = self.target_volatility / estimated_vol if estimated_vol > 0 else 1.0
        # vol_scalar = np.clip(vol_scalar, 0.5, 2.0)
        # position_return *= vol_scalar
        
        # Time penalty (encourage faster trades)
        time_penalty = self.time_penalty_weight * (1 - minutes_to_close / 360)
        if abs(self.position) > 0:
            time_penalty *= abs(self.position) / self.max_lots
        
        # Combine
        reward = position_return - time_penalty
        
        # Normalize to reasonable scale
        reward = reward / (self.initial_capital / 100)  # Scale to % of capital
        
        return reward
    
    def _calculate_transaction_cost(self, trade_value: float, is_sell: bool) -> float:
        """
        Calculate transaction costs (brokerage + STT + charges)
        """
        # Brokerage (₹20 flat per order)
        brokerage = 20.0
        
        # STT (only on sell side)
        if is_sell:
            stt = trade_value * (self.stt_sell_pct / 100)
        else:
            stt = 0.0
        
        # Exchange charges + SEBI + Stamp duty
        other_charges = trade_value * (self.transaction_cost_bp / 10000)
        
        # GST (18% on brokerage + exchange)
        gst = (brokerage + other_charges) * 0.18
        
        total_cost = brokerage + stt + other_charges + gst
        
        return total_cost
    
    def _get_minutes_to_close(self, current_time: dt_time) -> float:
        """Calculate minutes remaining to market close (3:15 PM)"""
        market_close = dt_time(15, 15)
        
        current_minutes = current_time.hour * 60 + current_time.minute
        close_minutes = market_close.hour * 60 + market_close.minute
        
        minutes_to_close = max(0, close_minutes - current_minutes)
        
        return minutes_to_close
    
    def _get_state(self) -> np.ndarray:
        """
        Get current state
        Note: In practice, this should use IntradayStateBuilder
        For environment, we return a placeholder
        """
        # This is a placeholder - actual state building done by IntradayStateBuilder
        if self.current_step < 30:
            return np.zeros((30, 32), dtype=np.float32)
        
        # Return window of recent data (simplified)
        window = self.current_day_data.iloc[max(0, self.current_step-30):self.current_step]
        
        if len(window) < 30:
            # Pad with zeros
            padding = np.zeros((30 - len(window), 32))
            state = np.vstack([padding, np.zeros((len(window), 32))])
        else:
            state = np.zeros((30, 32))
        
        return state.astype(np.float32)
    
    def get_episode_metrics(self) -> Dict[str, float]:
        """Calculate episode-level metrics"""
        
        if len(self.episode_pnl) == 0:
            return {}
        
        pnl_series = pd.Series(self.episode_pnl)
        
        # Total P&L
        total_pnl = self.realized_pnl + self.unrealized_pnl
        
        # Returns
        returns = pnl_series.pct_change().fillna(0)
        
        # Sharpe ratio (intraday - annualize differently)
        # Intraday: 6.5 hours * 252 days
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 6.5) if returns.std() > 0 else 0.0
        
        # Max drawdown
        cumulative = pnl_series.cumsum()
        running_max = cumulative.cummax()
        drawdown = cumulative - running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        trades_pnl = pd.Series(self.episode_pnl).diff().dropna()
        win_rate = (trades_pnl > 0).sum() / len(trades_pnl) if len(trades_pnl) > 0 else 0.0
        
        # Average hold time (bars)
        positions_array = np.array(self.episode_positions)
        position_changes = np.diff(np.abs(positions_array) > 0).astype(int)
        avg_hold_time = len(positions_array) / (np.abs(position_changes).sum() + 1)
        
        metrics = {
            'total_pnl': total_pnl,
            'pnl_pct': (total_pnl / self.initial_capital) * 100,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': self.num_trades,
            'avg_hold_time_bars': avg_hold_time,
            'final_position': self.position
        }
        
        return metrics
    
    def render(self, mode: str = 'human') -> None:
        """Render environment state"""
        if mode == 'human':
            metrics = self.get_episode_metrics()
            print(f"\n{'='*60}")
            print(f"Day {self.current_day_idx} | Step {self.current_step}/{len(self.current_day_data)}")
            print(f"Position: {self.position} lots | P&L: ₹{metrics.get('total_pnl', 0):,.2f}")
            print(f"Trades: {self.num_trades} | Win Rate: {metrics.get('win_rate', 0)*100:.1f}%")
            print(f"{'='*60}\n")


def create_intraday_env(data: pd.DataFrame, config: Dict[str, Any]) -> IntradayTradingEnv:
    """Factory function to create environment from config"""
    
    env = IntradayTradingEnv(
        data=data,
        initial_capital=config.get('initial_capital', 500000),
        lot_size=config.get('lot_size', 50),
        target_volatility=config.get('target_volatility', 0.12),
        max_lots=config.get('max_lots', 5),
        transaction_cost_bp=config.get('transaction_cost_bp', 2.0),
        stt_sell_pct=config.get('stt_sell_pct', 0.0125),
        time_penalty_weight=config.get('time_penalty_weight', 0.1),
        flat_by_close_bonus=config.get('flat_by_close_bonus', 0.5),
        mis_mode=config.get('mis_mode', True)
    )
    
    return env