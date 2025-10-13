"""
Intraday Trading Environment - PRODUCTION
Gym-like environment for RL training with intraday-specific reward function
Phase 5 Enhanced with monitoring integration
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any
import logging
from datetime import datetime, time as dt_time

# Phase 5 Monitoring Integration
try:
    from utils.monitoring_integration import (
        monitor_rl_decision, track_performance, get_health_monitor
    )
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    logging.warning("Monitoring integration not available")

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
        Initialize environment - Phase 5 Enhanced
        
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
        
        # Phase 5: Detect Phase 4 feature availability
        self._detect_phase4_features()
        
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
        logger.info(f"Phase 4 features detected: {self._has_phase4_features}, Feature count: {self._get_feature_count()}")
    
    def _detect_phase4_features(self):
        """
        Detect if Phase 4 enhanced features are available in the dataset
        """
        self._has_phase4_features = False
        self._phase4_feature_categories = {
            'pcr_features': False,
            'confluence_features': False, 
            'regime_features': False,
            'oi_features': False,
            'quality_features': False
        }
        
        if self.data.empty:
            return
            
        # Check for Phase 4 feature categories
        columns = set(self.data.columns)
        
        # PCR features
        pcr_indicators = ['pcr_percentile', 'pcr_oversold', 'oi_pcr', 'composite_pcr', 'options_flow_bias']
        if any(col in columns for col in pcr_indicators):
            self._phase4_feature_categories['pcr_features'] = True
            
        # Confluence features  
        confluence_indicators = ['momentum_confluence_score', 'overall_confluence_score', 'strong_market_confluence']
        if any(col in columns for col in confluence_indicators):
            self._phase4_feature_categories['confluence_features'] = True
            
        # Market regime features
        regime_indicators = ['market_regime_strong_trend', 'volatility_regime_high', 'high_risk_environment']
        if any(col in columns for col in regime_indicators):
            self._phase4_feature_categories['regime_features'] = True
            
        # Enhanced OI features
        oi_indicators = ['oi_concentration_index', 'oi_flow_1d', 'price_oi_efficiency', 'volume_oi_divergence_score']
        if any(col in columns for col in oi_indicators):
            self._phase4_feature_categories['oi_features'] = True
            
        # Quality & Risk features
        quality_indicators = ['setup_quality_score', 'market_risk_score', 'price_action_quality_score']
        if any(col in columns for col in quality_indicators):
            self._phase4_feature_categories['quality_features'] = True
        
        # Overall Phase 4 detection
        self._has_phase4_features = any(self._phase4_feature_categories.values())
        
        if self._has_phase4_features:
            enabled_categories = [cat for cat, enabled in self._phase4_feature_categories.items() if enabled]
            logger.info(f"Phase 4 features detected: {enabled_categories}")
        else:
            logger.info("Using legacy feature set (Phase 3 and below)")
    
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
        
        # Phase 5 Monitoring Integration
        if MONITORING_AVAILABLE:
            try:
                # Record RL decision for monitoring
                action_names = {0: 'short', 1: 'hold', 2: 'long'}
                confidence = abs(reward) / (abs(reward) + 1.0)  # Simple confidence proxy
                phase4_enhanced = hasattr(self, 'phase4_features_detected') and self.phase4_features_detected
                
                monitor = get_health_monitor()
                monitor.record_rl_decision(
                    action=action_names.get(action, 'unknown'),
                    confidence=confidence,
                    phase4_enhanced=phase4_enhanced
                )
            except Exception as e:
                logger.warning(f"Failed to record RL decision metrics: {e}")
        
        info = {
            'position': self.position,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'total_pnl': self.realized_pnl + self.unrealized_pnl,
            'num_trades': self.num_trades,
            'minutes_to_close': minutes_to_close,
            'phase4_enhanced': hasattr(self, 'phase4_features_detected') and self.phase4_features_detected
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
        Get current state - Phase 5 Enhanced with Phase 4 features
        Expanded to include PCR, confluence, and regime indicators
        """
        # Phase 5: Expanded feature space from 32 to 65+ features
        feature_count = self._get_feature_count()
        
        if self.current_step < 30:
            return np.zeros((30, feature_count), dtype=np.float32)
        
        # Return window of recent data with Phase 4 enhancements
        window = self.current_day_data.iloc[max(0, self.current_step-30):self.current_step]
        
        if len(window) < 30:
            # Pad with zeros
            padding = np.zeros((30 - len(window), feature_count))
            state_data = self._extract_enhanced_features(window)
            if state_data.shape[0] > 0:
                state = np.vstack([padding, state_data])
            else:
                state = np.zeros((30, feature_count))
        else:
            state_data = self._extract_enhanced_features(window)
            state = state_data if state_data.shape[0] == 30 else np.zeros((30, feature_count))
        
        return state.astype(np.float32)
    
    def _get_feature_count(self) -> int:
        """
        Determine feature count based on available Phase 4 enhancements
        """
        base_features = 32  # Original OHLCV + basic indicators
        
        # Phase 4 feature additions
        if hasattr(self, '_has_phase4_features') and self._has_phase4_features:
            return 68  # Base + Phase 4 enhancements
        else:
            return base_features
    
    def _extract_enhanced_features(self, window: pd.DataFrame) -> np.ndarray:
        """
        Extract Phase 5 enhanced feature vectors from data window
        """
        if window.empty:
            return np.zeros((0, self._get_feature_count()))
        
        try:
            features_list = []
            
            for _, row in window.iterrows():
                feature_vector = self._build_feature_vector(row)
                features_list.append(feature_vector)
            
            if features_list:
                return np.array(features_list)
            else:
                return np.zeros((len(window), self._get_feature_count()))
                
        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            return np.zeros((len(window), self._get_feature_count()))
    
    def _build_feature_vector(self, row: pd.Series) -> np.ndarray:
        """
        Build Phase 5 enhanced feature vector from row data
        """
        feature_count = self._get_feature_count()
        features = np.zeros(feature_count)
        
        try:
            # Base features (0-31): OHLCV + basic technical indicators
            base_cols = ['open', 'high', 'low', 'close', 'volume']
            tech_cols = ['rsi_14', 'macd', 'macd_signal', 'atr_14', 'bb_upper', 'bb_lower', 'sma_20', 'ema_12']
            
            idx = 0
            
            # OHLCV features (normalized)
            for col in base_cols:
                if col in row.index:
                    if col in ['open', 'high', 'low', 'close']:
                        features[idx] = row[col] / 20000.0  # Price normalization
                    elif col == 'volume':
                        features[idx] = min(row[col] / 100000.0, 5.0)  # Volume normalization
                    idx += 1
            
            # Basic technical indicators (normalized)
            for col in tech_cols:
                if col in row.index:
                    if 'rsi' in col:
                        features[idx] = row[col] / 100.0
                    elif 'macd' in col:
                        features[idx] = np.tanh(row[col] / 50.0)
                    elif 'atr' in col:
                        features[idx] = min(row[col] / 100.0, 1.0)
                    elif any(x in col for x in ['bb_', 'sma_', 'ema_']):
                        features[idx] = row[col] / 20000.0
                    else:
                        features[idx] = np.tanh(row[col])
                    idx += 1
            
            # Phase 4 features (32-67) if available
            if feature_count > 32:
                phase4_features = self._extract_phase4_features(row)
                features[32:32+len(phase4_features)] = phase4_features
                
        except Exception as e:
            logger.warning(f"Feature vector building failed: {e}")
            
        return features
    
    def _extract_phase4_features(self, row: pd.Series) -> np.ndarray:
        """
        Extract Phase 4 enhanced features from row data
        Phase 5 Enhanced with monitoring
        """
        phase4_features = np.zeros(36)  # 36 Phase 4 features
        
        # Phase 5 Monitoring Integration
        if MONITORING_AVAILABLE:
            performance_tracker = track_performance('feature_generation', 'phase4_extraction')
            performance_tracker.__enter__()
        else:
            performance_tracker = None
        
        try:
            idx = 0
            
            # PCR features (0-7)
            pcr_cols = ['pcr_percentile', 'pcr_oversold', 'pcr_overbought', 'pcr_volatility',
                       'oi_pcr', 'vol_pcr', 'composite_pcr', 'options_flow_bias']
            for col in pcr_cols:
                if col in row.index:
                    if 'percentile' in col:
                        phase4_features[idx] = row[col] / 100.0
                    elif col in ['pcr_oversold', 'pcr_overbought']:
                        phase4_features[idx] = float(row[col])
                    elif 'pcr' in col:
                        phase4_features[idx] = np.tanh(row[col])
                    else:
                        phase4_features[idx] = np.tanh(row[col])
                idx += 1
            
            # Confluence features (8-15)
            confluence_cols = ['momentum_confluence_score', 'rsi_confluence_bullish', 'macd_confluence_score',
                             'trend_confluence_score', 'overall_confluence_score', 'strong_market_confluence',
                             'rsi_strong_confluence', 'macd_strong_confluence']
            for col in confluence_cols:
                if col in row.index:
                    if 'score' in col:
                        phase4_features[idx] = min(row[col] / 3.0, 1.0)
                    else:
                        phase4_features[idx] = float(row[col])
                idx += 1
            
            # Market regime features (16-23)
            regime_cols = ['market_regime_strong_trend', 'market_regime_consolidation',
                          'volatility_regime_high', 'volatility_regime_low', 'volatility_expanding',
                          'volatility_contracting', 'high_risk_environment', 'low_risk_environment']
            for col in regime_cols:
                if col in row.index:
                    phase4_features[idx] = float(row[col])
                idx += 1
            
            # Enhanced OI features (24-31)
            oi_cols = ['oi_concentration_index', 'oi_percentile_rank', 'oi_flow_1d', 'oi_acceleration',
                      'price_oi_efficiency', 'oi_trend_strength', 'volume_oi_divergence_score', 'oi_strong_trend']
            for col in oi_cols:
                if col in row.index:
                    if 'percentile' in col or 'concentration' in col:
                        phase4_features[idx] = row[col] / 100.0
                    elif 'efficiency' in col or 'flow' in col:
                        phase4_features[idx] = np.tanh(row[col])
                    else:
                        phase4_features[idx] = float(row[col]) if col in ['oi_strong_trend'] else np.tanh(row[col])
                idx += 1
            
            # Quality & Risk features (32-35)
            quality_cols = ['setup_quality_score', 'market_risk_score', 'high_quality_setup', 'price_action_quality_score']
            for col in quality_cols:
                if col in row.index:
                    if 'score' in col:
                        phase4_features[idx] = row[col]  # Already 0-1 range
                    else:
                        phase4_features[idx] = float(row[col])
                idx += 1
                
        except Exception as e:
            logger.warning(f"Phase 4 feature extraction failed: {e}")
            if performance_tracker:
                performance_tracker.success = False
                
        finally:
            # Complete performance tracking
            if performance_tracker:
                performance_tracker.__exit__(None, None, None)
            
        return phase4_features
    
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