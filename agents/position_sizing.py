"""
DEPRECATED MODULE (position_sizing.py)

This legacy implementation is retained only for backward compatibility.
Use `agents.volatility_position_sizing` for all new sizing logic.
This file will be removed after migration is complete.
"""

import logging
logger = logging.getLogger(__name__)

# Compatibility imports for legacy code below (will be fully removed later)
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
import math
from utils.config import load_config

# Re-export public APIs from the new module for compatibility
from agents.volatility_position_sizing import (
    VolatilityPositionSizer,
    PositionSizingResult,
    calculate_position_size as volatility_calculate_position_size,
    get_recommended_lots,
    is_position_viable,
)

__all__ = [
    "VolatilityPositionSizer",
    "PositionSizingResult",
    "volatility_calculate_position_size",
    "get_recommended_lots",
    "is_position_viable",
]

@dataclass
class PositionSizingResult:
    """Result of position sizing calculation"""
    symbol: str
    recommended_quantity: int  # In units (not lots)
    recommended_lots: int      # In standard lots
    notional_value: float      # Total notional exposure
    risk_amount: float         # Amount at risk
    volatility_target: float   # Target volatility used
    actual_volatility: float   # Current realized volatility
    vol_scaling_factor: float  # Volatility adjustment factor
    rl_signal_strength: float  # RL signal (-1 to +1)
    leverage_used: float       # Effective leverage
    kelly_fraction: float      # Kelly criterion fraction
    margin_required: float     # Estimated margin requirement
    confidence_score: float    # Confidence in sizing (0-1)
    warnings: List[str]        # Any sizing warnings

@dataclass
class IndexVolatilityProfile:
    """Volatility profile for each index"""
    index_name: str
    current_volatility: float     # Current realized volatility
    long_term_volatility: float   # Long-term average volatility
    volatility_regime: str        # 'low', 'normal', 'high', 'extreme'
    vol_percentile: float         # Percentile in historical distribution
    adjustment_factor: float      # Factor to adjust position size
    confidence: float            # Confidence in volatility estimate


class VolatilityPositionSizer:
    """
    Advanced position sizing using volatility targeting with futures-specific considerations
    Based on Oxford paper methodology with enhancements for Indian futures
    """
    
    def __init__(self, config_path: str = None):
        """Initialize position sizer with configuration"""
        self.config = load_config(config_path or "configs/risk.yaml")
        self.position_config = self.config.get('position_sizing', {})
        
        # Index-specific configurations with Indian futures specifications
        self.index_configs = {
            'NIFTY': {
                'lot_size': 75,
                'tick_size': 0.05,
                'typical_volatility': 0.15,
                'vol_floor': 0.08,          # Minimum volatility assumption
                'vol_ceiling': 0.40,        # Maximum volatility cap
                'leverage_cap': 10,         # Max leverage for NIFTY
                'liquidity_score': 1.0,     # Highest liquidity
                'margin_multiplier': 0.10,  # ~10% margin requirement
                'contract_multiplier': 1    # Standard multiplier
            },
            'BANKNIFTY': {
                'lot_size': 15,
                'tick_size': 0.05,
                'typical_volatility': 0.25,
                'vol_floor': 0.12,
                'vol_ceiling': 0.60,
                'leverage_cap': 8,          # Slightly lower due to higher volatility
                'liquidity_score': 0.95,    # Very high liquidity
                'margin_multiplier': 0.12,  # Higher margin due to volatility
                'contract_multiplier': 1
            },
            'FINNIFTY': {
                'lot_size': 25,
                'tick_size': 0.05,
                'typical_volatility': 0.20,
                'vol_floor': 0.10,
                'vol_ceiling': 0.50,
                'leverage_cap': 8,
                'liquidity_score': 0.85,    # Good liquidity
                'margin_multiplier': 0.11,
                'contract_multiplier': 1
            }
        }
        
        # Position sizing parameters
        self.base_volatility_target = self.position_config.get('base_volatility_target', 0.20)
        self.max_kelly_fraction = self.position_config.get('max_kelly_fraction', 0.25)
        self.min_kelly_fraction = self.position_config.get('min_kelly_fraction', 0.02)
        self.confidence_threshold = self.position_config.get('confidence_threshold', 0.70)
        
        logger.info("Volatility position sizer initialized")
    
    def compute_position_size(
        self,
        symbol: str,
        current_price: float,
        rl_signal: float,  # -1 to +1
        account_balance: float,
        available_margin: float,
        win_rate: float = 0.55,
        avg_win_loss_ratio: float = 1.2,
        current_positions: Dict[str, Any] = None,
        market_data: Dict[str, Any] = None
    ) -> PositionSizingResult:
        """
        Compute optimal position size using volatility targeting
        
        Args:
            symbol: Futures symbol (e.g., 'NIFTY25OCT')
            current_price: Current price of the instrument
            rl_signal: RL strategy signal strength (-1 to +1)
            account_balance: Total account balance
            available_margin: Available margin for new positions
            win_rate: Historical win rate (0-1)
            avg_win_loss_ratio: Average win/loss ratio
            current_positions: Current portfolio positions
            market_data: Additional market data for volatility calculation
            
        Returns:
            PositionSizingResult with recommended position size
        """
        try:
            # Extract index information
            index_name = self._extract_index_name(symbol)
            index_config = self.index_configs.get(index_name, self.index_configs['NIFTY'])
            
            warnings = []
            
            # 1. Calculate current volatility
            vol_profile = self._calculate_volatility_profile(symbol, current_price, market_data)
            
            # 2. Calculate Kelly fraction
            kelly_fraction = self._calculate_kelly_fraction(win_rate, avg_win_loss_ratio)
            
            # 3. Apply volatility scaling (Oxford paper methodology)
            vol_scaling_factor = self._calculate_volatility_scaling(vol_profile, index_config)
            
            # 4. Apply RL signal scaling
            signal_scaling = abs(rl_signal)  # Use absolute value for position size
            
            # 5. Calculate base position size
            base_position_value = account_balance * kelly_fraction * signal_scaling * vol_scaling_factor
            
            # 6. Apply futures-specific adjustments
            adjusted_position_value = self._apply_futures_adjustments(
                base_position_value, index_config, available_margin, warnings
            )
            
            # 7. Convert to lots and handle lot size constraints
            quantity, lots, notional_value = self._convert_to_lots(
                adjusted_position_value, current_price, index_config
            )
            
            # 8. Final validations and risk checks
            final_result = self._validate_and_finalize(
                symbol=symbol,
                quantity=quantity,
                lots=lots,
                current_price=current_price,
                notional_value=notional_value,
                account_balance=account_balance,
                available_margin=available_margin,
                vol_profile=vol_profile,
                kelly_fraction=kelly_fraction,
                vol_scaling_factor=vol_scaling_factor,
                rl_signal=rl_signal,
                index_config=index_config,
                warnings=warnings
            )
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error in position sizing for {symbol}: {e}")
            return self._create_error_result(symbol, str(e))


    def _calculate_volatility_profile(
        self, 
        symbol: str, 
        current_price: float, 
        market_data: Dict[str, Any] = None
    ) -> IndexVolatilityProfile:
    """
    Calculate annualized volatility for position sizing
    
    Methods:
    - EWMA: Exponentially weighted moving average (default)
    - Yang-Zhang: High-frequency volatility estimator
    - Parkinson: High-Low range estimator
    - Simple: Simple historical volatility
    
    Args:
        price_data: DataFrame with OHLC data
        method: Volatility calculation method
        lookback_days: Number of days for calculation
        
    Returns:
        {
            'volatility': float (annualized),
            'method': str,
            'confidence': float (0-1),
            'data_points': int
        }
    """
    if price_data.empty or len(price_data) < 20:
        logger.warning("Insufficient price data for volatility calculation")
        return {'volatility': 0.0, 'method': method, 'confidence': 0.0, 'data_points': 0}
    
    try:
        # Ensure we have required columns
        required_cols = ['close']
        if method in ['yang_zhang', 'parkinson']:
            required_cols.extend(['open', 'high', 'low'])
        
        missing_cols = [col for col in required_cols if col not in price_data.columns]
        if missing_cols:
            logger.warning(f"Missing columns {missing_cols}, falling back to simple method")
            method = 'simple'
        
        # Use last N days of data
        recent_data = price_data.tail(lookback_days).copy()
        
        if method == 'ewma':
            volatility = _calculate_ewma_volatility(recent_data)
        elif method == 'yang_zhang':
            volatility = _calculate_yang_zhang_volatility(recent_data)
        elif method == 'parkinson':
            volatility = _calculate_parkinson_volatility(recent_data)
        else:  # simple
            volatility = _calculate_simple_volatility(recent_data)
        
        # Calculate confidence based on data quality
        confidence = min(len(recent_data) / lookback_days, 1.0) * 0.9  # Max 90% confidence
        
        return {
            'volatility': volatility,
            'method': method,
            'confidence': confidence,
            'data_points': len(recent_data)
        }
        
    except Exception as e:
        logger.error(f"Volatility calculation failed: {str(e)}")
        return {'volatility': 0.0, 'method': method, 'confidence': 0.0, 'data_points': 0}


def _calculate_ewma_volatility(price_data: pd.DataFrame, decay_factor: float = 0.94) -> float:
    """Calculate EWMA volatility with 64-day half-life (decay = 0.94)"""
    if len(price_data) < 2:
        return 0.0
    
    # Calculate returns
    returns = price_data['close'].pct_change().dropna()
    
    if len(returns) < 2:
        return 0.0
    
    # EWMA calculation
    ewma_var = returns.iloc[0] ** 2  # Initialize with first return squared
    
    for ret in returns.iloc[1:]:
        ewma_var = decay_factor * ewma_var + (1 - decay_factor) * (ret ** 2)
    
    # Annualize (252 trading days, intraday: 78 5-minute bars * 252 days)
    annual_vol = np.sqrt(ewma_var * 252)
    return annual_vol


def _calculate_yang_zhang_volatility(price_data: pd.DataFrame) -> float:
    """Yang-Zhang high-frequency volatility estimator"""
    if len(price_data) < 2:
        return 0.0
    
    # Calculate components
    overnight = np.log(price_data['open'] / price_data['close'].shift(1)).dropna()
    open_to_close = np.log(price_data['close'] / price_data['open'])
    high_to_close = np.log(price_data['high'] / price_data['close'])
    low_to_close = np.log(price_data['low'] / price_data['close'])
    
    # Yang-Zhang formula
    k = 0.34 / (1.34 + (len(price_data) + 1) / (len(price_data) - 1))
    
    rs_var = (high_to_close * (high_to_close - open_to_close) + 
              low_to_close * (low_to_close - open_to_close)).mean()
    
    overnight_var = overnight.var()
    open_close_var = open_to_close.var()
    
    yang_zhang_var = overnight_var + k * open_close_var + (1 - k) * rs_var
    
    return np.sqrt(yang_zhang_var * 252)


def _calculate_parkinson_volatility(price_data: pd.DataFrame) -> float:
    """Parkinson high-low range volatility estimator"""
    if len(price_data) < 2:
        return 0.0
    
    # Parkinson formula: (1/(4*ln(2))) * ln(High/Low)^2
    hl_ratios = np.log(price_data['high'] / price_data['low']) ** 2
    parkinson_var = (hl_ratios / (4 * np.log(2))).mean()
    
    return np.sqrt(parkinson_var * 252)


def _calculate_simple_volatility(price_data: pd.DataFrame) -> float:
    """Simple historical volatility"""
    if len(price_data) < 2:
        return 0.0
    
    returns = price_data['close'].pct_change().dropna()
    return returns.std() * np.sqrt(252)


def apply_leverage_scaling(
    base_position_value: float,
    symbol: str,
    config: Dict[str, Any],
    portfolio: Dict[str, Any]
) -> float:
    """
    Apply symbol-specific leverage limits and scaling
    
    Futures Leverage Guidelines:
    - NIFTY: 8x max (most liquid, stable)
    - BANKNIFTY: 6x max (higher volatility)
    - FINNIFTY: 7x max (moderate volatility)
    - Sectoral indices: 5x max (higher risk)
    """
    # Get symbol-specific leverage limits
    leverage_limits = config.get('leverage_limits', {
        'NIFTY': 8.0,
        'BANKNIFTY': 6.0,
        'FINNIFTY': 7.0,
        'default': 5.0
    })
    
    max_leverage = leverage_limits.get(symbol, leverage_limits.get('default', 5.0))
    
    # Calculate margin requirement for base position
    margin_requirements = {
        'NIFTY': 60000,     # Per lot MIS margin
        'BANKNIFTY': 75000,
        'FINNIFTY': 50000
    }
    
    lot_sizes = {'NIFTY': 50, 'BANKNIFTY': 25, 'FINNIFTY': 40}
    lot_size = lot_sizes.get(symbol, 50)
    
    # Estimate lots needed for base position (rough calculation)
    current_price = config.get('current_price', 22000)  # Should come from market data
    approx_lots = base_position_value / (current_price * lot_size)
    
    margin_per_lot = margin_requirements.get(symbol, 60000)
    total_margin_required = approx_lots * margin_per_lot
    
    # Calculate implied leverage
    if total_margin_required > 0:
        implied_leverage = base_position_value / total_margin_required
    else:
        implied_leverage = 1.0
    
    # Apply leverage cap
    if implied_leverage > max_leverage:
        scaling_factor = max_leverage / implied_leverage
        scaled_position_value = base_position_value * scaling_factor
        
        logger.info(f"Leverage capped: {implied_leverage:.1f}x -> {max_leverage:.1f}x for {symbol}")
    else:
        scaled_position_value = base_position_value
    
    # Check portfolio-wide leverage
    total_capital = portfolio.get('total_capital', 500000)
    current_notional = portfolio.get('total_notional', 0)
    new_total_notional = current_notional + scaled_position_value
    
    portfolio_leverage = new_total_notional / total_capital
    max_portfolio_leverage = config.get('max_portfolio_leverage', 6.0)
    
    if portfolio_leverage > max_portfolio_leverage:
        portfolio_scaling = max_portfolio_leverage / portfolio_leverage
        scaled_position_value *= portfolio_scaling
        
        logger.info(f"Portfolio leverage capped: {portfolio_leverage:.1f}x -> {max_portfolio_leverage:.1f}x")
    
    return scaled_position_value


def optimize_lot_sizing(
    target_position_value: float,
    symbol: str,
    signal: float,
    market_data: Dict[str, Any],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Round to exchange lot sizes with optimal allocation
    
    NSE Lot Sizes and Considerations:
    - NIFTY: 50 shares per lot (₹9-11 lakh per lot at current levels)
    - BANKNIFTY: 25 shares per lot (₹13-15 lakh per lot)
    - FINNIFTY: 40 shares per lot (₹8-10 lakh per lot)
    
    Returns optimal number of lots considering:
    - Minimum tick size (₹0.05)
    - Margin efficiency
    - Position concentration limits
    """
    if target_position_value <= 0:
        return {
            'lots': 0,
            'position_value': 0,
            'margin_required': 0,
            'efficiency_score': 0
        }
    
    # Get contract specifications
    lot_sizes = {'NIFTY': 50, 'BANKNIFTY': 25, 'FINNIFTY': 40}
    lot_size = lot_sizes.get(symbol, 50)
    
    current_price = market_data.get('current_price', 22000)
    value_per_lot = current_price * lot_size
    
    # Calculate target lots (floating point)
    target_lots_float = target_position_value / value_per_lot
    
    # Consider different rounding strategies
    strategies = {
        'floor': math.floor(target_lots_float),
        'ceil': math.ceil(target_lots_float),
        'round': round(target_lots_float)
    }
    
    # Remove zero and negative strategies
    strategies = {k: v for k, v in strategies.items() if v > 0}
    
    if not strategies:
        return {
            'lots': 0,
            'position_value': 0,
            'margin_required': 0,
            'efficiency_score': 0
        }
    
    # Evaluate each strategy
    best_strategy = None
    best_score = -1
    
    for strategy_name, lots in strategies.items():
        actual_value = lots * value_per_lot
        
        # Calculate efficiency score
        target_deviation = abs(actual_value - target_position_value) / target_position_value
        size_penalty = 0 if lots <= 10 else (lots - 10) * 0.02  # Penalty for large positions
        
        efficiency_score = 1.0 - target_deviation - size_penalty
        
        if efficiency_score > best_score:
            best_score = efficiency_score
            best_strategy = {
                'strategy': strategy_name,
                'lots': lots,
                'position_value': actual_value,
                'target_deviation': target_deviation,
                'efficiency_score': efficiency_score
            }
    
    # Calculate margin requirement
    margin_per_lot = {'NIFTY': 60000, 'BANKNIFTY': 75000, 'FINNIFTY': 50000}
    margin_required = best_strategy['lots'] * margin_per_lot.get(symbol, 60000)
    
    best_strategy['margin_required'] = margin_required
    
    logger.debug(f"Lot optimization: {symbol} {best_strategy['lots']} lots "
                f"(₹{best_strategy['position_value']:,.0f}, efficiency: {best_strategy['efficiency_score']:.3f})")
    
    return best_strategy


def apply_position_caps(
    lot_optimization: Dict[str, Any],
    symbol: str,
    portfolio: Dict[str, Any],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Apply final position and exposure caps"""
    
    if lot_optimization['lots'] <= 0:
        return lot_optimization
    
    # Get position limits
    position_limits = config.get('position_limits', {
        'NIFTY': {'daily_lots': 15, 'single_trade': 10},
        'BANKNIFTY': {'daily_lots': 10, 'single_trade': 5},
        'FINNIFTY': {'daily_lots': 12, 'single_trade': 8}
    })
    
    symbol_limits = position_limits.get(symbol, {'daily_lots': 8, 'single_trade': 5})
    
    # Check current positions
    current_positions = portfolio.get('positions', {})
    current_lots = abs(current_positions.get(symbol, {}).get('lots', 0))
    
    # Apply single trade limit
    max_single_trade = symbol_limits['single_trade']
    if lot_optimization['lots'] > max_single_trade:
        logger.warning(f"Single trade limit applied: {lot_optimization['lots']} -> {max_single_trade} lots")
        lot_optimization['lots'] = max_single_trade
        lot_optimization['cap_applied'] = 'single_trade_limit'
    
    # Apply daily limit
    max_daily_lots = symbol_limits['daily_lots']
    new_total_lots = current_lots + lot_optimization['lots']
    
    if new_total_lots > max_daily_lots:
        available_lots = max_daily_lots - current_lots
        if available_lots <= 0:
            logger.warning(f"Daily position limit reached for {symbol}")
            lot_optimization['lots'] = 0
            lot_optimization['cap_applied'] = 'daily_limit_reached'
        else:
            logger.warning(f"Daily limit applied: {lot_optimization['lots']} -> {available_lots} lots")
            lot_optimization['lots'] = available_lots
            lot_optimization['cap_applied'] = 'daily_limit_partial'
    
    # Recalculate values after capping
    if 'cap_applied' in lot_optimization:
        lot_sizes = {'NIFTY': 50, 'BANKNIFTY': 25, 'FINNIFTY': 40}
        lot_size = lot_sizes.get(symbol, 50)
        current_price = config.get('current_price', 22000)
        
        lot_optimization['position_value'] = lot_optimization['lots'] * current_price * lot_size
        
        margin_per_lot = {'NIFTY': 60000, 'BANKNIFTY': 75000, 'FINNIFTY': 50000}
        lot_optimization['margin_required'] = lot_optimization['lots'] * margin_per_lot.get(symbol, 60000)
    
    return lot_optimization


def _calculate_base_position_oxford(
    target_volatility: float,
    instrument_volatility: float,
    signal_strength: float,
    available_capital: float,
    current_price: float
) -> float:
    """Calculate base position size using Oxford formula"""
    
    if instrument_volatility <= 0 or current_price <= 0:
        return 0.0
    
    # Oxford formula: Position = (Target_Vol / Instrument_Vol) * Signal * Capital
    volatility_ratio = target_volatility / instrument_volatility
    base_allocation = volatility_ratio * signal_strength * available_capital
    
    # Cap at reasonable levels
    max_allocation = available_capital * 0.5  # Max 50% of capital per position
    
    return min(base_allocation, max_allocation)


def _create_zero_position_result(reason: str, signal: float) -> Dict[str, Any]:
    """Create result structure for zero position"""
    return {
        'recommended_lots': 0,
        'position_value': 0.0,
        'leverage_used': 0.0,
        'margin_required': 0.0,
        'volatility_estimate': 0.0,
        'sizing_breakdown': {
            'reason': reason,
            'signal': signal,
            'target_lots_before_caps': 0,
            'caps_applied': []
        },
        'risk_metrics': {
            'position_risk_pct': 0.0,
            'leverage_efficiency': 0.0,
            'margin_efficiency': 0.0
        }
    }


def _compile_sizing_result(
    final_sizing: Dict[str, Any],
    volatility_data: Dict[str, float],
    signal: float,
    symbol: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Compile final sizing result with all metrics"""
    
    lots = final_sizing.get('lots', 0)
    position_value = final_sizing.get('position_value', 0)
    margin_required = final_sizing.get('margin_required', 0)
    
    # Calculate leverage and efficiency metrics
    leverage_used = position_value / margin_required if margin_required > 0 else 0.0
    
    total_capital = config.get('total_capital', 500000)
    position_risk_pct = (position_value / total_capital) * 100 if total_capital > 0 else 0.0
    
    leverage_efficiency = leverage_used / config.get('leverage_limits', {}).get(symbol, 5.0)
    margin_efficiency = margin_required / (total_capital * 0.6) if total_capital > 0 else 0.0  # vs 60% max usage
    
    return {
        'recommended_lots': lots,
        'position_value': position_value,
        'leverage_used': leverage_used,
        'margin_required': margin_required,
        'volatility_estimate': volatility_data.get('volatility', 0.0),
        'sizing_breakdown': {
            'signal_strength': signal,
            'volatility_method': volatility_data.get('method', 'unknown'),
            'volatility_confidence': volatility_data.get('confidence', 0.0),
            'target_lots_before_caps': final_sizing.get('target_lots_float', 0),
            'optimization_strategy': final_sizing.get('strategy', 'none'),
            'caps_applied': [final_sizing.get('cap_applied')] if 'cap_applied' in final_sizing else [],
            'efficiency_score': final_sizing.get('efficiency_score', 0.0)
        },
        'risk_metrics': {
            'position_risk_pct': position_risk_pct,
            'leverage_efficiency': leverage_efficiency,
            'margin_efficiency': margin_efficiency,
            'volatility_risk_score': min(volatility_data.get('volatility', 0.0) / 0.25, 1.0)  # Normalized to 25% vol
        }
    }


# Convenience function for integration
def calculate_position_size(
    signal: float,
    symbol: str,
    market_data: Dict[str, Any],
    portfolio: Dict[str, Any],
    risk_config: Dict[str, Any]
) -> int:
    """
    Simplified interface for position size calculation
    Returns number of lots (integer)
    """
    result = compute_position_size_oxford(signal, symbol, market_data, portfolio, risk_config)
    return result['recommended_lots']


if __name__ == "__main__":
    # Test the position sizing system
    import random
    
    # Mock data for testing
    test_signal = 0.75
    test_symbol = 'NIFTY'
    
    # Create mock price data
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    prices = []
    price = 22000
    
    for _ in dates:
        price *= (1 + random.gauss(0, 0.015))  # 1.5% daily volatility
        prices.append(price)
    
    mock_market_data = {
        'price_data': pd.DataFrame({
            'close': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'open': prices
        }, index=dates),
        'current_price': prices[-1]
    }
    
    mock_portfolio = {
        'total_capital': 1000000,
        'available_capital': 800000,
        'positions': {},
        'total_notional': 0
    }
    
    mock_config = {
        'volatility_target': 0.12,
        'volatility_method': 'ewma',
        'vol_lookback_days': 64,
        'leverage_limits': {'NIFTY': 8.0},
        'max_portfolio_leverage': 6.0,
        'current_price': prices[-1]
    }
    
    # Test position sizing
    result = compute_position_size_oxford(
        test_signal, test_symbol, mock_market_data, mock_portfolio, mock_config
    )
    
    print("Oxford Position Sizing Test Results:")
    print(f"Signal: {test_signal:.3f}")
    print(f"Recommended Lots: {result['recommended_lots']}")
    print(f"Position Value: ₹{result['position_value']:,.0f}")
    print(f"Leverage Used: {result['leverage_used']:.1f}x")
    print(f"Margin Required: ₹{result['margin_required']:,.0f}")
    print(f"Volatility Estimate: {result['volatility_estimate']:.1%}")
    print(f"Position Risk: {result['risk_metrics']['position_risk_pct']:.1f}%")