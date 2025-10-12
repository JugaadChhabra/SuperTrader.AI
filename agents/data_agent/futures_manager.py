"""
Futures Contract Management Module

Handles futures-specific operations including:
- Contract rollover logic
- Continuous contract construction
- Basis calculations
- Futures symbol generation
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from .constants import (
    MONTH_CODES,
    TRADING_DAYS_PER_YEAR,
    DEFAULT_RISK_FREE_RATE,
    PRICE_COLUMNS
)
from .validators import validate_rollover_continuity

logger = logging.getLogger(__name__)


def track_rollover_costs(current_df: pd.DataFrame, next_df: pd.DataFrame, 
                        rollover_date: str, symbol: str,
                        config: Optional[Dict] = None) -> Dict[str, float]:
    """Calculate comprehensive rollover transaction costs.
    
    Args:
        current_df: Current contract data around rollover
        next_df: Next contract data around rollover
        rollover_date: Date of rollover
        symbol: Symbol name for cost parameters
        config: Market configuration with cost parameters
        
    Returns:
        Dict with cost breakdown:
        {
            'rollover_cost_bp': Total cost in basis points,
            'bid_ask_spread_bp': Estimated bid-ask spread cost,
            'price_impact_bp': Market impact cost,
            'volume_weighted_cost': Volume-adjusted cost,
            'execution_quality_score': Quality of rollover execution
        }
    """
    try:
        rollover_dt = pd.to_datetime(rollover_date)
        
        # Get data around rollover point (±2 bars for context)
        current_window = current_df[
            (current_df.index >= rollover_dt - pd.Timedelta(minutes=10)) &
            (current_df.index <= rollover_dt)
        ]
        
        next_window = next_df[
            (next_df.index >= rollover_dt) &
            (next_df.index <= rollover_dt + pd.Timedelta(minutes=10))
        ]
        
        if len(current_window) == 0 or len(next_window) == 0:
            logger.warning("Insufficient data for rollover cost calculation: %s", symbol)
            return {
                'rollover_cost_bp': 0.0,
                'error': 'Insufficient data for cost calculation'
            }
        
        # 1. Estimate bid-ask spread using high-low range
        current_spread = (current_window['high'] - current_window['low']) / current_window['close']
        next_spread = (next_window['high'] - next_window['low']) / next_window['close']
        
        avg_current_spread = current_spread.mean()
        avg_next_spread = next_spread.mean()
        
        # Bid-ask cost = exit current + enter next (half spread each way)
        bid_ask_cost_bp = ((avg_current_spread + avg_next_spread) / 2) * 10000
        
        # 2. Price impact based on volume patterns
        current_volume = current_window['volume'].mean()
        next_volume = next_window['volume'].mean()
        
        # Higher volume = lower impact, use inverse relationship
        volume_ratio = min(current_volume, next_volume) / max(current_volume, next_volume, 1)
        volume_impact_bp = max(1.0, 5.0 * (1 - volume_ratio))  # 1-5 bps based on volume
        
        # 3. Symbol-specific cost adjustments from config
        symbol_costs = {}
        if config and 'transaction_costs' in config:
            futures_costs = config['transaction_costs'].get('futures', {})
            symbol_costs = {
                'base_cost_bp': futures_costs.get('total_cost_bp', 2.0),
                'rollover_multiplier': 1.5  # Rollover trades typically more expensive
            }
        
        base_cost_bp = symbol_costs.get('base_cost_bp', 2.0)
        rollover_multiplier = symbol_costs.get('rollover_multiplier', 1.5)
        
        # 4. Calculate total rollover cost
        total_cost_bp = (bid_ask_cost_bp + volume_impact_bp) * rollover_multiplier + base_cost_bp
        
        # 5. Execution quality score (0-100, higher = better)
        price_continuity = abs(current_window.iloc[-1]['close'] - next_window.iloc[0]['close']) / current_window.iloc[-1]['close']
        volume_consistency = min(volume_ratio, 1.0)
        spread_tightness = 1 - min((avg_current_spread + avg_next_spread) / 2, 0.01)  # Cap at 1%
        
        execution_quality = (
            (1 - min(price_continuity * 100, 1.0)) * 0.4 +  # Price continuity 40%
            volume_consistency * 0.3 +                       # Volume consistency 30%  
            spread_tightness * 0.3                          # Spread tightness 30%
        ) * 100
        
        # 6. Volume-weighted cost calculation
        total_volume = current_volume + next_volume
        if total_volume > 0:
            volume_weighted_cost = total_cost_bp * (1000000 / total_volume) if total_volume < 1000000 else total_cost_bp
        else:
            volume_weighted_cost = total_cost_bp * 2  # Penalty for low volume
        
        cost_result = {
            'rollover_cost_bp': round(total_cost_bp, 2),
            'bid_ask_spread_bp': round(bid_ask_cost_bp, 2),
            'price_impact_bp': round(volume_impact_bp, 2),
            'base_transaction_cost_bp': round(base_cost_bp, 2),
            'volume_weighted_cost': round(volume_weighted_cost, 2),
            'execution_quality_score': round(execution_quality, 1),
            'volume_ratio': round(volume_ratio, 3),
            'current_volume': int(current_volume),
            'next_volume': int(next_volume),
            'rollover_timestamp': rollover_date
        }
        
        logger.debug("Rollover costs for %s: %.2f bps total, quality: %.1f", 
                    symbol, total_cost_bp, execution_quality)
        
        return cost_result
        
    except Exception as e:
        logger.error("Rollover cost calculation failed for %s: %s", symbol, e)
        return {
            'rollover_cost_bp': 10.0,  # Conservative fallback
            'error': str(e),
            'execution_quality_score': 50.0  # Neutral score
        }


def auto_detect_rollover_dates(contracts_data: Dict[str, pd.DataFrame], 
                              symbol: str, config: Optional[Dict] = None) -> List[str]:
    """Auto-detect optimal rollover dates based on volume/OI patterns.
    
    Args:
        contracts_data: Dict of contract DataFrames
        symbol: Symbol name for logging
        config: Market config with rollover parameters
        
    Returns:
        List of rollover dates in chronological order
    """
    rollover_dates = []
    
    try:
        # Get rollover config parameters
        if config and 'rollover_config' in config:
            trigger_days = config['rollover_config'].get('trigger_days_before_expiry', 3)
            oi_threshold = config['rollover_config'].get('oi_ratio_threshold', 0.3)
            volume_threshold = config['rollover_config'].get('volume_ratio_threshold', 0.2)
        else:
            trigger_days = 3
            oi_threshold = 0.3
            volume_threshold = 0.2
        
        contract_names = sorted(contracts_data.keys())
        
        for i in range(len(contract_names) - 1):
            current_name = contract_names[i]
            next_name = contract_names[i + 1]
            
            current_df = contracts_data[current_name]
            next_df = contracts_data[next_name]
            
            if current_df.empty or next_df.empty:
                continue
                
            # Find overlapping period between contracts
            overlap_start = max(current_df.index.min(), next_df.index.min())
            overlap_end = min(current_df.index.max(), next_df.index.max())
            
            if overlap_start >= overlap_end:
                logger.warning("No overlap between %s and %s contracts for %s", 
                             current_name, next_name, symbol)
                continue
            
            # Get overlapping data
            current_overlap = current_df[
                (current_df.index >= overlap_start) & 
                (current_df.index <= overlap_end)
            ]
            next_overlap = next_df[
                (next_df.index >= overlap_start) & 
                (next_df.index <= overlap_end)
            ]
            
            # Method 1: Volume-based detection
            rollover_date_volume = detect_rollover_by_volume(
                current_overlap, next_overlap, volume_threshold, trigger_days
            )
            
            # Method 2: Open Interest-based detection (if available)
            rollover_date_oi = None
            if 'open_interest' in current_overlap.columns and 'open_interest' in next_overlap.columns:
                rollover_date_oi = detect_rollover_by_oi(
                    current_overlap, next_overlap, oi_threshold, trigger_days
                )
            
            # Choose the earlier rollover date (more conservative)
            rollover_candidates = [d for d in [rollover_date_volume, rollover_date_oi] if d is not None]
            
            if rollover_candidates:
                rollover_date = min(rollover_candidates)
                rollover_dates.append(rollover_date)
                logger.info("Auto-detected rollover for %s: %s -> %s on %s", 
                           symbol, current_name, next_name, rollover_date)
            else:
                # Fallback: Use fixed days before end of overlap
                fallback_date = (overlap_end - pd.Timedelta(days=trigger_days)).strftime('%Y-%m-%d')
                rollover_dates.append(fallback_date)
                logger.warning("Using fallback rollover date for %s: %s", symbol, fallback_date)
        
        return rollover_dates
        
    except Exception as e:
        logger.error("Auto-detection of rollover dates failed for %s: %s", symbol, e)
        return []


def detect_rollover_by_volume(current_df: pd.DataFrame, next_df: pd.DataFrame, 
                             threshold: float, max_days: int) -> Optional[str]:
    """Detect rollover point based on volume crossover."""
    try:
        # Calculate volume ratio (next/current) for overlapping period
        common_index = current_df.index.intersection(next_df.index)
        
        if len(common_index) < 5:  # Need minimum data
            return None
            
        current_vol = current_df.loc[common_index, 'volume']
        next_vol = next_df.loc[common_index, 'volume']
        
        # Avoid division by zero
        volume_ratio = next_vol / (current_vol + 1)
        
        # Find when next contract volume exceeds threshold
        crossover_points = volume_ratio[volume_ratio > threshold]
        
        if len(crossover_points) > 0:
            return crossover_points.index[0].strftime('%Y-%m-%d')
        
        return None
        
    except Exception as e:
        logger.debug("Volume-based rollover detection failed: %s", e)
        return None


def detect_rollover_by_oi(current_df: pd.DataFrame, next_df: pd.DataFrame,
                         threshold: float, max_days: int) -> Optional[str]:
    """Detect rollover point based on open interest crossover."""
    try:
        common_index = current_df.index.intersection(next_df.index)
        
        if len(common_index) < 5:
            return None
            
        current_oi = current_df.loc[common_index, 'open_interest']
        next_oi = next_df.loc[common_index, 'open_interest']
        
        # Calculate OI ratio
        oi_ratio = next_oi / (current_oi + 1)
        
        # Find when next contract OI exceeds threshold
        crossover_points = oi_ratio[oi_ratio > threshold]
        
        if len(crossover_points) > 0:
            return crossover_points.index[0].strftime('%Y-%m-%d')
            
        return None
        
    except Exception as e:
        logger.debug("OI-based rollover detection failed: %s", e)
        return None


def handle_contract_rollover(current_contract: pd.DataFrame, next_contract: pd.DataFrame,
                           rollover_date: str, method: str = 'ratio', 
                           symbol: str = "", config: Optional[Dict] = None) -> Dict[str, Any]:
    """Enhanced contract rollover with comprehensive validation and cost tracking.
    
    Args:
        current_contract: DataFrame for expiring contract
        next_contract: DataFrame for new contract  
        rollover_date: Date to perform rollover (YYYY-MM-DD)
        method: 'ratio' or 'difference' adjustment method
        symbol: Symbol name for logging and cost calculations
        config: Market configuration dict
        
    Returns:
        Dict with rollover results:
        {
            'continuous_contract': pd.DataFrame,
            'rollover_costs': Dict,
            'adjustment_factor': float,
            'validation_results': Dict,
            'rollover_metadata': Dict
        }
    """
    if current_contract.empty or next_contract.empty:
        logger.error("Empty contract data for rollover of %s", symbol)
        return {
            'continuous_contract': pd.DataFrame(),
            'error': 'Empty contract data',
            'rollover_costs': {},
            'validation_results': {}
        }
    
    rollover_dt = pd.to_datetime(rollover_date)
    
    # 1. Validate rollover continuity using Phase 1 validation
    continuity_results = validate_rollover_continuity(
        current_contract, next_contract, rollover_date, symbol
    )
    
    if 'error' in continuity_results:
        logger.error("Rollover validation failed for %s: %s", symbol, continuity_results['error'])
        return {
            'continuous_contract': current_contract,
            'error': continuity_results['error'],
            'rollover_costs': {},
            'validation_results': continuity_results
        }
    
    # 2. Find the rollover point data with buffer
    current_before = current_contract[current_contract.index <= rollover_dt]
    next_after = next_contract[next_contract.index >= rollover_dt]
    
    if current_before.empty or next_after.empty:
        logger.error("Insufficient data around rollover date %s for %s", rollover_date, symbol)
        return {
            'continuous_contract': current_contract,
            'error': f'Insufficient data around {rollover_date}',
            'rollover_costs': {},
            'validation_results': continuity_results
        }
    
    # 3. Calculate rollover costs before adjustment
    rollover_costs = track_rollover_costs(
        current_before, next_after, rollover_date, symbol, config
    )
    
    # 4. Get prices at rollover point (use last/first valid prices)
    current_price = current_before.iloc[-1]['close']
    next_price = next_after.iloc[0]['close']
    
    # 5. Apply price adjustment based on method
    adjusted_next = next_after.copy()
    adjustment_factor = 0.0
    
    if method == 'ratio':
        # Ratio adjustment (multiplicative) - preserves percentage moves
        adjustment_factor = current_price / next_price
        
        for col in PRICE_COLUMNS:
            if col in adjusted_next.columns:
                adjusted_next[col] = adjusted_next[col] * adjustment_factor
        
        logger.info("Applied ratio adjustment for %s: factor=%.6f", symbol, adjustment_factor)
        
    elif method == 'difference':
        # Difference adjustment (additive) - preserves absolute moves
        adjustment_factor = current_price - next_price
        
        for col in PRICE_COLUMNS:
            if col in adjusted_next.columns:
                adjusted_next[col] = adjusted_next[col] + adjustment_factor
        
        logger.info("Applied difference adjustment for %s: diff=%.2f", symbol, adjustment_factor)
        
    else:
        logger.error("Unknown rollover method: %s for %s", method, symbol)
        return {
            'continuous_contract': current_contract,
            'error': f'Unknown rollover method: {method}',
            'rollover_costs': rollover_costs,
            'validation_results': continuity_results
        }
    
    # 6. Combine the series seamlessly
    continuous_series = pd.concat([current_before, adjusted_next])
    continuous_series = continuous_series.sort_index()
    
    # 7. Add comprehensive metadata
    rollover_metadata = {
        'rollover_date': rollover_date,
        'rollover_method': method,
        'adjustment_factor': adjustment_factor,
        'original_gap_bps': continuity_results.get('price_gap_bps', 0),
        'contracts_merged': 2,
        'data_quality_score': continuity_results.get('continuity_score', 0),
        'needs_review': continuity_results.get('needs_adjustment', False),
        'rollover_timestamp': datetime.now().isoformat()
    }
    
    continuous_series.attrs.update(rollover_metadata)
    
    # 8. Final validation check
    if len(continuous_series) < len(current_before) + len(next_after) * 0.5:
        logger.warning("Significant data loss during rollover for %s", symbol)
        rollover_metadata['data_loss_warning'] = True
    
    logger.info("Rollover completed for %s: %.1f bps gap, %.1f bps cost, quality: %.1f", 
               symbol, continuity_results.get('price_gap_bps', 0),
               rollover_costs.get('rollover_cost_bp', 0), 
               continuity_results.get('continuity_score', 0))
    
    return {
        'continuous_contract': continuous_series,
        'rollover_costs': rollover_costs,
        'adjustment_factor': adjustment_factor,
        'validation_results': continuity_results,
        'rollover_metadata': rollover_metadata,
        'success': True
    }


def construct_continuous_contract(contracts_data: Dict[str, pd.DataFrame], 
                                symbol: str, method: str = 'ratio',
                                config: Optional[Dict] = None,
                                rollover_schedule: Optional[List[str]] = None) -> Dict[str, Any]:
    """Construct continuous futures contract from multiple expiry series with auto-rollover.
    
    Args:
        contracts_data: Dict mapping contract_month -> DataFrame (e.g., 'current', 'next', 'far')
        symbol: Symbol name for logging and configuration
        method: 'ratio' or 'difference' adjustment method
        config: Market configuration dict
        rollover_schedule: Optional manual rollover dates, otherwise auto-detected
        
    Returns:
        Dict with continuous contract results:
        {
            'continuous_contract': pd.DataFrame,
            'rollover_history': List[Dict],
            'total_rollover_cost': float,
            'construction_metadata': Dict
        }
    """
    if not contracts_data:
        logger.error("No contract data provided for %s continuous construction", symbol)
        return {
            'continuous_contract': pd.DataFrame(),
            'error': 'No contract data provided',
            'rollover_history': [],
            'total_rollover_cost': 0.0
        }
    
    contract_names = sorted(contracts_data.keys())
    if len(contract_names) < 2:
        logger.warning("Need at least 2 contracts for %s continuous series, got %d", 
                      symbol, len(contract_names))
        return {
            'continuous_contract': list(contracts_data.values())[0],
            'rollover_history': [],
            'total_rollover_cost': 0.0,
            'construction_metadata': {'single_contract': True}
        }
    
    # Auto-detect rollover dates if not provided
    if rollover_schedule is None:
        rollover_schedule = auto_detect_rollover_dates(contracts_data, symbol, config)
    
    # Start with the first contract
    continuous_series = contracts_data[contract_names[0]].copy()
    rollover_history = []
    total_cost_bp = 0.0
    
    # Apply rollovers sequentially with full tracking
    for i, rollover_date in enumerate(rollover_schedule):
        if i + 1 >= len(contract_names):
            break
            
        current_contract = continuous_series
        next_contract_name = contract_names[i + 1]
        next_contract = contracts_data[next_contract_name]
        
        # Execute rollover with full cost tracking
        rollover_result = handle_contract_rollover(
            current_contract, next_contract, rollover_date, method, symbol, config
        )
        
        if rollover_result.get('success', False):
            continuous_series = rollover_result['continuous_contract']
            
            # Track rollover for history
            rollover_record = {
                'rollover_date': rollover_date,
                'from_contract': contract_names[i] if i == 0 else 'continuous',
                'to_contract': next_contract_name,
                'method': method,
                'cost_bp': rollover_result['rollover_costs'].get('rollover_cost_bp', 0),
                'adjustment_factor': rollover_result['adjustment_factor'],
                'quality_score': rollover_result['validation_results'].get('continuity_score', 0),
                'data_bars_added': len(next_contract)
            }
            
            rollover_history.append(rollover_record)
            total_cost_bp += rollover_result['rollover_costs'].get('rollover_cost_bp', 0)
            
            logger.info("Rollover %d/%d completed for %s: %s -> %s (%.2f bps)", 
                       i + 1, len(rollover_schedule), symbol,
                       rollover_record['from_contract'], next_contract_name,
                       rollover_record['cost_bp'])
        else:
            logger.error("Rollover %d failed for %s: %s", i + 1, symbol, 
                        rollover_result.get('error', 'Unknown error'))
            break
    
    # Add comprehensive continuous contract metadata
    construction_metadata = {
        'contract_type': 'continuous',
        'symbol': symbol,
        'rollover_method': method,
        'num_rollovers_completed': len(rollover_history),
        'num_rollovers_planned': len(rollover_schedule),
        'contracts_used': contract_names[:len(rollover_history) + 1],
        'construction_success': len(rollover_history) == len(rollover_schedule),
        'total_bars': len(continuous_series),
        'date_range': {
            'start': continuous_series.index.min().isoformat() if len(continuous_series) > 0 else None,
            'end': continuous_series.index.max().isoformat() if len(continuous_series) > 0 else None
        },
        'avg_rollover_cost_bp': total_cost_bp / max(len(rollover_history), 1),
        'construction_timestamp': datetime.now().isoformat()
    }
    
    continuous_series.attrs.update(construction_metadata)
    
    logger.info("Continuous contract construction completed for %s: %d rollovers, %.2f bps total cost", 
               symbol, len(rollover_history), total_cost_bp)
    
    return {
        'continuous_contract': continuous_series,
        'rollover_history': rollover_history,
        'total_rollover_cost': total_cost_bp,
        'construction_metadata': construction_metadata,
        'success': construction_metadata['construction_success']
    }


def compute_basis(futures_df: pd.DataFrame, spot_df: pd.DataFrame, 
                 expiry_date: str, symbol: str = "",
                 risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
                 config: Optional[Dict] = None) -> pd.DataFrame:
    """Enhanced basis calculation with mean reversion signals and trading insights.
    
    Args:
        futures_df: DataFrame with futures OHLCV data
        spot_df: DataFrame with spot index OHLCV data  
        expiry_date: Contract expiry date (YYYY-MM-DD)
        symbol: Symbol name for logging
        risk_free_rate: Risk-free rate for fair value calculation
        config: Market configuration dict
        
    Returns:
        DataFrame with comprehensive basis analysis:
        - basis, basis_pct, annualized_basis
        - theoretical_fair_value, basis_deviation  
        - basis_zscore, basis_percentile
        - mean_reversion_signal, carry_trade_signal
    """
    try:
        # Align data by timestamp
        common_index = futures_df.index.intersection(spot_df.index)
        
        if len(common_index) == 0:
            logger.error("No common timestamps between futures and spot data for %s", symbol)
            return pd.DataFrame()
        
        futures_aligned = futures_df.loc[common_index]
        spot_aligned = spot_df.loc[common_index]
        
        # Calculate days to expiry
        expiry_dt = pd.to_datetime(expiry_date)
        days_to_expiry = (expiry_dt - common_index).days
        days_to_expiry = pd.Series(days_to_expiry, index=common_index)
        
        # Basic basis calculations
        futures_price = futures_aligned['close']
        spot_price = spot_aligned['close']
        
        basis = futures_price - spot_price
        basis_pct = (basis / spot_price) * 100
        
        # Annualized basis
        years_to_expiry = days_to_expiry / 365.25
        annualized_basis = basis_pct / years_to_expiry.replace(0, np.nan)
        
        # Theoretical fair value (cost of carry model)
        # Fair Value = Spot * exp(r * T) where r = risk-free rate, T = time to expiry
        theoretical_futures = spot_price * np.exp(risk_free_rate * years_to_expiry)
        theoretical_basis = theoretical_futures - spot_price
        basis_deviation = basis - theoretical_basis
        
        # Enhanced mean reversion indicators
        basis_rolling_mean = basis_pct.rolling(20, min_periods=10).mean()
        basis_rolling_std = basis_pct.rolling(20, min_periods=10).std()
        
        # Z-score for mean reversion (how many std devs from mean)
        basis_zscore = (basis_pct - basis_rolling_mean) / basis_rolling_std
        
        # Percentile ranking (0-100, where 50 = median)
        basis_percentile = basis_pct.rolling(50, min_periods=20).rank(pct=True) * 100
        
        # Mean reversion signal (-1 to +1)
        # Negative basis z-score = oversold = buy signal
        mean_reversion_signal = -np.tanh(basis_zscore / 2)  # Smooth between -1 and +1
        
        # Carry trade signal (positive carry = contango)
        carry_signal = np.where(
            annualized_basis > risk_free_rate * 100,  # Contango
            1,   # Positive carry (sell futures, buy spot)
            np.where(
                annualized_basis < -risk_free_rate * 50,  # Strong backwardation  
                -1,  # Negative carry (buy futures, sell spot)
                0    # Neutral
            )
        )
        
        # Basis momentum (rate of change)
        basis_momentum = basis_pct.pct_change(periods=5) * 100
        
        # Convergence indicator (basis should converge to 0 near expiry)
        convergence_expected = theoretical_basis * (days_to_expiry / days_to_expiry.iloc[0])
        convergence_deviation = abs(basis - convergence_expected) / spot_price * 100
        
        # Combine all metrics into result DataFrame
        result_df = pd.DataFrame({
            'futures_price': futures_price,
            'spot_price': spot_price,
            'basis': basis,
            'basis_pct': basis_pct,
            'basis_bps': basis_pct * 100,  # Basis in basis points
            'annualized_basis_pct': annualized_basis,
            'days_to_expiry': days_to_expiry,
            'theoretical_futures': theoretical_futures,
            'theoretical_basis': theoretical_basis,
            'basis_deviation': basis_deviation,
            'basis_zscore': basis_zscore,
            'basis_percentile': basis_percentile,
            'mean_reversion_signal': mean_reversion_signal,
            'carry_trade_signal': carry_signal,
            'basis_momentum': basis_momentum,
            'convergence_deviation_pct': convergence_deviation
        }, index=common_index)
        
        # Add metadata
        result_df.attrs = {
            'symbol': symbol,
            'expiry_date': expiry_date,
            'risk_free_rate': risk_free_rate,
            'calculation_timestamp': datetime.now().isoformat(),
            'basis_stats': {
                'mean_basis_bps': float(basis_pct.mean() * 100),
                'basis_volatility_bps': float(basis_pct.std() * 100),
                'max_basis_bps': float(basis_pct.max() * 100),
                'min_basis_bps': float(basis_pct.min() * 100)
            }
        }
        
        logger.info("Basis calculation completed for %s: avg %.1f bps, vol %.1f bps", 
                   symbol, result_df.attrs['basis_stats']['mean_basis_bps'],
                   result_df.attrs['basis_stats']['basis_volatility_bps'])
        
        return result_df
        
    except Exception as e:
        logger.error("Basis calculation failed for %s: %s", symbol, e)
        return pd.DataFrame()


def analyze_rollover_patterns(rollover_history: List[Dict], symbol: str) -> Dict[str, Any]:
    """Analyze historical rollover patterns for optimization and insights.
    
    Args:
        rollover_history: List of rollover records from continuous contract construction
        symbol: Symbol name for analysis
        
    Returns:
        Dict with rollover pattern analysis
    """
    if not rollover_history:
        return {'error': 'No rollover history provided'}
    
    try:
        rollover_df = pd.DataFrame(rollover_history)
        
        # Convert dates for analysis
        rollover_df['rollover_date'] = pd.to_datetime(rollover_df['rollover_date'])
        rollover_df['month'] = rollover_df['rollover_date'].dt.month
        rollover_df['year'] = rollover_df['rollover_date'].dt.year
        
        analysis_results = {
            'symbol': symbol,
            'total_rollovers': len(rollover_history),
            'analysis_period': {
                'start_date': rollover_df['rollover_date'].min().strftime('%Y-%m-%d'),
                'end_date': rollover_df['rollover_date'].max().strftime('%Y-%m-%d'),
                'years_covered': (rollover_df['rollover_date'].max() - rollover_df['rollover_date'].min()).days / 365.25
            }
        }
        
        # Cost analysis
        costs = rollover_df['cost_bp'].dropna()
        if len(costs) > 0:
            analysis_results['cost_analysis'] = {
                'average_cost_bp': float(costs.mean()),
                'median_cost_bp': float(costs.median()),
                'std_cost_bp': float(costs.std()),
                'min_cost_bp': float(costs.min()),
                'max_cost_bp': float(costs.max()),
                'total_cost_bp': float(costs.sum()),
                'cost_percentiles': {
                    '25th': float(costs.quantile(0.25)),
                    '75th': float(costs.quantile(0.75)),
                    '90th': float(costs.quantile(0.90))
                }
            }
        
        # Quality analysis
        quality_scores = rollover_df['quality_score'].dropna()
        if len(quality_scores) > 0:
            analysis_results['quality_analysis'] = {
                'average_quality': float(quality_scores.mean()),
                'median_quality': float(quality_scores.median()),
                'poor_quality_count': int((quality_scores < 70).sum()),
                'excellent_quality_count': int((quality_scores > 90).sum())
            }
        
        # Seasonal patterns
        if len(rollover_df) >= 12:  # Need at least a year of data
            monthly_costs = rollover_df.groupby('month')['cost_bp'].agg(['mean', 'count', 'std']).round(2)
            analysis_results['seasonal_patterns'] = {
                'monthly_avg_costs': monthly_costs['mean'].to_dict(),
                'monthly_rollover_counts': monthly_costs['count'].to_dict(),
                'most_expensive_month': int(monthly_costs['mean'].idxmax()),
                'least_expensive_month': int(monthly_costs['mean'].idxmin())
            }
        
        # Method effectiveness
        if 'method' in rollover_df.columns:
            method_analysis = rollover_df.groupby('method').agg({
                'cost_bp': ['mean', 'std', 'count'],
                'quality_score': ['mean', 'std']
            }).round(2)
            
            analysis_results['method_analysis'] = {}
            for method in method_analysis.index:
                analysis_results['method_analysis'][method] = {
                    'avg_cost_bp': float(method_analysis.loc[method, ('cost_bp', 'mean')]),
                    'cost_std_bp': float(method_analysis.loc[method, ('cost_bp', 'std')]),
                    'usage_count': int(method_analysis.loc[method, ('cost_bp', 'count')]),
                    'avg_quality': float(method_analysis.loc[method, ('quality_score', 'mean')])
                }
        
        # Recommendations based on analysis
        recommendations = []
        
        if len(costs) > 0:
            if costs.mean() > 10:
                recommendations.append("High average rollover cost detected - consider optimizing rollover timing")
            
            if costs.std() > costs.mean() * 0.5:
                recommendations.append("High cost variability - review rollover trigger conditions")
        
        if len(quality_scores) > 0:
            if quality_scores.mean() < 80:
                recommendations.append("Low average rollover quality - review data continuity")
        
        analysis_results['recommendations'] = recommendations
        analysis_results['analysis_timestamp'] = datetime.now().isoformat()
        
        logger.info("Rollover pattern analysis completed for %s: %d rollovers, avg cost %.1f bps", 
                   symbol, len(rollover_history), analysis_results.get('cost_analysis', {}).get('average_cost_bp', 0))
        
        return analysis_results
        
    except Exception as e:
        logger.error("Rollover pattern analysis failed for %s: %s", symbol, e)
        return {'error': str(e)}


def auto_rollover_trigger(current_contract: pd.DataFrame, next_contract: pd.DataFrame,
                         expiry_date: str, symbol: str, 
                         config: Optional[Dict] = None) -> Dict[str, Any]:
    """Determine optimal rollover timing based on multiple market factors.
    
    Args:
        current_contract: Current contract DataFrame with OHLCV data
        next_contract: Next contract DataFrame with OHLCV data
        expiry_date: Contract expiry date (YYYY-MM-DD)
        symbol: Symbol name for configuration
        config: Market configuration dict
        
    Returns:
        Dict with rollover recommendation:
        {
            'should_rollover': bool,
            'rollover_date': str,
            'confidence_score': float (0-100),
            'trigger_reasons': List[str],
            'market_conditions': Dict
        }
    """
    try:
        expiry_dt = pd.to_datetime(expiry_date)
        current_date = datetime.now()
        days_to_expiry = (expiry_dt - pd.to_datetime(current_date)).days
        
        # Get configuration parameters
        if config and 'rollover_config' in config:
            rollover_cfg = config['rollover_config']
            trigger_days = rollover_cfg.get('trigger_days_before_expiry', 3)
            oi_threshold = rollover_cfg.get('oi_ratio_threshold', 0.3)
            volume_threshold = rollover_cfg.get('volume_ratio_threshold', 0.2)
        else:
            trigger_days = 3
            oi_threshold = 0.3
            volume_threshold = 0.2
        
        recommendation = {
            'symbol': symbol,
            'expiry_date': expiry_date,
            'days_to_expiry': days_to_expiry,
            'should_rollover': False,
            'confidence_score': 0.0,
            'trigger_reasons': [],
            'rollover_date': None,
            'market_conditions': {}
        }
        
        # Check 1: Time-based trigger
        time_trigger_met = days_to_expiry <= trigger_days
        if time_trigger_met:
            recommendation['trigger_reasons'].append(f"Within {trigger_days} days of expiry")
            recommendation['confidence_score'] += 30
        
        # Check 2: Volume analysis
        if not current_contract.empty and not next_contract.empty:
            # Get recent volume data (last 5 days)
            recent_current = current_contract.tail(20)  # Last 20 bars
            recent_next = next_contract.tail(20)
            
            if len(recent_current) > 0 and len(recent_next) > 0:
                avg_current_volume = recent_current['volume'].mean()
                avg_next_volume = recent_next['volume'].mean()
                
                volume_ratio = avg_next_volume / (avg_current_volume + 1)  # Avoid div by zero
                
                recommendation['market_conditions']['volume_ratio'] = round(volume_ratio, 3)
                recommendation['market_conditions']['current_avg_volume'] = int(avg_current_volume)
                recommendation['market_conditions']['next_avg_volume'] = int(avg_next_volume)
                
                if volume_ratio > volume_threshold:
                    recommendation['trigger_reasons'].append(f"Volume shifted to next contract ({volume_ratio:.2f})")
                    recommendation['confidence_score'] += 25
        
        # Check 3: Open Interest analysis (if available)
        if ('open_interest' in current_contract.columns and 
            'open_interest' in next_contract.columns and
            not current_contract.empty and not next_contract.empty):
            
            recent_current_oi = current_contract.tail(10)['open_interest'].mean()
            recent_next_oi = next_contract.tail(10)['open_interest'].mean()
            
            oi_ratio = recent_next_oi / (recent_current_oi + 1)
            
            recommendation['market_conditions']['oi_ratio'] = round(oi_ratio, 3)
            recommendation['market_conditions']['current_avg_oi'] = int(recent_current_oi)
            recommendation['market_conditions']['next_avg_oi'] = int(recent_next_oi)
            
            if oi_ratio > oi_threshold:
                recommendation['trigger_reasons'].append(f"OI shifted to next contract ({oi_ratio:.2f})")
                recommendation['confidence_score'] += 25
        
        # Check 4: Price continuity assessment
        if not current_contract.empty and not next_contract.empty:
            current_price = current_contract.iloc[-1]['close']
            next_price = next_contract.iloc[-1]['close']
            price_gap_pct = abs(current_price - next_price) / current_price * 100
            
            recommendation['market_conditions']['price_gap_pct'] = round(price_gap_pct, 2)
            
            if price_gap_pct < 1.0:  # Good price continuity
                recommendation['confidence_score'] += 10
            elif price_gap_pct > 3.0:  # Large gap might indicate issues
                recommendation['trigger_reasons'].append(f"Large price gap detected ({price_gap_pct:.1f}%)")
                recommendation['confidence_score'] -= 10
        
        # Check 5: Market volatility consideration
        if not current_contract.empty and len(current_contract) >= 20:
            recent_returns = current_contract['close'].pct_change().tail(20)
            volatility = recent_returns.std() * np.sqrt(252) * 100  # Annualized volatility
            
            recommendation['market_conditions']['recent_volatility_pct'] = round(volatility, 1)
            
            # High volatility = less confidence in rollover timing
            if volatility > 30:  # High vol regime
                recommendation['confidence_score'] -= 5
                recommendation['trigger_reasons'].append("High volatility - monitor closely")
        
        # Final decision logic
        mandatory_rollover = days_to_expiry <= 1  # Must rollover if 1 day or less
        
        if mandatory_rollover:
            recommendation['should_rollover'] = True
            recommendation['confidence_score'] = max(recommendation['confidence_score'], 90)
            recommendation['trigger_reasons'].append("Mandatory rollover - expiry imminent")
        elif recommendation['confidence_score'] >= 50:
            recommendation['should_rollover'] = True
        
        # Set recommended rollover date
        if recommendation['should_rollover']:
            if mandatory_rollover:
                recommendation['rollover_date'] = current_date.strftime('%Y-%m-%d')
            else:
                recommendation['rollover_date'] = (current_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Ensure confidence score is between 0-100
        recommendation['confidence_score'] = max(0, min(100, recommendation['confidence_score']))
        
        logger.info("Rollover trigger analysis for %s: %s (confidence: %.0f%%, reasons: %d)", 
                   symbol, "ROLLOVER" if recommendation['should_rollover'] else "HOLD",
                   recommendation['confidence_score'], len(recommendation['trigger_reasons']))
        
        return recommendation
        
    except Exception as e:
        logger.error("Rollover trigger analysis failed for %s: %s", symbol, e)
        return {
            'should_rollover': False,
            'error': str(e),
            'confidence_score': 0
        }


def generate_futures_symbol(index_symbol: str, contract_month: str, 
                           market_config: Dict[str, Any]) -> str:
    """Generate NSE futures symbol based on index and contract month.
    
    Args:
        index_symbol: Index name (e.g., 'NIFTY')
        contract_month: Contract month ('current', 'next', 'far')
        market_config: Market configuration dict
        
    Returns:
        NSE futures symbol (e.g., 'NIFTY25JANFUT')
    """
    # Get current date to determine contract expiry months
    current_date = datetime.now()
    
    # Map contract month to actual month
    if contract_month == 'current':
        expiry_date = current_date
    elif contract_month == 'next':
        expiry_date = current_date + timedelta(days=30)
    elif contract_month == 'far':
        expiry_date = current_date + timedelta(days=60)
    else:
        # Assume it's already a month name or date
        expiry_date = current_date
    
    # Get month codes from config or use defaults
    config_month_codes = market_config.get('contract_naming', {}).get('month_codes', {})
    month_codes_map = {f"{i:02d}": month for i, month in MONTH_CODES.items()}
    month_codes_map.update(config_month_codes)
    
    year = str(expiry_date.year)[-2:]  # Last 2 digits of year
    month_num = expiry_date.strftime("%m")
    month_code = month_codes_map.get(month_num, "JAN")
    
    # NSE naming convention: {INDEX}{YY}{MON}FUT
    futures_symbol = f"{index_symbol}{year}{month_code}FUT"
    
    return futures_symbol