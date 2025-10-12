"""
Data Validation Module

Provides comprehensive data quality checks and validation for:
- Futures data integrity
- OHLCV data consistency  
- Price spikes and gaps detection
- Quality scoring
"""

import logging
from typing import Optional, Dict, Any, Tuple, List
from datetime import time

import pandas as pd
import numpy as np

from .constants import (
    OHLCVI_COLUMNS,
    PRICE_COLUMNS,
    PRICE_SPIKE_THRESHOLD_MULTIPLIER,
    TIME_GAP_THRESHOLD_MULTIPLIER,
    ZERO_VOLUME_THRESHOLD,
    OI_CHANGE_THRESHOLD
)

logger = logging.getLogger(__name__)


def validate_trading_hours(df: pd.DataFrame, symbol: str, config: Optional[Dict] = None) -> pd.DataFrame:
    """Filter out after-hours data for index futures using market config.
    
    Args:
        df: DataFrame with datetime index
        symbol: Symbol name for logging
        config: Market config dict with trading sessions
        
    Returns:
        Filtered DataFrame with only regular trading hours data
    """
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return df
    
    # Get trading hours from config (fallback to hardcoded NSE hours)
    if config and 'market_sessions' in config:
        regular_session = config['market_sessions']['nse_derivatives']['regular_trading']
        start_time_str, end_time_str = regular_session.split('-')
    else:
        start_time_str, end_time_str = "09:15", "15:30"  # NSE regular hours
    
    # Convert to time objects
    market_open = pd.to_datetime(start_time_str, format='%H:%M').time()
    market_close = pd.to_datetime(end_time_str, format='%H:%M').time()
    
    # Filter data within trading hours
    time_mask = (df.index.time >= market_open) & (df.index.time <= market_close)
    filtered_df = df[time_mask].copy()
    
    excluded_count = len(df) - len(filtered_df)
    if excluded_count > 0:
        logger.info("Filtered %d after-hours bars for %s (%d remaining)", 
                   excluded_count, symbol, len(filtered_df))
    
    return filtered_df


def detect_rollover_gaps(df: pd.DataFrame, symbol: str, config: Optional[Dict] = None) -> pd.DataFrame:
    """Detect artificial gaps from contract rollovers using price jump analysis.
    
    Args:
        df: DataFrame with OHLCV data
        symbol: Symbol name for logging
        config: Market config dict with rollover thresholds
        
    Returns:
        DataFrame with rollover gap flags and metadata
    """
    if len(df) < 10:
        return df
    
    # Get rollover config (fallback to defaults)
    if config and 'rollover_config' in config:
        gap_threshold = config['rollover_config']['gap_threshold_pct'] / 100
        volume_multiple = config['rollover_config']['volume_multiple']
    else:
        gap_threshold = 0.02  # 2% default
        volume_multiple = 1.5  # 1.5x volume average
    
    # Calculate overnight gaps (open vs previous close)
    overnight_gaps = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    overnight_gaps = overnight_gaps.fillna(0)
    
    # Detect large gaps with volume confirmation
    large_gaps = abs(overnight_gaps) > gap_threshold
    volume_avg = df['volume'].rolling(20, min_periods=5).mean()
    high_volume = df['volume'] > volume_avg * volume_multiple
    
    # Rollover gaps = large price gaps + high volume
    rollover_flags = large_gaps & high_volume
    
    # Add rollover metadata
    result_df = df.copy()
    result_df['rollover_gap'] = rollover_flags
    result_df['gap_size_pct'] = overnight_gaps * 100
    
    if rollover_flags.sum() > 0:
        gap_dates = result_df[rollover_flags].index.strftime('%Y-%m-%d %H:%M').tolist()
        gap_sizes = result_df.loc[rollover_flags, 'gap_size_pct'].round(2).tolist()
        logger.warning("Detected %d rollover gaps in %s on dates: %s (sizes: %s%%)", 
                      rollover_flags.sum(), symbol, gap_dates[:3], gap_sizes[:3])
    
    return result_df


def validate_futures_data(df: pd.DataFrame, symbol: str, config: Optional[Dict] = None) -> Dict[str, Any]:
    """Enhanced validation for futures data with comprehensive quality checks.
    
    Args:
        df: DataFrame with OHLCVI data
        symbol: Futures symbol for logging
        config: Market configuration dict
        
    Returns:
        Dict with validation results and cleaned dataframe
    """
    if df.empty:
        logger.warning("Empty DataFrame for %s", symbol)
        return {
            'valid': False,
            'dataframe': df,
            'issues': ['Empty DataFrame'],
            'quality_score': 0.0
        }
    
    validation_results = {
        'symbol': symbol,
        'total_bars': len(df),
        'issues': [],
        'valid': True
    }
    
    validated_df = df.copy()
    
    # 1. Check for missing OHLCVI columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        validation_results['issues'].append(f"Missing columns: {missing_cols}")
        logger.error("Missing columns for %s: %s", symbol, missing_cols)
        validation_results['valid'] = False
        return validation_results
    
    try:
        # 2. Trading hours validation (Phase 1.1)
        df_filtered = validate_trading_hours(validated_df, symbol, config)
        excluded_bars = len(validated_df) - len(df_filtered)
        
        validation_results['trading_hours_validation'] = {
            'excluded_after_hours_bars': excluded_bars,
            'remaining_bars': len(df_filtered),
            'passed': True
        }
        
        # 3. Rollover gap detection (Phase 1.3)
        df_with_rollover = detect_rollover_gaps(df_filtered, symbol, config)
        rollover_gaps = df_with_rollover['rollover_gap'].sum() if 'rollover_gap' in df_with_rollover.columns else 0
        
        validation_results['rollover_detection'] = {
            'detected_gaps': int(rollover_gaps),
            'gap_dates': df_with_rollover[df_with_rollover.get('rollover_gap', pd.Series(False, index=df_with_rollover.index))].index.strftime('%Y-%m-%d %H:%M').tolist()[:5]
        }
        
        # Use filtered data for remaining validations
        validated_df = df_with_rollover
        
    except Exception as e:
        logger.error("Enhanced validation failed for %s: %s", symbol, e)
        validation_results['issues'].append(f'Enhanced validation error: {str(e)}')
    
    # 4. Check for data gaps (missing time periods) 
    validated_df = _check_time_gaps(validated_df, symbol)
    
    # 5. Enhanced price spike detection (Phase 1.2)
    try:
        spike_passed, spike_issues, validated_df = _check_price_spikes(validated_df, symbol)
        validation_results['spike_detection'] = {
            'passed': spike_passed,
            'spike_count': validated_df['price_spike'].sum() if 'price_spike' in validated_df.columns else 0,
            'issues': spike_issues
        }
        validation_results['issues'].extend(spike_issues)
        
    except Exception as e:
        logger.error("Spike detection failed for %s: %s", symbol, e)
        validation_results['issues'].append(f'Spike detection error: {str(e)}')
    
    # 6. Check OHLC logic consistency
    validated_df = _check_ohlc_consistency(validated_df, symbol)
    
    # 7. Check for zero/negative values
    _check_invalid_prices(validated_df, symbol)
    
    # 8. Volume and Open Interest checks
    _check_volume_patterns(validated_df, symbol)
    validated_df = _check_open_interest_patterns(validated_df, symbol)
    
    # 9. Calculate overall data quality score
    validated_df = _calculate_quality_score(validated_df, symbol)
    
    # Final validation results
    validation_results.update({
        'dataframe': validated_df,
        'final_bars': len(validated_df),
        'quality_score': validated_df.attrs.get('quality_score', 0.0),
        'valid': len(validation_results['issues']) == 0,
        'warnings_count': len([issue for issue in validation_results['issues'] if 'warning' in issue.lower()]),
        'errors_count': len([issue for issue in validation_results['issues'] if 'error' in issue.lower()])
    })
    
    # Log summary
    logger.info("Validation complete for %s: %d/%d bars, quality: %.1f%%, %d issues", 
               symbol, validation_results['final_bars'], validation_results['total_bars'],
               validation_results['quality_score'], len(validation_results['issues']))
    
    return validation_results


def _check_time_gaps(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Check for missing time periods in the data with enhanced gap analysis."""
    if len(df) <= 1:
        return df
    
    try:
        time_diff = df.index.to_series().diff()
        expected_freq = time_diff.mode()[0] if not time_diff.mode().empty else pd.Timedelta('5min')
        
        # Find gaps larger than threshold
        gap_threshold = expected_freq * TIME_GAP_THRESHOLD_MULTIPLIER
        gaps = time_diff[time_diff > gap_threshold]
        
        if len(gaps) > 0:
            gap_durations = gaps / expected_freq
            logger.warning("Found %d time gaps in %s data (avg gap: %.1fx expected frequency)", 
                          len(gaps), symbol, gap_durations.mean())
            
            df = df.copy()
            df['has_gap'] = time_diff > gap_threshold
            df['gap_duration'] = time_diff / expected_freq
        
    except Exception as e:
        logger.warning("Time gap analysis failed for %s: %s", symbol, e)
    
    return df


def _check_price_spikes(df: pd.DataFrame, symbol: str) -> Tuple[bool, List[str], pd.DataFrame]:
    """Enhanced price spike detection with volatility adjustment.
    
    Args:
        df: DataFrame with OHLCV data
        symbol: Symbol name for volatility-specific thresholds
        
    Returns:
        Tuple of (validation_passed, issues_list, enhanced_dataframe)
    """
    issues = []
    result_df = df.copy()
    
    if 'close' not in df.columns or len(df) < 20:  # Need minimum data for volatility
        return True, issues, result_df
    
    # Calculate returns for spike detection
    returns = df['close'].pct_change().dropna()
    
    # Dynamic threshold based on symbol volatility characteristics
    volatility_multipliers = {
        'NIFTY': 2.5,        # Less volatile, tighter threshold
        'BANKNIFTY': 3.0,    # More volatile, looser threshold  
        'FINNIFTY': 2.8,     # Moderate volatility
        'MIDCPNIFTY': 3.2,   # Most volatile midcap
        'CNXPHARMA': 3.1,    # Sector volatility
        'CNXIT': 2.9         # Tech sector volatility
    }
    
    # Extract base symbol name and get multiplier
    base_symbol = symbol.upper()
    vol_multiplier = 3.0  # Default multiplier
    
    for known_symbol in volatility_multipliers:
        if known_symbol in base_symbol:
            vol_multiplier = volatility_multipliers[known_symbol]
            break
    
    # Calculate rolling volatility (20-day window)
    rolling_std = returns.rolling(20, min_periods=10).std()
    
    # Dynamic spike threshold = vol_multiplier * rolling_volatility
    spike_threshold = vol_multiplier * rolling_std
    
    # Detect spikes (absolute returns vs threshold)
    spike_mask = abs(returns) > spike_threshold
    spike_count = spike_mask.sum()
    
    if spike_count > 0:
        spike_dates = returns[spike_mask].index.strftime('%Y-%m-%d %H:%M').tolist()
        spike_sizes = (returns[spike_mask] * 100).round(2).tolist()
        
        issues.append(f"Found {spike_count} price spikes in {symbol}")
        logger.warning("Price spikes in %s: %s (sizes: %s%%) using %.1fσ threshold", 
                      symbol, spike_dates[:3], spike_sizes[:3], vol_multiplier)
        
        # Mark spikes in dataframe for potential cleaning
        result_df['price_spike'] = spike_mask.reindex(df.index, fill_value=False)
        result_df['spike_threshold'] = spike_threshold.reindex(df.index)
    
    # Consider validation failed if >5% of data points are spikes
    spike_ratio = spike_count / len(returns) if len(returns) > 0 else 0
    validation_passed = spike_ratio <= 0.05
    
    if not validation_passed:
        issues.append(f"High spike ratio: {spike_ratio:.1%} exceeds 5% threshold")
    
    return validation_passed, issues, result_df


def _check_ohlc_consistency(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Check OHLC data for logical consistency."""
    ohlc_issues = (
        (df['high'] < df['low']) |  # High < Low
        (df['high'] < df['open']) | (df['high'] < df['close']) |  # High < Open/Close
        (df['low'] > df['open']) | (df['low'] > df['close'])      # Low > Open/Close
    )
    
    if ohlc_issues.sum() > 0:
        logger.error("Found %d OHLC logic errors in %s data", ohlc_issues.sum(), symbol)
        df['ohlc_error'] = ohlc_issues
    
    return df


def _check_invalid_prices(df: pd.DataFrame, symbol: str) -> None:
    """Check for zero/negative price values."""
    for col in PRICE_COLUMNS:
        if col in df.columns:
            invalid_prices = (df[col] <= 0) | df[col].isna()
            if invalid_prices.sum() > 0:
                logger.warning("Found %d invalid %s prices in %s", 
                             invalid_prices.sum(), col, symbol)


def _check_volume_patterns(df: pd.DataFrame, symbol: str) -> None:
    """Check volume data for unusual patterns."""
    if 'volume' in df.columns:
        zero_volume = df['volume'] == 0
        if zero_volume.sum() > len(df) * ZERO_VOLUME_THRESHOLD:
            logger.warning("%s has %d%% zero volume bars", 
                         symbol, (zero_volume.sum() / len(df)) * 100)


def _check_open_interest_patterns(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Check open interest data for rollover patterns."""
    if 'open_interest' in df.columns and len(df) > 1:
        oi_change = df['open_interest'].pct_change().abs()
        large_oi_changes = oi_change > OI_CHANGE_THRESHOLD
        if large_oi_changes.sum() > 0:
            logger.info("Found %d large OI changes in %s (potential rollovers)", 
                      large_oi_changes.sum(), symbol)
            df['oi_rollover'] = large_oi_changes
    
    return df


def _calculate_quality_score(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Calculate overall data quality score."""
    quality_issues = 0
    
    if 'has_gap' in df.columns:
        quality_issues += df['has_gap'].sum()
    if 'price_spike' in df.columns:
        quality_issues += df['price_spike'].sum()
    if 'ohlc_error' in df.columns:
        quality_issues += df['ohlc_error'].sum()
    
    quality_score = max(0, 100 - (quality_issues / len(df)) * 100)
    df.attrs['quality_score'] = quality_score
    
    logger.info("Data quality score for %s: %.1f%%", symbol, quality_score)
    
    return df


def validate_dataframe_structure(df: pd.DataFrame, required_columns: Optional[list] = None) -> bool:
    """Validate that DataFrame has required structure for processing.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        True if valid, False otherwise
    """
    if df.empty:
        logger.error("DataFrame is empty")
        return False
    
    if required_columns is None:
        required_columns = ['close']
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        logger.error("Missing required columns: %s", missing_cols)
        return False
    
    # Check for datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.warning("DataFrame index is not DatetimeIndex")
    
    return True


def clean_ohlcv_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize OHLCV data.
    
    Args:
        df: Raw OHLCV DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    if df.empty:
        return df
    
    cleaned_df = df.copy()
    
    # Remove rows with all NaN values
    cleaned_df = cleaned_df.dropna(how='all')
    
    # Forward fill missing values for price columns
    price_cols = [col for col in PRICE_COLUMNS if col in cleaned_df.columns]
    cleaned_df[price_cols] = cleaned_df[price_cols].ffill()
    
    # Fill zero volumes with small positive number to avoid division errors
    if 'volume' in cleaned_df.columns:
        cleaned_df['volume'] = cleaned_df['volume'].replace(0, 1)
    
    # Ensure positive prices
    for col in price_cols:
        if col in cleaned_df.columns:
            cleaned_df.loc[cleaned_df[col] <= 0, col] = np.nan
            cleaned_df[col] = cleaned_df[col].ffill()
    
    return cleaned_df


def forward_fill_missing_bars(df: pd.DataFrame, symbol: str, max_fill_periods: int = 3) -> pd.DataFrame:
    """Forward-fill missing bars with volume=0 and limits to prevent stale data.
    
    Args:
        df: DataFrame with OHLCV data and potential gaps
        symbol: Symbol name for logging
        max_fill_periods: Maximum consecutive periods to forward-fill
        
    Returns:
        DataFrame with missing bars filled (limited forward-fill)
    """
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return df
    
    original_length = len(df)
    
    try:
        # Identify expected frequency
        freq = pd.infer_freq(df.index)
        if freq is None:
            # Fallback: use most common time difference
            time_diffs = df.index.to_series().diff().dropna()
            freq = time_diffs.mode()[0] if not time_diffs.empty else pd.Timedelta('5min')
        
        # Create complete time range
        full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
        
        # Reindex to full range with forward-fill
        df_filled = df.reindex(full_range)
        
        # Forward-fill prices (OHLC get same value as previous close)
        price_cols = [col for col in ['open', 'high', 'low', 'close'] if col in df_filled.columns]
        df_filled[price_cols] = df_filled[price_cols].ffill(limit=max_fill_periods)
        
        # Set volume=0 for filled bars
        if 'volume' in df_filled.columns:
            volume_filled = df_filled['volume'].ffill(limit=max_fill_periods).fillna(0)
            # Mark filled volume as 0
            df_filled.loc[df_filled['volume'].isna(), 'volume'] = 0
            df_filled['volume'] = volume_filled
        
        # Mark filled bars
        df_filled['is_filled'] = df_filled.index.isin(df.index) == False
        
        filled_count = len(df_filled) - original_length
        if filled_count > 0:
            logger.info("Forward-filled %d missing bars for %s (limit: %d periods)", 
                       filled_count, symbol, max_fill_periods)
        
        return df_filled
        
    except Exception as e:
        logger.error("Forward-fill failed for %s: %s", symbol, e)
        return df


def validate_rollover_continuity(current_df: pd.DataFrame, next_df: pd.DataFrame, 
                               rollover_date: str, symbol: str) -> Dict[str, Any]:
    """Validate price continuity during contract rollover transitions.
    
    Args:
        current_df: Current contract data
        next_df: Next contract data  
        rollover_date: Date of rollover
        symbol: Symbol name for logging
        
    Returns:
        Dict with continuity analysis results
    """
    try:
        rollover_dt = pd.to_datetime(rollover_date)
        
        # Get prices around rollover date
        current_before = current_df[current_df.index <= rollover_dt]['close'].iloc[-5:]  # Last 5 bars
        next_after = next_df[next_df.index >= rollover_dt]['close'].iloc[:5]  # First 5 bars
        
        if len(current_before) == 0 or len(next_after) == 0:
            return {'error': 'Insufficient data around rollover date'}
        
        # Calculate price gap
        price_gap = (next_after.iloc[0] - current_before.iloc[-1]) / current_before.iloc[-1]
        
        # Calculate volatilities
        current_vol = current_before.pct_change().std() * np.sqrt(252)
        next_vol = next_after.pct_change().std() * np.sqrt(252)
        
        # Continuity score (lower gap = better continuity)
        continuity_score = max(0, 100 - abs(price_gap) * 1000)  # Gap in bps
        
        result = {
            'rollover_date': rollover_date,
            'price_gap_pct': price_gap * 100,
            'price_gap_bps': price_gap * 10000,
            'current_price': current_before.iloc[-1],
            'next_price': next_after.iloc[0],
            'current_volatility': current_vol,
            'next_volatility': next_vol,
            'continuity_score': continuity_score,
            'needs_adjustment': abs(price_gap) > 0.005  # >50bps gap
        }
        
        logger.info("Rollover continuity for %s: %.1f bps gap, score: %.1f", 
                   symbol, result['price_gap_bps'], continuity_score)
        
        return result
        
    except Exception as e:
        logger.error("Rollover continuity check failed for %s: %s", symbol, e)
        return {'error': str(e)}