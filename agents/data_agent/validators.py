"""
Data Validation Module

Provides comprehensive data quality checks and validation for:
- Futures data integrity
- OHLCV data consistency  
- Price spikes and gaps detection
- Quality scoring
"""

import logging
from typing import Optional

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


def validate_futures_data(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Validate futures data for gaps, price spikes, and data quality issues.
    
    Args:
        df: DataFrame with OHLCVI data
        symbol: Futures symbol for logging
        
    Returns:
        Validated DataFrame with quality flags
    """
    if df.empty:
        logger.warning("Empty DataFrame for %s", symbol)
        return df
    
    validated_df = df.copy()
    
    # 1. Check for missing OHLCVI columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error("Missing columns for %s: %s", symbol, missing_cols)
        return pd.DataFrame()
    
    # 2. Check for data gaps (missing time periods)
    validated_df = _check_time_gaps(validated_df, symbol)
    
    # 3. Check for price spikes (outliers)
    validated_df = _check_price_spikes(validated_df, symbol)
    
    # 4. Check OHLC logic consistency
    validated_df = _check_ohlc_consistency(validated_df, symbol)
    
    # 5. Check for zero/negative values
    _check_invalid_prices(df, symbol)
    
    # 6. Volume and Open Interest checks
    _check_volume_patterns(df, symbol)
    _check_open_interest_patterns(validated_df, symbol)
    
    # 7. Add data quality score
    validated_df = _calculate_quality_score(validated_df, symbol)
    
    return validated_df


def _check_time_gaps(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Check for missing time periods in the data."""
    if len(df) > 1:
        time_diff = df.index.to_series().diff()
        expected_freq = time_diff.mode()[0] if not time_diff.mode().empty else pd.Timedelta('1D')
        
        # Find gaps larger than 2x expected frequency
        gaps = time_diff[time_diff > expected_freq * TIME_GAP_THRESHOLD_MULTIPLIER]
        if len(gaps) > 0:
            logger.warning("Found %d time gaps in %s data", len(gaps), symbol)
            df['has_gap'] = time_diff > expected_freq * TIME_GAP_THRESHOLD_MULTIPLIER
    
    return df


def _check_price_spikes(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Check for abnormal price spikes."""
    if 'close' in df.columns and len(df) > 10:
        returns = df['close'].pct_change().abs()
        spike_threshold = returns.quantile(0.99) * PRICE_SPIKE_THRESHOLD_MULTIPLIER
        
        price_spikes = returns > spike_threshold
        if price_spikes.sum() > 0:
            logger.warning("Found %d price spikes in %s data", price_spikes.sum(), symbol)
            df['price_spike'] = price_spikes
    
    return df


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