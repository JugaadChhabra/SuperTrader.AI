"""
Feature Building Module

Handles feature engineering and technical analysis including:
- Price-based features (returns, momentum)
- Volatility features 
- Volume/Open Interest features
- Time-based features
- Market regime features
- Cross-contract features
"""

import logging
from typing import Dict, Optional

import pandas as pd
import numpy as np

from .constants import (
    TRADING_DAYS_PER_YEAR,
    ROLLING_PERIODS,
    VOLATILITY_ANNUALIZATION_FACTOR
)

logger = logging.getLogger(__name__)


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add price-based features to DataFrame.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with price features added
    """
    if 'close' not in df.columns:
        logger.warning("No close column found for price features")
        return df
    
    result_df = df.copy()
    
    # Returns at different horizons
    result_df['return_1d'] = result_df['close'].pct_change()
    result_df['return_5d'] = result_df['close'].pct_change(5)
    result_df['return_10d'] = result_df['close'].pct_change(10)
    
    # Log returns (more stable for modeling)
    result_df['log_return_1d'] = np.log(result_df['close'] / result_df['close'].shift(1))
    
    # Price momentum features
    result_df['price_momentum_5'] = result_df['close'] / result_df['close'].shift(5) - 1
    result_df['price_momentum_10'] = result_df['close'] / result_df['close'].shift(10) - 1
    
    logger.debug("Added price-based features")
    return result_df


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add volatility-based features to DataFrame.
    
    Args:
        df: DataFrame with price data
        
    Returns:
        DataFrame with volatility features added
    """
    if 'close' not in df.columns or len(df) < 20:
        logger.warning("Insufficient data for volatility features")
        return df
    
    result_df = df.copy()
    returns = result_df['close'].pct_change()
    
    # Realized volatility at different horizons
    result_df['realized_vol_5'] = returns.rolling(5).std() * np.sqrt(VOLATILITY_ANNUALIZATION_FACTOR)
    result_df['realized_vol_20'] = returns.rolling(20).std() * np.sqrt(VOLATILITY_ANNUALIZATION_FACTOR)
    result_df['vol_ratio'] = result_df['realized_vol_5'] / result_df['realized_vol_20']
    
    # Parkinson volatility (using high-low range)
    if all(col in result_df.columns for col in ['high', 'low', 'open']):
        hl_ratio = np.log(result_df['high'] / result_df['low'])
        co_ratio = np.log(result_df['close'] / result_df['open'])
        parkinson_vol = np.sqrt((hl_ratio ** 2 / 4) - (2 * np.log(2) - 1) * (co_ratio ** 2))
        result_df['parkinson_vol'] = parkinson_vol.rolling(20).mean()
    
    logger.debug("Added volatility features")
    return result_df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add volume and open interest features.
    
    Args:
        df: DataFrame with volume/OI data
        
    Returns:
        DataFrame with volume features added
    """
    result_df = df.copy()
    
    # Volume features
    if 'volume' in df.columns:
        result_df['volume_ma_ratio'] = result_df['volume'] / result_df['volume'].rolling(20).mean()
        result_df['volume_momentum'] = result_df['volume'].pct_change(5)
        
        # Volume surge indicator
        if len(df) >= 20:
            volume = result_df['volume']
            volume_mean = volume.rolling(20).mean()
            volume_std = volume.rolling(20).std()
            result_df['volume_surge'] = ((volume - volume_mean) / volume_std).fillna(0)
    
    # Open Interest features
    if 'open_interest' in df.columns:
        result_df['oi_ma_ratio'] = result_df['open_interest'] / result_df['open_interest'].rolling(20).mean()
        result_df['oi_momentum'] = result_df['open_interest'].pct_change(5)
        
        # Volume-OI divergence (if both available)
        if 'volume' in df.columns:
            vol_norm = (result_df['volume'] - result_df['volume'].rolling(20).mean()) / result_df['volume'].rolling(20).std()
            oi_norm = (result_df['open_interest'] - result_df['open_interest'].rolling(20).mean()) / result_df['open_interest'].rolling(20).std()
            result_df['vol_oi_divergence'] = abs(vol_norm - oi_norm)
            result_df['vol_oi_signal'] = result_df['vol_oi_divergence'] > 1.5
    
    logger.debug("Added volume/OI features")
    return result_df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features to DataFrame.
    
    Args:
        df: DataFrame with datetime index
        
    Returns:
        DataFrame with time features added
    """
    result_df = df.copy()
    
    # Basic time features
    result_df['hour'] = result_df.index.hour
    result_df['day_of_week'] = result_df.index.dayofweek
    result_df['month'] = result_df.index.month
    result_df['is_month_end'] = (result_df.index + pd.DateOffset(days=1)).month != result_df.index.month
    
    # Market session indicators (assuming IST market hours)
    result_df['is_opening_hour'] = result_df['hour'] == 9
    result_df['is_closing_hour'] = result_df['hour'] == 15
    result_df['is_lunch_time'] = result_df['hour'] == 12
    
    # Weekend proximity
    result_df['days_to_weekend'] = 4 - result_df['day_of_week']  # Friday = 4
    result_df['is_friday'] = result_df['day_of_week'] == 4
    result_df['is_monday'] = result_df['day_of_week'] == 0
    
    logger.debug("Added time-based features")
    return result_df


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add market regime features using TA-Lib indicators.
    
    Args:
        df: DataFrame with price data and technical indicators
        
    Returns:
        DataFrame with regime features added
    """
    if len(df) < 50:
        logger.warning("Insufficient data for regime features")
        return df
    
    result_df = df.copy()
    
    # Trend regime (price vs long-term MA)
    if 'sma_50' in df.columns:
        result_df['trend_regime'] = (result_df['close'] > result_df['sma_50']).astype(int)
    elif 'close' in df.columns:
        # Fallback: create simple moving average
        result_df['sma_50'] = result_df['close'].rolling(50).mean()
        result_df['trend_regime'] = (result_df['close'] > result_df['sma_50']).astype(int)
    
    # ADX trend strength regime
    if 'adx' in df.columns:
        result_df['strong_trend_regime'] = (result_df['adx'] > 25).astype(int)
        
    # Volatility regime (current vol vs historical)
    if 'realized_vol_20' in df.columns:
        vol_median = result_df['realized_vol_20'].rolling(100).median()
        result_df['vol_regime'] = (result_df['realized_vol_20'] > vol_median).astype(int)
    
    # ATR-based volatility regime
    if 'atr_14' in df.columns:
        atr_ma = result_df['atr_14'].rolling(50).mean()
        result_df['high_vol_atr_regime'] = (result_df['atr_14'] > atr_ma * 1.5).astype(int)
    
    # RSI momentum regime
    if 'rsi_14' in df.columns:
        result_df['bullish_momentum'] = (result_df['rsi_14'] > 50).astype(int)
        result_df['extreme_oversold'] = (result_df['rsi_14'] < 20).astype(int)
        result_df['extreme_overbought'] = (result_df['rsi_14'] > 80).astype(int)
    
    # Bollinger Bands regime
    if all(col in df.columns for col in ['bb_upper', 'bb_lower', 'close']):
        result_df['bb_squeeze'] = ((result_df['bb_upper'] - result_df['bb_lower']) / result_df['close']) < 0.1
        result_df['bb_breakout_up'] = result_df['close'] > result_df['bb_upper']
        result_df['bb_breakout_down'] = result_df['close'] < result_df['bb_lower']
    
    logger.debug("Added regime features")
    return result_df


def add_cross_contract_features(primary_df: pd.DataFrame, futures_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Add cross-contract features (calendar spreads, etc.).
    
    Args:
        primary_df: Primary contract DataFrame
        futures_data: Dict of all contract DataFrames
        
    Returns:
        DataFrame with cross-contract features added
    """
    if len(futures_data) < 2:
        logger.debug("Insufficient contracts for cross-contract features")
        return primary_df
    
    result_df = primary_df.copy()
    contract_names = list(futures_data.keys())
    
    # Calendar spread (current - next month)
    if len(contract_names) >= 2:
        next_contract_data = futures_data[contract_names[1]]
        if not next_contract_data.empty and 'close' in next_contract_data.columns:
            # Align timeframes
            aligned_next = next_contract_data.reindex(result_df.index, method='ffill')
            result_df['calendar_spread'] = result_df['close'] - aligned_next['close']
            result_df['calendar_spread_pct'] = (result_df['calendar_spread'] / result_df['close']) * 100
            
            # Calendar spread momentum
            result_df['calendar_spread_momentum'] = result_df['calendar_spread'].pct_change(5)
            
    # Inter-contract volatility spread
    if len(contract_names) >= 2:
        primary_vol = result_df['close'].pct_change().rolling(20).std()
        next_vol = aligned_next['close'].pct_change().rolling(20).std() if 'aligned_next' in locals() else None
        
        if next_vol is not None:
            result_df['vol_spread'] = primary_vol - next_vol
            result_df['vol_spread_ratio'] = primary_vol / next_vol.replace(0, np.nan)
    
    logger.debug("Added cross-contract features")
    return result_df


def add_futures_specific_features(df: pd.DataFrame, spot_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Add futures-specific features including basis calculations.
    
    Args:
        df: Futures DataFrame
        spot_df: Optional spot index DataFrame
        
    Returns:
        DataFrame with futures-specific features
    """
    result_df = df.copy()
    
    # Basis indicators (if spot data provided)
    if spot_df is not None and 'close' in spot_df.columns:
        from .futures_manager import compute_basis
        
        # Align timeframes
        aligned_spot = spot_df.reindex(df.index, method='ffill')
        
        if not aligned_spot['close'].isna().all():
            # Calculate days to expiry (simplified - should be calculated properly)
            days_to_expiry = pd.Series(30, index=df.index)  # Placeholder
            
            basis_metrics = compute_basis(
                df['close'], 
                aligned_spot['close'], 
                days_to_expiry
            )
            
            for metric_name, metric_series in basis_metrics.items():
                result_df[f'basis_{metric_name}'] = metric_series
            
            # Basis momentum and mean reversion
            if 'basis_basis_pct' in result_df.columns:
                result_df['basis_momentum'] = result_df['basis_basis_pct'].rolling(5).mean()
                result_df['basis_mean_reversion'] = (
                    result_df['basis_basis_pct'] - result_df['basis_basis_pct'].rolling(20).mean()
                ) / result_df['basis_basis_pct'].rolling(20).std()
    
    # Contango/Backwardation indicators
    if 'calendar_spread' in result_df.columns:
        result_df['is_contango'] = (result_df['calendar_spread'] > 0).astype(int)
        result_df['is_backwardation'] = (result_df['calendar_spread'] < 0).astype(int)
    
    logger.debug("Added futures-specific features")
    return result_df


def build_comprehensive_features(df: pd.DataFrame, 
                               futures_data: Optional[Dict[str, pd.DataFrame]] = None,
                               spot_data: Optional[pd.DataFrame] = None,
                               minimal_mode: bool = False) -> pd.DataFrame:
    """Build comprehensive feature set by combining all feature types.
    
    Args:
        df: Primary DataFrame
        futures_data: Dict of futures contract data
        spot_data: Spot index data
        minimal_mode: If True, only add essential features
        
    Returns:
        DataFrame with comprehensive feature set
    """
    logger.info("Building comprehensive feature set (minimal_mode=%s)", minimal_mode)
    
    result_df = df.copy()
    
    # Always add basic price features
    result_df = add_price_features(result_df)
    result_df = add_time_features(result_df)
    
    if not minimal_mode:
        # Add advanced features only in full mode
        result_df = add_volatility_features(result_df)
        result_df = add_volume_features(result_df)
        result_df = add_regime_features(result_df)
        
        # Add cross-contract features if multiple contracts available
        if futures_data and len(futures_data) > 1:
            result_df = add_cross_contract_features(result_df, futures_data)
        
        # Add futures-specific features
        result_df = add_futures_specific_features(result_df, spot_data)
    else:
        # Minimal mode: only essential volume feature
        if 'volume' in result_df.columns and len(result_df) >= 20:
            volume = result_df['volume']
            volume_mean = volume.rolling(20).mean()
            volume_std = volume.rolling(20).std()
            result_df['volume_surge'] = ((volume - volume_mean) / volume_std).fillna(0)
    
    # Forward-fill any remaining NaN values for stability
    numeric_cols = result_df.select_dtypes(include=[np.number]).columns
    result_df[numeric_cols] = result_df[numeric_cols].ffill()
    
    # Count features added
    original_cols = set(['open', 'high', 'low', 'close', 'volume', 'open_interest'])
    feature_count = len([col for col in result_df.columns if col not in original_cols])
    
    logger.info("Built %d features for %d rows", feature_count, len(result_df))
    
    return result_df