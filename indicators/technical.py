"""
Technical Indicators for SuperTrader.AI using TA-Lib
Clean, standalone functions for technical analysis of futures data.

This module uses TA-Lib's optimized C functions for maximum performance.
All functions are independent and can be used separately.

Functions:
- macd_multi_timeframe(df): Multi-timeframe MACD analysis
- rsi_multi_period(df): RSI with multiple periods
- atr_volatility(df): Average True Range volatility
- bollinger_bands(df): Bollinger Bands analysis  
- volume_indicators(df): Volume-based indicators
- momentum_oscillators(df): Stochastic, Williams %R, CCI
- trend_indicators(df): Moving averages, SAR, ADX
- compute_all_indicators(df): Complete indicator suite

Usage:
    from indicators.technical import macd_multi_timeframe, rsi_multi_period
    
    df_with_macd = macd_multi_timeframe(df)
    df_with_rsi = rsi_multi_period(df)
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

# Try to import TA-Lib
try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False
    print("[WARNING] TA-Lib not available. Install with: pip install TA-Lib")

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# MACD INDICATORS
# =============================================================================

def macd_multi_timeframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate MACD for multiple timeframes using TA-Lib.
    
    Timeframes:
    - Fast: 8/21/9 for short-term signals
    - Standard: 12/26/9 for medium-term signals  
    - Slow: 19/39/9 for long-term signals
    
    Args:
        df: DataFrame with 'close' column
        
    Returns:
        DataFrame with MACD indicators added
    """
    result_df = df.copy()
    
    if 'close' not in df.columns:
        logger.warning("Missing 'close' column for MACD calculation")
        return result_df
    
    if not HAS_TALIB:
        logger.warning("TA-Lib not available for MACD calculation")
        return result_df
    
    try:
        close_prices = df['close'].astype(np.float64).values
        
        # MACD parameter sets optimized for futures trading
        timeframes = [
            ('fast', 8, 21, 9),
            ('standard', 12, 26, 9), 
            ('slow', 19, 39, 9)
        ]
        
        for name, fast, slow, signal in timeframes:
            # TA-Lib MACD calculation
            macd, macd_signal, macd_hist = talib.MACD(
                close_prices, 
                fastperiod=fast, 
                slowperiod=slow, 
                signalperiod=signal
            )
            
            # Core MACD indicators
            result_df[f'macd_{name}'] = macd
            result_df[f'macd_{name}_signal'] = macd_signal
            result_df[f'macd_{name}_histogram'] = macd_hist
            
            # MACD signals
            macd_s = pd.Series(macd, index=df.index)
            signal_s = pd.Series(macd_signal, index=df.index)
            
            result_df[f'macd_{name}_bullish'] = (
                (macd_s > signal_s) & (macd_s.shift(1) <= signal_s.shift(1))
            )
            result_df[f'macd_{name}_bearish'] = (
                (macd_s < signal_s) & (macd_s.shift(1) >= signal_s.shift(1))
            )
        
        logger.info("Calculated TA-Lib MACD for %d rows", len(result_df))
        
    except Exception as exc:
        logger.error("Error calculating MACD: %s", exc)
        
    return result_df


# =============================================================================
# RSI INDICATORS
# =============================================================================

def rsi_multi_period(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate RSI for multiple periods using TA-Lib RSI function.
    
    Periods: 9, 14, 21, 30 for different momentum timeframes
    
    Args:
        df: DataFrame with 'close' column
        
    Returns:
        DataFrame with RSI indicators added
    """
    result_df = df.copy()
    
    if 'close' not in df.columns:
        logger.warning("Missing 'close' column for RSI calculation")
        return result_df
    
    if not HAS_TALIB:
        logger.warning("TA-Lib not available for RSI calculation")
        return result_df
    
    try:
        close_prices = df['close'].astype(np.float64).values
        
        # RSI periods optimized for futures
        periods = [9, 14, 21, 30]
        
        for period in periods:
            # TA-Lib RSI calculation
            rsi = talib.RSI(close_prices, timeperiod=period)
            result_df[f'rsi_{period}'] = rsi
            
            # RSI levels
            result_df[f'rsi_{period}_overbought'] = rsi > 70
            result_df[f'rsi_{period}_oversold'] = rsi < 30
            result_df[f'rsi_{period}_extreme_ob'] = rsi > 80
            result_df[f'rsi_{period}_extreme_os'] = rsi < 20
        
        # RSI momentum
        result_df['rsi_momentum_14_9'] = result_df['rsi_14'] - result_df['rsi_9']
        result_df['rsi_momentum_30_14'] = result_df['rsi_30'] - result_df['rsi_14']
        
        logger.info("Calculated TA-Lib RSI for %d rows", len(result_df))
        
    except Exception as exc:
        logger.error("Error calculating RSI: %s", exc)
        
    return result_df


# =============================================================================
# VOLATILITY INDICATORS
# =============================================================================

def atr_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate ATR volatility indicators using TA-Lib ATR and TRANGE functions.
    
    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        
    Returns:
        DataFrame with ATR indicators added
    """
    result_df = df.copy()
    
    required_cols = ['high', 'low', 'close']
    if not all(col in df.columns for col in required_cols):
        logger.warning("Missing required OHLC columns for ATR: %s", required_cols)
        return result_df
    
    if not HAS_TALIB:
        logger.warning("TA-Lib not available for ATR calculation")
        return result_df
    
    try:
        high = df['high'].astype(np.float64).values
        low = df['low'].astype(np.float64).values
        close = df['close'].astype(np.float64).values
        
        # TA-Lib ATR for multiple periods
        periods = [7, 14, 21, 30]
        
        for period in periods:
            atr = talib.ATR(high, low, close, timeperiod=period)
            result_df[f'atr_{period}'] = atr
            result_df[f'atr_pct_{period}'] = (atr / close) * 100
        
        # TA-Lib True Range
        true_range = talib.TRANGE(high, low, close)
        result_df['true_range'] = true_range
        
        # Volatility regime (using ATR_14)
        if len(df) >= 50:
            atr_14 = result_df['atr_14']
            atr_ma = atr_14.rolling(50, min_periods=25).mean()
            
            result_df['high_vol_regime'] = atr_14 > atr_ma * 1.5
            result_df['low_vol_regime'] = atr_14 < atr_ma * 0.6
        
        # Position sizing (1% and 2% risk)
        result_df['position_size_1pct'] = (close * 0.01) / result_df['atr_14']
        result_df['position_size_2pct'] = (close * 0.02) / result_df['atr_14']
        
        logger.info("Calculated TA-Lib ATR for %d rows", len(result_df))
        
    except Exception as exc:
        logger.error("Error calculating ATR: %s", exc)
        
    return result_df


# =============================================================================
# BOLLINGER BANDS
# =============================================================================

def bollinger_bands(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Bollinger Bands using TA-Lib BBANDS function.
    
    Args:
        df: DataFrame with 'close' column
        
    Returns:
        DataFrame with Bollinger Bands indicators added
    """
    result_df = df.copy()
    
    if 'close' not in df.columns or not HAS_TALIB:
        return result_df
    
    try:
        close_prices = df['close'].astype(np.float64).values
        
        # TA-Lib Bollinger Bands (20 period, 2 standard deviations)
        bb_upper, bb_middle, bb_lower = talib.BBANDS(
            close_prices, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
        )
        
        result_df['bb_upper'] = bb_upper
        result_df['bb_middle'] = bb_middle  
        result_df['bb_lower'] = bb_lower
        result_df['bb_width'] = (bb_upper - bb_lower) / bb_middle
        result_df['bb_position'] = (close_prices - bb_lower) / (bb_upper - bb_lower)
        
        # Bollinger Band signals
        result_df['bb_upper_breach'] = close_prices > bb_upper
        result_df['bb_lower_breach'] = close_prices < bb_lower
        
        logger.info("Calculated TA-Lib Bollinger Bands for %d rows", len(result_df))
        
    except Exception as exc:
        logger.error("Error calculating Bollinger Bands: %s", exc)
        
    return result_df


# =============================================================================
# VOLUME INDICATORS
# =============================================================================

def volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate volume indicators using TA-Lib OBV, AD, SMA, and ROC functions.
    
    Args:
        df: DataFrame with 'close' and 'volume' columns
        
    Returns:
        DataFrame with volume indicators added
    """
    result_df = df.copy()
    
    required_cols = ['close', 'volume']
    if not all(col in df.columns for col in required_cols) or not HAS_TALIB:
        return result_df
    
    try:
        close = df['close'].astype(np.float64).values
        volume = df['volume'].astype(np.float64).values
        
        # TA-Lib volume indicators
        result_df['obv'] = talib.OBV(close, volume)  # On Balance Volume
        result_df['volume_roc_5'] = talib.ROC(volume, timeperiod=5)  # Volume ROC
        result_df['volume_roc_10'] = talib.ROC(volume, timeperiod=10)
        
        # Volume moving averages using TA-Lib SMA
        for period in [10, 20]:
            vol_sma = talib.SMA(volume, timeperiod=period)
            result_df[f'volume_sma_{period}'] = vol_sma
            result_df[f'volume_ratio_{period}'] = volume / vol_sma
        
        # Accumulation/Distribution Line (if OHLC available)
        if all(col in df.columns for col in ['high', 'low']):
            high = df['high'].astype(np.float64).values
            low = df['low'].astype(np.float64).values
            result_df['ad_line'] = talib.AD(high, low, close, volume)
        
        # Volume surge detection
        if len(df) >= 20:
            vol_series = pd.Series(volume, index=df.index)
            vol_mean = vol_series.rolling(20).mean()
            vol_std = vol_series.rolling(20).std()
            zscore = (vol_series - vol_mean) / vol_std
            
            result_df['volume_zscore'] = zscore
            result_df['volume_surge'] = zscore > 2.0
        
        logger.info("Calculated TA-Lib volume indicators for %d rows", len(result_df))
        
    except Exception as exc:
        logger.error("Error calculating volume indicators: %s", exc)
        
    return result_df


# =============================================================================
# MOMENTUM OSCILLATORS
# =============================================================================

def momentum_oscillators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate momentum oscillators using TA-Lib STOCH, WILLR, CCI, ROC, and MOM.
    
    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        
    Returns:
        DataFrame with momentum oscillators added
    """
    result_df = df.copy()
    
    required_cols = ['high', 'low', 'close']
    if not all(col in df.columns for col in required_cols) or not HAS_TALIB:
        return result_df
    
    try:
        high = df['high'].astype(np.float64).values
        low = df['low'].astype(np.float64).values  
        close = df['close'].astype(np.float64).values
        
        # TA-Lib momentum oscillators
        
        # Stochastic Oscillator
        slowk, slowd = talib.STOCH(high, low, close, 
                                 fastk_period=14, slowk_period=3, 
                                 slowk_matype=0, slowd_period=3, slowd_matype=0)
        result_df['stoch_k'] = slowk
        result_df['stoch_d'] = slowd
        result_df['stoch_overbought'] = slowk > 80
        result_df['stoch_oversold'] = slowk < 20
        
        # Williams %R
        result_df['williams_r'] = talib.WILLR(high, low, close, timeperiod=14)
        
        # Commodity Channel Index
        cci = talib.CCI(high, low, close, timeperiod=14)
        result_df['cci'] = cci
        result_df['cci_overbought'] = cci > 100
        result_df['cci_oversold'] = cci < -100
        
        # Rate of Change and Momentum
        result_df['roc_10'] = talib.ROC(close, timeperiod=10)
        result_df['momentum_10'] = talib.MOM(close, timeperiod=10)
        
        logger.info("Calculated TA-Lib momentum oscillators for %d rows", len(result_df))
        
    except Exception as exc:
        logger.error("Error calculating momentum oscillators: %s", exc)
        
    return result_df


# =============================================================================
# TREND INDICATORS
# =============================================================================

def trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate trend indicators using TA-Lib SMA, EMA, SAR, and ADX functions.
    
    Args:
        df: DataFrame with OHLC data
        
    Returns:
        DataFrame with trend indicators added
    """
    result_df = df.copy()
    
    if 'close' not in df.columns or not HAS_TALIB:
        return result_df
    
    try:
        close = df['close'].astype(np.float64).values
        
        # TA-Lib Simple Moving Averages
        sma_periods = [10, 20, 50, 100, 200]
        for period in sma_periods:
            if len(df) >= period:
                sma = talib.SMA(close, timeperiod=period)
                result_df[f'sma_{period}'] = sma
                result_df[f'price_above_sma_{period}'] = close > sma
        
        # TA-Lib Exponential Moving Averages  
        ema_periods = [12, 26, 50]
        for period in ema_periods:
            if len(df) >= period:
                ema = talib.EMA(close, timeperiod=period)
                result_df[f'ema_{period}'] = ema
        
        # Moving Average crossovers
        if len(df) >= 50:
            result_df['sma_20_above_sma_50'] = result_df['sma_20'] > result_df['sma_50']
            result_df['ema_12_above_ema_26'] = result_df['ema_12'] > result_df['ema_26']
        
        # TA-Lib Parabolic SAR and ADX (if OHLC available)
        if all(col in df.columns for col in ['high', 'low']):
            high = df['high'].astype(np.float64).values
            low = df['low'].astype(np.float64).values
            
            # Parabolic SAR
            result_df['sar'] = talib.SAR(high, low, acceleration=0.02, maximum=0.2)
            
            # Average Directional Index
            adx = talib.ADX(high, low, close, timeperiod=14)
            result_df['adx'] = adx
            result_df['adx_strong_trend'] = adx > 25
            
            # Directional Indicators
            result_df['plus_di'] = talib.PLUS_DI(high, low, close, timeperiod=14)
            result_df['minus_di'] = talib.MINUS_DI(high, low, close, timeperiod=14)
        
        logger.info("Calculated TA-Lib trend indicators for %d rows", len(result_df))
        
    except Exception as exc:
        logger.error("Error calculating trend indicators: %s", exc)
        
    return result_df


# =============================================================================
# FUTURES SPECIFIC INDICATORS
# =============================================================================

def futures_specific_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate futures-specific indicators including Open Interest analysis.
    
    Args:
        df: DataFrame with 'open_interest' column (if available)
        
    Returns:
        DataFrame with futures-specific indicators added
    """
    result_df = df.copy()
    
    if not HAS_TALIB:
        return result_df
    
    try:
        # Open Interest analysis (if available)
        if 'open_interest' in df.columns:
            oi_data = df['open_interest'].astype(np.float64).values
            
            # OI moving averages using TA-Lib
            result_df['oi_sma_10'] = talib.SMA(oi_data, timeperiod=10)
            result_df['oi_sma_20'] = talib.SMA(oi_data, timeperiod=20)
            result_df['oi_ema_10'] = talib.EMA(oi_data, timeperiod=10)
            
            # OI momentum using TA-Lib
            result_df['oi_roc_5'] = talib.ROC(oi_data, timeperiod=5)
            result_df['oi_momentum_3'] = talib.MOM(oi_data, timeperiod=3)
            
            # OI trend analysis
            result_df['oi_increasing'] = result_df['oi_sma_10'] > result_df['oi_sma_20']
            result_df['oi_decreasing'] = result_df['oi_sma_10'] < result_df['oi_sma_20']
            
            # Large OI changes (potential rollover signals)
            oi_pct_change = pd.Series(oi_data).pct_change()
            result_df['oi_large_increase'] = oi_pct_change > 0.15  # 15% increase
            result_df['oi_large_decrease'] = oi_pct_change < -0.15  # 15% decrease
            result_df['potential_rollover'] = abs(oi_pct_change) > 0.30  # 30% change
        
        # Contract-specific calculations
        if 'close' in df.columns:
            close_prices = df['close'].astype(np.float64).values
            
            # Daily returns for futures using TA-Lib ROC
            returns = talib.ROC(close_prices, timeperiod=1)
            result_df['daily_return'] = returns
            
            # Realized volatility (rolling)
            if len(df) >= 20:
                returns_series = pd.Series(returns, index=df.index)
                result_df['realized_vol_10'] = returns_series.rolling(10).std() * np.sqrt(252)
                result_df['realized_vol_20'] = returns_series.rolling(20).std() * np.sqrt(252)
            
            # Price gaps (futures specific)
            if 'open' in df.columns:
                gap = (df['open'] - df['close'].shift(1)) / df['close'].shift(1) * 100
                result_df['gap_pct'] = gap
                result_df['gap_up'] = gap > 1.0  # >1% gap up
                result_df['gap_down'] = gap < -1.0  # >1% gap down
        
        logger.info("Calculated futures-specific indicators for %d rows", len(result_df))
        
    except Exception as exc:
        logger.error("Error calculating futures indicators: %s", exc)
        
    return result_df


# =============================================================================
# COMPREHENSIVE INDICATOR SUITE
# =============================================================================

def compute_all_indicators(df: pd.DataFrame, include_futures_indicators: bool = True) -> pd.DataFrame:
    """
    Compute complete suite of technical indicators using TA-Lib.
    
    Args:
        df: DataFrame with OHLCV(I) data
        include_futures_indicators: Whether to include futures-specific indicators
        
    Returns:
        DataFrame with all TA-Lib indicators computed
    """
    if df.empty:
        logger.warning("Empty DataFrame provided")
        return df
    
    if not HAS_TALIB:
        logger.error("TA-Lib not available - cannot compute indicators")
        return df
    
    logger.info("Computing TA-Lib indicators for %d rows", len(df))
    
    result_df = df.copy()
    
    # Apply all TA-Lib indicator functions
    result_df = macd_multi_timeframe(result_df)
    result_df = rsi_multi_period(result_df) 
    result_df = atr_volatility(result_df)
    result_df = bollinger_bands(result_df)
    result_df = volume_indicators(result_df)
    result_df = momentum_oscillators(result_df)
    result_df = trend_indicators(result_df)
    
    # Futures-specific indicators (if requested)
    if include_futures_indicators:
        result_df = futures_specific_indicators(result_df)
    
    indicators_added = len(result_df.columns) - len(df.columns)
    logger.info("Added %d TA-Lib indicators", indicators_added)
    
    return result_df


# =============================================================================
# DATA VALIDATION FUNCTIONS (moved from loaders.py)
# =============================================================================

def validate_ohlc_logic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate OHLC relationships and fix obvious data entry errors.
    
    Ensures that:
    - High >= max(Open, Close, Low)
    - Low <= min(Open, Close, High)
    
    Args:
        df: DataFrame with OHLC columns
        
    Returns:
        DataFrame with corrected OHLC values
    """
    df = df.copy()
    
    # Convert to numeric
    for col in ['open', 'high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Check OHLC logic violations
    invalid_high = (df['high'] < df[['open', 'close', 'low']].max(axis=1))
    invalid_low = (df['low'] > df[['open', 'close', 'high']].min(axis=1))
    
    invalid_count = invalid_high.sum() + invalid_low.sum()
    
    if invalid_count > 0:
        logger.info("Found %d OHLC logic violations - fixing...", invalid_count)
        
        # Fix high values: set to maximum of open, close, low
        df.loc[invalid_high, 'high'] = df.loc[invalid_high, ['open', 'close', 'low']].max(axis=1)
        
        # Fix low values: set to minimum of open, close, high  
        df.loc[invalid_low, 'low'] = df.loc[invalid_low, ['open', 'close', 'high']].min(axis=1)
    
    return df


def validate_volume_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean volume data by handling negative and missing values.
    
    Args:
        df: DataFrame with volume column
        
    Returns:
        DataFrame with cleaned volume data (negatives set to 0)
    """
    df = df.copy()
    
    # Convert volume to numeric
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
    
    # Handle negative volumes (data errors)
    negative_volumes = (df['volume'] < 0).sum()
    if negative_volumes > 0:
        logger.info("Fixing %d negative volume entries...", negative_volumes)
        df.loc[df['volume'] < 0, 'volume'] = 0
    
    return df


def remove_duplicate_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate timestamp entries keeping the first occurrence.
    
    Args:
        df: DataFrame with date and time columns
        
    Returns:
        DataFrame with duplicate timestamps removed
    """
    df = df.copy()
    
    # Create datetime column for duplicate detection
    df['datetime'] = df['date'] + ' ' + df['time']
    
    initial_count = len(df)
    df = df.drop_duplicates(subset=['datetime'], keep='first')
    duplicates_removed = initial_count - len(df)
    
    if duplicates_removed > 0:
        logger.info("Removed %d duplicate timestamps...", duplicates_removed)
    
    # Drop the temporary datetime column
    df = df.drop('datetime', axis=1)
    
    return df


def drop_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop all rows with missing critical data to maintain 100% authenticity.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with complete records only (no imputed values)
    """
    df = df.copy()
    initial_count = len(df)
    
    # Drop rows with missing OHLC data (critical for analysis)
    df_clean = df.dropna(subset=['open', 'high', 'low', 'close'])
    
    # For volume, set NaN to 0 (common in some data feeds)
    df_clean['volume'] = df_clean['volume'].fillna(0)
    
    dropped_count = initial_count - len(df_clean)
    if dropped_count > 0:
        logger.info("Dropped %d rows with missing price data", dropped_count)
        logger.info("Kept %d complete records (%.1f%% retention)", 
                   len(df_clean), (len(df_clean)/initial_count)*100)
    
    return df_clean


def validate_and_clean_data(historical_data: List[Dict]) -> List[Dict]:
    """
    Comprehensive data validation pipeline using DROP missing data approach.
    
    Performs:
    - OHLC logic validation and correction
    - Volume data cleaning  
    - Duplicate timestamp removal
    - Missing data elimination (no imputation)
    
    Args:
        historical_data: List of raw OHLCV dictionaries from API
        
    Returns:
        List of validated and cleaned OHLCV dictionaries
    """
    logger.info("Starting data validation pipeline...")
    
    if not historical_data:
        logger.warning("Empty historical data provided")
        return []
    
    # Convert to DataFrame for processing
    df = pd.DataFrame(historical_data)
    
    if df.empty:
        logger.warning("Could not create DataFrame from historical data")
        return []
    
    # Apply validation pipeline
    df = validate_ohlc_logic(df)
    df = validate_volume_data(df) 
    df = remove_duplicate_timestamps(df)
    df = drop_missing_data(df)
    
    logger.info("Data validation complete!")
    
    # Convert back to list of dictionaries
    return df.to_dict('records')


# =============================================================================
# TESTING AND EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("Testing TA-Lib Technical Indicators")
    
    try:
        # Create sample OHLCV data
        np.random.seed(42)
        n = 100
        
        # Generate realistic price data
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + abs(np.random.randn(n) * 0.3)
        low = close - abs(np.random.randn(n) * 0.3)
        open_price = close + np.random.randn(n) * 0.2
        volume = np.random.randint(1000, 5000, n)
        
        # Create DataFrame
        dates = pd.date_range('2024-01-01', periods=n, freq='5min')
        test_df = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low, 
            'close': close,
            'volume': volume
        }, index=dates)
        
        print(f"Created sample data: {len(test_df)} rows")
        
        # Test individual functions
        if HAS_TALIB:
            print("Testing individual indicator functions...")
            
            # Test MACD
            macd_df = macd_multi_timeframe(test_df)
            print(f"MACD: Added {len(macd_df.columns) - len(test_df.columns)} indicators")
            
            # Test RSI
            rsi_df = rsi_multi_period(test_df)
            print(f"RSI: Added {len(rsi_df.columns) - len(test_df.columns)} indicators")
            
            # Test complete suite
            result_df = compute_all_indicators(test_df)
            new_cols = len(result_df.columns) - len(test_df.columns)
            print(f"Complete suite: Added {new_cols} TA-Lib indicators")
            
            # Show sample values
            sample_cols = [col for col in result_df.columns if 'macd' in col or 'rsi' in col or 'atr' in col][:5]
            if sample_cols:
                print("\nSample indicator values:")
                print(result_df[sample_cols].tail(3).round(4))
        else:
            print("TA-Lib not available - skipping test")
            
    except Exception as e:
        print(f"Test failed: {e}")
        
    print("Test completed")