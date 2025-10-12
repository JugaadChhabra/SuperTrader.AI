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

def enhanced_oi_indicators(df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
    """
    Advanced Open Interest analysis beyond basic implementation.
    
    Args:
        df: DataFrame with 'open_interest' column and OHLCV data
        symbol: Symbol name for logging
        
    Returns:
        DataFrame with enhanced OI indicators added
    """
    result_df = df.copy()
    
    if 'open_interest' not in df.columns or not HAS_TALIB:
        logger.debug("No OI data or TA-Lib unavailable for enhanced OI analysis")
        return result_df
    
    try:
        oi_data = df['open_interest'].astype(np.float64).values
        close_data = df['close'].astype(np.float64).values if 'close' in df.columns else None
        volume_data = df['volume'].astype(np.float64).values if 'volume' in df.columns else None
        
        # 1. OI Concentration Analysis
        if len(df) >= 50:
            oi_series = pd.Series(oi_data, index=df.index)
            
            # Rolling maximum OI (concentration indicator)
            oi_rolling_max = oi_series.rolling(50, min_periods=25).max()
            oi_concentration = oi_series / oi_rolling_max
            result_df['oi_concentration_index'] = oi_concentration
            
            # High concentration periods (>90% of recent max)
            result_df['high_oi_concentration'] = (oi_concentration > 0.9).astype(int)
            
            # OI distribution analysis (percentile-based)
            oi_percentile = oi_series.rolling(100, min_periods=50).rank(pct=True) * 100
            result_df['oi_percentile_rank'] = oi_percentile
            
            # Extreme OI levels
            result_df['oi_extreme_high'] = (oi_percentile > 95).astype(int)  # Top 5%
            result_df['oi_extreme_low'] = (oi_percentile < 5).astype(int)    # Bottom 5%
        
        # 2. OI Flow Momentum (acceleration analysis)
        oi_roc_1 = talib.ROC(oi_data, timeperiod=1)  # 1-period change
        oi_roc_5 = talib.ROC(oi_data, timeperiod=5)  # 5-period change
        
        result_df['oi_flow_1d'] = oi_roc_1
        result_df['oi_flow_5d'] = oi_roc_5
        
        # OI acceleration (second derivative)
        if len(df) >= 10:
            oi_roc_series = pd.Series(oi_roc_5, index=df.index)
            oi_acceleration = oi_roc_series.diff()
            result_df['oi_acceleration'] = oi_acceleration
            
            # OI momentum regime
            result_df['oi_accelerating'] = (oi_acceleration > 0).astype(int)
            result_df['oi_decelerating'] = (oi_acceleration < 0).astype(int)
        
        # 3. Price-OI Efficiency Ratio
        if close_data is not None and len(df) >= 10:
            price_roc_5 = talib.ROC(close_data, timeperiod=5)
            
            # Efficiency = Price change per unit OI change
            oi_change = pd.Series(oi_roc_5, index=df.index)
            price_change = pd.Series(price_roc_5, index=df.index)
            
            # Avoid division by zero
            efficiency_ratio = price_change / (abs(oi_change) + 0.001)
            result_df['price_oi_efficiency'] = efficiency_ratio
            
            # High efficiency (large price moves with small OI changes)
            efficiency_zscore = (efficiency_ratio - efficiency_ratio.rolling(20).mean()) / efficiency_ratio.rolling(20).std()
            result_df['price_oi_efficiency_zscore'] = efficiency_zscore
            result_df['high_efficiency_regime'] = (abs(efficiency_zscore) > 1.5).astype(int)
        
        # 4. OI Support/Resistance Levels
        if len(df) >= 100:
            oi_series = pd.Series(oi_data, index=df.index)
            
            # Find local OI maxima (resistance levels)
            oi_rolling_max_20 = oi_series.rolling(20, center=True).max()
            oi_local_max = (oi_series == oi_rolling_max_20) & (oi_series > oi_series.rolling(40).median() * 1.2)
            result_df['oi_resistance_level'] = oi_local_max.astype(int)
            
            # Find local OI minima (support levels)
            oi_rolling_min_20 = oi_series.rolling(20, center=True).min()
            oi_local_min = (oi_series == oi_rolling_min_20) & (oi_series < oi_series.rolling(40).median() * 0.8)
            result_df['oi_support_level'] = oi_local_min.astype(int)
        
        # 5. Volume-OI Divergence Analysis
        if volume_data is not None:
            vol_roc_5 = talib.ROC(volume_data, timeperiod=5)
            
            # Normalize both indicators
            vol_roc_norm = pd.Series(vol_roc_5, index=df.index).rolling(20).rank(pct=True)
            oi_roc_norm = pd.Series(oi_roc_5, index=df.index).rolling(20).rank(pct=True)
            
            # Divergence score
            vol_oi_divergence = abs(vol_roc_norm - oi_roc_norm)
            result_df['volume_oi_divergence_score'] = vol_oi_divergence
            
            # Strong divergence signal
            result_df['volume_oi_divergence_signal'] = (vol_oi_divergence > 0.7).astype(int)
        
        # 6. OI Trend Strength Indicator
        oi_sma_10 = talib.SMA(oi_data, timeperiod=10)
        oi_sma_30 = talib.SMA(oi_data, timeperiod=30)
        
        if len(df) >= 30:
            # OI trend strength (similar to ADX for price)
            oi_trend_up = oi_sma_10 > oi_sma_30
            oi_trend_strength = abs(oi_sma_10 - oi_sma_30) / oi_sma_30
            
            result_df['oi_trend_direction'] = oi_trend_up.astype(int)  # 1=up, 0=down
            result_df['oi_trend_strength'] = oi_trend_strength
            result_df['oi_strong_trend'] = (oi_trend_strength > 0.1).astype(int)  # >10% difference
        
        logger.info("Added enhanced OI indicators for %s: %d new features", 
                   symbol or "futures", len(result_df.columns) - len(df.columns))
        
    except Exception as e:
        logger.error("Enhanced OI indicators calculation failed: %s", e)
    
    return result_df


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
        # Basic Open Interest analysis (if available)
        if 'open_interest' in df.columns:
            oi_data = df['open_interest'].astype(np.float64).values
            
            # Basic OI indicators using TA-Lib
            result_df['oi_sma_10'] = talib.SMA(oi_data, timeperiod=10)
            result_df['oi_sma_20'] = talib.SMA(oi_data, timeperiod=20)
            result_df['oi_ema_10'] = talib.EMA(oi_data, timeperiod=10)
            
            # OI momentum using TA-Lib
            result_df['oi_roc_5'] = talib.ROC(oi_data, timeperiod=5)
            result_df['oi_momentum_3'] = talib.MOM(oi_data, timeperiod=3)
            
            # Basic OI trend analysis
            result_df['oi_increasing'] = result_df['oi_sma_10'] > result_df['oi_sma_20']
            result_df['oi_decreasing'] = result_df['oi_sma_10'] < result_df['oi_sma_20']
            
            # Large OI changes (potential rollover signals)
            oi_pct_change = pd.Series(oi_data).pct_change()
            result_df['oi_large_increase'] = oi_pct_change > 0.15  # 15% increase
            result_df['oi_large_decrease'] = oi_pct_change < -0.15  # 15% decrease
            result_df['potential_rollover'] = abs(oi_pct_change) > 0.30  # 30% change
            
            # Add enhanced OI analysis
            result_df = enhanced_oi_indicators(result_df)        # Contract-specific calculations
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

def compute_all_indicators(df: pd.DataFrame, include_futures_indicators: bool = True, 
                          pcr_data: dict = None, symbol: str = None) -> pd.DataFrame:
    """
    Compute complete suite of technical indicators using TA-Lib.
    Phase 4 Enhanced Version with PCR integration and advanced composites.
    
    Args:
        df: DataFrame with OHLCV(I) data
        include_futures_indicators: Whether to include futures-specific indicators
        pcr_data: Optional PCR data dictionary for integration
        symbol: Optional symbol name for logging
        
    Returns:
        DataFrame with all TA-Lib indicators computed
    """
    if df.empty:
        logger.warning("Empty DataFrame provided")
        return df
    
    if not HAS_TALIB:
        logger.error("TA-Lib not available - cannot compute indicators")
        return df
    
    logger.info("Computing enhanced TA-Lib indicators for %s: %d rows", symbol or "data", len(df))
    
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
    
    # Phase 4: PCR Integration
    if pcr_data:
        try:
            pcr_computed = pcr_indicators(pcr_data, symbol=symbol)
            if pcr_computed:
                # Add timestamp if missing for integration
                if 'timestamp' not in result_df.columns:
                    result_df_with_ts = result_df.copy()
                    result_df_with_ts['timestamp'] = result_df.index
                    result_df = integrate_pcr_with_price(result_df_with_ts, pcr_computed, 'timestamp')
                    result_df = result_df.drop('timestamp', axis=1, errors='ignore')
                else:
                    result_df = integrate_pcr_with_price(result_df, pcr_computed, 'timestamp')
                logger.info("Successfully integrated PCR data")
        except Exception as e:
            logger.warning("PCR integration failed: %s", e)
    
    # Phase 4: Advanced Composite Indicators
    try:
        result_df = add_advanced_composite_indicators(result_df, symbol=symbol)
    except Exception as e:
        logger.warning("Advanced composite indicators failed: %s", e)
    
    indicators_added = len(result_df.columns) - len(df.columns)
    logger.info("Added %d enhanced TA-Lib indicators for %s", indicators_added, symbol or "data")
    
    return result_df


def add_advanced_composite_indicators(df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
    """
    Add advanced composite indicators that combine multiple basic indicators.
    Phase 4 Enhancement: Market regime, momentum confluence, and risk indicators.
    
    Args:
        df: DataFrame with basic indicators already calculated
        symbol: Symbol name for logging
        
    Returns:
        DataFrame with composite indicators added
    """
    result_df = df.copy()
    
    try:
        # 1. Market Regime Indicator (Trend + Volatility + Volume)
        if all(col in df.columns for col in ['adx', 'atr_percentile']):
            # Strong trend regime
            strong_trend_conditions = []
            if 'adx' in df.columns:
                strong_trend_conditions.append(df['adx'] > 25)
            if 'atr_percentile' in df.columns:
                strong_trend_conditions.append(df['atr_percentile'] < 80)  # Not excessive volatility
            if 'volume_sma_ratio' in df.columns:
                strong_trend_conditions.append(df['volume_sma_ratio'] > 1.2)  # Above average volume
                
            if strong_trend_conditions:
                strong_trend = pd.concat(strong_trend_conditions, axis=1).all(axis=1).astype(int)
                result_df['market_regime_strong_trend'] = strong_trend
            
            # Consolidation regime
            consolidation_conditions = []
            if 'adx' in df.columns:
                consolidation_conditions.append(df['adx'] < 20)
            if 'atr_percentile' in df.columns:
                consolidation_conditions.append(df['atr_percentile'] < 40)
                
            if consolidation_conditions:
                consolidation = pd.concat(consolidation_conditions, axis=1).all(axis=1).astype(int)
                result_df['market_regime_consolidation'] = consolidation
        
        # 2. Momentum Confluence Score
        momentum_signals = []
        
        # RSI momentum (bullish > 50)
        if 'rsi_14' in df.columns:
            momentum_signals.append((df['rsi_14'] > 50).astype(int))
        
        # MACD momentum
        macd_cols = [col for col in df.columns if 'macd_bullish' in col]
        if macd_cols:
            momentum_signals.append(df[macd_cols[0]].astype(int))
        
        # Price momentum (above EMA)
        if 'ema_12' in df.columns and 'close' in df.columns:
            momentum_signals.append((df['close'] > df['ema_12']).astype(int))
        
        if momentum_signals:
            momentum_confluence = pd.concat(momentum_signals, axis=1).sum(axis=1)
            result_df['momentum_confluence_score'] = momentum_confluence
            result_df['strong_momentum_confluence'] = (momentum_confluence >= len(momentum_signals) * 0.75).astype(int)
        
        # 3. Volatility Regime Analysis
        if 'atr_percentile' in df.columns:
            atr_pct = df['atr_percentile']
            
            # Volatility regime classification
            result_df['volatility_regime_low'] = (atr_pct < 25).astype(int)
            result_df['volatility_regime_normal'] = ((atr_pct >= 25) & (atr_pct <= 75)).astype(int)
            result_df['volatility_regime_high'] = (atr_pct > 75).astype(int)
            
            # Volatility expansion/contraction (momentum)
            if len(df) >= 10:
                atr_momentum = atr_pct.diff(5)  # 5-period change in percentile
                result_df['volatility_expanding'] = (atr_momentum > 15).astype(int)
                result_df['volatility_contracting'] = (atr_momentum < -15).astype(int)
        
        # 4. Risk Assessment Composite
        risk_components = []
        
        # Volatility risk (high volatility = high risk)
        if 'atr_percentile' in df.columns:
            vol_risk = (df['atr_percentile'] / 100).fillna(0.5)
            risk_components.append(vol_risk)
        
        # Liquidity risk (low volume = high risk)
        if 'volume_percentile' in df.columns:
            liquidity_risk = 1 - (df['volume_percentile'] / 100).fillna(0.5)
            risk_components.append(liquidity_risk)
        
        # Trend uncertainty risk (low ADX = high uncertainty)
        if 'adx' in df.columns:
            trend_uncertainty = 1 - (df['adx'].clip(0, 50) / 50).fillna(0.5)
            risk_components.append(trend_uncertainty)
        
        if risk_components:
            overall_risk = pd.concat(risk_components, axis=1).mean(axis=1)
            result_df['market_risk_score'] = overall_risk
            result_df['high_risk_environment'] = (overall_risk > 0.7).astype(int)
            result_df['low_risk_environment'] = (overall_risk < 0.3).astype(int)
        
        # 5. Setup Quality Score (for entry signals)
        quality_components = []
        
        # Volume confirmation
        if 'volume_sma_ratio' in df.columns:
            quality_components.append((df['volume_sma_ratio'] > 1.0).astype(int))
        
        # Trend alignment
        if all(col in df.columns for col in ['close', 'ema_12', 'ema_26']):
            uptrend_alignment = ((df['close'] > df['ema_12']) & (df['ema_12'] > df['ema_26'])).astype(int)
            quality_components.append(uptrend_alignment)
        
        # Appropriate volatility (not too high, not too low)
        if 'atr_percentile' in df.columns:
            good_volatility = ((df['atr_percentile'] > 20) & (df['atr_percentile'] < 80)).astype(int)
            quality_components.append(good_volatility)
        
        if quality_components:
            setup_quality = pd.concat(quality_components, axis=1).mean(axis=1)
            result_df['setup_quality_score'] = setup_quality
            result_df['high_quality_setup'] = (setup_quality > 0.75).astype(int)
        
        # 6. Options-specific signals (if PCR data available)
        pcr_columns = [col for col in df.columns if 'pcr' in col.lower()]
        if pcr_columns:
            # PCR extreme levels (contrarian signals)
            if 'pcr_percentile' in df.columns:
                result_df['pcr_extreme_bearish'] = (df['pcr_percentile'] > 85).astype(int)  # Very high PCR
                result_df['pcr_extreme_bullish'] = (df['pcr_percentile'] < 15).astype(int)  # Very low PCR
            
            # PCR momentum divergence
            if all(col in df.columns for col in ['pcr_roc_3', 'close']):
                price_roc = df['close'].pct_change(3) * 100
                pcr_price_divergence = (
                    (price_roc > 0) & (df['pcr_roc_3'] > 0)  # Price up, PCR up (bearish divergence)
                ) | (
                    (price_roc < 0) & (df['pcr_roc_3'] < 0)  # Price down, PCR down (bullish divergence)
                )
                result_df['pcr_momentum_divergence'] = pcr_price_divergence.astype(int)
        
        added_features = len(result_df.columns) - len(df.columns)
        logger.info("Added %d advanced composite indicators for %s", added_features, symbol or "data")
        
    except Exception as e:
        logger.error("Advanced composite indicators calculation failed: %s", e)
    
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
# PCR (PUT-CALL RATIO) INDICATORS
# =============================================================================

def pcr_indicators(pcr_data: dict, symbol: str = None) -> dict:
    """
    Calculate Put-Call Ratio (PCR) based indicators.
    
    Args:
        pcr_data: Dictionary with PCR data structure:
                 {'timestamp': [...], 'total_pcr': [...], 'index_pcr': [...], 
                  'ce_oi': [...], 'pe_oi': [...], 'ce_volume': [...], 'pe_volume': [...]}
        symbol: Symbol name for logging
        
    Returns:
        Dictionary with PCR-based indicators
    """
    indicators = {}
    
    if not pcr_data or not HAS_TALIB:
        logger.debug("No PCR data or TA-Lib unavailable")
        return indicators
    
    try:
        # Convert to pandas DataFrame for easier manipulation
        df = pd.DataFrame(pcr_data)
        
        if len(df) < 10:
            logger.warning("Insufficient PCR data for indicators: %d periods", len(df))
            return indicators
        
        # 1. Total PCR Analysis
        if 'total_pcr' in df.columns:
            pcr_values = df['total_pcr'].astype(np.float64).values
            
            # PCR moving averages
            indicators['pcr_sma_5'] = talib.SMA(pcr_values, timeperiod=5)
            indicators['pcr_sma_20'] = talib.SMA(pcr_values, timeperiod=20)
            indicators['pcr_ema_10'] = talib.EMA(pcr_values, timeperiod=10)
            
            # PCR momentum
            indicators['pcr_roc_3'] = talib.ROC(pcr_values, timeperiod=3)
            indicators['pcr_momentum_5'] = talib.MOM(pcr_values, timeperiod=5)
            
            # PCR extremes (contrarian signals)
            pcr_series = pd.Series(pcr_values, index=df.index)
            pcr_percentile = pcr_series.rolling(50, min_periods=25).rank(pct=True) * 100
            
            indicators['pcr_percentile'] = pcr_percentile.values
            indicators['pcr_oversold'] = (pcr_percentile > 80).astype(int).values   # High PCR = Oversold market
            indicators['pcr_overbought'] = (pcr_percentile < 20).astype(int).values # Low PCR = Overbought market
            
            # PCR volatility (uncertainty indicator)
            pcr_volatility = pcr_series.rolling(10).std()
            indicators['pcr_volatility'] = pcr_volatility.values
            indicators['pcr_high_uncertainty'] = (pcr_volatility > pcr_volatility.rolling(50).quantile(0.8)).astype(int).values
        
        # 2. Index-specific PCR Analysis
        if 'index_pcr' in df.columns:
            index_pcr_values = df['index_pcr'].astype(np.float64).values
            
            indicators['index_pcr_sma_10'] = talib.SMA(index_pcr_values, timeperiod=10)
            indicators['index_pcr_rsi_14'] = talib.RSI(index_pcr_values, timeperiod=14)
            
            # Index PCR divergence from total PCR
            if 'total_pcr' in df.columns:
                total_pcr_norm = (df['total_pcr'] - df['total_pcr'].rolling(20).mean()) / df['total_pcr'].rolling(20).std()
                index_pcr_norm = (df['index_pcr'] - df['index_pcr'].rolling(20).mean()) / df['index_pcr'].rolling(20).std()
                
                pcr_divergence = abs(total_pcr_norm - index_pcr_norm)
                indicators['pcr_divergence_score'] = pcr_divergence.values
                indicators['pcr_significant_divergence'] = (pcr_divergence > 1.5).astype(int).values
        
        # 3. Options Flow Analysis (CE vs PE)
        if all(col in df.columns for col in ['ce_oi', 'pe_oi']):
            ce_oi = df['ce_oi'].astype(np.float64).values
            pe_oi = df['pe_oi'].astype(np.float64).values
            
            # OI-based PCR (more stable than volume-based)
            oi_pcr = pe_oi / (ce_oi + 1e-6)  # Avoid division by zero
            indicators['oi_pcr'] = oi_pcr
            indicators['oi_pcr_sma_10'] = talib.SMA(oi_pcr, timeperiod=10)
            
            # CE/PE OI momentum
            ce_oi_roc = talib.ROC(ce_oi, timeperiod=5)
            pe_oi_roc = talib.ROC(pe_oi, timeperiod=5)
            
            indicators['ce_oi_momentum'] = ce_oi_roc
            indicators['pe_oi_momentum'] = pe_oi_roc
            
            # Options flow bias
            flow_bias = (ce_oi_roc - pe_oi_roc) / (abs(ce_oi_roc) + abs(pe_oi_roc) + 1e-6)
            indicators['options_flow_bias'] = flow_bias
            indicators['bullish_flow'] = (flow_bias > 0.3).astype(int)
            indicators['bearish_flow'] = (flow_bias < -0.3).astype(int)
        
        # 4. Volume-based Analysis
        if all(col in df.columns for col in ['ce_volume', 'pe_volume']):
            ce_vol = df['ce_volume'].astype(np.float64).values
            pe_vol = df['pe_volume'].astype(np.float64).values
            
            # Volume PCR (more reactive than OI)
            vol_pcr = pe_vol / (ce_vol + 1e-6)
            indicators['vol_pcr'] = vol_pcr
            indicators['vol_pcr_sma_5'] = talib.SMA(vol_pcr, timeperiod=5)
            
            # Volume activity levels
            total_options_vol = ce_vol + pe_vol
            indicators['total_options_volume'] = total_options_vol
            indicators['options_vol_sma_10'] = talib.SMA(total_options_vol, timeperiod=10)
            
            # High activity periods
            vol_percentile = pd.Series(total_options_vol).rolling(50).rank(pct=True) * 100
            indicators['options_activity_percentile'] = vol_percentile.values
            indicators['high_options_activity'] = (vol_percentile > 90).astype(int).values
        
        # 5. Combined PCR Signal
        if 'total_pcr' in indicators and 'vol_pcr' in indicators:
            # Create composite signal
            composite_values = []
            for i in range(len(indicators['total_pcr'])):
                total_val = indicators['total_pcr'][i]
                vol_val = indicators['vol_pcr'][i]
                
                if not (np.isnan(total_val) or np.isnan(vol_val)):
                    composite_val = 0.7 * total_val + 0.3 * vol_val
                    composite_values.append(composite_val)
                else:
                    composite_values.append(np.nan)
            
            indicators['composite_pcr'] = np.array(composite_values)
            
            # Calculate SMA for composite
            valid_composite = [x for x in composite_values if not np.isnan(x)]
            if len(valid_composite) >= 10:
                indicators['composite_pcr_sma_10'] = talib.SMA(np.array(valid_composite), timeperiod=10)
        
        logger.info("Calculated PCR indicators for %s: %d indicators", 
                   symbol or "index", len(indicators))
        
    except Exception as e:
        logger.error("PCR indicators calculation failed: %s", e)
    
    return indicators


def integrate_pcr_with_price(price_df: pd.DataFrame, pcr_indicators: dict, 
                           sync_column: str = 'timestamp') -> pd.DataFrame:
    """
    Integrate PCR indicators with price data using timestamp synchronization.
    
    Args:
        price_df: Price DataFrame with timestamp column
        pcr_indicators: Dictionary of PCR indicators from pcr_indicators()
        sync_column: Column name for timestamp synchronization
        
    Returns:
        DataFrame with synchronized PCR and price data
    """
    result_df = price_df.copy()
    
    if not pcr_indicators or sync_column not in price_df.columns:
        logger.warning("No PCR data or sync column missing")
        return result_df
    
    try:
        # Create PCR DataFrame (handle missing timestamps)
        pcr_df_data = pcr_indicators.copy()
        
        # If no timestamp, create sequential timestamps
        if 'timestamp' not in pcr_df_data:
            n_periods = len(next(iter(pcr_df_data.values())))
            start_time = price_df[sync_column].iloc[0] if len(price_df) > 0 else pd.Timestamp.now()
            pcr_df_data['timestamp'] = pd.date_range(start_time, periods=n_periods, freq='5min')
        
        pcr_df = pd.DataFrame(pcr_df_data)
        
        # Ensure timestamp columns are datetime
        if not pd.api.types.is_datetime64_any_dtype(price_df[sync_column]):
            price_df_temp = price_df.copy()
            price_df_temp[sync_column] = pd.to_datetime(price_df[sync_column])
        else:
            price_df_temp = price_df.copy()
            
        if not pd.api.types.is_datetime64_any_dtype(pcr_df['timestamp']):
            pcr_df['timestamp'] = pd.to_datetime(pcr_df['timestamp'])
        
        # Forward-fill merge (use most recent PCR data for each price point)
        price_df_sorted = price_df_temp.sort_values(sync_column).reset_index(drop=True)
        pcr_df_sorted = pcr_df.sort_values('timestamp').reset_index(drop=True)
        
        # Merge using nearest timestamp (forward fill)
        merged_df = pd.merge_asof(
            price_df_sorted, 
            pcr_df_sorted,
            left_on=sync_column,
            right_on='timestamp',
            direction='backward',
            suffixes=('', '_pcr')
        )
        
        # Drop duplicate timestamp column
        if 'timestamp_pcr' in merged_df.columns:
            merged_df = merged_df.drop('timestamp_pcr', axis=1)
        
        result_df = merged_df
        
        logger.info("Integrated PCR indicators: %d PCR features added to price data", 
                   len(pcr_df.columns) - 1)  # -1 for timestamp
        
    except Exception as e:
        logger.error("PCR integration failed: %s", e)
    
    return result_df


# =============================================================================
# MULTI-TIMEFRAME SYNCHRONIZATION
# =============================================================================

def multi_timeframe_sync(df_dict: dict, base_timeframe: str = '5min') -> pd.DataFrame:
    """
    Synchronize indicators across multiple timeframes for confluence analysis.
    
    Args:
        df_dict: Dictionary of {timeframe: dataframe} with calculated indicators
        base_timeframe: Primary timeframe to use as base
        
    Returns:
        DataFrame with synchronized multi-timeframe indicators
    """
    if base_timeframe not in df_dict:
        logger.error("Base timeframe %s not found in data", base_timeframe)
        return pd.DataFrame()
    
    base_df = df_dict[base_timeframe].copy()
    
    # Validate base DataFrame has datetime index
    if not isinstance(base_df.index, pd.DatetimeIndex):
        if 'timestamp' in base_df.columns:
            base_df = base_df.set_index('timestamp')
        else:
            logger.error("No datetime index or timestamp column in base DataFrame")
            return base_df
    
    try:
        for timeframe, tf_df in df_dict.items():
            if timeframe == base_timeframe:
                continue
                
            # Ensure datetime index
            sync_df = tf_df.copy()
            if not isinstance(sync_df.index, pd.DatetimeIndex):
                if 'timestamp' in sync_df.columns:
                    sync_df = sync_df.set_index('timestamp')
                else:
                    continue
            
            # Select key indicators for synchronization
            sync_columns = []
            for col in sync_df.columns:
                if any(indicator in col.lower() for indicator in 
                      ['rsi', 'macd', 'atr', 'adx', 'bb_', 'sma', 'ema']):
                    sync_columns.append(col)
            
            if sync_columns:
                # Resample higher timeframe to base timeframe (forward fill)
                try:
                    sync_data = sync_df[sync_columns].resample(
                        pd.Timedelta(base_timeframe.replace('min', 'T'))
                    ).ffill()
                except:
                    # Fallback to simple forward fill
                    sync_data = sync_df[sync_columns].fillna(method='ffill')
                
                # Rename columns with timeframe suffix
                rename_dict = {col: f"{col}_{timeframe}" for col in sync_columns}
                sync_data = sync_data.rename(columns=rename_dict)
                
                # Merge with base DataFrame
                base_df = pd.merge(base_df, sync_data, left_index=True, right_index=True, how='left')
        
        # Add confluence indicators
        if len(df_dict) > 1:
            base_df = add_confluence_indicators(base_df, list(df_dict.keys()))
        
        logger.info("Synchronized %d timeframes to %s: %d total columns", 
                   len(df_dict), base_timeframe, len(base_df.columns))
        
    except Exception as e:
        logger.error("Multi-timeframe synchronization failed: %s", e)
    
    return base_df


def add_confluence_indicators(df: pd.DataFrame, timeframes: list) -> pd.DataFrame:
    """
    Add confluence indicators based on multi-timeframe alignment.
    
    Args:
        df: DataFrame with multi-timeframe indicators
        timeframes: List of timeframes used
        
    Returns:
        DataFrame with confluence indicators added
    """
    result_df = df.copy()
    
    try:
        # RSI Confluence (bullish when RSI > 50 across timeframes)
        rsi_columns = [col for col in df.columns if 'rsi_14' in col.lower()]
        if len(rsi_columns) > 1:
            rsi_bullish = pd.DataFrame()
            for col in rsi_columns:
                rsi_bullish[col] = (df[col] > 50).astype(int)
            
            result_df['rsi_confluence_bullish'] = rsi_bullish.sum(axis=1)
            result_df['rsi_confluence_bearish'] = len(rsi_columns) - result_df['rsi_confluence_bullish']
            result_df['rsi_strong_confluence'] = (result_df['rsi_confluence_bullish'] >= len(rsi_columns) * 0.75).astype(int)
        
        # MACD Confluence (bullish when MACD > Signal across timeframes)
        macd_columns = [col for col in df.columns if 'macd_bullish' in col.lower()]
        if len(macd_columns) > 1:
            macd_confluence = df[macd_columns].sum(axis=1)
            result_df['macd_confluence_score'] = macd_confluence
            result_df['macd_strong_confluence'] = (macd_confluence >= len(macd_columns) * 0.75).astype(int)
        
        # Trend Confluence (using moving averages)
        trend_columns = [col for col in df.columns if any(x in col.lower() for x in ['sma_20_above_sma_50', 'ema_12_above_ema_26'])]
        if len(trend_columns) > 1:
            trend_confluence = df[trend_columns].sum(axis=1)
            result_df['trend_confluence_score'] = trend_confluence
            result_df['trend_strong_confluence'] = (trend_confluence >= len(trend_columns) * 0.75).astype(int)
        
        # Volatility Confluence (using ATR percentiles)
        atr_columns = [col for col in df.columns if 'atr_percentile' in col.lower()]
        if len(atr_columns) > 1:
            atr_mean = df[atr_columns].mean(axis=1)
            result_df['volatility_confluence_level'] = atr_mean
            result_df['high_volatility_confluence'] = (atr_mean > 80).astype(int)
            result_df['low_volatility_confluence'] = (atr_mean < 20).astype(int)
        
        # Overall market confluence score
        confluence_indicators = [col for col in result_df.columns if 'confluence' in col.lower() and 'score' in col.lower()]
        if confluence_indicators:
            # Normalize scores to 0-1 range
            normalized_scores = pd.DataFrame()
            for col in confluence_indicators:
                max_val = result_df[col].max() if result_df[col].max() > 0 else 1
                normalized_scores[col] = result_df[col] / max_val
            
            result_df['overall_confluence_score'] = normalized_scores.mean(axis=1)
            result_df['strong_market_confluence'] = (result_df['overall_confluence_score'] > 0.75).astype(int)
        
        logger.info("Added confluence indicators for %d timeframes: %d new features", 
                   len(timeframes), len(result_df.columns) - len(df.columns))
        
    except Exception as e:
        logger.error("Confluence indicators calculation failed: %s", e)
    
    return result_df


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