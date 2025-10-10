"""
Technical Indicators for SuperTrader.AI using TA-Lib
Comprehensive technical analysis indicators optimized for index futures trading.

This module prioritizes TA-Lib's industry-standard C-optimized functions for
maximum performance and reliability. All core indicators use TA-Lib implementations.

Features:
- Multi-timeframe MACD (8/21, 12/26, 19/39) using TA-Lib MACD
- Multi-period RSI (9, 14, 21, 30) using TA-Lib RSI
- ATR volatility indicators using TA-Lib ATR and TRANGE
- Volume analysis using TA-Lib OBV, AD, ROC
- Bollinger Bands using TA-Lib BBANDS
- Momentum oscillators: TA-Lib STOCH, WILLR, CCI, MOM
- Trend indicators: TA-Lib SMA, EMA, SAR, ADX
- Futures-specific Open Interest analysis

Usage:
    from indicators.technical import TechnicalIndicators
    
    indicators = TechnicalIndicators()
    result_df = indicators.compute_all_indicators(df)
"""

# Standard library imports
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party imports
import numpy as np
import pandas as pd

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False
    print("[WARNING] TA-Lib not found. Install with: pip install TA-Lib")
    print("[INFO] Falling back to basic pandas calculations where possible")

# Configure logging
logger = logging.getLogger(__name__)

# Constants
DEFAULT_RISK_FREE_RATE = 0.06  # 6% annual risk-free rate for basis calculations
TRADING_DAYS_PER_YEAR = 252
MARKET_OPEN_TIME = "09:15:00"
MARKET_CLOSE_TIME = "15:30:00"


class TechnicalIndicators:
    """
    Comprehensive technical indicators class using TA-Lib for futures trading analysis.
    
    Provides both traditional technical indicators and futures-specific metrics
    including open interest analysis, basis calculations, and rollover detection.
    All calculations use TA-Lib's optimized C implementations for maximum performance.
    """
    
    def __init__(self, risk_free_rate: float = DEFAULT_RISK_FREE_RATE):
        """
        Initialize technical indicators calculator.
        
        Args:
            risk_free_rate: Annual risk-free rate for basis calculations
        """
        self.risk_free_rate = risk_free_rate
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.has_talib = HAS_TALIB
        
        if self.has_talib:
            # Verify TA-Lib is working
            try:
                talib.SMA(np.array([1, 2, 3, 4, 5]), 3)
                self.logger.info("TA-Lib initialized successfully")
            except Exception as e:
                self.logger.warning("TA-Lib verification failed: %s", e)
                self.has_talib = False
        
        if not self.has_talib:
            self.logger.warning("Using pandas fallback implementations")
    
    def _ensure_numpy_arrays(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Convert DataFrame columns to numpy arrays for TA-Lib functions.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dict with numpy arrays for OHLCV data
        """
        arrays = {}
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                # Convert to float64 and handle NaN values
                arrays[col] = df[col].astype(np.float64).values
            else:
                self.logger.warning("Missing column: %s", col)
                arrays[col] = np.full(len(df), np.nan, dtype=np.float64)
        
        return arrays
    
    def macd_multi_timeframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate MACD for multiple timeframes using TA-Lib.
        
        Timeframes:
        - Fast: 8/21/9 for short-term signals
        - Standard: 12/26/9 for medium-term signals  
        - Slow: 19/39/9 for long-term signals
        """
        result_df = df.copy()
        
        if 'close' not in df.columns:
            self.logger.warning("Missing 'close' column for MACD calculation")
            return result_df
        
        if not self.has_talib:
            self.logger.warning("TA-Lib not available for MACD calculation")
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
            
            self.logger.info("Calculated TA-Lib MACD for %d rows", len(result_df))
            
        except Exception as exc:
            self.logger.error("Error calculating MACD: %s", exc)
            
        return result_df
    
    def rsi_multi_period(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RSI for multiple periods using TA-Lib RSI function.
        
        Periods: 9, 14, 21, 30 for different momentum timeframes
        """
        result_df = df.copy()
        
        if 'close' not in df.columns:
            self.logger.warning("Missing 'close' column for RSI calculation")
            return result_df
        
        if not self.has_talib:
            self.logger.warning("TA-Lib not available for RSI calculation")
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
            
            self.logger.info("Calculated TA-Lib RSI for %d rows", len(result_df))
            
        except Exception as exc:
            self.logger.error("Error calculating RSI: %s", exc)
            
        return result_df
    
    def atr_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate ATR volatility indicators using TA-Lib ATR and TRANGE functions.
        """
        result_df = df.copy()
        
        required_cols = ['high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            self.logger.warning("Missing required OHLC columns for ATR: %s", required_cols)
            return result_df
        
        if not self.has_talib:
            self.logger.warning("TA-Lib not available for ATR calculation")
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
            
            self.logger.info("Calculated TA-Lib ATR for %d rows", len(result_df))
            
        except Exception as exc:
            self.logger.error("Error calculating ATR: %s", exc)
            
        return result_df
    
    def bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Bollinger Bands using TA-Lib.
        
        Args:
            df: DataFrame with 'close' column
            
        Returns:
            DataFrame with Bollinger Bands indicators added
        """
        result_df = df.copy()
        
        if 'close' not in df.columns:
            self.logger.warning("Missing 'close' column for Bollinger Bands")
            return result_df
        
        try:
            arrays = self._ensure_numpy_arrays(df)
            close_prices = arrays['close']
            
            # Standard Bollinger Bands (20, 2)
            bb_upper, bb_middle, bb_lower = talib.BBANDS(
                close_prices, 
                timeperiod=20, 
                nbdevup=2, 
                nbdevdn=2, 
                matype=0
            )
            
            result_df['bb_upper'] = bb_upper
            result_df['bb_middle'] = bb_middle
            result_df['bb_lower'] = bb_lower
            
            # Bollinger Band derived indicators
            result_df['bb_width'] = (bb_upper - bb_lower) / bb_middle
            result_df['bb_position'] = (close_prices - bb_lower) / (bb_upper - bb_lower)
            result_df['bb_squeeze'] = result_df['bb_width'] < result_df['bb_width'].rolling(20).quantile(0.1)
            
            # Bollinger Band signals
            result_df['bb_upper_breach'] = close_prices > bb_upper
            result_df['bb_lower_breach'] = close_prices < bb_lower
            result_df['bb_mean_reversion_long'] = (close_prices < bb_lower) & (close_prices.shift(1) >= bb_lower.shift(1))
            result_df['bb_mean_reversion_short'] = (close_prices > bb_upper) & (close_prices.shift(1) <= bb_upper.shift(1))
            
            self.logger.info("Calculated Bollinger Bands for %d rows", len(result_df))
            
        except Exception as exc:
            self.logger.error("Error calculating Bollinger Bands: %s", exc)
            
        return result_df
    
    def volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume-based indicators using TA-Lib.
        
        Args:
            df: DataFrame with 'close' and 'volume' columns
            
        Returns:
            DataFrame with volume indicators added
        """
        result_df = df.copy()
        
        required_cols = ['close', 'volume']
        if not all(col in df.columns for col in required_cols):
            self.logger.warning("Missing required columns for volume indicators: %s", required_cols)
            return result_df
        
        try:
            arrays = self._ensure_numpy_arrays(df)
            
            # On Balance Volume using TA-Lib
            obv = talib.OBV(arrays['close'], arrays['volume'])
            result_df['obv'] = obv
            
            # Volume moving averages
            for period in [10, 20, 50]:
                vol_ma = talib.SMA(arrays['volume'], timeperiod=period)
                result_df[f'volume_sma_{period}'] = vol_ma
                result_df[f'volume_ratio_{period}'] = arrays['volume'] / vol_ma
            
            # Volume Rate of Change
            volume_roc_5 = talib.ROC(arrays['volume'], timeperiod=5)
            volume_roc_10 = talib.ROC(arrays['volume'], timeperiod=10)
            result_df['volume_roc_5'] = volume_roc_5
            result_df['volume_roc_10'] = volume_roc_10
            
            # Volume surge detection using z-score
            if len(df) >= 20:
                volume_series = pd.Series(arrays['volume'], index=df.index)
                volume_mean = volume_series.rolling(20, min_periods=10).mean()
                volume_std = volume_series.rolling(20, min_periods=10).std()
                
                result_df['volume_zscore'] = (volume_series - volume_mean) / volume_std
                result_df['volume_surge_mild'] = result_df['volume_zscore'] > 1.5
                result_df['volume_surge_strong'] = result_df['volume_zscore'] > 2.0
                result_df['volume_surge_extreme'] = result_df['volume_zscore'] > 3.0
            
            # Accumulation/Distribution Line
            if all(col in df.columns for col in ['high', 'low', 'close', 'volume']):
                ad_line = talib.AD(arrays['high'], arrays['low'], arrays['close'], arrays['volume'])
                result_df['ad_line'] = ad_line
            
            self.logger.info("Calculated volume indicators for %d rows", len(result_df))
            
        except Exception as exc:
            self.logger.error("Error calculating volume indicators: %s", exc)
            
        return result_df
    
    def momentum_oscillators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate momentum oscillators using TA-Lib.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with momentum oscillators added
        """
        result_df = df.copy()
        
        required_cols = ['high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            self.logger.warning("Missing required columns for momentum oscillators: %s", required_cols)
            return result_df
        
        try:
            arrays = self._ensure_numpy_arrays(df)
            
            # Stochastic Oscillator
            slowk, slowd = talib.STOCH(
                arrays['high'], 
                arrays['low'], 
                arrays['close'], 
                fastk_period=14, 
                slowk_period=3, 
                slowk_matype=0, 
                slowd_period=3, 
                slowd_matype=0
            )
            result_df['stoch_k'] = slowk
            result_df['stoch_d'] = slowd
            result_df['stoch_overbought'] = slowk > 80
            result_df['stoch_oversold'] = slowk < 20
            
            # Williams %R
            willr = talib.WILLR(arrays['high'], arrays['low'], arrays['close'], timeperiod=14)
            result_df['williams_r'] = willr
            result_df['williams_r_overbought'] = willr > -20
            result_df['williams_r_oversold'] = willr < -80
            
            # Commodity Channel Index (CCI)
            cci = talib.CCI(arrays['high'], arrays['low'], arrays['close'], timeperiod=14)
            result_df['cci'] = cci
            result_df['cci_overbought'] = cci > 100
            result_df['cci_oversold'] = cci < -100
            result_df['cci_extreme_overbought'] = cci > 200
            result_df['cci_extreme_oversold'] = cci < -200
            
            # Rate of Change (ROC)
            roc_5 = talib.ROC(arrays['close'], timeperiod=5)
            roc_10 = talib.ROC(arrays['close'], timeperiod=10)
            roc_20 = talib.ROC(arrays['close'], timeperiod=20)
            
            result_df['roc_5'] = roc_5
            result_df['roc_10'] = roc_10
            result_df['roc_20'] = roc_20
            
            # Momentum
            momentum_5 = talib.MOM(arrays['close'], timeperiod=5)
            momentum_10 = talib.MOM(arrays['close'], timeperiod=10)
            
            result_df['momentum_5'] = momentum_5
            result_df['momentum_10'] = momentum_10
            
            self.logger.info("Calculated momentum oscillators for %d rows", len(result_df))
            
        except Exception as exc:
            self.logger.error("Error calculating momentum oscillators: %s", exc)
            
        return result_df
    
    def trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trend indicators using TA-Lib.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with trend indicators added
        """
        result_df = df.copy()
        
        if 'close' not in df.columns:
            self.logger.warning("Missing 'close' column for trend indicators")
            return result_df
        
        try:
            arrays = self._ensure_numpy_arrays(df)
            close_prices = arrays['close']
            
            # Simple Moving Averages
            sma_periods = [5, 10, 20, 50, 100, 200]
            for period in sma_periods:
                if len(df) >= period:
                    sma = talib.SMA(close_prices, timeperiod=period)
                    result_df[f'sma_{period}'] = sma
                    result_df[f'price_above_sma_{period}'] = close_prices > sma
            
            # Exponential Moving Averages
            ema_periods = [9, 12, 21, 26, 50]
            for period in ema_periods:
                if len(df) >= period:
                    ema = talib.EMA(close_prices, timeperiod=period)
                    result_df[f'ema_{period}'] = ema
                    result_df[f'price_above_ema_{period}'] = close_prices > ema
            
            # Moving Average crossovers
            if len(df) >= 50:
                result_df['sma_20_above_sma_50'] = result_df['sma_20'] > result_df['sma_50']
                result_df['ema_12_above_ema_26'] = result_df['ema_12'] > result_df['ema_26']
                
                # Golden/Death cross signals
                result_df['golden_cross'] = (
                    (result_df['sma_20'] > result_df['sma_50']) & 
                    (result_df['sma_20'].shift(1) <= result_df['sma_50'].shift(1))
                )
                result_df['death_cross'] = (
                    (result_df['sma_20'] < result_df['sma_50']) & 
                    (result_df['sma_20'].shift(1) >= result_df['sma_50'].shift(1))
                )
            
            # Parabolic SAR
            if all(col in df.columns for col in ['high', 'low']):
                sar = talib.SAR(arrays['high'], arrays['low'], acceleration=0.02, maximum=0.2)
                result_df['sar'] = sar
                result_df['price_above_sar'] = close_prices > sar
            
            # Average Directional Index (ADX)
            if all(col in df.columns for col in ['high', 'low', 'close']):
                adx = talib.ADX(arrays['high'], arrays['low'], arrays['close'], timeperiod=14)
                plus_di = talib.PLUS_DI(arrays['high'], arrays['low'], arrays['close'], timeperiod=14)
                minus_di = talib.MINUS_DI(arrays['high'], arrays['low'], arrays['close'], timeperiod=14)
                
                result_df['adx'] = adx
                result_df['plus_di'] = plus_di
                result_df['minus_di'] = minus_di
                result_df['adx_strong_trend'] = adx > 25
                result_df['adx_very_strong_trend'] = adx > 40
                result_df['bullish_di'] = plus_di > minus_di
            
            self.logger.info("Calculated trend indicators for %d rows", len(result_df))
            
        except Exception as exc:
            self.logger.error("Error calculating trend indicators: %s", exc)
            
        return result_df
    
    def futures_specific_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate futures-specific indicators including Open Interest analysis.
        
        Args:
            df: DataFrame with 'open_interest' column (if available)
            
        Returns:
            DataFrame with futures-specific indicators added
        """
        result_df = df.copy()
        
        try:
            # Open Interest analysis (if available)
            if 'open_interest' in df.columns:
                oi_data = df['open_interest'].astype(np.float64).values
                
                # OI moving averages using TA-Lib
                oi_sma_10 = talib.SMA(oi_data, timeperiod=10)
                oi_sma_20 = talib.SMA(oi_data, timeperiod=20)
                oi_ema_10 = talib.EMA(oi_data, timeperiod=10)
                
                result_df['oi_sma_10'] = oi_sma_10
                result_df['oi_sma_20'] = oi_sma_20
                result_df['oi_ema_10'] = oi_ema_10
                
                # OI momentum using TA-Lib
                oi_roc = talib.ROC(oi_data, timeperiod=5)
                oi_momentum = talib.MOM(oi_data, timeperiod=3)
                
                result_df['oi_roc_5'] = oi_roc
                result_df['oi_momentum_3'] = oi_momentum
                
                # OI trend analysis
                result_df['oi_increasing'] = oi_sma_10 > oi_sma_20
                result_df['oi_decreasing'] = oi_sma_10 < oi_sma_20
                
                # Large OI changes (potential rollover signals)
                oi_pct_change = pd.Series(oi_data).pct_change()
                result_df['oi_large_increase'] = oi_pct_change > 0.15  # 15% increase
                result_df['oi_large_decrease'] = oi_pct_change < -0.15  # 15% decrease
                result_df['potential_rollover'] = abs(oi_pct_change) > 0.30  # 30% change
                
                # Volume-OI divergence analysis
                if 'volume' in df.columns:
                    # Normalize both for comparison using z-score
                    if len(df) >= 20:
                        volume_series = pd.Series(df['volume'].values)
                        oi_series = pd.Series(oi_data)
                        
                        vol_zscore = (volume_series - volume_series.rolling(20).mean()) / volume_series.rolling(20).std()
                        oi_zscore = (oi_series - oi_series.rolling(20).mean()) / oi_series.rolling(20).std()
                        
                        result_df['vol_oi_divergence'] = abs(vol_zscore - oi_zscore)
                        result_df['vol_oi_extreme_divergence'] = result_df['vol_oi_divergence'] > 2.0
            
            # Contract-specific calculations
            if 'close' in df.columns:
                close_prices = df['close'].astype(np.float64).values
                
                # Daily returns for futures
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
            
            self.logger.info("Calculated futures-specific indicators for %d rows", len(result_df))
            
        except Exception as exc:
            self.logger.error("Error calculating futures indicators: %s", exc)
            
        return result_df
    
    def compute_all_indicators(self, df: pd.DataFrame, 
                             include_futures_indicators: bool = True) -> pd.DataFrame:
        """
        Compute complete suite of technical indicators using TA-Lib.
        
        Args:
            df: DataFrame with OHLCV(I) data
            include_futures_indicators: Whether to include futures-specific indicators
            
        Returns:
            DataFrame with all indicators computed
        """
        if df.empty:
            self.logger.warning("Empty DataFrame provided for indicator calculation")
            return df
        
        self.logger.info("Computing complete TA-Lib indicator suite for %d rows", len(df))
        
        result_df = df.copy()
        
        # Apply all indicator categories
        result_df = self.macd_multi_timeframe(result_df)
        result_df = self.rsi_multi_period(result_df)
        result_df = self.atr_volatility(result_df)
        result_df = self.bollinger_bands(result_df)
        result_df = self.volume_indicators(result_df)
        result_df = self.momentum_oscillators(result_df)
        result_df = self.trend_indicators(result_df)
        
        # Futures-specific indicators
        if include_futures_indicators:
            result_df = self.futures_specific_indicators(result_df)
        
        # Summary statistics
        indicators_added = len(result_df.columns) - len(df.columns)
        self.logger.info("Completed TA-Lib indicator calculation - added %d indicator columns", 
                        indicators_added)
        
        return result_df


# Data validation functions (moved from loaders.py)

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
        logger.info("[WARNING] Found %d OHLC logic violations - fixing...", invalid_count)
        
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
        logger.info("[FIX] Fixing %d negative volume entries...", negative_volumes)
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
        logger.info("[FIX] Removed %d duplicate timestamps...", duplicates_removed)
    
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
        logger.info("[CLEANED] Dropped %d rows with missing price data", dropped_count)
        logger.info("[INFO] Kept %d complete records (%.1f%% retention)", 
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
    logger.info("[VALIDATION] Starting data validation pipeline...")
    
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
    
    logger.info("[SUCCESS] Data validation complete!")
    
    # Convert back to list of dictionaries
    return df.to_dict('records')


# Example usage and testing
if __name__ == "__main__":
    # Example usage with sample data
    import os
    import sys
    
    try:
        # Create sample data for testing
        dates = pd.date_range('2024-01-01', periods=100, freq='5min')
        np.random.seed(42)
        
        # Generate realistic OHLCV data
        close_prices = 100 + np.cumsum(np.random.randn(100) * 0.1)
        high_prices = close_prices + abs(np.random.randn(100) * 0.2)
        low_prices = close_prices - abs(np.random.randn(100) * 0.2)
        open_prices = close_prices + np.random.randn(100) * 0.1
        volumes = np.random.randint(1000, 10000, 100)
        
        sample_df = pd.DataFrame({
            'datetime': dates,
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volumes
        })
        sample_df.set_index('datetime', inplace=True)
        
        print("[INFO] Testing TA-Lib indicators with sample data")
        
        # Initialize indicators
        indicators = TechnicalIndicators()
        
        # Compute all indicators
        result_df = indicators.compute_all_indicators(sample_df, include_futures_indicators=False)
        
        print(f"[SUCCESS] Computed indicators for {len(result_df)} rows")
        print(f"[INFO] Added {len(result_df.columns) - len(sample_df.columns)} indicator columns")
        
        # Show sample results
        indicator_cols = [col for col in result_df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        if indicator_cols:
            print(f"\n[INFO] Sample indicator values (last 5 rows):")
            print(result_df[indicator_cols[:10]].tail().round(4))  # Show first 10 indicators
            
    except ImportError as e:
        print(f"[ERROR] TA-Lib not installed: {e}")
        print("[INFO] Install with: pip install TA-Lib")
        
    except Exception as e:
        print(f"[ERROR] Error testing indicators: {e}")