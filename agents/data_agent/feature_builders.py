"""
Feature Building Module - Phase 4 Enhanced

Handles feature engineering and technical analysis including:
- Price-based features (returns, momentum)
- Volatility features 
- Volume/Open Interest features
- Time-based features
- Market regime features
- Cross-contract features
- Phase 4: Enhanced OI analysis, PCR integration, multi-timeframe confluence
"""

import logging
from typing import Dict, Optional, List, Tuple

import pandas as pd
import numpy as np

from .constants import (
    TRADING_DAYS_PER_YEAR,
    ROLLING_PERIODS,
    VOLATILITY_ANNUALIZATION_FACTOR
)
# Phase 4: Import enhanced technical indicators
from indicators.technical import (
    compute_all_indicators, 
    pcr_indicators, 
    integrate_pcr_with_price,
    multi_timeframe_sync,
    enhanced_oi_indicators
)

logger = logging.getLogger(__name__)


def add_inter_index_correlations(primary_df: pd.DataFrame, secondary_df: pd.DataFrame, 
                                primary_name: str, secondary_name: str,
                                windows: List[int] = [10, 20, 50]) -> pd.DataFrame:
    """Add inter-index correlation features (e.g., NIFTY-BANKNIFTY).
    
    Args:
        primary_df: Primary index DataFrame (e.g., NIFTY)
        secondary_df: Secondary index DataFrame (e.g., BANKNIFTY)
        primary_name: Name of primary index
        secondary_name: Name of secondary index
        windows: Rolling windows for correlation calculation
        
    Returns:
        Primary DataFrame with correlation features added
    """
    if primary_df.empty or secondary_df.empty or 'close' not in primary_df.columns or 'close' not in secondary_df.columns:
        logger.warning("Insufficient data for inter-index correlation between %s and %s", 
                      primary_name, secondary_name)
        return primary_df
    
    result_df = primary_df.copy()
    
    try:
        # Align DataFrames by timestamp
        aligned_secondary = secondary_df.reindex(primary_df.index, method='ffill')
        
        # Calculate returns for both indices
        primary_returns = result_df['close'].pct_change()
        secondary_returns = aligned_secondary['close'].pct_change()
        
        # Rolling correlations for different windows
        for window in windows:
            if len(primary_df) >= window:
                correlation = primary_returns.rolling(window).corr(secondary_returns)
                result_df[f'{primary_name}_{secondary_name}_corr_{window}'] = correlation
                
                # Beta coefficient (sensitivity of primary to secondary)
                covariance = primary_returns.rolling(window).cov(secondary_returns)
                secondary_variance = secondary_returns.rolling(window).var()
                beta = covariance / secondary_variance.replace(0, np.nan)
                result_df[f'{primary_name}_vs_{secondary_name}_beta_{window}'] = beta
        
        # Correlation regime detection
        if f'{primary_name}_{secondary_name}_corr_20' in result_df.columns:
            corr_20 = result_df[f'{primary_name}_{secondary_name}_corr_20']
            
            # High correlation regime (>0.8)
            result_df[f'{primary_name}_{secondary_name}_high_corr_regime'] = (corr_20 > 0.8).astype(int)
            
            # Low correlation regime (<0.4) - diversification opportunity
            result_df[f'{primary_name}_{secondary_name}_low_corr_regime'] = (corr_20 < 0.4).astype(int)
            
            # Negative correlation regime (<-0.2) - unusual for Indian indices
            result_df[f'{primary_name}_{secondary_name}_negative_corr'] = (corr_20 < -0.2).astype(int)
        
        # Relative strength analysis
        price_ratio = result_df['close'] / aligned_secondary['close']
        result_df[f'{primary_name}_vs_{secondary_name}_ratio'] = price_ratio
        
        # Relative strength momentum (20-day)
        rs_momentum = price_ratio / price_ratio.shift(20) - 1
        result_df[f'{primary_name}_vs_{secondary_name}_rs_momentum'] = rs_momentum
        
        # Z-score of relative strength (mean reversion signal)
        price_ratio_ma = price_ratio.rolling(50).mean()
        price_ratio_std = price_ratio.rolling(50).std()
        rs_zscore = (price_ratio - price_ratio_ma) / price_ratio_std
        result_df[f'{primary_name}_vs_{secondary_name}_rs_zscore'] = rs_zscore
        
        # Divergence signals (when correlation breaks down)
        if len(windows) >= 2:
            short_corr = result_df[f'{primary_name}_{secondary_name}_corr_{windows[0]}']
            long_corr = result_df[f'{primary_name}_{secondary_name}_corr_{windows[-1]}']
            
            # Divergence when short-term corr significantly different from long-term
            correlation_divergence = abs(short_corr - long_corr) > 0.3
            result_df[f'{primary_name}_{secondary_name}_corr_divergence'] = correlation_divergence.astype(int)
        
        logger.info("Added inter-index correlation features: %s vs %s", primary_name, secondary_name)
        
    except Exception as e:
        logger.error("Inter-index correlation calculation failed for %s vs %s: %s", 
                    primary_name, secondary_name, e)
    
    return result_df


def add_volatility_adjusted_features(df: pd.DataFrame, target_vol: float = 0.12) -> pd.DataFrame:
    """Add volatility-adjusted return features using EWMA.
    
    Args:
        df: DataFrame with price data
        target_vol: Target volatility for scaling (12% default for intraday)
        
    Returns:
        DataFrame with volatility-adjusted features
    """
    if 'close' not in df.columns or len(df) < 20:
        logger.warning("Insufficient data for volatility adjustment")
        return df
    
    result_df = df.copy()
    
    try:
        # Calculate returns
        returns = result_df['close'].pct_change().dropna()
        
        # EWMA volatility with different decay parameters
        ewma_spans = [10, 20, 50]  # Different lookback periods
        
        for span in ewma_spans:
            # EWMA volatility
            ewma_vol = returns.ewm(span=span).std() * np.sqrt(VOLATILITY_ANNUALIZATION_FACTOR)
            result_df[f'ewma_vol_{span}'] = ewma_vol
            
            # Volatility-adjusted returns (returns scaled to target volatility)
            vol_scalar = target_vol / ewma_vol.replace(0, np.nan)
            vol_adjusted_returns = returns * vol_scalar
            result_df[f'vol_adjusted_returns_{span}'] = vol_adjusted_returns
            
            # Multi-horizon volatility-adjusted returns
            for horizon in [1, 2, 5, 10]:
                horizon_returns = result_df['close'].pct_change(horizon)
                horizon_vol_adj = horizon_returns * vol_scalar
                result_df[f'vol_adj_return_{horizon}d_ewma{span}'] = horizon_vol_adj
        
        # Volatility regime indicators
        current_vol = result_df['ewma_vol_20']
        vol_ma = current_vol.rolling(100).mean()
        
        # Low, normal, high vol regimes
        result_df['low_vol_regime'] = (current_vol < vol_ma * 0.7).astype(int)
        result_df['high_vol_regime'] = (current_vol > vol_ma * 1.5).astype(int)
        result_df['vol_spike_regime'] = (current_vol > vol_ma * 2.0).astype(int)
        
        # Vol-of-vol (volatility of volatility)
        vol_returns = current_vol.pct_change()
        vol_of_vol = vol_returns.rolling(20).std()
        result_df['vol_of_vol'] = vol_of_vol
        
        # Volatility momentum
        vol_momentum = current_vol / current_vol.shift(10) - 1
        result_df['vol_momentum'] = vol_momentum
        
        # Risk-adjusted momentum (return / volatility)
        if f'vol_adjusted_returns_20' in result_df.columns:
            momentum_5d = result_df['close'].pct_change(5)
            risk_adj_momentum = momentum_5d / current_vol.replace(0, np.nan)
            result_df['risk_adjusted_momentum'] = risk_adj_momentum
        
        logger.info("Added volatility-adjusted features with target vol: %.1f%%", target_vol * 100)
        
    except Exception as e:
        logger.error("Volatility adjustment calculation failed: %s", e)
    
    return result_df


def add_pcr_sentiment_features(df: pd.DataFrame, options_data: Optional[Dict] = None) -> pd.DataFrame:
    """Add Put-Call Ratio sentiment features (framework + real data integration).
    
    Args:
        df: DataFrame with price data
        options_data: Optional dict with PCR and options flow data
        
    Returns:
        DataFrame with PCR sentiment features
    """
    result_df = df.copy()
    
    try:
        if options_data and 'pcr' in options_data:
            # Real PCR data integration
            pcr_series = options_data['pcr']
            
            if isinstance(pcr_series, (int, float)):
                # Static PCR value
                result_df['pcr'] = pcr_series
            elif isinstance(pcr_series, pd.Series):
                # Time series PCR data
                aligned_pcr = pcr_series.reindex(result_df.index, method='ffill')
                result_df['pcr'] = aligned_pcr
            
            # PCR-based sentiment signals
            pcr_values = result_df['pcr']
            
            # Standard PCR interpretation thresholds
            result_df['pcr_bullish'] = (pcr_values < 0.7).astype(int)     # Low PCR = bullish
            result_df['pcr_bearish'] = (pcr_values > 1.3).astype(int)     # High PCR = bearish  
            result_df['pcr_extreme_bullish'] = (pcr_values < 0.5).astype(int)  # Very low PCR
            result_df['pcr_extreme_bearish'] = (pcr_values > 1.8).astype(int)  # Very high PCR
            
            # PCR momentum (change in sentiment)
            pcr_momentum = pcr_values.pct_change(5)
            result_df['pcr_momentum'] = pcr_momentum
            
            # PCR mean reversion signal
            pcr_ma = pcr_values.rolling(20).mean()
            pcr_std = pcr_values.rolling(20).std()
            pcr_zscore = (pcr_values - pcr_ma) / pcr_std
            result_df['pcr_zscore'] = pcr_zscore
            result_df['pcr_mean_reversion_signal'] = -np.tanh(pcr_zscore / 2)  # Contrarian signal
            
            # Additional options flow features (if available)
            if 'call_volume' in options_data and 'put_volume' in options_data:
                call_vol = options_data['call_volume']
                put_vol = options_data['put_volume']
                
                # Volume-based PCR
                vol_pcr = put_vol / (call_vol + 1)  # Avoid div by zero
                result_df['volume_pcr'] = vol_pcr
                
                # Options activity surge
                total_options_vol = call_vol + put_vol
                options_vol_ma = total_options_vol.rolling(10).mean() if hasattr(total_options_vol, 'rolling') else total_options_vol
                result_df['options_activity_surge'] = (total_options_vol > options_vol_ma * 1.5).astype(int) if hasattr(total_options_vol, 'rolling') else 0
            
            logger.info("Added real PCR sentiment features")
            
        else:
            # Placeholder implementation for MVP (when PCR data not available)
            # Use price-based proxy signals
            
            if 'close' in result_df.columns and len(result_df) >= 20:
                # Use RSI as proxy for sentiment
                if 'rsi_14' in result_df.columns:
                    rsi = result_df['rsi_14']
                    
                    # Map RSI to PCR-like signals
                    result_df['pcr_proxy_bullish'] = (rsi > 70).astype(int)  # Overbought -> high put demand
                    result_df['pcr_proxy_bearish'] = (rsi < 30).astype(int)  # Oversold -> high call demand
                else:
                    # Calculate simple RSI proxy
                    returns = result_df['close'].pct_change()
                    up_moves = returns.where(returns > 0, 0).rolling(14).mean()
                    down_moves = (-returns.where(returns < 0, 0)).rolling(14).mean()
                    rsi_proxy = 100 - (100 / (1 + up_moves / down_moves.replace(0, 1)))
                    
                    result_df['pcr_proxy_bullish'] = (rsi_proxy > 70).astype(int)
                    result_df['pcr_proxy_bearish'] = (rsi_proxy < 30).astype(int)
                
                # VIX proxy using realized volatility
                if 'realized_vol_20' in result_df.columns:
                    vol = result_df['realized_vol_20']
                    vol_ma = vol.rolling(50).mean()
                    
                    # High volatility -> fear -> high put demand
                    result_df['fear_regime'] = (vol > vol_ma * 1.3).astype(int)
                    result_df['complacency_regime'] = (vol < vol_ma * 0.7).astype(int)
            
            # Default neutral values for missing real PCR
            result_df['pcr'] = 1.0  # Neutral PCR
            result_df['pcr_bullish'] = 0
            result_df['pcr_bearish'] = 0
            
            logger.debug("Added PCR sentiment features (placeholder mode - no real PCR data)")
    
    except Exception as e:
        logger.error("PCR sentiment feature calculation failed: %s", e)
        # Ensure basic columns exist even on error
        result_df['pcr'] = 1.0
        result_df['pcr_bullish'] = 0
        result_df['pcr_bearish'] = 0
    
    return result_df


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


def add_enhanced_oi_analysis(df: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
    """Add enhanced Open Interest analysis for trend strength confirmation.
    
    Args:
        df: DataFrame with OHLCV + OI data
        symbol: Symbol name for logging
        
    Returns:
        DataFrame with enhanced OI features
    """
    if 'open_interest' not in df.columns or df.empty:
        logger.debug("No OI data available for enhanced analysis: %s", symbol)
        return df
    
    result_df = df.copy()
    
    try:
        oi = result_df['open_interest']
        
        # OI trend analysis
        oi_ma_5 = oi.rolling(5).mean()
        oi_ma_20 = oi.rolling(20).mean()
        
        # OI trend signals
        result_df['oi_rising_trend'] = (oi_ma_5 > oi_ma_20).astype(int)
        result_df['oi_falling_trend'] = (oi_ma_5 < oi_ma_20).astype(int)
        
        # OI momentum
        oi_momentum_5 = oi.pct_change(5)
        oi_momentum_10 = oi.pct_change(10)
        
        result_df['oi_momentum_5'] = oi_momentum_5
        result_df['oi_momentum_10'] = oi_momentum_10
        
        # OI surge detection (significant increase)
        oi_std = oi.rolling(20).std()
        oi_mean = oi.rolling(20).mean()
        oi_zscore = (oi - oi_mean) / oi_std
        
        result_df['oi_zscore'] = oi_zscore
        result_df['oi_surge'] = (oi_zscore > 2.0).astype(int)  # 2 sigma surge
        result_df['oi_collapse'] = (oi_zscore < -2.0).astype(int)  # 2 sigma drop
        
        # Price-OI divergence analysis (key trend strength indicator)
        if 'close' in result_df.columns:
            price_direction = (result_df['close'] > result_df['close'].shift(1)).astype(int)  # 1 = up, 0 = down
            oi_direction = (oi > oi.shift(1)).astype(int)  # 1 = increasing, 0 = decreasing
            
            # Trend strength combinations
            # Rising price + Rising OI = Strong bullish trend
            result_df['strong_bullish_trend'] = (price_direction & oi_direction).astype(int)
            
            # Falling price + Rising OI = Strong bearish trend  
            result_df['strong_bearish_trend'] = ((1 - price_direction) & oi_direction).astype(int)
            
            # Rising price + Falling OI = Weak bullish (profit taking)
            result_df['weak_bullish_trend'] = (price_direction & (1 - oi_direction)).astype(int)
            
            # Falling price + Falling OI = Weak bearish (short covering)
            result_df['weak_bearish_trend'] = ((1 - price_direction) & (1 - oi_direction)).astype(int)
            
            # Price-OI divergence score (-1 to +1)
            price_change = result_df['close'].pct_change(5)
            oi_change = oi.pct_change(5)
            
            # Normalize changes
            price_norm = np.sign(price_change) * np.minimum(abs(price_change) * 100, 1)  # Cap at 1%
            oi_norm = np.sign(oi_change) * np.minimum(abs(oi_change) * 100, 1)  # Cap at 1%
            
            # Divergence = difference between normalized price and OI changes
            price_oi_divergence = price_norm - oi_norm
            result_df['price_oi_divergence'] = price_oi_divergence
        
        # Volume-OI relationship analysis
        if 'volume' in result_df.columns:
            # Volume/OI ratio (turnover indicator)
            volume_oi_ratio = result_df['volume'] / (oi + 1)  # Avoid division by zero
            result_df['volume_oi_ratio'] = volume_oi_ratio
            
            # High turnover (high volume relative to OI)
            vo_ratio_ma = volume_oi_ratio.rolling(20).mean()
            result_df['high_turnover'] = (volume_oi_ratio > vo_ratio_ma * 1.5).astype(int)
            
            # Volume surge with OI growth (new money entering)
            volume_surge = (result_df['volume'] > result_df['volume'].rolling(10).mean() * 2)
            oi_growth = (oi_momentum_5 > 0.05)  # 5% OI growth
            result_df['new_money_signal'] = (volume_surge & oi_growth).astype(int)
            
            # Volume surge with OI decline (position unwinding)
            oi_decline = (oi_momentum_5 < -0.05)  # 5% OI decline
            result_df['unwinding_signal'] = (volume_surge & oi_decline).astype(int)
        
        # OI concentration analysis (for market structure)
        # Rolling max OI (to detect concentration points)
        oi_rolling_max = oi.rolling(50).max()
        oi_concentration = oi / oi_rolling_max
        result_df['oi_concentration'] = oi_concentration
        
        # Near maximum OI (high concentration)
        result_df['high_oi_concentration'] = (oi_concentration > 0.9).astype(int)
        
        # OI buildup pattern (sustained growth)
        oi_buildup = (oi > oi.rolling(5).mean()) & (oi.rolling(5).mean() > oi.rolling(10).mean())
        result_df['oi_buildup_pattern'] = oi_buildup.astype(int)
        
        logger.info("Added enhanced OI analysis features for %s", symbol)
        
    except Exception as e:
        logger.error("Enhanced OI analysis failed for %s: %s", symbol, e)
    
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


def add_enhanced_basis_features(futures_df: pd.DataFrame, spot_df: Optional[pd.DataFrame] = None,
                               expiry_date: Optional[str] = None, symbol: str = "") -> pd.DataFrame:
    """Add enhanced basis spread analysis for mean reversion trading.
    
    Args:
        futures_df: Futures DataFrame
        spot_df: Spot index DataFrame
        expiry_date: Contract expiry date for accurate basis calculation
        symbol: Symbol name for logging
        
    Returns:
        DataFrame with enhanced basis features
    """
    if spot_df is None or futures_df.empty or spot_df.empty:
        logger.warning("Insufficient data for basis analysis: %s", symbol)
        return futures_df
    
    result_df = futures_df.copy()
    
    try:
        from .futures_manager import compute_basis
        
        # Use enhanced basis calculation from futures_manager
        if expiry_date:
            basis_analysis = compute_basis(
                futures_df, spot_df, expiry_date, symbol
            )
            
            if not basis_analysis.empty:
                # Merge basis analysis with result DataFrame
                common_columns = ['futures_price', 'spot_price']  # Avoid duplicating price columns
                basis_columns = [col for col in basis_analysis.columns if col not in common_columns]
                
                for col in basis_columns:
                    result_df[col] = basis_analysis[col]
                
                # Additional basis trading signals
                if 'basis_zscore' in result_df.columns:
                    basis_zscore = result_df['basis_zscore']
                    
                    # Mean reversion entry signals
                    result_df['basis_oversold'] = (basis_zscore < -2.0).astype(int)    # Strong buy signal
                    result_df['basis_overbought'] = (basis_zscore > 2.0).astype(int)   # Strong sell signal
                    result_df['basis_entry_long'] = (basis_zscore < -1.5).astype(int)  # Long entry
                    result_df['basis_entry_short'] = (basis_zscore > 1.5).astype(int)  # Short entry
                    
                    # Exit signals (return to mean)
                    result_df['basis_exit_long'] = (basis_zscore > -0.5).astype(int)   # Exit long position
                    result_df['basis_exit_short'] = (basis_zscore < 0.5).astype(int)   # Exit short position
                
                # Basis trend analysis
                if 'basis_pct' in result_df.columns:
                    basis_pct = result_df['basis_pct']
                    
                    # Basis trend (5-day moving average)
                    basis_trend = basis_pct.rolling(5).mean()
                    result_df['basis_trend'] = basis_trend
                    
                    # Basis acceleration (second derivative)
                    basis_acceleration = basis_trend.diff()
                    result_df['basis_acceleration'] = basis_acceleration
                    
                    # Basis volatility (for position sizing)
                    basis_volatility = basis_pct.rolling(20).std()
                    result_df['basis_volatility'] = basis_volatility
                
                # Convergence signals (near expiry)
                if 'days_to_expiry' in result_df.columns:
                    days_to_expiry = result_df['days_to_expiry']
                    
                    # Time decay factor (basis should converge faster near expiry)
                    time_decay_factor = np.exp(-days_to_expiry / 30)  # Exponential decay
                    result_df['basis_time_decay'] = time_decay_factor
                    
                    # Expected convergence rate
                    if 'basis_pct' in result_df.columns:
                        expected_convergence = result_df['basis_pct'] * time_decay_factor
                        result_df['basis_expected_convergence'] = expected_convergence
                        
                        # Convergence deviation (actual vs expected)
                        convergence_deviation = result_df['basis_pct'] - expected_convergence
                        result_df['basis_convergence_deviation'] = convergence_deviation
                
                logger.info("Added enhanced basis features for %s", symbol)
        
        else:
            # Fallback: Simple basis calculation without expiry
            aligned_spot = spot_df.reindex(futures_df.index, method='ffill')
            
            if 'close' in aligned_spot.columns:
                # Basic basis calculation
                basis = result_df['close'] - aligned_spot['close']
                basis_pct = (basis / aligned_spot['close']) * 100
                
                result_df['basis'] = basis
                result_df['basis_pct'] = basis_pct
                
                # Simple mean reversion signals
                basis_ma = basis_pct.rolling(20).mean()
                basis_std = basis_pct.rolling(20).std()
                basis_zscore = (basis_pct - basis_ma) / basis_std
                
                result_df['basis_zscore'] = basis_zscore
                result_df['basis_mean_reversion_signal'] = -np.tanh(basis_zscore / 2)
                
                logger.debug("Added basic basis features for %s (no expiry date)", symbol)
    
    except Exception as e:
        logger.error("Enhanced basis calculation failed for %s: %s", symbol, e)
    
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
    
    # Enhanced basis analysis
    result_df = add_enhanced_basis_features(result_df, spot_df, symbol="futures")
    
    # Contango/Backwardation indicators
    if 'calendar_spread' in result_df.columns:
        result_df['is_contango'] = (result_df['calendar_spread'] > 0).astype(int)
        result_df['is_backwardation'] = (result_df['calendar_spread'] < 0).astype(int)
        
        # Contango strength
        if 'close' in result_df.columns:
            contango_strength = result_df['calendar_spread'] / result_df['close'] * 100
            result_df['contango_strength_pct'] = contango_strength
    
    # Cost of carry analysis
    if 'basis_pct' in result_df.columns and 'days_to_expiry' in result_df.columns:
        # Implied cost of carry (annualized)
        days_to_expiry = result_df['days_to_expiry'].replace(0, 1)  # Avoid division by zero
        implied_carry = (result_df['basis_pct'] / 100) * (365 / days_to_expiry)
        result_df['implied_cost_of_carry'] = implied_carry
        
        # Carry trade opportunity (when implied carry > risk-free rate)
        risk_free_rate = 0.06  # 6% default
        result_df['positive_carry_opportunity'] = (implied_carry > risk_free_rate).astype(int)
        result_df['negative_carry_opportunity'] = (implied_carry < -risk_free_rate/2).astype(int)
    
    logger.debug("Added futures-specific features")
    return result_df


def build_feature_frame(futures_data: Dict[str, pd.DataFrame], 
                       spot_data: Optional[pd.DataFrame] = None,
                       include_indicators: bool = True,
                       minimal_mode: bool = False,
                       config: Optional[Dict] = None) -> pd.DataFrame:
    """Enhanced feature frame building with Phase 3 advanced features.
    
    Args:
        futures_data: Dict of futures contract DataFrames
        spot_data: Optional spot index DataFrame
        include_indicators: Whether to include technical indicators
        minimal_mode: If True, only essential features
        config: Market configuration dict
        
    Returns:
        DataFrame with comprehensive feature set including Phase 3 enhancements
    """
    if not futures_data:
        logger.error("No futures data provided for feature building")
        return pd.DataFrame()
    
    # Get primary contract (usually 'current' or first available)
    primary_key = 'current' if 'current' in futures_data else list(futures_data.keys())[0]
    primary_df = futures_data[primary_key].copy()
    
    if primary_df.empty:
        logger.warning("Primary contract data is empty")
        return pd.DataFrame()
    
    logger.info("Building enhanced feature frame with %d contracts (minimal_mode=%s)", 
               len(futures_data), minimal_mode)
    
    result_df = primary_df.copy()
    
    # Phase 1 Integration: Trading hours validation
    if config and not minimal_mode:
        try:
            from .validators import validate_trading_hours, detect_rollover_gaps
            
            # Apply trading hours validation
            result_df = validate_trading_hours(result_df, primary_key, config)
            
            # Detect rollover gaps for data quality awareness
            result_df = detect_rollover_gaps(result_df, primary_key, config)
            
            logger.debug("Applied Phase 1 data validation to features")
        except Exception as e:
            logger.warning("Phase 1 validation integration failed: %s", e)
    
    # Always add basic features
    result_df = add_price_features(result_df)
    result_df = add_time_features(result_df)
    
    if not minimal_mode:
        # Phase 3.1: Inter-index correlations
        secondary_indices = ['BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY']
        primary_symbol = extract_symbol_name(primary_key)
        
        for secondary in secondary_indices:
            secondary_key = find_matching_contract(futures_data, secondary)
            if secondary_key and secondary != primary_symbol:
                secondary_df = futures_data[secondary_key]
                result_df = add_inter_index_correlations(
                    result_df, secondary_df, primary_symbol, secondary
                )
                logger.debug("Added correlation features: %s vs %s", primary_symbol, secondary)
        
        # Phase 3.2: Volatility-adjusted features
        target_vol = config.get('risk_limits', {}).get('target_volatility', 0.12) if config else 0.12
        result_df = add_volatility_adjusted_features(result_df, target_vol)
        
        # Phase 3.3: Enhanced Open Interest analysis
        result_df = add_enhanced_oi_analysis(result_df, primary_symbol)
        
        # Phase 4.1: Enhanced Technical Indicators with PCR Integration
        pcr_data = config.get('pcr_data') if config else None
        result_df = add_phase4_enhanced_indicators(result_df, pcr_data, primary_symbol)
        
        # Phase 4.2: Multi-timeframe confluence analysis
        if config and config.get('enable_multi_timeframe', True):
            result_df = add_multi_timeframe_features(result_df, futures_data, primary_key)
        
        # Phase 3.4: PCR sentiment features (enhanced in Phase 4)
        options_data = config.get('options_data') if config else None
        result_df = add_pcr_sentiment_features(result_df, options_data)
        
        # Standard feature additions
        result_df = add_volatility_features(result_df)
        result_df = add_volume_features(result_df)
        result_df = add_regime_features(result_df)
        
        # Cross-contract features (calendar spreads, etc.)
        if len(futures_data) > 1:
            result_df = add_cross_contract_features(result_df, futures_data)
        
        # Enhanced futures-specific features with basis analysis
        if spot_data is not None:
            # Get expiry date from config if available
            expiry_date = None
            if config and 'expiry_dates' in config:
                expiry_date = config['expiry_dates'].get(primary_key)
            
            result_df = add_enhanced_basis_features(
                result_df, spot_data, expiry_date, primary_symbol
            )
        
        result_df = add_futures_specific_features(result_df, spot_data)
        
        # Phase 2 Integration: Rollover cost awareness
        if len(futures_data) >= 2 and config:
            try:
                # Add rollover cost estimates as features
                contracts = list(futures_data.keys())
                if len(contracts) >= 2:
                    current_contract = futures_data[contracts[0]]
                    next_contract = futures_data[contracts[1]]
                    
                    if not current_contract.empty and not next_contract.empty:
                        from .futures_manager import track_rollover_costs
                        
                        # Estimate rollover costs for latest available date
                        latest_date = current_contract.index[-1].strftime('%Y-%m-%d')
                        rollover_costs = track_rollover_costs(
                            current_contract.tail(10), next_contract.head(10),
                            latest_date, primary_symbol, config
                        )
                        
                        # Add rollover cost as constant feature (for strategy awareness)
                        result_df['estimated_rollover_cost_bp'] = rollover_costs.get('rollover_cost_bp', 5.0)
                        result_df['rollover_quality_score'] = rollover_costs.get('execution_quality_score', 75.0)
                        
                        logger.debug("Added rollover cost features")
            except Exception as e:
                logger.debug("Rollover cost integration failed: %s", e)
    
    else:
        # Minimal mode: only essential features
        if 'volume' in result_df.columns and len(result_df) >= 20:
            volume = result_df['volume']
            volume_mean = volume.rolling(20).mean()
            volume_std = volume.rolling(20).std()
            result_df['volume_surge'] = ((volume - volume_mean) / volume_std).fillna(0)
        
        # Add basic volatility feature even in minimal mode
        if 'close' in result_df.columns and len(result_df) >= 10:
            returns = result_df['close'].pct_change()
            result_df['realized_vol_10'] = returns.rolling(10).std() * np.sqrt(VOLATILITY_ANNUALIZATION_FACTOR)
    
    # Clean up features
    # Forward-fill NaN values for stability
    numeric_cols = result_df.select_dtypes(include=[np.number]).columns
    result_df[numeric_cols] = result_df[numeric_cols].ffill()
    
    # Remove infinite values
    result_df = result_df.replace([np.inf, -np.inf], np.nan)
    result_df[numeric_cols] = result_df[numeric_cols].fillna(0)
    
    # Feature summary
    original_cols = {'open', 'high', 'low', 'close', 'volume', 'open_interest'}
    feature_cols = [col for col in result_df.columns if col not in original_cols]
    
    logger.info("Built %d features for %s: %d rows, %d total columns", 
               len(feature_cols), primary_symbol, len(result_df), len(result_df.columns))
    
    # Add metadata
    result_df.attrs = {
        'primary_symbol': primary_symbol,
        'feature_count': len(feature_cols),
        'minimal_mode': minimal_mode,
        'include_indicators': include_indicators,
        'contracts_used': list(futures_data.keys()),
        'has_spot_data': spot_data is not None,
        'build_timestamp': pd.Timestamp.now().isoformat()
    }
    
    return result_df


def extract_symbol_name(contract_key: str) -> str:
    """Extract base symbol name from contract key."""
    # Handle keys like 'NIFTY_current', 'BANKNIFTY_next', etc.
    if '_' in contract_key:
        return contract_key.split('_')[0]
    return contract_key.upper()


def find_matching_contract(futures_data: Dict[str, pd.DataFrame], symbol: str) -> Optional[str]:
    """Find contract key matching the given symbol."""
    for key in futures_data.keys():
        if symbol.upper() in key.upper():
            return key
    return None


def build_comprehensive_features(df: pd.DataFrame, 
                               futures_data: Optional[Dict[str, pd.DataFrame]] = None,
                               spot_data: Optional[pd.DataFrame] = None,
                               minimal_mode: bool = False) -> pd.DataFrame:
    """Legacy function - redirects to enhanced build_feature_frame.
    
    Args:
        df: Primary DataFrame
        futures_data: Dict of futures contract data
        spot_data: Spot index data
        minimal_mode: If True, only add essential features
        
    Returns:
        DataFrame with comprehensive feature set
    """
    # Create futures_data dict if not provided
    if futures_data is None:
        futures_data = {'primary': df}
    
    return build_feature_frame(
        futures_data=futures_data,
        spot_data=spot_data,
        minimal_mode=minimal_mode
    )


# =============================================================================
# PHASE 4: ENHANCED TECHNICAL INDICATORS & MULTI-TIMEFRAME ANALYSIS
# =============================================================================

def add_phase4_enhanced_indicators(df: pd.DataFrame, pcr_data: Optional[Dict] = None, 
                                  symbol: str = None) -> pd.DataFrame:
    """
    Phase 4: Add enhanced technical indicators with PCR integration.
    
    Args:
        df: DataFrame with OHLCV data
        pcr_data: Optional PCR data for integration
        symbol: Symbol name for logging
        
    Returns:
        DataFrame with Phase 4 enhanced technical indicators
    """
    result_df = df.copy()
    
    try:
        # Apply comprehensive technical indicators with Phase 4 enhancements
        result_df = compute_all_indicators(
            result_df, 
            include_futures_indicators=True,
            pcr_data=pcr_data,
            symbol=symbol
        )
        
        # Additional Phase 4 specific enhancements
        if 'open_interest' in df.columns:
            result_df = enhanced_oi_indicators(result_df, symbol)
        
        # Add Phase 4 market microstructure features
        result_df = add_microstructure_features(result_df, symbol)
        
        logger.info("Added Phase 4 enhanced indicators for %s: %d total features", 
                   symbol or "contract", len(result_df.columns) - len(df.columns))
        
    except Exception as e:
        logger.error("Phase 4 enhanced indicators failed: %s", e)
    
    return result_df


def add_multi_timeframe_features(df: pd.DataFrame, futures_data: Dict[str, pd.DataFrame], 
                               primary_key: str) -> pd.DataFrame:
    """
    Phase 4: Add multi-timeframe confluence features.
    
    Args:
        df: Primary DataFrame
        futures_data: Dict of futures contract DataFrames
        primary_key: Key for primary contract
        
    Returns:
        DataFrame with multi-timeframe features
    """
    result_df = df.copy()
    
    try:
        # Create synthetic multi-timeframe data from single timeframe
        # In practice, this would use actual different timeframe data
        
        # Simulate 15min and 1H data by resampling
        if isinstance(df.index, pd.DatetimeIndex) or 'timestamp' in df.columns:
            
            # Use timestamp column if index is not datetime
            time_col = df.index if isinstance(df.index, pd.DatetimeIndex) else df['timestamp']
            df_with_time = df.copy()
            if not isinstance(df.index, pd.DatetimeIndex):
                df_with_time = df_with_time.set_index('timestamp')
            
            # Create higher timeframe data
            timeframe_data = {}
            
            # 5min base (original)
            timeframe_data['5min'] = df_with_time
            
            # 15min resample
            if len(df_with_time) >= 10:
                df_15min = df_with_time.resample('15T').agg({
                    'open': 'first',
                    'high': 'max', 
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
                
                if len(df_15min) >= 5:
                    # Add basic indicators to 15min
                    df_15min = compute_all_indicators(df_15min, symbol='15min')
                    timeframe_data['15min'] = df_15min
            
            # 1H resample
            if len(df_with_time) >= 20:
                df_1h = df_with_time.resample('1H').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min', 
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
                
                if len(df_1h) >= 3:
                    # Add basic indicators to 1H
                    df_1h = compute_all_indicators(df_1h, symbol='1H')
                    timeframe_data['1H'] = df_1h
            
            # Synchronize timeframes if we have multiple
            if len(timeframe_data) > 1:
                synchronized_df = multi_timeframe_sync(timeframe_data, base_timeframe='5min')
                
                # Extract only the new multi-timeframe columns
                original_cols = set(result_df.columns)
                new_cols = [col for col in synchronized_df.columns if col not in original_cols]
                
                if new_cols:
                    # Align indices for merging
                    if isinstance(result_df.index, pd.DatetimeIndex):
                        sync_subset = synchronized_df[new_cols].reindex(result_df.index)
                    else:
                        # Use position-based alignment
                        min_len = min(len(result_df), len(synchronized_df))
                        sync_subset = synchronized_df[new_cols].iloc[:min_len]
                        sync_subset.index = result_df.index[:min_len]
                    
                    # Merge new columns
                    result_df = pd.concat([result_df, sync_subset], axis=1)
                    
                    logger.info("Added multi-timeframe features: %d new columns", len(new_cols))
        
    except Exception as e:
        logger.warning("Multi-timeframe feature generation failed: %s", e)
    
    return result_df


def add_microstructure_features(df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
    """
    Add market microstructure features for enhanced market analysis.
    
    Args:
        df: DataFrame with OHLCV data
        symbol: Symbol name for logging
        
    Returns:
        DataFrame with microstructure features
    """
    result_df = df.copy()
    
    try:
        # 1. Price Impact and Efficiency Features
        if all(col in df.columns for col in ['high', 'low', 'close', 'volume']):
            
            # Intraday price range efficiency
            price_range = (df['high'] - df['low']) / df['close']
            result_df['price_range_efficiency'] = price_range
            
            # Price range percentile (relative to recent history)
            result_df['price_range_percentile'] = price_range.rolling(50).rank(pct=True) * 100
            
            # High-Low midpoint vs Close (market maker edge indicator)
            hl_midpoint = (df['high'] + df['low']) / 2
            result_df['close_vs_hl_midpoint'] = (df['close'] - hl_midpoint) / df['close']
        
        # 2. Volume Profile Features
        if 'volume' in df.columns:
            # Volume-weighted price levels
            if 'close' in df.columns:
                volume_weighted_price = (df['close'] * df['volume']).rolling(10).sum() / df['volume'].rolling(10).sum()
                result_df['vwap_10'] = volume_weighted_price
                result_df['price_vs_vwap_10'] = (df['close'] - volume_weighted_price) / df['close']
            
            # Volume clustering (periods of similar volume)
            volume_zscore = (df['volume'] - df['volume'].rolling(20).mean()) / df['volume'].rolling(20).std()
            result_df['volume_zscore'] = volume_zscore
            result_df['volume_cluster'] = (abs(volume_zscore) < 0.5).astype(int)  # Normal volume periods
        
        # 3. Order Flow Imbalance (synthetic)
        if all(col in df.columns for col in ['high', 'low', 'close', 'volume']):
            # Estimate buying/selling pressure from price action
            
            # Up moves (buyers in control)
            up_moves = (df['close'] > (df['high'] + df['low']) / 2).astype(int)
            result_df['buyer_pressure'] = up_moves.rolling(10).mean()
            
            # Price momentum vs volume (efficiency of moves)
            if len(df) >= 5:
                price_momentum = df['close'].pct_change(3)
                volume_momentum = df['volume'].pct_change(3)
                
                # Avoid division by zero
                momentum_efficiency = price_momentum / (abs(volume_momentum) + 1e-6)
                result_df['momentum_efficiency'] = momentum_efficiency
                
                # Strong moves with low volume (potential false breakouts)
                result_df['low_volume_breakout'] = (
                    (abs(price_momentum) > 0.01) & (volume_momentum < 0)
                ).astype(int)
        
        # 4. Liquidity Indicators
        if all(col in df.columns for col in ['high', 'low', 'volume']):
            # Amihud illiquidity measure (price impact per unit volume)
            price_impact = (df['high'] - df['low']) / df['volume'].clip(lower=1)
            result_df['amihud_illiquidity'] = price_impact
            
            # Liquidity percentile
            result_df['liquidity_percentile'] = (1 / price_impact).rolling(50).rank(pct=True) * 100
        
        # 5. Market Regime Microstructure
        if 'close' in df.columns:
            # Price clustering around round numbers (psychological levels)
            close_rounded = np.round(df['close'], -1)  # Round to nearest 10
            result_df['near_round_number'] = (abs(df['close'] - close_rounded) < df['close'] * 0.002).astype(int)
            
            # Intraday momentum persistence
            if len(df) >= 10:
                intraday_returns = df['close'].pct_change()
                momentum_persistence = intraday_returns.rolling(5).apply(
                    lambda x: (x > 0).sum() if len(x) > 0 else 0
                ) / 5
                result_df['momentum_persistence'] = momentum_persistence
        
        # 6. Volatility Microstructure
        if all(col in df.columns for col in ['high', 'low', 'open', 'close']):
            # Garman-Klass volatility estimator (more efficient than close-to-close)
            gk_vol = np.log(df['high']/df['low']) * np.log(df['high']/df['close']) + \
                     np.log(df['low']/df['close']) * np.log(df['low']/df['high'])
            result_df['garman_klass_vol'] = gk_vol
            
            # Overnight gap analysis
            if 'open' in df.columns:
                prev_close = df['close'].shift(1)
                overnight_gap = (df['open'] - prev_close) / prev_close
                result_df['overnight_gap'] = overnight_gap
                result_df['large_overnight_gap'] = (abs(overnight_gap) > 0.005).astype(int)
        
        added_features = len(result_df.columns) - len(df.columns)
        logger.info("Added microstructure features for %s: %d new features", 
                   symbol or "contract", added_features)
        
    except Exception as e:
        logger.error("Microstructure features calculation failed: %s", e)
    
    return result_df