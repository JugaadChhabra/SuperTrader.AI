"""
Technical Indicators Package for SuperTrader.AI

This package provides comprehensive technical analysis indicators
using the industry-standard TA-Lib library for optimal performance.

Modules:
- technical: Core technical indicators using TA-Lib (MACD, RSI, ATR, Volume, etc.)
- features: Feature engineering and ML preprocessing (to be implemented)

Usage:
    from indicators.technical import macd_multi_timeframe, rsi_multi_period
    from indicators.technical import compute_all_indicators
    
    # Individual functions
    df_with_macd = macd_multi_timeframe(df)
    df_with_rsi = rsi_multi_period(df)
    
    # Complete suite
    df_with_all = compute_all_indicators(df)
"""

from .technical import (
    macd_multi_timeframe,
    rsi_multi_period,
    atr_volatility,
    bollinger_bands,
    volume_indicators,
    momentum_oscillators,
    trend_indicators,
    futures_specific_indicators,
    compute_all_indicators,
    validate_and_clean_data
)

__version__ = "1.0.0"
__all__ = [
    "macd_multi_timeframe",
    "rsi_multi_period", 
    "atr_volatility",
    "bollinger_bands",
    "volume_indicators",
    "momentum_oscillators",
    "trend_indicators",
    "futures_specific_indicators",
    "compute_all_indicators",
    "validate_and_clean_data"
]

from .technical import (
    macd_multi_timeframe,
    rsi_multi_period,
    atr_volatility,
    bollinger_bands,
    volume_indicators,
    momentum_oscillators,
    trend_indicators,
    futures_specific_indicators,
    compute_all_indicators,
    validate_and_clean_data
)

__version__ = "1.0.0"
__all__ = [
    "macd_multi_timeframe",
    "rsi_multi_period",
    "atr_volatility", 
    "bollinger_bands",
    "volume_indicators",
    "momentum_oscillators",
    "trend_indicators",
    "futures_specific_indicators",
    "compute_all_indicators",
    "validate_and_clean_data"
]