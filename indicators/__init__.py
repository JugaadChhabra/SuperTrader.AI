"""
Technical Indicators Package for SuperTrader.AI

This package provides comprehensive technical analysis indicators
using the industry-standard TA-Lib library for optimal performance.

Modules:
- technical: Core technical indicators using TA-Lib (MACD, RSI, ATR, Volume, etc.)
- features: Feature engineering and ML preprocessing (to be implemented)

Usage:
    from indicators.technical import TechnicalIndicators
    
    indicators = TechnicalIndicators()
    result_df = indicators.compute_all_indicators(df)
"""

from .technical import TechnicalIndicators

__version__ = "1.0.0"
__all__ = ["TechnicalIndicators"]