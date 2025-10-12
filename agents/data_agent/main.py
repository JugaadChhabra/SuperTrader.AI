"""
Main Data Agent Module

Core orchestration module that provides the main API for data agent functionality.
This module focuses on high-level data flow and delegates specific tasks to specialized modules.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests
import pandas as pd
import numpy as np
import yaml
from dotenv import load_dotenv

from .constants import (
    ICICI_API_BASE_URL,
    ICICI_HISTORICAL_DATA_ENDPOINT,
    API_HEADERS_TEMPLATE,
    OHLCVI_COLUMN_MAPPING,
    NUMERIC_COLUMNS,
    DEFAULT_CONFIG_PATH
)
from .validators import validate_futures_data, validate_dataframe_structure
from .futures_manager import generate_futures_symbol
from .feature_builders import build_comprehensive_features

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def init_data_agent(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Initialize the data agent with configuration and environment variables."""
    load_dotenv()
    ctx = {
        "app_key": (config or {}).get("icici", {}).get("app_key")
        or os.getenv("APP_KEY"),
        "api_session_token": (config or {}).get("icici", {}).get("api_session_token")
        or os.getenv("API_SESSION_TOKEN"),
        "initialized_at": datetime.now(timezone.utc).isoformat(),
    }
    logger.info("Data Agent initialized")
    return ctx


def fetch_index_futures_ohlcv(index_symbols: List[str], contract_months: List[str], 
                              interval: str, start: str, end: str,
                              api_keys: Dict[str, str]) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Fetch OHLCVI data for index futures contracts.
    
    Args:
        index_symbols: List of index symbols (e.g., ['NIFTY', 'BANKNIFTY'])
        contract_months: List of contract months (e.g., ['current', 'next', 'far'])
        interval: Time interval ('1minute', '5minute', '15minute', '1day')
        start: Start date in 'YYYY-MM-DD' format
        end: End date in 'YYYY-MM-DD' format  
        api_keys: Dict with 'app_key' and 'api_session_token'
        
    Returns:
        Dict mapping index_symbol -> contract_month -> DataFrame with OHLCVI data
    """
    if not api_keys.get("app_key") or not api_keys.get("api_session_token"):
        raise ValueError("Missing required API keys")
        
    results = {}
    market_config = _load_market_config()
    
    for index_symbol in index_symbols:
        results[index_symbol] = {}
        
        for contract_month in contract_months:
            try:
                # Fetch raw data
                raw_data = _fetch_raw_ohlcv_data(
                    index_symbol, contract_month, interval, start, end, 
                    api_keys, market_config
                )
                
                if raw_data:
                    # Transform and validate
                    df = _transform_ohlcv_response(raw_data, index_symbol, contract_month, interval)
                    validated_df = validate_futures_data(df, f"{index_symbol}_{contract_month}")
                    results[index_symbol][contract_month] = validated_df
                    
                    logger.info("Fetched %d bars for %s (%s)", len(validated_df), 
                              index_symbol, contract_month)
                else:
                    results[index_symbol][contract_month] = pd.DataFrame()
                    
            except Exception as e:
                logger.error("Error fetching %s (%s): %s", index_symbol, contract_month, e)
                results[index_symbol][contract_month] = pd.DataFrame()
    
    return results


def _load_market_config() -> Dict[str, Any]:
    """Load market configuration, with fallback to empty dict."""
    try:
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(script_dir, DEFAULT_CONFIG_PATH)
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning("Market config not found, using default contract naming")
        return {}


def _fetch_raw_ohlcv_data(index_symbol: str, contract_month: str, interval: str, 
                         start: str, end: str, api_keys: Dict[str, str], 
                         market_config: Dict[str, Any]) -> Optional[List[Dict]]:
    """Fetch raw OHLCV data from ICICI API."""
    # Generate futures symbol
    futures_symbol = generate_futures_symbol(index_symbol, contract_month, market_config)
    
    # Prepare API request
    url = f"{ICICI_API_BASE_URL}{ICICI_HISTORICAL_DATA_ENDPOINT}"
    headers = {
        **API_HEADERS_TEMPLATE,
        "X-AppKey": api_keys["app_key"],
        "X-SessionToken": api_keys["api_session_token"]
    }
    
    payload = {
        "stock_code": futures_symbol,
        "exchange_code": "NSE",
        "product_type": "F",  # Futures segment
        "interval": interval,
        "from_date": start,
        "to_date": end
    }
    
    logger.info("Fetching futures OHLCVI for %s (%s) from %s to %s", 
               futures_symbol, contract_month, start, end)
    
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    
    data = response.json()
    
    if data.get("Status") == "Success" and "Success" in data:
        return data["Success"]
    else:
        logger.error("API error for %s (%s): %s", index_symbol, contract_month, data)
        return None


def _transform_ohlcv_response(raw_data: List[Dict], index_symbol: str, 
                             contract_month: str, interval: str) -> pd.DataFrame:
    """Transform raw API response into standardized DataFrame."""
    # Convert to DataFrame
    df = pd.DataFrame(raw_data)
    
    # Standardize column names
    df = df.rename(columns=OHLCVI_COLUMN_MAPPING)
    
    # Ensure timestamp is datetime and set as index
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    
    # Convert numeric columns
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Add metadata
    df.attrs = {
        'symbol': f"{index_symbol}_{contract_month}",
        'index': index_symbol,
        'contract_month': contract_month,
        'interval': interval
    }
    
    return df


def compute_indicators(df: pd.DataFrame, minimal_mode: bool = False) -> pd.DataFrame:
    """Compute technical indicators using TA-Lib functions.
    
    Args:
        df: DataFrame with OHLCV data (expects 'close' column)
        minimal_mode: If True, compute only essential indicators for MVP
        
    Returns:
        DataFrame with TA-Lib indicators added
    """
    if not validate_dataframe_structure(df, ['close']):
        logger.warning("Invalid DataFrame for indicators - missing 'close' column")
        return df
    
    try:
        if minimal_mode:
            # Use minimal indicators for MVP compliance
            from indicators.technical import macd_multi_timeframe, rsi_multi_period, atr_volatility
            
            result_df = df.copy()
            result_df = macd_multi_timeframe(result_df)  # Gets macd_standard
            result_df = rsi_multi_period(result_df)      # Gets rsi_30
            result_df = atr_volatility(result_df)        # Gets atr_14
            
            # Add volume surge (simple version)
            if 'volume' in df.columns and len(df) >= 20:
                volume = df['volume']
                volume_mean = volume.rolling(20).mean()
                volume_std = volume.rolling(20).std()
                result_df['volume_surge'] = ((volume - volume_mean) / volume_std).fillna(0)
            
            logger.info("Computed minimal indicators (MACD, RSI, ATR, volume surge) for %d rows", len(result_df))
        else:
            # Use the comprehensive TA-Lib indicator suite
            from indicators.technical import compute_all_indicators
            result_df = compute_all_indicators(df, include_futures_indicators=False)
            logger.info("Computed full TA-Lib indicators for %d rows", len(result_df))
        
        return result_df
        
    except Exception as exc:
        logger.exception("Error computing indicators: %s", exc)
        return df


def compute_futures_indicators(df: pd.DataFrame, spot_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Compute futures-specific technical indicators using TA-Lib functions.
    
    Args:
        df: DataFrame with futures OHLCVI data
        spot_df: Optional DataFrame with spot index data for basis calculation
        
    Returns:
        DataFrame with TA-Lib futures indicators added
    """
    if not validate_dataframe_structure(df, ['close']):
        logger.warning("Invalid DataFrame for futures indicators")
        return df
    
    try:
        # Use comprehensive TA-Lib indicator suite with futures-specific indicators
        from indicators.technical import compute_all_indicators
        result_df = compute_all_indicators(df, include_futures_indicators=True)
        
        # Add futures-specific features
        from .feature_builders import add_futures_specific_features
        result_df = add_futures_specific_features(result_df, spot_df)
        
        logger.info("Computed TA-Lib futures indicators for %d rows", len(result_df))
        return result_df
        
    except Exception as exc:
        logger.exception("Error computing TA-Lib futures indicators: %s", exc)
        return df


def build_feature_frame(futures_data: Dict[str, pd.DataFrame], 
                       spot_data: Optional[pd.DataFrame] = None,
                       include_indicators: bool = True,
                       minimal_mode: bool = False) -> pd.DataFrame:
    """Build final feature frame for model training from futures data.
    
    Args:
        futures_data: Dict mapping contract -> OHLCVI DataFrame
        spot_data: Optional spot index data for basis calculations
        include_indicators: Whether to compute technical indicators
        minimal_mode: If True, use only essential indicators for MVP
        
    Returns:
        DataFrame with model-ready features
    """
    if not futures_data:
        logger.error("No futures data provided for feature frame")
        return pd.DataFrame()
    
    # Use the most liquid contract (usually current month)
    primary_contract = list(futures_data.keys())[0]
    df = futures_data[primary_contract].copy()
    
    if df.empty:
        logger.error("Empty primary contract data")
        return pd.DataFrame()
    
    try:
        # 1. Add technical indicators if requested
        if include_indicators:
            if spot_data is not None and not minimal_mode:
                df = compute_futures_indicators(df, spot_data)
            else:
                df = compute_indicators(df, minimal_mode=minimal_mode)
        
        # 2. Build comprehensive feature set using the feature builders
        df = build_comprehensive_features(df, futures_data, spot_data, minimal_mode)
        
        # 3. Add metadata
        df.attrs = {
            'primary_contract': primary_contract,
            'feature_count': len([col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'open_interest']]),
            'has_indicators': include_indicators,
            'has_spot_data': spot_data is not None,
            'minimal_mode': minimal_mode
        }
        
        logger.info("Built feature frame with %d features for %d rows", 
                   df.attrs['feature_count'], len(df))
        
        return df
        
    except Exception as exc:
        logger.exception("Error building feature frame: %s", exc)
        return pd.DataFrame()


def fetch_ohlcv(symbols: List[str], interval: str, start: str, end: str, 
                api_keys: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    """Legacy OHLCV fetch function for backward compatibility.
    
    For index futures, use fetch_index_futures_ohlcv() instead.
    """
    logger.warning("Using legacy OHLCV fetch. Consider using fetch_index_futures_ohlcv() for futures.")
    return {}


def shutdown_data_agent():
    """Close sessions, flush buffers."""
    logger.info("Data Agent shutdown")
    pass


# Stub functions for future implementation
def fetch_market_depth(symbol):
    """Placeholder for level-1/level-2 snapshot functionality."""
    pass


def fetch_corporate_actions(symbols, start, end):
    """Placeholder for splits/dividends/earnings functionality."""
    pass


def interpolate_missing(df, method):
    """Placeholder for forward/back fill, stitching functionality."""
    pass


def resample_bars(df, interval):
    """Placeholder for unifying to target timeframe functionality.""" 
    pass