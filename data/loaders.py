"""
Data Warehouse and Preprocessing for SuperTrader.AI
Handles loading, cleaning, and preprocessing historical OHLCV data from CSV files.

IMPORTANT: This module works with local CSV data warehouse only.
- For LIVE trading data, use agents/data_agent.py ICICI WebSocket
- For HISTORICAL data, this module loads from local CSV files and cleans/preprocesses them

Usage:
- Data Warehouse: CSV files containing historical OHLCV data
- Data Cleaning: Remove outliers, fill gaps, validate schema
- Preprocessing: Generate features, normalize data, prepare for ML
- Backtesting: Load clean datasets for strategy testing
- Model Training: Prepare preprocessed datasets for RL training
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, timezone, timedelta
import warnings
import os

logger = logging.getLogger(__name__)

# OHLCV schema definition - Enhanced for Index Futures
OHLCV_SCHEMA = {
    'timestamp': 'datetime64[ns]',
    'open': 'float64',
    'high': 'float64', 
    'low': 'float64',
    'close': 'float64',
    'volume': 'int64'
}

# Extended schema for Index Futures (includes open interest)
FUTURES_OHLCVI_SCHEMA = {
    'timestamp': 'datetime64[ns]',
    'open': 'float64',
    'high': 'float64', 
    'low': 'float64',
    'close': 'float64',
    'volume': 'int64',
    'open_interest': 'int64'
}

# Contract-specific schema for futures (includes metadata)
FUTURES_CONTRACT_SCHEMA = {
    'timestamp': 'datetime64[ns]',
    'open': 'float64',
    'high': 'float64', 
    'low': 'float64',
    'close': 'float64',
    'volume': 'int64',
    'open_interest': 'int64',
    'contract_name': 'string',           # e.g., "NIFTY24DEC26"
    'expiry_date': 'datetime64[ns]',
    'days_to_expiry': 'int32'
}

REQUIRED_COLUMNS = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
FUTURES_REQUIRED_COLUMNS = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'open_interest']
CONTRACT_REQUIRED_COLUMNS = FUTURES_REQUIRED_COLUMNS + ['contract_name', 'expiry_date']

# Index Futures Specifications
INDEX_FUTURES_SPECS = {
    'NIFTY': {
        'lot_size': 50,
        'tick_size': 0.05,
        'point_value': 50.0,
        'symbol_pattern': r'NIFTY\d{2}[A-Z]{3}\d{2}',  # NIFTY24DEC26
        'underlying': 'NIFTY_50'
    },
    'BANKNIFTY': {
        'lot_size': 15, 
        'tick_size': 0.05,
        'point_value': 15.0,
        'symbol_pattern': r'BANKNIFTY\d{2}[A-Z]{3}\d{2}',
        'underlying': 'BANK_NIFTY'
    },
    'FINNIFTY': {
        'lot_size': 25,
        'tick_size': 0.05,
        'point_value': 25.0,
        'symbol_pattern': r'FINNIFTY\d{2}[A-Z]{3}\d{2}',
        'underlying': 'NIFTY_FIN_SERVICE'
    },
    'MIDCPNIFTY': {
        'lot_size': 75,
        'tick_size': 0.05,
        'point_value': 75.0,
        'symbol_pattern': r'MIDCPNIFTY\d{2}[A-Z]{3}\d{2}',
        'underlying': 'NIFTY_MIDCAP_SELECT'
    },
    'CNXPHARMA': {
        'lot_size': 30,
        'tick_size': 0.05,
        'point_value': 30.0,
        'symbol_pattern': r'CNXPHARMA\d{2}[A-Z]{3}\d{2}',
        'underlying': 'CNX_PHARMA'
    },
    'CNXIT': {
        'lot_size': 50,
        'tick_size': 0.05,
        'point_value': 50.0,
        'symbol_pattern': r'CNXIT\d{2}[A-Z]{3}\d{2}',
        'underlying': 'CNX_IT'
    }
}

# Data Warehouse Configuration
DATA_WAREHOUSE_ROOT = Path("data/warehouse")
SUPPORTED_INTERVALS = ['1m', '5m', '15m', '1h', '1d']

# Data cleaning configuration
CLEANING_CONFIG = {
    'max_price_gap_pct': 20.0,          # Max allowed price gap %
    'max_volume_spike_multiple': 10.0,   # Max volume spike multiple
    'min_price_value': 1.0,             # Minimum valid price
    'max_price_value': 100000.0,        # Maximum valid price  
    'outlier_z_threshold': 4.0,         # Z-score threshold for outliers
    'missing_data_max_gap_minutes': 60, # Max gap to interpolate (minutes)
}


# Data Warehouse Manager
class DataWarehouse:
    """Manages the local CSV data warehouse for historical OHLCV data."""
    
    def __init__(self, warehouse_root: Union[str, Path] = None):
        self.warehouse_root = Path(warehouse_root) if warehouse_root else DATA_WAREHOUSE_ROOT
        self.warehouse_root.mkdir(parents=True, exist_ok=True)
        logger.info(f"Data warehouse initialized at: {self.warehouse_root}")
    
    def get_symbol_path(self, symbol: str, interval: str) -> Path:
        """Get the file path for a symbol's data."""
        return self.warehouse_root / f"{symbol}_{interval}.csv"
    
    def list_available_symbols(self, interval: str = None) -> List[str]:
        """List all symbols available in the warehouse."""
        pattern = "*_*.csv" if not interval else f"*_{interval}.csv"
        files = list(self.warehouse_root.glob(pattern))
        
        symbols = []
        for file in files:
            # Extract symbol from filename (e.g., "RELIANCE_1d.csv" -> "RELIANCE")
            parts = file.stem.split('_')
            if len(parts) >= 2:
                symbol = '_'.join(parts[:-1])  # Handle symbols with underscores
                if not interval or parts[-1] == interval:
                    symbols.append(symbol)
        
        return sorted(list(set(symbols)))
    
    def symbol_exists(self, symbol: str, interval: str) -> bool:
        """Check if symbol data exists in warehouse."""
        return self.get_symbol_path(symbol, interval).exists()
    
    def get_data_info(self, symbol: str, interval: str) -> Dict:
        """Get information about symbol data in warehouse."""
        file_path = self.get_symbol_path(symbol, interval)
        
        if not file_path.exists():
            return {'exists': False}
        
        try:
            # Read just the first and last few rows to get date range
            df_head = pd.read_csv(file_path, nrows=5, parse_dates=['timestamp'])
            df_tail = pd.read_csv(file_path).tail(5)
            df_tail['timestamp'] = pd.to_datetime(df_tail['timestamp'])
            
            return {
                'exists': True,
                'file_size_mb': file_path.stat().st_size / (1024*1024),
                'start_date': df_head['timestamp'].min(),
                'end_date': df_tail['timestamp'].max(),
                'estimated_rows': sum(1 for _ in open(file_path)) - 1  # Subtract header
            }
        except Exception as e:
            logger.error(f"Error reading data info for {symbol}: {e}")
            return {'exists': True, 'error': str(e)}


def load_csv_ohlcv(file_path: Union[str, Path], 
                   symbol: Optional[str] = None,
                   date_column: str = 'timestamp',
                   validate_schema: bool = True) -> pd.DataFrame:
    """
    Load OHLCV data from CSV file with automatic format detection.
    
    Args:
        file_path: Path to CSV file
        symbol: Optional symbol name to add to DataFrame
        date_column: Name of the timestamp column
        validate_schema: Whether to validate OHLCV schema
        
    Returns:
        DataFrame with standardized OHLCV schema
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If required columns missing or invalid data
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    
    logger.info(f"Loading CSV data from {file_path}")
    
    try:
        # Try to read CSV with automatic delimiter detection
        df = pd.read_csv(file_path, parse_dates=[date_column], index_col=None)
        
        # Handle common column name variations
        df = _standardize_column_names(df, date_column)
        
        # Validate required columns exist
        if validate_schema:
            _validate_required_columns(df)
        
        # Set timestamp as index
        if 'timestamp' in df.columns:
            df.set_index('timestamp', inplace=True)
        
        # Add symbol column if provided
        if symbol:
            df['symbol'] = symbol
            
        # Sort by timestamp
        df.sort_index(inplace=True)
        
        # Convert data types
        df = _convert_data_types(df)
        
        logger.info(f"Successfully loaded {len(df)} rows from {file_path}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading CSV {file_path}: {e}")
        raise ValueError(f"Failed to load CSV: {e}")


def load_symbol_data(symbol: str, 
                    interval: str = '1d',
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None,
                    warehouse: Optional[DataWarehouse] = None) -> pd.DataFrame:
    """
    Load symbol data from the data warehouse.
    
    Args:
        symbol: Symbol name (e.g., 'RELIANCE')
        interval: Data interval ('1m', '5m', '15m', '1h', '1d')
        start_date: Optional start date filter ('YYYY-MM-DD')
        end_date: Optional end date filter ('YYYY-MM-DD')
        warehouse: Optional DataWarehouse instance
        
    Returns:
        DataFrame with OHLCV data
    """
    if warehouse is None:
        warehouse = DataWarehouse()
    
    file_path = warehouse.get_symbol_path(symbol, interval)
    
    if not warehouse.symbol_exists(symbol, interval):
        raise FileNotFoundError(f"Symbol {symbol} with interval {interval} not found in warehouse")
    
    # Load the data
    df = load_csv_ohlcv(file_path, symbol=symbol)
    
    # Apply date filters if provided
    if start_date:
        df = df[df.index >= start_date]
    if end_date:
        df = df[df.index <= end_date]
    
    logger.info(f"Loaded {symbol} ({interval}): {len(df)} bars")
    return df


def load_multiple_symbols(symbols: List[str],
                         interval: str = '1d',
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None,
                         combine: bool = False) -> Union[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Load data for multiple symbols from warehouse.
    
    Args:
        symbols: List of symbol names
        interval: Data interval
        start_date: Optional start date filter
        end_date: Optional end date filter
        combine: If True, return single DataFrame with 'symbol' column
        
    Returns:
        Dict of symbol -> DataFrame or single combined DataFrame
    """
    warehouse = DataWarehouse()
    results = {}
    failed_symbols = []
    
    for symbol in symbols:
        try:
            df = load_symbol_data(symbol, interval, start_date, end_date, warehouse)
            results[symbol] = df
        except Exception as e:
            logger.error(f"Failed to load {symbol}: {e}")
            failed_symbols.append(symbol)
    
    if failed_symbols:
        logger.warning(f"Failed to load symbols: {failed_symbols}")
    
    if combine and results:
        # Combine all DataFrames with symbol column
        combined_frames = []
        for symbol, df in results.items():
            df_copy = df.copy()
            df_copy['symbol'] = symbol
            combined_frames.append(df_copy)
        
        combined_df = pd.concat(combined_frames, ignore_index=False)
        combined_df.sort_index(inplace=True)
        logger.info(f"Combined {len(results)} symbols: {len(combined_df)} total bars")
        return combined_df
    
    return results


# Data Cleaning Functions
def clean_ohlcv_data(df: pd.DataFrame, 
                     config: Optional[Dict] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Clean OHLCV data by removing outliers, fixing gaps, and validating consistency.
    
    Args:
        df: Raw OHLCV DataFrame
        config: Optional cleaning configuration
        
    Returns:
        Tuple of (cleaned_df, cleaning_report)
    """
    if config is None:
        config = CLEANING_CONFIG
    
    cleaning_report = {
        'original_rows': len(df),
        'issues_found': [],
        'issues_fixed': [],
        'final_rows': 0
    }
    
    df_clean = df.copy()
    
    # 1. Validate OHLC consistency
    df_clean, ohlc_issues = _fix_ohlc_consistency(df_clean)
    if ohlc_issues:
        cleaning_report['issues_fixed'].extend(ohlc_issues)
    
    # 2. Remove price outliers
    df_clean, price_outliers = _remove_price_outliers(df_clean, config)
    if price_outliers:
        cleaning_report['issues_fixed'].append(f"Removed {price_outliers} price outliers")
    
    # 3. Remove volume spikes
    df_clean, volume_spikes = _remove_volume_spikes(df_clean, config)
    if volume_spikes:
        cleaning_report['issues_fixed'].append(f"Removed {volume_spikes} volume spikes")
    
    # 4. Handle missing data
    df_clean, missing_fixed = _handle_missing_data(df_clean, config)
    if missing_fixed:
        cleaning_report['issues_fixed'].append(f"Fixed {missing_fixed} missing data points")
    
    # 5. Remove invalid price ranges
    initial_count = len(df_clean)
    price_mask = (
        (df_clean['open'] >= config['min_price_value']) & 
        (df_clean['open'] <= config['max_price_value']) &
        (df_clean['high'] >= config['min_price_value']) & 
        (df_clean['high'] <= config['max_price_value']) &
        (df_clean['low'] >= config['min_price_value']) & 
        (df_clean['low'] <= config['max_price_value']) &
        (df_clean['close'] >= config['min_price_value']) & 
        (df_clean['close'] <= config['max_price_value'])
    )
    df_clean = df_clean[price_mask]
    invalid_prices = initial_count - len(df_clean)
    if invalid_prices:
        cleaning_report['issues_fixed'].append(f"Removed {invalid_prices} invalid price ranges")
    
    cleaning_report['final_rows'] = len(df_clean)
    
    logger.info(f"Data cleaning complete: {cleaning_report['original_rows']} -> {cleaning_report['final_rows']} rows")
    
    return df_clean, cleaning_report


def detect_data_gaps(df: pd.DataFrame, expected_interval_minutes: int = 1440) -> List[Dict]:
    """
    Detect gaps in time series data.
    
    Args:
        df: OHLCV DataFrame with timestamp index
        expected_interval_minutes: Expected interval in minutes (1440 for daily)
        
    Returns:
        List of gap information dictionaries
    """
    gaps = []
    
    if len(df) < 2:
        return gaps
    
    time_diffs = df.index.to_series().diff()
    expected_delta = timedelta(minutes=expected_interval_minutes)
    
    # Find gaps larger than expected
    gap_mask = time_diffs > expected_delta * 1.5  # Allow 50% tolerance
    gap_indices = df.index[gap_mask]
    
    for gap_end in gap_indices:
        gap_start_idx = df.index.get_loc(gap_end) - 1
        if gap_start_idx >= 0:
            gap_start = df.index[gap_start_idx]
            gap_duration = gap_end - gap_start
            
            gaps.append({
                'start': gap_start,
                'end': gap_end,
                'duration': gap_duration,
                'missing_periods': int(gap_duration.total_seconds() / (expected_interval_minutes * 60)) - 1
            })
    
    return gaps


def generate_data_quality_report(df: pd.DataFrame, symbol: str) -> Dict:
    """
    Generate comprehensive data quality report for a symbol.
    
    Args:
        df: OHLCV DataFrame
        symbol: Symbol name
        
    Returns:
        Data quality report dictionary
    """
    report = {
        'symbol': symbol,
        'total_rows': len(df),
        'date_range': {
            'start': df.index.min(),
            'end': df.index.max(),
            'days': (df.index.max() - df.index.min()).days
        },
        'missing_data': {},
        'data_consistency': {},
        'outliers': {},
        'completeness_score': 0.0
    }
    
    # Check for missing values
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            missing_count = df[col].isna().sum()
            report['missing_data'][col] = {
                'count': missing_count,
                'percentage': (missing_count / len(df)) * 100
            }
    
    # Check OHLC consistency
    ohlc_issues = 0
    if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        # High should be >= max(open, close)
        high_issues = (df['high'] < np.maximum(df['open'], df['close'])).sum()
        # Low should be <= min(open, close)  
        low_issues = (df['low'] > np.minimum(df['open'], df['close'])).sum()
        ohlc_issues = high_issues + low_issues
    
    report['data_consistency'] = {
        'ohlc_violations': ohlc_issues,
        'ohlc_violations_pct': (ohlc_issues / len(df)) * 100
    }
    
    # Detect outliers using z-score
    if 'close' in df.columns:
        returns = df['close'].pct_change().dropna()
        if len(returns) > 0:
            z_scores = np.abs((returns - returns.mean()) / returns.std())
            outliers = (z_scores > CLEANING_CONFIG['outlier_z_threshold']).sum()
            report['outliers'] = {
                'price_outliers': outliers,
                'price_outliers_pct': (outliers / len(returns)) * 100
            }
    
    # Calculate completeness score
    missing_pct = sum(col['percentage'] for col in report['missing_data'].values()) / len(report['missing_data'])
    consistency_pct = report['data_consistency']['ohlc_violations_pct']
    outlier_pct = report['outliers'].get('price_outliers_pct', 0)
    
    report['completeness_score'] = max(0, 100 - missing_pct - consistency_pct - (outlier_pct * 0.5))
    
    return report


# Helper functions for data cleaning
def _standardize_column_names(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """Standardize column names to OHLCV schema."""
    df = df.copy()
    
    # Common column name mappings
    column_mappings = {
        'date': 'timestamp',
        'time': 'timestamp', 
        'datetime': 'timestamp',
        'ts': 'timestamp',
        date_column: 'timestamp',
        'o': 'open',
        'h': 'high',
        'l': 'low', 
        'c': 'close',
        'adj_close': 'close',
        'adj close': 'close',
        'vol': 'volume',
        'v': 'volume',
        'quantity': 'volume'
    }
    
    # Apply mappings (case-insensitive)
    for old_col in df.columns:
        old_col_lower = old_col.lower().strip()
        if old_col_lower in column_mappings:
            new_col = column_mappings[old_col_lower]
            if old_col != new_col:
                df.rename(columns={old_col: new_col}, inplace=True)
                logger.debug(f"Renamed column '{old_col}' -> '{new_col}'")
    
    return df


def _validate_required_columns(df: pd.DataFrame) -> None:
    """Validate that DataFrame has required OHLCV columns."""
    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for empty DataFrame
    if df.empty:
        raise ValueError("DataFrame is empty")


def _convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convert columns to appropriate data types."""
    df = df.copy()
    
    # Convert price columns to float64
    price_columns = ['open', 'high', 'low', 'close']
    for col in price_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert volume to int64 (handle NaN as 0)
    if 'volume' in df.columns:
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype('int64')
    
    # Ensure timestamp index is datetime
    if df.index.name == 'timestamp':
        df.index = pd.to_datetime(df.index)
    
    return df


def _fix_ohlc_consistency(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Fix OHLC consistency issues."""
    df_fixed = df.copy()
    issues_fixed = []
    
    if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        # Fix high values that are too low
        high_mask = df_fixed['high'] < np.maximum(df_fixed['open'], df_fixed['close'])
        if high_mask.any():
            df_fixed.loc[high_mask, 'high'] = np.maximum(
                df_fixed.loc[high_mask, 'open'], 
                df_fixed.loc[high_mask, 'close']
            )
            issues_fixed.append(f"Fixed {high_mask.sum()} high values")
        
        # Fix low values that are too high
        low_mask = df_fixed['low'] > np.minimum(df_fixed['open'], df_fixed['close'])
        if low_mask.any():
            df_fixed.loc[low_mask, 'low'] = np.minimum(
                df_fixed.loc[low_mask, 'open'], 
                df_fixed.loc[low_mask, 'close']
            )
            issues_fixed.append(f"Fixed {low_mask.sum()} low values")
    
    return df_fixed, issues_fixed


def _remove_price_outliers(df: pd.DataFrame, config: Dict) -> Tuple[pd.DataFrame, int]:
    """Remove price outliers using z-score method."""
    df_clean = df.copy()
    outliers_removed = 0
    
    if 'close' in df.columns and len(df) > 10:
        returns = df['close'].pct_change().dropna()
        
        if len(returns) > 0:
            z_scores = np.abs((returns - returns.mean()) / returns.std())
            outlier_mask = z_scores > config['outlier_z_threshold']
            
            # Map back to original DataFrame indices
            outlier_indices = returns[outlier_mask].index
            
            # Remove outliers
            df_clean = df_clean.drop(outlier_indices, errors='ignore')
            outliers_removed = len(outlier_indices)
    
    return df_clean, outliers_removed


def _remove_volume_spikes(df: pd.DataFrame, config: Dict) -> Tuple[pd.DataFrame, int]:
    """Remove volume spikes."""
    df_clean = df.copy()
    spikes_removed = 0
    
    if 'volume' in df.columns and len(df) > 10:
        # Calculate rolling average volume
        volume_avg = df['volume'].rolling(window=20, min_periods=5).mean()
        
        # Find volume spikes
        spike_mask = df['volume'] > (volume_avg * config['max_volume_spike_multiple'])
        spike_mask = spike_mask.fillna(False)
        
        # Remove spikes
        df_clean = df_clean[~spike_mask]
        spikes_removed = spike_mask.sum()
    
    return df_clean, spikes_removed


def _handle_missing_data(df: pd.DataFrame, config: Dict) -> Tuple[pd.DataFrame, int]:
    """Handle missing data through forward fill and interpolation."""
    df_filled = df.copy()
    missing_fixed = 0
    
    # Count initial missing values
    initial_missing = df_filled.isna().sum().sum()
    
    # Forward fill small gaps
    df_filled = df_filled.fillna(method='ffill', limit=3)
    
    # Interpolate remaining small gaps
    df_filled = df_filled.interpolate(method='linear', limit=5)
    
    # Count remaining missing values
    final_missing = df_filled.isna().sum().sum()
    missing_fixed = initial_missing - final_missing
    
    # Drop any remaining rows with NaN values
    df_filled = df_filled.dropna()
    
    return df_filled, missing_fixed


# Index Futures Specific Functions
def validate_futures_contract_name(contract_name: str) -> Dict[str, Union[str, datetime]]:
    """
    Validate and parse futures contract name.
    
    Args:
        contract_name: Contract name like "NIFTY24DEC26"
        
    Returns:
        Dict with parsed contract information
        
    Raises:
        ValueError: If contract name format is invalid
    """
    import re
    from datetime import datetime
    
    # Contract naming patterns
    patterns = {
        'monthly': r'([A-Z]+)(\d{2})([A-Z]{3})(\d{2})',      # NIFTY24DEC26
        'weekly': r'([A-Z]+)(\d{2})([A-Z]{3})(\d{1})W(\d{2})'  # NIFTY24DEC1W26
    }
    
    contract_info = {}
    
    for contract_type, pattern in patterns.items():
        match = re.match(pattern, contract_name.upper())
        if match:
            if contract_type == 'monthly':
                index, year, month, day = match.groups()
                contract_info = {
                    'index': index,
                    'year': 2000 + int(year),
                    'month': month,
                    'day': int(day),
                    'contract_type': 'monthly',
                    'is_weekly': False
                }
            else:  # weekly
                index, year, month, week, day = match.groups()
                contract_info = {
                    'index': index,
                    'year': 2000 + int(year),
                    'month': month,
                    'week': int(week),
                    'day': int(day),
                    'contract_type': 'weekly',
                    'is_weekly': True
                }
            
            # Calculate expiry date
            month_map = {
                'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
            }
            
            try:
                expiry_date = datetime(
                    contract_info['year'], 
                    month_map[contract_info['month']], 
                    contract_info['day']
                )
                contract_info['expiry_date'] = expiry_date
                contract_info['days_to_expiry'] = (expiry_date - datetime.now()).days
                
            except ValueError as e:
                raise ValueError(f"Invalid date in contract {contract_name}: {e}")
            
            # Validate index exists
            if index not in INDEX_FUTURES_SPECS:
                raise ValueError(f"Unknown index: {index}")
            
            contract_info['specifications'] = INDEX_FUTURES_SPECS[index]
            return contract_info
    
    raise ValueError(f"Invalid contract name format: {contract_name}")


def calculate_notional_value(index_price: float, contract_name: str) -> float:
    """
    Calculate notional value of index futures position.
    
    Args:
        index_price: Current index price/level
        contract_name: Contract name to get specifications
        
    Returns:
        Notional value in rupees
    """
    contract_info = validate_futures_contract_name(contract_name)
    specs = contract_info['specifications']
    
    # Notional Value = Index Price × Lot Size × Point Value
    # For index futures: Point Value = Lot Size (e.g., NIFTY has lot size 50 and point value ₹50)
    notional_value = index_price * specs['lot_size']
    
    return notional_value


def calculate_futures_margin(contract_name: str, index_price: float, 
                           margin_type: str = 'NRML') -> Dict[str, float]:
    """
    Calculate margin requirements for index futures position.
    
    Args:
        contract_name: Futures contract name
        index_price: Current index level
        margin_type: 'NRML' (overnight) or 'MIS' (intraday)
        
    Returns:
        Dict with margin breakdown
    """
    contract_info = validate_futures_contract_name(contract_name)
    index = contract_info['index']
    notional_value = calculate_notional_value(index_price, contract_name)
    
    # Get margin percentages from market config (simplified here)
    margin_config = {
        'NIFTY': {'NRML': 10.0, 'MIS': 5.0},
        'BANKNIFTY': {'NRML': 12.0, 'MIS': 6.0},
        'FINNIFTY': {'NRML': 11.0, 'MIS': 5.5},
        'MIDCPNIFTY': {'NRML': 13.0, 'MIS': 6.5},
        'CNXPHARMA': {'NRML': 15.0, 'MIS': 7.0},
        'CNXIT': {'NRML': 12.0, 'MIS': 6.0}
    }
    
    margin_pct = margin_config.get(index, {}).get(margin_type, 10.0)
    
    # Calculate margin components
    span_margin = notional_value * (margin_pct * 0.7) / 100  # 70% is typically SPAN
    exposure_margin = notional_value * (margin_pct * 0.3) / 100  # 30% is exposure
    total_margin = span_margin + exposure_margin
    
    return {
        'notional_value': notional_value,
        'span_margin': span_margin,
        'exposure_margin': exposure_margin,
        'total_margin': total_margin,
        'margin_percentage': margin_pct,
        'margin_type': margin_type
    }


def load_futures_csv_ohlcvi(file_path: Union[str, Path], 
                           symbol: Optional[str] = None,
                           validate_futures: bool = True) -> pd.DataFrame:
    """
    Load OHLCVI data for index futures from CSV file.
    
    Args:
        file_path: Path to CSV file
        symbol: Optional symbol name
        validate_futures: Whether to validate futures-specific data
        
    Returns:
        DataFrame with OHLCVI data for futures
    """
    df = load_csv_ohlcv(file_path, symbol, validate_schema=False)
    
    # Check if open_interest column exists
    if 'open_interest' not in df.columns:
        logger.warning(f"No open_interest column found in {file_path}")
        df['open_interest'] = 0  # Add dummy OI column
    
    # Validate futures schema
    if validate_futures:
        _validate_futures_columns(df)
    
    # Convert open interest to int64
    if 'open_interest' in df.columns:
        df['open_interest'] = pd.to_numeric(df['open_interest'], errors='coerce').fillna(0).astype('int64')
    
    # Add contract metadata if symbol provided and matches futures pattern
    if symbol and validate_futures:
        try:
            contract_info = validate_futures_contract_name(symbol)
            df['contract_name'] = symbol
            df['expiry_date'] = contract_info['expiry_date']
            df['days_to_expiry'] = (contract_info['expiry_date'] - pd.to_datetime(df.index)).dt.days
            
        except ValueError:
            logger.warning(f"Symbol {symbol} doesn't match futures contract pattern")
    
    logger.info(f"Loaded futures data: {len(df)} rows from {file_path}")
    return df


def _validate_futures_columns(df: pd.DataFrame) -> None:
    """Validate DataFrame has required futures columns."""
    missing_columns = set(FUTURES_REQUIRED_COLUMNS) - set(df.columns)
    
    if missing_columns:
        raise ValueError(f"Missing required futures columns: {missing_columns}")
    
    if df.empty:
        raise ValueError("DataFrame is empty")


def detect_rollover_gaps(df: pd.DataFrame, contract_column: str = 'contract_name') -> List[Dict]:
    """
    Detect rollover gaps in continuous futures data.
    
    Args:
        df: DataFrame with multiple contracts
        contract_column: Column containing contract names
        
    Returns:
        List of detected rollover points
    """
    rollover_points = []
    
    if contract_column not in df.columns:
        return rollover_points
    
    # Group by contract and find transitions
    contract_changes = df[contract_column].ne(df[contract_column].shift()).cumsum()
    
    for i in range(1, contract_changes.max() + 1):
        # Find transition point
        transition_idx = df[contract_changes == i].index[0]
        prev_idx = df[contract_changes == i - 1].index[-1]
        
        if transition_idx in df.index and prev_idx in df.index:
            old_contract = df.loc[prev_idx, contract_column]
            new_contract = df.loc[transition_idx, contract_column]
            
            # Calculate price gap
            old_price = df.loc[prev_idx, 'close']
            new_price = df.loc[transition_idx, 'open']
            price_gap_pct = ((new_price - old_price) / old_price) * 100
            
            rollover_points.append({
                'date': transition_idx,
                'old_contract': old_contract,
                'new_contract': new_contract,
                'old_price': old_price,
                'new_price': new_price,
                'price_gap_pct': price_gap_pct
            })
    
    return rollover_points


def build_continuous_contract(index_name: str, contracts_data: Dict[str, pd.DataFrame], 
                             rollover_schedule: Dict[str, str], 
                             rollover_days_before: int = 5) -> pd.DataFrame:
    """
    Build continuous contract from individual contract data.
    
    Args:
        index_name: Name of the index (e.g., 'NIFTY', 'BANKNIFTY')
        contracts_data: Dict mapping contract symbols to their OHLCVI DataFrames
        rollover_schedule: Dict mapping contract symbols to expiry dates
        rollover_days_before: Days before expiry to rollover
        
    Returns:
        DataFrame with continuous contract data
    """
    if not contracts_data:
        raise ValueError("No contract data provided")
    
    # Sort contracts by expiry date
    sorted_contracts = sorted(rollover_schedule.items(), 
                            key=lambda x: pd.to_datetime(x[1]))
    
    continuous_data = []
    active_contract = None
    rollover_adjustments = []
    
    # Get full date range
    all_dates = []
    for df in contracts_data.values():
        if not df.empty:
            all_dates.extend(df.index.tolist())
    
    if not all_dates:
        return pd.DataFrame()
    
    date_range = pd.date_range(min(all_dates), max(all_dates), freq='D')
    
    for current_date in date_range:
        # Find active contract for current date
        new_active_contract = None
        
        for contract, expiry_date in sorted_contracts:
            expiry_dt = pd.to_datetime(expiry_date)
            rollover_date = expiry_dt - pd.Timedelta(days=rollover_days_before)
            
            if current_date <= rollover_date and contract in contracts_data:
                # Check if we have data for this date
                contract_df = contracts_data[contract]
                if current_date.strftime('%Y-%m-%d') in contract_df.index.strftime('%Y-%m-%d'):
                    new_active_contract = contract
                    break
        
        # Handle contract switch
        if new_active_contract != active_contract and active_contract is not None:
            # Record rollover
            old_df = contracts_data.get(active_contract)
            new_df = contracts_data.get(new_active_contract)
            
            if old_df is not None and new_df is not None:
                # Get prices at rollover date
                old_data = old_df[old_df.index.strftime('%Y-%m-%d') == current_date.strftime('%Y-%m-%d')]
                new_data = new_df[new_df.index.strftime('%Y-%m-%d') == current_date.strftime('%Y-%m-%d')]
                
                if not old_data.empty and not new_data.empty:
                    old_price = old_data['close'].iloc[0]
                    new_price = new_data['close'].iloc[0]
                    adjustment_factor = new_price / old_price if old_price != 0 else 1.0
                    
                    rollover_adjustments.append({
                        'date': current_date,
                        'from_contract': active_contract,
                        'to_contract': new_active_contract,
                        'adjustment_factor': adjustment_factor,
                        'price_gap': new_price - old_price
                    })
                    
                    logger.info(f"Rollover detected: {active_contract} -> {new_active_contract} "
                              f"on {current_date}, adjustment: {adjustment_factor:.4f}")
        
        active_contract = new_active_contract
        
        # Get data for active contract
        if active_contract and active_contract in contracts_data:
            contract_df = contracts_data[active_contract]
            day_data = contract_df[contract_df.index.strftime('%Y-%m-%d') == current_date.strftime('%Y-%m-%d')]
            
            if not day_data.empty:
                # Apply rollover adjustments to maintain price continuity
                adjusted_data = day_data.copy()
                
                # Apply all previous rollover adjustments
                for adj in rollover_adjustments:
                    if current_date >= adj['date']:
                        # Adjust OHLC prices
                        for col in ['open', 'high', 'low', 'close']:
                            if col in adjusted_data.columns:
                                adjusted_data[col] = adjusted_data[col] / adj['adjustment_factor']
                
                # Add metadata
                for _, row in adjusted_data.iterrows():
                    continuous_row = row.copy()
                    continuous_row['active_contract'] = active_contract
                    continuous_row['rollover_adjusted'] = len(rollover_adjustments) > 0
                    
                    continuous_data.append(continuous_row)
    
    if not continuous_data:
        return pd.DataFrame()
    
    # Create continuous DataFrame
    continuous_df = pd.DataFrame(continuous_data)
    continuous_df.index.name = 'timestamp'
    
    # Add continuous contract metadata
    continuous_df.attrs['index_name'] = index_name
    continuous_df.attrs['rollover_adjustments'] = rollover_adjustments
    continuous_df.attrs['construction_method'] = 'calendar_rollover'
    continuous_df.attrs['rollover_days_before'] = rollover_days_before
    
    logger.info(f"Built continuous contract for {index_name}: {len(continuous_df)} bars, "
              f"{len(rollover_adjustments)} rollovers")
    
    return continuous_df


def generate_rollover_calendar(index_name: str, start_date: str, end_date: str) -> Dict[str, str]:
    """
    Generate rollover calendar for index futures contracts.
    
    Args:
        index_name: Name of the index (e.g., 'NIFTY', 'BANKNIFTY')
        start_date: Start date for calendar generation
        end_date: End date for calendar generation
        
    Returns:
        Dict mapping contract symbols to their expiry dates
    """
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    rollover_calendar = {}
    current_date = start_dt
    
    while current_date <= end_dt:
        # NSE futures expire on last Thursday of the month
        year = current_date.year
        month = current_date.month
        
        # Find last Thursday of the month
        last_day = pd.Timestamp(year, month, 1) + pd.DateOffset(months=1) - pd.Timedelta(days=1)
        
        # Find last Thursday
        while last_day.weekday() != 3:  # Thursday is 3
            last_day -= pd.Timedelta(days=1)
        
        # Generate contract symbol
        month_codes = {
            1: 'JAN', 2: 'FEB', 3: 'MAR', 4: 'APR',
            5: 'MAY', 6: 'JUN', 7: 'JUL', 8: 'AUG',
            9: 'SEP', 10: 'OCT', 11: 'NOV', 12: 'DEC'
        }
        
        month_code = month_codes[month]
        year_code = str(year)[2:]  # Last 2 digits
        
        contract_symbol = f"{index_name}{year_code}{month_code}{last_day.day}"
        rollover_calendar[contract_symbol] = last_day.strftime('%Y-%m-%d')
        
        # Move to next month
        if month == 12:
            current_date = pd.Timestamp(year + 1, 1, 1)
        else:
            current_date = pd.Timestamp(year, month + 1, 1)
    
    logger.info(f"Generated rollover calendar for {index_name}: {len(rollover_calendar)} contracts")
    return rollover_calendar


def detect_contract_gaps(contracts_data: Dict[str, pd.DataFrame], 
                        rollover_schedule: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    Detect gaps in contract data coverage.
    
    Args:
        contracts_data: Dict mapping contract symbols to DataFrames
        rollover_schedule: Dict mapping contract symbols to expiry dates
        
    Returns:
        List of detected gaps with details
    """
    gaps = []
    
    # Sort contracts by expiry
    sorted_contracts = sorted(rollover_schedule.items(), 
                            key=lambda x: pd.to_datetime(x[1]))
    
    for i, (contract, expiry_date) in enumerate(sorted_contracts[:-1]):
        current_df = contracts_data.get(contract)
        next_contract, next_expiry = sorted_contracts[i + 1]
        next_df = contracts_data.get(next_contract)
        
        if current_df is None or next_df is None:
            gaps.append({
                'type': 'missing_contract_data',
                'contract': contract if current_df is None else next_contract,
                'expiry_date': expiry_date if current_df is None else next_expiry,
                'severity': 'HIGH'
            })
            continue
        
        if current_df.empty or next_df.empty:
            gaps.append({
                'type': 'empty_contract_data',
                'contract': contract if current_df.empty else next_contract,
                'expiry_date': expiry_date if current_df.empty else next_expiry,
                'severity': 'HIGH'
            })
            continue
        
        # Check for date gaps around expiry
        expiry_dt = pd.to_datetime(expiry_date)
        rollover_window = pd.date_range(expiry_dt - pd.Timedelta(days=7), 
                                      expiry_dt + pd.Timedelta(days=2))
        
        # Check current contract data near expiry
        current_dates = set(current_df.index.strftime('%Y-%m-%d'))
        next_dates = set(next_df.index.strftime('%Y-%m-%d'))
        
        missing_current = []
        missing_next = []
        
        for date in rollover_window:
            date_str = date.strftime('%Y-%m-%d')
            
            # Skip weekends
            if date.weekday() >= 5:
                continue
            
            if date <= expiry_dt and date_str not in current_dates:
                missing_current.append(date_str)
            elif date > expiry_dt and date_str not in next_dates:
                missing_next.append(date_str)
        
        if missing_current:
            gaps.append({
                'type': 'rollover_gap',
                'contract': contract,
                'missing_dates': missing_current,
                'expiry_date': expiry_date,
                'severity': 'MEDIUM'
            })
        
        if missing_next:
            gaps.append({
                'type': 'rollover_gap',
                'contract': next_contract,
                'missing_dates': missing_next,
                'expiry_date': next_expiry,
                'severity': 'MEDIUM'
            })
    
    logger.info(f"Detected {len(gaps)} contract gaps")
    return gaps


def adjust_for_rollover(df: pd.DataFrame, method: str = 'ratio') -> pd.DataFrame:
    """
    Adjust continuous futures data for rollover gaps.
    
    Args:
        df: DataFrame with rollover gaps
        method: 'ratio' or 'difference' adjustment method
        
    Returns:
        DataFrame with rollover-adjusted prices
    """
    df_adj = df.copy()
    rollover_points = detect_rollover_gaps(df)
    
    if not rollover_points:
        return df_adj
    
    # Apply adjustments backwards from most recent
    for rollover in reversed(rollover_points):
        rollover_date = rollover['date']
        gap_pct = rollover['price_gap_pct']
        
        # Get data before rollover
        mask = df_adj.index < rollover_date
        
        if method == 'ratio':
            # Ratio adjustment (multiply by adjustment factor)
            adjustment_factor = 1 - (gap_pct / 100)
            price_columns = ['open', 'high', 'low', 'close']
            
            for col in price_columns:
                if col in df_adj.columns:
                    df_adj.loc[mask, col] *= adjustment_factor
                    
        elif method == 'difference':
            # Difference adjustment (subtract gap amount)
            gap_amount = rollover['new_price'] - rollover['old_price']
            price_columns = ['open', 'high', 'low', 'close']
            
            for col in price_columns:
                if col in df_adj.columns:
                    df_adj.loc[mask, col] -= gap_amount
        
        else:
            raise ValueError(f"Unknown rollover adjustment method: {method}")
    
    logger.info(f"Applied {method} rollover adjustment for {len(rollover_points)} rollover points")
    return df_adj


def save_ohlcv_csv(df: pd.DataFrame, 
                   file_path: Union[str, Path],
                   symbol: Optional[str] = None) -> None:
    """
    Save DataFrame to CSV in standardized format.
    
    Args:
        df: DataFrame with OHLCV data
        file_path: Output CSV path
        symbol: Optional symbol name for logging
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Reset index to include timestamp as column
    output_df = df.reset_index()
    
    # Ensure standard column order
    standard_order = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    if 'symbol' in output_df.columns:
        standard_order.append('symbol')
    
    # Reorder columns
    available_cols = [col for col in standard_order if col in output_df.columns]
    output_df = output_df[available_cols]
    
    # Save to CSV
    output_df.to_csv(file_path, index=False, float_format='%.2f')
    
    symbol_info = f" for {symbol}" if symbol else ""
    logger.info(f"Saved {len(output_df)} rows to {file_path}{symbol_info}")


# Example usage and testing functions
if __name__ == "__main__":
    # Initialize data warehouse
    warehouse = DataWarehouse()
    
    # List available symbols
    symbols = warehouse.list_available_symbols('1d')
    print(f"Available daily symbols: {symbols}")
    
    if symbols:
        # Load and clean data for first symbol
        symbol = symbols[0]
        print(f"\nLoading data for {symbol}...")
        
        try:
            df = load_symbol_data(symbol, '1d')
            print(f"Loaded {len(df)} rows")
            
            # Generate quality report
            report = generate_data_quality_report(df, symbol)
            print(f"Data quality score: {report['completeness_score']:.1f}%")
            
            # Clean the data
            df_clean, cleaning_report = clean_ohlcv_data(df)
            print(f"Cleaned data: {cleaning_report['original_rows']} -> {cleaning_report['final_rows']} rows")
            
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
    
    print("\n" + "="*50)
    print("Data Warehouse and Preprocessing System Ready!")
    print("Add your CSV files to data/warehouse/ directory")
    print("Format: SYMBOL_INTERVAL.csv (e.g., RELIANCE_1d.csv)")
    print("="*50)


def _standardize_column_names(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """Standardize column names to OHLCV schema."""
    df = df.copy()
    
    # Common column name mappings
    column_mappings = {
        'date': 'timestamp',
        'time': 'timestamp', 
        'datetime': 'timestamp',
        'ts': 'timestamp',
        date_column: 'timestamp',
        'o': 'open',
        'h': 'high',
        'l': 'low', 
        'c': 'close',
        'adj_close': 'close',
        'adj close': 'close',
        'vol': 'volume',
        'v': 'volume',
        'quantity': 'volume'
    }
    
    # Apply mappings (case-insensitive)
    for old_col in df.columns:
        old_col_lower = old_col.lower().strip()
        if old_col_lower in column_mappings:
            new_col = column_mappings[old_col_lower]
            if old_col != new_col:
                df.rename(columns={old_col: new_col}, inplace=True)
                logger.debug(f"Renamed column '{old_col}' -> '{new_col}'")
    
    return df


def _validate_required_columns(df: pd.DataFrame) -> None:
    """Validate that DataFrame has required OHLCV columns."""
    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for empty DataFrame
    if df.empty:
        raise ValueError("DataFrame is empty")


def _convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convert columns to appropriate data types."""
    df = df.copy()
    
    # Convert price columns to float64
    price_columns = ['open', 'high', 'low', 'close']
    for col in price_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert volume to int64 (handle NaN as 0)
    if 'volume' in df.columns:
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype('int64')
    
    # Ensure timestamp index is datetime
    if df.index.name == 'timestamp':
        df.index = pd.to_datetime(df.index)
    
    return df


def save_ohlcv_csv(df: pd.DataFrame, 
                   file_path: Union[str, Path],
                   symbol: Optional[str] = None) -> None:
    """
    Save DataFrame to CSV in standardized format.
    
    Args:
        df: DataFrame with OHLCV data
        file_path: Output CSV path
        symbol: Optional symbol name for logging
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Reset index to include timestamp as column
    output_df = df.reset_index()
    
    # Ensure standard column order
    standard_order = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    if 'symbol' in output_df.columns:
        standard_order.append('symbol')
    
    # Reorder columns
    available_cols = [col for col in standard_order if col in output_df.columns]
    output_df = output_df[available_cols]
    
    # Save to CSV
    output_df.to_csv(file_path, index=False, float_format='%.2f')
    
    symbol_info = f" for {symbol}" if symbol else ""
    logger.info(f"Saved {len(output_df)} rows to {file_path}{symbol_info}")


def create_sample_ohlcv_csv(file_path: Union[str, Path], 
                           symbol: str = "SAMPLE", 
                           num_days: int = 30) -> None:
    """
    Create a sample OHLCV CSV file for testing.
    
    Args:
        file_path: Output CSV path
        symbol: Symbol name
        num_days: Number of days of data to generate
    """
    from datetime import timedelta
    
    # Generate sample timestamps (1-minute bars for market hours)
    start_date = datetime(2025, 1, 1, 9, 15)  # Market open time
    timestamps = []
    
    for day in range(num_days):
        day_start = start_date + timedelta(days=day)
        
        # Generate 1-minute bars for trading session (9:15 to 15:30)
        for minute in range(375):  # 6h15m = 375 minutes
            ts = day_start + timedelta(minutes=minute)
            timestamps.append(ts)
    
    # Generate sample OHLCV data
    np.random.seed(42)  # For reproducible sample data
    base_price = 1000.0
    
    data = []
    current_price = base_price
    
    for ts in timestamps:
        # Random walk with some volatility
        change_pct = np.random.normal(0, 0.001)  # 0.1% volatility per minute
        new_price = current_price * (1 + change_pct)
        
        # Generate OHLC around the price
        high = new_price * (1 + abs(np.random.normal(0, 0.0005)))
        low = new_price * (1 - abs(np.random.normal(0, 0.0005)))
        open_price = current_price
        close_price = new_price
        
        # Ensure OHLC consistency
        high = max(high, open_price, close_price)
        low = min(low, open_price, close_price)
        
        # Random volume
        volume = int(np.random.exponential(1000) + 100)
        
        data.append({
            'timestamp': ts,
            'open': round(open_price, 2),
            'high': round(high, 2), 
            'low': round(low, 2),
            'close': round(close_price, 2),
            'volume': volume
        })
        
        current_price = new_price
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    save_ohlcv_csv(df, file_path, symbol)
    logger.info(f"Created sample CSV with {len(df)} rows for {symbol}")


# Sample data generation for testing
def create_sample_ohlcv_csv(file_path: Union[str, Path], 
                           symbol: str = "SAMPLE", 
                           num_days: int = 30) -> None:
    """
    Create a sample OHLCV CSV file for testing.
    
    Args:
        file_path: Output CSV path
        symbol: Symbol name
        num_days: Number of days of data to generate
    """
    from datetime import timedelta
    
    # Generate sample timestamps (1-minute bars for market hours)
    start_date = datetime(2025, 1, 1, 9, 15)  # Market open time
    timestamps = []
    
    for day in range(num_days):
        day_start = start_date + timedelta(days=day)
        
        # Generate 1-minute bars for trading session (9:15 to 15:30)
        for minute in range(375):  # 6h15m = 375 minutes
            ts = day_start + timedelta(minutes=minute)
            timestamps.append(ts)
    
    # Generate sample OHLCV data
    np.random.seed(42)  # For reproducible sample data
    base_price = 1000.0
    
    data = []
    current_price = base_price
    
    for ts in timestamps:
        # Random walk with some volatility
        change_pct = np.random.normal(0, 0.001)  # 0.1% volatility per minute
        new_price = current_price * (1 + change_pct)
        
        # Generate OHLC around the price
        high = new_price * (1 + abs(np.random.normal(0, 0.0005)))
        low = new_price * (1 - abs(np.random.normal(0, 0.0005)))
        open_price = current_price
        close_price = new_price
        
        # Ensure OHLC consistency
        high = max(high, open_price, close_price)
        low = min(low, open_price, close_price)
        
        # Random volume
        volume = int(np.random.exponential(1000) + 100)
        
        data.append({
            'timestamp': ts,
            'open': round(open_price, 2),
            'high': round(high, 2), 
            'low': round(low, 2),
            'close': round(close_price, 2),
            'volume': volume
        })
        
        current_price = new_price
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    save_ohlcv_csv(df, file_path, symbol)
    logger.info(f"Created sample CSV with {len(df)} rows for {symbol}")


# Example usage and testing functions
if __name__ == "__main__":
    # Initialize data warehouse
    warehouse = DataWarehouse()
    
    # List available symbols
    symbols = warehouse.list_available_symbols('1d')
    print(f"Available daily symbols: {symbols}")
    
    if symbols:
        # Load and clean data for first symbol
        symbol = symbols[0]
        print(f"\nLoading data for {symbol}...")
        
        try:
            df = load_symbol_data(symbol, '1d')
            print(f"Loaded {len(df)} rows")
            
            # Generate quality report
            report = generate_data_quality_report(df, symbol)
            print(f"Data quality score: {report['completeness_score']:.1f}%")
            
            # Clean the data
            df_clean, cleaning_report = clean_ohlcv_data(df)
            print(f"Cleaned data: {cleaning_report['original_rows']} -> {cleaning_report['final_rows']} rows")
            
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
    
    else:
        # Create sample data for testing if no data available
        print("No data found in warehouse. Creating sample data...")
        sample_path = warehouse.get_symbol_path("SAMPLE", "1d")
        create_sample_ohlcv_csv(sample_path, "SAMPLE", num_days=30)
        print(f"Created sample data at: {sample_path}")


# Futures Data Validation Pipeline
def validate_futures_data_integrity(df: pd.DataFrame, contract_symbol: str) -> Dict[str, Any]:
    """
    Comprehensive validation for futures data integrity.
    
    Args:
        df: DataFrame with OHLCVI futures data
        contract_symbol: Futures contract symbol
        
    Returns:
        Dict with validation results and recommendations
    """
    validation_results = {
        'contract_symbol': contract_symbol,
        'total_rows': len(df),
        'validation_timestamp': pd.Timestamp.now(),
        'issues': [],
        'warnings': [],
        'data_quality_score': 0.0,
        'recommendations': []
    }
    
    if df.empty:
        validation_results['issues'].append('Empty dataset')
        validation_results['data_quality_score'] = 0.0
        return validation_results
    
    # 1. Schema validation
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    futures_cols = ['open_interest']
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        validation_results['issues'].append(f'Missing columns: {missing_cols}')
    
    missing_futures_cols = [col for col in futures_cols if col not in df.columns]
    if missing_futures_cols:
        validation_results['warnings'].append(f'Missing futures columns: {missing_futures_cols}')
    
    # 2. OHLC relationship validation
    if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        # High should be >= Open, Low, Close
        high_violations = (df['high'] < df[['open', 'low', 'close']].max(axis=1)).sum()
        if high_violations > 0:
            validation_results['issues'].append(f'High < max(O,L,C) in {high_violations} rows')
        
        # Low should be <= Open, High, Close
        low_violations = (df['low'] > df[['open', 'high', 'close']].min(axis=1)).sum()
        if low_violations > 0:
            validation_results['issues'].append(f'Low > min(O,H,C) in {low_violations} rows')
        
        # Check for zero prices
        zero_prices = (df[['open', 'high', 'low', 'close']] <= 0).any(axis=1).sum()
        if zero_prices > 0:
            validation_results['issues'].append(f'Zero or negative prices in {zero_prices} rows')
    
    # 3. Volume validation
    if 'volume' in df.columns:
        negative_volume = (df['volume'] < 0).sum()
        if negative_volume > 0:
            validation_results['issues'].append(f'Negative volume in {negative_volume} rows')
        
        zero_volume_pct = (df['volume'] == 0).mean() * 100
        if zero_volume_pct > 10:
            validation_results['warnings'].append(f'High zero volume: {zero_volume_pct:.1f}% of rows')
    
    # 4. Open Interest validation (futures-specific)
    if 'open_interest' in df.columns:
        negative_oi = (df['open_interest'] < 0).sum()
        if negative_oi > 0:
            validation_results['issues'].append(f'Negative open interest in {negative_oi} rows')
        
        # OI should generally increase as we approach expiry, then drop to zero
        oi_trend = df['open_interest'].diff().dropna()
        sudden_drops = (oi_trend < -df['open_interest'].mean() * 0.5).sum()
        if sudden_drops > 1:  # Allow one major drop for rollover
            validation_results['warnings'].append(f'Unusual OI drops detected: {sudden_drops} occurrences')
    
    # 5. Price continuity validation
    if 'close' in df.columns and len(df) > 1:
        price_changes = df['close'].pct_change().dropna()
        
        # Check for extreme price movements (>10% in one bar)
        extreme_moves = (abs(price_changes) > 0.1).sum()
        if extreme_moves > 0:
            validation_results['warnings'].append(f'Extreme price movements (>10%): {extreme_moves} occurrences')
        
        # Check for price gaps (>5% gap between bars)
        price_gaps = (abs(price_changes) > 0.05).sum()
        if price_gaps > len(df) * 0.02:  # More than 2% of data has large gaps
            validation_results['warnings'].append(f'High frequency of price gaps: {price_gaps} occurrences')
    
    # 6. Time series validation
    if isinstance(df.index, pd.DatetimeIndex):
        # Check for duplicate timestamps
        duplicates = df.index.duplicated().sum()
        if duplicates > 0:
            validation_results['issues'].append(f'Duplicate timestamps: {duplicates} occurrences')
        
        # Check for chronological order
        if not df.index.is_monotonic_increasing:
            validation_results['issues'].append('Data not in chronological order')
        
        # Check for missing trading days (basic check)
        if len(df) > 5:
            date_diffs = df.index.to_series().diff().dropna()
            # Look for gaps > 3 days (accounting for weekends)
            large_gaps = (date_diffs > pd.Timedelta(days=3)).sum()
            if large_gaps > 0:
                validation_results['warnings'].append(f'Potential data gaps: {large_gaps} large time gaps')
    
    # 7. Contract-specific validation
    if contract_symbol:
        # Extract contract details
        contract_info = validate_futures_contract_name(contract_symbol)
        if not contract_info['valid']:
            validation_results['issues'].append(f'Invalid contract symbol format: {contract_symbol}')
        else:
            # Check if data dates align with contract lifecycle
            if isinstance(df.index, pd.DatetimeIndex):
                data_start = df.index.min()
                data_end = df.index.max()
                
                # Basic expiry check (this would need actual expiry calendar)
                if len(df) > 0:
                    # Check for data after typical contract expiry patterns
                    contract_month = contract_info.get('month')
                    if contract_month and len(df) > 200:  # More than ~8 months of daily data
                        validation_results['warnings'].append('Data span exceeds typical contract lifecycle')
    
    # 8. Calculate overall data quality score
    total_checks = 8
    issue_penalty = len(validation_results['issues']) * 15  # Each issue reduces score by 15%
    warning_penalty = len(validation_results['warnings']) * 5  # Each warning reduces score by 5%
    
    base_score = 100
    validation_results['data_quality_score'] = max(0, base_score - issue_penalty - warning_penalty)
    
    # Generate recommendations
    if validation_results['data_quality_score'] < 70:
        validation_results['recommendations'].append('Data quality is poor - manual review recommended')
    
    if len(validation_results['issues']) > 0:
        validation_results['recommendations'].append('Critical issues detected - data cleaning required')
    
    if 'open_interest' not in df.columns:
        validation_results['recommendations'].append('Add open interest data for better futures analysis')
    
    logger.info(f"Futures validation for {contract_symbol}: "
              f"Quality score {validation_results['data_quality_score']:.1f}%, "
              f"{len(validation_results['issues'])} issues, "
              f"{len(validation_results['warnings'])} warnings")
    
    return validation_results


def validate_rollover_consistency(continuous_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate consistency of rollover adjustments in continuous futures data.
    
    Args:
        continuous_df: Continuous futures DataFrame with rollover metadata
        
    Returns:
        Dict with rollover validation results
    """
    rollover_validation = {
        'rollover_points': 0,
        'price_discontinuities': [],
        'volume_anomalies': [],
        'oi_inconsistencies': [],
        'validation_passed': False
    }
    
    if 'active_contract' not in continuous_df.columns:
        rollover_validation['issues'] = ['No active_contract column found']
        return rollover_validation
    
    # Find rollover points
    contract_changes = continuous_df['active_contract'].ne(continuous_df['active_contract'].shift()).cumsum()
    rollover_dates = continuous_df.groupby(contract_changes)['active_contract'].first()
    rollover_validation['rollover_points'] = len(rollover_dates) - 1
    
    # Check for price discontinuities at rollover points
    if 'close' in continuous_df.columns:
        price_changes = continuous_df['close'].pct_change().abs()
        rollover_price_changes = price_changes[continuous_df['active_contract'].ne(continuous_df['active_contract'].shift())]
        
        # Flag discontinuities > 2% at rollover points
        large_discontinuities = rollover_price_changes[rollover_price_changes > 0.02]
        if len(large_discontinuities) > 0:
            rollover_validation['price_discontinuities'] = large_discontinuities.index.tolist()
    
    # Check volume patterns around rollovers
    if 'volume' in continuous_df.columns:
        for rollover_idx in rollover_price_changes.index:
            window_start = max(0, continuous_df.index.get_loc(rollover_idx) - 5)
            window_end = min(len(continuous_df), continuous_df.index.get_loc(rollover_idx) + 5)
            
            pre_rollover_volume = continuous_df.iloc[window_start:continuous_df.index.get_loc(rollover_idx)]['volume'].mean()
            post_rollover_volume = continuous_df.iloc[continuous_df.index.get_loc(rollover_idx):window_end]['volume'].mean()
            
            # Flag significant volume changes
            if pre_rollover_volume > 0 and abs(post_rollover_volume - pre_rollover_volume) / pre_rollover_volume > 0.5:
                rollover_validation['volume_anomalies'].append({
                    'date': rollover_idx,
                    'pre_volume': pre_rollover_volume,
                    'post_volume': post_rollover_volume,
                    'change_pct': ((post_rollover_volume - pre_rollover_volume) / pre_rollover_volume) * 100
                })
    
    # Overall validation
    rollover_validation['validation_passed'] = (
        len(rollover_validation['price_discontinuities']) == 0 and
        len(rollover_validation['volume_anomalies']) <= rollover_validation['rollover_points'] * 0.5
    )
    
    return rollover_validation


def check_basis_convergence_patterns(futures_df: pd.DataFrame, spot_df: pd.DataFrame, 
                                   expiry_date: str) -> Dict[str, Any]:
    """
    Validate basis convergence patterns as futures approach expiry.
    
    Args:
        futures_df: Futures price data
        spot_df: Spot index data
        expiry_date: Contract expiry date
        
    Returns:
        Dict with basis convergence analysis
    """
    convergence_analysis = {
        'expiry_date': expiry_date,
        'basis_at_start': None,
        'basis_at_expiry': None,
        'convergence_rate': None,
        'anomalies': [],
        'convergence_quality': 'UNKNOWN'
    }
    
    # Align data
    common_dates = futures_df.index.intersection(spot_df.index)
    if len(common_dates) < 10:
        convergence_analysis['convergence_quality'] = 'INSUFFICIENT_DATA'
        return convergence_analysis
    
    aligned_futures = futures_df.loc[common_dates]
    aligned_spot = spot_df.loc[common_dates]
    
    # Calculate basis (futures - spot)
    basis = aligned_futures['close'] - aligned_spot['close']
    
    # Time to expiry calculation
    expiry_dt = pd.to_datetime(expiry_date)
    time_to_expiry = (expiry_dt - basis.index).days
    
    # Filter to last 30 days before expiry
    last_30_days = basis[time_to_expiry <= 30]
    
    if len(last_30_days) < 5:
        convergence_analysis['convergence_quality'] = 'INSUFFICIENT_EXPIRY_DATA'
        return convergence_analysis
    
    # Analyze convergence
    convergence_analysis['basis_at_start'] = last_30_days.iloc[0]
    convergence_analysis['basis_at_expiry'] = last_30_days.iloc[-1]
    
    # Expected: basis should converge to near zero
    if abs(convergence_analysis['basis_at_expiry']) > aligned_spot['close'].iloc[-1] * 0.002:  # > 0.2% of spot
        convergence_analysis['anomalies'].append('Poor convergence at expiry')
    
    # Check convergence trend
    if len(last_30_days) > 1:
        convergence_slope = np.polyfit(range(len(last_30_days)), last_30_days.values, 1)[0]
        convergence_analysis['convergence_rate'] = convergence_slope
        
        # Expected: basis should trend towards zero
        if abs(convergence_analysis['basis_at_expiry']) > abs(convergence_analysis['basis_at_start']):
            convergence_analysis['anomalies'].append('Diverging basis near expiry')
    
    # Quality assessment
    if len(convergence_analysis['anomalies']) == 0:
        convergence_analysis['convergence_quality'] = 'GOOD'
    elif len(convergence_analysis['anomalies']) <= 1:
        convergence_analysis['convergence_quality'] = 'ACCEPTABLE'
    else:
        convergence_analysis['convergence_quality'] = 'POOR'
    
    return convergence_analysis


def run_comprehensive_futures_validation(contracts_data: Dict[str, pd.DataFrame], 
                                       spot_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Run comprehensive validation across all futures contracts.
    
    Args:
        contracts_data: Dict mapping contract symbols to DataFrames
        spot_data: Optional spot index data for basis analysis
        
    Returns:
        Comprehensive validation report
    """
    validation_report = {
        'validation_timestamp': pd.Timestamp.now(),
        'total_contracts': len(contracts_data),
        'contract_validations': {},
        'cross_contract_issues': [],
        'overall_quality_score': 0.0,
        'recommendations': []
    }
    
    quality_scores = []
    
    # Validate each contract individually
    for contract, df in contracts_data.items():
        contract_validation = validate_futures_data_integrity(df, contract)
        validation_report['contract_validations'][contract] = contract_validation
        quality_scores.append(contract_validation['data_quality_score'])
    
    # Cross-contract validation
    if len(contracts_data) > 1:
        # Check for overlapping contract periods
        contract_dates = {}
        for contract, df in contracts_data.items():
            if not df.empty and isinstance(df.index, pd.DatetimeIndex):
                contract_dates[contract] = (df.index.min(), df.index.max())
        
        # Look for suspicious overlaps
        for i, (contract1, (start1, end1)) in enumerate(contract_dates.items()):
            for contract2, (start2, end2) in list(contract_dates.items())[i+1:]:
                overlap_days = (min(end1, end2) - max(start1, start2)).days
                if overlap_days > 30:  # More than 1 month overlap
                    validation_report['cross_contract_issues'].append(
                        f'Suspicious overlap: {contract1} and {contract2} overlap {overlap_days} days'
                    )
    
    # Calculate overall quality score
    if quality_scores:
        validation_report['overall_quality_score'] = np.mean(quality_scores)
    
    # Generate recommendations
    if validation_report['overall_quality_score'] < 80:
        validation_report['recommendations'].append('Overall data quality needs improvement')
    
    if len(validation_report['cross_contract_issues']) > 0:
        validation_report['recommendations'].append('Review contract period overlaps')
    
    poor_contracts = [contract for contract, validation in validation_report['contract_validations'].items() 
                     if validation['data_quality_score'] < 70]
    if poor_contracts:
        validation_report['recommendations'].append(f'Focus on improving: {", ".join(poor_contracts)}')
    
    logger.info(f"Comprehensive futures validation completed: "
              f"Overall score {validation_report['overall_quality_score']:.1f}%, "
              f"{len(poor_contracts)} contracts need attention")
    
    return validation_report


    print("\n" + "="*50)
    print("Data Warehouse and Preprocessing System Ready!")
    print("Add your CSV files to data/warehouse/ directory")
    print("Format: SYMBOL_INTERVAL.csv (e.g., RELIANCE_1d.csv)")
    print("="*50)
