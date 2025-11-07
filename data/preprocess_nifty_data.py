#!/usr/bin/env python3
"""
NSE ALL STOCKS DATA PREPROCESSOR
Handles multiple stocks from NSE_AllStocks_historical_data_5min.csv
Splits into 70% train, 20% validate, 10% test

Run: python preprocess_nse_allstocks.py
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

print("=" * 80)
print("NSE ALL STOCKS DATA PREPROCESSOR")
print("=" * 80)

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FILE = "NSE_AllStocks_historical_data_5min.csv"
OUTPUT_DIR = "processed_data"
TRAIN_FILE = "train.csv"
VALIDATE_FILE = "validate.csv"
TEST_FILE = "test.csv"

MARKET_OPEN = "09:15:00"
MARKET_CLOSE = "15:30:00"

# Split ratios
TRAIN_RATIO = 0.7
VALIDATE_RATIO = 0.2
TEST_RATIO = 0.1

print(f"\nConfiguration:")
print(f"  Input file: {INPUT_FILE}")
print(f"  Output directory: {OUTPUT_DIR}")
print(f"  Split ratios: {TRAIN_RATIO*100:.0f}% train, {VALIDATE_RATIO*100:.0f}% validate, {TEST_RATIO*100:.0f}% test")


# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

def load_data(filepath):
    """Load NSE All Stocks CSV data"""
    print(f"\n{'='*80}")
    print("STEP 1: LOADING NSE ALL STOCKS DATA")
    print(f"{'='*80}")
    
    if not os.path.exists(filepath):
        print(f"\n❌ ERROR: File not found: {filepath}")
        sys.exit(1)
    
    # Try different delimiters
    for delim in [',', '\t', '|', ';']:
        try:
            df = pd.read_csv(filepath, sep=delim)
            if len(df.columns) > 5:  # Should have multiple columns for stock data
                print(f"✓ Success with delimiter: {repr(delim)}")
                break
        except Exception as e:
            print(f"Failed with delimiter {repr(delim)}: {e}")
            continue
    
    if df is None or len(df.columns) <= 1:
        print(f"\n❌ Could not parse file")
        sys.exit(1)
    
    print(f"\n✓ Loaded: {len(df):,} rows")
    print(f"✓ Columns: {list(df.columns)}")
    
    # Normalize column names
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
    
    # Expected columns for NSE data (adjust based on actual file structure)
    possible_symbol_cols = ['symbol', 'stock', 'ticker', 'instrument']
    possible_date_cols = ['date', 'datetime', 'timestamp']
    possible_time_cols = ['time']
    
    # Find the correct column names
    symbol_col = None
    date_col = None
    time_col = None
    
    for col in df.columns:
        if any(x in col for x in possible_symbol_cols):
            symbol_col = col
        if any(x in col for x in possible_date_cols):
            date_col = col
        if any(x in col for x in possible_time_cols) and col != date_col:
            time_col = col
    
    # Check if we found required columns
    required_price_cols = ['open', 'high', 'low', 'close']
    missing_price_cols = [col for col in required_price_cols if col not in df.columns]
    
    if missing_price_cols:
        print(f"\n❌ ERROR: Missing price columns: {missing_price_cols}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)
    
    if not symbol_col:
        print(f"\n❌ ERROR: Could not find symbol column")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)
    
    if not date_col:
        print(f"\n❌ ERROR: Could not find date column")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)
    
    print(f"✓ Symbol column: {symbol_col}")
    print(f"✓ Date column: {date_col}")
    if time_col:
        print(f"✓ Time column: {time_col}")
    
    # Rename columns for consistency
    column_mapping = {symbol_col: 'symbol', date_col: 'date'}
    if time_col:
        column_mapping[time_col] = 'time'
    
    df = df.rename(columns=column_mapping)
    
    # Convert price columns to numeric
    for col in ['open', 'high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Show summary
    print(f"\n✓ Unique symbols: {df['symbol'].nunique()}")
    print(f"✓ Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Show sample
    print(f"\nFirst 3 rows:")
    display_cols = ['symbol', 'date']
    if 'time' in df.columns:
        display_cols.append('time')
    display_cols.extend(['open', 'high', 'low', 'close'])
    print(df[display_cols].head(3))
    
    return df


# ============================================================================
# STEP 2: CLEAN DATA
# ============================================================================

def clean_data(df):
    """Clean data - remove pre/post market, handle datetime"""
    print(f"\n{'='*80}")
    print("STEP 2: CLEANING DATA")
    print(f"{'='*80}")
    
    original_len = len(df)
    
    # Handle datetime parsing
    print(f"\nParsing datetime...")
    
    if 'time' in df.columns:
        # Separate date and time columns
        df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str), errors='coerce')
    else:
        # Date column might contain datetime
        df['datetime'] = pd.to_datetime(df['date'], errors='coerce')
        df['time'] = df['datetime'].dt.time
    
    # Remove rows with failed datetime parsing
    before_datetime_drop = len(df)
    df = df.dropna(subset=['datetime'])
    if len(df) < before_datetime_drop:
        print(f"⚠️  Dropped {before_datetime_drop - len(df)} rows with invalid datetime")
    
    # Filter market hours
    market_open_time = datetime.strptime(MARKET_OPEN, "%H:%M:%S").time()
    market_close_time = datetime.strptime(MARKET_CLOSE, "%H:%M:%S").time()
    
    df['time'] = df['datetime'].dt.time
    df = df[
        (df['time'] >= market_open_time) & 
        (df['time'] <= market_close_time)
    ].copy()
    
    print(f"✓ After time filter: {len(df):,} rows ({original_len - len(df):,} removed)")
    
    # Remove flat bars (no price movement)
    before_flat = len(df)
    flat_bars = (df['open'] == df['high']) & (df['high'] == df['low']) & (df['low'] == df['close'])
    df = df[~flat_bars].copy()
    
    print(f"✓ After removing flat bars: {len(df):,} rows ({before_flat - len(df):,} removed)")
    
    # Remove rows with NaN in price columns
    before_nan = len(df)
    df = df.dropna(subset=['open', 'high', 'low', 'close'])
    if len(df) < before_nan:
        print(f"⚠️  Dropped {before_nan - len(df)} rows with NaN prices")
    
    # Remove rows with zero or negative prices
    before_price = len(df)
    price_mask = (df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0) & (df['close'] > 0)
    df = df[price_mask].copy()
    if len(df) < before_price:
        print(f"⚠️  Dropped {before_price - len(df)} rows with zero/negative prices")
    
    # Sort by symbol and datetime
    df = df.set_index('datetime')
    df = df.sort_values(['symbol', 'datetime'])
    
    print(f"\n✓ Final data summary:")
    print(f"  - Total rows: {len(df):,}")
    print(f"  - Unique symbols: {df['symbol'].nunique()}")
    print(f"  - Date range: {df.index.min()} to {df.index.max()}")
    
    # Show symbol distribution
    symbol_counts = df['symbol'].value_counts()
    print(f"  - Top 5 symbols by data points:")
    for symbol, count in symbol_counts.head().items():
        print(f"    {symbol}: {count:,} rows")
    
    # Check if we have enough data
    if len(df) < 1000:
        print(f"\n❌ ERROR: Too few rows after cleaning ({len(df)})")
        print(f"   Need at least 1000 rows for training")
        sys.exit(1)
    
    return df


# ============================================================================
# STEP 3: ENGINEER FEATURES (ROBUST)
# ============================================================================

def calculate_rsi(prices, period=14):
    """Calculate RSI - robust version"""
    delta = prices.diff()
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    
    avg_gains = gains.ewm(span=period, adjust=False).mean()
    avg_losses = losses.ewm(span=period, adjust=False).mean()
    
    rs = avg_gains / (avg_losses + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Replace inf/nan
    rsi = rsi.replace([np.inf, -np.inf], np.nan)
    
    return rsi


def calculate_macd(prices):
    """Calculate MACD - robust version"""
    ema_12 = prices.ewm(span=12, adjust=False).mean()
    ema_26 = prices.ewm(span=26, adjust=False).mean()
    
    macd = ema_12 - ema_26
    macd_std = macd.rolling(window=50, min_periods=1).std()
    macd_norm = macd / (macd_std + 1e-10)
    
    # Replace inf/nan
    macd_norm = macd_norm.replace([np.inf, -np.inf], np.nan)
    
    return macd_norm


def calculate_atr(group_df, period=14):
    """Calculate ATR for a group - robust version"""
    high = group_df['high']
    low = group_df['low']
    close = group_df['close'].shift(1)
    
    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    atr_norm = atr / (group_df['close'] + 1e-10)
    
    # Replace inf/nan
    atr_norm = atr_norm.replace([np.inf, -np.inf], np.nan)
    
    return atr_norm


def engineer_features_for_symbol(symbol_df):
    """Engineer features for a single symbol"""
    df = symbol_df.copy()
    
    # Price-based features
    rolling_min = df['close'].rolling(window=50, min_periods=1).min()
    rolling_max = df['close'].rolling(window=50, min_periods=1).max()
    df['norm_close'] = (df['close'] - rolling_min) / (rolling_max - rolling_min + 1e-10)
    df['norm_close'] = df['norm_close'].replace([np.inf, -np.inf], np.nan)
    
    # Returns
    df['returns'] = df['close'].pct_change()
    df['returns'] = df['returns'].replace([np.inf, -np.inf], np.nan)
    
    # Log returns
    df['log_returns'] = np.log(df['close'] / (df['close'].shift(1) + 1e-10))
    df['log_returns'] = df['log_returns'].replace([np.inf, -np.inf], np.nan)
    
    # Technical indicators
    df['rsi_14'] = calculate_rsi(df['close'], period=14)
    df['macd'] = calculate_macd(df['close'])
    df['atr_14'] = calculate_atr(df, period=14)
    
    # Moving averages
    df['sma_10'] = df['close'].rolling(window=10, min_periods=1).mean()
    df['sma_20'] = df['close'].rolling(window=20, min_periods=1).mean()
    df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    
    # EMA distance
    df['ema_distance'] = (df['close'] - df['ema_21']) / (df['ema_21'] + 1e-10)
    df['ema_distance'] = df['ema_distance'].replace([np.inf, -np.inf], np.nan)
    
    # Volatility
    df['volatility'] = df['returns'].rolling(window=20, min_periods=1).std()
    df['volatility'] = df['volatility'].replace([np.inf, -np.inf], np.nan)
    
    # Volume features (if available)
    if 'volume' in df.columns:
        df['volume_sma'] = df['volume'].rolling(window=20, min_periods=1).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-10)
        df['volume_ratio'] = df['volume_ratio'].replace([np.inf, -np.inf], np.nan)
    
    return df


def engineer_features(df):
    """Engineer features for all symbols"""
    print(f"\n{'='*80}")
    print("STEP 3: ENGINEERING FEATURES")
    print(f"{'='*80}")
    
    original_len = len(df)
    processed_symbols = []
    
    print(f"\nProcessing features by symbol...")
    
    # Process each symbol separately
    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol].copy()
        
        if len(symbol_df) < 50:  # Skip symbols with too little data
            continue
            
        try:
            processed_df = engineer_features_for_symbol(symbol_df)
            processed_symbols.append(processed_df)
        except Exception as e:
            print(f"⚠️  Error processing {symbol}: {e}")
            continue
    
    if not processed_symbols:
        print(f"\n❌ ERROR: No symbols could be processed")
        sys.exit(1)
    
    # Combine all processed symbols
    df = pd.concat(processed_symbols, ignore_index=False)
    df = df.sort_index()
    
    print(f"✓ Processed {len(processed_symbols)} symbols")
    
    # Add time-based features
    print(f"\nAdding time-based features...")
    
    df['hour'] = df.index.hour / 24.0
    df['minute'] = df.index.minute / 60.0
    df['day_of_week'] = df.index.dayofweek / 6.0
    
    market_open_minutes = 9 * 60 + 15
    current_minutes = df.index.hour * 60 + df.index.minute
    df['minutes_since_open'] = (current_minutes - market_open_minutes) / 375.0
    
    market_close_minutes = 15 * 60 + 30
    df['minutes_to_close'] = (market_close_minutes - current_minutes) / 375.0
    df['session'] = df['minutes_since_open']
    
    # Define feature columns
    feature_cols = [
        'norm_close', 'returns', 'log_returns',
        'rsi_14', 'macd', 'atr_14',
        'sma_10', 'sma_20', 'ema_9', 'ema_21', 'ema_distance', 'volatility',
        'hour', 'minute', 'day_of_week', 'minutes_since_open', 'minutes_to_close', 'session'
    ]
    
    # Add volume features if available
    if 'volume_ratio' in df.columns:
        feature_cols.extend(['volume_ratio'])
    
    # Check for NaN/inf in features
    print(f"\nChecking for NaN/inf values...")
    for col in feature_cols:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            inf_count = np.isinf(df[col]).sum()
            if nan_count > 0 or inf_count > 0:
                print(f"   ⚠️  {col}: {nan_count} NaN, {inf_count} inf")
    
    # Drop rows with NaN in ANY feature
    print(f"\nDropping rows with NaN/inf...")
    df = df.replace([np.inf, -np.inf], np.nan)
    available_features = [col for col in feature_cols if col in df.columns]
    df = df.dropna(subset=available_features)
    
    dropped = original_len - len(df)
    print(f"   ✓ Kept: {len(df):,} rows")
    print(f"   ✓ Dropped: {dropped:,} rows ({dropped/original_len*100:.1f}%)")
    
    if len(df) < 1000:
        print(f"\n❌ ERROR: Too few rows after feature engineering ({len(df)})")
        sys.exit(1)
    
    print(f"\n✓ Total features: {len(available_features)}")
    print(f"✓ Available features: {available_features}")
    
    return df


# ============================================================================
# STEP 4: SPLIT TRAIN/VALIDATE/TEST
# ============================================================================

def split_train_validate_test(df, train_ratio=0.7, validate_ratio=0.2, test_ratio=0.1):
    """Split into train, validate, and test"""
    print(f"\n{'='*80}")
    print("STEP 4: SPLITTING TRAIN/VALIDATE/TEST")
    print(f"{'='*80}")
    
    if len(df) == 0:
        print(f"\n❌ ERROR: No data to split!")
        sys.exit(1)
    
    # Ensure ratios sum to 1
    total_ratio = train_ratio + validate_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        print(f"⚠️  Ratios don't sum to 1.0, normalizing...")
        train_ratio /= total_ratio
        validate_ratio /= total_ratio
        test_ratio /= total_ratio
    
    train_end = int(len(df) * train_ratio)
    validate_end = int(len(df) * (train_ratio + validate_ratio))
    
    # Ensure minimum data in each split
    min_rows = 100
    if train_end < min_rows or (validate_end - train_end) < min_rows or (len(df) - validate_end) < min_rows:
        print(f"\n❌ ERROR: Not enough data to split properly")
        print(f"   Total rows: {len(df)}")
        print(f"   Need at least {min_rows * 3} rows")
        sys.exit(1)
    
    train_df = df.iloc[:train_end].copy()
    validate_df = df.iloc[train_end:validate_end].copy()
    test_df = df.iloc[validate_end:].copy()
    
    print(f"\n✓ Total rows: {len(df):,}")
    print(f"✓ Train rows: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"✓ Validate rows: {len(validate_df):,} ({len(validate_df)/len(df)*100:.1f}%)")
    print(f"✓ Test rows: {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)")
    
    # Show symbol distribution in each split
    print(f"\n✓ Symbol distribution:")
    print(f"  Train: {train_df['symbol'].nunique()} unique symbols")
    print(f"  Validate: {validate_df['symbol'].nunique()} unique symbols")
    print(f"  Test: {test_df['symbol'].nunique()} unique symbols")
    
    return train_df, validate_df, test_df


# ============================================================================
# STEP 5: SAVE DATA
# ============================================================================

def save_data(train_df, validate_df, test_df, output_dir):
    """Save to CSV"""
    print(f"\n{'='*80}")
    print("STEP 5: SAVING DATA")
    print(f"{'='*80}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, TRAIN_FILE)
    validate_path = os.path.join(output_dir, VALIDATE_FILE)
    test_path = os.path.join(output_dir, TEST_FILE)
    
    train_df.to_csv(train_path)
    validate_df.to_csv(validate_path)
    test_df.to_csv(test_path)
    
    print(f"\n✓ Saved train data: {train_path}")
    print(f"  - Rows: {len(train_df):,}")
    print(f"  - Size: {os.path.getsize(train_path) / 1024 / 1024:.1f} MB")
    
    print(f"\n✓ Saved validate data: {validate_path}")
    print(f"  - Rows: {len(validate_df):,}")
    print(f"  - Size: {os.path.getsize(validate_path) / 1024 / 1024:.1f} MB")
    
    print(f"\n✓ Saved test data: {test_path}")
    print(f"  - Rows: {len(test_df):,}")
    print(f"  - Size: {os.path.getsize(test_path) / 1024 / 1024:.1f} MB")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main preprocessing pipeline"""
    
    try:
        df = load_data(INPUT_FILE)
        df_clean = clean_data(df)
        df_features = engineer_features(df_clean)
        train_df, validate_df, test_df = split_train_validate_test(
            df_features, 
            train_ratio=TRAIN_RATIO, 
            validate_ratio=VALIDATE_RATIO, 
            test_ratio=TEST_RATIO
        )
        save_data(train_df, validate_df, test_df, OUTPUT_DIR)
        
        print(f"\n{'='*80}")
        print("✅ PREPROCESSING COMPLETE!")
        print(f"{'='*80}")
        print(f"\nOutput files:")
        print(f"  - {os.path.join(OUTPUT_DIR, TRAIN_FILE)}")
        print(f"  - {os.path.join(OUTPUT_DIR, VALIDATE_FILE)}")
        print(f"  - {os.path.join(OUTPUT_DIR, TEST_FILE)}")
        print(f"\nNext step: python train.py")
        print(f"\n{'='*80}")
        
    except KeyboardInterrupt:
        print(f"\n\n❌ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"❌ ERROR OCCURRED")
        print(f"{'='*80}")
        print(f"\nError: {e}")
        print(f"\nTraceback:")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    print("\n")
    main()#!/usr/bin/env python3
"""
ROBUST NIFTY DATA PREPROCESSOR
Handles NaN values properly

Run: python preprocess_robust.py
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

print("=" * 80)
print("NIFTY DATA PREPROCESSOR - ROBUST VERSION")
print("=" * 80)

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FILE = "NSE_AllStocks_historical_data_5min.csv"
OUTPUT_DIR = "processed_data"
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"

MARKET_OPEN = "09:15:00"
MARKET_CLOSE = "15:30:00"
TEST_SIZE = 0.2

print(f"\nConfiguration:")
print(f"  Input file: {INPUT_FILE}")
print(f"  Output directory: {OUTPUT_DIR}")


# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

def load_data(filepath):
    """Load CSV/TSV data with automatic delimiter detection"""
    print(f"\n{'='*80}")
    print("STEP 1: LOADING DATA")
    print(f"{'='*80}")
    
    if not os.path.exists(filepath):
        print(f"\n❌ ERROR: File not found: {filepath}")
        sys.exit(1)
    
    # Try different delimiters
    for delim in ['\t', ',', '|', ';']:
        try:
            df = pd.read_csv(filepath, sep=delim)
            if len(df.columns) > 1 and 'date' in df.columns.str.lower().tolist():
                print(f"✓ Success with delimiter: {repr(delim)}")
                break
        except:
            continue
    
    if df is None or len(df.columns) <= 1:
        print(f"\n❌ Could not parse file")
        sys.exit(1)
    
    print(f"\n✓ Loaded: {len(df):,} rows")
    print(f"✓ Columns: {list(df.columns)}")
    
    # Normalize column names
    df.columns = df.columns.str.lower().str.strip()
    
    # Check required columns
    required_cols = ['date', 'time', 'open', 'high', 'low', 'close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"\n❌ ERROR: Missing columns: {missing_cols}")
        sys.exit(1)
    
    print(f"✓ All required columns present")
    
    # Convert to numeric (handle any string values)
    for col in ['open', 'high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Show sample
    print(f"\nFirst 3 rows:")
    print(df[['date', 'time', 'open', 'high', 'low', 'close']].head(3))
    
    return df


# ============================================================================
# STEP 2: CLEAN DATA
# ============================================================================

def clean_data(df):
    """Clean data - remove pre/post market"""
    print(f"\n{'='*80}")
    print("STEP 2: CLEANING DATA")
    print(f"{'='*80}")
    
    original_len = len(df)
    
    # Parse time
    print(f"\nParsing datetime...")
    df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S', errors='coerce').dt.time
    
    # Remove rows with failed parsing
    before_time_drop = len(df)
    df = df.dropna(subset=['time'])
    if len(df) < before_time_drop:
        print(f"⚠️  Dropped {before_time_drop - len(df)} rows with invalid time")
    
    # Filter market hours
    market_open_time = datetime.strptime(MARKET_OPEN, "%H:%M:%S").time()
    market_close_time = datetime.strptime(MARKET_CLOSE, "%H:%M:%S").time()
    
    df = df[
        (df['time'] >= market_open_time) & 
        (df['time'] <= market_close_time)
    ].copy()
    
    print(f"✓ After time filter: {len(df):,} rows ({original_len - len(df):,} removed)")
    
    # Remove flat bars
    before_flat = len(df)
    flat_bars = (df['open'] == df['high']) & (df['high'] == df['low']) & (df['low'] == df['close'])
    df = df[~flat_bars].copy()
    
    print(f"✓ After removing flat bars: {len(df):,} rows ({before_flat - len(df):,} removed)")
    
    # Remove rows with NaN in price columns
    before_nan = len(df)
    df = df.dropna(subset=['open', 'high', 'low', 'close'])
    if len(df) < before_nan:
        print(f"⚠️  Dropped {before_nan - len(df)} rows with NaN prices")
    
    # Combine date + time
    print(f"\nCombining date and time...")
    df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str))
    df = df.set_index('datetime')
    df = df.sort_index()
    
    print(f"✓ Date range: {df.index.min()} to {df.index.max()}")
    print(f"✓ Clean data: {len(df):,} rows")
    
    # Check if we have enough data
    if len(df) < 100:
        print(f"\n❌ ERROR: Too few rows after cleaning ({len(df)})")
        print(f"   Need at least 100 rows for training")
        sys.exit(1)
    
    return df


# ============================================================================
# STEP 3: ENGINEER FEATURES (ROBUST)
# ============================================================================

def calculate_rsi(prices, period=30):
    """Calculate RSI - robust version"""
    delta = prices.diff()
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    
    avg_gains = gains.ewm(span=period, adjust=False).mean()
    avg_losses = losses.ewm(span=period, adjust=False).mean()
    
    rs = avg_gains / (avg_losses + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Replace inf/nan
    rsi = rsi.replace([np.inf, -np.inf], np.nan)
    
    return rsi


def calculate_macd(prices):
    """Calculate MACD - robust version"""
    ema_12 = prices.ewm(span=12, adjust=False).mean()
    ema_26 = prices.ewm(span=26, adjust=False).mean()
    
    macd = ema_12 - ema_26
    macd_std = macd.rolling(window=75, min_periods=1).std()
    macd_norm = macd / (macd_std + 1e-10)
    
    # Replace inf/nan
    macd_norm = macd_norm.replace([np.inf, -np.inf], np.nan)
    
    return macd_norm


def calculate_atr(df, period=14):
    """Calculate ATR - robust version"""
    high = df['high']
    low = df['low']
    close = df['close'].shift(1)
    
    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    atr_norm = atr / (df['close'] + 1e-10)
    
    # Replace inf/nan
    atr_norm = atr_norm.replace([np.inf, -np.inf], np.nan)
    
    return atr_norm


def engineer_features(df):
    """Engineer all features - robust version"""
    print(f"\n{'='*80}")
    print("STEP 3: ENGINEERING FEATURES")
    print(f"{'='*80}")
    
    df = df.copy()
    original_len = len(df)
    
    print(f"\n1. Price-based features...")
    
    # Normalized close
    rolling_min = df['close'].rolling(window=75, min_periods=1).min()
    rolling_max = df['close'].rolling(window=75, min_periods=1).max()
    df['norm_close'] = (df['close'] - rolling_min) / (rolling_max - rolling_min + 1e-10)
    df['norm_close'] = df['norm_close'].replace([np.inf, -np.inf], np.nan)
    
    # Returns
    df['returns'] = df['close'].pct_change()
    df['returns'] = df['returns'].replace([np.inf, -np.inf], np.nan)
    
    # Log returns
    df['log_returns'] = np.log(df['close'] / (df['close'].shift(1) + 1e-10))
    df['log_returns'] = df['log_returns'].replace([np.inf, -np.inf], np.nan)
    
    print(f"   ✓ norm_close, returns, log_returns")
    
    print(f"\n2. Technical indicators...")
    
    df['rsi_30'] = calculate_rsi(df['close'], period=30)
    df['macd'] = calculate_macd(df['close'])
    df['atr_14'] = calculate_atr(df, period=14)
    
    df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema_distance'] = (df['close'] - df['ema_21']) / (df['ema_21'] + 1e-10)
    df['ema_distance'] = df['ema_distance'].replace([np.inf, -np.inf], np.nan)
    
    df['volatility'] = df['returns'].rolling(window=20, min_periods=1).std()
    df['volatility'] = df['volatility'].replace([np.inf, -np.inf], np.nan)
    
    print(f"   ✓ rsi_30, macd, atr_14, ema_9, ema_21, ema_distance, volatility")
    
    print(f"\n3. Time-based features...")
    
    df['hour'] = df.index.hour / 24.0
    df['minute'] = df.index.minute / 60.0
    
    market_open_minutes = 9 * 60 + 15
    current_minutes = df.index.hour * 60 + df.index.minute
    df['minutes_since_open'] = (current_minutes - market_open_minutes) / 375.0
    
    market_close_minutes = 15 * 60 + 30
    df['minutes_to_close'] = (market_close_minutes - current_minutes) / 375.0
    df['session'] = df['minutes_since_open']
    
    print(f"   ✓ hour, minute, minutes_since_open, minutes_to_close, session")
    
    # Check for NaN/inf in features
    print(f"\n4. Checking for NaN/inf values...")
    feature_cols = [
        'norm_close', 'returns', 'log_returns',
        'rsi_30', 'macd', 'atr_14',
        'ema_9', 'ema_21', 'ema_distance', 'volatility',
        'hour', 'minute', 'minutes_since_open', 'minutes_to_close', 'session'
    ]
    
    for col in feature_cols:
        nan_count = df[col].isna().sum()
        inf_count = np.isinf(df[col]).sum()
        if nan_count > 0 or inf_count > 0:
            print(f"   ⚠️  {col}: {nan_count} NaN, {inf_count} inf")
    
    # Drop rows with NaN in ANY feature
    print(f"\n5. Dropping rows with NaN/inf...")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_cols)
    
    dropped = original_len - len(df)
    print(f"   ✓ Kept: {len(df):,} rows")
    print(f"   ✓ Dropped: {dropped:,} rows ({dropped/original_len*100:.1f}%)")
    
    if len(df) < 100:
        print(f"\n❌ ERROR: Too few rows after feature engineering ({len(df)})")
        print(f"   This usually means:")
        print(f"   1. Not enough data to calculate features (need >75 rows)")
        print(f"   2. Price data has issues (zeros, NaN, negative values)")
        sys.exit(1)
    
    print(f"\n✓ Total features: {len(feature_cols)}")
    
    return df


# ============================================================================
# STEP 4: SPLIT TRAIN/TEST
# ============================================================================

def split_train_test(df, test_size=0.2):
    """Split into train and test"""
    print(f"\n{'='*80}")
    print("STEP 4: SPLITTING TRAIN/TEST")
    print(f"{'='*80}")
    
    if len(df) == 0:
        print(f"\n❌ ERROR: No data to split!")
        sys.exit(1)
    
    split_idx = int(len(df) * (1 - test_size))
    
    # Ensure at least some data in each split
    if split_idx < 10 or len(df) - split_idx < 10:
        print(f"\n❌ ERROR: Not enough data to split properly")
        print(f"   Total rows: {len(df)}")
        print(f"   Need at least 50 rows")
        sys.exit(1)
    
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    print(f"\n✓ Total rows: {len(df):,}")
    print(f"✓ Train rows: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"✓ Test rows: {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, test_df


# ============================================================================
# STEP 5: SAVE DATA
# ============================================================================

def save_data(train_df, test_df, output_dir):
    """Save to CSV"""
    print(f"\n{'='*80}")
    print("STEP 5: SAVING DATA")
    print(f"{'='*80}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, TRAIN_FILE)
    test_path = os.path.join(output_dir, TEST_FILE)
    
    train_df.to_csv(train_path)
    test_df.to_csv(test_path)
    
    print(f"\n✓ Saved train data: {train_path}")
    print(f"  - Rows: {len(train_df):,}")
    print(f"  - Size: {os.path.getsize(train_path) / 1024:.1f} KB")
    
    print(f"\n✓ Saved test data: {test_path}")
    print(f"  - Rows: {len(test_df):,}")
    print(f"  - Size: {os.path.getsize(test_path) / 1024:.1f} KB")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main preprocessing pipeline"""
    
    try:
        df = load_data(INPUT_FILE)
        df_clean = clean_data(df)
        df_features = engineer_features(df_clean)
        train_df, test_df = split_train_test(df_features, test_size=TEST_SIZE)
        save_data(train_df, test_df, OUTPUT_DIR)
        
        print(f"\n{'='*80}")
        print("✅ PREPROCESSING COMPLETE!")
        print(f"{'='*80}")
        print(f"\nOutput files:")
        print(f"  - {os.path.join(OUTPUT_DIR, TRAIN_FILE)}")
        print(f"  - {os.path.join(OUTPUT_DIR, TEST_FILE)}")
        print(f"\nNext step: python train.py")
        print(f"\n{'='*80}")
        
    except KeyboardInterrupt:
        print(f"\n\n❌ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"❌ ERROR OCCURRED")
        print(f"{'='*80}")
        print(f"\nError: {e}")
        print(f"\nTraceback:")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    print("\n")
    main()