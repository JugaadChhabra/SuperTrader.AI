#!/usr/bin/env python3
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

INPUT_FILE = "nifty_data.csv"
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