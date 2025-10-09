"""
Data Warehouse and Preprocessing for SuperTrader.AI
Handles loading, cleaning, and preprocessing historical OHLCV data from CSV files.

IMPORTANT: This module works with local CSV data warehouse only.
- For LIVE trading data, use agents/data_agent.py ICICI WebSocket  
- For HISTORICAL data, this module loads from local CSV files and preprocesses them

Features:
- Market hours filtering (9:15 AM - 3:30 PM IST only)
- IST timestamp handling (no timezone conversion)
- Clean CSV output with separate date/time columns
- Comprehensive data validation and reporting

Usage:
- Data Warehouse: CSV files containing historical OHLCV data
- Data Cleaning: Remove outliers, fill gaps, validate schema
- Preprocessing: Generate features, normalize data, prepare for ML
- Backtesting: Load clean datasets for strategy testing
- Model Training: Prepare preprocessed datasets for RL training
"""

# Standard library imports
import csv
import hashlib
import json
import os
from datetime import datetime, timezone, timedelta, time
from typing import List, Dict, Tuple

# Third-party imports
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

# Constants
ICICI_CUSTOMER_DETAIL_URL = "https://api.icicidirect.com/breezeapi/api/v1/customerdetails"
ICICI_HISTORICAL_URL = "https://api.icicidirect.com/breezeapi/api/v1/historicalcharts"
MARKET_OPEN_TIME = time(9, 15)  # 9:15 AM
MARKET_CLOSE_TIME = time(15, 30)  # 3:30 PM
REQUIRED_OHLCV_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
load_dotenv()

stock_code: str = input("Enter Stock Symbol: ")
print(f"Selected stock: {stock_code}")

# Credentials - Replace with your actual keys
secret_key: str | None = os.getenv("SECRET_KEY")
appkey: str | None = os.getenv("APP_KEY") 
session_key: str | None = os.getenv("API_SESSION_TOKEN")

# Debug: Check if environment variables are loaded
print("Environment Variables Check:")
print(f"SECRET_KEY: {'[OK]' if secret_key else '[MISSING]'}")
print(f"APP_KEY: {'[OK]' if appkey else '[MISSING]'}")  
print(f"SESSION_KEY: {'[OK]' if session_key else '[MISSING]'}")
print()

if not all([secret_key, appkey, session_key]):
    print("[ERROR] Missing required environment variables. Please check your .env file.")
    print("Required variables: SECRET_KEY, APP_KEY, SESSION_KEY")
    exit(1)

# ============================================================================
# DATA VALIDATION FUNCTIONS
# ============================================================================



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
        print(f"[WARNING] Found {invalid_count} OHLC logic violations - fixing...")
        
        # Fix high values (data entry errors)
        df.loc[invalid_high, 'high'] = df.loc[invalid_high, ['open', 'close', 'low']].max(axis=1)
        
        # Fix low values (data entry errors)
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
        print(f"[FIX] Fixing {negative_volumes} negative volume entries...")
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
    final_count = len(df)
    
    duplicates_removed = initial_count - final_count
    if duplicates_removed > 0:
        print(f"[FIX] Removed {duplicates_removed} duplicate timestamps...")
    
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
    
    # Drop any rows with missing OHLC data
    df_clean = df.dropna(subset=['open', 'high', 'low', 'close'])
    
    # For volume, set NaN to 0 (common in some data feeds)
    df_clean['volume'] = df_clean['volume'].fillna(0)
    
    dropped_count = initial_count - len(df_clean)
    if dropped_count > 0:
        print(f"[CLEANED] Dropped {dropped_count} rows with missing price data")
        print(f"[INFO] Kept {len(df_clean)} complete records ({(len(df_clean)/initial_count)*100:.1f}% retention)")
    
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
    print("\n[VALIDATION] Starting data validation pipeline...")
    
    df = pd.DataFrame(historical_data)
    
    df = remove_duplicate_timestamps(df)
    
    df = validate_ohlc_logic(df)
    
    df = validate_volume_data(df)
    
    df = drop_missing_data(df)
    
    columns_to_keep = ['exchange_name', 'stock_code', 'date', 'time', 'open', 'high', 'low', 'close', 'volume']
    df_final = df[columns_to_keep]
    
    cleaned_data = df_final.to_dict('records')
    
    print("[SUCCESS] Data validation complete!")
    
    return cleaned_data

print("Fetching session token...")
time_stamp = datetime.now(timezone.utc).isoformat()[:19] + '.000Z'

customerDetail_payload = json.dumps({
    "SessionToken": session_key,
    "AppKey": appkey
})

customerDetail_headers = {
    'Content-Type': 'application/json',
}

customerDetail_response = requests.request("GET", ICICI_CUSTOMER_DETAIL_URL, 
                                          headers=customerDetail_headers, 
                                          data=customerDetail_payload)

print(f"Response Status Code: {customerDetail_response.status_code}")
print(f"Response Text: {customerDetail_response.text}")

try:
    data = json.loads(customerDetail_response.text)
    print(f"Parsed Data: {data}")
    
    if data and "Success" in data and "session_token" in data["Success"]:
        session_token = data["Success"]["session_token"]
        print(f"Session token retrieved: {session_token[:20]}...")
    else:
        print("[ERROR] Could not find session token in response")
        print(f"Available keys in response: {list(data.keys()) if data else 'None'}")
        if data and "Error" in data:
            print(f"API Error: {data['Error']}")
        exit(1)
        
except json.JSONDecodeError as e:
    print(f"[ERROR] JSON Decode Error: {e}")
    print(f"Raw response: {customerDetail_response.text}")
    exit(1)
except Exception as e:
    print(f"[ERROR] Unexpected error: {e}")
    exit(1)

print("\nFetching 5-minute interval data from April 1 to October 1, 2025...")

time_stamp = datetime.now(timezone.utc).isoformat()[:19] + '.000Z'

payload = json.dumps({
    "interval": "5minute",
    "from_date": "2025-04-01T10:20:00.000Z",  
    "to_date": "2025-10-01T15:30:00.000Z", 
    "stock_code": stock_code,
    "exchange_code": "NSE",
    "product_type": "cash"
}, separators=(',', ':'))

checksum = hashlib.sha256((time_stamp + payload + secret_key).encode("utf-8")).hexdigest()

headers = {
    'Content-Type': 'application/json',
    'X-Checksum': 'token ' + checksum,
    'X-Timestamp': time_stamp,
    'X-AppKey': appkey,
    'X-SessionToken': session_token
}

response = requests.request("GET", ICICI_HISTORICAL_URL, headers=headers, data=payload)

print("\n" + "="*60)
print("Processing API Response...")
print("="*60)

try:
    response_data = json.loads(response.text)
    
    if "Success" in response_data and response_data["Success"]:
        historical_data = response_data["Success"]
        
        initial_cleaned_data = []
        
        for record in historical_data:
            datetime_str = record.get("datetime", "")
            if datetime_str:
                try:
                    dt_obj = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
                    
                    if not (time(9, 15) <= dt_obj.time() <= time(15, 30)):
                        continue
                    
                    date_part = dt_obj.strftime("%Y-%m-%d")
                    time_part = dt_obj.strftime("%H:%M:%S")
                    
                    cleaned_record = {
                        'exchange_name': 'NSE',
                        'stock_code': stock_code,
                        'date': date_part,
                        'time': time_part,
                        'open': record.get('open', ''),
                        'high': record.get('high', ''),
                        'low': record.get('low', ''),
                        'close': record.get('close', ''),
                        'volume': record.get('volume', '')
                    }
                    initial_cleaned_data.append(cleaned_record)
                except:
                    continue
        
        cleaned_data = validate_and_clean_data(initial_cleaned_data)
        
        if cleaned_data:
            df = pd.DataFrame(cleaned_data)
            csv_filename = f'{stock_code}_historical_data_5min.csv'
            df.to_csv(csv_filename, index=False)
            
            print(f"[SUCCESS] Clean data saved to '{csv_filename}'")
            print(f"[INFO] Records processed: {len(cleaned_data)}")
            print(f"[INFO] Date range: {df['date'].min()} to {df['date'].max()}")
            
            print("\n� Data Quality Report:")
            
            print("\n🔍 Sample data (first 5 rows):")
            print(df[['exchange_name', 'stock_code', 'date', 'time', 'open', 'high', 'low', 'close', 'volume']].head().to_string(index=False))
            
        else:
            print("[ERROR] No valid data remaining after cleaning")

    else:
        print("[ERROR] API response:")
        if "Error" in response_data:
            print(f"API Error: {response_data['Error']}")
        else:
            print(f"Unexpected response format: {response_data}")

except json.JSONDecodeError as e:
    print(f"[ERROR] JSON Decode Error: {e}")
    print(f"Raw response: {response.text[:500]}...")

except Exception as e:
    print(f"[ERROR] Unexpected error while processing data: {e}")
    print(f"Response status: {response.status_code}")
    print(f"Response text: {response.text[:500]}...") 