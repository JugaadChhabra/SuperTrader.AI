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
import sys
from datetime import datetime, timezone, timedelta, time
from typing import List, Dict, Tuple

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Third-party imports
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

# Constants
ICICI_CUSTOMER_DETAIL_URL = "https://api.icicidirect.com/breezeapi/api/v1/customerdetails"
ICICI_HISTORICAL_URL = "https://api.icicidirect.com/breezeapi/api/v1/historicalcharts"
MARKET_OPEN_TIME = time(9, 15)
MARKET_CLOSE_TIME = time(15, 30)
REQUIRED_OHLCV_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
load_dotenv()

stock_code: str = input("Enter Stock Symbol: ")
print(f"Selected stock: {stock_code}")

# Credentials - Replace with your actual keys
secret_key = os.getenv("SECRET_KEY")
appkey = os.getenv("APP_KEY") 
session_key = os.getenv("API_SESSION_TOKEN")

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
# DATA VALIDATION - Import from technical indicators module
# ============================================================================

# Import data validation functions from technical indicators module
from indicators.technical import validate_and_clean_data

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

        print(f"Response data: (display suppressed, {len(historical_data)} records)")
        
        # Debug: Print first record to see ALL available fields and values
        if historical_data:
            print("DEBUG: First API record (all fields):")
            first_record = historical_data[0]
            for key, value in first_record.items():
                print(f"  {key}: '{value}' (type: {type(value)})")
            print()

        initial_cleaned_data = []
        total_records = 0
        missing_volume_count = 0

        for record in historical_data:
            total_records += 1
            datetime_str = record.get("datetime", "")
            if datetime_str:
                try:
                    dt_obj = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")

                    # keep only market hours
                    if not (time(9, 15) <= dt_obj.time() <= time(15, 30)):
                        continue

                    date_part = dt_obj.strftime("%Y-%m-%d")
                    time_part = dt_obj.strftime("%H:%M:%S")

                                                            # Extract volume - for indices like NIFTY, API returns empty volume field
                    # Set to 0 to indicate no volume data available for index
                    api_volume = record.get('volume', '')
                    if api_volume and api_volume.strip():
                        volume_val = api_volume  # Use actual volume if provided
                    else:
                        volume_val = 0  # No volume data for indices
                        missing_volume_count += 1

                    cleaned_record = {
                        'exchange_name': 'NSE',
                        'stock_code': stock_code,
                        'date': date_part,
                        'time': time_part,
                        'open': record.get('open', ''),
                        'high': record.get('high', ''),
                        'low': record.get('low', ''),
                        'close': record.get('close', ''),
                        'volume': volume_val
                    }
                    initial_cleaned_data.append(cleaned_record)
                except Exception:
                    # skip malformed record but continue processing
                    continue

        cleaned_data = validate_and_clean_data(initial_cleaned_data)

        if cleaned_data:
            df = pd.DataFrame(cleaned_data)
            csv_filename = f'{stock_code}_historical_data_5min.csv'
            df.to_csv(csv_filename, index=False)

            print(f"[SUCCESS] Clean data saved to '{csv_filename}'")
            print(f"[INFO] Records processed: {len(cleaned_data)} (raw fetched: {total_records})")
            print(f"[INFO] Date range: {df['date'].min()} to {df['date'].max()}")
            print(f"[INFO] Volume missing/coercion issues (approx): {missing_volume_count} / {total_records}")

            print("\nData Quality Report:")

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