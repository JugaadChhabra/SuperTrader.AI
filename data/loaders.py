"""
Data Warehouse and Preprocessing for SuperTrader.AI
Handles loading, cleaning, and preprocessing historical OHLCV data from CSV files.

IMPORTANT: This module works with local CSV data warehouse only.
- For LIVE trading data, use agents/data_agent.py ICICI WebSocket  
- For HISTORICAL data, this module loads from local CSV files and cleans/preprocesses them

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

import requests
import json
import hashlib
from datetime import datetime, timezone, timedelta
import os
import pandas as pd
import csv

# API Configuration
customerDetail_url = "https://api.icicidirect.com/breezeapi/api/v1/customerdetails"
historical_url = "https://api.icicidirect.com/breezeapi/api/v1/historicalcharts"

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

stock_code = input("Enter Stock Symbol: ")
print(f"Selected stock: {stock_code}")

# Credentials - Replace with your actual keys
secret_key = os.getenv("SECRET_KEY")
appkey = os.getenv("APP_KEY") 
session_key = os.getenv("API_SESSION_TOKEN")

# Debug: Check if environment variables are loaded
print("🔐 Environment Variables Check:")
print(f"SECRET_KEY: {'✓ Loaded' if secret_key else '❌ Missing'}")
print(f"APP_KEY: {'✓ Loaded' if appkey else '❌ Missing'}")  
print(f"SESSION_KEY: {'✓ Loaded' if session_key else '❌ Missing'}")
print()

if not all([secret_key, appkey, session_key]):
    print("❌ Missing required environment variables. Please check your .env file.")
    print("Required variables: SECRET_KEY, APP_KEY, SESSION_KEY")
    exit(1)

# Step 1: Get Session Token
print("Fetching session token...")
time_stamp = datetime.now(timezone.utc).isoformat()[:19] + '.000Z'

customerDetail_payload = json.dumps({
    "SessionToken": session_key,
    "AppKey": appkey
})

customerDetail_headers = {
    'Content-Type': 'application/json',
}

customerDetail_response = requests.request("GET", customerDetail_url, 
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
        print("❌ Error: Could not find session token in response")
        print(f"Available keys in response: {list(data.keys()) if data else 'None'}")
        if data and "Error" in data:
            print(f"API Error: {data['Error']}")
        exit(1)
        
except json.JSONDecodeError as e:
    print(f"❌ JSON Decode Error: {e}")
    print(f"Raw response: {customerDetail_response.text}")
    exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    exit(1)

# Step 2: Fetch Historical Data (5-minute intervals)
print("\nFetching 5-minute interval data from April 1 to October 1, 2025...")

# Update timestamp for the second request
time_stamp = datetime.now(timezone.utc).isoformat()[:19] + '.000Z'

# Payload for 5-minute interval data
payload = json.dumps({
    "interval": "5minute",
    "from_date": "2025-04-01T10:20:00.000Z",  # April 1, 2025 (market opening)
    "to_date": "2025-10-01T15:30:00.000Z",    # October 1, 2025 (market closing)
    "stock_code": stock_code,
    "exchange_code": "NSE",
    "product_type": "cash"
}, separators=(',', ':'))

# Generate checksum
checksum = hashlib.sha256((time_stamp + payload + secret_key).encode("utf-8")).hexdigest()

# Headers for historical data request
headers = {
    'Content-Type': 'application/json',
    'X-Checksum': 'token ' + checksum,
    'X-Timestamp': time_stamp,
    'X-AppKey': appkey,
    'X-SessionToken': session_token
}

# Make the request
response = requests.request("GET", historical_url, headers=headers, data=payload)

# Process and clean the response
print("\n" + "="*60)
print("Processing API Response...")
print("="*60)

try:
    response_data = json.loads(response.text)
    
    # Check if response contains data
    if "Success" in response_data and response_data["Success"]:
        historical_data = response_data["Success"]
        
        # Prepare cleaned data for CSV (Market Hours Only)
        cleaned_data = []
        
        for record in historical_data:
            # Extract and clean each record
            datetime_str = record.get('datetime', '')
            
            # Handle IST timestamps from API (no conversion needed)
            if ' ' in datetime_str:
                # Format: "2025-04-01 09:15:00" (IST from API)
                date_part = datetime_str.split(' ')[0]
                time_part = datetime_str.split(' ')[1] if len(datetime_str.split(' ')) > 1 else ''
            else:
                # Fallback for other formats
                date_part = datetime_str.split('T')[0] if 'T' in datetime_str else datetime_str[:10]
                time_part = datetime_str.split('T')[1].replace('Z', '') if 'T' in datetime_str else ''
            
            # Filter for market hours only (9:15 AM to 3:30 PM IST)
            if time_part:
                try:
                    time_obj = datetime.strptime(time_part, "%H:%M:%S").time()
                    market_start = datetime.strptime("09:15:00", "%H:%M:%S").time()
                    market_end = datetime.strptime("15:30:00", "%H:%M:%S").time()
                    
                    # Skip records outside market hours
                    if not (market_start <= time_obj <= market_end):
                        continue
                except:
                    # If time parsing fails, skip this record
                    continue
            
            cleaned_record = {
                'exchange_name': 'NSE',  # From our request
                'stock_code': stock_code,  # From user input
                'date': date_part,  # IST date from API
                'time': time_part,  # IST time from API
                'open': record.get('open', ''),
                'high': record.get('high', ''),
                'low': record.get('low', ''),
                'close': record.get('close', ''),
                'volume': record.get('volume', '')
            }
            cleaned_data.append(cleaned_record)
        
        # Save to CSV
        csv_filename = f'{stock_code}_historical_data_5min.csv'
        
        if cleaned_data:
            # Convert to DataFrame for better CSV handling
            df = pd.DataFrame(cleaned_data)
            
            # Save to CSV
            df.to_csv(csv_filename, index=False)
            
            print(f"✅ Success! Market hours data saved to '{csv_filename}'")
            print(f"📊 Records processed (9:15 AM - 3:30 PM IST only): {len(cleaned_data)}")
            print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
            
            # Market hours summary
            print("\n🕐 Market Hours Data (IST):")
            print(f"   First timestamp: {df.iloc[0]['date']} {df.iloc[0]['time']}")
            print(f"   Last timestamp:  {df.iloc[-1]['date']} {df.iloc[-1]['time']}")
            print(f"   Filtered to: 09:15:00 - 15:30:00 IST only")
            
            # Time range analysis
            df['time_only'] = pd.to_datetime(df['time'], format='%H:%M:%S').dt.time
            earliest_time = df['time_only'].min()
            latest_time = df['time_only'].max()
            
            print(f"\n📈 Trading Session Summary:")
            print(f"   Earliest time in data: {earliest_time}")
            print(f"   Latest time in data:   {latest_time}")
            print(f"   Total trading days:    {df['date'].nunique()}")
            print(f"   Average records/day:   {len(df) // df['date'].nunique():.0f}")
            
            # Show sample data
            print("\n🔍 Sample data (first 5 rows):")
            print(df[['exchange_name', 'stock_code', 'date', 'time', 'open', 'high', 'low', 'close', 'volume']].head().to_string(index=False))
            
        else:
            print("⚠️  No historical data found in response")
    
    else:
        print("❌ Error in API response:")
        if "Error" in response_data:
            print(f"API Error: {response_data['Error']}")
        else:
            print(f"Unexpected response format: {response_data}")
            
except json.JSONDecodeError as e:
    print(f"❌ JSON Decode Error: {e}")
    print(f"Raw response: {response.text[:500]}...")  # Show first 500 chars
except Exception as e:
    print(f"❌ Unexpected error while processing data: {e}")
    print(f"Response status: {response.status_code}")
    print(f"Response text: {response.text[:500]}...")  # Show first 500 chars