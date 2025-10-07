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

import requests
import json
import hashlib
from datetime import datetime, timezone, timedelta
import os
import pandas as pd
import csv

customerDetail_url = "https://api.icicidirect.com/breezeapi/api/v1/customerdetails"
historical_url = "https://api.icicidirect.com/breezeapi/api/v1/historicalcharts"

from dotenv import load_dotenv
load_dotenv()

stock_code = input("Enter Stock Symbol: ")
print(f"Selected stock: {stock_code}")

secret_key = os.getenv("SECRET_KEY")
appkey = os.getenv("APP_KEY") 
session_key = os.getenv("API_SESSION_TOKEN")

#env variables check
print("Environment Variables Check:")
print(f"SECRET_KEY: {'✓ Loaded' if secret_key else 'Missing'}")
print(f"APP_KEY: {'✓ Loaded' if appkey else 'Missing'}")  
print(f"SESSION_KEY: {'✓ Loaded' if session_key else 'Missing'}")
print()

if not all([secret_key, appkey, session_key]):
    print("Missing required environment variables. Please check your .env file.")
    print("Required variables: SECRET_KEY, APP_KEY, SESSION_KEY")
    exit(1)

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
        print("Error: Could not find session token in response")
        print(f"Available keys in response: {list(data.keys()) if data else 'None'}")
        if data and "Error" in data:
            print(f"API Error: {data['Error']}")
        exit(1)
        
except json.JSONDecodeError as e:
    print(f"JSON Decode Error: {e}")
    print(f"Raw response: {customerDetail_response.text}")
    exit(1)
except Exception as e:
    print(f"Unexpected error: {e}")
    exit(1)

print("\nFetching 5-minute interval data from April 1 to October 1, 2025...")

time_stamp = datetime.now(timezone.utc).isoformat()[:19] + '.000Z'

payload = json.dumps({
    "interval": "5minute",
    "from_date": "2025-04-01T10:20:00.000Z",  # April 1, 2025 (market opening)
    "to_date": "2025-10-01T15:30:00.000Z",    # October 1, 2025 (market closing)
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

response = requests.request("GET", historical_url, headers=headers, data=payload)

print("\n" + "="*60)
print("Processing API Response...")
print("="*60)

try:
    response_data = json.loads(response.text)
    
    if "Success" in response_data and response_data["Success"]:
        historical_data = response_data["Success"]
        
        cleaned_data = []
        
        for record in historical_data:
            datetime_str = record.get('datetime', '')
            
            if ' ' in datetime_str:
                date_part = datetime_str.split(' ')[0]
                time_part = datetime_str.split(' ')[1] if len(datetime_str.split(' ')) > 1 else ''
            else:
                date_part = datetime_str.split('T')[0] if 'T' in datetime_str else datetime_str[:10]
                time_part = datetime_str.split('T')[1].replace('Z', '') if 'T' in datetime_str else ''
            
            if time_part:
                try:
                    time_obj = datetime.strptime(time_part, "%H:%M:%S").time()
                    market_start = datetime.strptime("09:15:00", "%H:%M:%S").time()
                    market_end = datetime.strptime("15:30:00", "%H:%M:%S").time()
                    
                    if not (market_start <= time_obj <= market_end):
                        continue
                except:
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
        
        csv_filename = f'{stock_code}_historical_data_5min.csv'
        
        if cleaned_data:
            df = pd.DataFrame(cleaned_data)
            
            df.to_csv(csv_filename, index=False)
            
            df['time_only'] = pd.to_datetime(df['time'], format='%H:%M:%S').dt.time
            earliest_time = df['time_only'].min()
            latest_time = df['time_only'].max()
            
            print("\nSample data (first 5 rows):")
            print(df[['exchange_name', 'stock_code', 'date', 'time', 'open', 'high', 'low', 'close', 'volume']].head().to_string(index=False))
            
        else:
            print("No historical data found in response")
    
    else:
        print("Error in API response:")
        if "Error" in response_data:
            print(f"API Error: {response_data['Error']}")
        else:
            print(f"Unexpected response format: {response_data}")
            
except json.JSONDecodeError as e:
    print(f"JSON Decode Error: {e}")
    print(f"Raw response: {response.text[:500]}...")  # Show first 500 chars
except Exception as e:
    print(f"Unexpected error while processing data: {e}")
    print(f"Response status: {response.status_code}")
    print(f"Response text: {response.text[:500]}...")  # Show first 500 chars