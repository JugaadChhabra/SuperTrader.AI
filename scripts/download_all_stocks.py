#!/usr/bin/env python3
"""
Batch downloader for NSE stocks - COMBINED DATASET VERSION
Downloads historical 5-minute data for all stocks in stock_names_symbol.csv and combines into one training dataset
"""

import os
import sys
import json
import hashlib
import pandas as pd
from datetime import datetime, timezone, time
from typing import List, Dict
import requests

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from dotenv import load_dotenv
from indicators.technical import validate_and_clean_data

# Load environment variables
load_dotenv()

# Read stock symbols from CSV file
def load_stock_symbols():
    """Load all stock symbols from stock_names_symbol.csv"""
    csv_path = os.path.join(project_root, "stock_names_symbol.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Stock symbols file not found: {csv_path}")
    
    df = pd.read_csv(csv_path, header=None, names=['company_name', 'symbol', 'code'])
    
    # Extract just the symbols
    symbols = df['symbol'].tolist()
    
    print(f"📊 Loaded {len(symbols)} stock symbols from {csv_path}")
    print(f"📋 Sample symbols: {', '.join(symbols[:10])}...")
    
    return symbols, df

# API Configuration
ICICI_CUSTOMER_DETAIL_URL = "https://api.icicidirect.com/breezeapi/api/v1/customerdetails"
ICICI_HISTORICAL_URL = "https://api.icicidirect.com/breezeapi/api/v1/historicalcharts"

# Output files
COMBINED_OUTPUT_CSV = "NSE_AllStocks_historical_data_5min.csv"
COMBINED_OUTPUT_JSON = "NSE_AllStocks_historical_data_5min.json"

def get_session_token():
    """Get session token from ICICI API"""
    secret_key = os.getenv("SECRET_KEY")
    appkey = os.getenv("APP_KEY") 
    session_key = os.getenv("API_SESSION_TOKEN")
    
    if not all([secret_key, appkey, session_key]):
        raise Exception("Missing environment variables: SECRET_KEY, APP_KEY, API_SESSION_TOKEN")
    
    print("🔑 Fetching session token...")
    time_stamp = datetime.now(timezone.utc).isoformat()[:19] + '.000Z'

    payload = json.dumps({
        "SessionToken": session_key,
        "AppKey": appkey
    })

    headers = {
        'Content-Type': 'application/json',
    }

    response = requests.request("GET", ICICI_CUSTOMER_DETAIL_URL, 
                               headers=headers, data=payload)

    if response.status_code != 200:
        raise Exception(f"Failed to get session token: {response.status_code}")
    
    data = json.loads(response.text)
    
    if data and "Success" in data and "session_token" in data["Success"]:
        session_token = data["Success"]["session_token"]
        print(f"✅ Session token retrieved")
        return session_token, secret_key, appkey
    else:
        raise Exception(f"Could not find session token in response: {data}")

def download_stock_data(symbol: str, session_token: str, secret_key: str, appkey: str) -> List[Dict]:
    """Download data for a single stock"""
    print(f"📥 Downloading {symbol}...", end=" ", flush=True)
    
    time_stamp = datetime.now(timezone.utc).isoformat()[:19] + '.000Z'

    payload = json.dumps({
        "interval": "5minute",
        "from_date": "2025-04-01T10:20:00.000Z",  
        "to_date": "2025-10-01T15:30:00.000Z", 
        "stock_code": symbol,
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

    try:
        response = requests.request("GET", ICICI_HISTORICAL_URL, headers=headers, data=payload)
        
        if response.status_code != 200:
            print(f"❌ HTTP {response.status_code}")
            return []

        response_data = json.loads(response.text)
        
        if "Success" not in response_data or not response_data["Success"]:
            error_msg = response_data.get("Error", "No data")
            print(f"⚠️ {error_msg}")
            return []
        
        historical_data = response_data["Success"]
        
        # Process the data
        processed_records = []
        
        for record in historical_data:
            datetime_str = record.get("datetime", "")
            if datetime_str:
                try:
                    dt_obj = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
                    
                    # Keep only market hours
                    if not (time(9, 15) <= dt_obj.time() <= time(15, 30)):
                        continue
                    
                    date_part = dt_obj.strftime("%Y-%m-%d")
                    time_part = dt_obj.strftime("%H:%M:%S")
                    
                    # Handle volume - use actual value from API
                    api_volume = record.get('volume', '')
                    volume_val = api_volume if api_volume and api_volume.strip() else 0
                    
                    cleaned_record = {
                        'exchange_name': 'NSE',
                        'stock_code': symbol,
                        'date': date_part,
                        'time': time_part,
                        'open': record.get('open', ''),
                        'high': record.get('high', ''),
                        'low': record.get('low', ''),
                        'close': record.get('close', ''),
                        'volume': volume_val
                    }
                    processed_records.append(cleaned_record)
                    
                except Exception:
                    continue
        
        print(f"✅ {len(processed_records)} records")
        return processed_records
        
    except Exception as e:
        print(f"❌ Error: {str(e)[:50]}")
        return []

def save_data_optimized(all_data: List[List[Dict]], output_csv: str, output_json: str):
    """Save data in both CSV and JSON formats with optimization"""
    print(f"\n💾 Saving combined dataset...")
    
    # Flatten all data into one list
    combined_data = []
    for stock_data in all_data:
        combined_data.extend(stock_data)
    
    print(f"   📊 Total combined records: {len(combined_data):,}")
    
    # Clean and validate data
    print(f"   🧹 Cleaning and validating data...")
    cleaned_data = validate_and_clean_data(combined_data)
    
    if not cleaned_data:
        raise Exception("No valid data remaining after cleaning")
    
    # Convert to DataFrame
    df = pd.DataFrame(cleaned_data)
    
    # Sort by date, time, then symbol
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    df = df.sort_values(['datetime', 'stock_code']).drop('datetime', axis=1)
    
    # Save CSV
    csv_path = os.path.join(project_root, output_csv)
    df.to_csv(csv_path, index=False)
    
    # Save JSON (more compact and faster for ML)
    json_path = os.path.join(project_root, output_json)
    
    # Create optimized JSON structure
    json_data = {
        "metadata": {
            "total_records": len(cleaned_data),
            "date_range": {
                "start": df['date'].min(),
                "end": df['date'].max()
            },
            "stocks": df['stock_code'].nunique(),
            "stock_list": sorted(df['stock_code'].unique().tolist()),
            "generated_at": datetime.now().isoformat()
        },
        "data": cleaned_data  # Raw list of dictionaries for fast loading
    }
    
    with open(json_path, 'w') as f:
        json.dump(json_data, f, separators=(',', ':'))  # Compact JSON
    
    # File sizes
    csv_size = os.path.getsize(csv_path)
    json_size = os.path.getsize(json_path)
    
    print(f"   ✅ Saved CSV: {output_csv} ({csv_size:,} bytes, {csv_size/1024/1024:.2f} MB)")
    print(f"   ✅ Saved JSON: {output_json} ({json_size:,} bytes, {json_size/1024/1024:.2f} MB)")
    print(f"   📈 Final records: {len(cleaned_data):,}")
    print(f"   📅 Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Show stock distribution
    print(f"\n📊 Top 10 stocks by record count:")
    stock_counts = df['stock_code'].value_counts().head(10)
    for stock, count in stock_counts.items():
        print(f"   {stock}: {count:,} records")
    
    # Show sample data
    print(f"\n📋 Sample data:")
    print(df[['stock_code', 'date', 'time', 'open', 'high', 'low', 'close', 'volume']].head(5).to_string(index=False))
    
    return csv_path, json_path

def main():
    """Download and combine data for all stocks"""
    print(f"🚀 NSE ALL STOCKS DATA DOWNLOADER (COMBINED DATASET)")
    print(f"=" * 80)
    print(f"📅 Started: {datetime.now()}")
    print(f"💾 Output: CSV + JSON formats")
    print()
    
    try:
        # Load stock symbols
        stock_symbols, stocks_df = load_stock_symbols()
        
        print(f"\n🎯 Will attempt to download data for {len(stock_symbols)} stocks")
        user_confirm = input("Continue? (y/n): ").lower().strip()
        
        if user_confirm != 'y':
            print("❌ Download cancelled by user")
            return
        
        # Get session token
        session_token, secret_key, appkey = get_session_token()
        
        # Download data for all stocks
        print(f"\n📥 Starting bulk download...")
        all_stock_data = []
        successful_stocks = []
        failed_stocks = []
        
        for i, symbol in enumerate(stock_symbols, 1):
            try:
                print(f"[{i:3d}/{len(stock_symbols)}] ", end="", flush=True)
                
                stock_data = download_stock_data(symbol, session_token, secret_key, appkey)
                
                if stock_data:  # Only add if we got data
                    all_stock_data.append(stock_data)
                    successful_stocks.append(symbol)
                else:
                    failed_stocks.append(symbol)
                
                # Small delay to avoid rate limiting
                if i % 10 == 0:
                    print(f"\n   📊 Progress: {i}/{len(stock_symbols)} completed ({len(successful_stocks)} successful)")
                    
            except KeyboardInterrupt:
                print(f"\n🛑 Download interrupted by user at stock {i}")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error with {symbol}: {e}")
                failed_stocks.append(symbol)
                continue
        
        if not successful_stocks:
            raise Exception("No stocks downloaded successfully")
        
        # Save combined data
        csv_path, json_path = save_data_optimized(all_stock_data, COMBINED_OUTPUT_CSV, COMBINED_OUTPUT_JSON)
        
        # Final summary
        print(f"\n" + "=" * 80)
        print(f"🎉 DOWNLOAD COMPLETE!")
        print(f"=" * 80)
        print(f"✅ Successful stocks: {len(successful_stocks)}/{len(stock_symbols)} ({len(successful_stocks)/len(stock_symbols)*100:.1f}%)")
        
        if failed_stocks:
            print(f"❌ Failed stocks: {len(failed_stocks)}")
            if len(failed_stocks) <= 20:
                print(f"   Failed: {', '.join(failed_stocks)}")
            else:
                print(f"   Too many to list (first 10): {', '.join(failed_stocks[:10])}...")
        
        print(f"\n📁 Output files:")
        print(f"   - {csv_path}")
        print(f"   - {json_path}")
        
        print(f"\n🏁 Completed: {datetime.now()}")
        
        print(f"\n💡 Usage:")
        print(f"   CSV: Use with pandas: df = pd.read_csv('{COMBINED_OUTPUT_CSV}')")
        print(f"   JSON: Use with json: data = json.load(open('{COMBINED_OUTPUT_JSON}'))")
        print(f"   JSON is more compact and includes metadata")
        
    except KeyboardInterrupt:
        print(f"\n🛑 Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()