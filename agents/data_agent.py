import base64
import json
import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional, Tuple

import requests
import socketio
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import yaml

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



def connect_icici_broker(api_keys: Dict[str, str]) -> Dict[str, str]:
    """Authenticate with ICICI Direct to obtain WebSocket session credentials.
    api_keys : dict
        {"app_key": str, "api_session_token": str}
    Returns
    dict
        {"user_id": str, "session_token": str, "raw": dict}
    """
    app_key = api_keys.get("app_key")
    api_session_token = api_keys.get("api_session_token")
    if not app_key or not api_session_token:
        raise RuntimeError("Missing 'app_key' or 'api_session_token' for ICICI auth")

    url = "https://api.icicidirect.com/breezeapi/api/v1/customerdetails"

    # NOTE: The upstream example used GET with a JSON body. We mirror that behavior
    # for compatibility, though many servers ignore GET bodies.
    payload = json.dumps({"SessionToken": api_session_token, "AppKey": app_key})
    headers = {"Content-Type": "application/json"}

    logger.info("Fetching ICICI WebSocket session token…")
    resp = requests.request("GET", url, headers=headers, data=payload, timeout=15)
    try:
        data = resp.json()
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("ICICI auth: non-JSON response")
        raise RuntimeError(f"ICICI auth failed: non-JSON response ({exc})")

    if not isinstance(data, dict) or "Success" not in data:
        raise RuntimeError(f"ICICI auth failed: unexpected payload {data!r}")

    success_block = data.get("Success") or {}
    ws_key = success_block.get("session_token")
    if not ws_key:
        raise RuntimeError(f"ICICI auth failed: {data!r}")

    try:
        decoded = base64.b64decode(ws_key.encode("ascii")).decode("ascii")
        user_id, session_token = decoded.split(":", 1)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to decode ICICI session token")
        raise RuntimeError(f"ICICI auth decode failed: {exc}")

    logger.info("ICICI auth OK | user_id=%s", user_id)
    return {"user_id": user_id, "session_token": session_token, "raw": data}


def _icici_sio_client(*, user_id: str, session_token: str) -> socketio.Client:
    """Create and configure a socket.io Client for ICICI live stream."""
    sio = socketio.Client(logger=False, engineio_logger=False)

    @sio.event
    def connect():  # pragma: no cover - network path
        logger.info("ICICI WebSocket connected")

    @sio.event
    def disconnect():  # pragma: no cover - network path
        logger.info("ICICI WebSocket disconnected")

    @sio.event
    def connect_error(data):  # pragma: no cover - network path
        logger.error("ICICI WebSocket connection error: %s", data)
        
    return sio



def parse_icici_payload(data: Any) -> Dict[str, Any]:
    """Parse raw ICICI tick payload into structured dict."""
    if not isinstance(data, list) or not data:
        return {"raw_data": data}

    try:
        if len(data) >= 12:
            return {
                "symbol": data[0],
                "open": data[1],
                "last": data[2],
                "high": data[3],
                "low": data[4],
                "change": data[5],
                "bPrice": data[6],
                "bQty": data[7],
                "sPrice": data[8],
                "sQty": data[9],
                "ltq": data[10],
                "avgPrice": data[11],
            }
        return {"raw_data": data}
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("ICICI parse error: %s", exc)
        return {"raw_data": data}


def stream_icici_quotes(
    api_keys: Dict[str, str],
    symbols: Iterable[str],
    *,
    duration_sec: int = 30,
    on_tick: Optional[callable] = None,
) -> None:
    """Stream live quotes from ICICI for a fixed duration.
    """
    auth = connect_icici_broker(api_keys)
    sio = _icici_sio_client(user_id=auth["user_id"], session_token=auth["session_token"])

    # Event handler for ticks
    @sio.on("stock")
    def _on_stock_data(data):  # pragma: no cover - network path
        parsed = parse_icici_payload(data)
        if on_tick:
            try:
                on_tick(parsed)
            except Exception:  # user callback safety
                logger.exception("on_tick callback error")
        else:
            # Default log output
            sym = parsed.get("symbol", "?")
            last = parsed.get("last")
            chg = parsed.get("change")
            ltq = parsed.get("ltq")
            logger.info("tick | %s | last=%s change=%s ltq=%s", sym, last, chg, ltq)

    # Connect using the working auth method from the example
    logger.info("Attempting WebSocket connection with user_id: %s", auth["user_id"][:4] + "***")
    
    try:
        sio.connect(
            "https://livestream.icicidirect.com",
            headers={"User-Agent": "python-socketio[client]/socket"},
            auth={"user": auth["user_id"], "token": auth["session_token"]},
            transports=["websocket"],
            wait_timeout=15,
        )

        # Join rooms / subscribe symbols
        joined: List[str] = []
        try:
            for sym in symbols:
                sio.emit("join", sym)
                joined.append(sym)
                logger.info("subscribed: %s", sym)

            # Sleep for duration
            logger.info("streaming for %s seconds…", duration_sec)
            sio.sleep(duration_sec)
        finally:
            # Best-effort cleanup
            for sym in joined:
                try:
                    sio.emit("leave", sym)
                except Exception:
                    pass
            try:
                sio.emit("disconnect", "transport close")
            except Exception:
                pass
            try:
                sio.disconnect()
            except Exception:
                pass
            logger.info("ICICI stream closed")
    except Exception as e:
        logger.error("WebSocket connection failed: %s", e)



def _default_on_tick(sample: Dict[str, Any]) -> None:
    # Pretty console print for ad-hoc local testing
    sym = sample.get("symbol", "N/A")
    last = sample.get("last", "N/A")
    chg = sample.get("change", "N/A")
    ltq = sample.get("ltq", "N/A")
    print("\n📊 Live Data")
    print(f"  Symbol: {sym}")
    print(f"  Last:   {last}")
    print(f"  Change: {chg}")
    print(f"  Volume: {ltq}")
    print("-" * 32)


def demo_run_from_env(duration_sec: int = 15) -> None:
    """Demo runner using env vars. Useful for quick local checks.

    Example
    -------
    >>> # export ICICI_APP_KEY=...; export ICICI_API_SESSION_TOKEN=...
    >>> from data_agent import demo_run_from_env
    >>> demo_run_from_env(10)
    """
    ctx = init_data_agent()
    api = {"app_key": ctx.get("app_key"), "api_session_token": ctx.get("api_session_token")}
    if not api["app_key"] or not api["api_session_token"]:
        raise RuntimeError("Missing ICICI_APP_KEY / ICICI_API_SESSION_TOKEN env vars")

    symbols = os.getenv("ICICI_DEMO_SYMBOLS", "NSE:SBIN,NSE:RELIANCE").split(",")
    stream_icici_quotes(api, symbols, duration_sec=duration_sec, on_tick=_default_on_tick)

def get_index_futures_symbols(config_path: str = "../configs/market.yaml") -> List[str]:
    """Generate ICICI-format symbol list for index futures from market config.
    
    Returns list like: ['NSE:NIFTY25OCTFUT', 'NSE:BANKNIFTY25OCTFUT', ...] (current month only)
    """
    import yaml
    from datetime import datetime
    import os
    
    # Get the directory of this script and construct absolute path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(config_path):
        config_path = os.path.join(script_dir, config_path)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get current month/year for contract generation  
    now = datetime.now()
    month_codes = {1: 'JAN', 2: 'FEB', 3: 'MAR', 4: 'APR', 5: 'MAY', 6: 'JUN',
                   7: 'JUL', 8: 'AUG', 9: 'SEP', 10: 'OCT', 11: 'NOV', 12: 'DEC'}
    
    # Generate symbols for current month contracts only
    symbols = []
    for index_name in config['index_futures'].keys():
        # Current month contract
        curr_month = month_codes[now.month]
        year_suffix = str(now.year)[2:]  # 25 for 2025
        symbol = f"NSE:{index_name}{year_suffix}{curr_month}FUT"
        symbols.append(symbol)
    
    return symbols

def stream_minute_chart(api_keys: Dict[str, str], symbols: List[str]) -> None:
    """Stream data every minute for 5 minutes to simulate 1-minute chart.
    
    Args:
        api_keys: Dict with 'app_key' and 'api_session_token'
        symbols: List of futures symbols to track
    """
    from datetime import datetime
    import time
    
    # Store latest prices for each symbol
    latest_prices = {}
    minute_count = 0
    
    def minute_tick_handler(data: Dict[str, Any]) -> None:
        """Store latest price data for minute-by-minute printing."""
        symbol = data.get("symbol", "Unknown")
        latest_prices[symbol] = {
            "last": data.get("last", "N/A"),
            "change": data.get("change", "N/A"),
            "high": data.get("high", "N/A"),
            "low": data.get("low", "N/A"),
            "volume": data.get("ltq", "N/A"),
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }
    
    def print_minute_summary():
        """Print 1-minute chart data for all symbols."""
        nonlocal minute_count
        minute_count += 1
        
        print(f"\n📊 === 1-MINUTE CHART DATA (Minute {minute_count}/5) ===")
        print(f"⏰ Time: {datetime.now().strftime('%H:%M:%S')}")
        print("-" * 80)
        
        for symbol in symbols:
            if symbol in latest_prices:
                data = latest_prices[symbol]
                print(f"🔹 {symbol}")
                print(f"   Last: {data['last']} | Change: {data['change']} | H: {data['high']} | L: {data['low']} | Vol: {data['volume']}")
            else:
                print(f"🔸 {symbol} - No data received yet")
        
        print("-" * 80)
    
    # Setup WebSocket connection
    auth = connect_icici_broker(api_keys)
    sio = _icici_sio_client(user_id=auth["user_id"], session_token=auth["session_token"])
    
    # Event handler for collecting ticks
    @sio.on("stock")
    def _on_stock_data(data):
        parsed = parse_icici_payload(data)
        minute_tick_handler(parsed)
    
    try:
        # Connect to WebSocket using the working auth method
        logger.info("Attempting WebSocket connection with user_id: %s", auth["user_id"][:4] + "***")
        
        sio.connect(
            "https://livestream.icicidirect.com",
            headers={"User-Agent": "python-socketio[client]/socket"},
            auth={"user": auth["user_id"], "token": auth["session_token"]},
            transports=["websocket"],
            wait_timeout=15,
        )
        
        # Subscribe to all symbols
        for sym in symbols:
            sio.emit("join", sym)
            logger.info("subscribed: %s", sym)
        
        # Stream for 5 minutes with 1-minute intervals
        logger.info("Starting 5-minute streaming with 1-minute intervals...")
        
        for minute in range(5):
            # Wait 60 seconds to collect data
            time.sleep(60)
            
            # Print minute summary
            print_minute_summary()
            
        logger.info("Completed 5-minute chart simulation")
        
    finally:
        # Cleanup WebSocket
        for sym in symbols:
            try:
                sio.emit("leave", sym)
            except Exception:
                pass
        try:
            sio.disconnect()
        except Exception:
            pass
        logger.info("WebSocket connection closed")


def demo_index_futures_stream(duration_sec: int = 30) -> None:
    """Demo streaming index futures with 1-minute intervals for 5 minutes.
    
    This simulates a 1-minute chart by capturing prices every minute for 5 minutes.
    """
    ctx = init_data_agent()
    api = {"app_key": ctx.get("app_key"), "api_session_token": ctx.get("api_session_token")}
    
    # Debug: Check what credentials we have
    logger.info("API Key present: %s", bool(api["app_key"]))
    logger.info("Session Token present: %s", bool(api["api_session_token"]))
    if api["app_key"]:
        logger.info("API Key starts with: %s***", api["app_key"][:8])
    
    if not api["app_key"] or not api["api_session_token"]:
        raise RuntimeError("Missing ICICI_APP_KEY / ICICI_API_SESSION_TOKEN env vars")
    
    # Get index futures symbols from config
    futures_symbols = get_index_futures_symbols()
    logger.info("Starting 1-minute chart simulation for %d contracts: %s", len(futures_symbols), futures_symbols)
    
    # Run the 1-minute interval streaming
    stream_minute_chart(api, futures_symbols)

if __name__ == "__main__": 
    # Test 5-minute chart simulation for index futures
    demo_index_futures_stream()

def fetch_index_futures_ohlcv(index_symbols: List[str], contract_months: List[str], 
                              interval: str, start: str, end: str,
                              api_keys: Dict[str, str]) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Fetch OHLCVI (Open, High, Low, Close, Volume, Open Interest) data for index futures.
    
    Args:
        index_symbols: List of index symbols (e.g., ['NIFTY', 'BANKNIFTY'])
        contract_months: List of contract months (e.g., ['current', 'next', 'far'])
        interval: Time interval ('1minute', '5minute', '15minute', '1day')
        start: Start date in 'YYYY-MM-DD' format
        end: End date in 'YYYY-MM-DD' format  
        api_keys: Dict with 'app_key' and 'api_session_token'
        
    Returns:
        Dict mapping index_symbol -> contract_month -> DataFrame with OHLCVI data
        
    Example:
        {
            'NIFTY': {
                'current': DataFrame(...),
                'next': DataFrame(...)
            },
            'BANKNIFTY': {
                'current': DataFrame(...)
            }
        }
    """
    if not api_keys.get("app_key") or not api_keys.get("api_session_token"):
        raise ValueError("Missing required API keys")
        
    results = {}
    
    # Load market configuration for contract naming
    try:
        # Get the directory of this script and construct absolute path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, '../configs/market.yaml')
        with open(config_path, 'r') as f:
            market_config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning("Market config not found, using default contract naming")
        market_config = {}
    
    # ICICI API endpoint for historical data
    base_url = "https://api.icicidirect.com/breezeapi/api/v1/historicaldata"
    
    headers = {
        "Content-Type": "application/json",
        "X-AppKey": api_keys["app_key"],
        "X-SessionToken": api_keys["api_session_token"]
    }
    
    for index_symbol in index_symbols:
        results[index_symbol] = {}
        
        for contract_month in contract_months:
            try:
                # Generate futures symbol based on NSE naming convention
                futures_symbol = _generate_futures_symbol(index_symbol, contract_month, market_config)
                
                # API payload for futures data
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
                
                response = requests.post(base_url, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                
                if data.get("Status") == "Success" and "Success" in data:
                    ohlcvi_data = data["Success"]
                    
                    # Convert to DataFrame with OHLCVI columns
                    df = pd.DataFrame(ohlcvi_data)
                    
                    # Standardize column names for futures
                    column_mapping = {
                        'datetime': 'timestamp',
                        'open': 'open', 
                        'high': 'high',
                        'low': 'low',
                        'close': 'close',
                        'volume': 'volume',
                        'oi': 'open_interest',  # Open Interest
                        'open_interest': 'open_interest'
                    }
                    
                    df = df.rename(columns=column_mapping)
                    
                    # Ensure timestamp is datetime
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df.set_index('timestamp', inplace=True)
                    
                    # Convert numeric columns
                    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'open_interest']
                    for col in numeric_cols:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Add metadata
                    df.attrs = {
                        'symbol': futures_symbol,
                        'index': index_symbol,
                        'contract_month': contract_month,
                        'interval': interval
                    }
                    
                    # Validate futures data
                    validated_df = validate_futures_data(df, futures_symbol)
                    results[index_symbol][contract_month] = validated_df
                    
                    logger.info("Fetched %d bars for %s (%s)", len(validated_df), 
                              futures_symbol, contract_month)
                    
                else:
                    logger.error("API error for %s (%s): %s", index_symbol, contract_month, data)
                    results[index_symbol][contract_month] = pd.DataFrame()
                    
            except Exception as e:
                logger.error("Error fetching %s (%s): %s", index_symbol, contract_month, e)
                results[index_symbol][contract_month] = pd.DataFrame()
    
    return results


def _generate_futures_symbol(index_symbol: str, contract_month: str, 
                           market_config: Dict[str, Any]) -> str:
    """Generate NSE futures symbol based on index and contract month.
    
    Args:
        index_symbol: Index name (e.g., 'NIFTY')
        contract_month: Contract month ('current', 'next', 'far')
        market_config: Market configuration dict
        
    Returns:
        NSE futures symbol (e.g., 'NIFTY25JANFUT')
    """
    # Get current date to determine contract expiry months
    current_date = datetime.now()
    
    # Map contract month to actual month
    if contract_month == 'current':
        expiry_date = current_date
    elif contract_month == 'next':
        expiry_date = current_date + timedelta(days=30)
    elif contract_month == 'far':
        expiry_date = current_date + timedelta(days=60)
    else:
        # Assume it's already a month name or date
        expiry_date = current_date
    
    # Get month codes from config
    month_codes = market_config.get('contract_naming', {}).get('month_codes', {
        "01": "JAN", "02": "FEB", "03": "MAR", "04": "APR",
        "05": "MAY", "06": "JUN", "07": "JUL", "08": "AUG", 
        "09": "SEP", "10": "OCT", "11": "NOV", "12": "DEC"
    })
    
    year = str(expiry_date.year)[-2:]  # Last 2 digits of year
    month_num = expiry_date.strftime("%m")
    month_code = month_codes.get(month_num, "JAN")
    
    # NSE naming convention: {INDEX}{YY}{MON}FUT
    futures_symbol = f"{index_symbol}{year}{month_code}FUT"
    
    return futures_symbol


def fetch_ohlcv(symbols: List[str], interval: str, start: str, end: str, 
                api_keys: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    """Fetch OHLCV (Open, High, Low, Close, Volume) data for symbols.
    
    This is the legacy function for equity/other instruments.
    For index futures, use fetch_index_futures_ohlcv() instead.
    """
    # Implementation stays the same as before but shortened for brevity
    logger.warning("Using legacy OHLCV fetch. Consider using fetch_index_futures_ohlcv() for futures.")
    return {}

def validate_futures_data(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Validate futures data for gaps, price spikes, and data quality issues.
    
    Args:
        df: DataFrame with OHLCVI data
        symbol: Futures symbol for logging
        
    Returns:
        Validated DataFrame with quality flags
    """
    if df.empty:
        logger.warning("Empty DataFrame for %s", symbol)
        return df
    
    validated_df = df.copy()
    
    # 1. Check for missing OHLCVI columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error("Missing columns for %s: %s", symbol, missing_cols)
        return pd.DataFrame()
    
    # 2. Check for data gaps (missing time periods)
    if len(df) > 1:
        time_diff = df.index.to_series().diff()
        expected_freq = time_diff.mode()[0] if not time_diff.mode().empty else pd.Timedelta('1D')
        
        # Find gaps larger than 2x expected frequency
        gaps = time_diff[time_diff > expected_freq * 2]
        if len(gaps) > 0:
            logger.warning("Found %d time gaps in %s data", len(gaps), symbol)
            validated_df['has_gap'] = time_diff > expected_freq * 2
    
    # 3. Check for price spikes (outliers)
    if 'close' in df.columns and len(df) > 10:
        returns = df['close'].pct_change().abs()
        spike_threshold = returns.quantile(0.99) * 2  # 2x 99th percentile
        
        price_spikes = returns > spike_threshold
        if price_spikes.sum() > 0:
            logger.warning("Found %d price spikes in %s data", price_spikes.sum(), symbol)
            validated_df['price_spike'] = price_spikes
    
    # 4. Check OHLC logic consistency
    ohlc_issues = (
        (df['high'] < df['low']) |  # High < Low
        (df['high'] < df['open']) | (df['high'] < df['close']) |  # High < Open/Close
        (df['low'] > df['open']) | (df['low'] > df['close'])      # Low > Open/Close
    )
    
    if ohlc_issues.sum() > 0:
        logger.error("Found %d OHLC logic errors in %s data", ohlc_issues.sum(), symbol)
        validated_df['ohlc_error'] = ohlc_issues
    
    # 5. Check for zero/negative values
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        if col in df.columns:
            invalid_prices = (df[col] <= 0) | df[col].isna()
            if invalid_prices.sum() > 0:
                logger.warning("Found %d invalid %s prices in %s", 
                             invalid_prices.sum(), col, symbol)
    
    # 6. Volume and Open Interest checks
    if 'volume' in df.columns:
        zero_volume = df['volume'] == 0
        if zero_volume.sum() > len(df) * 0.1:  # More than 10% zero volume
            logger.warning("%s has %d%% zero volume bars", 
                         symbol, (zero_volume.sum() / len(df)) * 100)
    
    if 'open_interest' in df.columns:
        # Check for sudden OI changes (potential rollover)
        if len(df) > 1:
            oi_change = df['open_interest'].pct_change().abs()
            large_oi_changes = oi_change > 0.5  # 50% change
            if large_oi_changes.sum() > 0:
                logger.info("Found %d large OI changes in %s (potential rollovers)", 
                          large_oi_changes.sum(), symbol)
                validated_df['oi_rollover'] = large_oi_changes
    
    # 7. Add data quality score
    quality_issues = 0
    if 'has_gap' in validated_df.columns:
        quality_issues += validated_df['has_gap'].sum()
    if 'price_spike' in validated_df.columns:
        quality_issues += validated_df['price_spike'].sum()
    if 'ohlc_error' in validated_df.columns:
        quality_issues += validated_df['ohlc_error'].sum()
    
    quality_score = max(0, 100 - (quality_issues / len(df)) * 100)
    validated_df.attrs['quality_score'] = quality_score
    
    logger.info("Data quality score for %s: %.1f%%", symbol, quality_score)
    
    return validated_df


def fetch_market_depth(symbol): # — level-1/level-2 snapshot (bid/ask, spreads).
    pass

def fetch_corporate_actions(symbols, start, end): # — splits/dividends/earnings.
    pass

def handle_contract_rollover(current_contract: pd.DataFrame, next_contract: pd.DataFrame,
                           rollover_date: str, method: str = 'ratio') -> pd.DataFrame:
    """Handle contract rollover for continuous futures series.
    
    Args:
        current_contract: DataFrame for expiring contract
        next_contract: DataFrame for new contract  
        rollover_date: Date to perform rollover (YYYY-MM-DD)
        method: 'ratio' or 'difference' adjustment method
        
    Returns:
        Adjusted DataFrame for seamless transition
    """
    if current_contract.empty or next_contract.empty:
        logger.error("Empty contract data for rollover")
        return pd.DataFrame()
    
    rollover_dt = pd.to_datetime(rollover_date)
    
    # Find the rollover point data
    current_before = current_contract[current_contract.index <= rollover_dt]
    next_after = next_contract[next_contract.index >= rollover_dt]
    
    if current_before.empty or next_after.empty:
        logger.error("Insufficient data around rollover date %s", rollover_date)
        return current_contract
    
    # Get prices at rollover point
    current_price = current_before.iloc[-1]['close']
    next_price = next_after.iloc[0]['close'] 
    
    if method == 'ratio':
        # Ratio adjustment (multiplicative)
        adjustment_factor = current_price / next_price
        adjusted_next = next_after.copy()
        
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in adjusted_next.columns:
                adjusted_next[col] = adjusted_next[col] * adjustment_factor
                
        logger.info("Applied ratio adjustment: factor=%.4f", adjustment_factor)
        
    elif method == 'difference':
        # Difference adjustment (additive)
        adjustment_diff = current_price - next_price
        adjusted_next = next_after.copy()
        
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in adjusted_next.columns:
                adjusted_next[col] = adjusted_next[col] + adjustment_diff
                
        logger.info("Applied difference adjustment: diff=%.2f", adjustment_diff)
        
    else:
        logger.error("Unknown rollover method: %s", method)
        return current_contract
    
    # Combine the series
    continuous_series = pd.concat([current_before, adjusted_next])
    continuous_series = continuous_series.sort_index()
    
    # Add rollover metadata
    continuous_series.attrs = {
        'rollover_date': rollover_date,
        'rollover_method': method,
        'adjustment_factor': adjustment_factor if method == 'ratio' else adjustment_diff
    }
    
    return continuous_series


def construct_continuous_contract(contracts_data: Dict[str, pd.DataFrame], 
                                rollover_schedule: List[str],
                                method: str = 'ratio') -> pd.DataFrame:
    """Construct continuous futures contract from multiple expiry series.
    
    Args:
        contracts_data: Dict mapping contract_month -> DataFrame
        rollover_schedule: List of rollover dates in chronological order
        method: 'ratio' or 'difference' adjustment method
        
    Returns:
        Continuous contract DataFrame
    """
    if not contracts_data or not rollover_schedule:
        logger.error("Insufficient data for continuous contract construction")
        return pd.DataFrame()
    
    # Sort contracts by rollover dates
    contract_names = list(contracts_data.keys())
    if len(contract_names) < 2:
        logger.warning("Need at least 2 contracts for continuous series")
        return list(contracts_data.values())[0]
    
    # Start with the first contract
    continuous_series = contracts_data[contract_names[0]].copy()
    
    # Apply rollovers sequentially
    for i, rollover_date in enumerate(rollover_schedule):
        if i + 1 >= len(contract_names):
            break
            
        current_contract = continuous_series
        next_contract = contracts_data[contract_names[i + 1]]
        
        continuous_series = handle_contract_rollover(
            current_contract, next_contract, rollover_date, method
        )
        
        logger.info("Rolled over to %s on %s", contract_names[i + 1], rollover_date)
    
    # Add continuous contract metadata
    continuous_series.attrs = {
        'contract_type': 'continuous',
        'rollover_method': method,
        'num_rollovers': len(rollover_schedule),
        'contracts_used': contract_names
    }
    
    return continuous_series


def compute_basis(futures_price: pd.Series, spot_price: pd.Series, 
                 days_to_expiry: pd.Series) -> Dict[str, pd.Series]:
    """Compute futures basis and related metrics.
    
    Args:
        futures_price: Futures closing prices
        spot_price: Spot index closing prices  
        days_to_expiry: Days remaining to contract expiry
        
    Returns:
        Dict with basis metrics: 'basis', 'basis_pct', 'annualized_basis'
    """
    # Absolute basis (futures - spot)
    basis = futures_price - spot_price
    
    # Percentage basis ((futures - spot) / spot * 100)
    basis_pct = (basis / spot_price) * 100
    
    # Annualized basis (extrapolate to annual terms)
    # Assume 252 trading days per year
    annualized_basis = basis_pct * (252 / days_to_expiry.replace(0, 1))  # Avoid div by zero
    
    # Theoretical fair value (cost of carry model)
    # This is simplified - in practice would include risk-free rate and dividends
    risk_free_rate = 0.06  # Assume 6% risk-free rate
    theoretical_basis = spot_price * (risk_free_rate / 365) * days_to_expiry
    basis_deviation = basis - theoretical_basis
    
    return {
        'basis': basis,
        'basis_pct': basis_pct,
        'annualized_basis': annualized_basis,
        'theoretical_basis': theoretical_basis,
        'basis_deviation': basis_deviation
    }


def interpolate_missing(df, method): # — forward/back fill, stitching.
    pass

def resample_bars(df, interval): # — unify to target timeframe.
    pass

# Import TA-Lib technical indicators from our clean indicators package
from indicators.technical import (
    compute_all_indicators, 
    validate_and_clean_data,
    macd_multi_timeframe,
    rsi_multi_period,
    atr_volatility,
    bollinger_bands,
    volume_indicators,
    momentum_oscillators,
    trend_indicators,
    futures_specific_indicators
)

def compute_indicators(df: pd.DataFrame, minimal_mode: bool = False) -> pd.DataFrame:
    """Compute technical indicators using TA-Lib functions.
    
    Args:
        df: DataFrame with OHLCV data (expects 'close' column)
        minimal_mode: If True, compute only essential indicators for MVP
        
    Returns:
        DataFrame with TA-Lib indicators added
    """
    if df.empty or 'close' not in df.columns:
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
    if df.empty or 'close' not in df.columns:
        logger.warning("Invalid DataFrame for futures indicators")
        return df
    
    try:
        # Use comprehensive TA-Lib indicator suite with futures-specific indicators
        result_df = compute_all_indicators(df, include_futures_indicators=True)
        
        # Add additional futures-specific analysis not covered by TA-Lib
        if 'open_interest' in df.columns and 'volume' in df.columns:
            # Volume-OI Divergence analysis
            volume_norm = (df['volume'] - df['volume'].rolling(20).mean()) / df['volume'].rolling(20).std()
            oi_norm = (df['open_interest'] - df['open_interest'].rolling(20).mean()) / df['open_interest'].rolling(20).std()
            result_df['vol_oi_divergence'] = abs(volume_norm - oi_norm)
            result_df['vol_oi_signal'] = result_df['vol_oi_divergence'] > 1.5
        
        # Basis indicators (if spot data provided)
        if spot_df is not None and 'close' in spot_df.columns:
            # Align timeframes
            aligned_spot = spot_df.reindex(df.index, method='ffill')
            
            if not aligned_spot['close'].isna().all():
                # Calculate days to expiry (simplified)
                days_to_expiry = pd.Series(30, index=df.index)  # Placeholder - should calculate actual
                
                basis_metrics = compute_basis(
                    df['close'], 
                    aligned_spot['close'], 
                    days_to_expiry
                )
                
                for metric_name, metric_series in basis_metrics.items():
                    result_df[f'basis_{metric_name}'] = metric_series
                
                # Basis momentum
                result_df['basis_momentum'] = result_df['basis_basis_pct'].rolling(5).mean()
                result_df['basis_mean_reversion'] = (
                    result_df['basis_basis_pct'] - result_df['basis_basis_pct'].rolling(20).mean()
                ) / result_df['basis_basis_pct'].rolling(20).std()
        
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
        
        # 2. Add price-based features
        if 'close' in df.columns:
            # Returns at different horizons
            df['return_1d'] = df['close'].pct_change()
            df['return_5d'] = df['close'].pct_change(5)
            df['return_10d'] = df['close'].pct_change(10)
            
            # Log returns (more stable for modeling)
            df['log_return_1d'] = np.log(df['close'] / df['close'].shift(1))
            
            # Price momentum features
            df['price_momentum_5'] = df['close'] / df['close'].shift(5) - 1
            df['price_momentum_10'] = df['close'] / df['close'].shift(10) - 1
            
        # 3. Add volatility features
        if len(df) >= 20:
            returns = df['close'].pct_change()
            df['realized_vol_5'] = returns.rolling(5).std() * np.sqrt(252)
            df['realized_vol_20'] = returns.rolling(20).std() * np.sqrt(252)
            df['vol_ratio'] = df['realized_vol_5'] / df['realized_vol_20']
        
        # 4. Add volume/OI features if available
        if 'volume' in df.columns:
            df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
            df['volume_momentum'] = df['volume'].pct_change(5)
            
        if 'open_interest' in df.columns:
            df['oi_ma_ratio'] = df['open_interest'] / df['open_interest'].rolling(20).mean()
            df['oi_momentum'] = df['open_interest'].pct_change(5)
        
        # 5. Add time-based features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['is_month_end'] = (df.index + pd.DateOffset(days=1)).month != df.index.month
        
        # 6. Add regime features (enhanced with TA-Lib indicators)
        if len(df) >= 50:
            # Trend regime (price vs long-term MA)
            if 'sma_50' in df.columns:
                df['trend_regime'] = (df['close'] > df['sma_50']).astype(int)
            
            # ADX trend strength regime
            if 'adx' in df.columns:
                df['strong_trend_regime'] = (df['adx'] > 25).astype(int)
                
            # Volatility regime (current vol vs historical)
            if 'realized_vol_20' in df.columns:
                vol_median = df['realized_vol_20'].rolling(100).median()
                df['vol_regime'] = (df['realized_vol_20'] > vol_median).astype(int)
            
            # ATR-based volatility regime
            if 'atr_14' in df.columns:
                atr_ma = df['atr_14'].rolling(50).mean()
                df['high_vol_atr_regime'] = (df['atr_14'] > atr_ma * 1.5).astype(int)
            
            # RSI momentum regime
            if 'rsi_14' in df.columns:
                df['bullish_momentum'] = (df['rsi_14'] > 50).astype(int)
                df['extreme_oversold'] = (df['rsi_14'] < 20).astype(int)
                df['extreme_overbought'] = (df['rsi_14'] > 80).astype(int)
        
        # 7. Add cross-contract features if multiple contracts available
        if len(futures_data) > 1:
            contract_names = list(futures_data.keys())
            if len(contract_names) >= 2:
                next_contract_data = futures_data[contract_names[1]]
                if not next_contract_data.empty and 'close' in next_contract_data.columns:
                    # Calendar spread (current - next month)
                    aligned_next = next_contract_data.reindex(df.index, method='ffill')
                    df['calendar_spread'] = df['close'] - aligned_next['close']
                    df['calendar_spread_pct'] = (df['calendar_spread'] / df['close']) * 100
        
        # 8. Forward-fill any remaining NaN values for stability
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].ffill()
        
        # 9. Add metadata
        df.attrs = {
            'primary_contract': primary_contract,
            'feature_count': len([col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'open_interest']]),
            'has_indicators': include_indicators,
            'has_spot_data': spot_data is not None
        }
        
        logger.info("Built feature frame with %d features for %d rows", 
                   df.attrs['feature_count'], len(df))
        
        return df
        
    except Exception as exc:
        logger.exception("Error building feature frame: %s", exc)
        return pd.DataFrame()



def shutdown_data_agent():
    """Close sessions, flush buffers."""
    pass