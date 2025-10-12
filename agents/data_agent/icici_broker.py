"""
ICICI Direct Broker Integration Module

Handles all ICICI Direct API interactions including:
- Authentication and WebSocket session management
- Live data streaming
- Demo functions for testing
- Symbol generation for index futures
"""

import base64
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

import requests
import socketio
import yaml
from dotenv import load_dotenv

from .constants import (
    ICICI_API_BASE_URL,
    ICICI_WEBSOCKET_URL,
    ICICI_CUSTOMER_DETAILS_ENDPOINT,
    DEFAULT_TIMEOUT,
    WEBSOCKET_HEADERS,
    API_HEADERS_TEMPLATE,
    MONTH_CODES,
    DEFAULT_DEMO_SYMBOLS,
    DEFAULT_CONFIG_PATH,
    MINUTE_CHART_DURATION,
    MINUTE_CHART_INTERVAL
)

logger = logging.getLogger(__name__)


def connect_icici_broker(api_keys: Dict[str, str]) -> Dict[str, str]:
    """Authenticate with ICICI Direct to obtain WebSocket session credentials.
    
    Args:
        api_keys: {"app_key": str, "api_session_token": str}
        
    Returns:
        {"user_id": str, "session_token": str, "raw": dict}
    """
    app_key = api_keys.get("app_key")
    api_session_token = api_keys.get("api_session_token")
    if not app_key or not api_session_token:
        raise RuntimeError("Missing 'app_key' or 'api_session_token' for ICICI auth")

    url = f"{ICICI_API_BASE_URL}{ICICI_CUSTOMER_DETAILS_ENDPOINT}"
    payload = json.dumps({"SessionToken": api_session_token, "AppKey": app_key})
    headers = API_HEADERS_TEMPLATE.copy()

    logger.info("Fetching ICICI WebSocket session token…")
    resp = requests.request("GET", url, headers=headers, data=payload, timeout=DEFAULT_TIMEOUT)
    
    try:
        data = resp.json()
    except Exception as exc:
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
    except Exception as exc:
        logger.exception("Failed to decode ICICI session token")
        raise RuntimeError(f"ICICI auth decode failed: {exc}")

    logger.info("ICICI auth OK | user_id=%s", user_id)
    return {"user_id": user_id, "session_token": session_token, "raw": data}


def _create_icici_socketio_client(*, user_id: str, session_token: str) -> socketio.Client:
    """Create and configure a socket.io Client for ICICI live stream."""
    sio = socketio.Client(logger=False, engineio_logger=False)

    @sio.event
    def connect():
        logger.info("ICICI WebSocket connected")

    @sio.event
    def disconnect():
        logger.info("ICICI WebSocket disconnected")

    @sio.event
    def connect_error(data):
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
    except Exception as exc:
        logger.exception("ICICI parse error: %s", exc)
        return {"raw_data": data}


def stream_icici_quotes(
    api_keys: Dict[str, str],
    symbols: Iterable[str],
    *,
    duration_sec: int = 30,
    on_tick: Optional[callable] = None,
) -> None:
    """Stream live quotes from ICICI for a fixed duration."""
    auth = connect_icici_broker(api_keys)
    sio = _create_icici_socketio_client(user_id=auth["user_id"], session_token=auth["session_token"])

    @sio.on("stock")
    def _on_stock_data(data):
        parsed = parse_icici_payload(data)
        if on_tick:
            try:
                on_tick(parsed)
            except Exception:
                logger.exception("on_tick callback error")
        else:
            # Default log output
            sym = parsed.get("symbol", "?")
            last = parsed.get("last")
            chg = parsed.get("change")
            ltq = parsed.get("ltq")
            logger.info("tick | %s | last=%s change=%s ltq=%s", sym, last, chg, ltq)

    logger.info("Attempting WebSocket connection with user_id: %s", auth["user_id"][:4] + "***")
    
    try:
        sio.connect(
            ICICI_WEBSOCKET_URL,
            headers=WEBSOCKET_HEADERS,
            auth={"user": auth["user_id"], "token": auth["session_token"]},
            transports=["websocket"],
            wait_timeout=DEFAULT_TIMEOUT,
        )

        # Join rooms / subscribe symbols
        joined: List[str] = []
        try:
            for sym in symbols:
                sio.emit("join", sym)
                joined.append(sym)
                logger.info("subscribed: %s", sym)

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
    """Pretty console print for ad-hoc local testing."""
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


def get_index_futures_symbols(config_path: str = DEFAULT_CONFIG_PATH) -> List[str]:
    """Generate ICICI-format symbol list for index futures from market config.
    
    Returns list like: ['NSE:NIFTY25OCTFUT', 'NSE:BANKNIFTY25OCTFUT', ...] (current month only)
    """
    # Get the directory of this script and construct absolute path
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(config_path):
        config_path = os.path.join(script_dir, config_path)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get current month/year for contract generation  
    now = datetime.now()
    
    # Generate symbols for current month contracts only
    symbols = []
    for index_name in config['index_futures'].keys():
        curr_month = MONTH_CODES[now.month]
        year_suffix = str(now.year)[2:]  # 25 for 2025
        symbol = f"NSE:{index_name}{year_suffix}{curr_month}FUT"
        symbols.append(symbol)
    
    return symbols


def stream_minute_chart(api_keys: Dict[str, str], symbols: List[str]) -> None:
    """Stream data every minute for 5 minutes to simulate 1-minute chart."""
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
        
        print(f"\n📊 === 1-MINUTE CHART DATA (Minute {minute_count}/{MINUTE_CHART_DURATION}) ===")
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
    sio = _create_icici_socketio_client(user_id=auth["user_id"], session_token=auth["session_token"])
    
    @sio.on("stock")
    def _on_stock_data(data):
        parsed = parse_icici_payload(data)
        minute_tick_handler(parsed)
    
    try:
        logger.info("Attempting WebSocket connection with user_id: %s", auth["user_id"][:4] + "***")
        
        sio.connect(
            ICICI_WEBSOCKET_URL,
            headers=WEBSOCKET_HEADERS,
            auth={"user": auth["user_id"], "token": auth["session_token"]},
            transports=["websocket"],
            wait_timeout=DEFAULT_TIMEOUT,
        )
        
        # Subscribe to all symbols
        for sym in symbols:
            sio.emit("join", sym)
            logger.info("subscribed: %s", sym)
        
        logger.info("Starting %d-minute streaming with 1-minute intervals...", MINUTE_CHART_DURATION)
        
        for minute in range(MINUTE_CHART_DURATION):
            time.sleep(MINUTE_CHART_INTERVAL)
            print_minute_summary()
            
        logger.info("Completed %d-minute chart simulation", MINUTE_CHART_DURATION)
        
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


def demo_run_from_env(duration_sec: int = 15) -> None:
    """Demo runner using env vars. Useful for quick local checks."""
    # Import here to avoid circular imports
    from dotenv import load_dotenv
    
    load_dotenv()
    api = {
        "app_key": os.getenv("APP_KEY"),
        "api_session_token": os.getenv("API_SESSION_TOKEN")
    }
    if not api["app_key"] or not api["api_session_token"]:
        raise RuntimeError("Missing ICICI_APP_KEY / ICICI_API_SESSION_TOKEN env vars")

    symbols = os.getenv("ICICI_DEMO_SYMBOLS", DEFAULT_DEMO_SYMBOLS).split(",")
    stream_icici_quotes(api, symbols, duration_sec=duration_sec, on_tick=_default_on_tick)


def demo_index_futures_stream(duration_sec: int = 30) -> None:
    """Demo streaming index futures with 1-minute intervals for 5 minutes."""
    # Import here to avoid circular imports
    from dotenv import load_dotenv
    
    load_dotenv()
    api = {
        "app_key": os.getenv("APP_KEY"),
        "api_session_token": os.getenv("API_SESSION_TOKEN")
    }
    
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