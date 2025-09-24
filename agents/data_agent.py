import asyncio
import base64
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional

import requests
import socketio
import pandas as pd
import numpy as np
from dotenv import load_dotenv

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
        or os.getenv("ICICI_APP_KEY"),
        "api_session_token": (config or {}).get("icici", {}).get("api_session_token")
        or os.getenv("ICICI_API_SESSION_TOKEN"),
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


async def _icici_sio_client(
    *, user_id: str, session_token: str
) -> socketio.AsyncClient:
    """Create and configure a socket.io AsyncClient for ICICI live stream."""
    sio = socketio.AsyncClient(logger=False, engineio_logger=False)

    @sio.event
    async def connect():  # pragma: no cover - network path
        logger.info("ICICI WebSocket connected")

    @sio.event
    async def disconnect():  # pragma: no cover - network path
        logger.info("ICICI WebSocket disconnected")

    @sio.event
    async def connect_error(data):  # pragma: no cover - network path
        logger.error("ICICI WebSocket connection error: %s", data)

    # Auth payload expected by upstream server
    sio.auth = {"user": user_id, "token": session_token}
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


async def stream_icici_quotes(
    api_keys: Dict[str, str],
    symbols: Iterable[str],
    *,
    duration_sec: int = 30,
    on_tick: Optional[callable] = None,
) -> None:
    """Stream live quotes from ICICI for a fixed duration.
    """
    auth = connect_icici_broker(api_keys)
    sio = await _icici_sio_client(user_id=auth["user_id"], session_token=auth["session_token"])

    # Event handler for ticks
    @sio.on("stock")
    async def _on_stock_data(data):  # pragma: no cover - network path
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

    # Connect
    await sio.connect(
        "https://livestream.icicidirect.com",
        headers={"User-Agent": "python-socketio[client]/socket"},
        transports=["websocket"],
        wait_timeout=10,
    )

    # Join rooms / subscribe symbols
    joined: List[str] = []
    try:
        for sym in symbols:
            await sio.emit("join", sym)
            joined.append(sym)
            logger.info("subscribed: %s", sym)

        # Sleep for duration
        logger.info("streaming for %s seconds…", duration_sec)
        await asyncio.sleep(duration_sec)
    finally:
        # Best-effort cleanup
        for sym in joined:
            try:
                await sio.emit("leave", sym)
            except Exception:
                pass
        try:
            await sio.emit("disconnect", "transport close")
        except Exception:
            pass
        try:
            await sio.disconnect()
        except Exception:
            pass
        logger.info("ICICI stream closed")



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
    asyncio.run(stream_icici_quotes(api, symbols, duration_sec=duration_sec, on_tick=_default_on_tick))

if __name__ == "__main__": 
    demo_run_from_env(10)

def fetch_ohlcv(symbols: List[str], interval: str, start: str, end: str, 
                api_keys: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    """Fetch OHLCV (Open, High, Low, Close, Volume) data for symbols.
    
    Args:
        symbols: List of symbols (e.g., ['NSE:RELIANCE', 'NSE:SBIN'])
        interval: Time interval ('1minute', '5minute', '15minute', '1day')
        start: Start date in 'YYYY-MM-DD' format
        end: End date in 'YYYY-MM-DD' format  
        api_keys: Dict with 'app_key' and 'api_session_token'
        
    Returns:
        Dict mapping symbol -> DataFrame with OHLCV data
        
    Note:
        This is a template implementation. You may need to adjust the API endpoint
        and parameters based on ICICI Direct's actual historical data API.
    """
    if not api_keys.get("app_key") or not api_keys.get("api_session_token"):
        raise ValueError("Missing required API keys")
        
    results = {}
    
    # ICICI API endpoint for historical data (this may need adjustment)
    base_url = "https://api.icicidirect.com/breezeapi/api/v1/historicaldata"
    
    headers = {
        "Content-Type": "application/json",
        "X-AppKey": api_keys["app_key"],
        "X-SessionToken": api_keys["api_session_token"]
    }
    
    for symbol in symbols:
        try:
            # API payload (adjust based on actual ICICI API format)
            payload = {
                "stock_code": symbol,
                "exchange_code": symbol.split(':')[0] if ':' in symbol else 'NSE',
                "product_type": "C",  # Cash segment
                "interval": interval,
                "from_date": start,
                "to_date": end
            }
            
            logger.info("Fetching OHLCV for %s from %s to %s", symbol, start, end)
            
            response = requests.post(base_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Parse response (adjust based on actual API response format)
            if data.get("Status") == "Success" and "Success" in data:
                ohlcv_data = data["Success"]
                
                # Convert to DataFrame (adjust column mapping as needed)
                df = pd.DataFrame(ohlcv_data)
                
                # Standardize column names
                column_mapping = {
                    'datetime': 'timestamp',
                    'open': 'open', 
                    'high': 'high',
                    'low': 'low',
                    'close': 'close',
                    'volume': 'volume'
                }
                
                df = df.rename(columns=column_mapping)
                
                # Ensure timestamp is datetime
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                
                # Convert price columns to numeric
                price_cols = ['open', 'high', 'low', 'close', 'volume']
                for col in price_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                results[symbol] = df
                logger.info("Fetched %d bars for %s", len(df), symbol)
                
            else:
                logger.error("API error for %s: %s", symbol, data)
                results[symbol] = pd.DataFrame()  # Empty DataFrame on error
                
        except requests.RequestException as e:
            logger.error("Network error fetching %s: %s", symbol, e)
            results[symbol] = pd.DataFrame()
            
        except Exception as e:
            logger.error("Unexpected error fetching %s: %s", symbol, e)
            results[symbol] = pd.DataFrame()
    
    return results

def fetch_market_depth(symbol): # — level-1/level-2 snapshot (bid/ask, spreads).
    pass

def fetch_corporate_actions(symbols, start, end): # — splits/dividends/earnings.
    pass

def validate_data(df): # — schema/type checks, outlier & gap detection.
    pass

def interpolate_missing(df, method): # — forward/back fill, stitching.
    pass

def resample_bars(df, interval): # — unify to target timeframe.
    pass

def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average.
    
    Args:
        prices: Series of closing prices
        period: Number of periods for moving average
        
    Returns:
        Series with SMA values
    """
    return prices.rolling(window=period, min_periods=period).mean()

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index.
    
    Args:
        prices: Series of closing prices  
        period: RSI period (default 14)
        
    Returns:
        Series with RSI values (0-100)
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
    """Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        prices: Series of closing prices
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26) 
        signal: Signal line EMA period (default 9)
        
    Returns:
        Dict containing 'macd', 'signal', and 'histogram' Series
    """
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    
    return {
        'macd': macd_line,
        'signal': signal_line, 
        'histogram': histogram
    }

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators for price data.
    
    Args:
        df: DataFrame with OHLCV data (expects 'close' column)
        
    Returns:
        DataFrame with original data plus indicator columns
    """
    if df.empty or 'close' not in df.columns:
        logger.warning("Invalid DataFrame for indicators - missing 'close' column")
        return df
    
    result_df = df.copy()
    
    try:
        # Simple Moving Averages (multiple timeframes)
        result_df['sma_5'] = calculate_sma(df['close'], 5)
        result_df['sma_10'] = calculate_sma(df['close'], 10)  
        result_df['sma_20'] = calculate_sma(df['close'], 20)
        result_df['sma_50'] = calculate_sma(df['close'], 50)
        result_df['sma_200'] = calculate_sma(df['close'], 200)
        
        # RSI
        result_df['rsi'] = calculate_rsi(df['close'], 14)
        
        # MACD
        macd_data = calculate_macd(df['close'])
        result_df['macd'] = macd_data['macd']
        result_df['macd_signal'] = macd_data['signal']
        result_df['macd_histogram'] = macd_data['histogram']
        
        # Volume indicators (if volume column exists)
        if 'volume' in df.columns:
            result_df['volume_sma_20'] = calculate_sma(df['volume'], 20)
            # Volume ratio (current vs average)
            result_df['volume_ratio'] = df['volume'] / result_df['volume_sma_20']
            
        # Volatility regime features
        if len(df) >= 20:
            # Rolling standard deviation of returns
            returns = df['close'].pct_change()
            result_df['volatility_20'] = returns.rolling(20).std() * np.sqrt(252)  # Annualized
            result_df['vol_regime'] = result_df['volatility_20'] > result_df['volatility_20'].rolling(50).mean()
            
        logger.info("Computed indicators for %d rows", len(result_df))
        
    except Exception as exc:
        logger.exception("Error computing indicators: %s", exc)
        
    return result_df

def build_feature_frame(df_prices, df_indicators): # — final model feature set.
    pass

def rank_universe(features, rules): # — screener logic → rank score per symbol.
    pass

def select_top_k(ranks, k, constraints): # — universe selection with liquidity caps.
    pass

def update_feature_store(symbol, features, ts): # — persist for RL/exec layers.
    pass

def get_feature_batch(symbols, ts_window): # — windowed features for RL.
    pass

def shutdown_data_agent(): # — close sessions, flush buffers.
    pass