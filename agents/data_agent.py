import asyncio
import base64
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional

import requests
import socketio
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

def fetch_ohlcv(symbols, interval, start, end): # — pull raw bars (1m/5m/15m/daily).
    pass

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

def compute_indicators(df): # — MACD, multi-TF SMA, RSI, volume/vol regime features.
    pass

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