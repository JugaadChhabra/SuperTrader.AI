"""
Constants and configuration values for the data agent
"""

# API Configuration
ICICI_API_BASE_URL = "https://api.icicidirect.com/breezeapi/api/v1"
ICICI_WEBSOCKET_URL = "https://livestream.icicidirect.com"
ICICI_CUSTOMER_DETAILS_ENDPOINT = "/customerdetails"
ICICI_HISTORICAL_DATA_ENDPOINT = "/historicaldata"

# Network Configuration
DEFAULT_TIMEOUT = 15
DEFAULT_WEBSOCKET_TIMEOUT = 15
MAX_RETRY_ATTEMPTS = 3

# Month codes for futures contract naming
MONTH_CODES = {
    1: 'JAN', 2: 'FEB', 3: 'MAR', 4: 'APR', 
    5: 'MAY', 6: 'JUN', 7: 'JUL', 8: 'AUG', 
    9: 'SEP', 10: 'OCT', 11: 'NOV', 12: 'DEC'
}

# DataFrame column mappings
OHLCVI_COLUMN_MAPPING = {
    'datetime': 'timestamp',
    'open': 'open', 
    'high': 'high',
    'low': 'low',
    'close': 'close',
    'volume': 'volume',
    'oi': 'open_interest',
    'open_interest': 'open_interest'
}

# Standard column sets
PRICE_COLUMNS = ['open', 'high', 'low', 'close']
OHLCVI_COLUMNS = ['open', 'high', 'low', 'close', 'volume', 'open_interest']
NUMERIC_COLUMNS = ['open', 'high', 'low', 'close', 'volume', 'open_interest']

# Financial calculations
TRADING_DAYS_PER_YEAR = 252
DEFAULT_RISK_FREE_RATE = 0.06
VOLATILITY_ANNUALIZATION_FACTOR = 252

# Data validation thresholds
PRICE_SPIKE_THRESHOLD_MULTIPLIER = 2.0  # 2x 99th percentile
TIME_GAP_THRESHOLD_MULTIPLIER = 2.0     # 2x expected frequency
ZERO_VOLUME_THRESHOLD = 0.1             # 10% of bars with zero volume
OI_CHANGE_THRESHOLD = 0.5               # 50% change in open interest

# Rolling window periods
ROLLING_PERIODS = [5, 10, 20, 50, 100]

# Default symbols for demo
DEFAULT_DEMO_SYMBOLS = "NSE:SBIN,NSE:RELIANCE"
DEFAULT_CONFIG_PATH = "../configs/market.yaml"

# Chart simulation settings
MINUTE_CHART_DURATION = 5  # 5 minutes
MINUTE_CHART_INTERVAL = 60  # 60 seconds

# WebSocket headers
WEBSOCKET_HEADERS = {"User-Agent": "python-socketio[client]/socket"}

# API Headers template
API_HEADERS_TEMPLATE = {
    "Content-Type": "application/json"
}