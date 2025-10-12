"""
Data Agent Package

This package contains all data agent related functionality organized by concern:
- icici_broker.py: ICICI Direct API integration and WebSocket streaming
- futures_manager.py: Contract rollover, basis calculation, and futures-specific logic
- validators.py: Data validation and quality checks
- feature_builders.py: Technical indicators and feature engineering
- constants.py: Configuration constants and hardcoded values
"""

from .main import (
    init_data_agent,
    fetch_index_futures_ohlcv,
    compute_indicators,
    compute_futures_indicators,
    build_feature_frame,
    shutdown_data_agent
)

from .icici_broker import (
    connect_icici_broker,
    stream_icici_quotes,
    demo_run_from_env,
    demo_index_futures_stream
)

from .futures_manager import (
    handle_contract_rollover,
    construct_continuous_contract,
    compute_basis
)

from .validators import (
    validate_futures_data
)

__all__ = [
    'init_data_agent',
    'fetch_index_futures_ohlcv',
    'compute_indicators',
    'compute_futures_indicators', 
    'build_feature_frame',
    'shutdown_data_agent',
    'connect_icici_broker',
    'stream_icici_quotes',
    'demo_run_from_env',
    'demo_index_futures_stream',
    'handle_contract_rollover',
    'construct_continuous_contract',
    'compute_basis',
    'validate_futures_data'
]