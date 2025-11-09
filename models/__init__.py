"""
SuperTrader.AI Models Package

This package contains the core trading models and environments for the SuperTrader.AI system.
"""

# Import main classes and functions to make them available at package level
from .dqn_network import TradingAgent, TradingBrain  # explicit
from .intraday_environment import (
    IntradayMarket,
    IntradayTradingEnv,
    create_intraday_env,
    TradingEnvironment,
)

# Training utilities (avoid wildcard to reduce namespace pollution)
try:
    from .training import Trainer as DQNTrainer
except Exception:  # training script optional
    DQNTrainer = None

__all__ = [
    'TradingAgent', 'TradingBrain',
    'IntradayMarket', 'IntradayTradingEnv', 'TradingEnvironment', 'create_intraday_env',
    'DQNTrainer'
]