"""
Configuration Constants for SuperTrader.AI Agentic Portfolio Manager

This module provides centralized configuration management for:
- Reinforcement Learning parameters (α, β)
- Model paths and feature scaling
- Trading session parameters
- Performance thresholds and limits
- Paper trading simulation settings

Author: SuperTrader.AI Team
Version: 2.0.0
Last Updated: 2024-11-08
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "notebooks" / "models"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"


@dataclass
class RLConfig:
    """Reinforcement Learning Configuration"""
    # Reward calculation parameters
    alpha_return: float = 0.7      # Weight for return component in reward
    beta_risk: float = 0.3         # Weight for risk component in reward
    
    # Model paths
    dqn_model_dir: str = str(MODEL_DIR / "intraday_dqn")
    feature_scaler_path: str = str(MODEL_DIR / "feature_scaler.pkl")
    
    # Training parameters
    epsilon: float = 0.05          # Exploration rate for inference
    gamma: float = 0.95            # Discount factor
    learning_rate: float = 0.001   # Model learning rate
    
    # State/action parameters
    state_size: int = 20           # Feature vector size
    action_size: int = 3           # BUY, SELL, HOLD
    lookback_window: int = 60      # Historical window for features
    
    # Model update thresholds
    performance_threshold: float = 0.6    # Min performance to update model
    update_frequency: int = 100           # Episodes between model updates


@dataclass
class TradingConfig:
    """Trading Session Configuration"""
    # Session parameters
    initial_capital: float = 100000.0     # Starting portfolio value
    max_session_duration: int = 390       # Max session length in minutes
    trade_frequency: int = 5              # Minutes between decision points
    
    # Position sizing
    max_position_size: float = 0.2        # Max 20% of portfolio per position
    min_position_size: float = 0.01       # Min 1% of portfolio per position
    volatility_adjustment: bool = True     # Adjust position size by volatility
    
    # Risk management
    max_daily_loss: float = 0.05          # Max 5% daily loss
    max_drawdown: float = 0.10            # Max 10% portfolio drawdown
    stop_loss_threshold: float = 0.02     # 2% stop loss
    
    # Transaction costs
    commission_rate: float = 0.001        # 0.1% commission
    slippage_basis_points: float = 5.0    # 5 basis points slippage
    
    # Market hours (Indian market)
    market_open: str = "09:15"
    market_close: str = "15:30"


@dataclass
class PerformanceConfig:
    """Performance Monitoring Configuration"""
    # Metrics thresholds
    excellent_sharpe: float = 1.5
    good_sharpe: float = 1.0
    acceptable_sharpe: float = 0.5
    
    # Win rate targets
    target_win_rate: float = 55.0         # 55% win rate target
    minimum_win_rate: float = 40.0        # 40% minimum acceptable
    
    # Drawdown limits
    max_acceptable_drawdown: float = 15.0  # 15% max acceptable
    warning_drawdown: float = 10.0         # 10% warning level
    
    # Return expectations (annualized %)
    target_annual_return: float = 20.0
    minimum_annual_return: float = 8.0
    
    # Risk-free rate for calculations
    risk_free_rate: float = 0.06          # 6% annualized
    
    # Reporting frequency
    real_time_updates: bool = True
    performance_log_frequency: int = 10    # Log every 10 trades


@dataclass
class AgentConfig:
    """Agent-specific Configuration"""
    # Data agent
    data_refresh_interval: int = 60       # Seconds between data updates
    feature_calculation_timeout: int = 30  # Max time for feature calc
    
    # News agent
    news_impact_weight: float = 0.1       # Weight of news in decisions
    sentiment_threshold: float = 0.6      # Min sentiment confidence
    
    # Execution agent
    decision_timeout: int = 5             # Max decision time in seconds
    retry_attempts: int = 3               # Max retry attempts for failed trades
    
    # Portfolio simulation
    enable_slippage: bool = True
    enable_transaction_costs: bool = True
    mock_latency_ms: int = 100            # Simulated execution latency


@dataclass
class LoggingConfig:
    """Logging Configuration"""
    # Log directories
    log_dir: str = str(LOGS_DIR)
    trade_log_dir: str = str(LOGS_DIR / "trades")
    performance_log_dir: str = str(LOGS_DIR / "performance")
    
    # Log levels
    console_level: str = "INFO"
    file_level: str = "DEBUG"
    
    # Log rotation
    max_log_size_mb: int = 100
    backup_count: int = 5
    
    # Export settings
    export_csv: bool = True
    export_json: bool = True
    real_time_metrics: bool = True


class SuperTraderConfig:
    """Main configuration class combining all settings"""
    
    def __init__(self):
        self.rl = RLConfig()
        self.trading = TradingConfig()
        self.performance = PerformanceConfig()
        self.agents = AgentConfig()
        self.logging = LoggingConfig()
        
        # Environment-specific overrides
        self._apply_environment_overrides()
        
        # Validation
        self._validate_config()
    
    def _apply_environment_overrides(self):
        """Apply environment variable overrides"""
        # RL overrides
        if os.getenv("ALPHA_RETURN"):
            self.rl.alpha_return = float(os.getenv("ALPHA_RETURN"))
        if os.getenv("BETA_RISK"):
            self.rl.beta_risk = float(os.getenv("BETA_RISK"))
        
        # Trading overrides
        if os.getenv("INITIAL_CAPITAL"):
            self.trading.initial_capital = float(os.getenv("INITIAL_CAPITAL"))
        if os.getenv("MAX_POSITION_SIZE"):
            self.trading.max_position_size = float(os.getenv("MAX_POSITION_SIZE"))
        
        # Performance overrides
        if os.getenv("TARGET_SHARPE"):
            self.performance.excellent_sharpe = float(os.getenv("TARGET_SHARPE"))
    
    def _validate_config(self):
        """Validate configuration parameters"""
        # RL validation
        assert 0 <= self.rl.alpha_return <= 1, "Alpha must be between 0 and 1"
        assert 0 <= self.rl.beta_risk <= 1, "Beta must be between 0 and 1"
        assert abs(self.rl.alpha_return + self.rl.beta_risk - 1.0) < 0.01, "Alpha + Beta must equal 1"
        
        # Trading validation
        assert self.trading.initial_capital > 0, "Initial capital must be positive"
        assert 0 < self.trading.max_position_size <= 1, "Max position size must be between 0 and 1"
        assert self.trading.min_position_size < self.trading.max_position_size, "Min < Max position size"
        
        # Performance validation
        assert self.performance.target_win_rate >= self.performance.minimum_win_rate, "Target >= Minimum win rate"
        assert self.performance.max_acceptable_drawdown > self.performance.warning_drawdown, "Max > Warning drawdown"
        
        # Path validation
        assert os.path.exists(self.rl.dqn_model_dir), f"DQN model directory not found: {self.rl.dqn_model_dir}"
    
    def get_reward_calculation_params(self) -> Dict[str, float]:
        """Get reward calculation parameters for RL agent"""
        return {
            'alpha': self.rl.alpha_return,
            'beta': self.rl.beta_risk,
            'gamma': self.rl.gamma,
            'risk_free_rate': self.performance.risk_free_rate
        }
    
    def get_model_paths(self) -> Dict[str, str]:
        """Get model file paths"""
        return {
            'dqn_model_dir': self.rl.dqn_model_dir,
            'feature_scaler': self.rl.feature_scaler_path,
            'latest_checkpoint': self._get_latest_checkpoint()
        }
    
    def _get_latest_checkpoint(self) -> Optional[str]:
        """Find latest Keras checkpoint (.weights.h5) with highest episode number.

        Filenames match pattern: 'checkpoint_episode_<N>.weights.h5'.
        Uses regex on the filename (not stem) to handle double extension.
        Returns absolute path string or None if none found.
        """
        import re
        model_dir = Path(self.rl.dqn_model_dir)
        if not model_dir.exists():
            return None

        pattern = re.compile(r"checkpoint_episode_(\d+)\.weights\.h5$")
        checkpoints = list(model_dir.glob("*.weights.h5"))
        if not checkpoints:
            return None

        episodes: list[tuple[int, str]] = []
        for cp in checkpoints:
            match = pattern.search(cp.name)
            if not match:
                continue
            try:
                ep = int(match.group(1))
                episodes.append((ep, str(cp)))
            except ValueError:
                continue

        if not episodes:
            return None
        episodes.sort(key=lambda x: x[0], reverse=True)
        return episodes[0][1]
    
    def get_trading_limits(self) -> Dict[str, float]:
        """Get trading risk limits"""
        return {
            'max_daily_loss': self.trading.max_daily_loss,
            'max_drawdown': self.trading.max_drawdown,
            'max_position_size': self.trading.max_position_size,
            'stop_loss_threshold': self.trading.stop_loss_threshold,
            'warning_drawdown': self.performance.warning_drawdown
        }
    
    def export_config(self) -> Dict[str, Any]:
        """Export complete configuration as dictionary"""
        return {
            'rl': self.rl.__dict__,
            'trading': self.trading.__dict__,
            'performance': self.performance.__dict__,
            'agents': self.agents.__dict__,
            'logging': self.logging.__dict__,
            'derived': {
                'reward_params': self.get_reward_calculation_params(),
                'model_paths': self.get_model_paths(),
                'trading_limits': self.get_trading_limits()
            }
        }


# Global configuration instance
_config: Optional[SuperTraderConfig] = None


def get_config() -> SuperTraderConfig:
    """Get or create global configuration instance"""
    global _config
    
    if _config is None:
        _config = SuperTraderConfig()
    
    return _config


# Convenience functions for quick access to common parameters
def get_reward_params() -> Dict[str, float]:
    """Get α, β parameters for reward calculation"""
    config = get_config()
    return config.get_reward_calculation_params()


def get_alpha_beta() -> tuple[float, float]:
    """Get alpha and beta parameters directly"""
    config = get_config()
    return config.rl.alpha_return, config.rl.beta_risk


def get_model_paths() -> Dict[str, str]:
    """Get DQN model paths"""
    config = get_config()
    return config.get_model_paths()


def get_trading_limits() -> Dict[str, float]:
    """Get trading risk management limits"""
    config = get_config()
    return config.get_trading_limits()


# Configuration validation at module import
if __name__ == "__main__":
    # Test configuration
    config = SuperTraderConfig()
    print("Configuration validation passed!")
    
    import json
    export = config.export_config()
    print(json.dumps(export, indent=2))