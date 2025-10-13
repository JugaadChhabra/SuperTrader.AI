"""
Enhanced Configuration Management - Phase 5

Dynamic feature selection, Phase 4 toggles, and comprehensive system configuration
for SuperTrader.AI with production-ready settings management.
"""

import logging
import os
import yaml
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, time as dt_time
import json

logger = logging.getLogger(__name__)


@dataclass
class Phase4Config:
    """Phase 4 feature configuration"""
    enabled: bool = True
    pcr_integration: bool = True
    multi_timeframe_analysis: bool = True
    enhanced_oi_analysis: bool = True
    market_regime_detection: bool = True
    microstructure_analysis: bool = True
    
    # PCR Configuration
    pcr_data_source: str = "live"  # "live", "historical", "mock"
    pcr_update_frequency: int = 60  # seconds
    pcr_lookback_periods: int = 50
    
    # Multi-timeframe Configuration
    timeframes: List[str] = field(default_factory=lambda: ["5min", "15min", "1H"])
    base_timeframe: str = "5min"
    confluence_threshold: float = 0.75
    
    # Risk & Quality Thresholds
    risk_threshold_high: float = 0.7
    quality_threshold_min: float = 0.4
    setup_quality_threshold: float = 0.75


@dataclass  
class TradingConfig:
    """Core trading configuration"""
    # Market Hours
    market_open: dt_time = dt_time(9, 15)
    market_close: dt_time = dt_time(15, 30)
    square_off_time: dt_time = dt_time(15, 10)
    
    # Position Management
    max_position_size: int = 5  # lots
    target_volatility: float = 0.12
    position_sizing_method: str = "volatility_adjusted"  # "fixed", "volatility_adjusted", "phase4_adaptive"
    
    # Risk Management  
    stop_loss_pct: float = 0.02  # 2%
    take_profit_pct: float = 0.04  # 4%
    max_daily_loss_pct: float = 0.05  # 5%
    max_drawdown_pct: float = 0.1  # 10%
    
    # Trading Costs
    brokerage_flat: float = 20.0  # ₹20 per order
    stt_sell_pct: float = 0.0125  # 1.25 basis points
    exchange_charges_pct: float = 0.002  # 0.2 basis points
    gst_pct: float = 0.18  # 18% on brokerage


@dataclass
class DataConfig:
    """Data management configuration"""
    # Data Sources
    primary_data_source: str = "icici"  # "icici", "zerodha", "mock"
    backup_data_sources: List[str] = field(default_factory=lambda: ["historical", "mock"])
    
    # Data Quality
    enable_data_validation: bool = True
    spike_detection_threshold: float = 3.0  # standard deviations
    missing_data_tolerance_pct: float = 0.05  # 5%
    
    # Caching
    enable_data_caching: bool = True
    cache_expiry_minutes: int = 5
    cache_directory: str = "/tmp/supertrader_cache"
    
    # Historical Data
    lookback_days: int = 30
    warmup_periods: int = 100


@dataclass
class ModelConfig:
    """ML/RL model configuration"""
    # Model Type
    model_type: str = "rule_based"  # "rule_based", "dqn", "ppo", "a2c"
    model_path: Optional[str] = None
    
    # Feature Selection
    feature_selection_method: str = "all"  # "all", "top_k", "phase4_only", "custom"
    max_features: int = 100
    feature_importance_threshold: float = 0.01
    
    # Model Parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    memory_size: int = 10000
    
    # Phase 4 Integration
    phase4_feature_weight: float = 0.3  # Weight for Phase 4 vs legacy features
    adaptive_feature_selection: bool = True


@dataclass
class MonitoringConfig:
    """System monitoring configuration"""
    # Logging
    log_level: str = "INFO"
    log_file_path: str = "logs/supertrader.log" 
    max_log_file_size_mb: int = 100
    log_retention_days: int = 7
    
    # Performance Monitoring
    enable_performance_tracking: bool = True
    latency_threshold_ms: float = 100.0
    memory_threshold_mb: float = 500.0
    
    # Health Checks
    health_check_frequency_seconds: int = 60
    feature_generation_timeout_seconds: int = 30
    
    # Alerts
    enable_alerts: bool = True
    alert_channels: List[str] = field(default_factory=lambda: ["log", "email"])


class SuperTraderConfig:
    """
    Comprehensive configuration manager for SuperTrader.AI system
    Phase 5 Enhanced with dynamic feature selection and health monitoring
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration from file or defaults"""
        self.config_path = config_path or "configs/supertrader_config.yaml"
        
        # Initialize with defaults
        self.phase4 = Phase4Config()
        self.trading = TradingConfig()
        self.data = DataConfig()
        self.model = ModelConfig()
        self.monitoring = MonitoringConfig()
        
        # Load from file if exists
        self.load_config()
        
        # Runtime state
        self._feature_cache = {}
        self._performance_metrics = {}
        self._health_status = {}
        
        logger.info(f"SuperTrader configuration initialized from {self.config_path}")
    
    def load_config(self) -> bool:
        """Load configuration from YAML file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                # Update configurations from file
                if 'phase4' in config_data:
                    self._update_dataclass(self.phase4, config_data['phase4'])
                if 'trading' in config_data:
                    self._update_dataclass(self.trading, config_data['trading'])
                if 'data' in config_data:
                    self._update_dataclass(self.data, config_data['data'])
                if 'model' in config_data:
                    self._update_dataclass(self.model, config_data['model'])
                if 'monitoring' in config_data:
                    self._update_dataclass(self.monitoring, config_data['monitoring'])
                
                logger.info(f"Configuration loaded from {self.config_path}")
                return True
            else:
                logger.info(f"Configuration file not found, using defaults: {self.config_path}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return False
    
    def save_config(self) -> bool:
        """Save current configuration to YAML file"""
        try:
            config_data = {
                'phase4': self._dataclass_to_dict(self.phase4),
                'trading': self._dataclass_to_dict(self.trading),
                'data': self._dataclass_to_dict(self.data),
                'model': self._dataclass_to_dict(self.model),
                'monitoring': self._dataclass_to_dict(self.monitoring)
            }
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                yaml.safe_dump(config_data, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def get_enabled_features(self) -> Dict[str, bool]:
        """Get currently enabled feature categories"""
        return {
            'basic_technical': True,  # Always enabled
            'phase4_pcr': self.phase4.enabled and self.phase4.pcr_integration,
            'phase4_multi_timeframe': self.phase4.enabled and self.phase4.multi_timeframe_analysis,
            'phase4_enhanced_oi': self.phase4.enabled and self.phase4.enhanced_oi_analysis,
            'phase4_market_regime': self.phase4.enabled and self.phase4.market_regime_detection,
            'phase4_microstructure': self.phase4.enabled and self.phase4.microstructure_analysis,
            'data_validation': self.data.enable_data_validation,
            'performance_monitoring': self.monitoring.enable_performance_tracking
        }
    
    def get_feature_selection_config(self) -> Dict[str, Any]:
        """Get feature selection configuration for pipeline"""
        return {
            'method': self.model.feature_selection_method,
            'max_features': self.model.max_features,
            'importance_threshold': self.model.feature_importance_threshold,
            'phase4_weight': self.model.phase4_feature_weight,
            'adaptive': self.model.adaptive_feature_selection,
            'enabled_categories': self.get_enabled_features()
        }
    
    def get_trading_session_config(self) -> Dict[str, Any]:
        """Get trading session configuration"""
        return {
            'market_open': self.trading.market_open,
            'market_close': self.trading.market_close,
            'square_off_time': self.trading.square_off_time,
            'max_position_size': self.trading.max_position_size,
            'target_volatility': self.trading.target_volatility,
            'position_sizing_method': self.trading.position_sizing_method,
            'risk_params': {
                'stop_loss_pct': self.trading.stop_loss_pct,
                'take_profit_pct': self.trading.take_profit_pct,
                'max_daily_loss_pct': self.trading.max_daily_loss_pct,
                'max_drawdown_pct': self.trading.max_drawdown_pct
            },
            'cost_params': {
                'brokerage_flat': self.trading.brokerage_flat,
                'stt_sell_pct': self.trading.stt_sell_pct,
                'exchange_charges_pct': self.trading.exchange_charges_pct,
                'gst_pct': self.trading.gst_pct
            }
        }
    
    def get_data_pipeline_config(self) -> Dict[str, Any]:
        """Get data pipeline configuration"""
        return {
            'primary_source': self.data.primary_data_source,
            'backup_sources': self.data.backup_data_sources,
            'validation_enabled': self.data.enable_data_validation,
            'spike_threshold': self.data.spike_detection_threshold,
            'missing_tolerance': self.data.missing_data_tolerance_pct,
            'caching_enabled': self.data.enable_data_caching,
            'cache_expiry': self.data.cache_expiry_minutes,
            'cache_directory': self.data.cache_directory,
            'lookback_days': self.data.lookback_days,
            'warmup_periods': self.data.warmup_periods
        }
    
    def get_phase4_config(self) -> Dict[str, Any]:
        """Get Phase 4 specific configuration"""
        return {
            'enabled': self.phase4.enabled,
            'pcr_config': {
                'enabled': self.phase4.pcr_integration,
                'data_source': self.phase4.pcr_data_source,
                'update_frequency': self.phase4.pcr_update_frequency,
                'lookback_periods': self.phase4.pcr_lookback_periods
            },
            'multi_timeframe_config': {
                'enabled': self.phase4.multi_timeframe_analysis,
                'timeframes': self.phase4.timeframes,
                'base_timeframe': self.phase4.base_timeframe,
                'confluence_threshold': self.phase4.confluence_threshold
            },
            'risk_thresholds': {
                'high_risk': self.phase4.risk_threshold_high,
                'min_quality': self.phase4.quality_threshold_min,
                'setup_quality': self.phase4.setup_quality_threshold
            },
            'feature_toggles': {
                'enhanced_oi': self.phase4.enhanced_oi_analysis,
                'market_regime': self.phase4.market_regime_detection,
                'microstructure': self.phase4.microstructure_analysis
            }
        }
    
    def update_runtime_performance(self, component: str, metrics: Dict[str, float]):
        """Update runtime performance metrics"""
        self._performance_metrics[component] = {
            **metrics,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_performance_status(self) -> Dict[str, Any]:
        """Get current performance status"""
        return {
            'metrics': self._performance_metrics,
            'thresholds': {
                'latency_ms': self.monitoring.latency_threshold_ms,
                'memory_mb': self.monitoring.memory_threshold_mb
            }
        }
    
    def update_health_status(self, component: str, status: Dict[str, Any]):
        """Update component health status"""
        self._health_status[component] = {
            **status,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current system health status"""
        return self._health_status
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if specific feature is enabled"""
        enabled_features = self.get_enabled_features()
        return enabled_features.get(feature_name, False)
    
    def adapt_features_for_market_conditions(self, market_volatility: float, 
                                           market_regime: str) -> Dict[str, bool]:
        """
        Dynamically adapt feature selection based on market conditions
        """
        adapted_features = self.get_enabled_features().copy()
        
        # High volatility adaptations
        if market_volatility > 0.3:  # High volatility
            adapted_features['phase4_enhanced_oi'] = True  # More granular OI analysis
            adapted_features['phase4_microstructure'] = True  # Market microstructure important
            
        # Low volatility adaptations
        elif market_volatility < 0.1:  # Low volatility
            adapted_features['phase4_multi_timeframe'] = True  # Cross-timeframe needed
            
        # Trending vs consolidating market adaptations
        if market_regime == 'trending':
            adapted_features['phase4_multi_timeframe'] = True  # Trend confirmation
        elif market_regime == 'consolidating':
            adapted_features['phase4_pcr'] = True  # Sentiment more important in ranges
            
        return adapted_features
    
    def _update_dataclass(self, dataclass_instance: Any, update_dict: Dict[str, Any]):
        """Update dataclass fields from dictionary"""
        for key, value in update_dict.items():
            if hasattr(dataclass_instance, key):
                setattr(dataclass_instance, key, value)
    
    def _dataclass_to_dict(self, dataclass_instance: Any) -> Dict[str, Any]:
        """Convert dataclass to dictionary for serialization"""
        result = {}
        for key, value in dataclass_instance.__dict__.items():
            if isinstance(value, (dt_time,)):
                result[key] = value.strftime('%H:%M:%S')
            elif isinstance(value, list):
                result[key] = value.copy()
            else:
                result[key] = value
        return result


# Global configuration instance
_config_instance = None

def get_config(config_path: Optional[str] = None) -> SuperTraderConfig:
    """Get global configuration instance (singleton pattern)"""
    global _config_instance
    if _config_instance is None:
        _config_instance = SuperTraderConfig(config_path)
    return _config_instance

def reload_config(config_path: Optional[str] = None) -> SuperTraderConfig:
    """Force reload configuration"""
    global _config_instance
    _config_instance = SuperTraderConfig(config_path)
    return _config_instance


if __name__ == "__main__":
    # Test configuration system
    config = SuperTraderConfig()
    
    print("Phase 4 Configuration:")
    print(json.dumps(config.get_phase4_config(), indent=2))
    
    print("\nEnabled Features:")
    print(json.dumps(config.get_enabled_features(), indent=2))
    
    print("\nFeature Selection Config:")
    print(json.dumps(config.get_feature_selection_config(), indent=2))
    
    # Test adaptive features
    adaptive_features = config.adapt_features_for_market_conditions(0.25, 'trending')
    print("\nAdaptive Features (High Vol + Trending):")
    print(json.dumps(adaptive_features, indent=2))
