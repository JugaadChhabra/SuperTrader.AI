"""
Risk Configuration Manager for SuperTrader.AI

This module provides centralized risk configuration management with dynamic
parameter adjustment, validation, and real-time risk limit enforcement.

Classes:
    RiskConfigManager: Main configuration manager with validation and dynamic updates
    RiskLimitViolation: Exception class for risk limit violations
    ConfigValidationError: Exception for configuration validation errors

Features:
    - Dynamic risk parameter adjustment based on market conditions
    - Real-time configuration validation and reloading
    - Risk limit breach detection and enforcement
    - Performance-based parameter optimization
    - Emergency override capabilities
    - Audit trail for configuration changes

Author: SuperTrader.AI Team
Version: 1.0.0
Last Updated: 2024-10-13
"""

import yaml
import os
import logging
from datetime import datetime, timedelta, time
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from threading import Lock
import copy


# Custom exceptions
class RiskLimitViolation(Exception):
    """Raised when a risk limit is violated"""
    def __init__(self, limit_type: str, current_value: float, limit_value: float, message: str = None):
        self.limit_type = limit_type
        self.current_value = current_value
        self.limit_value = limit_value
        self.message = message or f"{limit_type}: {current_value} exceeds limit of {limit_value}"
        super().__init__(self.message)


class ConfigValidationError(Exception):
    """Raised when configuration validation fails"""
    pass


@dataclass
class RiskMetrics:
    """Container for current risk metrics"""
    portfolio_leverage: float = 0.0
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    monthly_pnl: float = 0.0
    var_95: float = 0.0
    var_99: float = 0.0
    margin_utilization: float = 0.0
    largest_position_pct: float = 0.0
    total_exposure: float = 0.0
    correlation_risk: float = 0.0
    consecutive_losses: int = 0
    consecutive_loss_amount: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging/monitoring"""
        return {
            'portfolio_leverage': self.portfolio_leverage,
            'daily_pnl': self.daily_pnl,
            'weekly_pnl': self.weekly_pnl,
            'monthly_pnl': self.monthly_pnl,
            'var_95': self.var_95,
            'var_99': self.var_99,
            'margin_utilization': self.margin_utilization,
            'largest_position_pct': self.largest_position_pct,
            'total_exposure': self.total_exposure,
            'correlation_risk': self.correlation_risk,
            'consecutive_losses': self.consecutive_losses,
            'consecutive_loss_amount': self.consecutive_loss_amount
        }


@dataclass
class ViolationResult:
    """Result of risk limit validation"""
    is_violation: bool
    violation_type: str = None
    current_value: float = None
    limit_value: float = None
    severity: str = None  # 'warning', 'error', 'critical'
    message: str = None
    recommended_action: str = None


class RiskConfigManager:
    """
    Centralized risk configuration management with dynamic parameter adjustment
    and real-time risk monitoring capabilities.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the Risk Configuration Manager
        
        Args:
            config_path: Path to the risk configuration YAML file
        """
        self.logger = logging.getLogger(__name__)
        
        # Default to configs/risk.yaml if not specified
        if config_path is None:
            base_dir = Path(__file__).parent.parent
            config_path = base_dir / "configs" / "risk.yaml"
        
        self.config_path = Path(config_path)
        self.config = {}
        self.config_lock = Lock()
        self.last_reload_time = None
        self.emergency_mode = False
        self.config_change_log = []
        
        # Load initial configuration
        self.reload_config()
        
        # Initialize monitoring
        self.current_metrics = RiskMetrics()
        self.violation_history = []
        
        self.logger.info(f"RiskConfigManager initialized with config: {self.config_path}")
    
    def reload_config(self) -> bool:
        """
        Reload configuration from file with validation
        
        Returns:
            bool: True if reload successful, False otherwise
        """
        try:
            with self.config_lock:
                if not self.config_path.exists():
                    raise FileNotFoundError(f"Config file not found: {self.config_path}")
                
                with open(self.config_path, 'r') as f:
                    new_config = yaml.safe_load(f)
                
                # Validate configuration
                self._validate_config(new_config)
                
                # Log configuration change if different
                if new_config != self.config:
                    self._log_config_change(new_config)
                
                self.config = new_config
                self.last_reload_time = datetime.now()
                
                self.logger.info("Risk configuration reloaded successfully")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to reload config: {e}")
            return False
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate configuration structure and values
        
        Args:
            config: Configuration dictionary to validate
        """
        required_sections = [
            'global_limits', 'index_limits', 'position_sizing',
            'kelly_parameters', 'margin_management', 'correlation_limits',
            'time_based_controls', 'circuit_breakers', 'contract_specs'
        ]
        
        # Check required sections
        for section in required_sections:
            if section not in config:
                raise ConfigValidationError(f"Missing required section: {section}")
        
        # Validate global limits
        global_limits = config['global_limits']
        if global_limits.get('max_portfolio_leverage', 0) <= 0:
            raise ConfigValidationError("max_portfolio_leverage must be positive")
        
        # Validate index limits
        required_indices = ['NIFTY', 'BANKNIFTY', 'FINNIFTY']
        for index in required_indices:
            if index not in config['index_limits']:
                raise ConfigValidationError(f"Missing index configuration: {index}")
            
            index_config = config['index_limits'][index]
            if index_config.get('max_leverage', 0) <= 0:
                raise ConfigValidationError(f"{index} max_leverage must be positive")
        
        # Validate position sizing parameters
        pos_sizing = config['position_sizing']
        if not 0 < pos_sizing.get('volatility_target', 0) < 1:
            raise ConfigValidationError("volatility_target must be between 0 and 1")
        
        # Validate Kelly parameters
        kelly = config['kelly_parameters']
        if not 0 < kelly.get('max_kelly_fraction', 0) <= 1:
            raise ConfigValidationError("max_kelly_fraction must be between 0 and 1")
        
        self.logger.debug("Configuration validation passed")
    
    def _log_config_change(self, new_config: Dict[str, Any]) -> None:
        """
        Log configuration changes for audit trail
        
        Args:
            new_config: New configuration to compare against current
        """
        change_entry = {
            'timestamp': datetime.now(),
            'old_version': self.config.get('version', 'unknown'),
            'new_version': new_config.get('version', 'unknown'),
            'changes': self._find_config_differences(self.config, new_config)
        }
        
        self.config_change_log.append(change_entry)
        
        # Keep only last 100 changes
        if len(self.config_change_log) > 100:
            self.config_change_log.pop(0)
        
        self.logger.info(f"Configuration change logged: {len(change_entry['changes'])} changes")
    
    def _find_config_differences(self, old_config: Dict, new_config: Dict, path: str = "") -> List[Dict]:
        """
        Find differences between two configuration dictionaries
        
        Args:
            old_config: Previous configuration
            new_config: New configuration
            path: Current path in nested dictionary
            
        Returns:
            List of change descriptions
        """
        changes = []
        
        # Check for modified/new keys
        for key, new_value in new_config.items():
            current_path = f"{path}.{key}" if path else key
            
            if key not in old_config:
                changes.append({
                    'type': 'added',
                    'path': current_path,
                    'new_value': new_value
                })
            elif isinstance(new_value, dict) and isinstance(old_config[key], dict):
                changes.extend(self._find_config_differences(
                    old_config[key], new_value, current_path
                ))
            elif old_config[key] != new_value:
                changes.append({
                    'type': 'modified',
                    'path': current_path,
                    'old_value': old_config[key],
                    'new_value': new_value
                })
        
        # Check for removed keys
        for key in old_config:
            if key not in new_config:
                current_path = f"{path}.{key}" if path else key
                changes.append({
                    'type': 'removed',
                    'path': current_path,
                    'old_value': old_config[key]
                })
        
        return changes
    
    def get_global_limit(self, limit_name: str) -> Any:
        """
        Get a global risk limit value
        
        Args:
            limit_name: Name of the limit parameter
            
        Returns:
            Limit value or None if not found
        """
        with self.config_lock:
            return self.config.get('global_limits', {}).get(limit_name)
    
    def get_index_limit(self, index: str, limit_name: str) -> Any:
        """
        Get an index-specific risk limit
        
        Args:
            index: Index symbol (NIFTY, BANKNIFTY, FINNIFTY)
            limit_name: Name of the limit parameter
            
        Returns:
            Limit value or None if not found
        """
        with self.config_lock:
            return self.config.get('index_limits', {}).get(index, {}).get(limit_name)
    
    def get_position_sizing_params(self) -> Dict[str, Any]:
        """
        Get position sizing parameters
        
        Returns:
            Dictionary of position sizing parameters
        """
        with self.config_lock:
            return copy.deepcopy(self.config.get('position_sizing', {}))
    
    def get_kelly_params(self) -> Dict[str, Any]:
        """
        Get Kelly criterion parameters
        
        Returns:
            Dictionary of Kelly parameters
        """
        with self.config_lock:
            return copy.deepcopy(self.config.get('kelly_parameters', {}))
    
    def get_margin_params(self) -> Dict[str, Any]:
        """
        Get margin management parameters
        
        Returns:
            Dictionary of margin parameters
        """
        with self.config_lock:
            return copy.deepcopy(self.config.get('margin_management', {}))
    
    def check_risk_limits(self, metrics: RiskMetrics) -> List[ViolationResult]:
        """
        Check current metrics against all risk limits
        
        Args:
            metrics: Current portfolio risk metrics
            
        Returns:
            List of violation results
        """
        violations = []
        
        # Update current metrics
        self.current_metrics = metrics
        
        # Check global limits
        violations.extend(self._check_global_limits(metrics))
        
        # Check margin limits
        violations.extend(self._check_margin_limits(metrics))
        
        # Check circuit breaker conditions
        violations.extend(self._check_circuit_breakers(metrics))
        
        # Log violations
        for violation in violations:
            if violation.is_violation:
                self.violation_history.append({
                    'timestamp': datetime.now(),
                    'violation': violation
                })
                
                self.logger.warning(
                    f"Risk limit violation: {violation.violation_type} - "
                    f"{violation.message}"
                )
        
        return violations
    
    def _check_global_limits(self, metrics: RiskMetrics) -> List[ViolationResult]:
        """Check global portfolio limits"""
        violations = []
        
        # Portfolio leverage
        max_leverage = self.get_global_limit('max_portfolio_leverage')
        if max_leverage and metrics.portfolio_leverage > max_leverage:
            violations.append(ViolationResult(
                is_violation=True,
                violation_type='portfolio_leverage',
                current_value=metrics.portfolio_leverage,
                limit_value=max_leverage,
                severity='error',
                message=f"Portfolio leverage {metrics.portfolio_leverage:.2f}x exceeds limit {max_leverage:.2f}x",
                recommended_action='reduce_positions'
            ))
        
        # Daily drawdown
        max_daily_dd = self.get_global_limit('max_daily_drawdown_pct')
        if max_daily_dd and metrics.daily_pnl < -max_daily_dd:
            violations.append(ViolationResult(
                is_violation=True,
                violation_type='daily_drawdown',
                current_value=abs(metrics.daily_pnl),
                limit_value=max_daily_dd,
                severity='critical',
                message=f"Daily drawdown {abs(metrics.daily_pnl):.2f}% exceeds limit {max_daily_dd:.2f}%",
                recommended_action='stop_trading'
            ))
        
        # VaR limits
        max_var_95 = self.get_global_limit('max_var_95_pct')
        if max_var_95 and metrics.var_95 > max_var_95:
            violations.append(ViolationResult(
                is_violation=True,
                violation_type='var_95',
                current_value=metrics.var_95,
                limit_value=max_var_95,
                severity='warning',
                message=f"VaR 95% {metrics.var_95:.2f}% exceeds limit {max_var_95:.2f}%",
                recommended_action='reduce_risk'
            ))
        
        # Single position concentration
        max_single_pos = self.get_global_limit('max_single_position_pct')
        if max_single_pos and metrics.largest_position_pct > max_single_pos:
            violations.append(ViolationResult(
                is_violation=True,
                violation_type='position_concentration',
                current_value=metrics.largest_position_pct,
                limit_value=max_single_pos,
                severity='error',
                message=f"Largest position {metrics.largest_position_pct:.2f}% exceeds limit {max_single_pos:.2f}%",
                recommended_action='reduce_largest_position'
            ))
        
        return violations
    
    def _check_margin_limits(self, metrics: RiskMetrics) -> List[ViolationResult]:
        """Check margin management limits"""
        violations = []
        
        margin_params = self.get_margin_params()
        
        # Margin utilization
        max_margin_util = margin_params.get('max_margin_utilization', 1.0)
        if metrics.margin_utilization > max_margin_util:
            violations.append(ViolationResult(
                is_violation=True,
                violation_type='margin_utilization',
                current_value=metrics.margin_utilization,
                limit_value=max_margin_util,
                severity='error',
                message=f"Margin utilization {metrics.margin_utilization:.2f} exceeds limit {max_margin_util:.2f}",
                recommended_action='reduce_positions'
            ))
        
        # Emergency liquidation threshold
        emergency_threshold = margin_params.get('emergency_liquidation_threshold', 0.95)
        if metrics.margin_utilization > emergency_threshold:
            violations.append(ViolationResult(
                is_violation=True,
                violation_type='emergency_margin',
                current_value=metrics.margin_utilization,
                limit_value=emergency_threshold,
                severity='critical',
                message=f"Emergency margin threshold breached: {metrics.margin_utilization:.2f}",
                recommended_action='emergency_liquidation'
            ))
        
        return violations
    
    def _check_circuit_breakers(self, metrics: RiskMetrics) -> List[ViolationResult]:
        """Check circuit breaker conditions"""
        violations = []
        
        cb_config = self.config.get('circuit_breakers', {})
        
        # Consecutive losses
        max_consecutive = cb_config.get('consecutive_loss_limit', float('inf'))
        if metrics.consecutive_losses >= max_consecutive:
            violations.append(ViolationResult(
                is_violation=True,
                violation_type='consecutive_losses',
                current_value=metrics.consecutive_losses,
                limit_value=max_consecutive,
                severity='critical',
                message=f"Consecutive losses {metrics.consecutive_losses} reached limit {max_consecutive}",
                recommended_action='stop_trading'
            ))
        
        # Consecutive loss amount
        max_consecutive_amount = cb_config.get('consecutive_loss_amount', float('inf'))
        if metrics.consecutive_loss_amount >= max_consecutive_amount:
            violations.append(ViolationResult(
                is_violation=True,
                violation_type='consecutive_loss_amount',
                current_value=metrics.consecutive_loss_amount,
                limit_value=max_consecutive_amount,
                severity='critical',
                message=f"Consecutive loss amount ₹{metrics.consecutive_loss_amount:,.0f} exceeds limit ₹{max_consecutive_amount:,.0f}",
                recommended_action='stop_trading'
            ))
        
        return violations
    
    def is_trading_allowed(self, current_time: datetime = None) -> bool:
        """
        Check if trading is allowed based on time controls
        
        Args:
            current_time: Current datetime (defaults to now)
            
        Returns:
            bool: True if trading is allowed
        """
        if current_time is None:
            current_time = datetime.now()
        
        time_controls = self.config.get('time_based_controls', {})
        
        # Check if in market hours
        market_open = time_controls.get('market_open_time', '09:15')
        market_close = time_controls.get('market_close_time', '15:30')
        
        current_time_only = current_time.time()
        open_time = datetime.strptime(market_open, '%H:%M').time()
        close_time = datetime.strptime(market_close, '%H:%M').time()
        
        if not (open_time <= current_time_only <= close_time):
            return False
        
        # Check no-trade periods
        no_trade_start = time_controls.get('no_trade_start_time', '09:15')
        no_trade_end = time_controls.get('no_trade_end_time', '09:30')
        
        if no_trade_start and no_trade_end:
            no_start = datetime.strptime(no_trade_start, '%H:%M').time()
            no_end = datetime.strptime(no_trade_end, '%H:%M').time()
            
            if no_start <= current_time_only <= no_end:
                return False
        
        # Check emergency mode
        if self.emergency_mode:
            return False
        
        return True
    
    def enable_emergency_mode(self, reason: str = "Manual trigger") -> None:
        """
        Enable emergency mode to halt all trading
        
        Args:
            reason: Reason for enabling emergency mode
        """
        self.emergency_mode = True
        
        self.logger.critical(f"Emergency mode ENABLED: {reason}")
        
        # Log to config change history
        self.config_change_log.append({
            'timestamp': datetime.now(),
            'type': 'emergency_mode_enabled',
            'reason': reason
        })
    
    def disable_emergency_mode(self, reason: str = "Manual reset") -> None:
        """
        Disable emergency mode to resume trading
        
        Args:
            reason: Reason for disabling emergency mode
        """
        self.emergency_mode = False
        
        self.logger.warning(f"Emergency mode DISABLED: {reason}")
        
        # Log to config change history
        self.config_change_log.append({
            'timestamp': datetime.now(),
            'type': 'emergency_mode_disabled',
            'reason': reason
        })
    
    def update_dynamic_limits(self, market_volatility: float, portfolio_performance: Dict[str, float]) -> None:
        """
        Dynamically adjust risk limits based on market conditions and performance
        
        Args:
            market_volatility: Current market volatility (annualized)
            portfolio_performance: Dictionary with performance metrics
        """
        try:
            # Get volatility-based adjustments
            pos_sizing = self.get_position_sizing_params()
            
            # Adjust position sizing multipliers based on volatility
            if market_volatility > 0.25:  # High volatility
                multiplier = pos_sizing.get('high_vol_multiplier', 0.7)
                self.logger.info(f"High volatility detected ({market_volatility:.2%}), reducing position sizes by {(1-multiplier)*100:.1f}%")
            elif market_volatility < 0.10:  # Low volatility
                multiplier = pos_sizing.get('low_vol_multiplier', 1.2)
                self.logger.info(f"Low volatility detected ({market_volatility:.2%}), increasing position sizes by {(multiplier-1)*100:.1f}%")
            else:
                multiplier = 1.0
            
            # Adjust based on recent performance
            sharpe_ratio = portfolio_performance.get('sharpe_ratio', 0)
            if sharpe_ratio < 0.5:
                # Poor performance, reduce risk
                performance_multiplier = 0.8
                self.logger.warning(f"Poor performance (Sharpe: {sharpe_ratio:.2f}), reducing position sizes by 20%")
            elif sharpe_ratio > 2.0:
                # Excellent performance, allow slightly larger positions
                performance_multiplier = 1.1
                self.logger.info(f"Excellent performance (Sharpe: {sharpe_ratio:.2f}), increasing position sizes by 10%")
            else:
                performance_multiplier = 1.0
            
            # Store dynamic adjustments (would be used by position sizing)
            self._dynamic_multiplier = multiplier * performance_multiplier
            
        except Exception as e:
            self.logger.error(f"Error updating dynamic limits: {e}")
    
    def get_dynamic_multiplier(self) -> float:
        """Get current dynamic position sizing multiplier"""
        return getattr(self, '_dynamic_multiplier', 1.0)
    
    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get summary of current configuration
        
        Returns:
            Dictionary with key configuration parameters
        """
        with self.config_lock:
            summary = {
                'version': self.config.get('version', 'unknown'),
                'last_updated': self.config.get('last_updated', 'unknown'),
                'last_reload': self.last_reload_time.isoformat() if self.last_reload_time else None,
                'emergency_mode': self.emergency_mode,
                'global_limits': {
                    'max_portfolio_leverage': self.get_global_limit('max_portfolio_leverage'),
                    'max_daily_drawdown_pct': self.get_global_limit('max_daily_drawdown_pct'),
                    'max_single_position_pct': self.get_global_limit('max_single_position_pct')
                },
                'dynamic_multiplier': self.get_dynamic_multiplier(),
                'config_changes_count': len(self.config_change_log),
                'violations_count': len(self.violation_history)
            }
        
        return summary
    
    def export_config_audit_trail(self) -> Dict[str, Any]:
        """
        Export complete audit trail for compliance
        
        Returns:
            Dictionary with complete audit information
        """
        return {
            'config_path': str(self.config_path),
            'current_config_version': self.config.get('version', 'unknown'),
            'last_reload_time': self.last_reload_time.isoformat() if self.last_reload_time else None,
            'emergency_mode': self.emergency_mode,
            'config_changes': copy.deepcopy(self.config_change_log),
            'recent_violations': [
                {
                    'timestamp': v['timestamp'].isoformat(),
                    'type': v['violation'].violation_type,
                    'severity': v['violation'].severity,
                    'message': v['violation'].message
                }
                for v in self.violation_history[-50:]  # Last 50 violations
            ],
            'current_metrics': self.current_metrics.to_dict()
        }


# Global instance (singleton pattern)
_risk_config_manager = None
_manager_lock = Lock()


def get_risk_config_manager(config_path: str = None) -> RiskConfigManager:
    """
    Get the global RiskConfigManager instance (singleton)
    
    Args:
        config_path: Path to configuration file (only used on first call)
        
    Returns:
        RiskConfigManager instance
    """
    global _risk_config_manager
    
    with _manager_lock:
        if _risk_config_manager is None:
            _risk_config_manager = RiskConfigManager(config_path)
    
    return _risk_config_manager


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize manager
    try:
        manager = RiskConfigManager()
        
        # Test configuration access
        print("Configuration Summary:")
        summary = manager.get_config_summary()
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        # Test risk limit checking
        test_metrics = RiskMetrics(
            portfolio_leverage=7.5,  # This should trigger a violation
            daily_pnl=-1.5,
            margin_utilization=0.65,
            largest_position_pct=15.0,
            consecutive_losses=3
        )
        
        violations = manager.check_risk_limits(test_metrics)
        print(f"\nFound {len(violations)} violations:")
        for violation in violations:
            if violation.is_violation:
                print(f"  - {violation.violation_type}: {violation.message}")
        
        # Test time-based controls
        is_allowed = manager.is_trading_allowed()
        print(f"\nTrading allowed: {is_allowed}")
        
        # Test dynamic adjustments
        manager.update_dynamic_limits(
            market_volatility=0.28,  # High volatility
            portfolio_performance={'sharpe_ratio': 0.3}  # Poor performance
        )
        print(f"Dynamic multiplier: {manager.get_dynamic_multiplier():.3f}")
        
    except Exception as e:
        print(f"Error testing RiskConfigManager: {e}")
        import traceback
        traceback.print_exc()