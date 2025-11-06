"""
Position Manager for SuperTrader.AI

Real-time position monitoring system with mark-to-market P&L calculation,
margin tracking, and position aggregation across multiple strategies and timeframes.

Features:
- Real-time position tracking and updates
- Mark-to-market P&L calculation
- Multi-strategy position aggregation
- Risk monitoring and alerts
- Margin utilization tracking
- Position-level stop-loss management
- Portfolio-wide risk metrics

Classes:
    PositionManager: Main position management system
    PositionMonitor: Real-time position monitoring
    RiskMetrics: Portfolio risk calculations
    MarginTracker: Margin utilization monitoring

Author: SuperTrader.AI Team
Version: 1.0.0
Last Updated: 2024-10-14
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from collections import defaultdict, deque

# Import trade ledger components
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from agents.trade_ledger import TradeLedger, Position, Trade, PositionSide, get_trade_ledger


class PositionAlert(Enum):
    """Position alert types"""
    PROFIT_TARGET = "PROFIT_TARGET"
    STOP_LOSS = "STOP_LOSS"
    MARGIN_WARNING = "MARGIN_WARNING"
    CONCENTRATION_RISK = "CONCENTRATION_RISK"
    CORRELATION_RISK = "CORRELATION_RISK"
    LARGE_MOVE = "LARGE_MOVE"
    TIME_DECAY = "TIME_DECAY"


@dataclass
class PositionMetrics:
    """Position-level risk and performance metrics"""
    
    # Basic metrics
    symbol: str = ""
    quantity: int = 0
    avg_price: float = 0.0
    current_price: float = 0.0
    market_value: float = 0.0
    
    # P&L metrics
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    intraday_high_pnl: float = 0.0
    intraday_low_pnl: float = 0.0
    
    # Risk metrics
    position_size_pct: float = 0.0      # % of total capital
    leverage: float = 0.0               # Notional / Margin
    margin_used: float = 0.0
    var_contribution: float = 0.0       # VaR contribution
    
    # Timing metrics
    holding_period_minutes: int = 0
    time_to_expiry_days: int = 0
    
    # Alerts and flags
    active_alerts: List[PositionAlert] = field(default_factory=list)
    is_profit_target_hit: bool = False
    is_stop_loss_hit: bool = False
    
    def calculate_derived_metrics(self, total_capital: float) -> None:
        """Calculate derived metrics"""
        if self.avg_price > 0 and self.current_price > 0:
            if self.quantity > 0:  # Long position
                self.unrealized_pnl_pct = ((self.current_price - self.avg_price) / self.avg_price) * 100
            else:  # Short position
                self.unrealized_pnl_pct = ((self.avg_price - self.current_price) / self.avg_price) * 100
        
        if total_capital > 0:
            self.position_size_pct = (abs(self.market_value) / total_capital) * 100
        
        if self.margin_used > 0:
            self.leverage = abs(self.market_value) / self.margin_used


@dataclass
class PortfolioMetrics:
    """Portfolio-level risk and performance metrics"""
    
    # Basic metrics
    total_capital: float = 0.0
    available_margin: float = 0.0
    used_margin: float = 0.0
    margin_utilization_pct: float = 0.0
    
    # Position metrics
    num_positions: int = 0
    num_long_positions: int = 0
    num_short_positions: int = 0
    
    # Exposure metrics
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    long_exposure: float = 0.0
    short_exposure: float = 0.0
    
    # P&L metrics
    total_unrealized_pnl: float = 0.0
    total_realized_pnl: float = 0.0
    daily_pnl: float = 0.0
    
    # Risk metrics
    portfolio_leverage: float = 0.0
    largest_position_pct: float = 0.0
    concentration_risk: float = 0.0
    var_95: float = 0.0
    expected_shortfall: float = 0.0
    
    # Correlation metrics
    avg_correlation: float = 0.0
    nifty_banknifty_correlation: float = 0.0
    
    def calculate_derived_metrics(self) -> None:
        """Calculate derived portfolio metrics"""
        if self.total_capital > 0:
            self.portfolio_leverage = self.gross_exposure / self.total_capital
            self.concentration_risk = self.largest_position_pct / 100.0
        
        if self.available_margin > 0:
            self.margin_utilization_pct = (self.used_margin / self.available_margin) * 100
        
        self.net_exposure = self.long_exposure - self.short_exposure


class PositionManager:
    """
    Real-time position management system with comprehensive monitoring
    """
    
    def __init__(self, trade_ledger: TradeLedger = None, update_frequency: int = 30):
        """
        Initialize Position Manager
        
        Args:
            trade_ledger: TradeLedger instance (creates new if None)
            update_frequency: Update frequency in seconds
        """
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.trade_ledger = trade_ledger or get_trade_ledger()
        self.update_frequency = update_frequency
        
        # Position tracking
        self.position_metrics: Dict[str, PositionMetrics] = {}
        self.portfolio_metrics = PortfolioMetrics()
        
        # Market data
        self.current_prices: Dict[str, float] = {}
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Monitoring and alerts
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.alert_callbacks: List[Callable] = []
        
        # Performance tracking
        self.pnl_history: deque = deque(maxlen=10000)  # Keep last 10k P&L updates
        self.last_update_time: Optional[datetime] = None
        
        # Risk parameters (can be configured)
        self.risk_params = {
            'max_position_size_pct': 25.0,      # Max 25% in single position
            'max_concentration_risk': 0.30,     # Max 30% concentration
            'margin_warning_threshold': 0.75,   # Warning at 75% margin usage
            'margin_critical_threshold': 0.90,  # Critical at 90% margin usage
            'stop_loss_pct': 2.0,              # 2% stop loss
            'profit_target_pct': 4.0,          # 4% profit target
            'max_holding_hours': 6,            # Max 6 hours holding
            'correlation_alert_threshold': 0.80 # Alert if correlation > 80%
        }
        
        self.logger.info("PositionManager initialized")
    
    def start_monitoring(self) -> None:
        """Start real-time position monitoring"""
        if self.monitoring_active:
            self.logger.warning("Position monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            name="PositionMonitor",
            daemon=True
        )
        self.monitor_thread.start()
        
        self.logger.info(f"Position monitoring started (update every {self.update_frequency}s)")
    
    def stop_monitoring(self) -> None:
        """Stop real-time position monitoring"""
        self.monitoring_active = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        
        self.logger.info("Position monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Update positions and calculate metrics
                self.update_all_positions()
                
                # Check for alerts
                self._check_position_alerts()
                
                # Sleep until next update
                time.sleep(self.update_frequency)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1)  # Short sleep on error
    
    def update_market_data(self, price_data: Dict[str, float]) -> None:
        """
        Update market prices for positions
        
        Args:
            price_data: Dictionary of symbol -> current_price
        """
        try:
            # Update current prices
            self.current_prices.update(price_data)
            
            # Store price history
            timestamp = datetime.now()
            for symbol, price in price_data.items():
                self.price_history[symbol].append((timestamp, price))
            
            # Update trade ledger with new prices
            self.trade_ledger.update_market_prices(price_data)
            
            # Recalculate position metrics
            self._calculate_position_metrics()
            
            self.last_update_time = timestamp
            
        except Exception as e:
            self.logger.error(f"Error updating market data: {e}")
    
    def update_all_positions(self) -> None:
        """Update all position metrics and portfolio calculations"""
        try:
            # Get current positions from trade ledger
            positions = self.trade_ledger.get_all_positions()
            
            # Calculate metrics for each position
            for position in positions:
                self._update_position_metrics(position)
            
            # Calculate portfolio-level metrics
            self._calculate_portfolio_metrics(positions)
            
            # Update P&L history
            self._update_pnl_history()
            
        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")
    
    def _update_position_metrics(self, position: Position) -> None:
        """Update metrics for a single position"""
        symbol = position.symbol
        current_price = self.current_prices.get(symbol, position.avg_price)
        
        # Calculate position metrics
        metrics = PositionMetrics(
            symbol=symbol,
            quantity=position.quantity,
            avg_price=position.avg_price,
            current_price=current_price,
            market_value=abs(position.quantity) * current_price,
            unrealized_pnl=position.unrealized_pnl,
            margin_used=position.margin_used
        )
        
        # Calculate derived metrics
        metrics.calculate_derived_metrics(self.portfolio_metrics.total_capital or 1000000)
        
        # Calculate holding period
        if position.first_entry_time:
            holding_period = datetime.now() - position.first_entry_time
            metrics.holding_period_minutes = int(holding_period.total_seconds() / 60)
        
        # Update intraday high/low P&L
        if symbol in self.position_metrics:
            old_metrics = self.position_metrics[symbol]
            metrics.intraday_high_pnl = max(old_metrics.intraday_high_pnl, metrics.unrealized_pnl)
            metrics.intraday_low_pnl = min(old_metrics.intraday_low_pnl, metrics.unrealized_pnl)
        else:
            metrics.intraday_high_pnl = metrics.unrealized_pnl
            metrics.intraday_low_pnl = metrics.unrealized_pnl
        
        # Store updated metrics
        self.position_metrics[symbol] = metrics
    
    def _calculate_position_metrics(self) -> None:
        """Recalculate all position metrics with current prices"""
        positions = self.trade_ledger.get_all_positions()
        
        for position in positions:
            if position.symbol in self.current_prices:
                # Recalculate P&L with current price
                current_price = self.current_prices[position.symbol]
                position.calculate_pnl(current_price)
                
                # Update position metrics
                self._update_position_metrics(position)
    
    def _calculate_portfolio_metrics(self, positions: List[Position]) -> None:
        """Calculate portfolio-level metrics"""
        # Initialize metrics
        metrics = PortfolioMetrics()
        
        # Basic portfolio information (would come from account data)
        metrics.total_capital = 1000000  # ₹10 lakh (should be dynamic)
        metrics.available_margin = 600000  # ₹6 lakh (should be dynamic)
        
        # Calculate position-based metrics
        metrics.num_positions = len([p for p in positions if p.quantity != 0])
        metrics.num_long_positions = len([p for p in positions if p.quantity > 0])
        metrics.num_short_positions = len([p for p in positions if p.quantity < 0])
        
        # Calculate exposures
        total_long_exposure = 0.0
        total_short_exposure = 0.0
        total_margin_used = 0.0
        total_unrealized_pnl = 0.0
        largest_position_value = 0.0
        
        for position in positions:
            if position.quantity == 0:
                continue
            
            current_price = self.current_prices.get(position.symbol, position.avg_price)
            position_value = abs(position.quantity) * current_price
            
            if position.quantity > 0:
                total_long_exposure += position_value
            else:
                total_short_exposure += position_value
            
            total_margin_used += position.margin_used
            total_unrealized_pnl += position.unrealized_pnl
            largest_position_value = max(largest_position_value, position_value)
        
        metrics.long_exposure = total_long_exposure
        metrics.short_exposure = total_short_exposure
        metrics.gross_exposure = total_long_exposure + total_short_exposure
        metrics.used_margin = total_margin_used
        metrics.total_unrealized_pnl = total_unrealized_pnl
        
        # Calculate percentage metrics
        if metrics.total_capital > 0:
            metrics.largest_position_pct = (largest_position_value / metrics.total_capital) * 100
        
        # Calculate derived metrics
        metrics.calculate_derived_metrics()
        
        # Calculate risk metrics
        self._calculate_portfolio_risk_metrics(metrics, positions)
        
        # Store updated metrics
        self.portfolio_metrics = metrics
    
    def _calculate_portfolio_risk_metrics(self, metrics: PortfolioMetrics, positions: List[Position]) -> None:
        """Calculate advanced portfolio risk metrics"""
        try:
            if not positions:
                return
            
            # Calculate VaR (simplified approach using position volatilities)
            position_values = []
            for position in positions:
                if position.quantity != 0:
                    current_price = self.current_prices.get(position.symbol, position.avg_price)
                    position_value = position.quantity * current_price  # Signed value
                    position_values.append(position_value)
            
            if position_values:
                # Simple VaR calculation (assuming 2% daily volatility)
                portfolio_value = sum(position_values)
                daily_volatility = 0.02  # 2% daily volatility assumption
                metrics.var_95 = abs(portfolio_value) * daily_volatility * 1.645  # 95% VaR
                metrics.expected_shortfall = abs(portfolio_value) * daily_volatility * 2.33  # 99% ES
            
            # Calculate correlation risk (simplified)
            index_symbols = ['NIFTY', 'BANKNIFTY', 'FINNIFTY']
            index_positions = [p for p in positions if p.symbol in index_symbols and p.quantity != 0]
            
            if len(index_positions) >= 2:
                metrics.avg_correlation = 0.75  # Simplified correlation estimate
            
            # Check NIFTY-BANKNIFTY specific correlation
            nifty_pos = next((p for p in positions if p.symbol == 'NIFTY' and p.quantity != 0), None)
            banknifty_pos = next((p for p in positions if p.symbol == 'BANKNIFTY' and p.quantity != 0), None)
            
            if nifty_pos and banknifty_pos:
                metrics.nifty_banknifty_correlation = 0.75  # Historical correlation estimate
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio risk metrics: {e}")
    
    def _update_pnl_history(self) -> None:
        """Update P&L history for tracking"""
        timestamp = datetime.now()
        total_pnl = self.portfolio_metrics.total_unrealized_pnl
        
        self.pnl_history.append((timestamp, total_pnl))
    
    def _check_position_alerts(self) -> None:
        """Check all positions for alert conditions"""
        for symbol, metrics in self.position_metrics.items():
            alerts = []
            
            # Check stop loss
            if abs(metrics.unrealized_pnl_pct) >= self.risk_params['stop_loss_pct']:
                if metrics.unrealized_pnl < 0:  # Only for losses
                    alerts.append(PositionAlert.STOP_LOSS)
                    metrics.is_stop_loss_hit = True
            
            # Check profit target
            if metrics.unrealized_pnl_pct >= self.risk_params['profit_target_pct']:
                alerts.append(PositionAlert.PROFIT_TARGET)
                metrics.is_profit_target_hit = True
            
            # Check position concentration
            if metrics.position_size_pct > self.risk_params['max_position_size_pct']:
                alerts.append(PositionAlert.CONCENTRATION_RISK)
            
            # Check holding time
            if metrics.holding_period_minutes > (self.risk_params['max_holding_hours'] * 60):
                alerts.append(PositionAlert.TIME_DECAY)
            
            # Check for large moves
            if abs(metrics.unrealized_pnl_pct) > 5.0:  # 5% move
                alerts.append(PositionAlert.LARGE_MOVE)
            
            # Update alerts
            metrics.active_alerts = alerts
            
            # Trigger alert callbacks
            for alert in alerts:
                self._trigger_alert(symbol, alert, metrics)
    
    def _trigger_alert(self, symbol: str, alert: PositionAlert, metrics: PositionMetrics) -> None:
        """Trigger alert callbacks"""
        try:
            alert_data = {
                'symbol': symbol,
                'alert_type': alert,
                'metrics': metrics,
                'timestamp': datetime.now(),
                'message': self._generate_alert_message(symbol, alert, metrics)
            }
            
            # Call registered alert callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert_data)
                except Exception as e:
                    self.logger.error(f"Error in alert callback: {e}")
            
            # Log the alert
            self.logger.warning(f"ALERT [{alert.value}] {symbol}: {alert_data['message']}")
            
        except Exception as e:
            self.logger.error(f"Error triggering alert: {e}")
    
    def _generate_alert_message(self, symbol: str, alert: PositionAlert, metrics: PositionMetrics) -> str:
        """Generate human-readable alert message"""
        if alert == PositionAlert.STOP_LOSS:
            return f"Stop loss hit: {metrics.unrealized_pnl_pct:.1f}% loss (₹{metrics.unrealized_pnl:,.0f})"
        elif alert == PositionAlert.PROFIT_TARGET:
            return f"Profit target reached: {metrics.unrealized_pnl_pct:.1f}% profit (₹{metrics.unrealized_pnl:,.0f})"
        elif alert == PositionAlert.CONCENTRATION_RISK:
            return f"Position concentration risk: {metrics.position_size_pct:.1f}% of portfolio"
        elif alert == PositionAlert.TIME_DECAY:
            return f"Long holding period: {metrics.holding_period_minutes} minutes"
        elif alert == PositionAlert.LARGE_MOVE:
            return f"Large move detected: {metrics.unrealized_pnl_pct:.1f}% ({metrics.unrealized_pnl:+,.0f})"
        else:
            return f"Alert triggered: {alert.value}"
    
    def register_alert_callback(self, callback: Callable) -> None:
        """Register a callback function for alerts"""
        self.alert_callbacks.append(callback)
        self.logger.info(f"Registered alert callback: {callback.__name__}")
    
    def get_position_summary(self, symbol: str = None) -> Dict[str, Any]:
        """
        Get position summary for a symbol or all positions
        
        Args:
            symbol: Symbol to get summary for (None for all)
            
        Returns:
            Position summary dictionary
        """
        if symbol:
            if symbol in self.position_metrics:
                metrics = self.position_metrics[symbol]
                position = self.trade_ledger.get_position(symbol)
                
                return {
                    'symbol': symbol,
                    'quantity': metrics.quantity,
                    'side': 'LONG' if metrics.quantity > 0 else 'SHORT' if metrics.quantity < 0 else 'FLAT',
                    'avg_price': metrics.avg_price,
                    'current_price': metrics.current_price,
                    'market_value': metrics.market_value,
                    'unrealized_pnl': metrics.unrealized_pnl,
                    'unrealized_pnl_pct': metrics.unrealized_pnl_pct,
                    'position_size_pct': metrics.position_size_pct,
                    'margin_used': metrics.margin_used,
                    'leverage': metrics.leverage,
                    'holding_period_minutes': metrics.holding_period_minutes,
                    'active_alerts': [alert.value for alert in metrics.active_alerts],
                    'intraday_high_pnl': metrics.intraday_high_pnl,
                    'intraday_low_pnl': metrics.intraday_low_pnl
                }
            else:
                return {'symbol': symbol, 'status': 'No position'}
        else:
            # Return summary for all positions
            return {
                'positions': {
                    symbol: self.get_position_summary(symbol) 
                    for symbol in self.position_metrics.keys()
                },
                'portfolio_metrics': {
                    'num_positions': self.portfolio_metrics.num_positions,
                    'gross_exposure': self.portfolio_metrics.gross_exposure,
                    'net_exposure': self.portfolio_metrics.net_exposure,
                    'portfolio_leverage': self.portfolio_metrics.portfolio_leverage,
                    'margin_utilization_pct': self.portfolio_metrics.margin_utilization_pct,
                    'total_unrealized_pnl': self.portfolio_metrics.total_unrealized_pnl,
                    'largest_position_pct': self.portfolio_metrics.largest_position_pct,
                    'var_95': self.portfolio_metrics.var_95
                }
            }
    
    def get_risk_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive risk dashboard data"""
        return {
            'timestamp': datetime.now().isoformat(),
            'portfolio_metrics': {
                'total_capital': self.portfolio_metrics.total_capital,
                'used_margin': self.portfolio_metrics.used_margin,
                'available_margin': self.portfolio_metrics.available_margin,
                'margin_utilization_pct': self.portfolio_metrics.margin_utilization_pct,
                'portfolio_leverage': self.portfolio_metrics.portfolio_leverage,
                'gross_exposure': self.portfolio_metrics.gross_exposure,
                'net_exposure': self.portfolio_metrics.net_exposure,
                'num_positions': self.portfolio_metrics.num_positions,
                'largest_position_pct': self.portfolio_metrics.largest_position_pct,
                'var_95': self.portfolio_metrics.var_95
            },
            'position_alerts': {
                symbol: [alert.value for alert in metrics.active_alerts]
                for symbol, metrics in self.position_metrics.items()
                if metrics.active_alerts
            },
            'top_positions': sorted(
                [
                    {
                        'symbol': symbol,
                        'pnl': metrics.unrealized_pnl,
                        'pnl_pct': metrics.unrealized_pnl_pct,
                        'size_pct': metrics.position_size_pct
                    }
                    for symbol, metrics in self.position_metrics.items()
                ],
                key=lambda x: abs(x['pnl']),
                reverse=True
            )[:5],  # Top 5 by absolute P&L
            'risk_warnings': self._get_risk_warnings()
        }
    
    def _get_risk_warnings(self) -> List[str]:
        """Get current risk warnings"""
        warnings = []
        
        # Margin warnings
        if self.portfolio_metrics.margin_utilization_pct > 90:
            warnings.append("CRITICAL: Margin utilization >90%")
        elif self.portfolio_metrics.margin_utilization_pct > 75:
            warnings.append("WARNING: High margin utilization")
        
        # Concentration warnings
        if self.portfolio_metrics.largest_position_pct > 30:
            warnings.append("WARNING: High position concentration")
        
        # Leverage warnings
        if self.portfolio_metrics.portfolio_leverage > 8:
            warnings.append("WARNING: High portfolio leverage")
        
        # Correlation warnings
        if self.portfolio_metrics.nifty_banknifty_correlation > 0.8 and self.portfolio_metrics.num_positions >= 2:
            warnings.append("WARNING: High correlation risk")
        
        return warnings
    
    def close_position(self, symbol: str, reason: str = "Manual close") -> Dict[str, Any]:
        """
        Close a position (generates exit signal)
        
        Args:
            symbol: Symbol to close
            reason: Reason for closure
            
        Returns:
            Close order details
        """
        position = self.trade_ledger.get_position(symbol)
        
        if not position or position.quantity == 0:
            return {'status': 'error', 'message': f'No position to close for {symbol}'}
        
        # Generate close order
        close_order = {
            'symbol': symbol,
            'action': 'SELL' if position.quantity > 0 else 'BUY',
            'quantity': abs(position.quantity),
            'order_type': 'MARKET',
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"Position close requested for {symbol}: {reason}")
        return {'status': 'success', 'close_order': close_order}


# Global position manager instance
_position_manager = None
_pm_lock = threading.Lock()


def get_position_manager(trade_ledger: TradeLedger = None) -> PositionManager:
    """
    Get the global PositionManager instance (singleton)
    
    Args:
        trade_ledger: TradeLedger instance (only used on first call)
        
    Returns:
        PositionManager instance
    """
    global _position_manager
    
    with _pm_lock:
        if _position_manager is None:
            _position_manager = PositionManager(trade_ledger)
    
    return _position_manager


# Example usage and alert handler
def example_alert_handler(alert_data: Dict[str, Any]) -> None:
    """Example alert handler function"""
    print(f"🚨 ALERT: {alert_data['message']}")
    
    # Here you could:
    # - Send Slack/email notifications
    # - Log to monitoring system
    # - Trigger automated responses
    # - Update dashboard


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize position manager
    pm = PositionManager()
    
    # Register alert handler
    pm.register_alert_callback(example_alert_handler)
    
    # Start monitoring
    pm.start_monitoring()
    
    # Simulate market data updates
    pm.update_market_data({
        'NIFTY': 19550.0,
        'BANKNIFTY': 44800.0,
        'FINNIFTY': 19200.0
    })
    
    # Get position summary
    summary = pm.get_position_summary()
    print(f"Position Summary: {summary}")
    
    # Get risk dashboard
    dashboard = pm.get_risk_dashboard()
    print(f"Risk Dashboard: {dashboard}")
    
    # Keep running for a while
    try:
        time.sleep(10)
    finally:
        pm.stop_monitoring()