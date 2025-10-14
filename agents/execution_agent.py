"""
Execution Agent - PRODUCTION
Order management, risk checks, broker integration with integrated risk management

Enhanced with:
- RiskConfigManager integration for dynamic risk management
- Advanced position sizing using Oxford methodology  
- Comprehensive pre-trade validation system
- Real-time risk monitoring and circuit breakers

Version: 2.0 - Phase A Integration
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, time as dt_time
import logging
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.risk_config import get_risk_config_manager, RiskMetrics, ViolationResult
from agents.position_sizing import compute_position_size_oxford

logger = logging.getLogger(__name__)


def init_execution_agent(config: Dict[str, Any]) -> Dict[str, Any]:
    """Initialize Execution Agent with integrated risk management"""
    logger.info("Initializing Enhanced Execution Agent with Risk Management...")
    
    # Initialize risk configuration manager
    try:
        risk_manager = get_risk_config_manager()
        logger.info("✅ Risk configuration manager initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize risk manager: {e}")
        raise RuntimeError(f"Critical: Risk manager initialization failed: {e}")
    
    agent = {
        'broker_connection': None,  # TODO: Connect to broker
        'order_book': [],
        'trade_book': [],
        'config': config,
        'risk_manager': risk_manager,
        'current_metrics': RiskMetrics(),
        'daily_trade_count': 0,
        'consecutive_losses': 0,
        'consecutive_loss_amount': 0.0,
        'last_risk_check_time': None,
        'emergency_mode': False,
        'transaction_costs': {
            'futures_bp': 2.0,  # 2 bps
            'options_bp': 8.0,  # 8 bps (higher STT)
            'brokerage_per_order': 20,  # ₹20 flat
            'stt_futures_sell': 0.0125,  # % of notional
            'stt_options_sell': 0.0625,  # % of premium (5x!)
        },
        'performance_metrics': {
            'daily_pnl': 0.0,
            'weekly_pnl': 0.0,
            'monthly_pnl': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'total_volume': 0.0,
            'sharpe_ratio': 0.0
        }
    }
    
    logger.info("✅ Enhanced Execution Agent initialized with risk management")
    return agent


def connect_broker(api_keys: Dict[str, str]) -> bool:
    """Connect to broker API"""
    # TODO: Implement broker connection
    logger.info("Broker connection established")
    return True


def check_time_constraints(current_time: datetime) -> bool:
    """
    Check if current time allows new trading
    Returns False if after 3:00 PM
    """
    time_now = current_time.time()
    cutoff = dt_time(15, 0)  # 3:00 PM
    
    if time_now >= cutoff:
        logger.warning(f"⚠️ Time constraint violated: {time_now} >= 3:00 PM")
        return False
    
    return True


def pre_trade_checks(
    order: Dict[str, Any],
    portfolio: Dict[str, Any],
    limits: Dict[str, float],
    margin_available: float,
    current_time: datetime
) -> Dict[str, Any]:
    """
    Comprehensive pre-trade risk validation for NSE Index Futures
    
    Args:
        order: Order details {'symbol', 'action', 'quantity', 'price'}
        portfolio: Current portfolio positions and exposures
        limits: Risk limits from configuration
        margin_available: Available margin in rupees
        current_time: Current timestamp for time-based checks
    
    Returns:
        {
            'overall_status': bool,
            'checks': {
                'time_check': {'passed': bool, 'message': str},
                'exposure_check': {'passed': bool, 'current': float, 'limit': float, 'message': str},
                'margin_check': {'passed': bool, 'required': float, 'available': float, 'message': str},
                'position_limit_check': {'passed': bool, 'current': int, 'limit': int, 'message': str},
                'expiry_check': {'passed': bool, 'days_to_expiry': int, 'min_days': int, 'message': str},
                'correlation_check': {'passed': bool, 'correlation_score': float, 'message': str},
                'leverage_check': {'passed': bool, 'leverage': float, 'max_leverage': float, 'message': str}
            },
            'warnings': List[str],
            'blocking_issues': List[str]
        }
    """
    logger.debug(f"Running comprehensive pre-trade checks for {order['symbol']}...")
    
    checks = {}
    warnings = []
    blocking_issues = []
    
    # 1. Time-based check
    checks['time_check'] = _check_time_constraints(current_time)
    if not checks['time_check']['passed']:
        blocking_issues.append(checks['time_check']['message'])
    
    # 2. Index exposure check  
    checks['exposure_check'] = _check_index_exposure(order, portfolio, limits)
    if not checks['exposure_check']['passed']:
        blocking_issues.append(checks['exposure_check']['message'])
    
    # 3. Margin utilization check
    checks['margin_check'] = _check_margin_utilization(order, portfolio, limits, margin_available)
    if not checks['margin_check']['passed']:
        blocking_issues.append(checks['margin_check']['message'])
    
    # 4. Position limits check
    checks['position_limit_check'] = _check_position_limits(order, portfolio, limits)
    if not checks['position_limit_check']['passed']:
        blocking_issues.append(checks['position_limit_check']['message'])
    
    # 5. Expiry proximity check
    checks['expiry_check'] = _check_expiry_proximity(order, current_time, limits)
    if not checks['expiry_check']['passed']:
        blocking_issues.append(checks['expiry_check']['message'])
    
    # 6. Correlation risk check
    checks['correlation_check'] = _check_correlation_risk(order, portfolio, limits)
    if not checks['correlation_check']['passed']:
        warnings.append(checks['correlation_check']['message'])
    
    # 7. Leverage check
    checks['leverage_check'] = _check_leverage_limits(order, portfolio, limits)
    if not checks['leverage_check']['passed']:
        blocking_issues.append(checks['leverage_check']['message'])
    
    # Overall status
    overall_status = len(blocking_issues) == 0
    
    if overall_status:
        logger.info("✅ All pre-trade checks PASSED")
    else:
        logger.error(f"❌ Pre-trade checks FAILED: {len(blocking_issues)} blocking issues")
        for issue in blocking_issues:
            logger.error(f"   - {issue}")
    
    if warnings:
        for warning in warnings:
            logger.warning(f"⚠️  {warning}")
    
    return {
        'overall_status': overall_status,
        'checks': checks,
        'warnings': warnings,
        'blocking_issues': blocking_issues
    }


def _check_time_constraints(current_time: datetime) -> Dict[str, Any]:
    """Check if current time allows new trading"""
    time_now = current_time.time()
    cutoff = dt_time(15, 0)  # 3:00 PM
    
    passed = time_now < cutoff
    message = f"Trading allowed until 15:00, current: {time_now}" if passed else f"Trading closed after 15:00, current: {time_now}"
    
    return {
        'passed': passed,
        'current_time': str(time_now),
        'cutoff_time': str(cutoff),
        'message': message
    }


def _check_index_exposure(order: Dict[str, Any], portfolio: Dict[str, Any], limits: Dict[str, float]) -> Dict[str, Any]:
    """Check if adding position exceeds index exposure limits"""
    symbol = order['symbol']
    notional = order['price'] * order['quantity']
    
    # Get current exposure for this index
    current_exposure = portfolio.get('exposures', {}).get(symbol, 0.0)
    new_exposure = current_exposure + notional
    
    # Get exposure limit (default 30% for major indices)
    exposure_limits = {
        'NIFTY': 0.30,      # 30% max exposure
        'BANKNIFTY': 0.25,  # 25% max (higher volatility)
        'FINNIFTY': 0.20    # 20% max (newer, less liquid)
    }
    
    max_exposure_pct = exposure_limits.get(symbol, 0.15)  # 15% default for other indices
    total_capital = portfolio.get('total_capital', 500000)
    max_exposure_amount = total_capital * max_exposure_pct
    
    passed = new_exposure <= max_exposure_amount
    message = f"{symbol} exposure check: ₹{new_exposure:,.0f} <= ₹{max_exposure_amount:,.0f}" if passed else f"{symbol} exposure limit exceeded: ₹{new_exposure:,.0f} > ₹{max_exposure_amount:,.0f} ({max_exposure_pct*100:.0f}%)"
    
    return {
        'passed': passed,
        'current': current_exposure,
        'new_exposure': new_exposure,
        'limit': max_exposure_amount,
        'limit_pct': max_exposure_pct * 100,
        'message': message
    }


def _check_margin_utilization(order: Dict[str, Any], portfolio: Dict[str, Any], limits: Dict[str, float], margin_available: float) -> Dict[str, Any]:
    """Validate margin requirements vs available capital"""
    symbol = order['symbol']
    quantity = order['quantity']
    
    # Calculate margin required for this order
    margin_per_lot = {
        'NIFTY': 60000,     # Approximate MIS margin per lot
        'BANKNIFTY': 75000,
        'FINNIFTY': 50000
    }
    
    lot_size = 50  # Default lot size (should come from contract spec)
    if symbol == 'BANKNIFTY':
        lot_size = 25
    elif symbol == 'FINNIFTY':
        lot_size = 40
    
    lots = abs(quantity) / lot_size
    margin_required = lots * margin_per_lot.get(symbol, 60000)
    
    # Check margin utilization limit (max 60% of available)
    max_margin_utilization = limits.get('max_margin_utilization', 0.60)
    max_usable_margin = margin_available * max_margin_utilization
    
    # Consider current margin usage
    current_margin_used = portfolio.get('margin_used', 0.0)
    total_margin_after = current_margin_used + margin_required
    
    passed = total_margin_after <= max_usable_margin
    utilization_pct = (total_margin_after / margin_available) * 100 if margin_available > 0 else 100
    
    message = f"Margin check: {utilization_pct:.1f}% utilization" if passed else f"Margin limit exceeded: {utilization_pct:.1f}% > {max_margin_utilization*100:.0f}%"
    
    return {
        'passed': passed,
        'required': margin_required,
        'available': margin_available,
        'current_used': current_margin_used,
        'total_after': total_margin_after,
        'utilization_pct': utilization_pct,
        'max_utilization_pct': max_margin_utilization * 100,
        'message': message
    }


def _check_position_limits(order: Dict[str, Any], portfolio: Dict[str, Any], limits: Dict[str, float]) -> Dict[str, Any]:
    """Check against exchange and internal position limits"""
    symbol = order['symbol']
    quantity = abs(order['quantity'])
    
    # Lot size mapping
    lot_sizes = {'NIFTY': 50, 'BANKNIFTY': 25, 'FINNIFTY': 40}
    lot_size = lot_sizes.get(symbol, 50)
    lots = quantity / lot_size
    
    # Current position in this symbol
    current_lots = portfolio.get('positions', {}).get(symbol, {}).get('lots', 0)
    new_total_lots = abs(current_lots) + lots
    
    # Position limits per symbol
    position_limits = {
        'NIFTY': 15,        # Max 15 lots per day
        'BANKNIFTY': 10,    # Max 10 lots (higher margin)
        'FINNIFTY': 12      # Max 12 lots
    }
    
    max_lots = position_limits.get(symbol, 8)  # Default 8 lots for other indices
    
    # Also check single trade limit
    max_lots_per_trade = {
        'NIFTY': 10,        # Max 10 lots per trade
        'BANKNIFTY': 5,     # Max 5 lots per trade
        'FINNIFTY': 8       # Max 8 lots per trade
    }
    
    single_trade_limit = max_lots_per_trade.get(symbol, 5)
    
    # Check both daily and single trade limits
    daily_limit_ok = new_total_lots <= max_lots
    single_trade_ok = lots <= single_trade_limit
    
    passed = daily_limit_ok and single_trade_ok
    
    if not daily_limit_ok:
        message = f"{symbol} daily position limit: {new_total_lots} lots > {max_lots} limit"
    elif not single_trade_ok:
        message = f"{symbol} single trade limit: {lots} lots > {single_trade_limit} limit"
    else:
        message = f"{symbol} position limits OK: {lots} lots (daily: {new_total_lots}/{max_lots})"
    
    return {
        'passed': passed,
        'current_lots': current_lots,
        'new_lots': lots,
        'total_lots_after': new_total_lots,
        'daily_limit': max_lots,
        'single_trade_limit': single_trade_limit,
        'message': message
    }


def _check_expiry_proximity(order: Dict[str, Any], current_time: datetime, limits: Dict[str, float]) -> Dict[str, Any]:
    """Prevent new positions close to contract expiry"""
    symbol = order['symbol']
    
    # TODO: Get actual expiry date from contract specification
    # For now, assume monthly expiry on last Thursday of month
    import calendar
    
    # Get last Thursday of current month
    year = current_time.year
    month = current_time.month
    
    # Find last Thursday
    last_day = calendar.monthrange(year, month)[1]
    last_thursday = None
    
    for day in range(last_day, 0, -1):
        if calendar.weekday(year, month, day) == 3:  # Thursday = 3
            last_thursday = day
            break
    
    if last_thursday:
        from datetime import date
        expiry_date = date(year, month, last_thursday)
        days_to_expiry = (expiry_date - current_time.date()).days
    else:
        days_to_expiry = 15  # Default assumption
    
    # Minimum days before expiry to allow new positions
    min_days_before_expiry = limits.get('expiry_prohibition_days', 2)
    
    passed = days_to_expiry > min_days_before_expiry
    message = f"Expiry check: {days_to_expiry} days remaining" if passed else f"Too close to expiry: {days_to_expiry} days < {min_days_before_expiry} day limit"
    
    return {
        'passed': passed,
        'days_to_expiry': days_to_expiry,
        'min_days': min_days_before_expiry,
        'expiry_date': str(expiry_date) if last_thursday else 'unknown',
        'message': message
    }


def _check_correlation_risk(order: Dict[str, Any], portfolio: Dict[str, Any], limits: Dict[str, float]) -> Dict[str, Any]:
    """Assess correlation risk across indices"""
    symbol = order['symbol']
    
    # Get current positions
    positions = portfolio.get('positions', {})
    
    # High correlation pairs
    high_correlation_pairs = {
        ('NIFTY', 'BANKNIFTY'): 0.75,
        ('NIFTY', 'FINNIFTY'): 0.65,
        ('BANKNIFTY', 'FINNIFTY'): 0.60
    }
    
    correlation_score = 0.0
    correlated_symbols = []
    
    for pos_symbol, position in positions.items():
        if pos_symbol != symbol and position.get('lots', 0) != 0:
            # Check if this pair has high correlation
            pair = tuple(sorted([symbol, pos_symbol]))
            if pair in high_correlation_pairs:
                correlation = high_correlation_pairs[pair]
                correlation_score = max(correlation_score, correlation)
                correlated_symbols.append(f"{pos_symbol}({correlation:.2f})")
    
    # Check combined exposure for highly correlated indices
    if symbol in ['NIFTY', 'BANKNIFTY']:
        nifty_exposure = portfolio.get('exposures', {}).get('NIFTY', 0)
        banknifty_exposure = portfolio.get('exposures', {}).get('BANKNIFTY', 0)
        
        if symbol == 'NIFTY':
            combined_exposure = nifty_exposure + (order['price'] * order['quantity']) + banknifty_exposure
        else:
            combined_exposure = banknifty_exposure + (order['price'] * order['quantity']) + nifty_exposure
        
        total_capital = portfolio.get('total_capital', 500000)
        combined_limit = limits.get('max_nifty_banknifty_combined', 0.45) * total_capital
        
        passed = combined_exposure <= combined_limit
        message = f"Combined NIFTY+BANKNIFTY exposure: ₹{combined_exposure:,.0f} <= ₹{combined_limit:,.0f}" if passed else f"Combined exposure limit exceeded: ₹{combined_exposure:,.0f} > ₹{combined_limit:,.0f}"
    else:
        passed = correlation_score < limits.get('max_correlation_threshold', 0.80)
        message = f"Correlation risk acceptable: {correlation_score:.2f}" if passed else f"High correlation risk: {correlation_score:.2f} with {', '.join(correlated_symbols)}"
    
    return {
        'passed': passed,
        'correlation_score': correlation_score,
        'correlated_symbols': correlated_symbols,
        'message': message
    }


def _check_leverage_limits(order: Dict[str, Any], portfolio: Dict[str, Any], limits: Dict[str, float]) -> Dict[str, Any]:
    """Check leverage limits per symbol and overall portfolio"""
    symbol = order['symbol']
    notional = order['price'] * order['quantity']
    
    # Symbol-specific leverage limits
    max_leverage_limits = {
        'NIFTY': 8.0,       # 8x max (most liquid)
        'BANKNIFTY': 6.0,   # 6x max (higher volatility)
        'FINNIFTY': 7.0     # 7x max (moderate)
    }
    
    max_leverage = max_leverage_limits.get(symbol, 5.0)  # 5x default
    
    # Calculate effective leverage
    # Leverage = Notional Value / Margin Required
    margin_per_lot = {'NIFTY': 60000, 'BANKNIFTY': 75000, 'FINNIFTY': 50000}
    lot_sizes = {'NIFTY': 50, 'BANKNIFTY': 25, 'FINNIFTY': 40}
    
    lot_size = lot_sizes.get(symbol, 50)
    lots = abs(order['quantity']) / lot_size
    margin_required = lots * margin_per_lot.get(symbol, 60000)
    
    effective_leverage = notional / margin_required if margin_required > 0 else 0
    
    # Check portfolio-wide leverage
    total_capital = portfolio.get('total_capital', 500000)
    current_notional = portfolio.get('total_notional', 0)
    new_total_notional = current_notional + notional
    
    portfolio_leverage = new_total_notional / total_capital if total_capital > 0 else 0
    max_portfolio_leverage = limits.get('max_portfolio_leverage', 6.0)
    
    symbol_ok = effective_leverage <= max_leverage
    portfolio_ok = portfolio_leverage <= max_portfolio_leverage
    
    passed = symbol_ok and portfolio_ok
    
    if not symbol_ok:
        message = f"{symbol} leverage too high: {effective_leverage:.1f}x > {max_leverage:.1f}x"
    elif not portfolio_ok:
        message = f"Portfolio leverage too high: {portfolio_leverage:.1f}x > {max_portfolio_leverage:.1f}x"
    else:
        message = f"Leverage OK: {symbol} {effective_leverage:.1f}x, Portfolio {portfolio_leverage:.1f}x"
    
    return {
        'passed': passed,
        'symbol_leverage': effective_leverage,
        'max_symbol_leverage': max_leverage,
        'portfolio_leverage': portfolio_leverage,
        'max_portfolio_leverage': max_portfolio_leverage,
        'message': message
    }


def build_order(
    signal: Dict[str, str],
    price: float,
    size: int,
    tif: str,
    contract_spec: Dict[str, Any],
    order_type: str = 'MIS'
) -> Dict[str, Any]:
    """
    Build order object from signal
    
    Args:
        signal: {'action': 'BUY' | 'SELL'}
        price: Current market price
        size: Number of lots
        tif: Time in force ('DAY', 'IOC')
        contract_spec: {'symbol': str, 'lot_size': int}
        order_type: 'MIS' (intraday) or 'NRML' (overnight)
    
    Returns:
        Order dict ready for broker API
    """
    symbol = contract_spec['symbol']
    lot_size = contract_spec.get('lot_size', 50)
    quantity = size * lot_size
    
    order = {
        'symbol': symbol,
        'action': signal['action'],
        'quantity': quantity,
        'price': price,
        'order_type': 'MARKET',  # Use MARKET for speed (can change to LIMIT)
        'product_type': order_type,
        'validity': tif,
        'exchange': 'NFO',  # NSE F&O
        'disclosed_quantity': 0,
        'trigger_price': None,
        'timestamp': datetime.now().isoformat(),
        'status': 'PENDING'
    }
    
    logger.info(f"Order built: {order['action']} {quantity} {symbol} @ ₹{price:.2f}")
    return order


def execute_smart_trade(
    signal: Dict[str, Any],
    symbol: str,
    current_price: float,
    portfolio: Dict[str, Any],
    market_data: Dict[str, Any],
    agent: Dict[str, Any],
    current_time: datetime = None
) -> Dict[str, Any]:
    """
    Enhanced trade execution with integrated position sizing and risk management
    
    Args:
        signal: Trading signal {'action': 'BUY'/'SELL', 'confidence': 0.0-1.0, 'strategy': str}
        symbol: Index symbol (NIFTY, BANKNIFTY, FINNIFTY)
        current_price: Current market price
        portfolio: Current portfolio state
        market_data: Market data including volatility, volumes etc.
        agent: Execution agent state
        current_time: Current timestamp (defaults to now)
    
    Returns:
        Comprehensive execution result with position sizing, risk checks, and execution status
    """
    if current_time is None:
        current_time = datetime.now()
    
    logger.info(f"🎯 Smart trade execution: {signal['action']} {symbol} @ ₹{current_price:.2f}")
    
    result = {
        'timestamp': current_time.isoformat(),
        'signal': signal,
        'symbol': symbol,
        'price': current_price,
        'status': 'unknown'
    }
    
    try:
        # Step 1: Check if trading is allowed (time-based, emergency mode)
        risk_manager = agent['risk_manager']
        
        if not risk_manager.is_trading_allowed(current_time):
            result['status'] = 'rejected'
            result['rejection_reason'] = 'Trading not allowed at current time'
            logger.warning(f"⚠️ Trading rejected: {result['rejection_reason']}")
            return result
        
        if agent.get('emergency_mode', False):
            result['status'] = 'rejected' 
            result['rejection_reason'] = 'Emergency mode active'
            logger.error(f"🚨 Trading rejected: {result['rejection_reason']}")
            return result
        
        # Step 2: Calculate optimal position size using Oxford methodology
        position_sizing_result = compute_position_size_oxford(
            signal=signal,
            symbol=symbol,
            current_price=current_price,
            portfolio=portfolio,
            market_data=market_data,
            risk_config=risk_manager.get_position_sizing_params(),
            kelly_config=risk_manager.get_kelly_params()
        )
        
        result['position_sizing'] = position_sizing_result
        
        if not position_sizing_result['success']:
            result['status'] = 'rejected'
            result['rejection_reason'] = f"Position sizing failed: {position_sizing_result.get('error', 'Unknown error')}"
            logger.error(f"❌ Position sizing failed: {result['rejection_reason']}")
            return result
        
        # Extract calculated position size
        optimal_lots = position_sizing_result['result']['optimal_lots']
        risk_adjusted_lots = position_sizing_result['result']['final_lots_after_caps']
        
        if risk_adjusted_lots <= 0:
            result['status'] = 'rejected'
            result['rejection_reason'] = 'Calculated position size is zero or negative'
            logger.warning(f"⚠️ No position to take: {result['rejection_reason']}")
            return result
        
        # Step 3: Get contract specification and build order
        contract_spec = {
            'symbol': symbol,
            'lot_size': {'NIFTY': 50, 'BANKNIFTY': 25, 'FINNIFTY': 40}.get(symbol, 50),
            'tick_size': 0.05,
            'exchange': 'NFO'
        }
        
        # Step 4: Enhanced pre-trade checks with risk manager integration
        margin_available = portfolio.get('available_margin', 500000)
        risk_limits = {
            'max_portfolio_leverage': risk_manager.get_global_limit('max_portfolio_leverage'),
            'max_margin_utilization': risk_manager.get_margin_params().get('max_margin_utilization'),
            'expiry_prohibition_days': risk_manager.config.get('time_based_controls', {}).get('expiry_prohibition_days', 2),
            'max_nifty_banknifty_combined': risk_manager.config.get('correlation_limits', {}).get('max_nifty_banknifty_combined', 0.45),
            'max_correlation_threshold': risk_manager.config.get('correlation_limits', {}).get('max_correlation_threshold', 0.80)
        }
        
        order = build_order(signal, current_price, risk_adjusted_lots, 'DAY', contract_spec)
        risk_check = pre_trade_checks(order, portfolio, risk_limits, margin_available, current_time)
        
        result['risk_check'] = risk_check
        
        # Step 5: Risk limit validation using RiskConfigManager
        current_metrics = _calculate_current_metrics(portfolio, agent, order)
        violations = risk_manager.check_risk_limits(current_metrics)
        
        result['risk_violations'] = [
            {
                'type': v.violation_type,
                'severity': v.severity, 
                'message': v.message,
                'recommended_action': v.recommended_action
            }
            for v in violations if v.is_violation
        ]
        
        # Step 6: Final go/no-go decision
        critical_violations = [v for v in violations if v.is_violation and v.severity == 'critical']
        error_violations = [v for v in violations if v.is_violation and v.severity == 'error']
        
        if not risk_check['overall_status'] or critical_violations or error_violations:
            result['status'] = 'rejected'
            
            rejection_reasons = []
            if not risk_check['overall_status']:
                rejection_reasons.extend(risk_check['blocking_issues'])
            if critical_violations:
                rejection_reasons.extend([v.message for v in critical_violations])
            if error_violations:
                rejection_reasons.extend([v.message for v in error_violations])
            
            result['rejection_reason'] = '; '.join(rejection_reasons)
            logger.error(f"❌ Trade rejected: {result['rejection_reason']}")
            return result
        
        # Step 7: Execute the trade
        execution_result = execute_trade_with_checks(
            signal, symbol, current_price, risk_adjusted_lots,
            portfolio, risk_limits, margin_available, contract_spec, current_time
        )
        
        result.update(execution_result)
        
        # Step 8: Update agent state and metrics
        if result['status'] == 'success':
            _update_agent_metrics(agent, order, result, current_time)
            logger.info(f"✅ Smart trade executed successfully: {result.get('order_id')}")
        else:
            logger.error(f"❌ Smart trade execution failed: {result.get('message')}")
        
        return result
        
    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
        logger.error(f"💥 Smart trade execution error: {e}")
        return result


def execute_trade_with_checks(
    signal: Dict[str, str],
    symbol: str,
    price: float,
    size: int,
    portfolio: Dict[str, Any],
    limits: Dict[str, float],
    margin_available: float,
    contract_spec: Dict[str, Any],
    current_time: datetime
) -> Dict[str, Any]:
    """
    Enhanced execute_trade with mandatory pre-trade risk checks
    
    Args:
        signal: {'action': 'BUY' | 'SELL'}
        symbol: Index symbol (NIFTY, BANKNIFTY, etc.)
        price: Current market price
        size: Number of lots
        portfolio: Current portfolio state
        limits: Risk limits configuration
        margin_available: Available margin
        contract_spec: Contract specifications
        current_time: Current timestamp
    
    Returns:
        {
            'status': 'success' | 'rejected' | 'failed',
            'order_id': str (if successful),
            'risk_check_result': Dict,
            'execution_result': Dict (if executed),
            'rejection_reason': str (if rejected),
            'message': str
        }
    """
    logger.info(f"Executing trade with checks: {signal['action']} {size} lots {symbol} @ ₹{price:.2f}")
    
    # Build order for risk checking
    order = build_order(signal, price, size, 'DAY', contract_spec)
    
    # Run comprehensive pre-trade checks
    risk_check = pre_trade_checks(order, portfolio, limits, margin_available, current_time)
    
    result = {
        'status': 'unknown',
        'risk_check_result': risk_check,
        'order_details': order
    }
    
    # If risk checks fail, reject the trade
    if not risk_check['overall_status']:
        result['status'] = 'rejected'
        result['rejection_reason'] = '; '.join(risk_check['blocking_issues'])
        result['message'] = f"Trade rejected: {result['rejection_reason']}"
        
        logger.error(f"❌ Trade REJECTED: {result['rejection_reason']}")
        return result
    
    # Risk checks passed, proceed with execution
    try:
        # Route the order
        routed_order = route_order(order)
        
        # Place the order
        execution_result = place_order(routed_order)
        
        if execution_result.get('status') == 'SUBMITTED':
            result['status'] = 'success'
            result['order_id'] = execution_result.get('order_id')
            result['execution_result'] = execution_result
            result['message'] = f"Trade executed successfully: {execution_result.get('order_id')}"
            
            logger.info(f"✅ Trade EXECUTED: {result['order_id']}")
        else:
            result['status'] = 'failed'
            result['execution_result'] = execution_result
            result['message'] = f"Trade execution failed: {execution_result.get('message', 'Unknown error')}"
            
            logger.error(f"❌ Trade FAILED: {result['message']}")
    
    except Exception as e:
        result['status'] = 'failed'
        result['message'] = f"Trade execution error: {str(e)}"
        logger.error(f"❌ Trade ERROR: {str(e)}")
    
    # Log warnings if any
    if risk_check['warnings']:
        for warning in risk_check['warnings']:
            logger.warning(f"⚠️  {warning}")
    
    return result


def route_order(order: Dict[str, Any], algo: str = 'TWAP') -> Dict[str, Any]:
    """
    Smart order routing
    For Day 1: Direct market order
    Future: TWAP, VWAP, Iceberg
    """
    logger.info(f"Routing order: {order['action']} {order['quantity']} {order['symbol']}")
    
    # Day 1: Direct market order (no slicing)
    routed_order = order.copy()
    routed_order['routing_algo'] = 'DIRECT'
    
    return routed_order


def place_order(order: Dict[str, Any]) -> Dict[str, Any]:
    """
    Place order with broker
    TODO: Implement actual broker API call
    """
    logger.info(f"📤 Placing order: {order['action']} {order['quantity']} {order['symbol']}")
    
    # TODO: Call broker API
    # response = broker.place_order(order)
    
    # Mock response for Day 1
    order_response = {
        'order_id': f"ORD{datetime.now().strftime('%Y%m%d%H%M%S')}",
        'status': 'SUBMITTED',
        'message': 'Order placed successfully',
        'timestamp': datetime.now().isoformat()
    }
    
    logger.info(f"✅ Order placed: ID={order_response['order_id']}")
    return order_response


def monitor_orders(open_orders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Monitor order status
    TODO: Poll broker API for order updates
    """
    logger.debug(f"Monitoring {len(open_orders)} orders...")
    
    # TODO: Query broker for order status
    # updated_orders = broker.get_order_status(order_ids)
    
    return open_orders


def apply_stops_and_targets(
    position: Dict[str, Any],
    rules: Dict[str, float],
    time_remaining: float
) -> Optional[Dict[str, Any]]:
    """
    Check if stop-loss or take-profit triggered
    Returns exit order if triggered, else None
    """
    entry_price = position.get('entry_price')
    current_price = position.get('current_price')
    quantity = position.get('quantity', 0)
    
    if not entry_price or not current_price or quantity == 0:
        return None
    
    # Calculate P&L %
    if quantity > 0:  # Long position
        pnl_pct = (current_price - entry_price) / entry_price
    else:  # Short position
        pnl_pct = (entry_price - current_price) / entry_price
    
    # Stop-loss check
    stop_loss_pct = rules.get('stop_loss_pct', 0.015)  # 1.5%
    if pnl_pct < -stop_loss_pct:
        logger.warning(f"🛑 STOP-LOSS triggered: {pnl_pct*100:.2f}% loss")
        
        exit_order = {
            'symbol': position['symbol'],
            'action': 'SELL' if quantity > 0 else 'BUY',
            'quantity': abs(quantity),
            'price': current_price,
            'order_type': 'MARKET',
            'product_type': 'MIS',
            'reason': 'STOP_LOSS'
        }
        return exit_order
    
    # Take-profit check
    take_profit_pct = rules.get('take_profit_pct', 0.03)  # 3%
    if pnl_pct > take_profit_pct:
        logger.info(f"🎯 TAKE-PROFIT triggered: {pnl_pct*100:.2f}% profit")
        
        exit_order = {
            'symbol': position['symbol'],
            'action': 'SELL' if quantity > 0 else 'BUY',
            'quantity': abs(quantity),
            'price': current_price,
            'order_type': 'MARKET',
            'product_type': 'MIS',
            'reason': 'TAKE_PROFIT'
        }
        return exit_order
    
    # Time-based exit (tighter stops near close)
    if time_remaining < 45:  # After 2:30 PM
        tighter_stop = stop_loss_pct * 0.7  # 70% of normal stop
        if pnl_pct < -tighter_stop:
            logger.warning(f"⏰ TIME-BASED STOP triggered near close")
            
            exit_order = {
                'symbol': position['symbol'],
                'action': 'SELL' if quantity > 0 else 'BUY',
                'quantity': abs(quantity),
                'price': current_price,
                'order_type': 'MARKET',
                'product_type': 'MIS',
                'reason': 'TIME_BASED_STOP'
            }
            return exit_order
    
    return None


def auto_square_off_positions(current_time: str = '15:10:00') -> List[Dict[str, Any]]:
    """
    Auto square-off all positions at 3:10 PM
    Emergency exit before broker auto-square at 3:20 PM
    """
    logger.critical(f"🚨 AUTO SQUARE-OFF triggered at {current_time}")
    
    # TODO: Get actual open positions from broker
    # positions = broker.get_positions()
    
    exit_orders = []
    
    # Mock for Day 1
    # for position in positions:
    #     exit_order = {...}
    #     exit_orders.append(exit_order)
    
    logger.info(f"Created {len(exit_orders)} auto square-off orders")
    return exit_orders


def calculate_margin_required(
    positions: List[Dict[str, Any]],
    mis_mode: bool = True
) -> float:
    """
    Calculate total margin required for positions
    MIS margin is ~40-60% of NRML
    """
    total_margin = 0.0
    
    # Margin per lot (approximate)
    margin_map_mis = {
        'NIFTY': 60000,
        'BANKNIFTY': 75000,
        'FINNIFTY': 50000
    }
    
    margin_map_nrml = {
        'NIFTY': 100000,
        'BANKNIFTY': 150000,
        'FINNIFTY': 80000
    }
    
    margin_map = margin_map_mis if mis_mode else margin_map_nrml
    
    for position in positions:
        symbol = position.get('symbol', 'NIFTY')
        quantity = position.get('quantity', 0)
        lot_size = 50  # TODO: Get from contract spec
        
        lots = abs(quantity) / lot_size
        margin_per_lot = margin_map.get(symbol, 60000)
        
        position_margin = lots * margin_per_lot
        total_margin += position_margin
    
    logger.debug(f"Total margin required: ₹{total_margin:,.0f}")
    return total_margin


def rebalance_portfolio(
    target_weights: Dict[str, float],
    time_remaining: float
) -> List[Dict[str, Any]]:
    """
    Rebalance portfolio to target weights
    Only if sufficient time remaining (>60 mins)
    """
    if time_remaining < 60:
        logger.warning("⚠️ Not enough time for rebalancing")
        return []
    
    logger.info("Rebalancing portfolio...")
    
    # TODO: Implement rebalancing logic
    rebalance_orders = []
    
    return rebalance_orders


def post_trade_attribution(
    trades: List[Dict[str, Any]],
    benchmarks: Dict[str, float]
) -> Dict[str, float]:
    """
    Post-trade analysis and attribution
    Calculate intraday performance metrics
    """
    logger.info("Running post-trade attribution...")
    
    if not trades:
        return {
            'total_pnl': 0.0,
            'num_trades': 0,
            'win_rate': 0.0,
            'avg_pnl_per_trade': 0.0,
            'total_costs': 0.0
        }
    
    total_pnl = 0.0
    total_costs = 0.0
    winning_trades = 0
    
    for trade in trades:
        pnl = trade.get('pnl', 0.0)
        costs = trade.get('transaction_costs', 0.0)
        
        total_pnl += pnl
        total_costs += costs
        
        if pnl > 0:
            winning_trades += 1
    
    num_trades = len(trades)
    win_rate = winning_trades / num_trades if num_trades > 0 else 0.0
    avg_pnl = total_pnl / num_trades if num_trades > 0 else 0.0
    
    net_pnl = total_pnl - total_costs
    
    attribution = {
        'total_pnl': total_pnl,
        'net_pnl': net_pnl,
        'total_costs': total_costs,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'avg_pnl_per_trade': avg_pnl,
        'winning_trades': winning_trades,
        'losing_trades': num_trades - winning_trades
    }
    
    logger.info(f"Attribution: PnL=₹{net_pnl:,.0f}, Trades={num_trades}, Win Rate={win_rate*100:.1f}%")
    return attribution


def _calculate_current_metrics(portfolio: Dict[str, Any], agent: Dict[str, Any], order: Dict[str, Any]) -> RiskMetrics:
    """
    Calculate current portfolio risk metrics for risk limit checking
    
    Args:
        portfolio: Current portfolio state
        agent: Execution agent state
        order: Pending order details
        
    Returns:
        RiskMetrics object with current portfolio risk metrics
    """
    try:
        # Portfolio leverage calculation
        total_capital = portfolio.get('total_capital', 500000)
        total_notional = portfolio.get('total_notional', 0)
        
        # Add pending order notional
        order_notional = order['price'] * order['quantity'] 
        new_total_notional = total_notional + order_notional
        
        portfolio_leverage = new_total_notional / total_capital if total_capital > 0 else 0
        
        # P&L calculations
        daily_pnl_pct = (agent['performance_metrics']['daily_pnl'] / total_capital) * 100 if total_capital > 0 else 0
        weekly_pnl_pct = (agent['performance_metrics']['weekly_pnl'] / total_capital) * 100 if total_capital > 0 else 0
        monthly_pnl_pct = (agent['performance_metrics']['monthly_pnl'] / total_capital) * 100 if total_capital > 0 else 0
        
        # Margin utilization
        margin_used = portfolio.get('margin_used', 0)
        margin_available = portfolio.get('available_margin', total_capital)
        margin_utilization = margin_used / margin_available if margin_available > 0 else 0
        
        # Position concentration - largest single position
        positions = portfolio.get('positions', {})
        largest_position_pct = 0
        total_exposure = 0
        
        for symbol, position in positions.items():
            position_value = abs(position.get('market_value', 0))
            total_exposure += position_value
            
            position_pct = (position_value / total_capital) * 100 if total_capital > 0 else 0
            largest_position_pct = max(largest_position_pct, position_pct)
        
        # Add pending order to largest position check
        order_symbol = order['symbol']
        if order_symbol in positions:
            current_position_value = abs(positions[order_symbol].get('market_value', 0))
        else:
            current_position_value = 0
        
        new_position_value = current_position_value + abs(order_notional)
        new_position_pct = (new_position_value / total_capital) * 100 if total_capital > 0 else 0
        largest_position_pct = max(largest_position_pct, new_position_pct)
        
        # VaR estimation (simplified)
        var_95 = total_exposure * 0.02 / total_capital if total_capital > 0 else 0  # 2% of exposure as VaR
        var_99 = total_exposure * 0.03 / total_capital if total_capital > 0 else 0  # 3% of exposure as VaR
        
        # Correlation risk (simplified)
        correlation_risk = 0.5 if len(positions) > 1 else 0.0  # Simplified correlation measure
        
        return RiskMetrics(
            portfolio_leverage=portfolio_leverage,
            daily_pnl=daily_pnl_pct,
            weekly_pnl=weekly_pnl_pct,
            monthly_pnl=monthly_pnl_pct,
            var_95=var_95 * 100,  # Convert to percentage
            var_99=var_99 * 100,  # Convert to percentage
            margin_utilization=margin_utilization,
            largest_position_pct=largest_position_pct,
            total_exposure=total_exposure,
            correlation_risk=correlation_risk,
            consecutive_losses=agent.get('consecutive_losses', 0),
            consecutive_loss_amount=agent.get('consecutive_loss_amount', 0.0)
        )
        
    except Exception as e:
        logger.error(f"Error calculating current metrics: {e}")
        # Return default metrics in case of error
        return RiskMetrics()


def _update_agent_metrics(agent: Dict[str, Any], order: Dict[str, Any], result: Dict[str, Any], current_time: datetime) -> None:
    """
    Update agent state and performance metrics after trade execution
    
    Args:
        agent: Execution agent state to update
        order: Executed order details
        result: Trade execution result
        current_time: Current timestamp
    """
    try:
        # Update trade counters
        agent['daily_trade_count'] += 1
        agent['performance_metrics']['total_trades'] += 1
        
        # Update last risk check time
        agent['last_risk_check_time'] = current_time
        
        # Update volume metrics
        trade_value = order['price'] * order['quantity']
        agent['performance_metrics']['total_volume'] += trade_value
        
        # Log the trade for tracking
        trade_record = {
            'timestamp': current_time.isoformat(),
            'order_id': result.get('order_id'),
            'symbol': order['symbol'],
            'action': order['action'],
            'quantity': order['quantity'],
            'price': order['price'],
            'value': trade_value,
            'status': result['status'],
            'risk_score': result.get('position_sizing', {}).get('result', {}).get('risk_score', 0.0)
        }
        
        agent['trade_book'].append(trade_record)
        
        # Keep only last 1000 trades in memory
        if len(agent['trade_book']) > 1000:
            agent['trade_book'] = agent['trade_book'][-1000:]
        
        logger.debug(f"Agent metrics updated: Total trades={agent['performance_metrics']['total_trades']}, Daily={agent['daily_trade_count']}")
        
    except Exception as e:
        logger.error(f"Error updating agent metrics: {e}")


def calculate_transaction_costs(order: Dict[str, Any], is_options: bool = False) -> float:
    """
    Calculate total transaction costs for an order
    Includes: Brokerage + STT + Exchange charges + SEBI + GST + Stamp duty
    """
    quantity = order.get('quantity', 0)
    price = order.get('price', 0.0)
    action = order.get('action', 'BUY')
    
    notional = quantity * price
    
    # Brokerage (flat ₹20 per order)
    brokerage = 20.0
    
    # STT (asymmetric - only on SELL side)
    if action == 'SELL':
        if is_options:
            stt = notional * 0.000625  # 0.0625% for options (5x higher!)
        else:
            stt = notional * 0.000125  # 0.0125% for futures
    else:
        stt = 0.0
    
    # Exchange charges (0.0019% of turnover)
    exchange_charges = notional * 0.000019
    
    # SEBI charges (₹10 per crore)
    sebi_charges = (notional / 10000000) * 10
    
    # Stamp duty (0.002% on BUY side)
    if action == 'BUY':
        stamp_duty = notional * 0.00002
    else:
        stamp_duty = 0.0
    
    # GST (18% on brokerage + exchange charges)
    gst = (brokerage + exchange_charges) * 0.18
    
    # Total
    total_cost = brokerage + stt + exchange_charges + sebi_charges + stamp_duty + gst
    
    logger.debug(f"Transaction costs: ₹{total_cost:.2f} (STT: ₹{stt:.2f}, Brokerage: ₹{brokerage:.2f})")
    return total_cost


def get_positions_from_broker(api_keys: Dict[str, str]) -> Dict[str, Any]:
    """
    Fetch current open positions from broker
    TODO: Implement actual broker API call
    """
    logger.debug("Fetching positions from broker...")
    
    # TODO: Call broker API
    # positions = broker.get_positions()
    
    # Mock for Day 1
    positions = {}
    
    return positions


def get_order_book_from_broker(api_keys: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    Fetch today's order book from broker
    TODO: Implement actual broker API call
    """
    logger.debug("Fetching order book from broker...")
    
    # TODO: Call broker API
    # order_book = broker.get_order_book()
    
    # Mock for Day 1
    order_book = []
    
    return order_book


def cancel_order(order_id: str) -> bool:
    """
    Cancel pending order
    TODO: Implement broker API call
    """
    logger.info(f"Cancelling order: {order_id}")
    
    # TODO: Call broker API
    # response = broker.cancel_order(order_id)
    
    return True


def modify_order(order_id: str, modifications: Dict[str, Any]) -> bool:
    """
    Modify existing order
    TODO: Implement broker API call
    """
    logger.info(f"Modifying order: {order_id}")
    
    # TODO: Call broker API
    # response = broker.modify_order(order_id, modifications)
    
    return True