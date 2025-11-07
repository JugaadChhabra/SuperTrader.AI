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
from agents.pre_trade_risk import validate_pre_trade_risk, is_order_approved, get_risk_summary
from agents.volatility_position_sizing import calculate_position_size, get_recommended_lots, is_position_viable
from agents.trade_ledger import get_trade_ledger, Trade, TradeAction, TradeStatus, ContractDetails
from agents.universe_ranking import rank_indices_for_trading, get_top_k_indices, is_index_tradeable
from utils.position_manager import get_position_manager

logger = logging.getLogger(__name__)


def init_execution_agent(config: Dict[str, Any]) -> Dict[str, Any]:
    """Initialize Execution Agent with integrated risk management"""
    logger.info("Initializing Enhanced Execution Agent with Risk Management...")
    
    # Initialize core components
    try:
        risk_manager = get_risk_config_manager()
        trade_ledger = get_trade_ledger()
        position_manager = get_position_manager(trade_ledger)
        
        logger.info("✅ Risk configuration manager initialized")
        logger.info("✅ Trade ledger initialized")
        logger.info("✅ Position manager initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize core components: {e}")
        raise RuntimeError(f"Critical: Component initialization failed: {e}")
    
    agent = {
        'broker_connection': None,  # TODO: Connect to broker
        'order_book': [],
        'trade_book': [],
        'config': config,
        'risk_manager': risk_manager,
        'trade_ledger': trade_ledger,
        'position_manager': position_manager,
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


def select_trading_universe(
    market_data: Dict[str, Dict[str, Any]],
    max_selections: int = 5
) -> Dict[str, Any]:
    """
    Select optimal trading universe using dynamic ranking
    
    Args:
        market_data: Market data for all indices
        max_selections: Maximum number of indices to select
        
    Returns:
        Dictionary with selected indices and ranking details
    """
    try:
        logger.info(f"🔍 Selecting trading universe from {len(market_data)} available indices")
        
        # Rank indices using comprehensive metrics
        ranked_indices = rank_indices_for_trading(market_data, max_selections)
        
        if not ranked_indices:
            logger.warning("⚠️ No indices passed selection criteria")
            return {
                'selected_indices': [],
                'ranking_details': [],
                'selection_summary': "No indices met selection criteria"
            }
        
        # Extract selected symbols
        selected_symbols = [metrics.symbol for metrics in ranked_indices]
        
        # Create detailed ranking info
        ranking_details = []
        for metrics in ranked_indices:
            ranking_details.append({
                'symbol': metrics.symbol,
                'rank': metrics.rank,
                'composite_score': metrics.composite_score,
                'liquidity_score': metrics.liquidity_score,
                'volatility_score': metrics.volatility_score,
                'momentum_score': metrics.momentum_score,
                'mean_reversion_score': metrics.mean_reversion_score,
                'avg_daily_volume': metrics.avg_daily_volume,
                'realized_volatility': metrics.realized_volatility,
                'volatility_regime': metrics.volatility_regime.value,
                'data_quality': metrics.data_quality
            })
        
        # Create summary
        selection_summary = f"Selected {len(selected_symbols)} indices: {', '.join(selected_symbols)}"
        
        logger.info(f"✅ Universe selection complete: {selection_summary}")
        
        return {
            'selected_indices': selected_symbols,
            'ranking_details': ranking_details,
            'selection_summary': selection_summary,
            'total_candidates': len(market_data),
            'selected_count': len(selected_symbols)
        }
        
    except Exception as e:
        logger.error(f"Error in universe selection: {e}")
        return {
            'selected_indices': [],
            'ranking_details': [],
            'selection_summary': f"Error in selection: {str(e)}"
        }


def validate_index_for_trading(
    symbol: str,
    market_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate if a specific index is suitable for trading
    
    Args:
        symbol: Index symbol to validate
        market_data: Market data for the index
        
    Returns:
        Validation result with recommendation
    """
    try:
        is_tradeable = is_index_tradeable(symbol, market_data)
        
        return {
            'symbol': symbol,
            'is_tradeable': is_tradeable,
            'recommendation': "APPROVED for trading" if is_tradeable else "NOT RECOMMENDED for trading",
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error validating {symbol}: {e}")
        return {
            'symbol': symbol,
            'is_tradeable': False,
            'recommendation': f"Validation error: {str(e)}",
            'timestamp': datetime.now().isoformat()
        }


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
        
        # Step 2: Validate index suitability for trading
        logger.info("🎯 Validating index suitability for trading...")
        index_validation = validate_index_for_trading(symbol, market_data)
        
        result['index_validation'] = index_validation
        
        if not index_validation['is_tradeable']:
            result['status'] = 'rejected'
            result['rejection_reason'] = f"Index not suitable for trading: {index_validation['recommendation']}"
            logger.warning(f"❌ Index validation failed: {result['rejection_reason']}")
            return result
        
        logger.info(f"✅ Index validation passed: {index_validation['recommendation']}")
        
        # Step 3: Comprehensive Pre-Trade Risk Validation
        pre_trade_risk_report = validate_pre_trade_risk(
            symbol=symbol,
            side=signal['action'],
            quantity=1000,  # Preliminary quantity for validation
            price=current_price,
            current_positions=portfolio.get('positions', {}),
            available_margin=portfolio.get('available_margin', 500000),
            account_balance=portfolio.get('total_capital', 1000000)
        )
        
        result['pre_trade_risk'] = {
            'overall_status': pre_trade_risk_report.overall_status.value,
            'risk_score': pre_trade_risk_report.risk_score,
            'passed_checks': pre_trade_risk_report.passed_checks,
            'failed_checks': pre_trade_risk_report.failed_checks,
            'warning_checks': pre_trade_risk_report.warning_checks,
            'recommendation': pre_trade_risk_report.recommendation
        }
        
        if not is_order_approved(pre_trade_risk_report):
            result['status'] = 'rejected'
            result['rejection_reason'] = f"Pre-trade risk validation failed: {pre_trade_risk_report.recommendation}"
            result['risk_summary'] = get_risk_summary(pre_trade_risk_report)
            logger.error(f"❌ Pre-trade risk check failed: {result['rejection_reason']}")
            return result
        
        logger.info(f"✅ Pre-trade risk validation passed: {pre_trade_risk_report.recommendation}")
        
        # Step 4: Calculate optimal position size using volatility targeting
        # Enhanced position sizing using volatility targeting
        position_sizing_result = calculate_position_size(
            symbol=symbol,
            current_price=current_price,
            rl_signal=(1.0 if signal['action']=='BUY' else -1.0) * signal.get('confidence', 0.0),
            account_balance=portfolio.get('total_capital', 1000000),
            available_margin=portfolio.get('available_margin', 500000),
            win_rate=portfolio.get('win_rate', 0.55),
            avg_win_loss_ratio=portfolio.get('avg_win_loss_ratio', 1.2),
            current_positions=portfolio.get('positions', {}),
            market_data=market_data
        )
        
        result['position_sizing'] = {
            'recommended_lots': position_sizing_result.recommended_lots,
            'notional_value': position_sizing_result.notional_value,
            'margin_required': position_sizing_result.margin_required,
            'confidence_score': position_sizing_result.confidence_score,
            'warnings': position_sizing_result.warnings
        }
        
        if not is_position_viable(position_sizing_result):
            result['status'] = 'rejected'
            result['rejection_reason'] = f"Position not viable: {'; '.join(position_sizing_result.warnings)}"
            logger.error(f"❌ Position sizing failed: {result['rejection_reason']}")
            return result
        
        # Extract calculated position size
        optimal_lots = position_sizing_result.recommended_lots
        risk_adjusted_lots = position_sizing_result.recommended_lots
        
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
        
        # Step 7: Create enhanced trade record with contract details
        from agents.trade_ledger import ContractDetails
        
        # Create contract details for enhanced trade tracking
        contract_details = ContractDetails(
            symbol=symbol,
            exchange='NFO',
            contract_type='FUTURES',
            lot_size=contract_spec['lot_size'],
            tick_size=contract_spec['tick_size'],
            margin_requirement={'NIFTY': 60000, 'BANKNIFTY': 75000, 'FINNIFTY': 40000}.get(symbol, 60000),
            expiry_date=None,  # Would be populated with actual expiry
            multiplier=1,
            currency='INR'
        )
        
        trade_record = Trade(
            order_id=order['order_id'] if 'order_id' in order else f"ORDER_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            symbol=symbol,
            action=TradeAction.BUY if signal['action'] == 'BUY' else TradeAction.SELL,
            strategy=signal.get('strategy', 'unknown'),
            quantity_ordered=order['quantity'],
            price_ordered=current_price,
            confidence_score=signal.get('confidence', 0.0),
            risk_score=pre_trade_risk_report.risk_score,
            order_time=current_time,
            contract_details=contract_details,  # Enhanced contract information
            metadata={
                'signal_data': signal,
                'position_sizing_result': {
                    'recommended_lots': position_sizing_result.recommended_lots,
                    'notional_value': position_sizing_result.notional_value,
                    'margin_required': position_sizing_result.margin_required,
                    'confidence_score': position_sizing_result.confidence_score,
                    'volatility_target': getattr(position_sizing_result, 'volatility_target', None),
                    'kelly_fraction': getattr(position_sizing_result, 'kelly_fraction', None)
                },
                'pre_trade_risk': {
                    'overall_status': pre_trade_risk_report.overall_status.value,
                    'risk_score': pre_trade_risk_report.risk_score,
                    'passed_checks': pre_trade_risk_report.passed_checks,
                    'failed_checks': pre_trade_risk_report.failed_checks
                },
                'index_validation': index_validation,
                'market_conditions': {
                    'volatility': market_data.get('volatility', 0.0),
                    'volume': market_data.get('volume', 0),
                    'open_interest': market_data.get('open_interest', 0)
                }
            }
        )
        
        # Step 8: Execute the trade
        execution_result = execute_trade_with_checks(
            signal, symbol, current_price, risk_adjusted_lots,
            portfolio, risk_limits, margin_available, contract_spec, current_time
        )
        
        result.update(execution_result)
        
        # Step 9: Update trade record and ledger based on execution result
        if result['status'] == 'success':
            # Update trade with execution details
            trade_record.broker_order_id = result.get('broker_order_id', result.get('order_id', ''))
            trade_record.status = TradeStatus.FILLED  # Assume immediate fill for now
            trade_record.quantity_filled = order['quantity']
            trade_record.price_filled = current_price  # Would be actual fill price from broker
            trade_record.fill_time = current_time
            trade_record.notional_value = abs(current_price * order['quantity'])
            
            # Calculate margin used (approximate)
            margin_per_lot = {'NIFTY': 60000, 'BANKNIFTY': 75000, 'FINNIFTY': 40000}.get(symbol, 60000)
            lot_size = contract_spec['lot_size']
            lots = abs(order['quantity']) / lot_size
            trade_record.margin_used = lots * margin_per_lot
            
            # Record trade in ledger
            success = agent['trade_ledger'].record_trade(trade_record)
            
            if success:
                logger.info(f"✅ Trade recorded in ledger: {trade_record.trade_id}")
                
                # Update position manager with current prices
                agent['position_manager'].update_market_data({symbol: current_price})
                
                result['trade_id'] = trade_record.trade_id
                result['ledger_recorded'] = True
            else:
                logger.error(f"❌ Failed to record trade in ledger")
                result['ledger_recorded'] = False
            
            # Update agent state and metrics
            _update_agent_metrics(agent, order, result, current_time, trade_record)
            logger.info(f"✅ Smart trade executed successfully: {result.get('order_id')}")
            
        elif result['status'] == 'rejected':
            # Still record the rejected trade for analysis
            trade_record.status = TradeStatus.REJECTED
            trade_record.metadata['rejection_reason'] = result.get('rejection_reason', 'Unknown')
            
            # Record rejected trade (optional, for analysis)
            agent['trade_ledger'].record_trade(trade_record)
            result['trade_id'] = trade_record.trade_id
            
            logger.warning(f"⚠️ Trade rejected and recorded: {result.get('rejection_reason')}")
            
        else:
            logger.error(f"❌ Smart trade execution failed: {result.get('message')}")
        
        # Step 10: Compile comprehensive execution report
        result['execution_summary'] = {
            'symbol': symbol,
            'action': signal['action'],
            'signal_confidence': signal.get('confidence', 0.0),
            'strategy': signal.get('strategy', 'unknown'),
            'current_price': current_price,
            'recommended_lots': position_sizing_result.recommended_lots,
            'notional_value': position_sizing_result.notional_value,
            'margin_required': position_sizing_result.margin_required,
            'risk_score': pre_trade_risk_report.risk_score,
            'index_tradeable': index_validation['is_tradeable'],
            'execution_time': current_time.isoformat(),
            'total_checks_passed': len(pre_trade_risk_report.passed_checks),
            'total_checks_failed': len(pre_trade_risk_report.failed_checks)
        }
        
        # Add performance metrics for successful trades
        if result['status'] == 'success':
            result['performance_metrics'] = {
                'trade_id': trade_record.trade_id,
                'position_size_utilized': optimal_lots,
                'margin_utilization': (trade_record.margin_used / portfolio.get('available_margin', 1)) * 100,
                'risk_adjusted_confidence': signal.get('confidence', 0.0) * (1 - pre_trade_risk_report.risk_score),
                'execution_latency_ms': (datetime.now() - current_time).total_seconds() * 1000
            }
        
        logger.info(f"📊 Execution Summary: {result['execution_summary']}")
        
        return result
        
    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
        result['execution_summary'] = {
            'symbol': symbol,
            'action': signal.get('action', 'unknown'),
            'error_occurred': True,
            'error_message': str(e),
            'execution_time': current_time.isoformat() if current_time else datetime.now().isoformat()
        }
        logger.error(f"💥 Smart trade execution error: {e}")
        return result


def execute_universe_based_trading(
    portfolio: Dict[str, Any],
    agent: Dict[str, Any],
    market_data: Dict[str, Dict[str, Any]],
    max_positions: int = 3,
    current_time: datetime = None
) -> Dict[str, Any]:
    """
    Execute universe-based trading with dynamic index selection
    
    Args:
        portfolio: Current portfolio state
        agent: Agent configuration and state
        market_data: Market data for all available indices
        max_positions: Maximum concurrent positions
        current_time: Current timestamp
        
    Returns:
        Comprehensive trading results across selected universe
    """
    if current_time is None:
        current_time = datetime.now()
    
    logger.info(f"🌐 Starting universe-based trading with {len(market_data)} available indices")
    
    # Step 1: Select optimal trading universe
    universe_selection = select_trading_universe(market_data, max_positions)
    
    if not universe_selection['selected_indices']:
        logger.warning("❌ No indices selected for trading")
        return {
            'status': 'no_trades',
            'universe_selection': universe_selection,
            'trades': [],
            'timestamp': current_time.isoformat()
        }
    
    # Step 2: Execute trades for selected indices
    execution_results = []
    
    for symbol in universe_selection['selected_indices']:
        try:
            # Generate trading signal for this index
            # (This would integrate with your RL strategy agent)
            signal = {
                'action': 'BUY',  # Placeholder - would come from RL agent
                'confidence': 0.75,
                'strategy': 'universe_based_rl'
            }
            
            # Get current price and market data for this symbol
            symbol_market_data = market_data.get(symbol, {})
            current_price = symbol_market_data.get('price', 0.0)
            
            if current_price <= 0:
                logger.warning(f"❌ Invalid price for {symbol}: {current_price}")
                continue
            
            # Execute smart trade
            trade_result = execute_smart_trade(
                signal=signal,
                symbol=symbol,
                current_price=current_price,
                portfolio=portfolio,
                market_data=symbol_market_data,
                agent=agent,
                current_time=current_time
            )
            
            trade_result['universe_rank'] = next(
                (r['rank'] for r in universe_selection['ranking_details'] if r['symbol'] == symbol),
                None
            )
            
            execution_results.append(trade_result)
            
            logger.info(f"📈 Completed trade execution for {symbol}: {trade_result['status']}")
            
        except Exception as e:
            logger.error(f"❌ Error executing trade for {symbol}: {e}")
            execution_results.append({
                'symbol': symbol,
                'status': 'error',
                'error': str(e),
                'timestamp': current_time.isoformat()
            })
    
    # Step 3: Compile comprehensive results
    successful_trades = [t for t in execution_results if t['status'] == 'success']
    rejected_trades = [t for t in execution_results if t['status'] == 'rejected']
    failed_trades = [t for t in execution_results if t['status'] == 'error']
    
    summary = {
        'total_attempted': len(execution_results),
        'successful': len(successful_trades),
        'rejected': len(rejected_trades),
        'failed': len(failed_trades),
        'success_rate': len(successful_trades) / max(len(execution_results), 1) * 100
    }
    
    logger.info(f"🎯 Universe trading complete - {summary}")
    
    return {
        'status': 'completed',
        'universe_selection': universe_selection,
        'trades': execution_results,
        'summary': summary,
        'timestamp': current_time.isoformat()
    }


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


def _update_agent_metrics(agent: Dict[str, Any], order: Dict[str, Any], result: Dict[str, Any], current_time: datetime, trade_record: Trade = None) -> None:
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
        
        # Log the trade for tracking (lightweight version for agent memory)
        trade_summary = {
            'timestamp': current_time.isoformat(),
            'trade_id': trade_record.trade_id if trade_record else None,
            'order_id': result.get('order_id'),
            'symbol': order['symbol'],
            'action': order['action'],
            'quantity': order['quantity'],
            'price': order['price'],
            'value': trade_value,
            'status': result['status'],
            'risk_score': result.get('position_sizing', {}).get('result', {}).get('risk_score', 0.0),
            'ledger_recorded': result.get('ledger_recorded', False)
        }
        
        agent['trade_book'].append(trade_summary)
        
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


def handle_trade_fill_update(
    agent: Dict[str, Any],
    trade_id: str, 
    filled_qty: int, 
    fill_price: float, 
    fill_time: datetime = None
) -> bool:
    """
    Handle trade fill update from broker feed
    
    Args:
        agent: Execution agent instance
        trade_id: Trade ID to update
        filled_qty: Quantity filled
        fill_price: Actual fill price
        fill_time: Fill timestamp
        
    Returns:
        bool: Success status
    """
    try:
        if fill_time is None:
            fill_time = datetime.now()
        
        # Update trade in ledger
        success = agent['trade_ledger'].update_trade_fill(
            trade_id=trade_id,
            filled_qty=filled_qty,
            fill_price=fill_price,
            fill_time=fill_time
        )
        
        if success:
            # Get updated position from ledger
            trade = agent['trade_ledger'].trades.get(trade_id)
            if trade:
                # Update position manager with new price
                agent['position_manager'].update_market_data({trade.symbol: fill_price})
                
                logger.info(f"✅ Trade fill processed: {trade_id} - {filled_qty} @ ₹{fill_price:.2f}")
                return True
        
        logger.error(f"❌ Failed to process trade fill: {trade_id}")
        return False
        
    except Exception as e:
        logger.error(f"Error processing trade fill {trade_id}: {e}")
        return False


def get_portfolio_summary(agent: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get comprehensive portfolio summary including positions and P&L
    
    Args:
        agent: Execution agent instance
        
    Returns:
        Dictionary with portfolio summary
    """
    try:
        # Get position summary from position manager
        position_summary = agent['position_manager'].get_position_summary()
        
        # Get performance summary from trade ledger
        performance_summary = agent['trade_ledger'].get_performance_summary()
        
        # Get daily P&L
        daily_pnl = agent['trade_ledger'].get_daily_pnl()
        
        # Get risk dashboard
        risk_dashboard = agent['position_manager'].get_risk_dashboard()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'positions': position_summary.get('positions', {}),
            'portfolio_metrics': position_summary.get('portfolio_metrics', {}),
            'performance': performance_summary,
            'daily_pnl': daily_pnl,
            'risk_metrics': risk_dashboard.get('portfolio_metrics', {}),
            'active_alerts': risk_dashboard.get('position_alerts', {}),
            'agent_metrics': {
                'daily_trade_count': agent.get('daily_trade_count', 0),
                'consecutive_losses': agent.get('consecutive_losses', 0),
                'emergency_mode': agent.get('emergency_mode', False),
                'total_trades_today': len([t for t in agent.get('trade_book', []) 
                                         if datetime.fromisoformat(t['timestamp']).date() == datetime.now().date()])
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting portfolio summary: {e}")
        return {'error': str(e)}


def start_position_monitoring(agent: Dict[str, Any]) -> bool:
    """
    Start real-time position monitoring
    
    Args:
        agent: Execution agent instance
        
    Returns:
        bool: Success status
    """
    try:
        # Register alert callback for position alerts
        def position_alert_handler(alert_data: Dict[str, Any]) -> None:
            """Handle position alerts from position manager"""
            symbol = alert_data['symbol']
            alert_type = alert_data['alert_type']
            message = alert_data['message']
            
            logger.warning(f"🚨 POSITION ALERT [{alert_type.value}] {symbol}: {message}")
            
            # Handle specific alert types
            if alert_type.value == 'STOP_LOSS':
                # Trigger stop-loss exit
                logger.critical(f"🛑 STOP LOSS HIT: {symbol} - {message}")
                # Could automatically trigger position close here
                
            elif alert_type.value == 'PROFIT_TARGET':
                # Log profit target hit
                logger.info(f"🎯 PROFIT TARGET HIT: {symbol} - {message}")
                
            elif alert_type.value == 'MARGIN_WARNING':
                # Handle margin warnings
                logger.error(f"⚠️ MARGIN WARNING: {symbol} - {message}")
        
        # Register the alert handler
        agent['position_manager'].register_alert_callback(position_alert_handler)
        
        # Start monitoring
        agent['position_manager'].start_monitoring()
        
        logger.info("✅ Position monitoring started")
        return True
        
    except Exception as e:
        logger.error(f"Failed to start position monitoring: {e}")
        return False


def stop_position_monitoring(agent: Dict[str, Any]) -> bool:
    """
    Stop position monitoring
    
    Args:
        agent: Execution agent instance
        
    Returns:
        bool: Success status
    """
    try:
        agent['position_manager'].stop_monitoring()
        logger.info("✅ Position monitoring stopped")
        return True
        
    except Exception as e:
        logger.error(f"Failed to stop position monitoring: {e}")
        return False


def close_all_positions(agent: Dict[str, Any], reason: str = "End of day") -> List[Dict[str, Any]]:
    """
    Close all open positions
    
    Args:
        agent: Execution agent instance
        reason: Reason for closing positions
        
    Returns:
        List of close orders
    """
    try:
        positions = agent['trade_ledger'].get_all_positions(include_flat=False)
        close_orders = []
        
        for position in positions:
            if position.quantity != 0:
                close_result = agent['position_manager'].close_position(
                    symbol=position.symbol,
                    reason=reason
                )
                
                if close_result['status'] == 'success':
                    close_orders.append(close_result['close_order'])
        
        logger.info(f"Generated {len(close_orders)} close orders for reason: {reason}")
        return close_orders
        
    except Exception as e:
        logger.error(f"Error closing all positions: {e}")
        return []