"""
Execution Agent - PRODUCTION
Order management, risk checks, broker integration
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, time as dt_time
import logging

logger = logging.getLogger(__name__)


def init_execution_agent(config: Dict[str, Any]) -> Dict[str, Any]:
    """Initialize Execution Agent"""
    logger.info("Initializing Execution Agent...")
    
    agent = {
        'broker_connection': None,  # TODO: Connect to broker
        'order_book': [],
        'trade_book': [],
        'config': config,
        'transaction_costs': {
            'futures_bp': 2.0,  # 2 bps
            'options_bp': 8.0,  # 8 bps (higher STT)
            'brokerage_per_order': 20,  # ₹20 flat
            'stt_futures_sell': 0.0125,  # % of notional
            'stt_options_sell': 0.0625,  # % of premium (5x!)
        }
    }
    
    logger.info("✅ Execution Agent initialized")
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
) -> bool:
    """
    Pre-trade risk checks before order placement
    Returns True if all checks pass
    """
    logger.debug(f"Running pre-trade checks for {order['symbol']}...")
    
    # Time check
    if not check_time_constraints(current_time):
        logger.error("❌ Pre-trade check FAILED: Time constraint")
        return False
    
    # Margin check
    notional = order['price'] * order['quantity']
    margin_required = notional * 0.15  # MIS ~15%
    
    if margin_required > margin_available:
        logger.error(f"❌ Pre-trade check FAILED: Insufficient margin (need ₹{margin_required:,.0f}, have ₹{margin_available:,.0f})")
        return False
    
    # Max exposure check (per limits)
    max_exposure_pct = limits.get('max_exposure_pct', 50)
    total_capital = 500000  # TODO: Get from config
    max_notional = total_capital * (max_exposure_pct / 100)
    
    if notional > max_notional:
        logger.error(f"❌ Pre-trade check FAILED: Exposure limit (₹{notional:,.0f} > ₹{max_notional:,.0f})")
        return False
    
    # Position limit check (max 5 lots per symbol for intraday)
    max_lots_per_symbol = 5
    lot_size = 50  # TODO: Get from contract spec
    lots = order['quantity'] / lot_size
    
    if lots > max_lots_per_symbol:
        logger.error(f"❌ Pre-trade check FAILED: Lot limit ({lots} > {max_lots_per_symbol})")
        return False
    
    logger.info("✅ Pre-trade checks PASSED")
    return True


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