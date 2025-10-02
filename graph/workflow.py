from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime, time as dt_time
import logging
import traceback

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== STATE SCHEMA ====================
class TradingState(TypedDict):
    """Shared state - this is passed through ALL nodes"""
    
    # Time Management (CRITICAL)
    timestamp: datetime
    current_time: str
    minutes_to_close: float
    session_phase: str  # 'opening_range' | 'morning' | 'afternoon' | 'closing'
    trading_allowed: bool
    
    # Market Data (raw from broker)
    price_data: Dict[str, Any]  # {index: {futures: df, spot: value, vix: value}}
    
    # Computed Features
    indicators: Dict[str, Any]
    oi_data: Dict[str, Any]
    basis_data: Dict[str, Any]
    options_data: Dict[str, Any]
    
    # Sentiment
    sentiment_scores: Dict[str, float]
    
    # RL Output
    rl_action: Dict[str, Any]
    
    # Execution
    orders: List[Dict[str, Any]]
    positions: Dict[str, Any]  # Current open positions from broker
    
    # Risk
    risk_status: Dict[str, Any]
    time_guard_status: Dict[str, Any]
    
    # Metrics
    metrics: Dict[str, Any]
    
    # Broker Context
    broker_ctx: Dict[str, Any]  # API keys, session tokens
    
    # Logging
    errors: List[str]
    logs: List[str]


# ==================== UTILITY FUNCTIONS ====================

def calculate_time_metrics(current_dt: datetime) -> Dict[str, Any]:
    """Calculate time-based metrics for intraday trading"""
    market_open = current_dt.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = current_dt.replace(hour=15, minute=15, second=0, microsecond=0)
    
    minutes_since_open = max(0, (current_dt - market_open).total_seconds() / 60)
    minutes_to_close = max(0, (market_close - current_dt).total_seconds() / 60)
    
    # Session phase
    if minutes_since_open < 30:
        session_phase = 'opening_range'
    elif minutes_since_open < 150:
        session_phase = 'morning'
    elif minutes_since_open < 300:
        session_phase = 'afternoon'
    else:
        session_phase = 'closing'
    
    # Trading allowed? (before 3:00 PM)
    trading_allowed = minutes_to_close >= 15
    
    return {
        'minutes_since_open': minutes_since_open,
        'minutes_to_close': minutes_to_close,
        'session_phase': session_phase,
        'trading_allowed': trading_allowed,
        'minutes_since_open_norm': min(1.0, minutes_since_open / 360),
        'minutes_to_close_norm': max(0.0, minutes_to_close / 360)
    }


# ==================== NODE 1: DATA AGENT ====================

def data_agent_node(state: TradingState) -> TradingState:
    """
    Fetch real market data from ICICI Direct broker
    Compute technical indicators, OI analysis, basis
    """
    logger.info("=" * 60)
    logger.info("📊 DATA AGENT - Fetching Market Data")
    logger.info("=" * 60)
    
    try:
        from agents.data_agent import (
            fetch_ohlcv,
            compute_indicators,
            init_data_agent
        )
        import pandas as pd
        
        # Time metrics
        time_metrics = calculate_time_metrics(state['timestamp'])
        state['minutes_to_close'] = time_metrics['minutes_to_close']
        state['session_phase'] = time_metrics['session_phase']
        state['trading_allowed'] = time_metrics['trading_allowed']
        
        logger.info(f"Session: {state['session_phase']} | Time to close: {state['minutes_to_close']:.1f} mins")
        
        # Broker API keys
        api_keys = state['broker_ctx']
        
        # Define indices to fetch (Phase 1: NIFTY only)
        indices = ['NIFTY']  # Start with one, expand to BANKNIFTY, FINNIFTY later
        
        # Map to broker symbols (ICICI format)
        symbol_map = {
            'NIFTY': 'NIFTY 50',  # Adjust based on ICICI's actual symbol format
            'BANKNIFTY': 'NIFTY BANK',
            'FINNIFTY': 'NIFTY FIN SERVICE'
        }
        
        price_data = {}
        indicators = {}
        oi_data = {}
        basis_data = {}
        
        for index in indices:
            broker_symbol = symbol_map.get(index, index)
            
            try:
                # Fetch last 30 bars of 1-min data
                end_date = state['timestamp'].strftime('%Y-%m-%d')
                start_date = (state['timestamp'] - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                
                logger.info(f"Fetching {index} data from broker...")
                
                # Real OHLCV fetch from broker
                ohlcv_data = fetch_ohlcv(
                    symbols=[broker_symbol],
                    interval='1minute',
                    start=start_date,
                    end=end_date,
                    api_keys=api_keys
                )
                
                df = ohlcv_data.get(broker_symbol, pd.DataFrame())
                
                if df.empty:
                    logger.warning(f"No data for {index}, using previous cached data")
                    # Use cached data if available
                    df = state.get('price_data', {}).get(index, {}).get('futures', pd.DataFrame())
                
                if not df.empty:
                    # Take only last 30 bars for lookback
                    df = df.tail(30)
                    
                    # Compute technical indicators
                    df_with_indicators = compute_indicators(df)
                    
                    # Extract latest values for state
                    latest = df_with_indicators.iloc[-1]
                    
                    indicators[index] = {
                        'macd': latest.get('macd', 0.0),
                        'rsi': latest.get('rsi', 50.0),
                        'sma_5': latest.get('sma_5', latest['close']),
                        'sma_20': latest.get('sma_20', latest['close']),
                        'volatility_20': latest.get('volatility_20', 0.15),
                        'volume_ratio': latest.get('volume_ratio', 1.0),
                        'close': latest['close'],
                        'high': latest['high'],
                        'low': latest['low'],
                        'volume': latest.get('volume', 0)
                    }
                    
                    # Compute intraday-specific metrics
                    session_high = df_with_indicators['high'].max()
                    session_low = df_with_indicators['low'].min()
                    current_price = latest['close']
                    
                    intraday_range_pct = (
                        (current_price - session_low) / (session_high - session_low)
                        if session_high > session_low else 0.5
                    )
                    
                    indicators[index]['intraday_range_pct'] = intraday_range_pct
                    indicators[index]['session_high'] = session_high
                    indicators[index]['session_low'] = session_low
                    
                    # VWAP calculation
                    if 'volume' in df_with_indicators.columns:
                        df_with_indicators['vwap'] = (
                            (df_with_indicators['close'] * df_with_indicators['volume']).cumsum() /
                            df_with_indicators['volume'].cumsum()
                        )
                        vwap = df_with_indicators['vwap'].iloc[-1]
                        vwap_dist = (current_price - vwap) / vwap if vwap > 0 else 0.0
                        indicators[index]['vwap'] = vwap
                        indicators[index]['vwap_dist'] = vwap_dist
                    
                    # Open Interest analysis (if available in data)
                    if 'oi' in df_with_indicators.columns:
                        oi_current = df_with_indicators['oi'].iloc[-1]
                        oi_prev = df_with_indicators['oi'].iloc[-2] if len(df_with_indicators) > 1 else oi_current
                        oi_5bars_ago = df_with_indicators['oi'].iloc[-6] if len(df_with_indicators) > 5 else oi_current
                        
                        oi_change_1bar = oi_current - oi_prev
                        oi_change_5bar = oi_current - oi_5bars_ago
                        oi_momentum = (oi_change_5bar / oi_5bars_ago * 100) if oi_5bars_ago > 0 else 0.0
                        
                        oi_data[index] = {
                            'oi_current': oi_current,
                            'oi_change_1bar': oi_change_1bar,
                            'oi_change_5bar': oi_change_5bar,
                            'oi_momentum': oi_momentum
                        }
                    else:
                        oi_data[index] = {
                            'oi_current': 0,
                            'oi_change_1bar': 0,
                            'oi_change_5bar': 0,
                            'oi_momentum': 0.0
                        }
                    
                    # Spot-Futures basis (for now, assume futures = spot, refine later)
                    # TODO: Fetch actual spot index value separately
                    spot_price = current_price  # Placeholder
                    futures_price = current_price
                    basis = futures_price - spot_price
                    basis_pct = (basis / spot_price * 100) if spot_price > 0 else 0.0
                    
                    basis_data[index] = {
                        'spot': spot_price,
                        'futures': futures_price,
                        'basis': basis,
                        'basis_pct': basis_pct
                    }
                    
                    # Store raw dataframe
                    price_data[index] = {
                        'futures': df_with_indicators,
                        'spot': spot_price,
                        'vix': None  # TODO: Fetch India VIX separately
                    }
                    
                    logger.info(f"✅ {index} | Price: {current_price:.2f} | RSI: {indicators[index]['rsi']:.1f} | MACD: {indicators[index]['macd']:.2f}")
                
                else:
                    logger.error(f"❌ No data available for {index}")
                    state['errors'].append(f"DataAgent: No data for {index}")
                    
            except Exception as e:
                logger.error(f"❌ Error processing {index}: {str(e)}")
                state['errors'].append(f"DataAgent: {index} error - {str(e)}")
        
        # Update state
        state['price_data'] = price_data
        state['indicators'] = indicators
        state['oi_data'] = oi_data
        state['basis_data'] = basis_data
        state['options_data'] = {}  # Placeholder for Day 1
        
        state['logs'].append(f"DataAgent: Processed {len(price_data)} indices")
        logger.info(f"✅ Data Agent Complete - {len(price_data)} indices loaded")
        
    except Exception as e:
        error_msg = f"DataAgent CRITICAL ERROR: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        state['errors'].append(error_msg)
    
    return state


# ==================== NODE 2: NEWS AGENT ====================

def news_agent_node(state: TradingState) -> TradingState:
    """
    Sentiment analysis - Day 1 returns zeros
    Production: Fetch from Twitter/Bloomberg APIs
    """
    logger.info("📰 NEWS AGENT - Analyzing Sentiment")
    
    try:
        # Day 1 MVP: Return neutral sentiment
        state['sentiment_scores'] = {
            'market_sentiment_5min': 0.0,
            'sentiment_momentum_15min': 0.0,
            'breaking_news_flag': False,
            'high_impact_news': None
        }
        
        state['logs'].append("NewsAgent: Sentiment neutral (MVP)")
        logger.info("✅ News Agent Complete - Neutral sentiment")
        
    except Exception as e:
        error_msg = f"NewsAgent ERROR: {str(e)}"
        logger.error(error_msg)
        state['errors'].append(error_msg)
        # Safe default
        state['sentiment_scores'] = {
            'market_sentiment_5min': 0.0,
            'sentiment_momentum_15min': 0.0,
            'breaking_news_flag': False,
            'high_impact_news': None
        }
    
    return state


# ==================== NODE 3: RL AGENT ====================

def rl_agent_node(state: TradingState) -> TradingState:
    """
    RL Strategy Agent - Make trading decision
    Day 1: Simple rule-based logic (momentum + RSI)
    Production: DQN/PPO model inference
    """
    logger.info("🤖 RL AGENT - Computing Actions")
    
    try:
        from agents.rl_strategy_agent import (
            build_state_representation,
            sample_action,
            compute_position_size
        )
        
        # Check if trading is allowed
        if not state['trading_allowed']:
            logger.warning("⚠️ Trading NOT allowed - too close to market close")
            state['rl_action'] = {}
            state['logs'].append("RLAgent: Trading blocked - near close")
            return state
        
        rl_actions = {}
        
        for index in state['indicators'].keys():
            try:
                # Build state vector
                state_vector = build_state_representation(
                    price_feats=state['price_data'].get(index, {}),
                    tech_feats=state['indicators'].get(index, {}),
                    senti_feats=state['sentiment_scores'],
                    basis=state['basis_data'].get(index, {}),
                    oi=state['oi_data'].get(index, {}),
                    vix=state['price_data'].get(index, {}).get('vix'),
                    time_feats={
                        'minutes_to_close': state['minutes_to_close'],
                        'session_phase': state['session_phase']
                    },
                    options_feats=state['options_data'].get(index, {})
                )
                
                # Sample action
                action_dict = sample_action(
                    state=state_vector,
                    mode='eval',
                    time_remaining=state['minutes_to_close']
                )
                
                # Compute position size
                position_size = compute_position_size(
                    state=state_vector,
                    raw_action=action_dict['action'],
                    risk_params={'max_lots': 3, 'target_vol': 0.12},  # Conservative for Day 1
                    margin_available=100000,  # TODO: Get from broker
                    time_remaining=state['minutes_to_close'],
                    mis_mode=True
                )
                
                rl_actions[index] = {
                    'raw_action': action_dict['action'],
                    'position_size': position_size,
                    'confidence': action_dict.get('confidence', 0.0),
                    'aggression': action_dict.get('aggression_multiplier', 1.0)
                }
                
                logger.info(f"✅ {index} | Action: {action_dict['action']:.2f} | Size: {position_size} lots")
                
            except Exception as e:
                logger.error(f"❌ Error processing RL for {index}: {str(e)}")
                state['errors'].append(f"RLAgent: {index} error - {str(e)}")
        
        state['rl_action'] = rl_actions
        state['logs'].append(f"RLAgent: Actions computed for {len(rl_actions)} indices")
        logger.info(f"✅ RL Agent Complete - {len(rl_actions)} actions")
        
    except Exception as e:
        error_msg = f"RLAgent CRITICAL ERROR: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        state['errors'].append(error_msg)
        state['rl_action'] = {}
    
    return state


# ==================== NODE 4: EXECUTION AGENT ====================

def execution_agent_node(state: TradingState) -> TradingState:
    """
    Convert RL actions into broker orders
    """
    logger.info("💼 EXECUTION AGENT - Building Orders")
    
    try:
        from agents.execution_agent import (
            build_order,
            pre_trade_checks,
            check_time_constraints
        )
        
        # Time check
        if not check_time_constraints(state['timestamp']):
            logger.warning("⚠️ Time constraints violated - no new orders")
            state['orders'] = []
            return state
        
        orders = []
        
        for index, action in state['rl_action'].items():
            position_size = action.get('position_size', 0)
            
            if position_size == 0:
                logger.info(f"{index}: No position change (size=0)")
                continue
            
            try:
                # Get current position from state
                # TODO: Fetch actual positions from broker
                current_position = state['positions'].get(index, {}).get('quantity', 0)
                
                # Lot size mapping
                lot_sizes = {
                    'NIFTY': 50,
                    'BANKNIFTY': 15,
                    'FINNIFTY': 40
                }
                lot_size = lot_sizes.get(index, 50)
                
                # Calculate delta
                target_lots = position_size
                current_lots = current_position // lot_size
                delta_lots = target_lots - current_lots
                
                if delta_lots != 0:
                    # Get current price
                    current_price = state['indicators'][index]['close']
                    
                    # Build order
                    order = build_order(
                        signal={'action': 'BUY' if delta_lots > 0 else 'SELL'},
                        price=current_price,
                        size=abs(delta_lots),
                        tif='DAY',
                        contract_spec={'symbol': index, 'lot_size': lot_size},
                        order_type='MIS'
                    )
                    
                    # Pre-trade checks
                    if pre_trade_checks(
                        order=order,
                        portfolio=state['positions'],
                        limits={'max_exposure_pct': 50},
                        margin_available=100000,
                        current_time=state['timestamp']
                    ):
                        orders.append(order)
                        logger.info(f"✅ {index} | Order: {order['action']} {order['quantity']} @ {order['price']:.2f}")
                    else:
                        logger.warning(f"⚠️ {index} | Pre-trade check FAILED")
                
            except Exception as e:
                logger.error(f"❌ Error building order for {index}: {str(e)}")
                state['errors'].append(f"ExecutionAgent: {index} error - {str(e)}")
        
        state['orders'] = orders
        state['logs'].append(f"ExecutionAgent: {len(orders)} orders created")
        logger.info(f"✅ Execution Agent Complete - {len(orders)} orders")
        
    except Exception as e:
        error_msg = f"ExecutionAgent CRITICAL ERROR: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        state['errors'].append(error_msg)
        state['orders'] = []
    
    return state


# ==================== NODE 5: RISK GUARD ====================

def risk_guard_node(state: TradingState) -> TradingState:
    """
    Risk management - validate exposure, margin, VIX
    """
    logger.info("🛡️ RISK GUARD - Checking Limits")
    
    try:
        # Calculate exposures
        total_notional = 0
        margin_required = 0
        
        for order in state['orders']:
            notional = order['price'] * order['quantity']
            total_notional += notional
            # MIS margin ~15% of notional
            margin_required += notional * 0.15
        
        # Capital limits
        total_capital = 500000  # TODO: Get from config
        exposure_pct = (total_notional / total_capital) * 100
        margin_pct = (margin_required / total_capital) * 100
        
        # VIX check (if available)
        vix_ok = True
        for index in state['price_data'].keys():
            vix = state['price_data'][index].get('vix')
            if vix and vix > 20:
                vix_ok = False
                logger.warning(f"⚠️ VIX HIGH: {vix:.1f}")
        
        # Risk status
        risk_status = {
            'exposure_pct': exposure_pct,
            'margin_pct': margin_pct,
            'total_notional': total_notional,
            'margin_required': margin_required,
            'max_exposure_ok': exposure_pct < 50,
            'margin_ok': margin_pct < 50,
            'vix_ok': vix_ok,
            'risk_breach': False
        }
        
        # Check for breaches
        if not (risk_status['max_exposure_ok'] and risk_status['margin_ok'] and risk_status['vix_ok']):
            risk_status['risk_breach'] = True
            state['orders'] = []  # CANCEL ALL ORDERS
            logger.critical("🚨 RISK BREACH - ALL ORDERS CANCELLED")
        else:
            logger.info(f"✅ Risk OK | Exposure: {exposure_pct:.1f}% | Margin: {margin_pct:.1f}%")
        
        state['risk_status'] = risk_status
        state['logs'].append(f"RiskGuard: Exposure {exposure_pct:.1f}%, Breach: {risk_status['risk_breach']}")
        
    except Exception as e:
        error_msg = f"RiskGuard ERROR: {str(e)}"
        logger.error(error_msg)
        state['errors'].append(error_msg)
        # On error, be conservative
        state['risk_status'] = {'risk_breach': True}
        state['orders'] = []
    
    return state


# ==================== NODE 6: TIME GUARD ====================

def time_guard_node(state: TradingState) -> TradingState:
    """
    TIME GUARD - The KILL SWITCH
    Force exit all positions if near market close
    """
    logger.info("⏰ TIME GUARD - Enforcing Time Limits")
    
    try:
        minutes_to_close = state['minutes_to_close']
        
        # Time thresholds
        REDUCE_AFTER = 45  # 2:30 PM
        NO_NEW_AFTER = 15  # 3:00 PM
        FORCE_EXIT_AT = 5  # 3:10 PM
        EMERGENCY_AT = 0   # 3:15 PM
        
        # Determine alert level
        if minutes_to_close < 0:
            alert_level = 'EMERGENCY'
        elif minutes_to_close < FORCE_EXIT_AT:
            alert_level = 'CRITICAL'
        elif minutes_to_close < NO_NEW_AFTER:
            alert_level = 'WARNING'
        elif minutes_to_close < REDUCE_AFTER:
            alert_level = 'CAUTION'
        else:
            alert_level = 'NORMAL'
        
        time_guard_status = {
            'can_enter_new': minutes_to_close >= NO_NEW_AFTER,
            'should_reduce': minutes_to_close < REDUCE_AFTER,
            'must_exit_all': minutes_to_close < FORCE_EXIT_AT,
            'emergency_mode': minutes_to_close < 0,
            'alert_level': alert_level,
            'minutes_to_close': minutes_to_close
        }
        
        # Log alert
        if alert_level == 'EMERGENCY':
            logger.critical("🚨🚨🚨 EMERGENCY: MARKET CLOSED - BROKER WILL AUTO-SQUARE!")
        elif alert_level == 'CRITICAL':
            logger.critical(f"🚨 CRITICAL: {minutes_to_close:.1f} mins to close - FORCE EXIT ALL!")
        elif alert_level == 'WARNING':
            logger.warning(f"⚠️ WARNING: {minutes_to_close:.1f} mins to close - No new positions!")
        elif alert_level == 'CAUTION':
            logger.warning(f"⚠️ CAUTION: {minutes_to_close:.1f} mins to close - Reduce exposure")
        
        # FORCE EXIT LOGIC
        if time_guard_status['must_exit_all']:
            logger.critical("🚨 TIME GUARD: FORCING EXIT OF ALL POSITIONS")
            
            # Cancel all entry orders
            state['orders'] = []
            
            # Create exit orders for all open positions
            exit_orders = []
            for symbol, position in state['positions'].items():
                if position.get('quantity', 0) != 0:
                    qty = position['quantity']
                    price = state['indicators'].get(symbol, {}).get('close', 0)
                    
                    exit_order = {
                        'symbol': symbol,
                        'action': 'SELL' if qty > 0 else 'BUY',
                        'quantity': abs(qty),
                        'price': price,
                        'order_type': 'MARKET',
                        'product_type': 'MIS',
                        'emergency_exit': True,
                        'reason': 'TIME_GUARD_FORCE_EXIT'
                    }
                    exit_orders.append(exit_order)
            
            state['orders'] = exit_orders
            logger.critical(f"🚨 Created {len(exit_orders)} emergency exit orders")
        
        # Block new entries after 3:00 PM
        elif not time_guard_status['can_enter_new']:
            # Keep only exit orders
            exit_only = [o for o in state['orders'] if o.get('emergency_exit', False)]
            state['orders'] = exit_only
            logger.warning(f"⚠️ TIME GUARD: Blocked new entries, {len(exit_only)} exits allowed")
        
        state['time_guard_status'] = time_guard_status
        state['logs'].append(f"TimeGuard: {alert_level} - {minutes_to_close:.1f} mins to close")
        logger.info(f"✅ Time Guard Complete - Alert: {alert_level}")
        
    except Exception as e:
        error_msg = f"TimeGuard ERROR: {str(e)}"
        logger.error(error_msg)
        state['errors'].append(error_msg)
        # On error, assume emergency
        state['time_guard_status'] = {
            'can_enter_new': False,
            'must_exit_all': True,
            'emergency_mode': True,
            'alert_level': 'EMERGENCY'
        }
    
    return state


# ==================== WORKFLOW BUILDER ====================

def create_intraday_workflow() -> Any:
    """Build and compile the workflow"""
    logger.info("🔧 Building LangGraph Workflow...")
    
    workflow = StateGraph(TradingState)
    
    # Add nodes
    workflow.add_node("data_agent", data_agent_node)
    workflow.add_node("news_agent", news_agent_node)
    workflow.add_node("rl_agent", rl_agent_node)
    workflow.add_node("execution_agent", execution_agent_node)
    workflow.add_node("risk_guard", risk_guard_node)
    workflow.add_node("time_guard", time_guard_node)
    
    # Linear flow
    workflow.set_entry_point("data_agent")
    workflow.add_edge("data_agent", "news_agent")
    workflow.add_edge("news_agent", "rl_agent")
    workflow.add_edge("rl_agent", "execution_agent")
    workflow.add_edge("execution_agent", "risk_guard")
    workflow.add_edge("risk_guard", "time_guard")
    workflow.add_edge("time_guard", END)
    
    app = workflow.compile()
    logger.info("✅ Workflow Compiled")
    
    return app


def initialize_state(broker_ctx: Dict[str, str]) -> TradingState:
    """Initialize trading state with broker context"""
    now = datetime.now()
    time_metrics = calculate_time_metrics(now)
    
    return {
        'timestamp': now,
        'current_time': now.strftime("%H:%M:%S"),
        'minutes_to_close': time_metrics['minutes_to_close'],
        'session_phase': time_metrics['session_phase'],
        'trading_allowed': time_metrics['trading_allowed'],
        'price_data': {},
        'indicators': {},
        'oi_data': {},
        'basis_data': {},
        'options_data': {},
        'sentiment_scores': {},
        'rl_action': {},
        'orders': [],
        'positions': {},  # TODO: Fetch from broker
        'risk_status': {},
        'time_guard_status': {},
        'metrics': {
            'realized_pnl': 0.0,
            'unrealized_pnl': 0.0,
            'num_trades_today': 0
        },
        'broker_ctx': broker_ctx,
        'errors': [],
        'logs': []
    }


# ==================== MAIN ====================

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    logger.info("=" * 80)
    logger.info("🚀 INTRADAY TRADING SYSTEM - DAY 1 PRODUCTION")
    logger.info("=" * 80)
    
    # Get broker credentials from environment
    broker_ctx = {
        'app_key': os.getenv('ICICI_APP_KEY'),
        'api_session_token': os.getenv('ICICI_API_SESSION_TOKEN')
    }
    
    if not broker_ctx['app_key'] or not broker_ctx['api_session_token']:
        logger.critical("❌ MISSING BROKER CREDENTIALS - Set ICICI_APP_KEY and ICICI_API_SESSION_TOKEN")
        exit(1)
    
    # Create workflow
    app = create_intraday_workflow()
    
    # Initialize state
    initial_state = initialize_state(broker_ctx)
    
    logger.info(f"\n📋 Initial State:")
    logger.info(f"  Timestamp: {initial_state['timestamp']}")
    logger.info(f"  Time to Close: {initial_state['minutes_to_close']:.1f} mins")
    logger.info(f"  Session: {initial_state['session_phase']}")
    logger.info(f"  Trading Allowed: {initial_state['trading_allowed']}\n")
    
    # Run workflow
    try:
        logger.info("🎬 Starting workflow execution...\n")
        final_state = app.invoke(initial_state)
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ WORKFLOW COMPLETED")
        logger.info("=" * 80)
        
        # Summary
        logger.info(f"\n📊 EXECUTION SUMMARY:")
        logger.info(f"  Orders Created: {len(final_state['orders'])}")
        logger.info(f"  Risk Breach: {final_state.get('risk_status', {}).get('risk_breach', False)}")
        logger.info(f"  Time Alert: {final_state.get('time_guard_status', {}).get('alert_level', 'N/A')}")
        logger.info(f"  Errors: {len(final_state['errors'])}")
        logger.info(f"  Logs: {len(final_state['logs'])}")
        
        # Show orders
        if final_state['orders']:
            logger.info(f"\n📝 ORDERS TO EXECUTE:")
            for i, order in enumerate(final_state['orders'], 1):
                logger.info(f"  {i}. {order['action']} {order['quantity']} {order['symbol']} @ {order.get('price', 'MARKET')}")
        else:
            logger.info("\n📝 No orders to execute")
        
        # Show errors
        if final_state['errors']:
            logger.error("\n❌ ERRORS ENCOUNTERED:")
            for error in final_state['errors']:
                logger.error(f"  - {error}")
        
        # Show execution log
        if final_state['logs']:
            logger.info("\n📋 EXECUTION LOG:")
            for log in final_state['logs']:
                logger.info(f"  - {log}")
        
        logger.info("\n" + "=" * 80)
        
    except Exception as e:
        logger.critical(f"\n💥 WORKFLOW FAILED: {str(e)}")
        logger.critical(traceback.format_exc())
        raise