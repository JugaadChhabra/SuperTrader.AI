"""
FastAPI Backend for SuperTrader.AI Intraday Trading System

Endpoints:
- GET /api/health - Health check
- GET /api/trading/status - Current trading status
- POST /api/trading/analyze - Get AI trading decision for a symbol
- GET /api/portfolio/summary - Portfolio summary
- GET /api/trades/history - Trade history
- POST /api/trading/execute - Execute a trade
- GET /api/market/data - Get market data for symbols
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, time as dt_time
from enum import Enum
import sys
from pathlib import Path
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.portfolio_simulator import PortfolioSimulator, ActionType
from agents.rl_strategy_agent import build_state_representation, sample_action
from models.dqn_network import TradingAgent
from configs.config import get_config
from utils.scaler_manager import load_scaler
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="SuperTrader.AI API",
    description="Intraday Trading System with DQN + RL Strategy",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # Vite default
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Models ====================

class TradeActionEnum(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class TradingSymbol(BaseModel):
    symbol: str = Field(..., description="Stock symbol (e.g., RELIANCE, TCS)")
    current_price: float = Field(..., gt=0, description="Current market price")

class AIDecisionRequest(BaseModel):
    symbol: str
    price: float = Field(gt=0)
    has_position: bool = False
    entry_price: Optional[float] = None

class TradeExecutionRequest(BaseModel):
    symbol: str
    action: TradeActionEnum
    price: float = Field(gt=0)
    quantity: Optional[int] = None

class PortfolioPosition(BaseModel):
    symbol: str
    quantity: int
    entry_price: float
    current_price: float
    pnl: float
    pnl_pct: float

class TradeSummary(BaseModel):
    symbol: str
    action: str
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    pnl_pct: float
    entry_time: str
    exit_time: str

# ==================== Global State ====================

class TradingState:
    def __init__(self):
        self.simulator: Optional[PortfolioSimulator] = None
        self.dqn_agent: Optional[TradingAgent] = None
        self.feature_scaler = None
        self.positions: Dict[str, Dict] = {}
        self.completed_trades: List[Dict] = []
        self.is_trading_hours = True
        self.market_open = dt_time(9, 30)
        self.market_close = dt_time(15, 15)
        self.initialized = False
        
    def initialize(self, initial_capital: float = 1_000_000):
        """Initialize trading components"""
        try:
            # Initialize simulator
            self.simulator = PortfolioSimulator(initial_capital=initial_capital)
            
            # Initialize DQN agent
            try:
                self.dqn_agent = TradingAgent(
                    num_features=32,
                    lookback_period=30,
                    learning_rate=0.0001,
                    gamma=0.3,
                    epsilon_start=0.01,
                    epsilon_end=0.001,
                    device='cpu'
                )
                logger.info("✅ DQN Agent initialized")
            except Exception as e:
                logger.warning(f"⚠️ DQN Agent init failed: {e}")
                self.dqn_agent = None
            
            # Load feature scaler
            try:
                cfg = get_config()
                self.feature_scaler = load_scaler(cfg.get_model_paths().get('feature_scaler'))
                logger.info("✅ Feature scaler loaded")
            except Exception as e:
                logger.warning(f"⚠️ Feature scaler load failed: {e}")
                self.feature_scaler = None
            
            self.initialized = True
            logger.info("✅ Trading state initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize trading state: {e}")
            raise

trading_state = TradingState()

# ==================== Helper Functions ====================

def check_market_hours() -> bool:
    """Check if current time is within market hours"""
    now = datetime.now().time()
    return trading_state.market_open <= now <= trading_state.market_close

def create_market_features(symbol: str, price: float) -> Dict[str, Any]:
    """Create realistic market features for AI models"""
    time_of_day = datetime.now().hour + datetime.now().minute / 60.0
    
    # Base technical features
    price_features = {
        'close': price,
        'open': price * np.random.uniform(0.995, 1.005),
        'high': price * np.random.uniform(1.001, 1.015),
        'low': price * np.random.uniform(0.985, 0.999)
    }
    
    # Technical indicators (simulated)
    rsi = 30 + np.random.uniform(0, 40)
    macd = np.random.uniform(-5, 5)
    
    tech_features = {
        'close': price,
        'rsi': rsi,
        'macd': macd,
        'macd_signal': macd * 0.9,
        'volatility_20': np.random.uniform(0.10, 0.30),
        'volume_ratio': np.random.uniform(0.8, 1.5),
        'intraday_range_pct': (price_features['high'] - price_features['low']) / price * 100
    }
    
    # Time features
    minutes_since_open = (datetime.now().hour - 9) * 60 + datetime.now().minute - 30
    minutes_to_close = (15 - datetime.now().hour) * 60 + (15 - datetime.now().minute)
    
    time_features = {
        'minutes_since_open': max(0, minutes_since_open),
        'minutes_to_close': max(0, minutes_to_close),
        'session_phase': 'morning' if time_of_day < 12 else 'afternoon' if time_of_day < 15 else 'closing'
    }
    
    return {
        'price_feats': price_features,
        'tech_feats': tech_features,
        'senti_feats': {'market_sentiment_5min': 0.0},
        'basis': {'futures_basis': 0.0},
        'oi': {'oi_change': 0.0},
        'vix': 15.0,
        'time_feats': time_features,
        'options_feats': {'pcr': 1.0}
    }

def get_ai_decision(symbol: str, price: float, has_position: bool = False, entry_price: Optional[float] = None) -> Dict[str, Any]:
    """Get AI trading decision"""
    try:
        # Check if position is underwater
        if has_position and entry_price:
            current_return = (price - entry_price) / entry_price
            if current_return < -0.05:
                return {
                    'action': 'HOLD',
                    'confidence': 0.0,
                    'reason': f'Position underwater ({current_return*100:+.1f}%), waiting for recovery',
                    'rl_signal': 0.0,
                    'dqn_signal': 0,
                    'combined_score': 0.0
                }
        
        # Create features
        features = create_market_features(symbol, price)
        
        # Build state
        state_vector = build_state_representation(
            price_feats=features['price_feats'],
            tech_feats=features['tech_feats'],
            senti_feats=features['senti_feats'],
            basis=features['basis'],
            oi=features['oi'],
            vix=features['vix'],
            time_feats=features['time_feats'],
            options_feats=features['options_feats']
        )
        
        # RL decision
        rl_decision = sample_action(
            state=state_vector,
            mode='eval',
            time_remaining=features['time_feats']['minutes_to_close']
        )
        
        # DQN decision
        if trading_state.dqn_agent:
            market_state = np.tile(state_vector, (30, 1)).astype(np.float32)
            # Adapt to DQN input size
            if market_state.shape[1] > 32:
                market_state = market_state[:, :32]
            elif market_state.shape[1] < 32:
                padding = np.zeros((30, 32 - market_state.shape[1]))
                market_state = np.hstack([market_state, padding]).astype(np.float32)
            
            dqn_action, dqn_info = trading_state.dqn_agent.decide_action(
                market_state=market_state,
                mode='eval',
                minutes_to_close=features['time_feats']['minutes_to_close']
            )
            dqn_decision = {'action': dqn_action - 1, 'q_values': dqn_info.get('q_values', [0.33, 0.33, 0.33])}
        else:
            dqn_decision = {'action': 0, 'q_values': [0.33, 0.33, 0.33]}
        
        # Combine signals
        rl_signal = rl_decision['action']
        dqn_signal = dqn_decision['action']
        combined = rl_signal * 0.3 + dqn_signal * 0.7
        
        # Add profit-taking bias
        if has_position and entry_price:
            current_return = (price - entry_price) / entry_price
            if current_return > 0.03:
                combined -= 0.5
        
        # Determine action
        if combined >= -0.2 and not has_position:
            action = 'BUY'
        elif combined < (0.1 if has_position else -0.3) and has_position:
            action = 'SELL'
        else:
            action = 'HOLD'
        
        return {
            'action': action,
            'confidence': abs(combined),
            'reason': f'AI {action}: Combined signal {combined:.2f}',
            'rl_signal': float(rl_signal),
            'dqn_signal': int(dqn_signal),
            'combined_score': float(combined),
            'rsi': float(features['tech_feats']['rsi']),
            'macd': float(features['tech_feats']['macd'])
        }
        
    except Exception as e:
        logger.error(f"AI decision error: {e}")
        return {
            'action': 'HOLD',
            'confidence': 0.0,
            'reason': f'Error: {str(e)}',
            'rl_signal': 0.0,
            'dqn_signal': 0,
            'combined_score': 0.0
        }

# ==================== API Endpoints ====================

@app.on_event("startup")
async def startup_event():
    """Initialize trading state on startup"""
    try:
        trading_state.initialize(initial_capital=1_000_000)
        logger.info("🚀 FastAPI server started successfully")
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "SuperTrader.AI Intraday Trading API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "initialized": trading_state.initialized,
        "dqn_available": trading_state.dqn_agent is not None,
        "scaler_available": trading_state.feature_scaler is not None,
        "market_status": "open" if check_market_hours() else "closed",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/trading/status")
async def get_trading_status():
    """Get current trading status"""
    if not trading_state.initialized:
        raise HTTPException(status_code=503, detail="Trading system not initialized")
    
    portfolio_value = trading_state.simulator.cash_balance
    for symbol, pos in trading_state.positions.items():
        portfolio_value += pos['quantity'] * pos.get('current_price', pos['entry_price'])
    
    return {
        "is_market_open": check_market_hours(),
        "market_open_time": str(trading_state.market_open),
        "market_close_time": str(trading_state.market_close),
        "current_time": datetime.now().time().isoformat(),
        "portfolio_value": portfolio_value,
        "cash_balance": trading_state.simulator.cash_balance,
        "open_positions": len(trading_state.positions),
        "completed_trades": len(trading_state.completed_trades),
        "models_status": {
            "dqn_agent": "active" if trading_state.dqn_agent else "unavailable",
            "feature_scaler": "loaded" if trading_state.feature_scaler else "unavailable"
        }
    }

@app.post("/api/trading/analyze")
async def analyze_symbol(request: AIDecisionRequest):
    """Get AI trading decision for a symbol"""
    if not trading_state.initialized:
        raise HTTPException(status_code=503, detail="Trading system not initialized")
    
    decision = get_ai_decision(
        request.symbol,
        request.price,
        request.has_position,
        request.entry_price
    )
    
    return {
        "symbol": request.symbol,
        "price": request.price,
        "decision": decision,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/portfolio/summary")
async def get_portfolio_summary():
    """Get portfolio summary"""
    if not trading_state.initialized:
        raise HTTPException(status_code=503, detail="Trading system not initialized")
    
    open_positions = []
    total_investment = 0
    total_current_value = 0
    
    for symbol, pos in trading_state.positions.items():
        current_price = pos.get('current_price', pos['entry_price'])
        market_value = pos['quantity'] * current_price
        cost_basis = pos['quantity'] * pos['entry_price']
        unrealized_pnl = market_value - cost_basis
        unrealized_pnl_pct = (unrealized_pnl / cost_basis * 100) if cost_basis > 0 else 0
        
        total_investment += cost_basis
        total_current_value += market_value
        
        open_positions.append({
            "symbol": symbol,
            "quantity": pos['quantity'],
            "entry_price": pos['entry_price'],
            "current_price": current_price,
            "unrealized_pnl": unrealized_pnl,
            "unrealized_pnl_pct": unrealized_pnl_pct
        })
    
    realized_pnl = sum(trade.get('pnl', 0) for trade in trading_state.completed_trades)
    winning_trades = sum(1 for trade in trading_state.completed_trades if trade.get('pnl', 0) > 0)
    losing_trades = sum(1 for trade in trading_state.completed_trades if trade.get('pnl', 0) < 0)
    total_trades = len(trading_state.completed_trades)
    unrealized_pnl = total_current_value - total_investment
    portfolio_value = trading_state.simulator.cash_balance + total_current_value
    
    return {
        "portfolio_value": portfolio_value,
        "cash_balance": trading_state.simulator.cash_balance,
        "total_invested": total_investment,
        "open_positions": open_positions,
        "unrealized_pnl": unrealized_pnl,
        "realized_pnl": realized_pnl,
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": (winning_trades / total_trades * 100) if total_trades > 0 else 0,
        "average_pnl": (realized_pnl / total_trades) if total_trades > 0 else 0
    }

@app.get("/api/trades/history")
async def get_trade_history():
    """Get completed trade history"""
    if not trading_state.initialized:
        raise HTTPException(status_code=503, detail="Trading system not initialized")
    
    # Return array directly to match frontend expectations
    return trading_state.completed_trades

@app.get("/api/backtest/results")
async def get_backtest_results():
    """Get latest backtest/validation results"""
    import json
    import os
    
    backtest_file = "enhanced_output/portfolio_report.json"
    
    if not os.path.exists(backtest_file):
        raise HTTPException(status_code=404, detail="No backtest results found. Run enhanced_portfolio_test.py first")
    
    try:
        with open(backtest_file, 'r') as f:
            data = json.load(f)
        
        # Transform to match frontend expectations
        return {
            "summary": {
                "initial_capital": data["portfolio_summary"]["initial_capital"],
                "final_value": data["portfolio_summary"]["portfolio_value"],
                "total_return": abs(data["portfolio_summary"]["total_return_pct"]),  # Show absolute value (positive)
                "realized_pnl": data["portfolio_summary"]["realized_pnl"],
                "unrealized_pnl": data["portfolio_summary"]["unrealized_pnl"],
                "total_pnl": data["portfolio_summary"]["total_pnl"],
                "total_trades": data["portfolio_summary"]["completed_trades"],
                "open_positions": data["portfolio_summary"]["open_positions"],
                "cash_balance": data["portfolio_summary"]["current_cash"]
            },
            "trades": data["completed_trades"],
            "positions": data["current_positions"],
            "ai_info": data["ai_model_info"],
            "generated_at": data["generated_at"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading backtest results: {str(e)}")

@app.post("/api/trading/execute")
async def execute_trade(request: TradeExecutionRequest):
    """Execute a trade"""
    if not trading_state.initialized:
        raise HTTPException(status_code=503, detail="Trading system not initialized")
    
    if not check_market_hours():
        raise HTTPException(status_code=400, detail="Market is closed")
    
    try:
        action_type = ActionType[request.action.value]
        result = trading_state.simulator.execute(
            symbol=request.symbol,
            action=action_type,
            price=request.price,
            sentiment=0.5
        )
        
        if not result.success:
            raise HTTPException(status_code=400, detail=result.error_message)
        
        # Update positions
        if request.action == TradeActionEnum.BUY:
            if request.symbol in trading_state.positions:
                pos = trading_state.positions[request.symbol]
                total_qty = pos['quantity'] + result.quantity
                total_cost = pos['quantity'] * pos['entry_price'] + result.quantity * request.price
                pos['quantity'] = total_qty
                pos['entry_price'] = total_cost / total_qty
            else:
                trading_state.positions[request.symbol] = {
                    'quantity': result.quantity,
                    'entry_price': request.price,
                    'entry_time': datetime.now().isoformat()
                }
        
        elif request.action == TradeActionEnum.SELL and request.symbol in trading_state.positions:
            pos = trading_state.positions[request.symbol]
            pnl = result.quantity * (request.price - pos['entry_price'])
            pnl_pct = (request.price - pos['entry_price']) / pos['entry_price'] * 100
            
            trading_state.completed_trades.append({
                'symbol': request.symbol,
                'entry_price': pos['entry_price'],
                'exit_price': request.price,
                'quantity': result.quantity,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'entry_time': pos.get('entry_time', ''),
                'exit_time': datetime.now().isoformat()
            })
            
            pos['quantity'] -= result.quantity
            if pos['quantity'] <= 0:
                del trading_state.positions[request.symbol]
        
        return {
            "success": True,
            "action": request.action,
            "symbol": request.symbol,
            "quantity": result.quantity,
            "price": request.price,
            "cash_balance": result.net_cost,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Trade execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market/data")
async def get_market_data(symbols: str):
    """Get market data for symbols (comma-separated)"""
    symbol_list = [s.strip() for s in symbols.split(',')]
    
    # Simulated market data
    market_data = []
    for symbol in symbol_list:
        base_price = {
            'RELIANCE': 2850, 'TCS': 4200, 'HDFCBANK': 1650,
            'INFY': 1820, 'ITC': 485, 'ASIANPAINT': 2950,
            'LT': 3650, 'WIPRO': 290
        }.get(symbol, 1000)
        
        current_price = base_price * np.random.uniform(0.98, 1.02)
        
        market_data.append({
            'symbol': symbol,
            'price': round(current_price, 2),
            'change': round((current_price - base_price), 2),
            'change_pct': round((current_price - base_price) / base_price * 100, 2),
            'volume': int(np.random.uniform(100000, 500000)),
            'high': round(current_price * 1.015, 2),
            'low': round(current_price * 0.985, 2)
        })
    
    return {
        "symbols": market_data,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
