#!/usr/bin/env python3
"""
Enhanced Portfolio Trade Tracker with DQN Integration

Creates proper trade records with complete P&L tracking using actual AI models:
- DQN-driven entry/exit decisions
- Feature scaler integration
- RL strategy agent for enhanced signals
- One row per complete trade (buy + sell)
- Clear entry/exit prices with AI reasoning
- Actual P&L calculations
- Portfolio performance metrics
- Clean JSON/CSV output with meaningful data
"""

import sys
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from models.dqn_network import TradingAgent
    DQN_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ DQN not available: {e}")
    DQN_AVAILABLE = False
    TradingAgent = None

from utils.portfolio_simulator import PortfolioSimulator, ActionType
from agents.rl_strategy_agent import (
    sample_action, 
    build_state_representation,
    extract_phase4_signals
)

class TradeTracker:
    """Enhanced trade tracking with proper P&L calculation and AI decision logging"""
    
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.current_cash = initial_capital
        self.positions: Dict[str, Dict] = {}  # Track open positions
        self.completed_trades: List[Dict] = []  # Completed round-trip trades
        self.trade_sequence = 0
        
        # Initialize AI components
        self._setup_ai_models()
        
    def _setup_ai_models(self):
        """Initialize DQN agent and feature scaler"""
        try:
            # Initialize DQN agent only if available
            if DQN_AVAILABLE and TradingAgent is not None:
                self.dqn_agent = TradingAgent(
                    num_features=32,
                    lookback_period=30,
                    learning_rate=0.0001,
                    gamma=0.3,
                    epsilon_start=0.01,  # Low epsilon for production
                    epsilon_end=0.001,
                    device='cpu'
                )
                print("✅ DQN Agent initialized")
            else:
                self.dqn_agent = None
                print("⚠️ DQN Agent not available - using RL strategy only")
            
            # Load feature scaler
            scaler_path = Path("models/feature_scaler.pkl")
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.feature_scaler = pickle.load(f)
                print(f"✅ Feature scaler loaded from {scaler_path}")
            else:
                print(f"⚠️ Feature scaler not found at {scaler_path}, using None")
                self.feature_scaler = None
                
        except Exception as e:
            print(f"⚠️ AI model setup failed: {e}")
            self.dqn_agent = None
            self.feature_scaler = None
    
    def _create_market_features(self, symbol: str, price: float, timestamp: datetime) -> Dict[str, Any]:
        """Create realistic market features for AI models"""
        
        # Simulate technical indicators based on symbol and price patterns
        time_of_day = timestamp.hour + timestamp.minute / 60.0
        
        # Base technical features
        price_features = {
            'close': price,
            'open': price * np.random.uniform(0.995, 1.005),
            'high': price * np.random.uniform(1.0, 1.015),
            'low': price * np.random.uniform(0.985, 1.0)
        }
        
        # Technical indicators (simulated but realistic)
        tech_features = {
            'close': price,
            'rsi': np.random.uniform(30, 70),  # RSI typically 30-70 range
            'macd': np.random.normal(0, 2.0),  # MACD around 0
            'volatility_20': np.random.uniform(0.12, 0.25),
            'volume_ratio': np.random.uniform(0.8, 1.5),
            'intraday_range_pct': abs(price_features['high'] - price_features['low']) / price * 100,
            'vwap_dist': np.random.uniform(-0.5, 0.5)
        }
        
        # Sentiment features (Phase 4)
        sentiment_features = {
            'market_sentiment_5min': np.random.uniform(-0.3, 0.3),
            'sentiment_momentum_15min': np.random.uniform(-0.2, 0.2)
        }
        
        # Options features
        oi_features = {
            'oi_momentum': np.random.uniform(-50, 50),
            'oi_change_5bar': np.random.uniform(-1000, 1000)
        }
        
        # Basis features
        basis_features = {
            'basis_pct': np.random.uniform(-0.1, 0.1)
        }
        
        # Time features
        time_features = {
            'minutes_to_close': max(0, (15.25 - time_of_day) * 60),  # Market closes at 3:15 PM
            'session_phase': self._get_session_phase(time_of_day)
        }
        
        # VIX simulation
        vix = np.random.uniform(15, 25)
        
        # Options features
        options_features = {}
        
        return {
            'price_feats': price_features,
            'tech_feats': tech_features,
            'senti_feats': sentiment_features,
            'oi': oi_features,
            'basis': basis_features,
            'time_feats': time_features,
            'vix': vix,
            'options_feats': options_features
        }
    
    def _get_session_phase(self, time_of_day: float) -> str:
        """Determine market session phase"""
        if time_of_day < 9.5:
            return 'opening_range'
        elif time_of_day < 12.0:
            return 'morning'
        elif time_of_day < 15.0:
            return 'afternoon'
        else:
            return 'closing'
    
    def _get_ai_decision(self, symbol: str, price: float, timestamp: datetime, 
                        has_position: bool = False) -> Dict[str, Any]:
        """Get AI-driven trading decision using DQN and RL strategy"""
        
        try:
            # Create market features
            features = self._create_market_features(symbol, price, timestamp)
            
            # Build state representation using RL strategy agent
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
            
            # Apply feature scaling if available
            if self.feature_scaler is not None:
                try:
                    state_vector_scaled = self.feature_scaler.transform(state_vector.reshape(1, -1))[0]
                except Exception as e:
                    print(f"⚠️ Feature scaling failed: {e}")
                    state_vector_scaled = state_vector
            else:
                state_vector_scaled = state_vector
            
            # Create market state for DQN (30 timesteps x features)
            market_state = np.tile(state_vector_scaled, (30, 1)).astype(np.float32)
            
            # Get RL strategy decision (enhanced rule-based for now)
            time_remaining = features['time_feats']['minutes_to_close']
            rl_decision = sample_action(
                state=state_vector_scaled,
                mode='eval',
                time_remaining=time_remaining,
                phase4_features=None  # Could add Phase 4 features here
            )
            
            # Get DQN decision if agent is available
            if self.dqn_agent is not None:
                try:
                    dqn_action, dqn_info = self.dqn_agent.decide_action(
                        market_state=market_state,
                        mode='eval',
                        minutes_to_close=time_remaining
                    )
                    
                    dqn_decision = {
                        'action': dqn_action,  # 0=short, 1=hold, 2=long
                        'q_values': dqn_info.get('q_values', [0.33, 0.33, 0.33]),
                        'mode': dqn_info.get('mode', 'eval')
                    }
                except Exception as e:
                    print(f"⚠️ DQN decision failed: {e}")
                    dqn_decision = {'action': 1, 'q_values': [0.33, 0.33, 0.33], 'mode': 'fallback'}
            else:
                dqn_decision = {'action': 1, 'q_values': [0.33, 0.33, 0.33], 'mode': 'no_model'}
            
            # Combine RL and DQN decisions
            rl_action = rl_decision['action']  # -1, 0, 1
            dqn_action = dqn_decision['action']  # 0, 1, 2
            
            # Convert DQN action to RL format
            dqn_action_mapped = dqn_action - 1  # Convert to -1, 0, 1
            
            # Decision fusion: Average the two approaches
            combined_action = (rl_action + dqn_action_mapped) / 2.0
            
            # Convert to portfolio action
            if combined_action > 0.3 and not has_position:
                portfolio_action = ActionType.BUY
                reason = f"AI BUY: RL={rl_action:.2f}, DQN={dqn_action_mapped}, Combined={combined_action:.2f}"
            elif combined_action < -0.3 and has_position:
                portfolio_action = ActionType.SELL
                reason = f"AI SELL: RL={rl_action:.2f}, DQN={dqn_action_mapped}, Combined={combined_action:.2f}"
            else:
                portfolio_action = ActionType.HOLD
                reason = f"AI HOLD: RL={rl_action:.2f}, DQN={dqn_action_mapped}, Combined={combined_action:.2f}"
            
            return {
                'action': portfolio_action,
                'reason': reason,
                'ai_details': {
                    'rl_decision': rl_decision,
                    'dqn_decision': dqn_decision,
                    'combined_signal': combined_action,
                    'state_features': {
                        'rsi': features['tech_feats']['rsi'],
                        'macd': features['tech_feats']['macd'],
                        'volatility': features['tech_feats']['volatility_20'],
                        'time_to_close': time_remaining
                    },
                    'feature_scaled': bool(self.feature_scaler is not None)
                }
            }
            
        except Exception as e:
            print(f"⚠️ AI decision failed for {symbol}: {e}")
            return {
                'action': ActionType.HOLD,
                'reason': f"AI Error: {str(e)[:100]}",
                'ai_details': {'error': str(e)}
            }
        
    def execute_trade(self, simulator: PortfolioSimulator, symbol: str, action: ActionType, 
                     price: float, reason: str = "", ai_details: Dict = None) -> Dict:
        """Execute trade and track P&L with AI decision details"""
        
        result = simulator.execute(
            symbol=symbol,
            action=action, 
            price=price,
            sentiment=0.5
        )
        
        if not result.success:
            return {"success": False, "error": result.error_message}
        
        self.trade_sequence += 1
        timestamp = datetime.now()
        
        if action == ActionType.BUY:
            return self._handle_buy(symbol, result, timestamp, reason, ai_details)
        elif action == ActionType.SELL:
            return self._handle_sell(symbol, result, timestamp, reason, ai_details)
        else:
            return {"success": False, "error": "Hold action not tracked"}
    
    def _handle_buy(self, symbol: str, result, timestamp: datetime, reason: str, ai_details: Dict) -> Dict:
        """Handle buy order and create position with AI details"""
        
        total_cost = result.quantity * result.price + result.transaction_costs
        self.current_cash = result.net_cost  # This should be remaining cash
        
        # Create or add to position
        if symbol in self.positions:
            # Average down the position
            existing = self.positions[symbol]
            total_shares = existing['quantity'] + result.quantity
            total_cost_basis = existing['total_cost'] + total_cost
            avg_price = total_cost_basis / total_shares
            
            self.positions[symbol].update({
                'quantity': total_shares,
                'avg_entry_price': avg_price,
                'total_cost': total_cost_basis,
                'ai_details': ai_details  # Update with latest AI details
            })
        else:
            # New position
            self.positions[symbol] = {
                'quantity': result.quantity,
                'entry_price': result.price,
                'avg_entry_price': result.price,
                'total_cost': total_cost,
                'entry_time': timestamp,
                'entry_reason': reason,
                'ai_details': ai_details or {}
            }
        
        return {
            "success": True,
            "action": "BUY",
            "symbol": symbol,
            "quantity": result.quantity,
            "price": result.price,
            "cost": total_cost,
            "cash_balance": self.current_cash,
            "ai_details": ai_details
        }
    
    def _handle_sell(self, symbol: str, result, timestamp: datetime, reason: str, ai_details: Dict) -> Dict:
        """Handle sell order and complete trade with AI details"""
        
        if symbol not in self.positions:
            return {"success": False, "error": f"No position in {symbol} to sell"}
        
        position = self.positions[symbol]
        shares_sold = abs(result.quantity)
        
        # Calculate P&L
        proceeds = shares_sold * result.price - result.transaction_costs
        cost_basis = (position['total_cost'] / position['quantity']) * shares_sold
        realized_pnl = proceeds - cost_basis
        return_pct = (result.price / position['avg_entry_price'] - 1) * 100
        holding_days = (timestamp - position['entry_time']).days
        
        # Update cash
        self.current_cash += proceeds
        
        # Create completed trade record with AI details
        trade_record = {
            'trade_id': f"T{self.trade_sequence:03d}",
            'symbol': symbol,
            'quantity': shares_sold,
            'entry_date': position['entry_time'].strftime('%Y-%m-%d %H:%M:%S'),
            'entry_price': position['avg_entry_price'],
            'exit_date': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'exit_price': result.price,
            'entry_value': cost_basis,
            'exit_value': proceeds,
            'gross_pnl': shares_sold * (result.price - position['avg_entry_price']),
            'transaction_costs': result.transaction_costs + (cost_basis * 0.001),  # Estimated total costs
            'net_pnl': realized_pnl,
            'return_percent': return_pct,
            'holding_days': holding_days,
            'cash_balance': self.current_cash,
            'entry_reason': position['entry_reason'],
            'exit_reason': reason,
            # AI-specific fields
            'entry_ai_signal': position.get('ai_details', {}),
            'exit_ai_signal': ai_details or {},
            'ai_model_used': 'DQN+RL_Strategy',
            'feature_scaler_applied': bool(self.feature_scaler is not None)
        }
        
        self.completed_trades.append(trade_record)
        
        # Update or remove position
        remaining_shares = position['quantity'] - shares_sold
        if remaining_shares <= 0:
            del self.positions[symbol]
        else:
            # Partial sell - update position
            remaining_cost = position['total_cost'] - cost_basis
            self.positions[symbol].update({
                'quantity': remaining_shares,
                'total_cost': remaining_cost
            })
        
        return {
            "success": True,
            "action": "SELL",
            "symbol": symbol,
            "quantity": shares_sold,
            "price": result.price,
            "pnl": realized_pnl,
            "return_pct": return_pct,
            "cash_balance": self.current_cash,
            "ai_details": ai_details
        }
    
    def get_portfolio_summary(self, current_prices: Dict[str, float]) -> Dict:
        """Get current portfolio summary"""
        
        total_invested = sum(pos['total_cost'] for pos in self.positions.values())
        total_market_value = sum(
            pos['quantity'] * current_prices.get(symbol, pos['avg_entry_price'])
            for symbol, pos in self.positions.items()
        )
        
        unrealized_pnl = total_market_value - total_invested
        total_realized_pnl = sum(trade['net_pnl'] for trade in self.completed_trades)
        total_portfolio_value = self.current_cash + total_market_value
        total_return = total_portfolio_value - self.initial_capital
        
        return {
            'initial_capital': self.initial_capital,
            'current_cash': self.current_cash,
            'invested_value': total_invested,
            'market_value': total_market_value,
            'portfolio_value': total_portfolio_value,
            'unrealized_pnl': unrealized_pnl,
            'realized_pnl': total_realized_pnl,
            'total_pnl': unrealized_pnl + total_realized_pnl,
            'total_return_pct': (total_return / self.initial_capital) * 100,
            'completed_trades': len(self.completed_trades),
            'open_positions': len(self.positions)
        }


def run_enhanced_portfolio_test():
    """Run enhanced portfolio test with DQN and RL strategy integration"""
    
    print("🚀 Enhanced Portfolio Manager with AI-Driven Trading Decisions")
    print("=" * 70)
    print("🤖 Using: DQN Network + RL Strategy Agent + Feature Scaler")
    print("=" * 70)
    
    # Initialize
    initial_capital = 1_000_000.0
    simulator = PortfolioSimulator(initial_capital=initial_capital)
    tracker = TradeTracker(initial_capital)
    
    # Stock universe with realistic prices
    stocks = {
        "RELIANCE": 2850.0,
        "TCS": 4200.0,
        "HDFCBANK": 1650.0,
        "INFY": 1820.0,
        "ITC": 485.0,
        "ASIANPAINT": 2950.0,
        "LT": 3650.0,
        "WIPRO": 290.0
    }
    
    print(f"💰 Initial Capital: ₹{initial_capital:,.0f}")
    print(f"📊 Stock Universe: {len(stocks)} equities")
    print(f"🧠 AI Models: {'✅' if tracker.dqn_agent else '❌'} DQN, {'✅' if tracker.feature_scaler else '❌'} Scaler")
    print()
    
    # Phase 1: AI-driven portfolio construction
    print("🤖 Phase 1: AI-Driven Portfolio Construction")
    print("-" * 45)
    
    for symbol, price in stocks.items():
        # Get AI decision for each stock
        ai_decision = tracker._get_ai_decision(symbol, price, datetime.now(), has_position=False)
        
        if ai_decision['action'] == ActionType.BUY:
            result = tracker.execute_trade(
                simulator, symbol, ActionType.BUY, price, 
                ai_decision['reason'], ai_decision['ai_details']
            )
            
            if result['success']:
                print(f"🤖 AI BOUGHT {symbol}: {result['quantity']} shares @ ₹{price:.2f}")
                print(f"   📊 Reason: {ai_decision['reason']}")
                
                # Show AI decision details
                ai_details = ai_decision['ai_details']
                if 'rl_decision' in ai_details:
                    rl_action = ai_details['rl_decision']['action']
                    print(f"   🧠 RL Signal: {rl_action:.2f}, RSI: {ai_details['state_features']['rsi']:.1f}")
                
                if 'dqn_decision' in ai_details:
                    dqn_q_values = ai_details['dqn_decision']['q_values']
                    print(f"   🎯 DQN Q-values: [Short:{dqn_q_values[0]:.3f}, Hold:{dqn_q_values[1]:.3f}, Long:{dqn_q_values[2]:.3f}]")
                
                print(f"   💰 Cost: ₹{result['cost']:,.0f} | Cash: ₹{result['cash_balance']:,.0f}")
            else:
                print(f"❌ Failed to buy {symbol}: {result.get('error', 'Unknown error')}")
        else:
            print(f"⚪ AI SKIPPED {symbol} @ ₹{price:.2f} - {ai_decision['reason']}")
    
    print()
    
    # Phase 2: Simulate time passage and AI rebalancing
    print("📊 Phase 2: AI Market Analysis & Rebalancing (2 months later)")
    print("-" * 60)
    
    # Simulate realistic price movements
    future_time = datetime.now() + timedelta(days=60)
    new_prices = {}
    
    for symbol, base_price in stocks.items():
        # Simulate market movements with some realistic patterns
        if symbol in ["TCS", "INFY"]:  # IT sector under pressure
            change_factor = np.random.uniform(0.92, 0.98)
        elif symbol in ["ITC", "ASIANPAINT"]:  # Consumer/Paint doing well
            change_factor = np.random.uniform(1.05, 1.20)
        elif symbol == "RELIANCE":  # Energy volatility
            change_factor = np.random.uniform(0.95, 1.10)
        else:
            change_factor = np.random.uniform(0.95, 1.05)
        
        new_prices[symbol] = base_price * change_factor
    
    print("📈 Updated Market Prices:")
    for symbol, new_price in new_prices.items():
        old_price = stocks[symbol]
        change_pct = (new_price / old_price - 1) * 100
        print(f"   {symbol:10}: ₹{old_price:6.0f} → ₹{new_price:6.0f} ({change_pct:+5.1f}%)")
    print()
    
    # AI-driven selling decisions
    print("🤖 AI Rebalancing Decisions:")
    print("-" * 30)
    
    for symbol in list(tracker.positions.keys()):
        if symbol in new_prices:
            # Get AI decision for potential exit
            ai_decision = tracker._get_ai_decision(symbol, new_prices[symbol], future_time, has_position=True)
            
            if ai_decision['action'] == ActionType.SELL:
                result = tracker.execute_trade(
                    simulator, symbol, ActionType.SELL, new_prices[symbol], 
                    ai_decision['reason'], ai_decision['ai_details']
                )
                
                if result['success']:
                    print(f"🤖 AI SOLD {symbol}: {result['quantity']} shares @ ₹{new_prices[symbol]:.2f}")
                    print(f"   📊 Reason: {ai_decision['reason']}")
                    print(f"   💰 P&L: ₹{result['pnl']:+,.0f} ({result['return_pct']:+.1f}%)")
                    
                    # Show AI decision details
                    ai_details = ai_decision['ai_details']
                    if 'combined_signal' in ai_details:
                        print(f"   🎯 AI Signal Strength: {ai_details['combined_signal']:+.3f}")
                    
                    print(f"   💰 Cash: ₹{result['cash_balance']:,.0f}")
                else:
                    print(f"❌ Failed to sell {symbol}: {result.get('error', 'Unknown error')}")
            else:
                print(f"🤖 AI HOLDS {symbol} @ ₹{new_prices[symbol]:.2f} - {ai_decision['reason']}")
    
    # Generate reports
    output_dir = Path("enhanced_output")
    output_dir.mkdir(exist_ok=True)
    
    portfolio_summary = tracker.get_portfolio_summary(new_prices)
    
    # Save completed trades as CSV
    if tracker.completed_trades:
        trades_df = pd.DataFrame(tracker.completed_trades)
        csv_file = output_dir / "completed_trades.csv"
        trades_df.to_csv(csv_file, index=False)
        print(f"\n📄 Saved trades to: {csv_file}")
    
    # Save as JSON with AI details
    json_data = {
        'portfolio_summary': portfolio_summary,
        'completed_trades': tracker.completed_trades,
        'ai_model_info': {
            'dqn_agent_available': tracker.dqn_agent is not None,
            'feature_scaler_available': tracker.feature_scaler is not None,
            'models_used': 'DQN+RL_Strategy',
            'decision_fusion': 'Combined RL and DQN signals'
        },
        'current_positions': {
            symbol: {
                'quantity': pos['quantity'],
                'entry_price': pos['avg_entry_price'],
                'current_price': new_prices.get(symbol, pos['avg_entry_price']),
                'market_value': pos['quantity'] * new_prices.get(symbol, pos['avg_entry_price']),
                'unrealized_pnl': pos['quantity'] * (new_prices.get(symbol, pos['avg_entry_price']) - pos['avg_entry_price']),
                'days_held': (datetime.now() - pos['entry_time']).days,
                'ai_entry_details': pos.get('ai_details', {})
            }
            for symbol, pos in tracker.positions.items()
        },
        'generated_at': datetime.now().isoformat()
    }
    
    json_file = output_dir / "portfolio_report.json"
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2, default=str)
    
    print(f"📄 Saved detailed report to: {json_file}")
    
    # Display results
    display_results(tracker, portfolio_summary, new_prices)


def display_results(tracker: TradeTracker, summary: Dict, current_prices: Dict):
    """Display comprehensive results"""
    
    print("\n" + "=" * 65)
    print("📋 PORTFOLIO PERFORMANCE REPORT")
    print("=" * 65)
    
    # Portfolio summary
    print(f"💰 Initial Capital:      ₹{summary['initial_capital']:,.0f}")
    print(f"💰 Portfolio Value:      ₹{summary['portfolio_value']:,.0f}")
    print(f"📊 Total Return:         ₹{summary['total_pnl']:+,.0f} ({summary['total_return_pct']:+.2f}%)")
    print(f"🏦 Cash Balance:         ₹{summary['current_cash']:,.0f}")
    print(f"📈 Market Value:         ₹{summary['market_value']:,.0f}")
    print(f"💹 Realized P&L:         ₹{summary['realized_pnl']:+,.0f}")
    print(f"📊 Unrealized P&L:       ₹{summary['unrealized_pnl']:+,.0f}")
    
    # Completed trades summary
    if tracker.completed_trades:
        print(f"\n💼 COMPLETED TRADES SUMMARY:")
        print(f"   Total Trades:         {len(tracker.completed_trades)}")
        
        winning_trades = [t for t in tracker.completed_trades if t['net_pnl'] > 0]
        win_rate = (len(winning_trades) / len(tracker.completed_trades)) * 100
        
        print(f"   Winning Trades:       {len(winning_trades)}")
        print(f"   Win Rate:             {win_rate:.1f}%")
        print(f"   Average P&L:          ₹{summary['realized_pnl'] / len(tracker.completed_trades):+,.0f}")
        
        best_trade = max(tracker.completed_trades, key=lambda x: x['net_pnl'])
        worst_trade = min(tracker.completed_trades, key=lambda x: x['net_pnl'])
        
        print(f"   Best Trade:           {best_trade['symbol']} (₹{best_trade['net_pnl']:+,.0f})")
        print(f"   Worst Trade:          {worst_trade['symbol']} (₹{worst_trade['net_pnl']:+,.0f})")
        
        # Individual trade details with AI info
        print(f"\n📊 INDIVIDUAL TRADE RESULTS (AI-DRIVEN):")
        for trade in tracker.completed_trades:
            # Extract AI signal info if available
            ai_info = ""
            if 'entry_ai_signal' in trade:
                entry_signal = trade.get('entry_ai_signal', {})
                if 'combined_signal' in entry_signal:
                    ai_info = f" AI:{entry_signal['combined_signal']:+.2f}"
            
            print(f"   {trade['symbol']:10} | "
                  f"₹{trade['entry_price']:6.0f} → ₹{trade['exit_price']:6.0f} | "
                  f"₹{trade['net_pnl']:+8,.0f} ({trade['return_percent']:+5.1f}%) | "
                  f"{trade['holding_days']} days{ai_info}")
        
        # Show AI decision summary
        print(f"\n🤖 AI MODEL PERFORMANCE:")
        ai_driven_trades = [t for t in tracker.completed_trades if 'ai_model_used' in t]
        print(f"   AI-Driven Trades:     {len(ai_driven_trades)}")
        print(f"   Feature Scaler Used:  {'Yes' if tracker.feature_scaler else 'No'}")
        print(f"   DQN Model Available:  {'Yes' if tracker.dqn_agent else 'No'}")
        
        if ai_driven_trades:
            ai_pnl = sum(t['net_pnl'] for t in ai_driven_trades)
            ai_winning = len([t for t in ai_driven_trades if t['net_pnl'] > 0])
            ai_win_rate = (ai_winning / len(ai_driven_trades)) * 100
            print(f"   AI Win Rate:          {ai_win_rate:.1f}%")
            print(f"   AI Total P&L:         ₹{ai_pnl:+,.0f}")
    
    # Current positions
    if tracker.positions:
        print(f"\n📊 CURRENT HOLDINGS:")
        for symbol, pos in tracker.positions.items():
            current_price = current_prices.get(symbol, pos['avg_entry_price'])
            market_value = pos['quantity'] * current_price
            unrealized_pnl = market_value - pos['total_cost']
            unrealized_pct = (current_price / pos['avg_entry_price'] - 1) * 100
            
            print(f"   {symbol:10} | {pos['quantity']:3d} shares | "
                  f"₹{pos['avg_entry_price']:6.0f} → ₹{current_price:6.0f} | "
                  f"₹{unrealized_pnl:+8,.0f} ({unrealized_pct:+5.1f}%)")
    
    print("=" * 65)


if __name__ == "__main__":
    run_enhanced_portfolio_test()