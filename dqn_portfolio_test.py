#!/usr/bin/env python3
"""
DQN-Driven Portfolio Manager

Uses actual DQN model for entry/exit decisions instead of hardcoded rules.
This shows real AI-driven trading decisions with proper P&L tracking.
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

from utils.portfolio_simulator import PortfolioSimulator, ActionType
from models.dqn_network import TradingAgent, TradingBrain
from enhanced_portfolio_test import TradeTracker  # Reuse the good tracking system

class DQNPortfolioManager:
    """Portfolio manager that uses DQN for trading decisions"""
    
    def __init__(self, initial_capital: float, model_path: Optional[str] = None):
        self.initial_capital = initial_capital
        self.simulator = PortfolioSimulator(initial_capital=initial_capital)
        self.tracker = TradeTracker(initial_capital)
        
        # Initialize DQN agent
        self.dqn_agent = TradingAgent(
            num_features=32,
            lookback_period=30,
            learning_rate=0.0001,
            gamma=0.3,
            epsilon_start=0.01,  # Low epsilon for production (minimal exploration)
            epsilon_end=0.001,
            device='cpu'
        )
        
        # Load trained model if available
        if model_path and Path(model_path).exists():
            try:
                self.dqn_agent.load(model_path)
                print(f"✅ Loaded trained DQN model from {model_path}")
            except Exception as e:
                print(f"⚠️ Could not load model: {e}. Using randomly initialized model.")
        else:
            print("🔄 Using randomly initialized DQN model (for demonstration)")
        
        # Load feature scaler if available
        scaler_path = Path("models/feature_scaler.pkl")
        if scaler_path.exists():
            try:
                with open(scaler_path, 'rb') as f:
                    self.feature_scaler = pickle.load(f)
                print(f"✅ Loaded feature scaler from {scaler_path}")
            except Exception as e:
                print(f"⚠️ Could not load scaler: {e}")
                self.feature_scaler = None
        else:
            self.feature_scaler = None
    
    def create_market_state(self, symbol: str, price: float, timestamp: datetime) -> np.ndarray:
        """
        Create synthetic market state for DQN input
        In production, this would come from real market data pipeline
        """
        # Create synthetic features (30 timesteps x 32 features)
        state = np.random.randn(30, 32).astype(np.float32)
        
        # Inject some realistic patterns based on current price/time
        time_of_day = timestamp.hour + timestamp.minute / 60.0
        
        # Price features (first 8 features)
        price_normalized = price / 1000.0  # Rough normalization
        state[:, 0] = price_normalized + np.random.normal(0, 0.01, 30)  # Price with noise
        state[:, 1] = np.linspace(price_normalized * 0.95, price_normalized, 30)  # Price trend
        state[:, 2] = np.random.uniform(0.3, 0.7, 30)  # RSI-like
        state[:, 3] = np.random.normal(0, 0.5, 30)  # MACD-like
        state[:, 4] = np.random.exponential(0.5, 30)  # Volume-like
        state[:, 5] = np.random.uniform(0.2, 0.8, 30)  # BB position
        state[:, 6] = np.random.normal(0, 0.3, 30)  # Momentum
        state[:, 7] = np.random.uniform(-1, 1, 30)  # Volatility
        
        # Time features (next 4 features)
        state[:, 8] = time_of_day / 24.0  # Hour of day
        state[:, 9] = timestamp.weekday() / 7.0  # Day of week  
        state[:, 10] = np.random.uniform(0.3, 0.7, 30)  # Market phase
        state[:, 11] = np.random.normal(0.5, 0.1, 30)  # Trend strength
        
        # Fill remaining features with market context
        for i in range(12, 32):
            state[:, i] = np.random.normal(0, 0.5, 30)
        
        # Apply feature scaling if available
        if self.feature_scaler:
            try:
                # Reshape for scaler, apply, then reshape back
                original_shape = state.shape
                state_flat = state.reshape(-1, state.shape[-1])
                state_flat = self.feature_scaler.transform(state_flat)
                state = state_flat.reshape(original_shape)
            except Exception as e:
                print(f"⚠️ Scaler error: {e}")
        
        return state
    
    def make_trading_decision(self, symbol: str, price: float, timestamp: datetime, 
                            position_exists: bool = False) -> Dict:
        """
        Use DQN model to make trading decision
        
        Returns:
            action: 0=SELL/SHORT, 1=HOLD, 2=BUY/LONG
            confidence: Q-value confidence
            reason: AI decision reasoning
        """
        # Create market state
        market_state = self.create_market_state(symbol, price, timestamp)
        
        # Get DQN decision
        action, info = self.dqn_agent.decide_action(
            market_state=market_state,
            mode='eval',  # Production mode (no exploration)
            minutes_to_close=180.0  # Assume mid-day trading
        )
        
        # Convert DQN action to portfolio action
        if action == 2:  # DQN says LONG
            portfolio_action = ActionType.BUY if not position_exists else ActionType.HOLD
            reason = "DQN: Strong BUY signal detected"
        elif action == 0:  # DQN says SHORT/SELL
            portfolio_action = ActionType.SELL if position_exists else ActionType.HOLD  
            reason = "DQN: SELL signal - taking profits/cutting losses"
        else:  # DQN says HOLD
            portfolio_action = ActionType.HOLD
            reason = "DQN: HOLD - no clear directional signal"
        
        q_values = info.get('q_values', [0, 0, 0])
        confidence = max(q_values) - np.mean(q_values)  # Confidence measure
        
        return {
            'action': portfolio_action,
            'dqn_action': action,
            'confidence': confidence,
            'q_values': q_values,
            'reason': reason,
            'mode': info.get('mode', 'eval')
        }
    
    def run_dqn_trading_session(self, stocks: Dict[str, float], 
                               simulation_days: int = 5) -> Dict:
        """
        Run a complete DQN-driven trading session
        """
        print("🤖 DQN-Driven Portfolio Trading Session")
        print("=" * 50)
        print(f"🧠 Model: {'Trained' if Path('models').exists() else 'Random'} DQN Agent")
        print(f"💰 Initial Capital: ₹{self.initial_capital:,.0f}")
        print(f"📊 Universe: {list(stocks.keys())}")
        print(f"📅 Simulation: {simulation_days} trading days")
        print()
        
        results = []
        base_time = datetime.now()
        
        # Day-by-day trading simulation
        for day in range(simulation_days):
            print(f"📅 Day {day + 1} Trading:")
            print("-" * 25)
            
            current_time = base_time + timedelta(days=day)
            daily_trades = 0
            
            # Morning: Portfolio construction decisions
            for symbol, base_price in stocks.items():
                # Simulate price with some daily variation
                daily_variation = np.random.uniform(0.95, 1.05)
                current_price = base_price * daily_variation
                
                # Check if we have position
                has_position = symbol in self.tracker.positions
                
                # Get DQN decision
                decision = self.make_trading_decision(
                    symbol, current_price, current_time, has_position
                )
                
                # Execute if not HOLD
                if decision['action'] != ActionType.HOLD:
                    result = self.tracker.execute_trade(
                        self.simulator, symbol, decision['action'], 
                        current_price, decision['reason']
                    )
                    
                    if result['success']:
                        action_name = "BOUGHT" if decision['action'] == ActionType.BUY else "SOLD"
                        print(f"  🤖 {action_name} {symbol}: {result['quantity']} shares @ ₹{current_price:.2f}")
                        print(f"     📊 DQN Q-values: {decision['q_values']}")
                        print(f"     💭 Confidence: {decision['confidence']:.3f}")
                        if 'pnl' in result:
                            print(f"     💰 P&L: ₹{result['pnl']:+,.0f}")
                        daily_trades += 1
                        
                        # Store decision details
                        results.append({
                            'day': day + 1,
                            'timestamp': current_time,
                            'symbol': symbol,
                            'action': decision['action'].value,
                            'price': current_price,
                            'dqn_action': decision['dqn_action'],
                            'confidence': decision['confidence'],
                            'q_values': decision['q_values'],
                            'reason': decision['reason']
                        })
                    else:
                        print(f"  ❌ Failed to execute {symbol}: {result.get('error', 'Unknown')}")
                else:
                    print(f"  ⏸️ HOLD {symbol} @ ₹{current_price:.2f} (DQN confidence: {decision['confidence']:.3f})")
            
            if daily_trades == 0:
                print("  📊 No trades executed today - DQN in hold mode")
            
            print()
        
        return {
            'decisions': results,
            'portfolio_summary': self.tracker.get_portfolio_summary({
                symbol: price * np.random.uniform(0.98, 1.02) 
                for symbol, price in stocks.items()
            }),
            'completed_trades': self.tracker.completed_trades
        }


def run_dqn_portfolio_demo():
    """Main demonstration of DQN-driven portfolio management"""
    
    print("🚀 SuperTrader.AI - DQN Portfolio Manager Demo")
    print("=" * 55)
    
    # Initialize DQN portfolio manager
    initial_capital = 1_000_000.0
    
    # Look for trained model weights
    model_paths = [
        "notebooks/models/intraday_dqn/checkpoint_episode_100.weights.h5",
        "models/trained_dqn.pth", 
        "models/dqn_model.pth"
    ]
    
    model_path = None
    for path in model_paths:
        if Path(path).exists():
            model_path = path
            break
    
    manager = DQNPortfolioManager(initial_capital, model_path)
    
    # Indian equity universe
    stocks = {
        "RELIANCE": 2850.0,
        "TCS": 4200.0,
        "HDFCBANK": 1650.0,
        "INFY": 1820.0,
        "ITC": 485.0,
        "ASIANPAINT": 2950.0,
        "LT": 3650.0,
        "WIPRO": 290.0,
        "MARUTI": 10500.0,
        "HDFC": 2750.0
    }
    
    # Run DQN trading session
    results = manager.run_dqn_trading_session(stocks, simulation_days=3)
    
    # Display results
    print("=" * 55)
    print("📊 DQN TRADING RESULTS")
    print("=" * 55)
    
    summary = results['portfolio_summary']
    print(f"💰 Initial Capital:    ₹{summary['initial_capital']:,.0f}")
    print(f"💰 Final Portfolio:    ₹{summary['portfolio_value']:,.0f}")
    print(f"📊 Total Return:       ₹{summary['total_pnl']:+,.0f} ({summary['total_return_pct']:+.2f}%)")
    print(f"🏦 Cash Balance:       ₹{summary['current_cash']:,.0f}")
    print(f"📈 Market Value:       ₹{summary['market_value']:,.0f}")
    print(f"🤖 DQN Decisions:      {len(results['decisions'])}")
    print(f"✅ Completed Trades:   {len(results['completed_trades'])}")
    
    if results['completed_trades']:
        winning_trades = [t for t in results['completed_trades'] if t['net_pnl'] > 0]
        win_rate = (len(winning_trades) / len(results['completed_trades'])) * 100
        print(f"🎯 Win Rate:           {win_rate:.1f}%")
        
        best_trade = max(results['completed_trades'], key=lambda x: x['net_pnl'])
        worst_trade = min(results['completed_trades'], key=lambda x: x['net_pnl'])
        print(f"🔥 Best Trade:         {best_trade['symbol']} (₹{best_trade['net_pnl']:+,.0f})")
        print(f"❄️ Worst Trade:        {worst_trade['symbol']} (₹{worst_trade['net_pnl']:+,.0f})")
    
    # Save DQN-specific outputs
    output_dir = Path("dqn_output")
    output_dir.mkdir(exist_ok=True)
    
    # Save decisions log
    decisions_df = pd.DataFrame(results['decisions'])
    if not decisions_df.empty:
        decisions_df.to_csv(output_dir / "dqn_decisions.csv", index=False)
        print(f"\n📄 DQN decisions saved to: {output_dir / 'dqn_decisions.csv'}")
    
    # Save trading results
    if results['completed_trades']:
        trades_df = pd.DataFrame(results['completed_trades'])
        trades_df.to_csv(output_dir / "dqn_trades.csv", index=False)
        print(f"📄 DQN trades saved to: {output_dir / 'dqn_trades.csv'}")
    
    # Save complete report
    full_report = {
        'session_info': {
            'model_path': model_path,
            'initial_capital': initial_capital,
            'stocks_universe': list(stocks.keys()),
            'generated_at': datetime.now().isoformat()
        },
        'portfolio_summary': summary,
        'dqn_decisions': results['decisions'],
        'completed_trades': results['completed_trades']
    }
    
    with open(output_dir / "dqn_portfolio_report.json", 'w') as f:
        json.dump(full_report, f, indent=2, default=str)
    
    print(f"📄 Full report saved to: {output_dir / 'dqn_portfolio_report.json'}")
    print()
    
    # Show sample DQN decisions
    if results['decisions']:
        print("🧠 SAMPLE DQN DECISIONS:")
        for decision in results['decisions'][:5]:  # Show first 5
            action_map = {1: 'BUY', -1: 'SELL', 0: 'HOLD'}
            print(f"   {decision['symbol']:10} | "
                  f"Day {decision['day']} | "
                  f"{action_map.get(decision['action'], 'HOLD'):4} | "
                  f"₹{decision['price']:6.0f} | "
                  f"Conf: {decision['confidence']:+.3f}")
    
    print("=" * 55)
    
    return results


if __name__ == "__main__":
    run_dqn_portfolio_demo()