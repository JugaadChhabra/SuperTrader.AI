#!/usr/bin/env python3
"""
Equity Portfolio Manager Test - Live Demo

This script demonstrates the agentic portfolio manager with:
- Real Indian equity stocks (RELIANCE, TCS, HDFC, etc.)
- Realistic stock prices
- Portfolio management with longer-term holdings
- Clear entry/exit prices and P&L tracking
- Performance metrics and ledger outputs

Run this to see the portfolio manager in action!
"""

import sys
import os
import time
import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import our modules
from utils.portfolio_simulator import PortfolioSimulator, ActionType
from agents.enhanced_trade_ledger import SessionLedger
from utils.logging import PerformanceLogger
from configs.config import get_config

class EquityPortfolioDemo:
    """
    Equity Portfolio Manager Demonstration
    
    Simulates a realistic portfolio management session with:
    - Indian equity stocks
    - Proper position sizing
    - Entry/exit tracking
    - P&L calculations
    - Performance analytics
    """
    
    def __init__(self, initial_capital: float = 500000.0):
        """Initialize with 5 lakhs initial capital"""
        self.initial_capital = initial_capital
        self.demo_dir = Path("demo_outputs")
        self.demo_dir.mkdir(exist_ok=True)
        
        # Indian equity stocks with realistic prices (Nov 2024)
        self.stock_universe = {
            "RELIANCE": {"price": 2850.0, "sector": "Energy", "market_cap": "Large"},
            "TCS": {"price": 4200.0, "sector": "IT", "market_cap": "Large"},
            "HDFCBANK": {"price": 1650.0, "sector": "Banking", "market_cap": "Large"},
            "INFY": {"price": 1820.0, "sector": "IT", "market_cap": "Large"},
            "ITC": {"price": 485.0, "sector": "FMCG", "market_cap": "Large"},
            "LT": {"price": 3650.0, "sector": "Infrastructure", "market_cap": "Large"},
            "WIPRO": {"price": 290.0, "sector": "IT", "market_cap": "Large"},
            "MARUTI": {"price": 11500.0, "sector": "Auto", "market_cap": "Large"},
            "BAJFINANCE": {"price": 7200.0, "sector": "NBFC", "market_cap": "Large"},
            "ASIANPAINT": {"price": 2950.0, "sector": "Paints", "market_cap": "Large"}
        }
        
        print("🚀 Equity Portfolio Manager - Live Demonstration")
        print(f"💰 Initial Capital: ₹{self.initial_capital:,.0f}")
        print(f"📊 Stock Universe: {len(self.stock_universe)} Indian equities")
        print("=" * 60)
    
    def run_portfolio_simulation(self):
        """Run complete portfolio simulation"""
        
        # Initialize components
        simulator = PortfolioSimulator(
            initial_capital=self.initial_capital,
            transaction_cost_bps=10.0,  # 0.1% total cost
            slippage_bps=2.0
        )
        
        ledger = SessionLedger(base_path=str(self.demo_dir))
        session_id = ledger.start_session(
            symbols=list(self.stock_universe.keys()),
            strategy="Equity Portfolio Manager",
            initial_capital=self.initial_capital
        )
        
        perf_logger = PerformanceLogger("equity_demo", str(self.demo_dir))
        
        print(f"📝 Session Started: {session_id}")
        print("💼 Building Portfolio...\n")
        
        # Phase 1: Initial Portfolio Construction
        print("Phase 1: Portfolio Construction")
        print("-" * 40)
        
        buy_decisions = [
            ("RELIANCE", 0.6, "Strong fundamentals, energy sector recovery"),
            ("TCS", 0.8, "IT sector leader, strong growth"),
            ("HDFCBANK", 0.7, "Banking sector strength"),
            ("INFY", 0.5, "IT diversification"),
            ("ITC", 0.4, "Defensive FMCG play"),
            ("ASIANPAINT", 0.3, "Quality paint sector leader")
        ]
        
        portfolio_value = self.initial_capital
        positions = {}
        
        for symbol, sentiment, reason in buy_decisions:
            stock_info = self.stock_universe[symbol]
            price = stock_info["price"]
            
            # Execute buy order
            result = simulator.execute(
                symbol=symbol,
                action=ActionType.BUY,
                price=price,
                sentiment=sentiment
            )
            
            if result.success:
                portfolio_value = simulator.cash_balance + sum(
                    pos.quantity * self.stock_universe[pos.symbol]["price"] 
                    for pos in simulator.positions.values()
                )
                
                positions[symbol] = {
                    'quantity': result.quantity,
                    'entry_price': result.price,
                    'entry_time': result.timestamp,
                    'cost': result.quantity * result.price
                }
                
                # Log to ledger
                trade_data = {
                    'trade_id': f"BUY_{symbol}_{datetime.now().strftime('%H%M%S')}",
                    'timestamp': result.timestamp,
                    'symbol': symbol,
                    'action': 'BUY',
                    'quantity': result.quantity,
                    'price': result.price,
                    'pnl': -result.net_cost,  # Negative for buy (cash outflow)
                    'portfolio_value': portfolio_value,
                    'reason': reason
                }
                ledger.log_trade(trade_data)
                
                # Log to performance tracker
                perf_logger.log_trade(
                    trade_id=trade_data['trade_id'],
                    symbol=symbol,
                    action='BUY',
                    quantity=result.quantity,
                    price=result.price,
                    pnl=-result.net_cost,
                    portfolio_value=portfolio_value
                )
                
                print(f"✅ BUY {symbol}: {result.quantity} shares @ ₹{result.price:.2f}")
                print(f"   💰 Investment: ₹{result.net_cost:,.0f}")
                print(f"   📝 Reason: {reason}")
                print(f"   📊 Portfolio Value: ₹{portfolio_value:,.0f}\n")
            else:
                print(f"❌ Failed to buy {symbol}: {result.error_message}\n")
        
        # Wait and simulate market movements
        print("⏰ Holding Period: 3 months simulation...")
        print("📈 Market movements and rebalancing...\n")
        time.sleep(1)
        
        # Phase 2: Portfolio Rebalancing (simulate price changes)
        print("Phase 2: Portfolio Rebalancing")
        print("-" * 40)
        
        # Simulate market movements (some stocks up, some down)
        market_moves = {
            "RELIANCE": 0.08,   # +8%
            "TCS": -0.05,       # -5%
            "HDFCBANK": 0.12,   # +12%
            "INFY": -0.03,      # -3%
            "ITC": 0.15,        # +15%
            "ASIANPAINT": 0.06  # +6%
        }
        
        rebalance_decisions = [
            ("TCS", "SELL", 0.5, "Taking profits, valuation concerns"),
            ("HDFCBANK", "SELL", 0.6, "Booking profits after 12% gain"),
            ("ITC", "SELL", 0.8, "Excellent 15% return, rebalancing")
        ]
        
        for symbol, action_type, sentiment, reason in rebalance_decisions:
            if symbol not in positions:
                continue
                
            original_price = positions[symbol]['entry_price']
            current_price = original_price * (1 + market_moves[symbol])
            
            # Execute sell order
            result = simulator.execute(
                symbol=symbol,
                action=ActionType.SELL,
                price=current_price,
                sentiment=sentiment
            )
            
            if result.success:
                # Calculate P&L
                entry_cost = positions[symbol]['cost']
                exit_value = abs(result.quantity) * current_price
                trade_pnl = exit_value - entry_cost - result.transaction_costs
                
                portfolio_value = simulator.cash_balance + sum(
                    pos.quantity * self.stock_universe[pos.symbol]["price"] * (1 + market_moves.get(pos.symbol, 0))
                    for pos in simulator.positions.values()
                )
                
                # Log to ledger
                trade_data = {
                    'trade_id': f"SELL_{symbol}_{datetime.now().strftime('%H%M%S')}",
                    'timestamp': result.timestamp,
                    'symbol': symbol,
                    'action': 'SELL',
                    'quantity': result.quantity,  # Negative for sell
                    'price': current_price,
                    'pnl': trade_pnl,
                    'portfolio_value': portfolio_value,
                    'reason': reason,
                    'entry_price': positions[symbol]['entry_price'],
                    'holding_period_days': (result.timestamp - positions[symbol]['entry_time']).days,
                    'return_pct': ((current_price / positions[symbol]['entry_price']) - 1) * 100
                }
                ledger.log_trade(trade_data)
                
                # Log to performance tracker
                perf_logger.log_trade(
                    trade_id=trade_data['trade_id'],
                    symbol=symbol,
                    action='SELL',
                    quantity=result.quantity,
                    price=current_price,
                    pnl=trade_pnl,
                    portfolio_value=portfolio_value
                )
                
                print(f"✅ SELL {symbol}: {abs(result.quantity)} shares @ ₹{current_price:.2f}")
                print(f"   📈 Entry Price: ₹{positions[symbol]['entry_price']:.2f}")
                print(f"   📊 Return: {((current_price/positions[symbol]['entry_price'])-1)*100:+.1f}%")
                print(f"   💰 P&L: ₹{trade_pnl:+,.0f}")
                print(f"   📝 Reason: {reason}")
                print(f"   📊 Portfolio Value: ₹{portfolio_value:,.0f}\n")
                
                # Remove from positions
                del positions[symbol]
            else:
                print(f"❌ Failed to sell {symbol}: {result.error_message}\n")
        
        # Finalize session
        duration_minutes = 5.0
        ledger.finalize_session(duration_minutes)
        
        # Generate final reports
        self.generate_reports(session_id, perf_logger, simulator, positions, market_moves)
    
    def generate_reports(self, session_id, perf_logger, simulator, remaining_positions, market_moves):
        """Generate comprehensive portfolio reports"""
        
        print("📋 Portfolio Performance Report")
        print("=" * 60)
        
        # Portfolio summary
        current_value = simulator.cash_balance
        for pos in simulator.positions.values():
            current_price = self.stock_universe[pos.symbol]["price"] * (1 + market_moves.get(pos.symbol, 0))
            current_value += pos.quantity * current_price
        
        total_return = current_value - self.initial_capital
        return_pct = (total_return / self.initial_capital) * 100
        
        print(f"Initial Capital: ₹{self.initial_capital:,.0f}")
        print(f"Current Value:   ₹{current_value:,.0f}")
        print(f"Total Return:    ₹{total_return:+,.0f} ({return_pct:+.2f}%)")
        print(f"Cash Balance:    ₹{simulator.cash_balance:,.0f}")
        
        # Current holdings
        if simulator.positions:
            print(f"\n📊 Current Holdings:")
            for symbol, pos in simulator.positions.items():
                current_price = self.stock_universe[symbol]["price"] * (1 + market_moves.get(symbol, 0))
                unrealized_pnl = (current_price - pos.avg_entry_price) * pos.quantity
                print(f"   {symbol}: {pos.quantity} shares @ ₹{current_price:.2f} "
                      f"(Entry: ₹{pos.avg_entry_price:.2f}, "
                      f"P&L: ₹{unrealized_pnl:+,.0f})")
        
        # Performance metrics
        metrics = perf_logger.get_comprehensive_metrics()
        print(f"\n📈 Performance Metrics:")
        print(f"   Total Trades:     {metrics.trades_count}")
        print(f"   Winning Trades:   {metrics.winning_trades}")
        print(f"   Win Rate:         {metrics.win_rate:.1f}%")
        print(f"   Average Trade:    ₹{metrics.avg_trade_return:,.0f}")
        print(f"   Best Trade:       ₹{metrics.best_trade:,.0f}")
        print(f"   Worst Trade:      ₹{metrics.worst_trade:,.0f}")
        if metrics.sharpe_ratio != 0:
            print(f"   Sharpe Ratio:     {metrics.sharpe_ratio:.2f}")
        
        # File locations
        print(f"\n📄 Generated Files:")
        print(f"   CSV Ledger:       {self.demo_dir}/ledgers/ledger_master.csv")
        print(f"   Session JSON:     {self.demo_dir}/sessions/session_{session_id}.json")
        
        # Display ledger
        self.show_trade_ledger()
    
    def show_trade_ledger(self):
        """Display the trade ledger in a readable format"""
        csv_file = self.demo_dir / "ledgers" / "ledger_master.csv"
        
        if csv_file.exists():
            print(f"\n📋 Trade Ledger Summary:")
            print("=" * 100)
            df = pd.read_csv(csv_file)
            
            # Format and display key columns
            display_columns = ['timestamp', 'symbol', 'action', 'quantity', 'price', 'pnl']
            if all(col in df.columns for col in display_columns):
                df_display = df[display_columns].copy()
                df_display['timestamp'] = pd.to_datetime(df_display['timestamp']).dt.strftime('%H:%M:%S')
                df_display['price'] = df_display['price'].apply(lambda x: f"₹{x:,.0f}")
                df_display['pnl'] = df_display['pnl'].apply(lambda x: f"₹{x:+,.0f}")
                
                print(df_display.to_string(index=False, max_colwidth=15))
            else:
                print("Ledger columns:", df.columns.tolist())
                print(df.head())
        else:
            print("❌ Ledger file not found")


def main():
    """Run the equity portfolio demonstration"""
    demo = EquityPortfolioDemo(initial_capital=500000.0)  # 5 lakhs
    demo.run_portfolio_simulation()
    
    print("\n" + "=" * 60)
    print("✅ Demo Complete! Check the 'demo_outputs' folder for detailed reports.")
    print("📊 This shows how the portfolio manager works with real equity trading.")


if __name__ == "__main__":
    main()