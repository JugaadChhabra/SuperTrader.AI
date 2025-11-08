#!/usr/bin/env python3
"""
Simple Equity Portfolio Test - Clear P&L Tracking

This is the main file to run to test the portfolio simulator.
Shows clear entry/exit prices and P&L for each transaction.
"""

import sys
import json
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.portfolio_simulator import PortfolioSimulator, ActionType
from agents.enhanced_trade_ledger import SessionLedger

def run_simple_portfolio_test():
    """
    Simple portfolio test with clear P&L tracking
    
    Data Used:
    - Real Indian stock prices (RELIANCE, TCS, HDFCBANK, etc.)
    - Realistic position sizing (2-5% of portfolio per stock)
    - Transaction costs and slippage modeling
    - Portfolio rebalancing scenarios
    """
    
    print("🚀 Simple Equity Portfolio Manager Test")
    print("=" * 50)
    
    # Initialize with 10 lakhs capital for equity trading
    initial_capital = 1_000_000.0
    simulator = PortfolioSimulator(initial_capital=initial_capital)
    
    # Create output directory
    output_dir = Path("simple_test_output")
    output_dir.mkdir(exist_ok=True)
    
    # Start trading session
    ledger = SessionLedger(base_path=str(output_dir))
    session_id = ledger.start_session(
        symbols=["RELIANCE", "TCS", "HDFCBANK", "INFY", "ITC"],
        strategy="Simple Portfolio Test",
        initial_capital=initial_capital
    )
    
    print(f"📊 Initial Capital: ₹{initial_capital:,.0f}")
    print(f"📝 Session ID: {session_id}")
    print("💼 Building equity portfolio...\n")
    
    # Track all transactions for clear P&L
    transactions = []
    
    # Stock data (realistic November 2024 prices)
    stocks = {
        "RELIANCE": 2850.0,
        "TCS": 4200.0, 
        "HDFCBANK": 1650.0,
        "INFY": 1820.0,
        "ITC": 485.0
    }
    
    # Phase 1: Buy some stocks
    print("Phase 1: Building Portfolio")
    print("-" * 30)
    
    for symbol, price in stocks.items():
        result = simulator.execute(
            symbol=symbol,
            action=ActionType.BUY,
            price=price,
            sentiment=0.5  # Moderate conviction
        )
        
        if result.success:
            cost = result.quantity * result.price + result.transaction_costs
            
            transactions.append({
                'type': 'BUY',
                'symbol': symbol,
                'quantity': result.quantity,
                'entry_price': result.price,
                'total_cost': cost,
                'timestamp': result.timestamp
            })
            
            # Log to ledger with proper data
            trade_data = {
                'trade_id': f"BUY_{symbol}_{datetime.now().strftime('%H%M%S')}",
                'timestamp': result.timestamp,
                'symbol': symbol,
                'action': 'BUY',
                'quantity': result.quantity,
                'price': result.price,
                'total_cost': cost,
                'pnl': -cost,  # Negative for purchases
                'portfolio_value': get_portfolio_value(simulator, stocks),
                'transaction_type': 'ENTRY',
                'notes': f"Initial purchase of {result.quantity} shares"
            }
            ledger.log_trade(trade_data)
            
            print(f"✅ BOUGHT {symbol}")
            print(f"   📊 Quantity: {result.quantity} shares")
            print(f"   💰 Entry Price: ₹{result.price:.2f}")
            print(f"   💸 Total Cost: ₹{cost:,.0f}")
            print(f"   🏦 Cash Remaining: ₹{simulator.cash_balance:,.0f}\n")
        else:
            print(f"❌ Failed to buy {symbol}: {result.error_message}\n")
    
    print("⏰ Holding period simulation (3 months)...")
    print("📈 Some stocks gained, others lost value\n")
    
    # Phase 2: Sell some positions (simulate market movements)
    print("Phase 2: Portfolio Rebalancing")
    print("-" * 30)
    
    # Simulate price changes after 3 months
    new_prices = {
        "RELIANCE": 3100.0,  # +8.8% gain
        "TCS": 3950.0,       # -6.0% loss  
        "HDFCBANK": 1850.0,  # +12.1% gain
        "INFY": 1750.0,      # -3.8% loss
        "ITC": 580.0         # +19.6% gain
    }
    
    # Sell some positions
    sell_decisions = ["RELIANCE", "TCS", "ITC"]
    
    for symbol in sell_decisions:
        if symbol not in [t['symbol'] for t in transactions if t['type'] == 'BUY']:
            continue
            
        new_price = new_prices[symbol]
        
        result = simulator.execute(
            symbol=symbol,
            action=ActionType.SELL,
            price=new_price,
            sentiment=0.6
        )
        
        if result.success:
            # Find original purchase
            original = next(t for t in transactions if t['symbol'] == symbol and t['type'] == 'BUY')
            
            # Calculate P&L
            shares_sold = abs(result.quantity)
            proceeds = shares_sold * result.price - result.transaction_costs
            cost_basis = (original['total_cost'] / original['quantity']) * shares_sold
            pnl = proceeds - cost_basis
            return_pct = (result.price / original['entry_price'] - 1) * 100
            holding_days = (result.timestamp - original['timestamp']).days
            
            transactions.append({
                'type': 'SELL',
                'symbol': symbol,
                'quantity': shares_sold,
                'exit_price': result.price,
                'proceeds': proceeds,
                'pnl': pnl,
                'return_pct': return_pct,
                'timestamp': result.timestamp
            })
            
            # Log to ledger
            trade_data = {
                'trade_id': f"SELL_{symbol}_{datetime.now().strftime('%H%M%S')}",
                'timestamp': result.timestamp,
                'symbol': symbol,
                'action': 'SELL',
                'quantity': -shares_sold,  # Negative for sales
                'price': result.price,
                'total_proceeds': proceeds,
                'pnl': pnl,
                'portfolio_value': get_portfolio_value(simulator, new_prices),
                'transaction_type': 'EXIT',
                'entry_price': original['entry_price'],
                'return_pct': return_pct,
                'holding_days': holding_days,
                'notes': f"Sold {shares_sold} shares for {return_pct:+.1f}% return"
            }
            ledger.log_trade(trade_data)
            
            print(f"✅ SOLD {symbol}")
            print(f"   📊 Quantity: {shares_sold} shares")
            print(f"   📈 Entry Price: ₹{original['entry_price']:.2f}")
            print(f"   📉 Exit Price: ₹{result.price:.2f}")
            print(f"   💰 Total Proceeds: ₹{proceeds:,.0f}")
            print(f"   📊 P&L: ₹{pnl:+,.0f} ({return_pct:+.1f}%)")
            print(f"   📅 Holding Period: {holding_days} days\n")
        else:
            print(f"❌ Failed to sell {symbol}: {result.error_message}\n")
    
    # Finalize session
    ledger.finalize_session(5.0)
    
    # Generate summary report
    generate_summary_report(simulator, transactions, new_prices, output_dir, session_id)

def get_portfolio_value(simulator, current_prices):
    """Calculate total portfolio value"""
    cash = simulator.cash_balance
    holdings_value = sum(
        pos.quantity * current_prices.get(pos.symbol, 0)
        for pos in simulator.positions.values()
    )
    return cash + holdings_value

def generate_summary_report(simulator, transactions, current_prices, output_dir, session_id):
    """Generate comprehensive summary report"""
    
    print("📋 PORTFOLIO SUMMARY REPORT")
    print("=" * 50)
    
    # Calculate portfolio performance
    initial_capital = 1_000_000.0
    current_value = get_portfolio_value(simulator, current_prices)
    total_return = current_value - initial_capital
    return_pct = (total_return / initial_capital) * 100
    
    print(f"💰 Initial Capital: ₹{initial_capital:,.0f}")
    print(f"💰 Current Value:   ₹{current_value:,.0f}")
    print(f"📊 Total Return:    ₹{total_return:+,.0f} ({return_pct:+.2f}%)")
    print(f"🏦 Cash Balance:    ₹{simulator.cash_balance:,.0f}")
    
    # Current holdings
    if simulator.positions:
        print(f"\n📊 CURRENT HOLDINGS:")
        total_unrealized = 0
        for symbol, pos in simulator.positions.items():
            current_price = current_prices.get(symbol, 0)
            market_value = pos.quantity * current_price
            cost_basis = pos.quantity * pos.avg_entry_price
            unrealized_pnl = market_value - cost_basis
            unrealized_pct = (current_price / pos.avg_entry_price - 1) * 100
            total_unrealized += unrealized_pnl
            
            print(f"   {symbol}: {pos.quantity} shares")
            print(f"      Entry: ₹{pos.avg_entry_price:.2f}")
            print(f"      Current: ₹{current_price:.2f}")
            print(f"      Market Value: ₹{market_value:,.0f}")
            print(f"      Unrealized P&L: ₹{unrealized_pnl:+,.0f} ({unrealized_pct:+.1f}%)")
        
        print(f"   Total Unrealized P&L: ₹{total_unrealized:+,.0f}")
    
    # Realized P&L summary
    realized_trades = [t for t in transactions if t['type'] == 'SELL']
    if realized_trades:
        total_realized = sum(t['pnl'] for t in realized_trades)
        winning_trades = len([t for t in realized_trades if t['pnl'] > 0])
        win_rate = (winning_trades / len(realized_trades)) * 100
        
        print(f"\n💹 REALIZED TRADES SUMMARY:")
        print(f"   Total Realized P&L: ₹{total_realized:+,.0f}")
        print(f"   Trades Executed: {len(realized_trades)}")
        print(f"   Winning Trades: {winning_trades}")
        print(f"   Win Rate: {win_rate:.1f}%")
        
        print(f"\n📈 INDIVIDUAL TRADE RESULTS:")
        for trade in realized_trades:
            print(f"   {trade['symbol']}: ₹{trade['pnl']:+,.0f} ({trade['return_pct']:+.1f}%)")
    
    # File locations
    print(f"\n📄 Generated Files:")
    print(f"   📊 CSV Ledger: {output_dir}/ledgers/ledger_master.csv")
    print(f"   📋 Session Data: {output_dir}/sessions/session_{session_id}.json")
    
    print(f"\n" + "=" * 50)
    print("✅ Portfolio simulation completed successfully!")
    print("📊 Check the generated files for detailed transaction logs")
    
    # Display CSV content
    csv_file = output_dir / "ledgers" / "ledger_master.csv"
    if csv_file.exists():
        print(f"\n📋 TRANSACTION LEDGER:")
        df = pd.read_csv(csv_file)
        
        # Create simplified view
        if len(df) > 0:
            display_cols = ['symbol', 'action', 'quantity', 'price', 'pnl', 'notes']
            available_cols = [col for col in display_cols if col in df.columns]
            
            if available_cols:
                print(df[available_cols].to_string(index=False, max_colwidth=30))
            else:
                print("Available columns:", df.columns.tolist())
                print(df.head())

if __name__ == "__main__":
    run_simple_portfolio_test()