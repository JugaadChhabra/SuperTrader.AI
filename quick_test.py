#!/usr/bin/env python3

import sys
sys.path.insert(0, '/Users/jugaadchhabra/Documents/Github/SuperTrader.AI')

from utils.portfolio_simulator import PortfolioSimulator, ActionType

# Simple test
simulator = PortfolioSimulator(initial_capital=100000.0)

print(f"Initial Capital: {simulator.initial_capital}")
print(f"Cash Balance: {simulator.cash_balance}")
print(f"Available Margin: {simulator.available_margin}")
print(f"Margin Requirement: {simulator.margin_requirement}")

# Test position sizing calculation
symbol = "NIFTY"
price = 19500.0
lot_size = 50
notional_per_lot = lot_size * price
margin_per_lot = notional_per_lot * simulator.margin_requirement

print(f"\nPosition sizing for {symbol}:")
print(f"Price: {price}")
print(f"Lot size: {lot_size}")
print(f"Notional per lot: {notional_per_lot}")
print(f"Margin per lot: {margin_per_lot}")
print(f"Max lots by margin: {int(simulator.available_margin / margin_per_lot)}")

result = simulator.execute(
    symbol="NIFTY",
    action=ActionType.BUY, 
    price=19500.0
)

print(f"\nTrade Success: {result.success}")
print(f"Trade Quantity: {result.quantity}")
print(f"Error Message: {getattr(result, 'error_message', 'None')}")
print(f"Result attributes: {[attr for attr in dir(result) if not attr.startswith('_')]}")