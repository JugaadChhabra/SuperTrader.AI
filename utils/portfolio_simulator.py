"""
Portfolio Simulator for SuperTrader.AI - Paper Trading Implementation

Handles mock trade execution without broker integration, managing:
- Cash and position balances
- P&L calculations
- Transaction cost simulation
- Risk metrics tracking
- Performance analytics

This simulator replicates real trading conditions for backtesting and
development without live market exposure.

Author: SuperTrader.AI Team
Version: 1.0.0
Last Updated: 2024-11-08
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
import json


logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Trade action types for the simulator"""
    BUY = 1
    SELL = -1
    HOLD = 0


@dataclass
class SimulatedPosition:
    """Represents a simulated position in the portfolio"""
    symbol: str
    quantity: int  # Positive for long, negative for short
    avg_entry_price: float
    current_price: float
    total_cost: float  # Including transaction costs
    entry_time: datetime
    unrealized_pnl: float = 0.0
    
    def update_price(self, new_price: float) -> None:
        """Update current price and recalculate unrealized P&L"""
        self.current_price = new_price
        
        if self.quantity > 0:  # Long position
            self.unrealized_pnl = (new_price - self.avg_entry_price) * self.quantity
        elif self.quantity < 0:  # Short position
            self.unrealized_pnl = (self.avg_entry_price - new_price) * abs(self.quantity)
        else:
            self.unrealized_pnl = 0.0
    
    def get_market_value(self) -> float:
        """Get current market value of the position"""
        return abs(self.quantity) * self.current_price
    
    def get_notional_value(self) -> float:
        """Get notional exposure (market value)"""
        return self.get_market_value()


@dataclass
class TradeResult:
    """Result of a simulated trade execution"""
    trade_id: str
    symbol: str
    action: ActionType
    quantity: int
    price: float
    timestamp: datetime
    
    # Financial details
    notional_value: float
    transaction_costs: float
    net_cost: float  # Cost including fees
    
    # Context
    sentiment: float
    q_values: List[float]
    confidence: float
    
    # Execution quality
    slippage_bps: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    
    # Position impact
    position_change: Dict[str, Any] = field(default_factory=dict)
    portfolio_impact: Dict[str, float] = field(default_factory=dict)


class PortfolioSimulator:
    """
    Paper trading portfolio simulator
    
    Simulates realistic trading conditions without broker integration:
    - Maintains cash and position balances
    - Calculates transaction costs based on NSE fee structure
    - Tracks P&L and performance metrics
    - Simulates market impact and slippage
    - Provides realistic execution delays
    """
    
    def __init__(self, 
                 initial_capital: float = 1_000_000.0,
                 max_leverage: float = 1.0,  # No leverage for equity cash trading
                 margin_requirement: float = 1.0,  # 100% cash required for equity
                 transaction_cost_bps: float = 10.0,  # 10 bps total costs (brokerage + taxes)
                 slippage_bps: float = 2.0):  # 2 basis points slippage
        
        self.initial_capital = initial_capital
        self.max_leverage = max_leverage
        self.margin_requirement = margin_requirement
        self.transaction_cost_bps = transaction_cost_bps / 10000  # Convert to decimal
        self.slippage_bps = slippage_bps / 10000
        
        # Portfolio state
        self.cash_balance = initial_capital
        self.positions: Dict[str, SimulatedPosition] = {}
        self.trade_history: List[TradeResult] = []
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_value = initial_capital
        
        # Risk metrics
        self.margin_used = 0.0
        self.total_exposure = 0.0
        
        logger.info(f"PortfolioSimulator initialized with ₹{initial_capital:,.0f} capital")
    
    @property
    def total_value(self) -> float:
        """Calculate total portfolio value (cash + unrealized P&L)"""
        return self.cash_balance + self.unrealized_pnl
    
    @property
    def available_margin(self) -> float:
        """Calculate available margin for new positions"""
        return max(0, self.cash_balance - self.margin_used)
    
    @property
    def leverage_ratio(self) -> float:
        """Calculate current leverage ratio"""
        if self.cash_balance <= 0:
            return float('inf')
        return self.total_exposure / self.cash_balance
    
    def execute(self, 
                symbol: str, 
                action: Union[ActionType, int, float], 
                price: float, 
                sentiment: float = 0.0, 
                q_values: List[float] = None,
                quantity_override: Optional[int] = None) -> TradeResult:
        """
        Execute a simulated trade
        
        Args:
            symbol: Trading symbol (e.g., 'NIFTY', 'BANKNIFTY')
            action: Action type (1=BUY, -1=SELL, 0=HOLD)
            price: Current market price
            sentiment: Market sentiment (-1 to 1)
            q_values: Q-values from RL model
            quantity_override: Override calculated position size
            
        Returns:
            TradeResult with execution details
        """
        trade_id = f"SIM_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        timestamp = datetime.now()
        
        # Normalize action
        if isinstance(action, (int, float)):
            if action > 0.1:
                action = ActionType.BUY
            elif action < -0.1:
                action = ActionType.SELL
            else:
                action = ActionType.HOLD
        
        # Handle HOLD action
        if action == ActionType.HOLD:
            return TradeResult(
                trade_id=trade_id,
                symbol=symbol,
                action=action,
                quantity=0,
                price=price,
                timestamp=timestamp,
                notional_value=0.0,
                transaction_costs=0.0,
                net_cost=0.0,
                sentiment=sentiment,
                q_values=q_values or [0.0, 1.0, 0.0],  # High HOLD confidence
                confidence=0.8,
                success=True
            )
        
        try:
            # Calculate position size
            quantity = self._calculate_position_size(symbol, action, price, sentiment, quantity_override)
            
            if quantity == 0:
                return TradeResult(
                    trade_id=trade_id,
                    symbol=symbol,
                    action=action,
                    quantity=0,
                    price=price,
                    timestamp=timestamp,
                    notional_value=0.0,
                    transaction_costs=0.0,
                    net_cost=0.0,
                    sentiment=sentiment,
                    q_values=q_values or [0.0, 0.0, 1.0],
                    confidence=0.3,
                    success=False,
                    error_message="Insufficient capital or margin"
                )
            
            # Apply slippage
            execution_price = self._apply_slippage(price, action, quantity, sentiment)
            slippage_bps = abs(execution_price - price) / price * 10000
            
            # Calculate costs
            notional_value = abs(quantity) * execution_price
            transaction_costs = self._calculate_transaction_costs(notional_value, action)
            net_cost = notional_value + transaction_costs
            
            # Check available capital
            required_margin = notional_value * self.margin_requirement
            
            if action == ActionType.BUY:
                if self.cash_balance < net_cost:
                    return TradeResult(
                        trade_id=trade_id,
                        symbol=symbol,
                        action=action,
                        quantity=quantity,
                        price=price,
                        timestamp=timestamp,
                        notional_value=notional_value,
                        transaction_costs=transaction_costs,
                        net_cost=net_cost,
                        sentiment=sentiment,
                        q_values=q_values or [0.0, 0.0, 1.0],
                        confidence=0.2,
                        success=False,
                        error_message=f"Insufficient cash: need ₹{net_cost:,.0f}, have ₹{self.cash_balance:,.0f}"
                    )
                
                # Execute buy order
                self._execute_buy(symbol, quantity, execution_price, net_cost, required_margin, timestamp)
                
            else:  # SELL
                if symbol not in self.positions or self.positions[symbol].quantity <= 0:
                    return TradeResult(
                        trade_id=trade_id,
                        symbol=symbol,
                        action=action,
                        quantity=quantity,
                        price=price,
                        timestamp=timestamp,
                        notional_value=notional_value,
                        transaction_costs=transaction_costs,
                        net_cost=net_cost,
                        sentiment=sentiment,
                        q_values=q_values or [0.0, 0.0, 1.0],
                        confidence=0.2,
                        success=False,
                        error_message=f"No position to sell in {symbol}"
                    )
                
                # Execute sell order
                self._execute_sell(symbol, quantity, execution_price, transaction_costs, timestamp)
            
            # Update portfolio metrics
            self._update_portfolio_metrics()
            
            # Create successful trade result
            trade_result = TradeResult(
                trade_id=trade_id,
                symbol=symbol,
                action=action,
                quantity=quantity,
                price=execution_price,
                timestamp=timestamp,
                notional_value=notional_value,
                transaction_costs=transaction_costs,
                net_cost=net_cost,
                sentiment=sentiment,
                q_values=q_values or [0.0, 0.0, 1.0],
                confidence=self._calculate_confidence(sentiment, q_values),
                slippage_bps=slippage_bps,
                success=True,
                position_change=self._get_position_change(symbol),
                portfolio_impact=self._get_portfolio_impact()
            )
            
            # Record trade
            self.trade_history.append(trade_result)
            self.total_trades += 1
            
            logger.info(f"✅ Executed: {action.name} {quantity} {symbol} @ ₹{execution_price:.2f} "
                       f"(slippage: {slippage_bps:.1f}bps)")
            
            return trade_result
            
        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}")
            return TradeResult(
                trade_id=trade_id,
                symbol=symbol,
                action=action,
                quantity=0,
                price=price,
                timestamp=timestamp,
                notional_value=0.0,
                transaction_costs=0.0,
                net_cost=0.0,
                sentiment=sentiment,
                q_values=q_values or [1.0, 0.0, 0.0],
                confidence=0.0,
                success=False,
                error_message=str(e)
            )
    
    def _calculate_position_size(self, 
                                symbol: str, 
                                action: ActionType, 
                                price: float, 
                                sentiment: float,
                                override: Optional[int] = None) -> int:
        """Calculate appropriate position size for equity portfolio management"""
        if override is not None:
            return abs(override)
        
        # Portfolio position sizing for equity stocks
        # Risk 2-5% of portfolio per position based on sentiment/conviction
        base_risk = 0.02  # 2% base allocation
        sentiment_boost = abs(sentiment) * 0.03  # Up to 3% additional based on sentiment
        position_risk = min(0.05, base_risk + sentiment_boost)  # Cap at 5% per position
        
        # Calculate position value
        position_value = self.cash_balance * position_risk
        
        # Calculate number of shares
        shares = int(position_value / price)
        
        # Minimum position size (at least 1 share if we have enough cash)
        if shares == 0 and self.cash_balance >= price:
            shares = 1
        
        # Maximum position check (don't exceed 10% of portfolio in any single stock)
        max_position_value = self.cash_balance * 0.10
        max_shares = int(max_position_value / price)
        shares = min(shares, max_shares)
        
        # Ensure we have enough cash for the purchase
        total_cost = shares * price * (1 + self.transaction_cost_bps)
        if total_cost > self.cash_balance:
            shares = int(self.cash_balance / (price * (1 + self.transaction_cost_bps)))
        
        logger.debug(f"Equity position sizing for {symbol}: {shares} shares "
                    f"(risk: {position_risk:.1%}, value: ₹{shares * price:,.0f}, "
                    f"sentiment: {sentiment:.2f})")
        
        return shares
    
    def _apply_slippage(self, price: float, action: ActionType, quantity: int, sentiment: float) -> float:
        """Apply realistic slippage based on market conditions"""
        base_slippage = self.slippage_bps
        
        # Increase slippage for large orders
        if quantity > 1000:  # Large order
            base_slippage *= 1.5
        elif quantity > 500:  # Medium order
            base_slippage *= 1.2
        
        # Increase slippage for negative sentiment (poor market conditions)
        if abs(sentiment) > 0.5:
            base_slippage *= (1 + abs(sentiment) * 0.5)
        
        # Apply slippage in the unfavorable direction
        if action == ActionType.BUY:
            slippage_amount = price * base_slippage
        else:
            slippage_amount = -price * base_slippage
        
        execution_price = price + slippage_amount
        return max(0.01, execution_price)  # Minimum price of 1 paisa
    
    def _calculate_transaction_costs(self, notional_value: float, action: ActionType) -> float:
        """Calculate realistic transaction costs based on NSE fee structure"""
        # Base brokerage: ₹20 per order
        brokerage = 20.0
        
        # STT: 0.0125% on sell side for futures
        stt = notional_value * 0.000125 if action == ActionType.SELL else 0.0
        
        # Exchange charges: 0.00345% of turnover
        exchange_charges = notional_value * 0.0000345
        
        # SEBI charges: ₹1 per crore
        sebi_charges = notional_value * 0.000001
        
        # GST: 18% on brokerage and charges (not on STT)
        taxable_amount = brokerage + exchange_charges + sebi_charges
        gst = taxable_amount * 0.18
        
        total_costs = brokerage + stt + exchange_charges + sebi_charges + gst
        
        return total_costs
    
    def _execute_buy(self, symbol: str, quantity: int, price: float, net_cost: float, margin_required: float, timestamp: datetime) -> None:
        """Execute a buy order"""
        self.cash_balance -= net_cost
        self.margin_used += margin_required
        self.total_exposure += quantity * price
        
        if symbol in self.positions:
            # Update existing position (average price)
            pos = self.positions[symbol]
            total_value = pos.quantity * pos.avg_entry_price + quantity * price
            total_quantity = pos.quantity + quantity
            
            pos.avg_entry_price = total_value / total_quantity
            pos.quantity = total_quantity
            pos.total_cost += net_cost
            pos.current_price = price
            pos.update_price(price)
        else:
            # Create new position
            self.positions[symbol] = SimulatedPosition(
                symbol=symbol,
                quantity=quantity,
                avg_entry_price=price,
                current_price=price,
                total_cost=net_cost,
                entry_time=timestamp
            )
    
    def _execute_sell(self, symbol: str, quantity: int, price: float, transaction_costs: float, timestamp: datetime) -> None:
        """Execute a sell order"""
        pos = self.positions[symbol]
        sell_quantity = min(quantity, pos.quantity)
        
        # Calculate realized P&L
        realized_pnl = (price - pos.avg_entry_price) * sell_quantity - transaction_costs
        self.realized_pnl += realized_pnl
        self.cash_balance += sell_quantity * price - transaction_costs
        
        # Update position
        pos.quantity -= sell_quantity
        margin_released = sell_quantity * price * self.margin_requirement
        self.margin_used -= margin_released
        self.total_exposure -= sell_quantity * price
        
        if realized_pnl > 0:
            self.winning_trades += 1
        
        # Remove position if fully closed
        if pos.quantity <= 0:
            del self.positions[symbol]
    
    def _update_portfolio_metrics(self) -> None:
        """Update portfolio performance metrics"""
        # Update current prices and unrealized P&L
        self.unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        
        # Update total P&L
        self.total_pnl = self.realized_pnl + self.unrealized_pnl
        
        # Update peak and drawdown
        current_value = self.total_value
        if current_value > self.peak_value:
            self.peak_value = current_value
        
        current_drawdown = (self.peak_value - current_value) / self.peak_value
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
    
    def _calculate_confidence(self, sentiment: float, q_values: Optional[List[float]]) -> float:
        """Calculate trade confidence based on sentiment and Q-values"""
        base_confidence = 0.5
        
        # Sentiment contribution (0.0 to 0.3)
        sentiment_boost = abs(sentiment) * 0.3
        
        # Q-values contribution (0.0 to 0.2)
        q_boost = 0.0
        if q_values and len(q_values) >= 3:
            max_q = max(q_values)
            avg_q = np.mean(q_values)
            if max_q > avg_q:
                q_boost = (max_q - avg_q) * 0.2
        
        return min(1.0, base_confidence + sentiment_boost + q_boost)
    
    def _get_position_change(self, symbol: str) -> Dict[str, Any]:
        """Get position change details"""
        if symbol in self.positions:
            pos = self.positions[symbol]
            return {
                'symbol': symbol,
                'quantity': pos.quantity,
                'avg_entry_price': pos.avg_entry_price,
                'current_price': pos.current_price,
                'unrealized_pnl': pos.unrealized_pnl,
                'market_value': pos.get_market_value()
            }
        return {'symbol': symbol, 'quantity': 0}
    
    def _get_portfolio_impact(self) -> Dict[str, float]:
        """Get portfolio-level impact metrics"""
        return {
            'total_value': self.total_value,
            'cash_balance': self.cash_balance,
            'total_exposure': self.total_exposure,
            'margin_used': self.margin_used,
            'leverage_ratio': self.leverage_ratio,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl
        }
    
    def update_market_prices(self, price_updates: Dict[str, float]) -> None:
        """Update current market prices for all positions"""
        for symbol, new_price in price_updates.items():
            if symbol in self.positions:
                self.positions[symbol].update_price(new_price)
        
        self._update_portfolio_metrics()
        
        logger.debug(f"Updated market prices: total value = ₹{self.total_value:,.0f}, "
                    f"unrealized P&L = ₹{self.unrealized_pnl:,.0f}")
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary"""
        win_rate = self.winning_trades / max(self.total_trades, 1)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'capital': {
                'initial_capital': self.initial_capital,
                'cash_balance': self.cash_balance,
                'total_value': self.total_value,
                'total_pnl': self.total_pnl,
                'pnl_percentage': (self.total_pnl / self.initial_capital) * 100
            },
            'positions': {
                symbol: {
                    'quantity': pos.quantity,
                    'avg_entry_price': pos.avg_entry_price,
                    'current_price': pos.current_price,
                    'market_value': pos.get_market_value(),
                    'unrealized_pnl': pos.unrealized_pnl,
                    'unrealized_pnl_pct': (pos.unrealized_pnl / (pos.avg_entry_price * abs(pos.quantity))) * 100 if pos.quantity != 0 else 0
                }
                for symbol, pos in self.positions.items()
            },
            'performance': {
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'win_rate': win_rate,
                'realized_pnl': self.realized_pnl,
                'unrealized_pnl': self.unrealized_pnl,
                'max_drawdown': self.max_drawdown,
                'peak_value': self.peak_value
            },
            'risk': {
                'total_exposure': self.total_exposure,
                'margin_used': self.margin_used,
                'available_margin': self.available_margin,
                'leverage_ratio': self.leverage_ratio,
                'max_leverage': self.max_leverage
            }
        }
    
    def reset_portfolio(self, new_capital: Optional[float] = None) -> None:
        """Reset portfolio to initial state"""
        if new_capital is not None:
            self.initial_capital = new_capital
        
        self.cash_balance = self.initial_capital
        self.positions.clear()
        self.trade_history.clear()
        
        # Reset metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_value = self.initial_capital
        self.margin_used = 0.0
        self.total_exposure = 0.0
        
        logger.info(f"Portfolio reset with ₹{self.initial_capital:,.0f} capital")
    
    def close_all_positions(self, current_prices: Dict[str, float]) -> List[TradeResult]:
        """Close all open positions at current market prices"""
        closing_trades = []
        
        for symbol, position in list(self.positions.items()):
            if symbol in current_prices and position.quantity > 0:
                close_result = self.execute(
                    symbol=symbol,
                    action=ActionType.SELL,
                    price=current_prices[symbol],
                    sentiment=0.0,
                    q_values=[0.0, 0.0, 1.0],  # High HOLD confidence for closing
                    quantity_override=position.quantity
                )
                closing_trades.append(close_result)
        
        return closing_trades
    
    def get_trade_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get trade history as serializable dictionaries"""
        trades = self.trade_history[-limit:] if limit else self.trade_history
        
        return [
            {
                'trade_id': trade.trade_id,
                'symbol': trade.symbol,
                'action': trade.action.name,
                'quantity': trade.quantity,
                'price': trade.price,
                'timestamp': trade.timestamp.isoformat(),
                'notional_value': trade.notional_value,
                'transaction_costs': trade.transaction_costs,
                'sentiment': trade.sentiment,
                'confidence': trade.confidence,
                'slippage_bps': trade.slippage_bps,
                'success': trade.success,
                'error_message': trade.error_message
            }
            for trade in trades
        ]


# Global singleton instance
_portfolio_simulator: Optional[PortfolioSimulator] = None


def get_portfolio_simulator(initial_capital: float = 1_000_000.0) -> PortfolioSimulator:
    """Get or create the global portfolio simulator instance"""
    global _portfolio_simulator
    
    if _portfolio_simulator is None:
        _portfolio_simulator = PortfolioSimulator(initial_capital=initial_capital)
    
    return _portfolio_simulator


if __name__ == "__main__":
    # Test the portfolio simulator
    simulator = PortfolioSimulator(initial_capital=500_000)
    
    # Execute some test trades
    result1 = simulator.execute('NIFTY', ActionType.BUY, 19500.0, sentiment=0.3)
    print(f"Trade 1: {result1.success}, {result1.action.name} {result1.quantity} @ ₹{result1.price}")
    
    # Update market price
    simulator.update_market_prices({'NIFTY': 19550.0})
    
    result2 = simulator.execute('BANKNIFTY', ActionType.BUY, 45000.0, sentiment=0.5)
    print(f"Trade 2: {result2.success}, {result2.action.name} {result2.quantity} @ ₹{result2.price}")
    
    # Get portfolio summary
    summary = simulator.get_portfolio_summary()
    print(f"\nPortfolio Summary:")
    print(f"Total Value: ₹{summary['capital']['total_value']:,.0f}")
    print(f"P&L: ₹{summary['capital']['total_pnl']:,.0f} ({summary['capital']['pnl_percentage']:.2f}%)")
    print(f"Positions: {len(summary['positions'])}")
    print(f"Win Rate: {summary['performance']['win_rate']:.1%}")