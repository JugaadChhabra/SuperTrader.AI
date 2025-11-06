"""
Trade Ledger System for SuperTrader.AI

Comprehensive trade tracking system with real-time P&L calculation,
position lifecycle management, and transaction cost tracking.

Features:
- Complete trade lifecycle tracking (Entry → Fill → P&L → Exit)
- Real-time mark-to-market P&L calculation
- Multi-strategy position aggregation
- Transaction cost tracking (STT, brokerage, slippage)
- Performance attribution and analytics
- Risk metrics monitoring

Classes:
    TradeLedger: Main trade tracking and P&L system
    Trade: Individual trade representation
    Position: Aggregated position tracking
    TransactionCosts: Cost calculation utilities

Author: SuperTrader.AI Team
Version: 1.0.0
Last Updated: 2024-10-14
"""

import sqlite3
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import pandas as pd
import numpy as np
from threading import Lock
import uuid


# Enums for trade states and types
class TradeStatus(Enum):
    """Trade status enumeration"""
    PENDING = "PENDING"           # Order placed but not filled
    PARTIALLY_FILLED = "PARTIALLY_FILLED"  # Partial execution
    FILLED = "FILLED"             # Completely filled
    CANCELLED = "CANCELLED"       # Order cancelled
    REJECTED = "REJECTED"         # Order rejected
    EXPIRED = "EXPIRED"           # Order expired


class TradeAction(Enum):
    """Trade action enumeration"""
    BUY = "BUY"
    SELL = "SELL"


class PositionSide(Enum):
    """Position side enumeration"""
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


@dataclass
class TransactionCosts:
    """Comprehensive transaction cost breakdown for NSE Index Futures"""
    brokerage: float = 0.0
    stt: float = 0.0          # Securities Transaction Tax (sell side only)
    exchange_fees: float = 0.0 # NSE F&O transaction charges
    clearing_charges: float = 0.0  # Clearing corporation charges
    sebi_fees: float = 0.0    # SEBI turnover fees
    stamp_duty: float = 0.0   # Stamp duty (buy side only)
    gst: float = 0.0          # 18% GST on brokerage and charges
    rollover_costs: float = 0.0  # Rollover costs for contract expiry
    impact_costs: float = 0.0    # Market impact costs
    total_costs: float = 0.0
    
    def __post_init__(self):
        """Calculate total costs after initialization"""
        self.total_costs = (
            self.brokerage + self.stt + self.exchange_fees + 
            self.clearing_charges + self.sebi_fees + self.stamp_duty + 
            self.gst + self.rollover_costs + self.impact_costs
        )

@dataclass
class ContractDetails:
    """Futures contract specific details"""
    contract_symbol: str      # Full contract symbol (e.g., NIFTY25OCT24800)
    underlying_symbol: str    # Underlying index (e.g., NIFTY)
    expiry_date: str         # Contract expiry date
    lot_size: int            # Contract lot size
    tick_size: float         # Minimum tick size
    contract_month: str      # Contract month (e.g., OCT2024)
    contract_type: str       # 'futures' or 'options'
    strike_price: Optional[float] = None  # For options
    option_type: Optional[str] = None     # 'CE' or 'PE' for options
    
    def calculate_total(self) -> float:
        """Calculate total transaction cost"""
        self.total_cost = (
            self.brokerage + self.stt + self.exchange_charges + 
            self.sebi_charges + self.gst + self.stamp_duty
        )
        return self.total_cost


@dataclass
class Trade:
    """Enhanced trade record with comprehensive futures-specific details"""
    trade_id: str
    symbol: str
    side: TradeAction          # BUY or SELL
    quantity: int
    entry_price: float
    entry_time: datetime
    status: TradeStatus
    
    # Contract-specific details
    contract_details: ContractDetails = field(default_factory=lambda: ContractDetails("", "", "", 25, 0.05, "", "futures"))
    
    # Optional fields for complete trade lifecycle
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    # Enhanced P&L tracking
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    gross_pnl: float = 0.0           # P&L before costs
    net_pnl: float = 0.0             # P&L after all costs
    transaction_costs: TransactionCosts = field(default_factory=TransactionCosts)
    
    # Strategy and signal context
    strategy_name: str = "default"
    signal_strength: float = 0.0
    confidence_score: float = 0.0
    rl_signal: float = 0.0           # Original RL signal (-1 to +1)
    
    # Risk and performance metrics
    max_adverse_excursion: float = 0.0    # Largest loss during trade
    max_favorable_excursion: float = 0.0  # Largest profit during trade
    risk_reward_ratio: float = 0.0        # Actual risk/reward achieved
    holding_period_minutes: int = 0       # Trade duration in minutes
    
    # Market context at entry
    entry_volatility: float = 0.0         # Market volatility at entry
    entry_volume: int = 0                 # Volume at entry
    entry_open_interest: int = 0          # Open interest at entry
    
    # Index type classification
    index_type: str = "broad"             # broad, banking, sectoral, thematic
    market_cap_segment: str = "large"     # large, mid, small
    
    # Position sizing details
    position_size_method: str = "kelly"   # kelly, fixed, volatility_target
    kelly_fraction_used: float = 0.0
    volatility_target_used: float = 0.0
    leverage_used: float = 0.0
    margin_utilized: float = 0.0
    
    # Execution quality metrics
    slippage_bps: float = 0.0            # Slippage in basis points
    execution_speed_ms: int = 0          # Execution time in milliseconds
    market_impact_bps: float = 0.0       # Market impact in basis points
    
    # Rollover tracking
    rollover_count: int = 0              # Number of times rolled over
    rollover_costs: float = 0.0          # Total rollover costs
    original_expiry: Optional[str] = None # Original contract expiry
    
    # Execution details and metadata
    fill_details: Dict[str, Any] = field(default_factory=dict)
    market_conditions: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Enhanced post-initialization processing"""
        if isinstance(self.entry_time, str):
            self.entry_time = datetime.fromisoformat(self.entry_time)
        if isinstance(self.exit_time, str) and self.exit_time:
            self.exit_time = datetime.fromisoformat(self.exit_time)
            
        # Calculate holding period if exit time is available
        if self.exit_time and self.entry_time:
            self.holding_period_minutes = int((self.exit_time - self.entry_time).total_seconds() / 60)
            
        # Calculate net P&L after transaction costs
        self.net_pnl = self.gross_pnl - self.transaction_costs.total_costs


@dataclass
class Position:
    """Aggregated position tracking across multiple trades"""
    
    symbol: str = ""
    strategy: str = "ALL"          # Strategy name or "ALL" for aggregated
    
    # Position details
    quantity: int = 0              # Net quantity (+ for long, - for short)
    avg_price: float = 0.0         # Average entry price
    side: PositionSide = PositionSide.FLAT
    
    # Financial metrics
    market_value: float = 0.0      # Current market value
    unrealized_pnl: float = 0.0    # Mark-to-market P&L
    realized_pnl: float = 0.0      # Realized P&L from closed trades
    total_pnl: float = 0.0         # Total P&L (realized + unrealized)
    
    # Risk metrics
    margin_used: float = 0.0       # Total margin blocked
    exposure: float = 0.0          # Total notional exposure
    
    # Trade tracking
    entry_trades: List[str] = field(default_factory=list)  # List of entry trade IDs
    exit_trades: List[str] = field(default_factory=list)   # List of exit trade IDs
    
    # Timing
    first_entry_time: Optional[datetime] = None
    last_update_time: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Post-initialization processing"""
        self.update_side()
        self.calculate_market_value()
    
    def update_side(self) -> None:
        """Update position side based on quantity"""
        if self.quantity > 0:
            self.side = PositionSide.LONG
        elif self.quantity < 0:
            self.side = PositionSide.SHORT
        else:
            self.side = PositionSide.FLAT
    
    def add_trade(self, trade: Trade, current_price: float = None) -> None:
        """Add a trade to the position"""
        if trade.quantity_filled == 0:
            return
        
        trade_quantity = trade.quantity_filled
        trade_price = trade.price_filled
        
        # Adjust quantity based on trade action
        if trade.action == TradeAction.SELL:
            trade_quantity = -trade_quantity
        
        if trade.is_entry:
            self.entry_trades.append(trade.trade_id)
        else:
            self.exit_trades.append(trade.trade_id)
        
        # Update position
        if self.quantity == 0:
            # New position
            self.quantity = trade_quantity
            self.avg_price = trade_price
            self.first_entry_time = trade.fill_time or trade.order_time
        else:
            if (self.quantity > 0 and trade_quantity > 0) or (self.quantity < 0 and trade_quantity < 0):
                # Adding to existing position
                total_value = (abs(self.quantity) * self.avg_price) + (abs(trade_quantity) * trade_price)
                total_quantity = abs(self.quantity) + abs(trade_quantity)
                self.avg_price = total_value / total_quantity
                self.quantity += trade_quantity
            else:
                # Reducing or reversing position
                if abs(trade_quantity) >= abs(self.quantity):
                    # Position reversal or closure
                    self.quantity += trade_quantity
                    if self.quantity != 0:
                        # Reversed position
                        self.avg_price = trade_price
                else:
                    # Partial reduction
                    self.quantity += trade_quantity
        
        # Update margin and exposure
        self.margin_used += trade.margin_used
        
        # Update timing
        self.last_update_time = datetime.now()
        
        # Update side and market value
        self.update_side()
        if current_price:
            self.calculate_market_value(current_price)
    
    def calculate_market_value(self, current_price: float = None) -> float:
        """Calculate current market value of the position"""
        if current_price is None:
            current_price = self.avg_price
        
        self.market_value = abs(self.quantity) * current_price
        return self.market_value
    
    def calculate_pnl(self, current_price: float) -> Tuple[float, float, float]:
        """Calculate P&L metrics"""
        if self.quantity == 0:
            self.unrealized_pnl = 0.0
        else:
            if self.quantity > 0:
                # Long position
                pnl_per_unit = current_price - self.avg_price
            else:
                # Short position
                pnl_per_unit = self.avg_price - current_price
            
            self.unrealized_pnl = pnl_per_unit * abs(self.quantity)
        
        self.total_pnl = self.realized_pnl + self.unrealized_pnl
        return self.unrealized_pnl, self.realized_pnl, self.total_pnl


class TradeLedger:
    """
    Comprehensive trade tracking system with real-time P&L calculation
    and position lifecycle management
    """
    
    def __init__(self, database_path: str = None):
        """
        Initialize Trade Ledger system
        
        Args:
            database_path: Path to SQLite database file
        """
        self.logger = logging.getLogger(__name__)
        
        # Database setup
        if database_path is None:
            database_path = "data/trade_ledger.db"
        
        self.db_path = Path(database_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Thread safety
        self.lock = Lock()
        
        # In-memory tracking
        self.trades: Dict[str, Trade] = {}          # trade_id -> Trade
        self.positions: Dict[str, Position] = {}    # symbol -> Position
        self.strategy_positions: Dict[Tuple[str, str], Position] = {}  # (symbol, strategy) -> Position
        
        # Performance tracking
        self.daily_pnl = 0.0
        self.total_realized_pnl = 0.0
        self.total_unrealized_pnl = 0.0
        self.total_transaction_costs = 0.0
        
        # Initialize database
        self.init_database()
        
        # Load existing data
        self.load_todays_data()
        
        self.logger.info(f"TradeLedger initialized with database: {self.db_path}")
    
    def init_database(self) -> None:
        """Initialize SQLite database with required tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create trades table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS trades (
                        trade_id TEXT PRIMARY KEY,
                        order_id TEXT,
                        broker_order_id TEXT,
                        symbol TEXT NOT NULL,
                        action TEXT NOT NULL,
                        strategy TEXT,
                        quantity_ordered INTEGER,
                        quantity_filled INTEGER,
                        price_ordered REAL,
                        price_filled REAL,
                        order_time TEXT,
                        fill_time TEXT,
                        exit_time TEXT,
                        status TEXT,
                        is_entry BOOLEAN,
                        parent_trade_id TEXT,
                        notional_value REAL,
                        margin_used REAL,
                        unrealized_pnl REAL,
                        realized_pnl REAL,
                        confidence_score REAL,
                        risk_score REAL,
                        transaction_costs TEXT,  -- JSON
                        metadata TEXT,           -- JSON
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create positions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS positions (
                        position_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        strategy TEXT DEFAULT 'ALL',
                        quantity INTEGER,
                        avg_price REAL,
                        side TEXT,
                        market_value REAL,
                        unrealized_pnl REAL,
                        realized_pnl REAL,
                        margin_used REAL,
                        exposure REAL,
                        entry_trades TEXT,  -- JSON list
                        exit_trades TEXT,   -- JSON list
                        first_entry_time TEXT,
                        last_update_time TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(symbol, strategy)
                    )
                """)
                
                # Create daily_performance table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS daily_performance (
                        date TEXT PRIMARY KEY,
                        total_pnl REAL,
                        realized_pnl REAL,
                        unrealized_pnl REAL,
                        transaction_costs REAL,
                        num_trades INTEGER,
                        winning_trades INTEGER,
                        losing_trades INTEGER,
                        largest_win REAL,
                        largest_loss REAL,
                        avg_trade_pnl REAL,
                        win_rate REAL,
                        profit_factor REAL,
                        sharpe_ratio REAL,
                        max_drawdown REAL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indices for better performance
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_order_time ON trades(order_time)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol)")
                
                conn.commit()
                self.logger.info("Database schema initialized successfully")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise
    
    def record_trade(self, trade: Trade) -> bool:
        """
        Record a new trade in the ledger
        
        Args:
            trade: Trade object to record
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with self.lock:
                # Add to in-memory tracking
                self.trades[trade.trade_id] = trade
                
                # Calculate transaction costs if not already calculated
                if trade.transaction_costs.total_cost == 0:
                    self.calculate_transaction_costs(trade)
                
                # Update positions
                self._update_positions(trade)
                
                # Persist to database
                self._save_trade_to_db(trade)
                
                self.logger.info(f"Trade recorded: {trade.trade_id} - {trade.action.value} {trade.quantity_filled} {trade.symbol}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to record trade {trade.trade_id}: {e}")
            return False
    
    def update_trade_fill(self, trade_id: str, filled_qty: int, fill_price: float, 
                         fill_time: datetime = None) -> bool:
        """
        Update trade with fill information
        
        Args:
            trade_id: ID of trade to update
            filled_qty: Quantity filled
            fill_price: Fill price
            fill_time: Fill timestamp
            
        Returns:
            bool: True if successful
        """
        try:
            with self.lock:
                if trade_id not in self.trades:
                    self.logger.warning(f"Trade {trade_id} not found for fill update")
                    return False
                
                trade = self.trades[trade_id]
                old_filled = trade.quantity_filled
                
                # Update the trade
                trade.update_fill(filled_qty, fill_price, fill_time)
                
                # Recalculate transaction costs
                self.calculate_transaction_costs(trade)
                
                # Update positions if additional quantity was filled
                if filled_qty > old_filled:
                    self._update_positions(trade)
                
                # Update in database
                self._save_trade_to_db(trade)
                
                self.logger.info(f"Trade fill updated: {trade_id} - {filled_qty} @ {fill_price}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to update trade fill {trade_id}: {e}")
            return False
    
    def calculate_transaction_costs(self, trade: Trade) -> TransactionCosts:
        """
        Calculate comprehensive transaction costs for a trade
        
        Args:
            trade: Trade object to calculate costs for
            
        Returns:
            TransactionCosts: Calculated transaction costs
        """
        if trade.quantity_filled == 0 or trade.price_filled == 0:
            return trade.transaction_costs
        
        notional = abs(trade.quantity_filled * trade.price_filled)
        
        # Brokerage (flat ₹20 per order)
        trade.transaction_costs.brokerage = 20.0
        
        # STT (asymmetric - only on SELL side for futures)
        if trade.action == TradeAction.SELL:
            trade.transaction_costs.stt = notional * 0.000125  # 0.0125%
        else:
            trade.transaction_costs.stt = 0.0
        
        # Exchange charges (0.0019% of turnover)
        trade.transaction_costs.exchange_charges = notional * 0.000019
        
        # SEBI charges (₹10 per crore)
        trade.transaction_costs.sebi_charges = (notional / 10000000) * 10
        
        # Stamp duty (0.002% on BUY side)
        if trade.action == TradeAction.BUY:
            trade.transaction_costs.stamp_duty = notional * 0.00002
        else:
            trade.transaction_costs.stamp_duty = 0.0
        
        # GST (18% on brokerage + exchange charges)
        gst_base = trade.transaction_costs.brokerage + trade.transaction_costs.exchange_charges
        trade.transaction_costs.gst = gst_base * 0.18
        
        # Calculate total
        trade.transaction_costs.calculate_total()
        
        return trade.transaction_costs
    
    def _update_positions(self, trade: Trade, current_price: float = None) -> None:
        """
        Update position tracking with new trade
        
        Args:
            trade: Trade to add to positions
            current_price: Current market price for P&L calculation
        """
        if trade.quantity_filled == 0:
            return
        
        symbol = trade.symbol
        strategy = trade.strategy or "DEFAULT"
        
        # Update aggregate position
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol, strategy="ALL")
        
        self.positions[symbol].add_trade(trade, current_price or trade.price_filled)
        
        # Update strategy-specific position
        strategy_key = (symbol, strategy)
        if strategy_key not in self.strategy_positions:
            self.strategy_positions[strategy_key] = Position(symbol=symbol, strategy=strategy)
        
        self.strategy_positions[strategy_key].add_trade(trade, current_price or trade.price_filled)
        
        # Save positions to database
        self._save_position_to_db(self.positions[symbol])
        self._save_position_to_db(self.strategy_positions[strategy_key])
    
    def update_market_prices(self, price_data: Dict[str, float]) -> None:
        """
        Update all positions with current market prices for P&L calculation
        
        Args:
            price_data: Dictionary of symbol -> current_price
        """
        try:
            with self.lock:
                total_unrealized = 0.0
                
                # Update position P&L
                for symbol, position in self.positions.items():
                    if symbol in price_data:
                        current_price = price_data[symbol]
                        unrealized_pnl, realized_pnl, total_pnl = position.calculate_pnl(current_price)
                        total_unrealized += unrealized_pnl
                        
                        # Update position in database
                        self._save_position_to_db(position)
                
                # Update strategy positions
                for (symbol, strategy), position in self.strategy_positions.items():
                    if symbol in price_data:
                        current_price = price_data[symbol]
                        position.calculate_pnl(current_price)
                        self._save_position_to_db(position)
                
                # Update individual trade P&L
                for trade in self.trades.values():
                    if trade.symbol in price_data and trade.quantity_filled > 0:
                        current_price = price_data[trade.symbol]
                        trade.calculate_unrealized_pnl(current_price)
                
                self.total_unrealized_pnl = total_unrealized
                
                self.logger.debug(f"Updated market prices for {len(price_data)} symbols")
                
        except Exception as e:
            self.logger.error(f"Failed to update market prices: {e}")
    
    def get_position(self, symbol: str, strategy: str = None) -> Optional[Position]:
        """
        Get position for a symbol and optional strategy
        
        Args:
            symbol: Symbol to get position for
            strategy: Optional strategy filter
            
        Returns:
            Position object or None if not found
        """
        if strategy is None:
            return self.positions.get(symbol)
        else:
            return self.strategy_positions.get((symbol, strategy))
    
    def get_all_positions(self, include_flat: bool = False) -> List[Position]:
        """
        Get all current positions
        
        Args:
            include_flat: Include flat positions (quantity = 0)
            
        Returns:
            List of Position objects
        """
        positions = list(self.positions.values())
        
        if not include_flat:
            positions = [p for p in positions if p.quantity != 0]
        
        return positions
    
    def get_daily_pnl(self, date: str = None) -> Dict[str, float]:
        """
        Get daily P&L summary
        
        Args:
            date: Date in YYYY-MM-DD format (defaults to today)
            
        Returns:
            Dictionary with P&L metrics
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            total_realized = sum(trade.realized_pnl for trade in self.trades.values() 
                               if trade.fill_time and trade.fill_time.strftime('%Y-%m-%d') == date)
            
            total_unrealized = sum(position.unrealized_pnl for position in self.positions.values())
            
            total_costs = sum(trade.transaction_costs.total_cost for trade in self.trades.values() 
                            if trade.fill_time and trade.fill_time.strftime('%Y-%m-%d') == date)
            
            num_trades = len([t for t in self.trades.values() 
                            if t.fill_time and t.fill_time.strftime('%Y-%m-%d') == date and t.status == TradeStatus.FILLED])
            
            return {
                'date': date,
                'realized_pnl': total_realized,
                'unrealized_pnl': total_unrealized,
                'total_pnl': total_realized + total_unrealized,
                'transaction_costs': total_costs,
                'net_pnl': total_realized + total_unrealized - total_costs,
                'num_trades': num_trades
            }
            
        except Exception as e:
            self.logger.error(f"Failed to calculate daily P&L: {e}")
            return {}
    
    def _save_trade_to_db(self, trade: Trade) -> None:
        """Save trade to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                trade_data = (
                    trade.trade_id, trade.order_id, trade.broker_order_id,
                    trade.symbol, trade.action.value, trade.strategy,
                    trade.quantity_ordered, trade.quantity_filled,
                    trade.price_ordered, trade.price_filled,
                    trade.order_time.isoformat() if trade.order_time else None,
                    trade.fill_time.isoformat() if trade.fill_time else None,
                    trade.exit_time.isoformat() if trade.exit_time else None,
                    trade.status.value, trade.is_entry, trade.parent_trade_id,
                    trade.notional_value, trade.margin_used,
                    trade.unrealized_pnl, trade.realized_pnl,
                    trade.confidence_score, trade.risk_score,
                    json.dumps(asdict(trade.transaction_costs)),
                    json.dumps(trade.metadata),
                    datetime.now().isoformat()
                )
                
                cursor.execute("""
                    INSERT OR REPLACE INTO trades (
                        trade_id, order_id, broker_order_id, symbol, action, strategy,
                        quantity_ordered, quantity_filled, price_ordered, price_filled,
                        order_time, fill_time, exit_time, status, is_entry, parent_trade_id,
                        notional_value, margin_used, unrealized_pnl, realized_pnl,
                        confidence_score, risk_score, transaction_costs, metadata, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, trade_data)
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to save trade to database: {e}")
    
    def _save_position_to_db(self, position: Position) -> None:
        """Save position to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                position_data = (
                    position.symbol, position.strategy, position.quantity, position.avg_price,
                    position.side.value, position.market_value, position.unrealized_pnl,
                    position.realized_pnl, position.margin_used, position.exposure,
                    json.dumps(position.entry_trades), json.dumps(position.exit_trades),
                    position.first_entry_time.isoformat() if position.first_entry_time else None,
                    position.last_update_time.isoformat()
                )
                
                cursor.execute("""
                    INSERT OR REPLACE INTO positions (
                        symbol, strategy, quantity, avg_price, side, market_value,
                        unrealized_pnl, realized_pnl, margin_used, exposure,
                        entry_trades, exit_trades, first_entry_time, last_update_time
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, position_data)
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to save position to database: {e}")
    
    def load_todays_data(self) -> None:
        """Load today's trades and positions from database"""
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Load today's trades
                cursor.execute("""
                    SELECT * FROM trades 
                    WHERE date(order_time) = ? OR date(fill_time) = ?
                    ORDER BY order_time
                """, (today, today))
                
                for row in cursor.fetchall():
                    # Reconstruct Trade object from database row
                    # This would need proper deserialization logic
                    pass  # Implementation details omitted for brevity
                
                self.logger.info(f"Loaded today's trading data from database")
                
        except Exception as e:
            self.logger.error(f"Failed to load today's data: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        try:
            positions = self.get_all_positions()
            
            return {
                'total_positions': len(positions),
                'long_positions': len([p for p in positions if p.side == PositionSide.LONG]),
                'short_positions': len([p for p in positions if p.side == PositionSide.SHORT]),
                'total_exposure': sum(p.exposure for p in positions),
                'total_margin_used': sum(p.margin_used for p in positions),
                'total_unrealized_pnl': sum(p.unrealized_pnl for p in positions),
                'total_realized_pnl': sum(trade.realized_pnl for trade in self.trades.values()),
                'total_transaction_costs': sum(trade.transaction_costs.total_cost for trade in self.trades.values()),
                'num_trades_today': len([t for t in self.trades.values() if t.fill_time and t.fill_time.date() == datetime.now().date()]),
                'largest_position': max(positions, key=lambda p: abs(p.market_value)) if positions else None
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get performance summary: {e}")
            return {}
    
    def calculate_enhanced_transaction_costs(
        self, 
        symbol: str, 
        quantity: int, 
        price: float, 
        side: str,
        contract_details: Optional[ContractDetails] = None
    ) -> TransactionCosts:
        """
        Calculate comprehensive transaction costs for NSE Index Futures
        Based on current NSE fee structure (October 2024)
        """
        try:
            turnover = quantity * price
            costs = TransactionCosts()
            
            # Load risk config for fee structure
            from utils.config import load_config
            risk_config = load_config("configs/risk.yaml")
            fee_config = risk_config.get('enhanced_fee_structure', {})
            
            # 1. Brokerage: ₹20 per order or 0.03% whichever is lower
            brokerage_config = fee_config.get('brokerage', {})
            brokerage_flat = brokerage_config.get('flat_rate', 20.0)
            brokerage_percent = turnover * brokerage_config.get('percentage_rate', 0.0003)
            costs.brokerage = min(brokerage_flat, brokerage_percent)
            
            # 2. STT: 0.01% on sell side only for futures
            if side.upper() == 'SELL':
                stt_config = fee_config.get('stt', {})
                costs.stt = turnover * stt_config.get('rate', 0.0001)
            
            # 3. Exchange fees: NSE F&O transaction charges
            exchange_config = fee_config.get('exchange_charges', {})
            costs.exchange_fees = turnover * exchange_config.get('nse_rate', 0.0000345)
            
            # 4. Clearing charges
            clearing_config = fee_config.get('clearing_charges', {})
            costs.clearing_charges = turnover * clearing_config.get('rate', 0.0000055)
            
            # 5. SEBI fees: ₹1 per crore turnover
            sebi_config = fee_config.get('sebi_charges', {})
            costs.sebi_fees = turnover * sebi_config.get('rate', 0.000001)
            
            # 6. Stamp duty: 0.003% on buy side only
            if side.upper() == 'BUY':
                stamp_config = fee_config.get('stamp_duty', {})
                costs.stamp_duty = turnover * stamp_config.get('rate', 0.00003)
            
            # 7. GST: 18% on brokerage and charges (not on STT and stamp duty)
            gst_config = fee_config.get('gst', {})
            gst_rate = gst_config.get('rate', 0.18)
            taxable_amount = (costs.brokerage + costs.exchange_fees + 
                            costs.clearing_charges + costs.sebi_fees)
            costs.gst = taxable_amount * gst_rate
            
            # 8. Estimate impact costs (for large orders)
            costs.impact_costs = self._calculate_impact_costs(symbol, quantity, turnover)
            
            # 9. Rollover costs (if applicable)
            if contract_details and self._is_near_expiry(contract_details.expiry_date):
                costs.rollover_costs = self._estimate_rollover_costs(symbol, turnover)
            
            # Update total costs
            costs.__post_init__()
            
            return costs
            
        except Exception as e:
            self.logger.error(f"Error calculating enhanced transaction costs: {e}")
            return TransactionCosts()
    
    def _calculate_impact_costs(self, symbol: str, quantity: int, turnover: float) -> float:
        """Calculate market impact costs for large orders"""
        try:
            # Extract index name for impact cost estimation
            index_name = self._extract_index_name(symbol)
            
            # Typical daily volumes (approximate)
            typical_volumes = {
                'NIFTY': 500000,      # ~5 lakh lots daily
                'BANKNIFTY': 300000,  # ~3 lakh lots daily
                'FINNIFTY': 100000    # ~1 lakh lots daily
            }
            
            daily_volume = typical_volumes.get(index_name, 50000)
            
            # Calculate order size as percentage of daily volume
            # Assuming lot size for calculation
            lot_sizes = {'NIFTY': 75, 'BANKNIFTY': 15, 'FINNIFTY': 25}
            lot_size = lot_sizes.get(index_name, 25)
            lots_traded = quantity // lot_size
            
            volume_pct = lots_traded / daily_volume
            
            # Impact cost model: square root of volume percentage
            # Typical impact: 2-5 bps for normal orders, higher for large orders
            if volume_pct < 0.001:  # <0.1% of daily volume
                impact_bps = 1  # 1 basis point
            elif volume_pct < 0.005:  # <0.5% of daily volume
                impact_bps = 3  # 3 basis points
            elif volume_pct < 0.01:  # <1% of daily volume
                impact_bps = 8  # 8 basis points
            else:  # >1% of daily volume
                impact_bps = 15  # 15 basis points
            
            return turnover * (impact_bps / 10000)  # Convert bps to decimal
            
        except Exception as e:
            self.logger.error(f"Error calculating impact costs: {e}")
            return 0.0
    
    def _is_near_expiry(self, expiry_date: str) -> bool:
        """Check if contract is near expiry (within 5 days)"""
        try:
            if not expiry_date:
                return False
                
            expiry = datetime.strptime(expiry_date, '%Y-%m-%d').date()
            today = datetime.now().date()
            days_to_expiry = (expiry - today).days
            
            return days_to_expiry <= 5
            
        except Exception as e:
            self.logger.error(f"Error checking expiry proximity: {e}")
            return False
    
    def _estimate_rollover_costs(self, symbol: str, turnover: float) -> float:
        """Estimate rollover costs for near-expiry contracts"""
        try:
            from utils.config import load_config
            risk_config = load_config("configs/risk.yaml")
            
            # Get rollover cost configuration
            rollover_config = risk_config.get('enhanced_fee_structure', {}).get('rollover_costs', {})
            
            # Basis cost: difference between near month and next month
            basis_cost_rate = rollover_config.get('basis_cost_estimate', 0.0005)  # 0.05%
            
            # Additional impact cost for rollover
            impact_cost_rate = rollover_config.get('impact_cost', 0.0002)  # 0.02%
            
            total_rollover_rate = basis_cost_rate + impact_cost_rate
            
            return turnover * total_rollover_rate
            
        except Exception as e:
            self.logger.error(f"Error estimating rollover costs: {e}")
            return 0.0
    
    def _extract_index_name(self, symbol: str) -> str:
        """Extract index name from futures symbol"""
        symbol_upper = symbol.upper()
        
        if 'BANKNIFTY' in symbol_upper or 'BANK NIFTY' in symbol_upper:
            return 'BANKNIFTY'
        elif 'FINNIFTY' in symbol_upper or 'FIN NIFTY' in symbol_upper:
            return 'FINNIFTY'
        elif 'NIFTY' in symbol_upper:
            return 'NIFTY'
        else:
            return 'OTHER'
    
    def get_trade_attribution_analysis(self, days: int = 30) -> Dict[str, Any]:
        """
        Get comprehensive trade attribution analysis
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dictionary with attribution metrics
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_trades = [
                trade for trade in self.trades.values() 
                if trade.fill_time and trade.fill_time >= cutoff_date
            ]
            
            if not recent_trades:
                return {'error': 'No trades found in the specified period'}
            
            # Index type analysis
            index_analysis = self._analyze_by_index_type(recent_trades)
            
            # Long vs Short analysis
            direction_analysis = self._analyze_by_direction(recent_trades)
            
            # Strategy analysis
            strategy_analysis = self._analyze_by_strategy(recent_trades)
            
            # Time-based analysis
            time_analysis = self._analyze_by_time(recent_trades)
            
            # Performance metrics
            performance_metrics = self._calculate_performance_metrics(recent_trades)
            
            return {
                'analysis_period': f'{days} days',
                'total_trades': len(recent_trades),
                'index_analysis': index_analysis,
                'direction_analysis': direction_analysis,
                'strategy_analysis': strategy_analysis,
                'time_analysis': time_analysis,
                'performance_metrics': performance_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Error in trade attribution analysis: {e}")
            return {'error': str(e)}
    
    def _analyze_by_index_type(self, trades: List[Trade]) -> Dict[str, Any]:
        """Analyze performance by index type"""
        index_stats = {}
        
        for trade in trades:
            index_type = getattr(trade, 'index_type', 'broad')  # Use enhanced Trade structure
            
            if index_type not in index_stats:
                index_stats[index_type] = {
                    'trade_count': 0,
                    'total_pnl': 0.0,
                    'total_costs': 0.0,
                    'winners': 0,
                    'losers': 0
                }
            
            stats = index_stats[index_type]
            stats['trade_count'] += 1
            stats['total_pnl'] += trade.net_pnl
            stats['total_costs'] += trade.transaction_costs.total_costs
            
            if trade.net_pnl > 0:
                stats['winners'] += 1
            elif trade.net_pnl < 0:
                stats['losers'] += 1
        
        # Calculate metrics
        for index_type, stats in index_stats.items():
            total_trades = stats['trade_count']
            stats['win_rate'] = stats['winners'] / total_trades if total_trades > 0 else 0
            stats['avg_pnl_per_trade'] = stats['total_pnl'] / total_trades if total_trades > 0 else 0
            stats['cost_ratio'] = stats['total_costs'] / abs(stats['total_pnl']) if stats['total_pnl'] != 0 else float('inf')
        
        return index_stats
    
    def _analyze_by_direction(self, trades: List[Trade]) -> Dict[str, Any]:
        """Analyze performance by long vs short"""
        direction_stats = {'long': {'trade_count': 0, 'total_pnl': 0.0, 'winners': 0, 'losers': 0},
                          'short': {'trade_count': 0, 'total_pnl': 0.0, 'winners': 0, 'losers': 0}}
        
        for trade in trades:
            direction = 'long' if trade.side == TradeAction.BUY else 'short'
            
            stats = direction_stats[direction]
            stats['trade_count'] += 1
            stats['total_pnl'] += trade.net_pnl
            
            if trade.net_pnl > 0:
                stats['winners'] += 1
            elif trade.net_pnl < 0:
                stats['losers'] += 1
        
        # Calculate metrics
        for direction, stats in direction_stats.items():
            total_trades = stats['trade_count']
            stats['win_rate'] = stats['winners'] / total_trades if total_trades > 0 else 0
            stats['avg_pnl_per_trade'] = stats['total_pnl'] / total_trades if total_trades > 0 else 0
        
        return direction_stats
    
    def _analyze_by_strategy(self, trades: List[Trade]) -> Dict[str, Any]:
        """Analyze performance by strategy"""
        strategy_stats = {}
        
        for trade in trades:
            strategy = trade.strategy_name
            
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {
                    'trade_count': 0,
                    'total_pnl': 0.0,
                    'winners': 0,
                    'losers': 0,
                    'avg_confidence': 0.0,
                    'total_leverage': 0.0
                }
            
            stats = strategy_stats[strategy]
            stats['trade_count'] += 1
            stats['total_pnl'] += trade.net_pnl
            stats['avg_confidence'] += trade.confidence_score
            stats['total_leverage'] += getattr(trade, 'leverage_used', 0.0)
            
            if trade.net_pnl > 0:
                stats['winners'] += 1
            elif trade.net_pnl < 0:
                stats['losers'] += 1
        
        # Calculate averages
        for strategy, stats in strategy_stats.items():
            total_trades = stats['trade_count']
            if total_trades > 0:
                stats['win_rate'] = stats['winners'] / total_trades
                stats['avg_pnl_per_trade'] = stats['total_pnl'] / total_trades
                stats['avg_confidence'] = stats['avg_confidence'] / total_trades
                stats['avg_leverage'] = stats['total_leverage'] / total_trades
        
        return strategy_stats
    
    def _analyze_by_time(self, trades: List[Trade]) -> Dict[str, Any]:
        """Analyze performance by time periods"""
        time_stats = {
            'hourly': {},
            'daily': {},
            'holding_period': {'0-1h': 0, '1-3h': 0, '3-6h': 0, '6h+': 0}
        }
        
        for trade in trades:
            if not trade.entry_time:
                continue
                
            # Hourly analysis
            hour = trade.entry_time.hour
            if hour not in time_stats['hourly']:
                time_stats['hourly'][hour] = {'trade_count': 0, 'total_pnl': 0.0}
            time_stats['hourly'][hour]['trade_count'] += 1
            time_stats['hourly'][hour]['total_pnl'] += trade.net_pnl
            
            # Daily analysis
            day = trade.entry_time.strftime('%Y-%m-%d')
            if day not in time_stats['daily']:
                time_stats['daily'][day] = {'trade_count': 0, 'total_pnl': 0.0}
            time_stats['daily'][day]['trade_count'] += 1
            time_stats['daily'][day]['total_pnl'] += trade.net_pnl
            
            # Holding period analysis
            holding_minutes = getattr(trade, 'holding_period_minutes', 0)
            if holding_minutes <= 60:
                time_stats['holding_period']['0-1h'] += 1
            elif holding_minutes <= 180:
                time_stats['holding_period']['1-3h'] += 1
            elif holding_minutes <= 360:
                time_stats['holding_period']['3-6h'] += 1
            else:
                time_stats['holding_period']['6h+'] += 1
        
        return time_stats
    
    def _calculate_performance_metrics(self, trades: List[Trade]) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        if not trades:
            return {}
        
        # Basic metrics
        total_pnl = sum(trade.net_pnl for trade in trades)
        total_costs = sum(trade.transaction_costs.total_costs for trade in trades)
        winners = [trade for trade in trades if trade.net_pnl > 0]
        losers = [trade for trade in trades if trade.net_pnl < 0]
        
        win_rate = len(winners) / len(trades) if trades else 0
        
        # Risk-reward metrics
        avg_winner = np.mean([trade.net_pnl for trade in winners]) if winners else 0
        avg_loser = np.mean([trade.net_pnl for trade in losers]) if losers else 0
        profit_factor = abs(avg_winner / avg_loser) if avg_loser != 0 else float('inf')
        
        # Cost analysis
        cost_ratio = total_costs / abs(total_pnl) if total_pnl != 0 else float('inf')
        avg_cost_per_trade = total_costs / len(trades) if trades else 0
        
        return {
            'total_trades': len(trades),
            'winners': len(winners),
            'losers': len(losers),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl_per_trade': total_pnl / len(trades),
            'avg_winner': avg_winner,
            'avg_loser': avg_loser,
            'profit_factor': profit_factor,
            'total_costs': total_costs,
            'cost_ratio': cost_ratio,
            'avg_cost_per_trade': avg_cost_per_trade,
            'net_profit_after_costs': total_pnl - total_costs
        }


# Global instance (singleton pattern)
_trade_ledger = None
_ledger_lock = Lock()


def get_trade_ledger(database_path: str = None) -> TradeLedger:
    """
    Get the global TradeLedger instance (singleton)
    
    Args:
        database_path: Path to database file (only used on first call)
        
    Returns:
        TradeLedger instance
    """
    global _trade_ledger
    
    with _ledger_lock:
        if _trade_ledger is None:
            _trade_ledger = TradeLedger(database_path)
    
    return _trade_ledger


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize trade ledger
    ledger = TradeLedger("test_trade_ledger.db")
    
    # Create a test trade
    test_trade = Trade(
        symbol="NIFTY",
        action=TradeAction.BUY,
        strategy="momentum_breakout",
        quantity_ordered=100,
        price_ordered=19500.0,
        confidence_score=0.75,
        risk_score=0.25
    )
    
    # Simulate trade fill
    test_trade.update_fill(100, 19505.0)
    
    # Record the trade
    success = ledger.record_trade(test_trade)
    print(f"Trade recorded: {success}")
    
    # Update market prices
    ledger.update_market_prices({"NIFTY": 19550.0})
    
    # Get position
    position = ledger.get_position("NIFTY")
    if position:
        print(f"Position: {position.quantity} @ {position.avg_price:.2f}, P&L: ₹{position.unrealized_pnl:.2f}")
    
    # Get daily P&L
    daily_pnl = ledger.get_daily_pnl()
    print(f"Daily P&L: {daily_pnl}")
    
    # Performance summary
    summary = ledger.get_performance_summary()
    print(f"Performance Summary: {summary}")