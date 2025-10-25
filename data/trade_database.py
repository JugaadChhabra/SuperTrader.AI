"""
Trade Database Management for SuperTrader.AI

Enhanced database utilities for efficient trade data storage, querying,
and analytics with SQLite backend and performance optimizations.

Features:
- Optimized database schema with indices
- Efficient bulk operations and batch processing
- Advanced querying capabilities with filters
- Data export and backup utilities
- Performance analytics queries
- Database maintenance and optimization

Classes:
    TradeDatabase: Enhanced database management
    QueryBuilder: SQL query construction utilities
    DataExporter: Export utilities for analysis
    DatabaseMaintenance: Optimization and cleanup

Author: SuperTrader.AI Team
Version: 1.0.0
Last Updated: 2024-10-14
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import json
import logging
import shutil
from contextlib import contextmanager
import threading
from dataclasses import asdict

# Import trade ledger types
import sys
sys.path.append(str(Path(__file__).parent.parent))
from agents.trade_ledger import Trade, Position, TradeStatus, TradeAction, PositionSide


class QueryBuilder:
    """SQL query construction utilities"""
    
    @staticmethod
    def build_trade_filter(
        symbols: List[str] = None,
        strategies: List[str] = None,
        start_date: datetime = None,
        end_date: datetime = None,
        statuses: List[TradeStatus] = None,
        min_pnl: float = None,
        max_pnl: float = None
    ) -> Tuple[str, List[Any]]:
        """
        Build WHERE clause for trade filtering
        
        Returns:
            Tuple of (where_clause, parameters)
        """
        conditions = []
        params = []
        
        if symbols:
            placeholders = ','.join('?' * len(symbols))
            conditions.append(f"symbol IN ({placeholders})")
            params.extend(symbols)
        
        if strategies:
            placeholders = ','.join('?' * len(strategies))
            conditions.append(f"strategy IN ({placeholders})")
            params.extend(strategies)
        
        if start_date:
            conditions.append("datetime(order_time) >= ?")
            params.append(start_date.isoformat())
        
        if end_date:
            conditions.append("datetime(order_time) <= ?")
            params.append(end_date.isoformat())
        
        if statuses:
            placeholders = ','.join('?' * len(statuses))
            conditions.append(f"status IN ({placeholders})")
            params.extend([status.value for status in statuses])
        
        if min_pnl is not None:
            conditions.append("(realized_pnl + unrealized_pnl) >= ?")
            params.append(min_pnl)
        
        if max_pnl is not None:
            conditions.append("(realized_pnl + unrealized_pnl) <= ?")
            params.append(max_pnl)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        return where_clause, params
    
    @staticmethod
    def build_performance_query(
        group_by: str = "date",
        start_date: datetime = None,
        end_date: datetime = None
    ) -> Tuple[str, List[Any]]:
        """
        Build performance aggregation query
        
        Args:
            group_by: Grouping field ('date', 'symbol', 'strategy', 'hour')
        """
        params = []
        
        if group_by == "date":
            select_fields = "date(order_time) as period"
        elif group_by == "symbol":
            select_fields = "symbol as period"
        elif group_by == "strategy":
            select_fields = "strategy as period"
        elif group_by == "hour":
            select_fields = "strftime('%Y-%m-%d %H:00', order_time) as period"
        else:
            select_fields = "date(order_time) as period"
        
        query = f"""
        SELECT 
            {select_fields},
            COUNT(*) as num_trades,
            SUM(CASE WHEN (realized_pnl + unrealized_pnl) > 0 THEN 1 ELSE 0 END) as winning_trades,
            SUM(CASE WHEN (realized_pnl + unrealized_pnl) < 0 THEN 1 ELSE 0 END) as losing_trades,
            SUM(realized_pnl + unrealized_pnl) as total_pnl,
            AVG(realized_pnl + unrealized_pnl) as avg_pnl,
            MAX(realized_pnl + unrealized_pnl) as max_win,
            MIN(realized_pnl + unrealized_pnl) as max_loss,
            SUM(notional_value) as total_volume,
            SUM(JSON_EXTRACT(transaction_costs, '$.total_cost')) as total_costs
        FROM trades
        WHERE status = 'FILLED'
        """
        
        if start_date:
            query += " AND datetime(order_time) >= ?"
            params.append(start_date.isoformat())
        
        if end_date:
            query += " AND datetime(order_time) <= ?"
            params.append(end_date.isoformat())
        
        query += f" GROUP BY {select_fields} ORDER BY {select_fields}"
        
        return query, params


class TradeDatabase:
    """
    Enhanced database management for trade data with advanced querying
    and analytics capabilities
    """
    
    def __init__(self, db_path: str = "data/trade_ledger.db"):
        """
        Initialize enhanced trade database
        
        Args:
            db_path: Path to SQLite database file
        """
        self.logger = logging.getLogger(__name__)
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Connection pool and threading
        self.connection_lock = threading.RLock()
        
        # Initialize database with enhanced schema
        self._init_enhanced_schema()
        
        # Create performance views
        self._create_performance_views()
        
        self.logger.info(f"Enhanced TradeDatabase initialized: {self.db_path}")
    
    @contextmanager
    def get_connection(self):
        """Get database connection with proper locking"""
        with self.connection_lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Enable column access by name
            try:
                yield conn
            finally:
                conn.close()
    
    def _init_enhanced_schema(self) -> None:
        """Initialize enhanced database schema with additional indices and constraints"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Create additional indices for performance
                performance_indices = [
                    "CREATE INDEX IF NOT EXISTS idx_trades_fill_time ON trades(fill_time)",
                    "CREATE INDEX IF NOT EXISTS idx_trades_pnl ON trades((realized_pnl + unrealized_pnl))",
                    "CREATE INDEX IF NOT EXISTS idx_trades_notional ON trades(notional_value)",
                    "CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)",
                    "CREATE INDEX IF NOT EXISTS idx_trades_composite ON trades(symbol, strategy, status, order_time)",
                    "CREATE INDEX IF NOT EXISTS idx_positions_last_update ON positions(last_update_time)",
                    "CREATE INDEX IF NOT EXISTS idx_positions_pnl ON positions(unrealized_pnl)",
                    "CREATE INDEX IF NOT EXISTS idx_daily_performance_date ON daily_performance(date)"
                ]
                
                for index_sql in performance_indices:
                    cursor.execute(index_sql)
                
                # Create trade_analytics table for pre-computed metrics
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS trade_analytics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        trade_id TEXT REFERENCES trades(trade_id),
                        symbol TEXT NOT NULL,
                        strategy TEXT,
                        entry_time TEXT,
                        exit_time TEXT,
                        holding_period_minutes INTEGER,
                        entry_price REAL,
                        exit_price REAL,
                        quantity INTEGER,
                        gross_pnl REAL,
                        net_pnl REAL,
                        total_costs REAL,
                        pnl_pct REAL,
                        max_adverse_excursion REAL,
                        max_favorable_excursion REAL,
                        win_loss_flag INTEGER,  -- 1 for win, 0 for loss
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(trade_id)
                    )
                """)
                
                # Create strategy_performance table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS strategy_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        strategy TEXT NOT NULL,
                        date TEXT NOT NULL,
                        num_trades INTEGER DEFAULT 0,
                        winning_trades INTEGER DEFAULT 0,
                        losing_trades INTEGER DEFAULT 0,
                        total_pnl REAL DEFAULT 0,
                        gross_profit REAL DEFAULT 0,
                        gross_loss REAL DEFAULT 0,
                        profit_factor REAL DEFAULT 0,
                        win_rate REAL DEFAULT 0,
                        avg_win REAL DEFAULT 0,
                        avg_loss REAL DEFAULT 0,
                        max_win REAL DEFAULT 0,
                        max_loss REAL DEFAULT 0,
                        total_volume REAL DEFAULT 0,
                        total_costs REAL DEFAULT 0,
                        sharpe_ratio REAL DEFAULT 0,
                        max_drawdown REAL DEFAULT 0,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(strategy, date)
                    )
                """)
                
                # Create market_data_cache table for price history
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS market_data_cache (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        open_price REAL,
                        high_price REAL,
                        low_price REAL,
                        close_price REAL,
                        volume INTEGER DEFAULT 0,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(symbol, timestamp)
                    )
                """)
                
                # Create additional indices
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_trade_analytics_symbol ON trade_analytics(symbol)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_trade_analytics_strategy ON trade_analytics(strategy)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_strategy_performance_strategy ON strategy_performance(strategy)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_data_symbol_time ON market_data_cache(symbol, timestamp)")
                
                conn.commit()
                self.logger.info("Enhanced database schema initialized")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize enhanced schema: {e}")
            raise
    
    def _create_performance_views(self) -> None:
        """Create database views for common performance queries"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Daily P&L view
                cursor.execute("""
                    CREATE VIEW IF NOT EXISTS v_daily_pnl AS
                    SELECT 
                        date(fill_time) as trade_date,
                        symbol,
                        strategy,
                        COUNT(*) as num_trades,
                        SUM(realized_pnl + unrealized_pnl) as total_pnl,
                        SUM(JSON_EXTRACT(transaction_costs, '$.total_cost')) as total_costs,
                        SUM(notional_value) as total_volume
                    FROM trades 
                    WHERE status = 'FILLED' AND fill_time IS NOT NULL
                    GROUP BY date(fill_time), symbol, strategy
                """)
                
                # Position summary view
                cursor.execute("""
                    CREATE VIEW IF NOT EXISTS v_position_summary AS
                    SELECT 
                        p.*,
                        CASE 
                            WHEN p.quantity > 0 THEN 'LONG'
                            WHEN p.quantity < 0 THEN 'SHORT'
                            ELSE 'FLAT'
                        END as position_side,
                        ABS(p.quantity * p.avg_price) as notional_value,
                        p.unrealized_pnl + p.realized_pnl as total_pnl
                    FROM positions p
                    WHERE p.quantity != 0
                """)
                
                # Trade performance view
                cursor.execute("""
                    CREATE VIEW IF NOT EXISTS v_trade_performance AS
                    SELECT 
                        t.*,
                        t.realized_pnl + t.unrealized_pnl as total_pnl,
                        CASE 
                            WHEN (t.realized_pnl + t.unrealized_pnl) > 0 THEN 1 
                            ELSE 0 
                        END as is_winner,
                        JSON_EXTRACT(t.transaction_costs, '$.total_cost') as total_transaction_cost,
                        (t.realized_pnl + t.unrealized_pnl) - JSON_EXTRACT(t.transaction_costs, '$.total_cost') as net_pnl
                    FROM trades t
                    WHERE t.status = 'FILLED'
                """)
                
                conn.commit()
                self.logger.info("Performance views created")
                
        except Exception as e:
            self.logger.error(f"Failed to create performance views: {e}")
    
    def bulk_insert_trades(self, trades: List[Trade]) -> bool:
        """
        Bulk insert trades for better performance
        
        Args:
            trades: List of Trade objects to insert
            
        Returns:
            bool: Success status
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                trade_data = []
                for trade in trades:
                    trade_tuple = (
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
                    trade_data.append(trade_tuple)
                
                cursor.executemany("""
                    INSERT OR REPLACE INTO trades (
                        trade_id, order_id, broker_order_id, symbol, action, strategy,
                        quantity_ordered, quantity_filled, price_ordered, price_filled,
                        order_time, fill_time, exit_time, status, is_entry, parent_trade_id,
                        notional_value, margin_used, unrealized_pnl, realized_pnl,
                        confidence_score, risk_score, transaction_costs, metadata, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, trade_data)
                
                conn.commit()
                self.logger.info(f"Bulk inserted {len(trades)} trades")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to bulk insert trades: {e}")
            return False
    
    def get_trades_by_filter(
        self,
        symbols: List[str] = None,
        strategies: List[str] = None,
        start_date: datetime = None,
        end_date: datetime = None,
        statuses: List[TradeStatus] = None,
        limit: int = None
    ) -> pd.DataFrame:
        """
        Get trades with advanced filtering
        
        Returns:
            pandas DataFrame with filtered trades
        """
        try:
            where_clause, params = QueryBuilder.build_trade_filter(
                symbols=symbols,
                strategies=strategies,
                start_date=start_date,
                end_date=end_date,
                statuses=statuses
            )
            
            query = f"""
            SELECT * FROM v_trade_performance 
            WHERE {where_clause}
            ORDER BY order_time DESC
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            with self.get_connection() as conn:
                df = pd.read_sql_query(query, conn, params=params)
                
                # Convert datetime columns
                datetime_cols = ['order_time', 'fill_time', 'exit_time', 'created_at', 'updated_at']
                for col in datetime_cols:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                
                return df
                
        except Exception as e:
            self.logger.error(f"Failed to get trades by filter: {e}")
            return pd.DataFrame()
    
    def get_performance_analytics(
        self,
        group_by: str = "date",
        start_date: datetime = None,
        end_date: datetime = None
    ) -> pd.DataFrame:
        """
        Get performance analytics with grouping
        
        Args:
            group_by: Group by 'date', 'symbol', 'strategy', or 'hour'
            
        Returns:
            DataFrame with performance metrics
        """
        try:
            query, params = QueryBuilder.build_performance_query(
                group_by=group_by,
                start_date=start_date,
                end_date=end_date
            )
            
            with self.get_connection() as conn:
                df = pd.read_sql_query(query, conn, params=params)
                
                # Calculate additional metrics
                df['win_rate'] = (df['winning_trades'] / df['num_trades'] * 100).round(2)
                df['profit_factor'] = np.where(
                    df['max_loss'] != 0,
                    abs(df['max_win'] / df['max_loss']),
                    0
                )
                df['net_pnl'] = df['total_pnl'] - df['total_costs']
                
                return df
                
        except Exception as e:
            self.logger.error(f"Failed to get performance analytics: {e}")
            return pd.DataFrame()
    
    def get_position_history(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """
        Get position history for a symbol
        
        Args:
            symbol: Symbol to get history for
            days: Number of days to look back
            
        Returns:
            DataFrame with position history
        """
        try:
            start_date = datetime.now() - timedelta(days=days)
            
            with self.get_connection() as conn:
                query = """
                SELECT 
                    date(last_update_time) as date,
                    symbol,
                    strategy,
                    quantity,
                    avg_price,
                    unrealized_pnl,
                    realized_pnl,
                    margin_used
                FROM positions 
                WHERE symbol = ? AND datetime(last_update_time) >= ?
                ORDER BY last_update_time
                """
                
                df = pd.read_sql_query(query, conn, params=[symbol, start_date.isoformat()])
                df['date'] = pd.to_datetime(df['date'])
                
                return df
                
        except Exception as e:
            self.logger.error(f"Failed to get position history for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_strategy_performance(self, strategy: str = None, days: int = 30) -> Dict[str, Any]:
        """
        Calculate comprehensive strategy performance metrics
        
        Args:
            strategy: Strategy name (None for all strategies)
            days: Number of days to analyze
            
        Returns:
            Dictionary with performance metrics
        """
        try:
            start_date = datetime.now() - timedelta(days=days)
            
            # Build query
            where_clause = "WHERE datetime(fill_time) >= ?"
            params = [start_date.isoformat()]
            
            if strategy:
                where_clause += " AND strategy = ?"
                params.append(strategy)
            
            with self.get_connection() as conn:
                query = f"""
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN total_pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(CASE WHEN total_pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                    SUM(total_pnl) as total_pnl,
                    AVG(total_pnl) as avg_pnl,
                    MAX(total_pnl) as max_win,
                    MIN(total_pnl) as max_loss,
                    SUM(CASE WHEN total_pnl > 0 THEN total_pnl ELSE 0 END) as gross_profit,
                    SUM(CASE WHEN total_pnl < 0 THEN ABS(total_pnl) ELSE 0 END) as gross_loss,
                    SUM(total_transaction_cost) as total_costs,
                    SUM(notional_value) as total_volume,
                    AVG(CASE WHEN total_pnl > 0 THEN total_pnl END) as avg_win,
                    AVG(CASE WHEN total_pnl < 0 THEN total_pnl END) as avg_loss
                FROM v_trade_performance 
                {where_clause}
                """
                
                cursor = conn.cursor()
                cursor.execute(query, params)
                row = cursor.fetchone()
                
                if not row or row['total_trades'] == 0:
                    return {'error': 'No trades found for the specified criteria'}
                
                # Calculate derived metrics
                win_rate = (row['winning_trades'] / row['total_trades']) * 100 if row['total_trades'] > 0 else 0
                profit_factor = row['gross_profit'] / row['gross_loss'] if row['gross_loss'] > 0 else 0
                avg_win = row['avg_win'] or 0
                avg_loss = row['avg_loss'] or 0
                expectancy = (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss)
                
                # Get daily P&L for Sharpe calculation
                daily_query = f"""
                SELECT 
                    date(fill_time) as trade_date,
                    SUM(total_pnl) as daily_pnl
                FROM v_trade_performance 
                {where_clause}
                GROUP BY date(fill_time)
                ORDER BY trade_date
                """
                
                daily_df = pd.read_sql_query(daily_query, conn, params=params)
                
                # Calculate Sharpe ratio (annualized)
                if len(daily_df) > 1:
                    daily_returns = daily_df['daily_pnl']
                    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0
                    
                    # Calculate maximum drawdown
                    cumulative_pnl = daily_returns.cumsum()
                    running_max = cumulative_pnl.expanding().max()
                    drawdown = cumulative_pnl - running_max
                    max_drawdown = drawdown.min()
                else:
                    sharpe_ratio = 0
                    max_drawdown = 0
                
                return {
                    'strategy': strategy or 'ALL',
                    'analysis_period_days': days,
                    'total_trades': row['total_trades'],
                    'winning_trades': row['winning_trades'],
                    'losing_trades': row['losing_trades'],
                    'win_rate_pct': round(win_rate, 2),
                    'total_pnl': round(row['total_pnl'], 2),
                    'net_pnl': round(row['total_pnl'] - row['total_costs'], 2),
                    'avg_pnl_per_trade': round(row['avg_pnl'], 2),
                    'max_win': round(row['max_win'], 2),
                    'max_loss': round(row['max_loss'], 2),
                    'avg_win': round(avg_win, 2),
                    'avg_loss': round(avg_loss, 2),
                    'profit_factor': round(profit_factor, 2),
                    'expectancy': round(expectancy, 2),
                    'sharpe_ratio': round(sharpe_ratio, 2),
                    'max_drawdown': round(max_drawdown, 2),
                    'total_volume': round(row['total_volume'], 2),
                    'total_transaction_costs': round(row['total_costs'], 2)
                }
                
        except Exception as e:
            self.logger.error(f"Failed to calculate strategy performance: {e}")
            return {'error': str(e)}
    
    def backup_database(self, backup_path: str = None) -> bool:
        """
        Create database backup
        
        Args:
            backup_path: Path for backup file
            
        Returns:
            bool: Success status
        """
        try:
            if backup_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_path = f"{self.db_path.stem}_backup_{timestamp}.db"
            
            backup_path = Path(backup_path)
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy database file
            shutil.copy2(self.db_path, backup_path)
            
            self.logger.info(f"Database backed up to: {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to backup database: {e}")
            return False
    
    def optimize_database(self) -> bool:
        """
        Optimize database performance
        
        Returns:
            bool: Success status
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Analyze tables for query optimization
                cursor.execute("ANALYZE")
                
                # Vacuum to reclaim space
                cursor.execute("VACUUM")
                
                # Update statistics
                cursor.execute("PRAGMA optimize")
                
                conn.commit()
                
                self.logger.info("Database optimized successfully")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to optimize database: {e}")
            return False
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics and health metrics"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                stats = {}
                
                # Table row counts
                tables = ['trades', 'positions', 'daily_performance', 'trade_analytics', 'strategy_performance']
                for table in tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    stats[f'{table}_count'] = cursor.fetchone()[0]
                
                # Database size
                stats['database_size_mb'] = round(self.db_path.stat().st_size / (1024 * 1024), 2)
                
                # Date ranges
                cursor.execute("SELECT MIN(order_time), MAX(order_time) FROM trades WHERE order_time IS NOT NULL")
                date_range = cursor.fetchone()
                if date_range[0]:
                    stats['first_trade_date'] = date_range[0]
                    stats['last_trade_date'] = date_range[1]
                
                # Recent activity
                cursor.execute("SELECT COUNT(*) FROM trades WHERE datetime(order_time) >= datetime('now', '-1 day')")
                stats['trades_last_24h'] = cursor.fetchone()[0]
                
                return stats
                
        except Exception as e:
            self.logger.error(f"Failed to get database stats: {e}")
            return {'error': str(e)}


class DataExporter:
    """Export utilities for trade data analysis"""
    
    def __init__(self, trade_db: TradeDatabase):
        """
        Initialize data exporter
        
        Args:
            trade_db: TradeDatabase instance
        """
        self.trade_db = trade_db
        self.logger = logging.getLogger(__name__)
    
    def export_trades_to_csv(
        self,
        file_path: str,
        start_date: datetime = None,
        end_date: datetime = None,
        symbols: List[str] = None
    ) -> bool:
        """
        Export trades to CSV file
        
        Args:
            file_path: Output CSV file path
            start_date: Start date filter
            end_date: End date filter  
            symbols: Symbol filter
            
        Returns:
            bool: Success status
        """
        try:
            # Get filtered trades
            df = self.trade_db.get_trades_by_filter(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date
            )
            
            if df.empty:
                self.logger.warning("No trades found for export")
                return False
            
            # Export to CSV
            df.to_csv(file_path, index=False)
            
            self.logger.info(f"Exported {len(df)} trades to: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export trades to CSV: {e}")
            return False
    
    def export_performance_report(self, file_path: str, days: int = 30) -> bool:
        """
        Export comprehensive performance report
        
        Args:
            file_path: Output file path (Excel format)
            days: Analysis period in days
            
        Returns:
            bool: Success status
        """
        try:
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                
                # Daily performance
                daily_perf = self.trade_db.get_performance_analytics(
                    group_by='date',
                    start_date=datetime.now() - timedelta(days=days)
                )
                daily_perf.to_excel(writer, sheet_name='Daily_Performance', index=False)
                
                # Symbol performance
                symbol_perf = self.trade_db.get_performance_analytics(
                    group_by='symbol',
                    start_date=datetime.now() - timedelta(days=days)
                )
                symbol_perf.to_excel(writer, sheet_name='Symbol_Performance', index=False)
                
                # Strategy performance
                strategy_perf = self.trade_db.get_performance_analytics(
                    group_by='strategy',
                    start_date=datetime.now() - timedelta(days=days)
                )
                strategy_perf.to_excel(writer, sheet_name='Strategy_Performance', index=False)
                
                # Recent trades
                recent_trades = self.trade_db.get_trades_by_filter(
                    start_date=datetime.now() - timedelta(days=7),
                    limit=100
                )
                recent_trades.to_excel(writer, sheet_name='Recent_Trades', index=False)
            
            self.logger.info(f"Performance report exported to: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export performance report: {e}")
            return False


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize enhanced database
    db = TradeDatabase("test_enhanced_trade_db.db")
    
    # Get database stats
    stats = db.get_database_stats()
    print(f"Database Stats: {stats}")
    
    # Get performance analytics
    daily_perf = db.get_performance_analytics(group_by='date', days=30)
    print(f"Daily Performance Shape: {daily_perf.shape}")
    
    # Calculate strategy performance
    strategy_metrics = db.calculate_strategy_performance(days=30)
    print(f"Strategy Metrics: {strategy_metrics}")
    
    # Initialize data exporter
    exporter = DataExporter(db)
    
    # Export sample data (would work with actual data)
    # exporter.export_trades_to_csv("sample_trades.csv", days=7)
    # exporter.export_performance_report("performance_report.xlsx", days=30)