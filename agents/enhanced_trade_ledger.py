"""
Enhanced Trade Ledger with Dual Logging for SuperTrader.AI

Extends the existing trade ledger with:
- CSV ledger (ledgers/ledger_master.csv) - one row per trade
- JSON session files (sessions/session_<timestamp>.json) - full session summary
- Session management for trading sessions
- Performance metrics calculation
- Export capabilities for analysis

This module works alongside the existing TradeLedger class and adds
session-based tracking and dual format logging.

Author: SuperTrader.AI Team  
Version: 2.0.0
Last Updated: 2024-11-08
"""

import json
import logging
import os
import csv
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import asdict
import uuid

# Import existing trade ledger components
from agents.trade_ledger import TradeLedger, Trade, TradeAction, TradeStatus, get_trade_ledger

logger = logging.getLogger(__name__)


class SessionLedger:
    """
    Enhanced ledger system with dual CSV/JSON logging and session management
    
    Features:
    - CSV trade log for individual trade records
    - JSON session files for comprehensive session summaries
    - Performance metrics calculation
    - Session-based analysis and reporting
    """
    
    def __init__(self, base_path: str = "trading_data"):
        """
        Initialize SessionLedger with dual logging
        
        Args:
            base_path: Base directory for all trading data files
        """
        self.base_path = Path(base_path)
        self.ledgers_dir = self.base_path / "ledgers" 
        self.sessions_dir = self.base_path / "sessions"
        
        # Create directories
        self.ledgers_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.csv_ledger_path = self.ledgers_dir / "ledger_master.csv"
        
        # Current session state
        self.current_session_id: Optional[str] = None
        self.current_session_data: Dict[str, Any] = {}
        self.session_start_time: Optional[datetime] = None
        self.session_trades: List[Dict[str, Any]] = []
        
        # Get reference to main trade ledger
        self.trade_ledger = get_trade_ledger()
        
        # Initialize CSV ledger if not exists
        self._init_csv_ledger()
        
        logger.info(f"SessionLedger initialized: CSV={self.csv_ledger_path}, Sessions={self.sessions_dir}")
    
    def _init_csv_ledger(self) -> None:
        """Initialize CSV ledger with headers if not exists"""
        if not self.csv_ledger_path.exists():
            headers = [
                'trade_id', 'session_id', 'timestamp', 'symbol', 'action', 'strategy',
                'quantity', 'price', 'notional_value', 'transaction_costs', 'net_cost',
                'sentiment', 'confidence', 'q_values', 'slippage_bps', 'success',
                'unrealized_pnl', 'realized_pnl', 'portfolio_value', 'cash_balance',
                'position_change', 'error_message', 'metadata'
            ]
            
            with open(self.csv_ledger_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
            
            logger.info(f"Created CSV ledger: {self.csv_ledger_path}")
    
    def start_session(self, 
                     symbols: List[str] = None,
                     strategy: str = "agentic_trading",
                     initial_capital: float = 1_000_000.0,
                     metadata: Dict[str, Any] = None) -> str:
        """
        Start a new trading session
        
        Args:
            symbols: List of symbols to trade in this session
            strategy: Strategy name for this session  
            initial_capital: Starting capital for the session
            metadata: Additional session metadata
            
        Returns:
            Session ID
        """
        # End current session if active
        if self.current_session_id:
            logger.warning(f"Ending previous session {self.current_session_id}")
            self.finalize_session()
        
        # Create new session
        self.current_session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self.session_start_time = datetime.now()
        self.session_trades = []
        
        self.current_session_data = {
            'session_id': self.current_session_id,
            'start_time': self.session_start_time.isoformat(),
            'end_time': None,
            'strategy': strategy,
            'symbols': symbols or [],
            'initial_capital': initial_capital,
            'metadata': metadata or {},
            'trades': [],
            'performance': {},
            'risk_metrics': {},
            'final_summary': {}
        }
        
        logger.info(f"Started trading session: {self.current_session_id} with strategy '{strategy}'")
        return self.current_session_id
    
    def log_trade(self, trade_data: Dict[str, Any], session_id: Optional[str] = None) -> bool:
        """
        Log a trade to both CSV and session data
        
        Args:
            trade_data: Trade information (from PortfolioSimulator.execute())
            session_id: Session ID (uses current session if None)
            
        Returns:
            Success status
        """
        try:
            # Use current session if none specified
            if session_id is None:
                session_id = self.current_session_id
            
            if session_id is None:
                logger.warning("No active session for trade logging")
                return False
            
            # Prepare trade record for CSV
            csv_record = self._prepare_csv_record(trade_data, session_id)
            
            # Append to CSV ledger
            with open(self.csv_ledger_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    csv_record['trade_id'], csv_record['session_id'], csv_record['timestamp'],
                    csv_record['symbol'], csv_record['action'], csv_record['strategy'],
                    csv_record['quantity'], csv_record['price'], csv_record['notional_value'],
                    csv_record['transaction_costs'], csv_record['net_cost'],
                    csv_record['sentiment'], csv_record['confidence'], 
                    json.dumps(csv_record['q_values']), csv_record['slippage_bps'], csv_record['success'],
                    csv_record['unrealized_pnl'], csv_record['realized_pnl'], 
                    csv_record['portfolio_value'], csv_record['cash_balance'],
                    json.dumps(csv_record['position_change']), csv_record['error_message'],
                    json.dumps(csv_record['metadata'])
                ])
            
            # Add to current session data
            if session_id == self.current_session_id:
                self.session_trades.append(csv_record)
                self.current_session_data['trades'].append(csv_record)
            
            logger.debug(f"Logged trade {csv_record['trade_id']} to CSV and session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to log trade: {e}")
            return False
    
    def _prepare_csv_record(self, trade_data: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Prepare trade data for CSV logging"""
        return {
            'trade_id': trade_data.get('trade_id', 'unknown'),
            'session_id': session_id,
            'timestamp': trade_data.get('timestamp', datetime.now().isoformat()),
            'symbol': trade_data.get('symbol', ''),
            'action': trade_data.get('action', ''),
            'strategy': self.current_session_data.get('strategy', 'unknown'),
            'quantity': trade_data.get('quantity', 0),
            'price': trade_data.get('price', 0.0),
            'notional_value': trade_data.get('notional_value', 0.0),
            'transaction_costs': trade_data.get('transaction_costs', 0.0),
            'net_cost': trade_data.get('net_cost', 0.0),
            'sentiment': trade_data.get('sentiment', 0.0),
            'confidence': trade_data.get('confidence', 0.0),
            'q_values': trade_data.get('q_values', []),
            'slippage_bps': trade_data.get('slippage_bps', 0.0),
            'success': trade_data.get('success', False),
            'unrealized_pnl': trade_data.get('portfolio_impact', {}).get('unrealized_pnl', 0.0),
            'realized_pnl': trade_data.get('portfolio_impact', {}).get('realized_pnl', 0.0),
            'portfolio_value': trade_data.get('portfolio_impact', {}).get('total_value', 0.0),
            'cash_balance': trade_data.get('portfolio_impact', {}).get('cash_balance', 0.0),
            'position_change': trade_data.get('position_change', {}),
            'error_message': trade_data.get('error_message', ''),
            'metadata': trade_data.get('metadata', {})
        }
    
    def finalize_session(self, simulator_instance=None) -> Optional[str]:
        """
        Finalize current session and create JSON summary
        
        Args:
            simulator_instance: PortfolioSimulator instance for final metrics
            
        Returns:
            Path to created JSON file
        """
        if not self.current_session_id:
            logger.warning("No active session to finalize")
            return None
        
        try:
            # Calculate session performance metrics
            session_end_time = datetime.now()
            self.current_session_data['end_time'] = session_end_time.isoformat()
            self.current_session_data['duration_minutes'] = (
                session_end_time - self.session_start_time
            ).total_seconds() / 60
            
            # Calculate performance metrics
            performance_metrics = self._calculate_session_metrics(simulator_instance)
            self.current_session_data['performance'] = performance_metrics
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(simulator_instance)
            self.current_session_data['risk_metrics'] = risk_metrics
            
            # Create final summary
            final_summary = self._create_final_summary(simulator_instance)
            self.current_session_data['final_summary'] = final_summary
            
            # Save JSON session file
            json_path = self.sessions_dir / f"{self.current_session_id}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(self.current_session_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Session {self.current_session_id} finalized: {json_path}")
            
            # Reset current session
            session_id = self.current_session_id
            self._reset_current_session()
            
            return str(json_path)
            
        except Exception as e:
            logger.error(f"Failed to finalize session {self.current_session_id}: {e}")
            return None
    
    def _calculate_session_metrics(self, simulator=None) -> Dict[str, Any]:
        """Calculate comprehensive session performance metrics"""
        trades = self.session_trades
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_pnl_per_trade': 0.0,
                'total_costs': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'profit_factor': 0.0
            }
        
        successful_trades = [t for t in trades if t['success']]
        winning_trades = [t for t in successful_trades if t['realized_pnl'] > 0]
        losing_trades = [t for t in successful_trades if t['realized_pnl'] < 0]
        
        total_pnl = sum(t['realized_pnl'] for t in successful_trades)
        total_costs = sum(t['transaction_costs'] for t in successful_trades)
        
        win_rate = len(winning_trades) / len(successful_trades) if successful_trades else 0.0
        avg_pnl_per_trade = total_pnl / len(successful_trades) if successful_trades else 0.0
        
        # Calculate Sharpe ratio (simplified)
        if simulator and len(successful_trades) > 1:
            returns = [t['realized_pnl'] / t['notional_value'] for t in successful_trades if t['notional_value'] > 0]
            if returns:
                mean_return = np.mean(returns)
                std_return = np.std(returns)
                sharpe_ratio = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0.0
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0
        
        # Calculate profit factor
        gross_profits = sum(t['realized_pnl'] for t in winning_trades) if winning_trades else 0
        gross_losses = abs(sum(t['realized_pnl'] for t in losing_trades)) if losing_trades else 0
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf') if gross_profits > 0 else 0.0
        
        # Get max drawdown from simulator
        max_drawdown = simulator.max_drawdown if simulator else 0.0
        
        return {
            'total_trades': len(trades),
            'successful_trades': len(successful_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'net_pnl': total_pnl - total_costs,
            'avg_pnl_per_trade': avg_pnl_per_trade,
            'total_costs': total_costs,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor,
            'largest_win': max((t['realized_pnl'] for t in winning_trades), default=0),
            'largest_loss': min((t['realized_pnl'] for t in losing_trades), default=0),
            'avg_win': np.mean([t['realized_pnl'] for t in winning_trades]) if winning_trades else 0,
            'avg_loss': np.mean([t['realized_pnl'] for t in losing_trades]) if losing_trades else 0
        }
    
    def _calculate_risk_metrics(self, simulator=None) -> Dict[str, Any]:
        """Calculate risk metrics for the session"""
        if not simulator:
            return {}
        
        return {
            'max_leverage_used': getattr(simulator, 'leverage_ratio', 0.0),
            'max_margin_utilized_pct': (simulator.margin_used / simulator.initial_capital) * 100 if simulator.initial_capital > 0 else 0,
            'max_exposure': simulator.total_exposure,
            'max_portfolio_value': simulator.peak_value,
            'final_portfolio_value': simulator.total_value,
            'cash_utilization_pct': ((simulator.initial_capital - simulator.cash_balance) / simulator.initial_capital) * 100 if simulator.initial_capital > 0 else 0,
            'active_positions': len(simulator.positions),
            'position_concentration': max(
                (abs(pos.quantity) * pos.current_price / simulator.total_value * 100 
                 for pos in simulator.positions.values()),
                default=0.0
            )
        }
    
    def _create_final_summary(self, simulator=None) -> Dict[str, Any]:
        """Create final session summary with key insights"""
        performance = self.current_session_data.get('performance', {})
        risk_metrics = self.current_session_data.get('risk_metrics', {})
        
        # Determine session outcome
        total_pnl = performance.get('total_pnl', 0)
        win_rate = performance.get('win_rate', 0)
        
        if total_pnl > 0 and win_rate > 0.5:
            outcome = "PROFITABLE"
        elif total_pnl > 0:
            outcome = "MARGINALLY_PROFITABLE" 
        elif total_pnl < 0 and win_rate < 0.3:
            outcome = "POOR_PERFORMANCE"
        else:
            outcome = "MIXED_RESULTS"
        
        # Key insights
        insights = []
        
        if performance.get('sharpe_ratio', 0) > 1.0:
            insights.append("Strong risk-adjusted returns")
        
        if performance.get('max_drawdown', 0) > 0.1:
            insights.append("High drawdown experienced")
        
        if performance.get('win_rate', 0) > 0.7:
            insights.append("High win rate achieved")
        elif performance.get('win_rate', 0) < 0.3:
            insights.append("Low win rate - review strategy")
        
        if risk_metrics.get('max_leverage_used', 0) > 3.0:
            insights.append("High leverage utilized")
        
        return {
            'session_outcome': outcome,
            'key_insights': insights,
            'profitability': 'PROFITABLE' if total_pnl > 0 else 'LOSS_MAKING',
            'risk_profile': 'HIGH' if risk_metrics.get('max_leverage_used', 0) > 2.5 else 'MODERATE',
            'execution_quality': 'GOOD' if performance.get('total_costs', 0) / max(performance.get('total_pnl', 1), 1) < 0.1 else 'NEEDS_IMPROVEMENT',
            'recommended_actions': self._get_recommended_actions(performance, risk_metrics)
        }
    
    def _get_recommended_actions(self, performance: Dict, risk_metrics: Dict) -> List[str]:
        """Generate recommended actions based on session results"""
        actions = []
        
        win_rate = performance.get('win_rate', 0)
        profit_factor = performance.get('profit_factor', 0)
        max_dd = performance.get('max_drawdown', 0)
        
        if win_rate < 0.4:
            actions.append("Review entry signals - low win rate")
        
        if profit_factor < 1.5:
            actions.append("Improve risk/reward ratio")
        
        if max_dd > 0.15:
            actions.append("Implement better position sizing")
        
        if performance.get('total_costs', 0) > abs(performance.get('total_pnl', 1)) * 0.2:
            actions.append("Reduce transaction frequency - high cost ratio")
        
        if risk_metrics.get('max_leverage_used', 0) > 4.0:
            actions.append("Reduce leverage usage")
        
        return actions or ["Continue current strategy"]
    
    def _reset_current_session(self) -> None:
        """Reset current session variables"""
        self.current_session_id = None
        self.current_session_data = {}
        self.session_start_time = None
        self.session_trades = []
    
    def get_session_summary(self, session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get summary of a specific session or current session"""
        if session_id is None and self.current_session_id:
            return self.current_session_data
        
        if session_id:
            json_path = self.sessions_dir / f"{session_id}.json"
            if json_path.exists():
                with open(json_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        
        return None
    
    def get_recent_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get list of recent sessions with basic info"""
        sessions = []
        
        for json_file in sorted(self.sessions_dir.glob("session_*.json"), reverse=True):
            if len(sessions) >= limit:
                break
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                
                sessions.append({
                    'session_id': session_data['session_id'],
                    'start_time': session_data['start_time'],
                    'end_time': session_data.get('end_time'),
                    'strategy': session_data.get('strategy'),
                    'total_trades': session_data.get('performance', {}).get('total_trades', 0),
                    'total_pnl': session_data.get('performance', {}).get('total_pnl', 0),
                    'win_rate': session_data.get('performance', {}).get('win_rate', 0),
                    'outcome': session_data.get('final_summary', {}).get('session_outcome', 'UNKNOWN')
                })
            except Exception as e:
                logger.warning(f"Failed to read session file {json_file}: {e}")
        
        return sessions
    
    def export_csv_analysis(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Export CSV data for analysis"""
        try:
            df = pd.read_csv(self.csv_ledger_path)
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Filter by date range if provided
                if start_date:
                    df = df[df['timestamp'] >= pd.to_datetime(start_date)]
                if end_date:
                    df = df[df['timestamp'] <= pd.to_datetime(end_date)]
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to export CSV analysis: {e}")
            return pd.DataFrame()
    
    def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive performance dashboard data"""
        try:
            # Load recent CSV data
            df = self.export_csv_analysis()
            
            if df.empty:
                return {'error': 'No trade data available'}
            
            # Recent sessions summary
            recent_sessions = self.get_recent_sessions(5)
            
            # Overall statistics
            successful_trades = df[df['success'] == True]
            
            dashboard = {
                'overview': {
                    'total_sessions': len(recent_sessions),
                    'total_trades': len(df),
                    'successful_trades': len(successful_trades),
                    'total_pnl': successful_trades['realized_pnl'].sum(),
                    'win_rate': (successful_trades['realized_pnl'] > 0).mean(),
                    'avg_trade_pnl': successful_trades['realized_pnl'].mean()
                },
                'recent_sessions': recent_sessions,
                'daily_pnl': successful_trades.groupby(successful_trades['timestamp'].dt.date)['realized_pnl'].sum().tail(10).to_dict(),
                'symbol_performance': successful_trades.groupby('symbol')['realized_pnl'].agg(['count', 'sum', 'mean']).to_dict(),
                'strategy_performance': successful_trades.groupby('strategy')['realized_pnl'].agg(['count', 'sum', 'mean']).to_dict()
            }
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Failed to create performance dashboard: {e}")
            return {'error': str(e)}


# Global instance
_session_ledger: Optional[SessionLedger] = None


def get_session_ledger(base_path: str = "trading_data") -> SessionLedger:
    """Get or create the global SessionLedger instance"""
    global _session_ledger
    
    if _session_ledger is None:
        _session_ledger = SessionLedger(base_path)
    
    return _session_ledger


# Convenience functions for integration
def start_trading_session(symbols: List[str] = None, 
                         strategy: str = "agentic_trading", 
                         initial_capital: float = 1_000_000.0) -> str:
    """Start a new trading session"""
    ledger = get_session_ledger()
    return ledger.start_session(symbols, strategy, initial_capital)


def log_trade_to_session(trade_data: Dict[str, Any], session_id: Optional[str] = None) -> bool:
    """Log a trade to the current session"""
    ledger = get_session_ledger()
    return ledger.log_trade(trade_data, session_id)


def finalize_trading_session(simulator=None) -> Optional[str]:
    """Finalize the current trading session"""
    ledger = get_session_ledger()
    return ledger.finalize_session(simulator)


if __name__ == "__main__":
    # Test the session ledger
    ledger = SessionLedger("test_trading_data")
    
    # Start session
    session_id = ledger.start_session(['NIFTY', 'BANKNIFTY'], 'test_strategy')
    print(f"Started session: {session_id}")
    
    # Log sample trade
    sample_trade = {
        'trade_id': 'TEST_001',
        'symbol': 'NIFTY',
        'action': 'BUY',
        'quantity': 50,
        'price': 19500.0,
        'notional_value': 975000.0,
        'transaction_costs': 150.0,
        'sentiment': 0.3,
        'confidence': 0.75,
        'success': True,
        'portfolio_impact': {
            'total_value': 1000000.0,
            'unrealized_pnl': 2500.0,
            'realized_pnl': 0.0,
            'cash_balance': 25000.0
        }
    }
    
    success = ledger.log_trade(sample_trade)
    print(f"Trade logged: {success}")
    
    # Finalize session
    json_path = ledger.finalize_session()
    print(f"Session finalized: {json_path}")
    
    # Get dashboard
    dashboard = ledger.get_performance_dashboard()
    print(f"Dashboard overview: {dashboard.get('overview', {})}")