"""
Enhanced Logging System for SuperTrader.AI

Extended logging capabilities including:
- Structured JSON logging
- Performance metrics calculation
- Trading session analytics
- Real-time metric tracking
- Performance attribution reporting

Author: SuperTrader.AI Team
Version: 2.0.0
Last Updated: 2024-11-08
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import numpy as np
from dataclasses import dataclass, field


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    volatility: float = 0.0
    calmar_ratio: float = 0.0
    avg_trade_return: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    trades_count: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win_streak: int = 0
    largest_loss_streak: int = 0
    current_streak: int = 0
    expectancy: float = 0.0
    ulcer_index: float = 0.0


class StructuredLogger:
    def __init__(self, name="main", log_dir="logs", level=logging.INFO):
        self.name=name
        os.makedirs(log_dir,exist_ok=True)
        self.log_path=os.path.join(log_dir,f"{name}.log")

        self.logger=logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate=False

        if not self.logger.handlers:
            file_handler=logging.FileHandler(self.log_path)
            console_handler=logging.StreamHandler()

            formatter=logging.Formatter('%(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    
    def _log(self,level,message,**metadata):
        record={
            "timestamp":datetime.utcnow().isoformat(),
            "level":logging.getLevelName(level),
            "message":message,
            "metadata":metadata
        }
        self.logger.log(level,json.dumps(record,ensure_ascii=False))

    def info(self,message,**metadata):
        self._log(logging.INFO,message,**metadata)

    def warning(self,message,**metadata):
        self._log(logging.WARNING,message,**metadata)

    def error(self,message,**metadata):
        self._log(logging.ERROR,message,**metadata)

    def critical(self,message,**metadata):
        self._log(logging.CRITICAL,message,**metadata)


class PerformanceLogger:
    """
    Enhanced performance metrics logger for trading systems
    
    Provides comprehensive performance analytics including:
    - Sharpe ratio calculation
    - Win rate tracking  
    - Maximum drawdown analysis
    - Risk-adjusted returns
    - Trade attribution analysis
    """
    
    def __init__(self, name: str = "performance", log_dir: str = "logs"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize structured logger
        self.logger = StructuredLogger(f"{name}_performance", str(self.log_dir))
        
        # Performance tracking data
        self.trade_history: List[Dict[str, Any]] = []
        self.portfolio_values: List[Tuple[datetime, float]] = []
        self.returns_series: List[float] = []
        self.drawdown_series: List[float] = []
        
        # Tracking state
        self.peak_value = 0.0
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.win_streak = 0
        self.loss_streak = 0
        self.current_streak_type = None  # 'win' or 'loss'
        
        # Risk-free rate for Sharpe calculation (6% annualized)
        self.risk_free_rate = 0.06
        
        self.logger.info("Performance logger initialized", 
                        name=name, log_dir=str(self.log_dir))
    
    def log_trade(self, 
                 trade_id: str,
                 symbol: str,
                 action: str, 
                 quantity: int,
                 price: float,
                 pnl: float,
                 portfolio_value: float,
                 timestamp: datetime = None,
                 metadata: Dict[str, Any] = None) -> None:
        """
        Log individual trade and update performance metrics
        
        Args:
            trade_id: Unique trade identifier
            symbol: Trading symbol  
            action: BUY/SELL/HOLD
            quantity: Trade quantity
            price: Execution price
            pnl: Profit/Loss for this trade
            portfolio_value: Current portfolio value
            timestamp: Trade execution time
            metadata: Additional trade context
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Create trade record
        trade_record = {
            'trade_id': trade_id,
            'timestamp': timestamp.isoformat(),
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'price': price,
            'pnl': pnl,
            'portfolio_value': portfolio_value,
            'metadata': metadata or {}
        }
        
        # Add to trade history
        self.trade_history.append(trade_record)
        
        # Update portfolio value series
        self.portfolio_values.append((timestamp, portfolio_value))
        
        # Calculate return for this trade
        if len(self.portfolio_values) > 1:
            prev_value = self.portfolio_values[-2][1]
            trade_return = (portfolio_value - prev_value) / prev_value if prev_value > 0 else 0
            self.returns_series.append(trade_return)
        
        # Update drawdown tracking
        self._update_drawdown(portfolio_value)
        
        # Update win/loss streaks
        self._update_streaks(pnl)
        
        # Log structured trade data
        self.logger.info("Trade executed",
                        trade_id=trade_id,
                        symbol=symbol,
                        action=action,
                        pnl=pnl,
                        portfolio_value=portfolio_value,
                        current_drawdown=self.current_drawdown,
                        win_streak=self.win_streak,
                        loss_streak=self.loss_streak)
    
    def _update_drawdown(self, portfolio_value: float) -> None:
        """Update drawdown metrics"""
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.peak_value - portfolio_value) / self.peak_value
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
        
        self.drawdown_series.append(self.current_drawdown)
    
    def _update_streaks(self, pnl: float) -> None:
        """Update win/loss streak tracking"""
        if pnl > 0:  # Winning trade
            if self.current_streak_type == 'win':
                self.win_streak += 1
            else:
                self.win_streak = 1
                self.loss_streak = 0
                self.current_streak_type = 'win'
        elif pnl < 0:  # Losing trade
            if self.current_streak_type == 'loss':
                self.loss_streak += 1
            else:
                self.loss_streak = 1
                self.win_streak = 0
                self.current_streak_type = 'loss'
        # pnl == 0 doesn't change streaks
    
    def calculate_sharpe_ratio(self, period_days: int = 252) -> float:
        """
        Calculate Sharpe ratio
        
        Args:
            period_days: Annualization factor (252 for daily, 365 for continuous)
            
        Returns:
            Sharpe ratio
        """
        if len(self.returns_series) < 2:
            return 0.0
        
        returns = np.array(self.returns_series)
        
        # Calculate excess returns
        daily_rf_rate = self.risk_free_rate / period_days
        excess_returns = returns - daily_rf_rate
        
        # Calculate Sharpe ratio
        if np.std(excess_returns) == 0:
            return 0.0
        
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(period_days)
        return float(sharpe)
    
    def calculate_sortino_ratio(self, period_days: int = 252) -> float:
        """
        Calculate Sortino ratio (using downside deviation)
        
        Args:
            period_days: Annualization factor
            
        Returns:
            Sortino ratio
        """
        if len(self.returns_series) < 2:
            return 0.0
        
        returns = np.array(self.returns_series)
        daily_rf_rate = self.risk_free_rate / period_days
        excess_returns = returns - daily_rf_rate
        
        # Calculate downside deviation
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf')  # No downside
        
        downside_deviation = np.sqrt(np.mean(downside_returns**2))
        
        if downside_deviation == 0:
            return 0.0
        
        sortino = np.mean(excess_returns) / downside_deviation * np.sqrt(period_days)
        return float(sortino)
    
    def calculate_win_rate(self) -> float:
        """Calculate win rate percentage"""
        if not self.trade_history:
            return 0.0
        
        profitable_trades = len([t for t in self.trade_history if t['pnl'] > 0])
        total_trades = len(self.trade_history)
        
        return (profitable_trades / total_trades) * 100 if total_trades > 0 else 0.0
    
    def calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profits / gross losses)"""
        if not self.trade_history:
            return 0.0
        
        profits = sum(t['pnl'] for t in self.trade_history if t['pnl'] > 0)
        losses = abs(sum(t['pnl'] for t in self.trade_history if t['pnl'] < 0))
        
        return profits / losses if losses > 0 else float('inf') if profits > 0 else 0.0
    
    def calculate_calmar_ratio(self) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)"""
        if self.max_drawdown == 0 or not self.portfolio_values:
            return 0.0
        
        # Calculate annualized return
        if len(self.portfolio_values) < 2:
            return 0.0
        
        start_value = self.portfolio_values[0][1]
        end_value = self.portfolio_values[-1][1]
        
        days = (self.portfolio_values[-1][0] - self.portfolio_values[0][0]).days
        if days == 0:
            return 0.0
        
        annualized_return = ((end_value / start_value) ** (365 / days) - 1) if start_value > 0 else 0
        
        return annualized_return / self.max_drawdown
    
    def calculate_ulcer_index(self) -> float:
        """Calculate Ulcer Index (measure of downside risk)"""
        if len(self.drawdown_series) < 2:
            return 0.0
        
        drawdowns = np.array(self.drawdown_series)
        squared_drawdowns = drawdowns ** 2
        ulcer = np.sqrt(np.mean(squared_drawdowns)) * 100
        
        return float(ulcer)
    
    def get_comprehensive_metrics(self) -> PerformanceMetrics:
        """Calculate and return comprehensive performance metrics"""
        if not self.trade_history:
            return PerformanceMetrics()
        
        # Basic trade statistics
        trades = [t['pnl'] for t in self.trade_history]
        winning_trades = [t for t in trades if t > 0]
        losing_trades = [t for t in trades if t < 0]
        
        # Portfolio performance
        total_return = 0.0
        annualized_return = 0.0
        
        if len(self.portfolio_values) >= 2:
            start_value = self.portfolio_values[0][1]
            end_value = self.portfolio_values[-1][1]
            total_return = ((end_value / start_value) - 1) * 100 if start_value > 0 else 0
            
            # Annualized return
            days = (self.portfolio_values[-1][0] - self.portfolio_values[0][0]).days
            if days > 0:
                annualized_return = ((end_value / start_value) ** (365 / days) - 1) * 100 if start_value > 0 else 0
        
        # Calculate metrics
        metrics = PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=self.calculate_sharpe_ratio(),
            sortino_ratio=self.calculate_sortino_ratio(),
            win_rate=self.calculate_win_rate(),
            profit_factor=self.calculate_profit_factor(),
            max_drawdown=self.max_drawdown * 100,  # Convert to percentage
            volatility=np.std(self.returns_series) * np.sqrt(252) * 100 if self.returns_series else 0,
            calmar_ratio=self.calculate_calmar_ratio(),
            avg_trade_return=np.mean(trades) if trades else 0,
            best_trade=max(trades) if trades else 0,
            worst_trade=min(trades) if trades else 0,
            trades_count=len(trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            avg_win=np.mean(winning_trades) if winning_trades else 0,
            avg_loss=np.mean(losing_trades) if losing_trades else 0,
            largest_win_streak=self.win_streak,
            largest_loss_streak=self.loss_streak,
            current_streak=max(self.win_streak, self.loss_streak),
            expectancy=(np.mean(winning_trades) * len(winning_trades) - abs(np.mean(losing_trades)) * len(losing_trades)) / len(trades) if trades else 0,
            ulcer_index=self.calculate_ulcer_index()
        )
        
        return metrics
    
    def log_session_summary(self, session_id: str, strategy: str, duration_minutes: float) -> None:
        """Log comprehensive session performance summary"""
        metrics = self.get_comprehensive_metrics()
        
        # Create summary
        summary = {
            'session_id': session_id,
            'strategy': strategy,
            'duration_minutes': duration_minutes,
            'metrics': {
                'total_return_pct': metrics.total_return,
                'annualized_return_pct': metrics.annualized_return,
                'sharpe_ratio': metrics.sharpe_ratio,
                'sortino_ratio': metrics.sortino_ratio,
                'win_rate_pct': metrics.win_rate,
                'profit_factor': metrics.profit_factor,
                'max_drawdown_pct': metrics.max_drawdown,
                'volatility_pct': metrics.volatility,
                'calmar_ratio': metrics.calmar_ratio,
                'trades_count': metrics.trades_count,
                'winning_trades': metrics.winning_trades,
                'losing_trades': metrics.losing_trades,
                'avg_trade_return': metrics.avg_trade_return,
                'best_trade': metrics.best_trade,
                'worst_trade': metrics.worst_trade,
                'avg_win': metrics.avg_win,
                'avg_loss': metrics.avg_loss,
                'expectancy': metrics.expectancy,
                'ulcer_index': metrics.ulcer_index
            }
        }
        
        # Performance assessment
        assessment = self._assess_performance(metrics)
        summary['assessment'] = assessment
        
        # Log comprehensive summary
        self.logger.info("Session performance summary",
                        session_id=session_id,
                        **summary)
        
        # Save to dedicated performance log file
        perf_file = self.log_dir / f"session_performance_{session_id}.json"
        with open(perf_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Performance summary saved: {perf_file}")
    
    def _assess_performance(self, metrics: PerformanceMetrics) -> Dict[str, str]:
        """Assess overall performance quality"""
        assessment = {
            'overall': 'POOR',
            'returns': 'LOW',
            'risk_management': 'POOR', 
            'consistency': 'LOW',
            'recommendations': []
        }
        
        # Returns assessment
        if metrics.total_return > 10:
            assessment['returns'] = 'EXCELLENT'
        elif metrics.total_return > 5:
            assessment['returns'] = 'GOOD'
        elif metrics.total_return > 0:
            assessment['returns'] = 'MODERATE'
        else:
            assessment['returns'] = 'POOR'
        
        # Risk management assessment
        if metrics.max_drawdown < 5 and metrics.sharpe_ratio > 1.5:
            assessment['risk_management'] = 'EXCELLENT'
        elif metrics.max_drawdown < 10 and metrics.sharpe_ratio > 1.0:
            assessment['risk_management'] = 'GOOD'
        elif metrics.max_drawdown < 15:
            assessment['risk_management'] = 'MODERATE'
        else:
            assessment['risk_management'] = 'POOR'
        
        # Consistency assessment
        if metrics.win_rate > 60 and metrics.profit_factor > 1.5:
            assessment['consistency'] = 'HIGH'
        elif metrics.win_rate > 50 and metrics.profit_factor > 1.2:
            assessment['consistency'] = 'MODERATE'
        else:
            assessment['consistency'] = 'LOW'
        
        # Overall assessment
        scores = {'EXCELLENT': 4, 'GOOD': 3, 'MODERATE': 2, 'POOR': 1}
        avg_score = np.mean([
            scores[assessment['returns']],
            scores[assessment['risk_management']],
            scores[assessment['consistency']]
        ])
        
        if avg_score >= 3.5:
            assessment['overall'] = 'EXCELLENT'
        elif avg_score >= 2.5:
            assessment['overall'] = 'GOOD'
        elif avg_score >= 1.5:
            assessment['overall'] = 'MODERATE'
        else:
            assessment['overall'] = 'POOR'
        
        # Recommendations
        recommendations = []
        
        if metrics.max_drawdown > 10:
            recommendations.append("Reduce position sizes to control drawdown")
        
        if metrics.win_rate < 40:
            recommendations.append("Review entry criteria - low win rate")
        
        if metrics.profit_factor < 1.2:
            recommendations.append("Improve risk/reward ratio")
        
        if metrics.sharpe_ratio < 0.5:
            recommendations.append("Focus on risk-adjusted returns")
        
        assessment['recommendations'] = recommendations or ["Continue current approach"]
        
        return assessment
    
    def export_performance_report(self, session_id: str) -> str:
        """Export detailed performance report"""
        metrics = self.get_comprehensive_metrics()
        assessment = self._assess_performance(metrics)
        
        report = {
            'session_id': session_id,
            'generated_at': datetime.now().isoformat(),
            'performance_metrics': metrics.__dict__,
            'assessment': assessment,
            'trade_count': len(self.trade_history),
            'portfolio_evolution': [
                {'timestamp': ts.isoformat(), 'value': val} 
                for ts, val in self.portfolio_values
            ],
            'daily_returns': self.returns_series,
            'drawdown_series': self.drawdown_series
        }
        
        # Save detailed report
        report_file = self.log_dir / f"detailed_report_{session_id}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return str(report_file)


# Global performance logger instance
_performance_logger: Optional[PerformanceLogger] = None


def get_performance_logger(name: str = "supertrader", log_dir: str = "logs") -> PerformanceLogger:
    """Get or create global performance logger instance"""
    global _performance_logger
    
    if _performance_logger is None:
        _performance_logger = PerformanceLogger(name, log_dir)
    
    return _performance_logger


# Convenience functions
def log_trade_performance(trade_id: str, symbol: str, action: str, quantity: int, 
                         price: float, pnl: float, portfolio_value: float) -> None:
    """Log trade to performance logger"""
    logger = get_performance_logger()
    logger.log_trade(trade_id, symbol, action, quantity, price, pnl, portfolio_value)


def get_session_metrics() -> PerformanceMetrics:
    """Get current session performance metrics"""
    logger = get_performance_logger()
    return logger.get_comprehensive_metrics()


def log_session_performance(session_id: str, strategy: str, duration_minutes: float) -> None:
    """Log session performance summary"""
    logger = get_performance_logger()
    logger.log_session_summary(session_id, strategy, duration_minutes)
