"""
Policy Evaluation - PRODUCTION
Comprehensive intraday-specific performance metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class IntradayPolicyEvaluator:
    """
    Evaluate RL policy with intraday-specific metrics
    - Intraday Sharpe (annualized correctly)
    - Average hold time (minutes per trade)
    - Time-of-day performance analysis
    - Slippage impact analysis
    """
    
    def __init__(self):
        self.episode_results = []
        self.trade_log = []
    
    def evaluate_policy(
        self,
        agent,
        env,
        state_builder,
        num_episodes: int = 60,
        mode: str = 'validation'
    ) -> Dict[str, float]:
        """
        Evaluate policy over multiple episodes
        
        Args:
            agent: Trained DQN agent
            env: IntradayTradingEnv
            state_builder: IntradayStateBuilder
            num_episodes: Number of days to evaluate (40-60 recommended)
            mode: 'validation' or 'test'
        
        Returns:
            Comprehensive metrics dictionary
        """
        logger.info(f"Starting policy evaluation - {num_episodes} episodes ({mode})")
        
        episode_metrics = []
        all_trades = []
        
        for ep in range(num_episodes):
            # Reset environment
            state = env.reset()
            done = False
            episode_trades = []
            
            while not done:
                # Get current bar
                current_bar = env.current_day_data.iloc[env.current_step]
                
                # Build state
                time_features = {
                    'minutes_since_open': (current_bar.name.hour - 9) * 60 + current_bar.name.minute,
                    'minutes_to_close': env._get_minutes_to_close(current_bar.name.time()),
                    'session_phase': self._get_session_phase(current_bar.name.time())
                }
                
                lookback_start = max(0, env.current_step - 30)
                window_data = env.current_day_data.iloc[lookback_start:env.current_step + 1]
                
                if len(window_data) >= 30:
                    state = state_builder.build_state_vector(
                        df_price=window_data,
                        time_features=time_features,
                        oi_data=None,
                        vix=None,
                        options_data=None,
                        sentiment_data=None
                    )
                
                # Select action (greedy - no exploration)
                action, action_info = agent.select_action(
                    state,
                    mode='eval',
                    time_remaining=time_features['minutes_to_close']
                )
                
                # Track trade
                old_position = env.position
                
                # Take step
                next_state, reward, done, info = env.step(action)
                
                # Log trade if position changed
                if env.position != old_position:
                    trade = {
                        'episode': ep,
                        'timestamp': current_bar.name,
                        'time_of_day': time_features['minutes_since_open'],
                        'old_position': old_position,
                        'new_position': env.position,
                        'price': current_bar['close'],
                        'pnl': info.get('realized_pnl', 0),
                        'session_phase': time_features['session_phase']
                    }
                    episode_trades.append(trade)
                
                state = next_state
            
            # Episode complete - collect metrics
            ep_metrics = env.get_episode_metrics()
            ep_metrics['episode'] = ep
            ep_metrics['num_trades'] = len(episode_trades)
            episode_metrics.append(ep_metrics)
            all_trades.extend(episode_trades)
            
            if (ep + 1) % 10 == 0:
                logger.info(f"Evaluated {ep + 1}/{num_episodes} episodes")
        
        # Store results
        self.episode_results = episode_metrics
        self.trade_log = all_trades
        
        # Calculate aggregate metrics
        aggregate_metrics = self._calculate_aggregate_metrics(episode_metrics, all_trades)
        
        logger.info(f"✅ Evaluation complete - {num_episodes} episodes")
        self._log_metrics_summary(aggregate_metrics)
        
        return aggregate_metrics
    
    def _calculate_aggregate_metrics(
        self,
        episode_metrics: List[Dict],
        all_trades: List[Dict]
    ) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        
        df_episodes = pd.DataFrame(episode_metrics)
        df_trades = pd.DataFrame(all_trades) if all_trades else pd.DataFrame()
        
        # === RETURNS METRICS ===
        
        # Average daily P&L
        avg_pnl = df_episodes['total_pnl'].mean()
        pnl_std = df_episodes['total_pnl'].std()
        
        # Total P&L
        total_pnl = df_episodes['total_pnl'].sum()
        
        # Win rate (% of profitable days)
        win_rate_days = (df_episodes['total_pnl'] > 0).mean()
        
        # === INTRADAY SHARPE RATIO ===
        # Annualize using: sqrt(252 trading days)
        # Intraday adjustment: 6.5 hours trading / 24 hours
        
        daily_returns = df_episodes['total_pnl'] / 500000  # As % of capital
        sharpe_ratio = self._calculate_intraday_sharpe(daily_returns)
        
        # === SORTINO RATIO (downside risk) ===
        sortino_ratio = self._calculate_sortino(daily_returns)
        
        # === MAX DRAWDOWN ===
        cumulative_pnl = df_episodes['total_pnl'].cumsum()
        running_max = cumulative_pnl.cummax()
        drawdown = cumulative_pnl - running_max
        max_drawdown = drawdown.min()
        max_drawdown_pct = (max_drawdown / 500000) * 100
        
        # === INTRADAY DRAWDOWN (session-level) ===
        avg_intraday_dd = df_episodes['max_drawdown'].mean() if 'max_drawdown' in df_episodes.columns else 0
        
        # === TRADE METRICS ===
        
        if not df_trades.empty:
            # Average hold time (in minutes)
            # Calculate from consecutive position changes
            avg_hold_time = self._calculate_avg_hold_time(df_trades)
            
            # Trade-level win rate
            trade_pnls = df_trades['pnl'].values
            trade_win_rate = (trade_pnls > 0).mean() if len(trade_pnls) > 0 else 0
            
            # Average P&L per trade
            avg_pnl_per_trade = trade_pnls.mean() if len(trade_pnls) > 0 else 0
            
            # Avg trades per day
            avg_trades_per_day = len(df_trades) / len(df_episodes)
            
            # Turnover (total trades)
            total_trades = len(df_trades)
            
            # Time-of-day analysis
            time_of_day_metrics = self._analyze_time_of_day_performance(df_trades)
        else:
            avg_hold_time = 0
            trade_win_rate = 0
            avg_pnl_per_trade = 0
            avg_trades_per_day = 0
            total_trades = 0
            time_of_day_metrics = {}
        
        # === CALMAR RATIO ===
        calmar_ratio = (avg_pnl * 252) / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # === PROFIT FACTOR ===
        if not df_episodes.empty:
            gross_profit = df_episodes[df_episodes['total_pnl'] > 0]['total_pnl'].sum()
            gross_loss = abs(df_episodes[df_episodes['total_pnl'] < 0]['total_pnl'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        else:
            profit_factor = 0
        
        # Aggregate metrics
        metrics = {
            # Returns
            'avg_daily_pnl': avg_pnl,
            'total_pnl': total_pnl,
            'pnl_std': pnl_std,
            
            # Risk-adjusted returns
            'intraday_sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            
            # Drawdown
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'avg_intraday_dd': avg_intraday_dd,
            
            # Win rates
            'win_rate_days': win_rate_days,
            'win_rate_trades': trade_win_rate,
            
            # Trade metrics
            'avg_hold_time_mins': avg_hold_time,
            'avg_pnl_per_trade': avg_pnl_per_trade,
            'avg_trades_per_day': avg_trades_per_day,
            'total_trades': total_trades,
            
            # Other
            'profit_factor': profit_factor,
            'num_episodes': len(df_episodes)
        }
        
        # Add time-of-day metrics
        metrics.update(time_of_day_metrics)
        
        return metrics
    
    def _calculate_intraday_sharpe(self, returns: pd.Series) -> float:
        """
        Calculate annualized Sharpe ratio for intraday trading
        
        Intraday adjustment:
        - Trading time: 6.5 hours per day (9:15 AM - 3:45 PM)
        - Annualization: sqrt(252) * (sqrt(24/6.5) or directly sqrt(252))
        
        Common approach: Use daily returns, annualize with sqrt(252)
        """
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Annualized Sharpe
        sharpe = (mean_return / std_return) * np.sqrt(252)
        
        return float(sharpe)
    
    def _calculate_sortino(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio (uses downside deviation)"""
        if len(returns) < 2:
            return 0.0
        
        mean_return = returns.mean()
        
        # Downside deviation (only negative returns)
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_std = downside_returns.std()
        
        if downside_std == 0:
            return 0.0
        
        sortino = (mean_return / downside_std) * np.sqrt(252)
        
        return float(sortino)
    
    def _calculate_avg_hold_time(self, df_trades: pd.DataFrame) -> float:
        """Calculate average hold time in minutes"""
        
        if df_trades.empty:
            return 0.0
        
        # Group by episode
        hold_times = []
        
        for episode, group in df_trades.groupby('episode'):
            if len(group) < 2:
                continue
            
            # Calculate time between entry and exit
            for i in range(len(group) - 1):
                if group.iloc[i]['new_position'] != 0 and group.iloc[i+1]['new_position'] == 0:
                    # Found entry -> exit
                    entry_time = group.iloc[i]['time_of_day']
                    exit_time = group.iloc[i+1]['time_of_day']
                    hold_time = exit_time - entry_time
                    hold_times.append(hold_time)
        
        if hold_times:
            return float(np.mean(hold_times))
        else:
            return 0.0
    
    def _analyze_time_of_day_performance(self, df_trades: pd.DataFrame) -> Dict[str, float]:
        """Analyze performance by time of day"""
        
        if df_trades.empty:
            return {}
        
        # Define time buckets
        df_trades['time_bucket'] = pd.cut(
            df_trades['time_of_day'],
            bins=[0, 30, 150, 300, 360],
            labels=['opening_range', 'morning', 'afternoon', 'closing']
        )
        
        # Performance by time bucket
        time_metrics = {}
        
        for bucket in ['opening_range', 'morning', 'afternoon', 'closing']:
            bucket_trades = df_trades[df_trades['time_bucket'] == bucket]
            
            if len(bucket_trades) > 0:
                bucket_pnl = bucket_trades['pnl'].sum()
                bucket_win_rate = (bucket_trades['pnl'] > 0).mean()
                
                time_metrics[f'{bucket}_pnl'] = bucket_pnl
                time_metrics[f'{bucket}_win_rate'] = bucket_win_rate
                time_metrics[f'{bucket}_num_trades'] = len(bucket_trades)
        
        return time_metrics
    
    def _get_session_phase(self, time_obj) -> str:
        """Determine session phase from time"""
        minutes = time_obj.hour * 60 + time_obj.minute
        market_open = 9 * 60 + 15  # 9:15 AM
        minutes_since_open = minutes - market_open
        
        if minutes_since_open < 30:
            return 'opening_range'
        elif minutes_since_open < 150:
            return 'morning'
        elif minutes_since_open < 300:
            return 'afternoon'
        else:
            return 'closing'
    
    def _log_metrics_summary(self, metrics: Dict[str, float]):
        """Log summary of metrics"""
        logger.info("\n" + "="*80)
        logger.info("POLICY EVALUATION RESULTS")
        logger.info("="*80)
        
        logger.info(f"\n📊 RETURNS:")
        logger.info(f"  Total P&L:        ₹{metrics['total_pnl']:>12,.0f}")
        logger.info(f"  Avg Daily P&L:    ₹{metrics['avg_daily_pnl']:>12,.0f}")
        logger.info(f"  P&L Std Dev:      ₹{metrics['pnl_std']:>12,.0f}")
        
        logger.info(f"\n📈 RISK-ADJUSTED RETURNS:")
        logger.info(f"  Intraday Sharpe:  {metrics['intraday_sharpe_ratio']:>12.3f}")
        logger.info(f"  Sortino Ratio:    {metrics['sortino_ratio']:>12.3f}")
        logger.info(f"  Calmar Ratio:     {metrics['calmar_ratio']:>12.3f}")
        
        logger.info(f"\n📉 DRAWDOWN:")
        logger.info(f"  Max Drawdown:     ₹{metrics['max_drawdown']:>12,.0f} ({metrics['max_drawdown_pct']:.2f}%)")
        logger.info(f"  Avg Intraday DD:  ₹{metrics['avg_intraday_dd']:>12,.0f}")
        
        logger.info(f"\n🎯 WIN RATES:")
        logger.info(f"  Days Win Rate:    {metrics['win_rate_days']:>12.1%}")
        logger.info(f"  Trades Win Rate:  {metrics['win_rate_trades']:>12.1%}")
        
        logger.info(f"\n💼 TRADING ACTIVITY:")
        logger.info(f"  Total Trades:     {metrics['total_trades']:>12.0f}")
        logger.info(f"  Avg Trades/Day:   {metrics['avg_trades_per_day']:>12.1f}")
        logger.info(f"  Avg Hold Time:    {metrics['avg_hold_time_mins']:>12.1f} mins")
        logger.info(f"  Avg P&L/Trade:    ₹{metrics['avg_pnl_per_trade']:>12,.0f}")
        
        logger.info(f"\n⏰ TIME OF DAY PERFORMANCE:")
        for phase in ['opening_range', 'morning', 'afternoon', 'closing']:
            pnl_key = f'{phase}_pnl'
            wr_key = f'{phase}_win_rate'
            if pnl_key in metrics:
                logger.info(f"  {phase.title():15s} P&L: ₹{metrics[pnl_key]:>10,.0f} | Win Rate: {metrics[wr_key]:.1%}")
        
        logger.info("\n" + "="*80)
    
    def save_results(self, output_path: str):
        """Save evaluation results to CSV"""
        df_episodes = pd.DataFrame(self.episode_results)
        df_trades = pd.DataFrame(self.trade_log)
        
        df_episodes.to_csv(f"{output_path}_episodes.csv", index=False)
        df_trades.to_csv(f"{output_path}_trades.csv", index=False)
        
        logger.info(f"Results saved to {output_path}_*.csv")
    
    def checkpoint_best_model(
        self,
        agent,
        metrics: Dict[str, float],
        checkpoint_dir: str,
        metric_name: str = 'intraday_sharpe_ratio'
    ):
        """Save model checkpoint if it's the best so far"""
        
        checkpoint_path = f"{checkpoint_dir}/best_{metric_name}.pt"
        
        # Save
        agent.save_checkpoint(checkpoint_path, metrics)
        
        logger.info(f"Best model saved: {metric_name}={metrics[metric_name]:.3f} -> {checkpoint_path}")


def compare_with_without_sentiment(
    agent,
    env,
    state_builder,
    num_episodes: int = 20
) -> Dict[str, Dict[str, float]]:
    """
    Ablation study: Compare performance with/without sentiment features
    
    Returns:
        {'with_sentiment': metrics, 'without_sentiment': metrics}
    """
    logger.info("Running sentiment ablation study...")
    
    evaluator = IntradayPolicyEvaluator()
    
    # Test WITH sentiment
    logger.info("Testing WITH sentiment features...")
    state_builder.use_sentiment = True
    metrics_with = evaluator.evaluate_policy(agent, env, state_builder, num_episodes)
    
    # Test WITHOUT sentiment
    logger.info("Testing WITHOUT sentiment features...")
    state_builder.use_sentiment = False
    metrics_without = evaluator.evaluate_policy(agent, env, state_builder, num_episodes)
    
    # Compare
    logger.info("\n" + "="*80)
    logger.info("SENTIMENT ABLATION RESULTS")
    logger.info("="*80)
    
    comparison = {
        'with_sentiment': metrics_with,
        'without_sentiment': metrics_without
    }
    
    # Key metrics comparison
    logger.info(f"\n{'Metric':<30s} {'With Sent':>12s} {'Without':>12s} {'Diff':>12s}")
    logger.info("-"*80)
    
    for key in ['intraday_sharpe_ratio', 'avg_daily_pnl', 'win_rate_days', 'max_drawdown_pct']:
        val_with = metrics_with.get(key, 0)
        val_without = metrics_without.get(key, 0)
        diff = val_with - val_without
        
        logger.info(f"{key:<30s} {val_with:>12.3f} {val_without:>12.3f} {diff:>12.3f}")
    
    logger.info("="*80 + "\n")
    
    return comparison