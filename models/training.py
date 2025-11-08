"""
================================================================================================
SIMPLIFIED TRAINING SCRIPT - HOW THE AI LEARNS
================================================================================================

This is where the magic happens - the AI learns to trade!

TRAINING PROCESS:
1. Agent observes market (gets state)
2. Agent decides action (buy/sell/hold)
3. Market responds (price changes, P&L updates)
4. Agent gets reward (positive for profit, negative for loss)
5. Agent learns from experience (updates neural network)
6. Repeat for many episodes (trading days)

TRAINING LOOP:
  for each episode (trading day):
      reset market
      while not done (until 3:15 PM):
          observe state
          decide action
          execute action
          get reward
          remember experience
          learn from batch of experiences
      
      evaluate performance
      save if best model

KEY CONCEPTS:
- Episode = One trading day (9:15 AM - 3:15 PM)
- Experience Replay = Learn from past experiences, not just current one
- Epsilon-Greedy = Sometimes explore randomly, sometimes exploit knowledge
- Target Network = Provides stable learning targets
================================================================================================
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
import logging
from typing import Dict, List
from tqdm import tqdm

# Import our simplified components
from .dqn_network import TradingAgent
from .intraday_environment import IntradayMarket

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """
    Handles the training process
    
    Orchestrates:
    - Agent (the AI brain)
    - Environment (market simulator)
    - Training loop
    - Performance evaluation
    - Model checkpointing
    """
    
    def __init__(
        self,
        agent: TradingAgent,
        env: IntradayMarket,
        episodes: int = 100,
        batch_size: int = 32,
        eval_frequency: int = 10,
        save_dir: str = 'models'
    ):
        """
        Initialize trainer
        
        Args:
            agent: The trading AI
            env: The market simulator
            episodes: Number of training days
            batch_size: How many experiences to learn from at once
            eval_frequency: Evaluate every N episodes
            save_dir: Where to save trained models
        """
        self.agent = agent
        self.env = env
        self.episodes = episodes
        self.batch_size = batch_size
        self.eval_frequency = eval_frequency
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Training metrics
        self.episode_rewards = []
        self.episode_pnls = []
        self.episode_lengths = []
        self.losses = []
        
        # Best model tracking
        self.best_sharpe = -float('inf')
        
        logger.info("Trainer initialized ✓")
    
    def train(self):
        """
        Main training loop
        
        This is where the AI learns to trade over many episodes
        """
        logger.info(f"Starting training for {self.episodes} episodes...")
        
        for episode in range(self.episodes):
            # === EPISODE START ===
            state = self.env.reset()
            done = False
            episode_reward = 0
            episode_steps = 0
            
            # Progress bar for this episode
            pbar = tqdm(total=len(self.env.current_day_data), desc=f"Episode {episode+1}/{self.episodes}")
            
            # === EPISODE LOOP (ONE TRADING DAY) ===
            while not done:
                # Get current time info
                current_bar = self.env.current_day_data.iloc[self.env.current_step]
                current_time = current_bar.name.time() if hasattr(current_bar.name, 'time') else None
                minutes_to_close = self.env._minutes_until_close(current_time) if current_time else 180
                
                # === AGENT DECIDES ACTION ===
                action, action_info = self.agent.decide_action(
                    state, 
                    mode='train',
                    minutes_to_close=minutes_to_close
                )
                
                # === EXECUTE ACTION IN MARKET ===
                next_state, reward, done, info = self.env.step(action)
                
                # === REMEMBER THIS EXPERIENCE ===
                self.agent.remember(state, action, reward, next_state, done)
                
                # === LEARN FROM PAST EXPERIENCES ===
                if len(self.agent.memory) >= self.batch_size:
                    metrics = self.agent.learn(self.batch_size)
                    if metrics:
                        self.losses.append(metrics['loss'])
                
                # Update for next step
                state = next_state
                episode_reward += reward
                episode_steps += 1
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'P&L': f"₹{info['total_pnl']:,.0f}",
                    'Pos': info['position'],
                    'ε': f"{self.agent.epsilon:.3f}"
                })
            
            pbar.close()
            
            # === EPISODE END ===
            summary = self.env.get_episode_summary()
            
            # Store metrics
            self.episode_rewards.append(episode_reward)
            self.episode_pnls.append(summary['total_pnl'])
            self.episode_lengths.append(episode_steps)
            
            # Log episode summary
            logger.info(f"\nEpisode {episode+1} Summary:")
            logger.info(f"  Total Reward: {episode_reward:.2f}")
            logger.info(f"  P&L: ₹{summary['total_pnl']:,.2f} ({summary['pnl_pct']:.2f}%)")
            logger.info(f"  Trades: {summary['num_trades']}")
            logger.info(f"  Max Drawdown: ₹{summary['max_drawdown']:,.2f}")
            logger.info(f"  Epsilon: {self.agent.epsilon:.3f}")
            
            # === PERIODIC EVALUATION ===
            if (episode + 1) % self.eval_frequency == 0:
                eval_metrics = self._evaluate()
                
                # Save if best model
                if eval_metrics['sharpe'] > self.best_sharpe:
                    self.best_sharpe = eval_metrics['sharpe']
                    self._save_best_model(episode, eval_metrics)
            
            # Move to next day
            self.env.current_day_idx += 1
        
        logger.info("\n" + "="*80)
        logger.info("TRAINING COMPLETE!")
        logger.info("="*80)
        self._plot_training_progress()
    
    def _evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate the agent's performance
        
        Runs the agent in evaluation mode (no exploration, greedy only)
        over multiple episodes to assess performance
        
        Args:
            num_episodes: Number of episodes to evaluate over
        
        Returns:
            metrics: Performance metrics (Sharpe, win rate, etc.)
        """
        logger.info(f"\n{'='*60}")
        logger.info("EVALUATION")
        logger.info(f"{'='*60}")
        
        eval_pnls = []
        eval_trades = []
        
        for ep in range(num_episodes):
            state = self.env.reset()
            done = False
            
            while not done:
                # Get time info
                current_bar = self.env.current_day_data.iloc[self.env.current_step]
                current_time = current_bar.name.time() if hasattr(current_bar.name, 'time') else None
                minutes_to_close = self.env._minutes_until_close(current_time) if current_time else 180
                
                # Greedy action (no exploration)
                action, _ = self.agent.decide_action(
                    state,
                    mode='eval',
                    minutes_to_close=minutes_to_close
                )
                
                next_state, reward, done, info = self.env.step(action)
                state = next_state
            
            summary = self.env.get_episode_summary()
            eval_pnls.append(summary['total_pnl'])
            eval_trades.append(summary['num_trades'])
            
            self.env.current_day_idx += 1
        
        # Calculate metrics
        pnls = np.array(eval_pnls)
        
        # Average P&L
        avg_pnl = pnls.mean()
        
        # Win rate (% of profitable days)
        win_rate = (pnls > 0).mean()
        
        # Sharpe ratio (annualized)
        returns = pnls / self.env.initial_capital
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        
        # Max drawdown
        cumulative = pnls.cumsum()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        max_dd = drawdown.min()
        
        metrics = {
            'avg_pnl': avg_pnl,
            'win_rate': win_rate,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'avg_trades': np.mean(eval_trades)
        }
        
        # Log results
        logger.info(f"\nEvaluation Results ({num_episodes} episodes):")
        logger.info(f"  Avg P&L: ₹{avg_pnl:,.2f}")
        logger.info(f"  Win Rate: {win_rate:.1%}")
        logger.info(f"  Sharpe Ratio: {sharpe:.3f}")
        logger.info(f"  Max Drawdown: ₹{max_dd:,.2f}")
        logger.info(f"  Avg Trades/Day: {metrics['avg_trades']:.1f}")
        logger.info(f"{'='*60}\n")
        
        return metrics
    
    def _save_best_model(self, episode: int, metrics: Dict):
        """Save the best performing model"""
        filepath = self.save_dir / f"best_model_sharpe_{metrics['sharpe']:.3f}.pt"
        self.agent.save(str(filepath))
        logger.info(f"✅ New best model saved! Sharpe: {metrics['sharpe']:.3f}")
    
    def _plot_training_progress(self):
        """Plot training metrics"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Episode P&Ls
        axes[0, 0].plot(self.episode_pnls)
        axes[0, 0].set_title('Episode P&L')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('P&L (₹)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Cumulative P&L
        cumulative_pnl = np.cumsum(self.episode_pnls)
        axes[0, 1].plot(cumulative_pnl)
        axes[0, 1].set_title('Cumulative P&L')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Cumulative P&L (₹)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Training Loss
        if self.losses:
            axes[1, 0].plot(self.losses)
            axes[1, 0].set_title('Training Loss')
            axes[1, 0].set_xlabel('Training Step')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Episode Lengths
        axes[1, 1].plot(self.episode_lengths)
        axes[1, 1].set_title('Episode Length (Steps)')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Steps')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.save_dir / 'training_progress.png'
        plt.savefig(plot_path, dpi=150)
        logger.info(f"Training plots saved to {plot_path}")
        plt.close()


# ==================== QUICK START EXAMPLE ====================

if __name__ == "__main__":
    """
    Complete training example
    """
    
    # === STEP 1: Create dummy data ===
    logger.info("Creating dummy market data...")
    
    # Generate 30 days of 1-minute data
    all_data = []
    for day in range(30):
        date_str = f"2025-{10:02d}-{day+1:02d}"
        timestamps = pd.date_range(f'{date_str} 09:15', f'{date_str} 15:15', freq='1min')
        
        # Random walk price
        base_price = 22000
        prices = base_price + np.cumsum(np.random.randn(len(timestamps)) * 5)
        
        day_data = pd.DataFrame({
            'open': prices + np.random.randn(len(timestamps)) * 2,
            'high': prices + abs(np.random.randn(len(timestamps))) * 5,
            'low': prices - abs(np.random.randn(len(timestamps))) * 5,
            'close': prices,
            'volume': np.random.randint(1000, 10000, len(timestamps))
        }, index=timestamps)
        
        all_data.append(day_data)
    
    data = pd.concat(all_data)
    logger.info(f"Created {len(data)} bars across {len(all_data)} days")
    
    # === STEP 2: Create agent ===
    logger.info("Creating trading agent...")
    agent = TradingAgent(
        num_features=32,
        lookback_period=30,
        learning_rate=0.0001,
        gamma=0.3,
        epsilon_start=0.5,
        epsilon_end=0.01
    )
    
    # === STEP 3: Create environment ===
    logger.info("Creating market environment...")
    env = IntradayMarket(
        data=data,
        initial_capital=500000,
        lot_size=50,
        max_lots=5
    )
    
    # === STEP 4: Create trainer ===
    logger.info("Creating trainer...")
    trainer = Trainer(
        agent=agent,
        env=env,
        episodes=20,  # Train for 20 days
        batch_size=32,
        eval_frequency=5
    )
    
    # === STEP 5: Train! ===
    trainer.train()
    
    logger.info("\n🎉 Training complete! Check the 'models' folder for saved models.")