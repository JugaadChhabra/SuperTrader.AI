"""
Training Script - PRODUCTION
Train DQN agent on historical intraday data
Episode = Single trading day (9:15 AM - 3:15 PM)
"""

import argparse
import logging
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import torch

# Import our modules
from models.dqn_network import DoubleDQNAgent, ReplayBuffer, benchmark_inference_speed
from models.intraday_environment import IntradayTradingEnv, create_intraday_env
from data.state_builder import IntradayStateBuilder
from agents.data_agent import fetch_ohlcv, compute_indicators

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)


class IntradayTrainer:
    """
    Trainer for intraday DQN agent
    """
    
    def __init__(self, config_path: str):
        """Initialize trainer with configuration"""
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        logger.info(f"Loaded config from {config_path}")
        
        # Setup paths
        self.artifact_dir = Path('artifacts/training')
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_dir = Path('artifacts/checkpoints')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components (will be set in setup())
        self.train_env = None
        self.val_env = None
        self.agent = None
        self.replay_buffer = None
        self.state_builder = None
        
        # Training state
        self.episode = 0
        self.best_val_sharpe = -np.inf
        self.patience_counter = 0
        
        logger.info("IntradayTrainer initialized")
    
    def setup(self):
        """Setup training components"""
        logger.info("Setting up training components...")
        
        # 1. Load and prepare data
        train_data, val_data = self._load_and_prepare_data()
        
        # 2. Create environments
        env_config = self.config.get('environment', {})
        self.train_env = create_intraday_env(train_data, env_config)
        self.val_env = create_intraday_env(val_data, env_config)
        
        logger.info(f"Train days: {len(self.train_env.trading_days)}, Val days: {len(self.val_env.trading_days)}")
        
        # 3. Create state builder
        self.state_builder = IntradayStateBuilder(
            lookback=self.config['state']['lookback_window'],
            bar_interval=self.config['state']['bar_interval']
        )
        
        # 4. Create DQN agent
        agent_config = {
            'lookback': self.config['state']['lookback_window'],
            'num_features': 32,  # From state builder
            'action_space': 3,
            'learning_rate': self.config['training']['learning_rate']['critic'],
            'gamma': self.config['training']['gamma'],
            'epsilon_start': self.config['training']['epsilon']['start'],
            'epsilon_end': self.config['training']['epsilon']['end'],
            'epsilon_decay': self.config['training']['epsilon']['decay_episodes'],
            'target_update_freq': self.config['training']['target_network']['update_frequency'],
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
        self.agent = DoubleDQNAgent(**agent_config)
        
        # 5. Create replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=self.config['training']['replay_buffer']['size']
        )
        
        # 6. Benchmark inference speed
        inference_time = benchmark_inference_speed(self.agent)
        if inference_time > 50:
            logger.warning("⚠️ Model may be too slow for live intraday trading!")
        
        logger.info("✅ Setup complete")
    
    def _load_and_prepare_data(self) -> tuple:
        """Load and prepare historical data"""
        logger.info("Loading historical data...")
        
        # TODO: Load from your data source
        # For now, create synthetic data for demonstration
        
        # In production, you would:
        # 1. Load 1-min OHLCV data for NIFTY
        # 2. Compute technical indicators
        # 3. Split into train/val
        
        # Placeholder: Create synthetic data
        dates = pd.date_range('2024-01-01', '2024-12-31', freq='1min')
        dates = dates[(dates.hour >= 9) & (dates.hour < 15)]  # Market hours only
        
        data = pd.DataFrame({
            'open': np.random.randn(len(dates)).cumsum() + 22000,
            'high': np.random.randn(len(dates)).cumsum() + 22050,
            'low': np.random.randn(len(dates)).cumsum() + 21950,
            'close': np.random.randn(len(dates)).cumsum() + 22000,
            'volume': np.random.randint(1000, 5000, len(dates))
        }, index=dates)
        
        # Compute indicators
        data = compute_indicators(data)
        
        # Split: 80% train, 20% validation
        split_idx = int(len(data) * 0.8)
        train_data = data.iloc[:split_idx]
        val_data = data.iloc[split_idx:]
        
        logger.info(f"Data loaded - Train: {len(train_data)} bars, Val: {len(val_data)} bars")
        
        return train_data, val_data
    
    def train(self):
        """Main training loop"""
        logger.info("=" * 80)
        logger.info("STARTING TRAINING")
        logger.info("=" * 80)
        
        max_episodes = self.config['training']['training_episodes']
        batch_size = self.config['training']['replay_buffer']['batch_size']
        early_stopping_patience = self.config['training']['early_stopping']['patience']
        
        # Training loop
        for episode in range(max_episodes):
            self.episode = episode
            
            # Train episode
            episode_metrics = self._train_episode()
            
            # Validation every 10 episodes
            if (episode + 1) % 10 == 0:
                val_metrics = self._validate()
                
                # Check for improvement
                if val_metrics['sharpe_ratio'] > self.best_val_sharpe:
                    self.best_val_sharpe = val_metrics['sharpe_ratio']
                    self.patience_counter = 0
                    
                    # Save best model
                    checkpoint_path = self.checkpoint_dir / f"best_model.pt"
                    self.agent.save_checkpoint(str(checkpoint_path), val_metrics)
                    logger.info(f"✅ New best model saved! Sharpe: {self.best_val_sharpe:.3f}")
                else:
                    self.patience_counter += 1
                
                # Log validation
                logger.info(f"\n{'='*80}")
                logger.info(f"VALIDATION - Episode {episode + 1}")
                logger.info(f"{'='*80}")
                logger.info(f"Sharpe: {val_metrics['sharpe_ratio']:.3f} (Best: {self.best_val_sharpe:.3f})")
                logger.info(f"PnL: ₹{val_metrics['avg_pnl']:,.0f} | Win Rate: {val_metrics['win_rate']*100:.1f}%")
                logger.info(f"Patience: {self.patience_counter}/{early_stopping_patience}")
                logger.info(f"{'='*80}\n")
                
                # Early stopping
                if self.patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered after {episode + 1} episodes")
                    break
            
            # Save checkpoint every 50 episodes
            if (episode + 1) % 50 == 0:
                checkpoint_path = self.checkpoint_dir / f"checkpoint_ep{episode+1}.pt"
                self.agent.save_checkpoint(str(checkpoint_path), episode_metrics)
        
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Best Validation Sharpe: {self.best_val_sharpe:.3f}")
        logger.info(f"Total Episodes: {self.episode + 1}")
        logger.info("=" * 80)
    
    def _train_episode(self) -> Dict[str, float]:
        """Train for one episode (one trading day)"""
        
        # Reset environment
        state = self.train_env.reset()
        done = False
        episode_reward = 0
        episode_loss = []
        
        while not done:
            # Get current bar for state building
            current_bar = self.train_env.current_day_data.iloc[self.train_env.current_step]
            
            # Build state using state builder
            time_features = {
                'minutes_since_open': (current_bar.name.hour - 9) * 60 + current_bar.name.minute,
                'minutes_to_close': self.train_env._get_minutes_to_close(current_bar.name.time()),
                'session_phase': 'morning'  # Simplified
            }
            
            # Get window of data for state building
            lookback_start = max(0, self.train_env.current_step - 30)
            window_data = self.train_env.current_day_data.iloc[lookback_start:self.train_env.current_step + 1]
            
            if len(window_data) >= 30:
                state = self.state_builder.build_state_vector(
                    df_price=window_data,
                    time_features=time_features,
                    oi_data=None,
                    vix=None,
                    options_data=None,
                    sentiment_data=None
                )
            
            # Select action
            action, action_info = self.agent.select_action(
                state,
                mode='train',
                time_remaining=time_features['minutes_to_close']
            )
            
            # Take step
            next_state, reward, done, info = self.train_env.step(action)
            
            # Store experience
            self.replay_buffer.push(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                time_of_day=time_features['minutes_since_open']
            )
            
            episode_reward += reward
            
            # Train if enough samples
            if len(self.replay_buffer) >= self.config['training']['replay_buffer']['batch_size']:
                batch = self.replay_buffer.sample(self.config['training']['replay_buffer']['batch_size'])
                metrics = self.agent.train_step(batch)
                episode_loss.append(metrics['loss'])
            
            state = next_state
        
        # Episode metrics
        env_metrics = self.train_env.get_episode_metrics()
        
        episode_metrics = {
            'episode': self.episode,
            'reward': episode_reward,
            'loss': np.mean(episode_loss) if episode_loss else 0.0,
            'pnl': env_metrics.get('total_pnl', 0),
            'sharpe': env_metrics.get('sharpe_ratio', 0),
            'num_trades': env_metrics.get('num_trades', 0),
            'epsilon': self.agent.epsilon
        }
        
        # Log every 10 episodes
        if (self.episode + 1) % 10 == 0:
            logger.info(f"Ep {self.episode + 1:4d} | "
                       f"PnL: ₹{episode_metrics['pnl']:>8,.0f} | "
                       f"Reward: {episode_metrics['reward']:>6.2f} | "
                       f"Loss: {episode_metrics['loss']:>6.4f} | "
                       f"ε: {episode_metrics['epsilon']:.3f} | "
                       f"Trades: {episode_metrics['num_trades']:2d}")
        
        return episode_metrics
    
    def _validate(self) -> Dict[str, float]:
        """Validate agent on validation set"""
        
        val_pnls = []
        val_sharpes = []
        val_win_rates = []
        val_num_trades = []
        
        # Run through all validation days
        num_val_days = min(20, len(self.val_env.trading_days))
        
        for _ in range(num_val_days):
            state = self.val_env.reset()
            done = False
            
            while not done:
                # Get current bar
                current_bar = self.val_env.current_day_data.iloc[self.val_env.current_step]
                
                # Build state
                time_features = {
                    'minutes_since_open': (current_bar.name.hour - 9) * 60 + current_bar.name.minute,
                    'minutes_to_close': self.val_env._get_minutes_to_close(current_bar.name.time()),
                    'session_phase': 'morning'
                }
                
                lookback_start = max(0, self.val_env.current_step - 30)
                window_data = self.val_env.current_day_data.iloc[lookback_start:self.val_env.current_step + 1]
                
                if len(window_data) >= 30:
                    state = self.state_builder.build_state_vector(
                        df_price=window_data,
                        time_features=time_features,
                        oi_data=None,
                        vix=None,
                        options_data=None,
                        sentiment_data=None
                    )
                
                # Select action (greedy - no exploration)
                action, _ = self.agent.select_action(
                    state,
                    mode='eval',
                    time_remaining=time_features['minutes_to_close']
                )
                
                # Take step
                next_state, reward, done, info = self.val_env.step(action)
                state = next_state
            
            # Collect metrics
            metrics = self.val_env.get_episode_metrics()
            val_pnls.append(metrics.get('total_pnl', 0))
            val_sharpes.append(metrics.get('sharpe_ratio', 0))
            val_win_rates.append(metrics.get('win_rate', 0))
            val_num_trades.append(metrics.get('num_trades', 0))
        
        # Aggregate
        val_metrics = {
            'avg_pnl': np.mean(val_pnls),
            'sharpe_ratio': np.mean(val_sharpes),
            'win_rate': np.mean(val_win_rates),
            'avg_trades': np.mean(val_num_trades),
            'pnl_std': np.std(val_pnls)
        }
        
        return val_metrics


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Train Intraday DQN Agent')
    parser.add_argument('--config', type=str, default='configs/rl.yaml',
                       help='Path to config file')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for training')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    # Create trainer
    trainer = IntradayTrainer(args.config)
    trainer.setup()
    
    # Resume if checkpoint provided
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.agent.load_checkpoint(args.resume)
    
    # Train
    trainer.train()
    
    logger.info("Training script completed successfully!")


if __name__ == "__main__":
    main()