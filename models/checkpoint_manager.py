"""
Model Checkpoint Manager - PRODUCTION
Handles saving/loading models with versioning and metadata
"""

import torch
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
import shutil

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manage model checkpoints with versioning and metadata
    - Save best models by metric
    - Version control
    - Config hashing for reproducibility
    - Automatic cleanup of old checkpoints
    """
    
    def __init__(self, checkpoint_dir: str = 'artifacts/checkpoints', max_keep: int = 5):
        """
        Initialize checkpoint manager
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_keep: Maximum number of checkpoints to keep per type
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_keep = max_keep
        
        # Subdirectories
        self.best_dir = self.checkpoint_dir / 'best'
        self.periodic_dir = self.checkpoint_dir / 'periodic'
        self.archive_dir = self.checkpoint_dir / 'archive'
        
        for dir_path in [self.best_dir, self.periodic_dir, self.archive_dir]:
            dir_path.mkdir(exist_ok=True)
        
        logger.info(f"CheckpointManager initialized - Dir: {checkpoint_dir}")
    
    def save_checkpoint(
        self,
        model,
        optimizer,
        metrics: Dict[str, float],
        config: Dict,
        step: int,
        checkpoint_type: str = 'periodic'
    ) -> str:
        """
        Save model checkpoint with full metadata
        
        Args:
            model: PyTorch model (DQN, PPO, A2C)
            optimizer: PyTorch optimizer
            metrics: Performance metrics
            config: Training configuration
            step: Training step/episode number
            checkpoint_type: 'best' or 'periodic'
        
        Returns:
            Path to saved checkpoint
        """
        # Generate config hash
        config_hash = self._hash_config(config)
        
        # Generate filename
        if checkpoint_type == 'best':
            metric_name = 'sharpe'
            metric_value = metrics.get('intraday_sharpe_ratio', 0.0)
            filename = f"best_{metric_name}_{metric_value:.3f}_{config_hash[:8]}.pt"
            save_dir = self.best_dir
        else:
            filename = f"checkpoint_ep{step}_{config_hash[:8]}.pt"
            save_dir = self.periodic_dir
        
        filepath = save_dir / filename
        
        # Prepare checkpoint data
        checkpoint = {
            # Model state
            'model_state_dict': model.state_dict() if hasattr(model, 'state_dict') else model.q_network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            
            # Training state
            'step': step,
            'episode': step,  # For RL, step = episode
            
            # Metrics
            'metrics': metrics,
            
            # Configuration
            'config': config,
            'config_hash': config_hash,
            
            # Metadata
            'timestamp': datetime.now().isoformat(),
            'model_type': config.get('algorithm', {}).get('type', 'DQN'),
            'version': '1.0.0',
            
            # Git info (if available)
            'git_commit': self._get_git_commit(),
            
            # Environment info
            'python_version': self._get_python_version(),
            'torch_version': torch.__version__
        }
        
        # If DQN agent, save target network too
        if hasattr(model, 'target_network'):
            checkpoint['target_network_state_dict'] = model.target_network.state_dict()
            checkpoint['epsilon'] = model.epsilon
            checkpoint['steps'] = model.steps
            checkpoint['episodes'] = model.episodes
        
        # Save checkpoint
        torch.save(checkpoint, filepath)
        
        logger.info(f"Checkpoint saved: {filepath}")
        logger.info(f"  Step: {step} | Sharpe: {metrics.get('intraday_sharpe_ratio', 0):.3f}")
        
        # Cleanup old checkpoints
        if checkpoint_type == 'periodic':
            self._cleanup_old_checkpoints(save_dir)
        
        return str(filepath)
    
    def load_checkpoint(self, filepath: str, model, optimizer=None) -> Dict:
        """
        Load checkpoint and restore model state
        
        Args:
            filepath: Path to checkpoint file
            model: Model to load weights into
            optimizer: Optimizer to load state into (optional)
        
        Returns:
            Checkpoint metadata
        """
        checkpoint = torch.load(filepath, map_location='cpu')
        
        # Load model weights
        if hasattr(model, 'state_dict'):
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.q_network.load_state_dict(checkpoint['model_state_dict'])
        
        # Load target network if exists
        if 'target_network_state_dict' in checkpoint and hasattr(model, 'target_network'):
            model.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            model.epsilon = checkpoint.get('epsilon', 0.01)
            model.steps = checkpoint.get('steps', 0)
            model.episodes = checkpoint.get('episodes', 0)
        
        # Load optimizer
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Checkpoint loaded: {filepath}")
        logger.info(f"  Step: {checkpoint['step']} | Config hash: {checkpoint['config_hash'][:8]}")
        
        return checkpoint
    
    def save_best_model(
        self,
        model,
        optimizer,
        metrics: Dict[str, float],
        config: Dict,
        step: int,
        metric_name: str = 'intraday_sharpe_ratio'
    ) -> Optional[str]:
        """
        Save model if it's the best so far
        
        Returns:
            Path to checkpoint if saved, None otherwise
        """
        current_value = metrics.get(metric_name, -float('inf'))
        
        # Find existing best checkpoint
        existing_best = list(self.best_dir.glob(f'best_{metric_name.split("_")[0]}*.pt'))
        
        if existing_best:
            # Extract best value from filename
            best_checkpoint = max(existing_best, key=lambda p: float(p.stem.split('_')[2]))
            best_value = float(best_checkpoint.stem.split('_')[2])
            
            if current_value <= best_value:
                logger.info(f"Not saving - Current {metric_name}: {current_value:.3f} <= Best: {best_value:.3f}")
                return None
            
            # Archive old best
            logger.info(f"New best model! {current_value:.3f} > {best_value:.3f}")
            shutil.move(str(best_checkpoint), str(self.archive_dir / best_checkpoint.name))
        
        # Save new best
        checkpoint_path = self.save_checkpoint(
            model, optimizer, metrics, config, step, checkpoint_type='best'
        )
        
        return checkpoint_path
    
    def list_checkpoints(self, checkpoint_type: str = 'all') -> List[Dict]:
        """
        List all checkpoints with metadata
        
        Args:
            checkpoint_type: 'best', 'periodic', or 'all'
        
        Returns:
            List of checkpoint info dictionaries
        """
        if checkpoint_type == 'best':
            checkpoint_files = list(self.best_dir.glob('*.pt'))
        elif checkpoint_type == 'periodic':
            checkpoint_files = list(self.periodic_dir.glob('*.pt'))
        else:
            checkpoint_files = list(self.best_dir.glob('*.pt')) + list(self.periodic_dir.glob('*.pt'))
        
        checkpoint_info = []
        
        for filepath in checkpoint_files:
            try:
                checkpoint = torch.load(filepath, map_location='cpu')
                
                info = {
                    'path': str(filepath),
                    'filename': filepath.name,
                    'step': checkpoint.get('step', 0),
                    'timestamp': checkpoint.get('timestamp', ''),
                    'sharpe': checkpoint.get('metrics', {}).get('intraday_sharpe_ratio', 0),
                    'model_type': checkpoint.get('model_type', ''),
                    'config_hash': checkpoint.get('config_hash', '')[:8]
                }
                
                checkpoint_info.append(info)
                
            except Exception as e:
                logger.warning(f"Could not read checkpoint {filepath}: {e}")
        
        # Sort by sharpe descending
        checkpoint_info.sort(key=lambda x: x['sharpe'], reverse=True)
        
        return checkpoint_info
    
    def get_best_checkpoint(self) -> Optional[str]:
        """Get path to best checkpoint by Sharpe ratio"""
        checkpoints = self.list_checkpoints('best')
        
        if not checkpoints:
            logger.warning("No best checkpoints found")
            return None
        
        return checkpoints[0]['path']
    
    def _cleanup_old_checkpoints(self, directory: Path):
        """Remove old checkpoints, keeping only max_keep newest"""
        checkpoints = sorted(
            directory.glob('*.pt'),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        if len(checkpoints) > self.max_keep:
            for old_checkpoint in checkpoints[self.max_keep:]:
                logger.info(f"Removing old checkpoint: {old_checkpoint.name}")
                old_checkpoint.unlink()
    
    def _hash_config(self, config: Dict) -> str:
        """Generate hash of configuration for reproducibility"""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()
    
    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash"""
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None
    
    def _get_python_version(self) -> str:
        """Get Python version"""
        import sys
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


# ==================== EARLY STOPPING ====================

class EarlyStopping:
    """
    Early stopping to prevent overfitting
    Monitors validation metric and stops if no improvement
    """
    
    def __init__(
        self,
        patience: int = 30,
        min_delta: float = 0.001,
        metric_name: str = 'intraday_sharpe_ratio',
        mode: str = 'max'
    ):
        """
        Initialize early stopping
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            metric_name: Metric to monitor
            mode: 'max' (higher is better) or 'min' (lower is better)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.metric_name = metric_name
        self.mode = mode
        
        self.counter = 0
        self.best_value = -float('inf') if mode == 'max' else float('inf')
        self.early_stop = False
        
        logger.info(f"EarlyStopping initialized - Patience: {patience}, Metric: {metric_name}")
    
    def __call__(self, metrics: Dict[str, float]) -> bool:
        """
        Check if training should stop
        
        Args:
            metrics: Current validation metrics
        
        Returns:
            True if should stop, False otherwise
        """
        current_value = metrics.get(self.metric_name, None)
        
        if current_value is None:
            logger.warning(f"Metric {self.metric_name} not found in metrics")
            return False
        
        # Check for improvement
        if self.mode == 'max':
            improved = current_value > (self.best_value + self.min_delta)
        else:
            improved = current_value < (self.best_value - self.min_delta)
        
        if improved:
            self.best_value = current_value
            self.counter = 0
            logger.info(f"Improvement! {self.metric_name}: {current_value:.4f}")
        else:
            self.counter += 1
            logger.info(f"No improvement ({self.counter}/{self.patience}) - Best: {self.best_value:.4f}, Current: {current_value:.4f}")
        
        if self.counter >= self.patience:
            logger.warning(f"Early stopping triggered after {self.counter} epochs without improvement")
            self.early_stop = True
            return True
        
        return False
    
    def reset(self):
        """Reset early stopping state"""
        self.counter = 0
        self.best_value = -float('inf') if self.mode == 'max' else float('inf')
        self.early_stop = False


# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
    # Example: Save and load checkpoint
    
    from models.dqn_network import DoubleDQNAgent
    
    # Create checkpoint manager
    manager = CheckpointManager(checkpoint_dir='artifacts/checkpoints')
    
    # Create dummy agent
    agent = DoubleDQNAgent(
        state_shape=(30, 32),
        action_space=3,
        learning_rate=0.0001,
        gamma=0.3
    )
    
    # Dummy metrics
    metrics = {
        'intraday_sharpe_ratio': 0.652,
        'avg_pnl': 15234.56,
        'win_rate': 0.643,
        'max_drawdown': -4567.89
    }
    
    # Dummy config
    config = {
        'algorithm': {'type': 'DQN'},
        'training': {
            'gamma': 0.3,
            'learning_rate': {'critic': 0.0001}
        }
    }
    
    # Save checkpoint
    checkpoint_path = manager.save_checkpoint(
        model=agent,
        optimizer=agent.optimizer,
        metrics=metrics,
        config=config,
        step=100,
        checkpoint_type='periodic'
    )
    
    print(f"Saved checkpoint: {checkpoint_path}")
    
    # Save best model
    best_path = manager.save_best_model(
        model=agent,
        optimizer=agent.optimizer,
        metrics=metrics,
        config=config,
        step=100
    )
    
    if best_path:
        print(f"Saved best model: {best_path}")
    
    # List checkpoints
    checkpoints = manager.list_checkpoints()
    print(f"\nFound {len(checkpoints)} checkpoints:")
    for ckpt in checkpoints:
        print(f"  {ckpt['filename']} - Sharpe: {ckpt['sharpe']:.3f}")
    
    # Early stopping example
    early_stopping = EarlyStopping(patience=5, metric_name='intraday_sharpe_ratio')
    
    # Simulate training
    for epoch in range(20):
        val_metrics = {
            'intraday_sharpe_ratio': 0.5 + epoch * 0.01 if epoch < 10 else 0.6
        }
        
        if early_stopping(val_metrics):
            print(f"Training stopped early at epoch {epoch}")
            break