"""
DQN Network - PRODUCTION
Smaller, faster architecture optimized for intraday trading
- 2-layer LSTM (32, 16) for <50ms inference
- Double DQN + Dueling DQN
- Target network with 500-step updates
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DuelingDQN(nn.Module):
    """
    Dueling DQN Architecture for intraday trading
    Separates state value and action advantages
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size_1: int = 32,
        hidden_size_2: int = 16,
        action_space: int = 3,  # {-1, 0, 1}
        dropout: float = 0.2
    ):
        super(DuelingDQN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.action_space = action_space
        
        # LSTM layers (smaller for faster inference)
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size_1,
            batch_first=True,
            dropout=dropout if hidden_size_2 > 0 else 0
        )
        
        self.lstm2 = nn.LSTM(
            input_size=hidden_size_1,
            hidden_size=hidden_size_2,
            batch_first=True
        )
        
        # Dueling streams
        # Value stream: V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size_2, 16),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(16, 1)
        )
        
        # Advantage stream: A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size_2, 16),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(16, action_space)
        )
        
        logger.info(f"DuelingDQN initialized - Input: {input_size}, LSTM: [{hidden_size_1}, {hidden_size_2}], Actions: {action_space}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (batch, lookback, features)
        
        Returns:
            Q-values: (batch, action_space)
        """
        batch_size = x.size(0)
        
        # LSTM layers
        lstm1_out, _ = self.lstm1(x)
        lstm2_out, (h_n, c_n) = self.lstm2(lstm1_out)
        
        # Take final hidden state
        final_hidden = h_n[-1]  # (batch, hidden_size_2)
        
        # Dueling streams
        value = self.value_stream(final_hidden)  # (batch, 1)
        advantages = self.advantage_stream(final_hidden)  # (batch, action_space)
        
        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values


class DoubleDQNAgent:
    """
    Double DQN Agent with Experience Replay
    Optimized for intraday trading
    """
    
    def __init__(
        self,
        state_shape: Tuple[int, int],  # (lookback, features)
        action_space: int = 3,
        learning_rate: float = 0.0001,
        gamma: float = 0.3,  # Shorter horizon for intraday
        epsilon_start: float = 0.5,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 100,
        target_update_freq: int = 500,  # Faster updates for intraday
        device: str = 'cpu'
    ):
        self.lookback, self.input_size = state_shape
        self.action_space = action_space
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)
        
        # Networks
        self.q_network = DuelingDQN(
            input_size=self.input_size,
            action_space=action_space
        ).to(self.device)
        
        self.target_network = DuelingDQN(
            input_size=self.input_size,
            action_space=action_space
        ).to(self.device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Training counters
        self.steps = 0
        self.episodes = 0
        
        logger.info(f"DoubleDQNAgent initialized - Gamma: {gamma}, Epsilon: {epsilon_start}→{epsilon_end}")
    
    def select_action(
        self,
        state: np.ndarray,
        mode: str = 'train',
        time_remaining: float = 180.0
    ) -> Tuple[int, Dict[str, float]]:
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: State array (lookback, features)
            mode: 'train' (with exploration) or 'eval' (greedy)
            time_remaining: Minutes to market close
        
        Returns:
            action: Integer action {0, 1, 2} → {-1, 0, 1}
            metadata: Q-values and confidence
        """
        # Time-based constraints (no new positions after 3:00 PM)
        if time_remaining < 15:
            action = 1  # Hold (action index)
            q_values = [0.0, 1.0, 0.0]
            return action, {'q_values': q_values, 'epsilon': 0.0, 'mode': 'time_constrained'}
        
        # Epsilon-greedy exploration
        if mode == 'train' and np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
            q_values = [0.0] * self.action_space
            return action, {'q_values': q_values, 'epsilon': self.epsilon, 'mode': 'explore'}
        
        # Greedy action selection
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # (1, lookback, features)
            q_vals = self.q_network(state_tensor).cpu().numpy()[0]
            action = int(np.argmax(q_vals))
            q_values = q_vals.tolist()
        
        return action, {'q_values': q_values, 'epsilon': self.epsilon, 'mode': 'exploit'}
    
    def train_step(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Perform one training step (Double DQN update)
        
        Args:
            batch: Dictionary with 'states', 'actions', 'rewards', 'next_states', 'dones'
        
        Returns:
            metrics: Training metrics
        """
        # Convert batch to tensors
        states = torch.FloatTensor(batch['states']).to(self.device)
        actions = torch.LongTensor(batch['actions']).to(self.device)
        rewards = torch.FloatTensor(batch['rewards']).to(self.device)
        next_states = torch.FloatTensor(batch['next_states']).to(self.device)
        dones = torch.FloatTensor(batch['dones']).to(self.device)
        
        # Current Q-values
        q_values = self.q_network(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN: Select action with online network, evaluate with target network
        with torch.no_grad():
            # Select best action using online network
            next_q_online = self.q_network(next_states)
            next_actions = next_q_online.argmax(dim=1)
            
            # Evaluate action using target network
            next_q_target = self.target_network(next_states)
            next_q_value = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            # TD target
            target_q_value = rewards + (1 - dones) * self.gamma * next_q_value
        
        # Loss (Huber loss for robustness)
        loss = F.smooth_l1_loss(q_value, target_q_value)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        # Update counters
        self.steps += 1
        
        # Update target network
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            logger.info(f"Target network updated at step {self.steps}")
        
        # Decay epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon = max(
                self.epsilon_end,
                self.epsilon - (0.5 - self.epsilon_end) / self.epsilon_decay
            )
        
        metrics = {
            'loss': loss.item(),
            'q_mean': q_value.mean().item(),
            'q_std': q_value.std().item(),
            'reward_mean': rewards.mean().item(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }
        
        return metrics
    
    def save_checkpoint(self, path: str, metrics: Dict[str, float]) -> None:
        """Save model checkpoint"""
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'episodes': self.episodes,
            'metrics': metrics
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str) -> Dict[str, float]:
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.episodes = checkpoint['episodes']
        logger.info(f"Checkpoint loaded from {path}")
        return checkpoint['metrics']


class ReplayBuffer:
    """
    Experience Replay Buffer for DQN
    Stores transitions for training
    """
    
    def __init__(self, capacity: int = 5000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        logger.info(f"ReplayBuffer initialized - Capacity: {capacity}")
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        time_of_day: Optional[float] = None
    ) -> None:
        """Add experience to buffer"""
        
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'time_of_day': time_of_day
        }
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample random batch"""
        
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch_experiences = [self.buffer[i] for i in indices]
        
        batch = {
            'states': np.array([exp['state'] for exp in batch_experiences]),
            'actions': np.array([exp['action'] for exp in batch_experiences]),
            'rewards': np.array([exp['reward'] for exp in batch_experiences]),
            'next_states': np.array([exp['next_state'] for exp in batch_experiences]),
            'dones': np.array([exp['done'] for exp in batch_experiences], dtype=np.float32)
        }
        
        return batch
    
    def __len__(self) -> int:
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer
    Samples important experiences more frequently
    """
    
    def __init__(self, capacity: int = 5000, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta = beta  # Importance sampling exponent
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        logger.info(f"PrioritizedReplayBuffer initialized - Capacity: {capacity}, Alpha: {alpha}")
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        time_of_day: Optional[float] = None
    ) -> None:
        """Add experience with max priority"""
        
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'time_of_day': time_of_day
        }
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """Sample batch with priorities"""
        
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        # Calculate sampling probabilities
        priorities = self.priorities[:len(self.buffer)]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        
        # Get experiences
        batch_experiences = [self.buffer[i] for i in indices]
        
        # Calculate importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        batch = {
            'states': np.array([exp['state'] for exp in batch_experiences]),
            'actions': np.array([exp['action'] for exp in batch_experiences]),
            'rewards': np.array([exp['reward'] for exp in batch_experiences]),
            'next_states': np.array([exp['next_state'] for exp in batch_experiences]),
            'dones': np.array([exp['done'] for exp in batch_experiences], dtype=np.float32)
        }
        
        return batch, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Update priorities for sampled experiences"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self) -> int:
        return len(self.buffer)


def create_dqn_agent(config: Dict[str, any]) -> DoubleDQNAgent:
    """
    Factory function to create DQN agent from config
    """
    state_shape = (
        config.get('lookback', 30),
        config.get('num_features', 32)
    )
    
    agent = DoubleDQNAgent(
        state_shape=state_shape,
        action_space=config.get('action_space', 3),
        learning_rate=config.get('learning_rate', 0.0001),
        gamma=config.get('gamma', 0.3),
        epsilon_start=config.get('epsilon_start', 0.5),
        epsilon_end=config.get('epsilon_end', 0.01),
        epsilon_decay=config.get('epsilon_decay', 100),
        target_update_freq=config.get('target_update_freq', 500),
        device=config.get('device', 'cpu')
    )
    
    return agent


# ==================== PERFORMANCE OPTIMIZATION ====================

def benchmark_inference_speed(agent: DoubleDQNAgent, num_runs: int = 100) -> float:
    """
    Benchmark model inference speed
    Target: <50ms for intraday trading
    """
    import time
    
    # Create dummy state
    dummy_state = np.random.randn(30, 32).astype(np.float32)
    
    # Warmup
    for _ in range(10):
        agent.select_action(dummy_state, mode='eval')
    
    # Benchmark
    start_time = time.time()
    for _ in range(num_runs):
        agent.select_action(dummy_state, mode='eval')
    elapsed_time = (time.time() - start_time) / num_runs * 1000  # ms
    
    logger.info(f"Inference speed: {elapsed_time:.2f}ms per prediction")
    
    if elapsed_time > 50:
        logger.warning("⚠️ Inference too slow for intraday trading! Target: <50ms")
    else:
        logger.info(f"✅ Inference speed OK for intraday trading")
    
    return elapsed_time