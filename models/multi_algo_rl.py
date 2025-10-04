"""
Multi-Algorithm RL Support - PRODUCTION
Support for DQN, PPO, and A2C
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


# ==================== A2C (Advantage Actor-Critic) ====================

class A2CNetwork(nn.Module):
    """
    A2C Network with shared LSTM backbone
    Outputs both policy (actor) and value (critic)
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size_1: int = 32,
        hidden_size_2: int = 16,
        action_space: int = 3,
        dropout: float = 0.2
    ):
        super(A2CNetwork, self).__init__()
        
        # Shared LSTM backbone
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size_1,
            batch_first=True,
            dropout=dropout
        )
        
        self.lstm2 = nn.LSTM(
            input_size=hidden_size_1,
            hidden_size=hidden_size_2,
            batch_first=True
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_size_2, 16),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(16, action_space),
            nn.Softmax(dim=-1)
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_size_2, 16),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(16, 1)
        )
        
        logger.info(f"A2CNetwork initialized - Actions: {action_space}")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input (batch, lookback, features)
        
        Returns:
            policy: Action probabilities (batch, actions)
            value: State value (batch, 1)
        """
        # LSTM encoding
        lstm1_out, _ = self.lstm1(x)
        lstm2_out, (h_n, _) = self.lstm2(lstm1_out)
        
        # Take final hidden state
        hidden = h_n[-1]
        
        # Actor and critic outputs
        policy = self.actor(hidden)
        value = self.critic(hidden)
        
        return policy, value


class A2CAgent:
    """
    A2C Agent for continuous learning with real-time updates
    Faster training than DQN, good for intraday
    """
    
    def __init__(
        self,
        state_shape: Tuple[int, int],
        action_space: int = 3,
        learning_rate_actor: float = 0.0001,
        learning_rate_critic: float = 0.001,
        gamma: float = 0.3,
        device: str = 'cpu'
    ):
        self.lookback, self.input_size = state_shape
        self.action_space = action_space
        self.gamma = gamma
        self.device = torch.device(device)
        
        # Network
        self.network = A2CNetwork(
            input_size=self.input_size,
            action_space=action_space
        ).to(self.device)
        
        # Separate optimizers for actor and critic
        self.optimizer_actor = torch.optim.Adam(
            self.network.actor.parameters(),
            lr=learning_rate_actor
        )
        self.optimizer_critic = torch.optim.Adam(
            self.network.critic.parameters(),
            lr=learning_rate_critic
        )
        
        self.steps = 0
        self.episodes = 0
        
        logger.info(f"A2CAgent initialized - Gamma: {gamma}")
    
    def select_action(
        self,
        state: np.ndarray,
        mode: str = 'train',
        time_remaining: float = 180.0
    ) -> Tuple[int, Dict]:
        """Select action using policy network"""
        
        # Time constraints
        if time_remaining < 15:
            return 1, {'policy': [0, 1, 0], 'value': 0, 'mode': 'time_constrained'}
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            policy, value = self.network(state_tensor)
            
            policy = policy.cpu().numpy()[0]
            value_scalar = value.cpu().item()
        
        if mode == 'train':
            # Sample from policy
            action = np.random.choice(self.action_space, p=policy)
        else:
            # Greedy
            action = int(np.argmax(policy))
        
        return action, {
            'policy': policy.tolist(),
            'value': value_scalar,
            'mode': mode
        }
    
    def train_step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> Dict[str, float]:
        """
        Single-step update (advantage actor-critic)
        Real-time learning - no replay buffer needed
        """
        # Convert to tensors
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        reward_tensor = torch.FloatTensor([reward]).to(self.device)
        done_tensor = torch.FloatTensor([done]).to(self.device)
        
        # Forward pass
        policy, value = self.network(state_tensor)
        _, next_value = self.network(next_state_tensor)
        
        # Calculate advantage
        td_target = reward_tensor + (1 - done_tensor) * self.gamma * next_value
        advantage = td_target - value
        
        # Actor loss (policy gradient with advantage)
        log_prob = torch.log(policy[0, action] + 1e-10)
        actor_loss = -log_prob * advantage.detach()
        
        # Critic loss (TD error)
        critic_loss = F.smooth_l1_loss(value, td_target.detach())
        
        # Update actor
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.actor.parameters(), max_norm=10.0)
        self.optimizer_actor.step()
        
        # Update critic
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.critic.parameters(), max_norm=10.0)
        self.optimizer_critic.step()
        
        self.steps += 1
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'advantage': advantage.item(),
            'value': value.item()
        }


# ==================== PPO (Proximal Policy Optimization) ====================

class PPOAgent:
    """
    PPO Agent - More stable than A2C
    Good for continuous action spaces (options trading)
    """
    
    def __init__(
        self,
        state_shape: Tuple[int, int],
        action_space: int = 3,
        learning_rate: float = 0.0001,
        gamma: float = 0.3,
        epsilon_clip: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        device: str = 'cpu'
    ):
        self.lookback, self.input_size = state_shape
        self.action_space = action_space
        self.gamma = gamma
        self.epsilon_clip = epsilon_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.device = torch.device(device)
        
        # Network
        self.network = A2CNetwork(
            input_size=self.input_size,
            action_space=action_space
        ).to(self.device)
        
        # Single optimizer for both actor and critic
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=learning_rate
        )
        
        # Storage for PPO updates
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        
        self.steps = 0
        self.episodes = 0
        
        logger.info(f"PPOAgent initialized - Epsilon clip: {epsilon_clip}")
    
    def select_action(
        self,
        state: np.ndarray,
        mode: str = 'train',
        time_remaining: float = 180.0
    ) -> Tuple[int, Dict]:
        """Select action and store log probability"""
        
        if time_remaining < 15:
            return 1, {'policy': [0, 1, 0], 'value': 0, 'mode': 'time_constrained'}
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            policy, value = self.network(state_tensor)
        
        # Sample action
        dist = Categorical(policy)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        action_int = action.item()
        
        # Store for PPO update (if training)
        if mode == 'train':
            self.states.append(state)
            self.actions.append(action_int)
            self.log_probs.append(log_prob.item())
        
        return action_int, {
            'policy': policy.cpu().numpy()[0].tolist(),
            'value': value.cpu().item(),
            'log_prob': log_prob.item()
        }
    
    def store_reward(self, reward: float, done: bool):
        """Store reward and done flag"""
        self.rewards.append(reward)
        self.dones.append(done)
    
    def train_step(self, num_epochs: int = 4) -> Dict[str, float]:
        """
        PPO update using collected experiences
        Called at end of episode
        """
        if len(self.states) == 0:
            return {}
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        
        # Calculate returns and advantages
        returns = self._calculate_returns()
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # PPO update for multiple epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for _ in range(num_epochs):
            # Forward pass
            policies, values = self.network(states)
            
            # Calculate advantages
            advantages = returns - values.squeeze()
            
            # Policy loss with clipping
            dist = Categorical(policies)
            new_log_probs = dist.log_prob(actions)
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages.detach()
            surr2 = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages.detach()
            
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values.squeeze(), returns)
            
            # Entropy bonus (encourage exploration)
            entropy = dist.entropy().mean()
            
            # Combined loss
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=10.0)
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
        
        # Clear buffers
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        
        self.steps += 1
        
        return {
            'policy_loss': total_policy_loss / num_epochs,
            'value_loss': total_value_loss / num_epochs,
            'entropy': total_entropy / num_epochs
        }
    
    def _calculate_returns(self) -> np.ndarray:
        """Calculate discounted returns"""
        returns = []
        R = 0
        
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                R = 0
            R = reward + self.gamma * R
            returns.insert(0, R)
        
        return np.array(returns)


# ==================== ALGORITHM FACTORY ====================

def create_rl_agent(algo_type: str, config: Dict, state_shape: Tuple[int, int]) -> object:
    """
    Factory function to create RL agent
    
    Args:
        algo_type: 'DQN', 'A2C', or 'PPO'
        config: Configuration dictionary
        state_shape: (lookback, num_features)
    
    Returns:
        RL agent instance
    """
    from models.dqn_network import DoubleDQNAgent
    
    if algo_type == 'DQN':
        agent = DoubleDQNAgent(
            state_shape=state_shape,
            action_space=config.get('action_space', 3),
            learning_rate=config.get('learning_rate', 0.0001),
            gamma=config.get('gamma', 0.3),
            device=config.get('device', 'cpu')
        )
    
    elif algo_type == 'A2C':
        agent = A2CAgent(
            state_shape=state_shape,
            action_space=config.get('action_space', 3),
            learning_rate_actor=config.get('learning_rate_actor', 0.0001),
            learning_rate_critic=config.get('learning_rate_critic', 0.001),
            gamma=config.get('gamma', 0.3),
            device=config.get('device', 'cpu')
        )
    
    elif algo_type == 'PPO':
        agent = PPOAgent(
            state_shape=state_shape,
            action_space=config.get('action_space', 3),
            learning_rate=config.get('learning_rate', 0.0001),
            gamma=config.get('gamma', 0.3),
            epsilon_clip=config.get('epsilon_clip', 0.2),
            device=config.get('device', 'cpu')
        )
    
    else:
        raise ValueError(f"Unknown algorithm: {algo_type}")
    
    logger.info(f"Created {algo_type} agent")
    return agent


# ==================== COMPARISON UTILITY ====================

def compare_algorithms(
    env,
    state_builder,
    algorithms: List[str] = ['DQN', 'A2C', 'PPO'],
    num_episodes: int = 50
) -> Dict[str, Dict]:
    """
    Compare multiple RL algorithms on same environment
    
    Returns:
        Dictionary with results for each algorithm
    """
    logger.info(f"Comparing algorithms: {algorithms}")
    
    results = {}
    
    for algo in algorithms:
        logger.info(f"\nTesting {algo}...")
        
        # Create agent
        config = {
            'action_space': 3,
            'learning_rate': 0.0001,
            'gamma': 0.3
        }
        
        agent = create_rl_agent(algo, config, state_shape=(30, 32))
        
        # Evaluate
        from models.policy_evaluator import IntradayPolicyEvaluator
        evaluator = IntradayPolicyEvaluator()
        
        metrics = evaluator.evaluate_policy(
            agent=agent,
            env=env,
            state_builder=state_builder,
            num_episodes=num_episodes
        )
        
        results[algo] = metrics
    
    # Log comparison
    logger.info("\n" + "="*80)
    logger.info("ALGORITHM COMPARISON")
    logger.info("="*80)
    
    for algo, metrics in results.items():
        logger.info(f"\n{algo}:")
        logger.info(f"  Sharpe: {metrics['intraday_sharpe_ratio']:.3f}")
        logger.info(f"  Avg P&L: ₹{metrics['avg_daily_pnl']:,.0f}")
        logger.info(f"  Win Rate: {metrics['win_rate_days']:.1%}")
    
    return results