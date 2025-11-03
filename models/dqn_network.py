"""
================================================================================================
SIMPLIFIED DQN NETWORK - THE TRADING BRAIN
================================================================================================

This is the AI "brain" that learns to make trading decisions.

WHAT IT DOES:
- Takes in market data (prices, indicators, time-of-day)
- Outputs 3 possible actions: BUY (long), SELL (short), or HOLD (stay flat)
- Learns from experience using "Deep Q-Learning"

ARCHITECTURE:
  Input (market state) 
    ↓
  LSTM Layer 1 (32 units) - learns short-term patterns
    ↓
  LSTM Layer 2 (16 units) - learns longer-term patterns
    ↓
  Dense layers - combines patterns
    ↓
  Output (Q-values for 3 actions: Short, Hold, Long)

WHY DUELING DQN?
- Separates "how good is this state" from "which action is best"
- More stable learning for trading
================================================================================================
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


# ==================== THE NEURAL NETWORK ====================

class TradingBrain(nn.Module):
    """
    The neural network that makes trading decisions
    
    Think of it as a trader's brain that:
    1. Looks at recent market data (last 30 minutes)
    2. Processes patterns using LSTM (remembers sequences)
    3. Decides: Should I go LONG, SHORT, or stay FLAT?
    """
    
    def __init__(self, input_features: int = 32):
        """
        Initialize the trading brain
        
        Args:
            input_features: Number of features per time step (default 32)
                           e.g., price, RSI, MACD, volume, time-of-day, etc.
        """
        super(TradingBrain, self).__init__()
        
        # === LAYER 1: First LSTM - Catches short-term patterns ===
        # Example: "Price has been rising for 5 minutes"
        self.lstm1 = nn.LSTM(
            input_size=input_features,
            hidden_size=32,
            batch_first=True,
            dropout=0.2
        )
        
        # === LAYER 2: Second LSTM - Catches longer patterns ===
        # Example: "We're in an uptrend that started 20 minutes ago"
        self.lstm2 = nn.LSTM(
            input_size=32,
            hidden_size=16,
            batch_first=True
        )
        
        # === VALUE STREAM: "How good is the current situation?" ===
        # Answers: "Is this a good time to have a position?"
        self.value_stream = nn.Sequential(
            nn.Linear(16, 16),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),
            nn.Linear(16, 1)  # Single value: state goodness
        )
        
        # === ADVANTAGE STREAM: "Which action is best?" ===
        # Answers: "Should I go long, short, or stay flat?"
        self.advantage_stream = nn.Sequential(
            nn.Linear(16, 16),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),
            nn.Linear(16, 3)  # 3 actions: Short (0), Hold (1), Long (2)
        )
        
        logger.info("TradingBrain initialized ✓")
    
    def forward(self, market_state: torch.Tensor) -> torch.Tensor:
        """
        Think and decide on an action
        
        Args:
            market_state: Recent market data
                         Shape: (batch_size, lookback_period, features)
                         Example: (32, 30, 32) = 32 samples, 30 time steps, 32 features
        
        Returns:
            q_values: "Quality scores" for each action
                     Shape: (batch_size, 3)
                     Higher score = better action
        """
        # Step 1: Process through LSTM layers
        lstm1_output, _ = self.lstm1(market_state)
        lstm2_output, (final_hidden, _) = self.lstm2(lstm1_output)
        
        # Step 2: Take the final hidden state (most recent understanding)
        brain_state = final_hidden[-1]  # Shape: (batch_size, 16)
        
        # Step 3: Calculate VALUE (how good is this situation?)
        state_value = self.value_stream(brain_state)  # Shape: (batch_size, 1)
        
        # Step 4: Calculate ADVANTAGES (which action is best?)
        action_advantages = self.advantage_stream(brain_state)  # Shape: (batch_size, 3)
        
        # Step 5: Combine into Q-values
        # Formula: Q(state, action) = V(state) + A(state, action) - mean(A)
        # This is the "Dueling" part - separates value from action selection
        q_values = state_value + (action_advantages - action_advantages.mean(dim=1, keepdim=True))
        
        return q_values


# ==================== THE TRADING AGENT ====================

class TradingAgent:
    """
    The complete trading agent that:
    1. Uses the TradingBrain to make decisions
    2. Learns from experience (replay buffer)
    3. Manages exploration vs exploitation
    """
    
    def __init__(
        self,
        num_features: int = 32,
        lookback_period: int = 30,
        learning_rate: float = 0.0001,
        gamma: float = 0.3,  # Low gamma for intraday (short horizon)
        epsilon_start: float = 0.5,  # Start with 50% random exploration
        epsilon_end: float = 0.01,  # End with 1% random exploration
        device: str = 'cpu'
    ):
        """
        Initialize the trading agent
        
        Args:
            num_features: Number of input features (price, indicators, etc.)
            lookback_period: How many time steps to look back (30 = 30 minutes)
            learning_rate: How fast to learn (0.0001 = conservative)
            gamma: Discount factor (0.3 = focus on immediate rewards, good for intraday)
            epsilon_start: Initial exploration rate (0.5 = 50% random at start)
            epsilon_end: Final exploration rate (0.01 = 1% random after training)
            device: 'cpu' or 'cuda' (GPU)
        """
        self.device = torch.device(device)
        
        # === Create TWO brains ===
        # Why two? One learns fast (main), one provides stable targets (target)
        
        # MAIN BRAIN: Updates every step, learns actively
        self.main_brain = TradingBrain(input_features=num_features).to(self.device)
        
        # TARGET BRAIN: Updates slowly, provides stable learning targets
        self.target_brain = TradingBrain(input_features=num_features).to(self.device)
        self.target_brain.load_state_dict(self.main_brain.state_dict())
        self.target_brain.eval()  # Always in evaluation mode
        
        # === Learning parameters ===
        self.optimizer = torch.optim.Adam(self.main_brain.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        
        # === Experience storage ===
        self.memory = []  # Stores past experiences for learning
        self.memory_capacity = 5000  # Keep last 5000 experiences
        
        # === Training counters ===
        self.steps = 0
        self.target_update_frequency = 500  # Update target brain every 500 steps
        
        logger.info(f"TradingAgent initialized | Gamma: {gamma} | Epsilon: {epsilon_start}→{epsilon_end}")
    
    def decide_action(
        self, 
        market_state: np.ndarray, 
        mode: str = 'train',
        minutes_to_close: float = 180.0
    ) -> Tuple[int, Dict]:
        """
        Decide what action to take
        
        Args:
            market_state: Recent market data (lookback, features)
            mode: 'train' (with exploration) or 'eval' (greedy only)
            minutes_to_close: Minutes until market close (3:15 PM)
        
        Returns:
            action: 0 (short), 1 (hold), or 2 (long)
            info: Additional information (Q-values, epsilon, etc.)
        """
        # === SAFETY CHECK: Don't trade close to market close ===
        if minutes_to_close < 15:
            # Force HOLD action if less than 15 minutes to close
            return 1, {'reason': 'time_constraint', 'q_values': [0, 1, 0]}
        
        # === EXPLORATION: Random action sometimes (helps learning) ===
        if mode == 'train' and np.random.random() < self.epsilon:
            action = np.random.randint(0, 3)  # Random: 0, 1, or 2
            return action, {'mode': 'explore', 'epsilon': self.epsilon}
        
        # === EXPLOITATION: Use brain to pick best action ===
        with torch.no_grad():
            # Convert to tensor and add batch dimension
            state_tensor = torch.FloatTensor(market_state).unsqueeze(0).to(self.device)
            
            # Get Q-values from brain
            q_values = self.main_brain(state_tensor).cpu().numpy()[0]
            
            # Pick action with highest Q-value
            action = int(np.argmax(q_values))
        
        return action, {'mode': 'exploit', 'q_values': q_values.tolist(), 'epsilon': self.epsilon}
    
    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        Store an experience in memory for later learning
        
        This is like the agent "journaling" its experiences:
        "I saw this market state, took this action, got this reward"
        """
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        }
        
        # Add to memory (with capacity limit)
        if len(self.memory) < self.memory_capacity:
            self.memory.append(experience)
        else:
            # Overwrite oldest experience
            self.memory[self.steps % self.memory_capacity] = experience
    
    def learn(self, batch_size: int = 32) -> Dict[str, float]:
        """
        Learn from past experiences (training step)
        
        This is where the magic happens:
        1. Sample random experiences from memory
        2. Calculate how "wrong" our predictions were
        3. Update the brain to be less wrong next time
        
        Args:
            batch_size: Number of experiences to learn from at once
        
        Returns:
            metrics: Training metrics (loss, Q-values, etc.)
        """
        if len(self.memory) < batch_size:
            return {}  # Not enough experiences yet
        
        # === STEP 1: Sample random experiences ===
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        experiences = [self.memory[i] for i in indices]
        
        # === STEP 2: Convert to tensors ===
        states = torch.FloatTensor(np.array([e['state'] for e in experiences])).to(self.device)
        actions = torch.LongTensor([e['action'] for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e['reward'] for e in experiences]).to(self.device)
        next_states = torch.FloatTensor(np.array([e['next_state'] for e in experiences])).to(self.device)
        dones = torch.FloatTensor([e['done'] for e in experiences]).to(self.device)
        
        # === STEP 3: Calculate current Q-values ===
        current_q_values = self.main_brain(states)
        current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze()
        
        # === STEP 4: Calculate target Q-values (what we SHOULD have predicted) ===
        with torch.no_grad():
            # Double DQN: Select action with main brain, evaluate with target brain
            next_q_main = self.main_brain(next_states)
            next_actions = next_q_main.argmax(dim=1)
            
            next_q_target = self.target_brain(next_states)
            next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze()
            
            # Target = reward + gamma * next_q (if not done)
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # === STEP 5: Calculate loss (how wrong we were) ===
        loss = nn.functional.smooth_l1_loss(current_q, target_q)
        
        # === STEP 6: Update the brain ===
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.main_brain.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        # === STEP 7: Update target brain periodically ===
        self.steps += 1
        if self.steps % self.target_update_frequency == 0:
            self.target_brain.load_state_dict(self.main_brain.state_dict())
            logger.info(f"Target brain updated at step {self.steps}")
        
        # === STEP 8: Decay exploration rate ===
        if self.epsilon > self.epsilon_end:
            self.epsilon = max(self.epsilon_end, self.epsilon * 0.995)
        
        return {
            'loss': loss.item(),
            'q_mean': current_q.mean().item(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }
    
    def save(self, filepath: str):
        """Save the agent's brain to disk"""
        checkpoint = {
            'main_brain': self.main_brain.state_dict(),
            'target_brain': self.target_brain.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Agent saved to {filepath}")
    
    def load(self, filepath: str):
        """Load the agent's brain from disk"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.main_brain.load_state_dict(checkpoint['main_brain'])
        self.target_brain.load_state_dict(checkpoint['target_brain'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        logger.info(f"Agent loaded from {filepath}")


# ==================== QUICK START EXAMPLE ====================

if __name__ == "__main__":
    """
    Quick example of how to use the TradingAgent
    """
    
    # Create agent
    agent = TradingAgent(
        num_features=32,
        lookback_period=30,
        learning_rate=0.0001,
        gamma=0.3
    )
    
    # Simulate market state (30 time steps, 32 features each)
    dummy_state = np.random.randn(30, 32).astype(np.float32)
    
    # Decide action
    action, info = agent.decide_action(dummy_state, mode='train', minutes_to_close=120)
    
    print(f"Decision: {'SHORT' if action == 0 else 'HOLD' if action == 1 else 'LONG'}")
    print(f"Q-values: {info.get('q_values', [])}")
    print(f"Mode: {info.get('mode', 'N/A')}")
    
    # Store experience
    dummy_reward = 0.5
    dummy_next_state = np.random.randn(30, 32).astype(np.float32)
    agent.remember(dummy_state, action, dummy_reward, dummy_next_state, done=False)
    
    # Learn (if enough experiences)
    metrics = agent.learn(batch_size=32)
    if metrics:
        print(f"Training metrics: {metrics}")