# models/rl_agents.py
"""
TensorFlow-based RL Agent Implementations
DQN and A2C agents following Oxford paper architecture
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class DQNAgent(Model):
    """
    Deep Q-Network Agent with LSTM layers
    Based on Oxford paper architecture with volatility scaling
    """
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: list = [64, 32],
                 dropout_rate: float = 0.2,
                 **kwargs):
        super(DQNAgent, self).__init__(**kwargs)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # Calculate LSTM input shape (assuming 60 timesteps)
        self.timesteps = 60
        self.features_per_timestep = state_dim // self.timesteps
        
        # Build network layers
        self.reshape_layer = layers.Reshape((self.timesteps, self.features_per_timestep))
        
        # LSTM layers as per Oxford paper
        self.lstm1 = layers.LSTM(
            hidden_dims[0], 
            return_sequences=True,
            dropout=dropout_rate,
            recurrent_dropout=dropout_rate,
            name='lstm1'
        )
        
        self.lstm2 = layers.LSTM(
            hidden_dims[1],
            dropout=dropout_rate,
            recurrent_dropout=dropout_rate,
            name='lstm2'
        )
        
        # Dense layers with LeakyReLU
        self.dense1 = layers.Dense(32, activation='relu', name='dense1')
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dense2 = layers.Dense(16, activation='relu', name='dense2')
        
        # Output layer for Q-values
        self.q_output = layers.Dense(action_dim, activation='linear', name='q_values')
    
    def call(self, inputs, training=None):
        """Forward pass through the network"""
        # Reshape input for LSTM (batch_size, timesteps, features)
        x = self.reshape_layer(inputs)
        
        # LSTM feature extraction
        x = self.lstm1(x, training=training)
        x = self.lstm2(x, training=training)
        
        # Dense layers
        x = self.dense1(x)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        
        # Q-values output
        q_values = self.q_output(x)
        
        return q_values
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'hidden_dims': self.hidden_dims,
            'dropout_rate': self.dropout_rate
        })
        return config

class A2CAgent(Model):
    """
    Advantage Actor-Critic Agent
    Separate actor and critic networks with shared feature extraction
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: list = [64, 32],
                 dropout_rate: float = 0.2,
                 action_space_type: str = 'continuous',
                 **kwargs):
        super(A2CAgent, self).__init__(**kwargs)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.action_space_type = action_space_type
        
        # Calculate LSTM input shape
        self.timesteps = 60
        self.features_per_timestep = state_dim // self.timesteps
        
        # Shared feature extractor
        self.reshape_layer = layers.Reshape((self.timesteps, self.features_per_timestep))
        
        self.lstm1 = layers.LSTM(
            hidden_dims[0],
            return_sequences=True,
            dropout=dropout_rate,
            recurrent_dropout=dropout_rate,
            name='shared_lstm1'
        )
        
        self.lstm2 = layers.LSTM(
            hidden_dims[1],
            dropout=dropout_rate,
            recurrent_dropout=dropout_rate,
            name='shared_lstm2'
        )
        
        self.shared_dense = layers.Dense(32, activation='relu', name='shared_features')
        self.shared_dropout = layers.Dropout(dropout_rate)
        
        # Actor network (policy)
        if action_space_type == 'discrete':
            self.actor_output = layers.Dense(action_dim, activation='softmax', name='actor_policy')
        else:
            # Continuous actions - output mean and log_std
            self.actor_mean = layers.Dense(action_dim, activation='tanh', name='actor_mean')
            self.actor_log_std = layers.Dense(action_dim, activation='linear', name='actor_log_std')
        
        # Critic network (value function)
        self.critic_dense = layers.Dense(16, activation='relu', name='critic_dense')
        self.critic_output = layers.Dense(1, activation='linear', name='state_value')
    
    def call(self, inputs, training=None):
        """Forward pass returning both actor and critic outputs"""
        # Shared feature extraction
        x = self.reshape_layer(inputs)
        x = self.lstm1(x, training=training)
        x = self.lstm2(x, training=training)
        
        shared_features = self.shared_dense(x)
        shared_features = self.shared_dropout(shared_features, training=training)
        
        # Actor output
        if self.action_space_type == 'discrete':
            actor_output = self.actor_output(shared_features)
        else:
            actor_mean = self.actor_mean(shared_features)
            actor_log_std = self.actor_log_std(shared_features)
            actor_output = tf.concat([actor_mean, actor_log_std], axis=-1)
        
        # Critic output
        critic_features = self.critic_dense(shared_features)
        critic_output = self.critic_output(critic_features)
        
        return actor_output, critic_output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'hidden_dims': self.hidden_dims,
            'dropout_rate': self.dropout_rate,
            'action_space_type': self.action_space_type
        })
        return config

class DuelingDQNAgent(Model):
    """
    Dueling DQN Architecture
    Separates state value and advantage estimation
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: list = [64, 32],
                 dropout_rate: float = 0.2,
                 **kwargs):
        super(DuelingDQNAgent, self).__init__(**kwargs)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        
        # Shared feature extractor
        self.timesteps = 60
        self.features_per_timestep = state_dim // self.timesteps
        
        self.reshape_layer = layers.Reshape((self.timesteps, self.features_per_timestep))
        self.lstm1 = layers.LSTM(hidden_dims[0], return_sequences=True, dropout=dropout_rate)
        self.lstm2 = layers.LSTM(hidden_dims[1], dropout=dropout_rate)
        self.shared_dense = layers.Dense(32, activation='relu')
        
        # Value stream
        self.value_dense = layers.Dense(16, activation='relu', name='value_dense')
        self.value_output = layers.Dense(1, activation='linear', name='state_value')
        
        # Advantage stream
        self.advantage_dense = layers.Dense(16, activation='relu', name='advantage_dense')
        self.advantage_output = layers.Dense(action_dim, activation='linear', name='advantages')
    
    def call(self, inputs, training=None):
        """Forward pass with dueling architecture"""
        # Shared features
        x = self.reshape_layer(inputs)
        x = self.lstm1(x, training=training)
        x = self.lstm2(x, training=training)
        shared_features = self.shared_dense(x)
        
        # Value stream
        value_features = self.value_dense(shared_features)
        state_value = self.value_output(value_features)
        
        # Advantage stream
        advantage_features = self.advantage_dense(shared_features)
        advantages = self.advantage_output(advantage_features)
        
        # Combine value and advantages to get Q-values
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = state_value + advantages - tf.reduce_mean(advantages, axis=-1, keepdims=True)
        
        return q_values
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'hidden_dims': self.hidden_dims
        })
        return config

# Training utilities for TensorFlow RL agents

@tf.function
def dqn_loss(q_values, target_q_values, actions, rewards, dones, gamma=0.99):
    """
    Compute DQN loss with target network
    
    Args:
        q_values: Current Q-values from main network
        target_q_values: Q-values from target network
        actions: Actions taken
        rewards: Rewards received
        dones: Episode termination flags
        gamma: Discount factor
        
    Returns:
        Mean squared error loss
    """
    batch_size = tf.shape(q_values)[0]
    
    # Get Q-values for taken actions
    action_indices = tf.stack([tf.range(batch_size), tf.cast(actions, tf.int32)], axis=1)
    current_q = tf.gather_nd(q_values, action_indices)
    
    # Compute target Q-values
    next_q_max = tf.reduce_max(target_q_values, axis=1)
    target_q = rewards + gamma * next_q_max * (1.0 - tf.cast(dones, tf.float32))
    
    # Mean squared error loss
    loss = tf.reduce_mean(tf.square(current_q - tf.stop_gradient(target_q)))
    
    return loss

@tf.function
def a2c_loss(actor_output, critic_output, actions, rewards, next_values, dones, 
             gamma=0.99, entropy_coef=0.01):
    """
    Compute A2C loss for both actor and critic
    
    Args:
        actor_output: Policy output from actor network
        critic_output: Value estimates from critic network
        actions: Actions taken
        rewards: Rewards received
        next_values: Next state values
        dones: Episode termination flags
        gamma: Discount factor
        entropy_coef: Entropy regularization coefficient
        
    Returns:
        Tuple of (total_loss, actor_loss, critic_loss, entropy_loss)
    """
    # Compute advantages
    returns = rewards + gamma * next_values * (1.0 - tf.cast(dones, tf.float32))
    advantages = returns - tf.squeeze(critic_output)
    
    # Critic loss (value function)
    critic_loss = tf.reduce_mean(tf.square(advantages))
    
    # Actor loss (policy gradient)
    if len(actor_output.shape) > 1 and actor_output.shape[-1] > 1:
        # Discrete actions
        action_probs = actor_output
        log_probs = tf.math.log(tf.maximum(action_probs, 1e-8))
        action_log_probs = tf.reduce_sum(
            log_probs * tf.one_hot(actions, tf.shape(action_probs)[-1]), axis=-1
        )
        
        # Entropy for exploration
        entropy = -tf.reduce_sum(action_probs * log_probs, axis=-1)
        entropy_loss = -entropy_coef * tf.reduce_mean(entropy)
        
    else:
        # Continuous actions
        action_mean = actor_output[:, :1]
        action_log_std = actor_output[:, 1:]
        
        # Calculate log probability of taken action
        action_std = tf.exp(action_log_std)
        action_log_probs = (-0.5 * tf.square((actions - action_mean) / action_std) 
                           - action_log_std - 0.5 * np.log(2 * np.pi))
        action_log_probs = tf.reduce_sum(action_log_probs, axis=-1)
        
        # Entropy for continuous actions
        entropy = 0.5 * (1 + 2 * action_log_std + np.log(2 * np.pi))
        entropy_loss = -entropy_coef * tf.reduce_mean(entropy)
    
    # Actor loss
    actor_loss = -tf.reduce_mean(action_log_probs * tf.stop_gradient(advantages))
    
    # Total loss
    total_loss = actor_loss + 0.5 * critic_loss + entropy_loss
    
    return total_loss, actor_loss, critic_loss, entropy_loss

def build_dqn_model(state_dim: int, action_dim: int, hidden_dims: list = [64, 32]) -> keras.Model:
    """
    Build DQN model with functional API for easier customization
    
    Args:
        state_dim: Dimension of state space
        action_dim: Number of possible actions
        hidden_dims: List of hidden layer dimensions
        
    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=(state_dim,), name='state_input')
    
    # Reshape for LSTM
    x = layers.Reshape((60, state_dim // 60))(inputs)
    
    # LSTM layers
    x = layers.LSTM(hidden_dims[0], return_sequences=True, dropout=0.2, 
                   recurrent_dropout=0.2, name='lstm1')(x)
    x = layers.LSTM(hidden_dims[1], dropout=0.2, recurrent_dropout=0.2, 
                   name='lstm2')(x)
    
    # Dense layers
    x = layers.Dense(32, activation='relu', name='dense1')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(16, activation='relu', name='dense2')(x)
    
    # Output layer
    outputs = layers.Dense(action_dim, activation='linear', name='q_values')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='DQN')
    
    return model

def build_actor_critic_models(state_dim: int, action_dim: int, 
                             action_space_type: str = 'discrete',
                             hidden_dims: list = [64, 32]) -> Tuple[keras.Model, keras.Model]:
    """
    Build separate Actor and Critic models for A2C
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        action_space_type: 'discrete' or 'continuous'
        hidden_dims: List of hidden layer dimensions
        
    Returns:
        Tuple of (actor_model, critic_model)
    """
    # Shared input
    inputs = layers.Input(shape=(state_dim,), name='state_input')
    
    # Shared feature extraction
    x = layers.Reshape((60, state_dim // 60))(inputs)
    x = layers.LSTM(hidden_dims[0], return_sequences=True, dropout=0.2, 
                   recurrent_dropout=0.2)(x)
    x = layers.LSTM(hidden_dims[1], dropout=0.2, recurrent_dropout=0.2)(x)
    shared_features = layers.Dense(32, activation='relu')(x)
    shared_features = layers.Dropout(0.2)(shared_features)
    
    # Actor network
    if action_space_type == 'discrete':
        actor_output = layers.Dense(action_dim, activation='softmax', 
                                   name='policy')(shared_features)
    else:
        actor_mean = layers.Dense(action_dim, activation='tanh', 
                                 name='action_mean')(shared_features)
        actor_log_std = layers.Dense(action_dim, activation='linear', 
                                    name='action_log_std')(shared_features)
        actor_output = layers.Concatenate(name='actor_output')([actor_mean, actor_log_std])
    
    actor_model = keras.Model(inputs=inputs, outputs=actor_output, name='Actor')
    
    # Critic network
    critic_hidden = layers.Dense(16, activation='relu')(shared_features)
    critic_output = layers.Dense(1, activation='linear', name='value')(critic_hidden)
    
    critic_model = keras.Model(inputs=inputs, outputs=critic_output, name='Critic')
    
    return actor_model, critic_model

class ModelUtils:
    """Utility functions for model management"""
    
    @staticmethod
    def save_model_weights(model: keras.Model, filepath: str) -> None:
        """Save model weights"""
        model.save_weights(filepath)
        logger.info(f"Model weights saved to {filepath}")
    
    @staticmethod
    def load_model_weights(model: keras.Model, filepath: str) -> None:
        """Load model weights"""
        model.load_weights(filepath)
        logger.info(f"Model weights loaded from {filepath}")
    
    @staticmethod
    def get_model_summary(model: keras.Model) -> str:
        """Get formatted model summary"""
        import io
        stream = io.StringIO()
        model.summary(print_fn=lambda x: stream.write(x + '\n'))
        return stream.getvalue()
    
    @staticmethod
    def count_parameters(model: keras.Model) -> int:
        """Count total trainable parameters"""
        return model.count_params()

# Example usage and testing
if __name__ == "__main__":
    # Test DQN model
    print("Testing DQN model...")
    dqn_model = build_dqn_model(state_dim=240, action_dim=3)
    print(f"DQN parameters: {dqn_model.count_params()}")
    
    # Test input
    test_input = tf.random.normal((1, 240))
    dqn_output = dqn_model(test_input)
    print(f"DQN output shape: {dqn_output.shape}")
    
    # Test A2C models
    print("\nTesting A2C models...")
    actor, critic = build_actor_critic_models(state_dim=240, action_dim=3)
    print(f"Actor parameters: {actor.count_params()}")
    print(f"Critic parameters: {critic.count_params()}")
    
    actor_output = actor(test_input)
    critic_output = critic(test_input)
    print(f"Actor output shape: {actor_output.shape}")
    print(f"Critic output shape: {critic_output.shape}")
    
    print("\n✅ All models tested successfully!")