"""Unified model loader for SuperTrader.AI

Supports loading either:
1. TensorFlow/Keras intraday DQN checkpoints (*.weights.h5)
2. PyTorch TradingAgent checkpoints (*.pth) if present

Falls back gracefully if a format isn't available.
"""
from pathlib import Path
from typing import Optional, Tuple, Union, Any

from configs.config import get_config

# Optional imports guarded
try:
    import tensorflow as tf
except Exception:
    tf = None

try:
    import torch
    from models.dqn_network import TradingAgent
except Exception:
    torch = None
    TradingAgent = None


def _build_keras_dueling_lstm(state_size: int = 20, lstm1: int = 128, lstm2: int = 64, action_size: int = 3):
    """Create dueling LSTM architecture matching training notebook assumptions."""
    if tf is None:
        return None
    inputs = tf.keras.Input(shape=(None, state_size), name="state_sequence")
    x = tf.keras.layers.Masking()(inputs)
    x = tf.keras.layers.LSTM(lstm1, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(lstm2)(x)
    # Dueling streams
    value = tf.keras.layers.Dense(64, activation='relu')(x)
    value = tf.keras.layers.Dense(1, name='state_value')(value)
    adv = tf.keras.layers.Dense(64, activation='relu')(x)
    adv = tf.keras.layers.Dense(action_size, name='action_advantage')(adv)
    # Combine
    adv_mean = tf.keras.layers.Lambda(lambda a: a - tf.reduce_mean(a, axis=1, keepdims=True))(adv)
    q_values = tf.keras.layers.Add(name='q_values')([value, adv_mean])
    model = tf.keras.Model(inputs=inputs, outputs=q_values, name='dueling_lstm_dqn')
    return model


def load_model() -> Tuple[Optional[Any], str]:
    """Load the best available model using configuration.

    Returns: (model_or_agent, description)
    """
    cfg = get_config()
    paths = cfg.get_model_paths()
    checkpoint = paths.get('latest_checkpoint')

    # If not found via pattern, try a common fallback filename
    if not checkpoint:
        fallback = Path(cfg.rl.dqn_model_dir) / 'final_model.weights.h5'
        if fallback.exists():
            checkpoint = str(fallback)

    # Priority 1: Keras checkpoint (.weights.h5)
    if checkpoint and checkpoint.endswith('.weights.h5') and tf is not None:
        try:
            # Build with RLConfig.state_size (assumed original training feature count)
            model = _build_keras_dueling_lstm(state_size=cfg.rl.state_size, action_size=cfg.rl.action_size)
            # Allow partial loading in case of minor naming differences
            model.load_weights(checkpoint, skip_mismatch=True)
            return model, f"Loaded Keras dueling LSTM weights: {checkpoint}"
        except Exception as e:
            return None, f"Failed to load Keras weights ({checkpoint}): {e}"

    # Priority 2: PyTorch TradingAgent checkpoint (.pth) in models directory
    if TradingAgent is not None and torch is not None:
        pth_candidates = list(Path('models').glob('*.pth'))
        if pth_candidates:
            # Choose the most recently modified
            pth_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            ckpt = pth_candidates[0]
            try:
                agent = TradingAgent(num_features=32, lookback_period=30)
                agent.load(str(ckpt))
                return agent, f"Loaded PyTorch TradingAgent: {ckpt}"
            except Exception as e:
                return None, f"Failed to load PyTorch TradingAgent ({ckpt}): {e}"

    return None, "No suitable model checkpoint found"
