"""
RL Strategy Agent - PRODUCTION
Day 1: Rule-based logic (RSI + MACD momentum)
Future: DQN/PPO model inference
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def init_rl_agent(
    config: Dict[str, Any],
    action_space: int,
    obs_space: int,
    intraday_mode: bool = True
) -> Dict[str, Any]:
    """Initialize RL Agent"""
    logger.info(f"Initializing RL Agent - Intraday: {intraday_mode}")
    
    agent = {
        'model': None,  # TODO: Load trained model
        'action_space': action_space,
        'obs_space': obs_space,
        'intraday_mode': intraday_mode,
        'config': config,
        'epsilon': config.get('epsilon_start', 0.1),
        'step_count': 0,
        'mode': 'rule_based'  # Day 1: rule-based, later: 'dqn' | 'ppo' | 'a2c'
    }
    
    logger.info(f"✅ RL Agent initialized - Mode: {agent['mode']}")
    return agent


def build_state_representation(
    price_feats: Dict[str, Any],
    tech_feats: Dict[str, Any],
    senti_feats: Dict[str, float],
    basis: Dict[str, float],
    oi: Dict[str, float],
    vix: Optional[float],
    time_feats: Dict[str, Any],
    options_feats: Dict[str, Any]
) -> np.ndarray:
    """
    Build state vector from all features
    Returns numpy array ready for model inference
    """
    logger.debug("Building state representation...")
    
    # Extract key features
    close = tech_feats.get('close', 0.0)
    rsi = tech_feats.get('rsi', 50.0)
    macd = tech_feats.get('macd', 0.0)
    volatility = tech_feats.get('volatility_20', 0.15)
    volume_ratio = tech_feats.get('volume_ratio', 1.0)
    intraday_range = tech_feats.get('intraday_range_pct', 0.5)
    vwap_dist = tech_feats.get('vwap_dist', 0.0)
    
    # OI features
    oi_momentum = oi.get('oi_momentum', 0.0)
    oi_change = oi.get('oi_change_5bar', 0.0)
    
    # Basis
    basis_pct = basis.get('basis_pct', 0.0)
    
    # Time features (CRITICAL)
    minutes_to_close = time_feats.get('minutes_to_close', 180)
    session_phase = time_feats.get('session_phase', 'morning')
    
    # Session encoding
    session_map = {
        'opening_range': [1, 0, 0, 0],
        'morning': [0, 1, 0, 0],
        'afternoon': [0, 0, 1, 0],
        'closing': [0, 0, 0, 1]
    }
    session_onehot = session_map.get(session_phase, [0, 1, 0, 0])
    
    # Normalize features
    minutes_to_close_norm = minutes_to_close / 360.0
    minutes_since_open_norm = 1.0 - minutes_to_close_norm
    rsi_norm = rsi / 100.0
    vix_norm = (vix / 30.0) if vix else 0.5
    
    # Build state vector
    state_features = [
        # Price/Technical
        rsi_norm,
        macd / 100.0,  # Normalize
        volatility,
        volume_ratio,
        intraday_range,
        vwap_dist,
        
        # OI
        oi_momentum / 100.0,  # Normalize
        oi_change / 10000.0,
        
        # Basis
        basis_pct / 100.0,
        
        # VIX
        vix_norm,
        
        # Time (CRITICAL for intraday)
        minutes_to_close_norm,
        minutes_since_open_norm,
        *session_onehot,
        
        # Sentiment (zeros for Day 1)
        senti_feats.get('market_sentiment_5min', 0.0),
        senti_feats.get('sentiment_momentum_15min', 0.0),
    ]
    
    state_vector = np.array(state_features, dtype=np.float32)
    
    logger.debug(f"State vector shape: {state_vector.shape}, RSI: {rsi:.1f}, MACD: {macd:.2f}")
    return state_vector


def sample_action(
    state: np.ndarray,
    mode: str = 'eval',
    time_remaining: float = 180.0
) -> Dict[str, Any]:
    """
    Sample action from RL model
    Day 1: Rule-based logic (RSI + MACD)
    Future: DQN/PPO inference
    """
    logger.debug(f"Sampling action - Mode: {mode}, Time: {time_remaining:.1f} mins")
    
    # Time-based aggression
    if time_remaining < 15:  # After 3:00 PM
        aggression = 0.0
        logger.warning("⚠️ No new positions - too close to close")
    elif time_remaining < 45:  # After 2:30 PM
        aggression = 0.5
        logger.info("Reduced aggression - approaching close")
    else:
        aggression = 1.0
    
    # Extract features from state vector
    rsi_norm = state[0]
    macd_norm = state[1]
    
    rsi = rsi_norm * 100
    macd = macd_norm * 100
    
    # RULE-BASED LOGIC (Day 1 MVP)
    # Buy: RSI < 40 AND MACD > 0 (oversold + positive momentum)
    # Sell: RSI > 60 AND MACD < 0 (overbought + negative momentum)
    # Hold: Otherwise
    
    if rsi < 40 and macd > 0:
        action = 1  # Long
        confidence = min(1.0, (40 - rsi) / 40 + abs(macd) / 50)
        logger.info(f"🟢 LONG signal | RSI: {rsi:.1f} | MACD: {macd:.2f}")
    elif rsi > 60 and macd < 0:
        action = -1  # Short
        confidence = min(1.0, (rsi - 60) / 40 + abs(macd) / 50)
        logger.info(f"🔴 SHORT signal | RSI: {rsi:.1f} | MACD: {macd:.2f}")
    else:
        action = 0  # Hold
        confidence = 0.3
        logger.info(f"⚪ HOLD | RSI: {rsi:.1f} | MACD: {macd:.2f}")
    
    # Apply aggression multiplier
    final_action = action * aggression
    
    # Mock Q-values for logging
    q_values = [0.3, 0.5, 0.2] if action == 0 else [0.1, 0.2, 0.7]
    
    action_dict = {
        'action': final_action,
        'raw_action': action,
        'aggression_multiplier': aggression,
        'q_values': q_values,
        'confidence': confidence,
        'mode': mode,
        'logic': 'rule_based',
        'rsi': rsi,
        'macd': macd
    }
    
    return action_dict


def compute_position_size(
    state: np.ndarray,
    raw_action: float,
    risk_params: Dict[str, Any],
    margin_available: float,
    time_remaining: float,
    mis_mode: bool = True
) -> int:
    """
    Compute position size with risk management
    Returns: Number of lots (integer)
    """
    logger.debug(f"Computing position size - Action: {raw_action}, MIS: {mis_mode}")
    
    # If action is hold (0), return 0 lots
    if abs(raw_action) < 0.1:
        return 0
    
    # Extract risk parameters
    max_lots = risk_params.get('max_lots', 5)
    target_vol = risk_params.get('target_vol', 0.12)
    
    # Extract volatility from state
    volatility = state[2] if len(state) > 2 else 0.15
    
    # Volatility scaling
    vol_scalar = target_vol / volatility if volatility > 0 else 1.0
    vol_scalar = np.clip(vol_scalar, 0.5, 2.0)
    
    # Base lots from action
    base_lots = int(abs(raw_action) * max_lots * vol_scalar)
    
    # Time-based scaling (reduce after 2:30 PM)
    if time_remaining < 45:
        time_scalar = 0.5
        logger.info("Reducing position size - near close")
    else:
        time_scalar = 1.0
    
    position_size = int(base_lots * time_scalar)
    
    # Apply action direction
    position_size = position_size if raw_action > 0 else -position_size
    
    # MIS margin check
    if mis_mode:
        margin_per_lot = 60000  # NIFTY MIS margin ~60k per lot
    else:
        margin_per_lot = 100000  # NRML margin
    
    required_margin = abs(position_size) * margin_per_lot
    
    if required_margin > margin_available:
        position_size_adjusted = int(margin_available / margin_per_lot)
        if raw_action < 0:
            position_size_adjusted = -position_size_adjusted
        logger.warning(f"Position reduced due to margin: {position_size} → {position_size_adjusted}")
        position_size = position_size_adjusted
    
    logger.info(f"Position size: {position_size} lots | Margin required: ₹{required_margin:,.0f}")
    return position_size


def set_risk_params(
    params: Dict[str, Any],
    intraday_constraints: bool = True
) -> Dict[str, Any]:
    """Set risk parameters with intraday constraints"""
    default_params = {
        'max_lots': 3,  # Conservative for Day 1
        'target_vol': 0.12,
        'max_position_pct': 0.5,
        'stop_loss_pct': 0.015,
        'take_profit_pct': 0.03,
        'max_leverage': 5.0,
        'vix_threshold': 20.0,
    }
    
    validated_params = {**default_params, **params}
    
    if intraday_constraints:
        validated_params['max_leverage'] = min(validated_params['max_leverage'], 5.0)
        validated_params['stop_loss_pct'] = min(validated_params['stop_loss_pct'], 0.02)
        logger.info("Applied intraday constraints")
    
    logger.info(f"Risk params: Max lots={validated_params['max_lots']}, Target vol={validated_params['target_vol']}")
    return validated_params


# ==================== TRAINING FUNCTIONS (Future) ====================

def add_experience(
    s: np.ndarray,
    a: int,
    r: float,
    s_next: np.ndarray,
    done: bool,
    time_of_day: float
) -> None:
    """Add experience to replay buffer - TODO"""
    pass


def train_step(batch: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Training step - TODO"""
    return {'loss': 0.0, 'q_mean': 0.0}


def evaluate_policy(env: Any, episodes: int) -> Dict[str, float]:
    """Evaluate policy - TODO"""
    return {
        'mean_return': 0.0,
        'sharpe_ratio': 0.0,
        'win_rate': 0.0
    }


def save_checkpoint(path: str, step: int, metrics: Dict[str, float]) -> None:
    """Save checkpoint - TODO"""
    logger.info(f"Checkpoint would save to: {path}")
    pass


def load_checkpoint(path: str) -> Dict[str, Any]:
    """Load checkpoint - TODO"""
    logger.info(f"Checkpoint would load from: {path}")
    return {}