"""
RL Strategy Agent - PRODUCTION
Day 1: Rule-based logic (RSI + MACD momentum)
Future: DQN/PPO model inference
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def extract_phase4_signals(state: np.ndarray, phase4_features: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    Extract Phase 4 enhanced signals from state vector and additional features.
    
    Args:
        state: State vector with normalized features
        phase4_features: Dictionary of Phase 4 feature values
        
    Returns:
        Dictionary of Phase 4 signal indicators
    """
    signals = {
        'pcr_bullish': False,
        'pcr_bearish': False,
        'momentum_confluence': 0,
        'market_regime_trend': False,
        'high_risk_environment': False,
        'high_quality_setup': False,
        'volatility_regime': 'normal',
        'options_flow_bullish': False,
        'options_flow_bearish': False
    }
    
    try:
        if phase4_features:
            # PCR sentiment analysis
            pcr_percentile = phase4_features.get('pcr_percentile', 50)
            signals['pcr_bullish'] = pcr_percentile > 80  # High PCR = oversold = bullish
            signals['pcr_bearish'] = pcr_percentile < 20  # Low PCR = overbought = bearish
            
            # Momentum confluence
            signals['momentum_confluence'] = int(phase4_features.get('momentum_confluence_score', 0))
            
            # Market regime
            signals['market_regime_trend'] = bool(phase4_features.get('market_regime_strong_trend', False))
            
            # Risk assessment
            risk_score = phase4_features.get('market_risk_score', 0.5)
            signals['high_risk_environment'] = risk_score > 0.7
            
            # Setup quality
            quality_score = phase4_features.get('setup_quality_score', 0.5)
            signals['high_quality_setup'] = quality_score > 0.75
            
            # Volatility regime
            vol_regime_high = phase4_features.get('volatility_regime_high', False)
            vol_regime_low = phase4_features.get('volatility_regime_low', False)
            if vol_regime_high:
                signals['volatility_regime'] = 'high'
            elif vol_regime_low:
                signals['volatility_regime'] = 'low'
            else:
                signals['volatility_regime'] = 'normal'
            
            # Options flow bias
            flow_bias = phase4_features.get('options_flow_bias', 0)
            signals['options_flow_bullish'] = flow_bias > 0.3
            signals['options_flow_bearish'] = flow_bias < -0.3
        
        # Extract from extended state vector if Phase 4 features are embedded
        elif len(state) > 10:  # Extended state with Phase 4 features
            
            # PCR features (positions 10-12 in state)
            if len(state) > 12:
                pcr_norm = state[10]  # Normalized PCR percentile
                signals['pcr_bullish'] = pcr_norm > 0.8
                signals['pcr_bearish'] = pcr_norm < 0.2
            
            # Confluence features (positions 13-15)
            if len(state) > 15:
                confluence_norm = state[13]  # Normalized confluence score
                signals['momentum_confluence'] = int(confluence_norm * 3)  # Scale to 0-3
            
            # Market regime (position 16)
            if len(state) > 16:
                signals['market_regime_trend'] = state[16] > 0.5
            
            # Risk level (position 17)
            if len(state) > 17:
                signals['high_risk_environment'] = state[17] > 0.7
            
            # Setup quality (position 18)
            if len(state) > 18:
                signals['high_quality_setup'] = state[18] > 0.75
        
    except Exception as e:
        logger.warning(f"Phase 4 signal extraction failed: {e}")
    
    return signals


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
    time_remaining: float = 180.0,
    phase4_features: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Sample action from RL model - Phase 4 Enhanced
    Enhanced with PCR sentiment, confluence indicators, and market regime analysis
    Day 1: Rule-based logic (RSI + MACD + Phase 4 enhancements)
    Future: DQN/PPO inference with expanded feature space
    """
    logger.debug(f"Sampling action (Phase 4) - Mode: {mode}, Time: {time_remaining:.1f} mins")
    
    # Time-based aggression
    if time_remaining < 15:  # After 3:00 PM
        aggression = 0.0
        logger.warning("⚠️ No new positions - too close to close")
    elif time_remaining < 45:  # After 2:30 PM
        aggression = 0.5
        logger.info("Reduced aggression - approaching close")
    else:
        aggression = 1.0
    
    # Extract basic features from state vector
    rsi_norm = state[0] if len(state) > 0 else 0.5
    macd_norm = state[1] if len(state) > 1 else 0.0
    
    rsi = rsi_norm * 100
    macd = macd_norm * 100
    
    # Phase 4: Extract enhanced features if available
    phase4_signals = extract_phase4_signals(state, phase4_features)
    
    # ENHANCED RULE-BASED LOGIC (Phase 4)
    # Multi-factor decision making with Phase 4 features
    
    # Base RSI + MACD signals
    rsi_oversold = rsi < 40
    rsi_overbought = rsi > 60
    macd_bullish = macd > 0
    macd_bearish = macd < 0
    
    # Phase 4: Additional signal filters
    pcr_bullish = phase4_signals.get('pcr_bullish', False)
    pcr_bearish = phase4_signals.get('pcr_bearish', False)
    momentum_confluence = phase4_signals.get('momentum_confluence', 0)
    market_regime_trend = phase4_signals.get('market_regime_trend', False)
    high_risk_env = phase4_signals.get('high_risk_environment', False)
    high_quality_setup = phase4_signals.get('high_quality_setup', False)
    
    # Enhanced decision logic
    if rsi_oversold and macd_bullish:
        # Base bullish signal - enhance with Phase 4 filters
        base_confidence = (40 - rsi) / 40 + abs(macd) / 50
        
        # Phase 4 enhancements
        pcr_boost = 0.2 if pcr_bullish else -0.1 if pcr_bearish else 0
        confluence_boost = 0.15 * (momentum_confluence / 3.0) if momentum_confluence else 0
        regime_boost = 0.1 if market_regime_trend else -0.05
        quality_boost = 0.15 if high_quality_setup else 0
        risk_penalty = -0.2 if high_risk_env else 0
        
        confidence = min(1.0, base_confidence + pcr_boost + confluence_boost + regime_boost + quality_boost + risk_penalty)
        
        # Only take position if Phase 4 filters support it
        if confidence > 0.5 and not high_risk_env:  # Raised threshold from 0.4 to 0.5
            action = 1  # Long
            logger.info(f"🟢 ENHANCED LONG | RSI: {rsi:.1f} | MACD: {macd:.2f} | PCR: {pcr_bullish} | Conf: {confidence:.2f}")
        else:
            action = 0
            confidence = 0.3
            logger.info(f"⚪ FILTERED LONG | Risk too high or weak confluence")
            
    elif rsi_overbought and macd_bearish:
        # Base bearish signal - enhance with Phase 4 filters
        base_confidence = (rsi - 60) / 40 + abs(macd) / 50
        
        # Phase 4 enhancements
        pcr_boost = 0.2 if pcr_bearish else -0.1 if pcr_bullish else 0
        confluence_boost = 0.15 * (momentum_confluence / 3.0) if momentum_confluence else 0
        regime_boost = 0.1 if market_regime_trend else -0.05  # Trend can support short in downtrend
        quality_boost = 0.15 if high_quality_setup else 0
        risk_penalty = -0.2 if high_risk_env else 0
        
        confidence = min(1.0, base_confidence + pcr_boost + confluence_boost + regime_boost + quality_boost + risk_penalty)
        
        # Only take position if Phase 4 filters support it
        if confidence > 0.5 and not high_risk_env:  # Raised threshold from 0.4 to 0.5
            action = -1  # Short
            logger.info(f"🔴 ENHANCED SHORT | RSI: {rsi:.1f} | MACD: {macd:.2f} | PCR: {pcr_bearish} | Conf: {confidence:.2f}")
        else:
            action = 0
            confidence = 0.3
            logger.info(f"⚪ FILTERED SHORT | Risk too high or weak confluence")
    else:
        action = 0  # Hold
        confidence = 0.3
        logger.info(f"⚪ HOLD | RSI: {rsi:.1f} | MACD: {macd:.2f} | Regime: {market_regime_trend}")
    
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
        'logic': 'phase4_enhanced_rule_based',
        'rsi': rsi,
        'macd': macd,
        # Phase 4 enhancements
        'phase4_signals': phase4_signals,
        'pcr_sentiment': 'bullish' if pcr_bullish else 'bearish' if pcr_bearish else 'neutral',
        'momentum_confluence_score': momentum_confluence,
        'market_regime': 'trending' if market_regime_trend else 'consolidating',
        'risk_level': 'high' if high_risk_env else 'normal',
        'setup_quality': 'high' if high_quality_setup else 'standard'
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