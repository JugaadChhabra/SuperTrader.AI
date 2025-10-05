"""
State Representation Builder - PRODUCTION
Intraday-optimized feature engineering with 30-step lookback
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class IntradayStateBuilder:
    """
    Build state vectors optimized for intraday trading
    - 30-step lookback (30 mins at 1-min bars)
    - Fast indicators (MACD 5,13,5 | RSI 9)
    - Time-based features (critical for intraday)
    - Options features (PCR, IV, max pain)
    """
    
    def __init__(self, lookback: int = 30, bar_interval: str = '1min'):
        self.lookback = lookback
        self.bar_interval = bar_interval
        self.feature_names = []
        logger.info(f"StateBuilder initialized - Lookback: {lookback}, Interval: {bar_interval}")
    
    def build_state_vector(
        self,
        df_price: pd.DataFrame,
        time_features: Dict[str, float],
        oi_data: Optional[Dict[str, float]] = None,
        vix: Optional[float] = None,
        options_data: Optional[Dict[str, float]] = None,
        sentiment_data: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Build complete state vector for RL agent
        
        Returns:
            state_vector: shape (lookback, num_features) for LSTM input
        """
        logger.debug("Building state vector...")
        
        if df_price.empty or len(df_price) < self.lookback:
            logger.warning(f"Insufficient data: {len(df_price)} bars, need {self.lookback}")
            return self._create_empty_state()
        
        # Take last N bars
        df = df_price.tail(self.lookback).copy()
        
        # 1. Price features (time series)
        price_features = self._build_price_features(df)
        
        # 2. Technical indicators (time series + current values)
        tech_features = self._build_technical_features(df)
        
        # 3. Time-based features (scalar, repeated for each timestep)
        time_feats = self._build_time_features(time_features, df)
        
        # 4. OI features (scalar)
        oi_feats = self._build_oi_features(oi_data)
        
        # 5. VIX features (scalar)
        vix_feats = self._build_vix_features(vix)
        
        # 6. Options features (scalar)
        options_feats = self._build_options_features(options_data)
        
        # 7. Sentiment features (scalar)
        sentiment_feats = self._build_sentiment_features(sentiment_data)
        
        # Combine all features
        # Shape: (lookback, total_features)
        state_vector = self._combine_features(
            price_features,
            tech_features,
            time_feats,
            oi_feats,
            vix_feats,
            options_feats,
            sentiment_feats
        )
        
        logger.debug(f"State vector built - Shape: {state_vector.shape}")
        return state_vector
    
    def _build_price_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Build price-based features (normalized)"""
        
        # Normalized close price (by ATR)
        atr = self._calculate_atr(df, period=10)
        if atr > 0:
            normalized_close = (df['close'] - df['close'].mean()) / atr
        else:
            normalized_close = (df['close'] - df['close'].mean()) / df['close'].std()
        
        # Returns at different horizons (intraday-specific)
        returns_1bar = df['close'].pct_change()
        returns_5bar = df['close'].pct_change(5)
        returns_15bar = df['close'].pct_change(15)
        
        # Intraday volatility (EWMA with 30-period)
        volatility_ewma = returns_1bar.ewm(span=30, adjust=False).std()
        
        # Intraday range proximity
        session_high = df['high'].max()
        session_low = df['low'].min()
        if session_high > session_low:
            high_prox = (df['close'] - session_low) / (session_high - session_low)
            low_prox = (session_high - df['close']) / (session_high - session_low)
        else:
            high_prox = pd.Series([0.5] * len(df), index=df.index)
            low_prox = pd.Series([0.5] * len(df), index=df.index)
        
        return {
            'normalized_close': normalized_close.values,
            'returns_1bar': returns_1bar.fillna(0).values,
            'returns_5bar': returns_5bar.fillna(0).values,
            'returns_15bar': returns_15bar.fillna(0).values,
            'volatility_ewma': volatility_ewma.fillna(0.15).values,
            'intraday_high_prox': high_prox.values,
            'intraday_low_prox': low_prox.values
        }
    
    def _build_technical_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Build technical indicators (intraday-optimized)"""
        
        # MACD (5, 13, 5) - faster for intraday
        macd_line, signal_line, histogram = self._calculate_macd(
            df['close'], fast=5, slow=13, signal=5
        )
        
        # RSI (9) - faster for intraday
        rsi = self._calculate_rsi(df['close'], period=9)
        
        # ATR (10) - intraday volatility
        atr = self._calculate_atr(df, period=10)
        atr_series = pd.Series([atr] * len(df), index=df.index)
        
        # VWAP and distance
        vwap = self._calculate_vwap(df)
        vwap_dist = (df['close'] - vwap) / vwap if vwap.mean() > 0 else pd.Series([0] * len(df))
        
        # Volume surge (current vs time-of-day average)
        volume_surge = self._calculate_volume_surge(df)
        
        return {
            'macd': macd_line.fillna(0).values,
            'macd_signal': signal_line.fillna(0).values,
            'macd_histogram': histogram.fillna(0).values,
            'rsi': rsi.fillna(50).values,
            'atr': atr_series.values,
            'vwap_dist': vwap_dist.fillna(0).values,
            'volume_surge': volume_surge.fillna(1.0).values
        }
    
    def _build_time_features(
        self, 
        time_dict: Dict[str, float],
        df: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        """Build time-based features (critical for intraday)"""
        
        minutes_since_open = time_dict.get('minutes_since_open', 0)
        minutes_to_close = time_dict.get('minutes_to_close', 180)
        session_phase = time_dict.get('session_phase', 'morning')
        
        # Normalize to [0, 1]
        minutes_since_open_norm = min(1.0, minutes_since_open / 360)
        minutes_to_close_norm = max(0.0, minutes_to_close / 360)
        
        # Session one-hot encoding
        session_map = {
            'opening_range': [1, 0, 0, 0],
            'morning': [0, 1, 0, 0],
            'afternoon': [0, 0, 1, 0],
            'closing': [0, 0, 0, 1]
        }
        session_onehot = session_map.get(session_phase, [0, 1, 0, 0])
        
        # Repeat for each timestep in lookback
        n = len(df)
        
        return {
            'minutes_since_open_norm': np.full(n, minutes_since_open_norm),
            'minutes_to_close_norm': np.full(n, minutes_to_close_norm),
            'session_opening': np.full(n, session_onehot[0]),
            'session_morning': np.full(n, session_onehot[1]),
            'session_afternoon': np.full(n, session_onehot[2]),
            'session_closing': np.full(n, session_onehot[3])
        }
    
    def _build_oi_features(self, oi_data: Optional[Dict[str, float]]) -> Dict[str, float]:
        """Build Open Interest features"""
        
        if not oi_data:
            return {
                'oi_change_1bar': 0.0,
                'oi_change_5bar': 0.0,
                'oi_momentum': 0.0
            }
        
        # Normalize OI changes
        oi_change_1bar = oi_data.get('oi_change_1bar', 0) / 10000
        oi_change_5bar = oi_data.get('oi_change_5bar', 0) / 50000
        oi_momentum = oi_data.get('oi_momentum', 0) / 100
        
        return {
            'oi_change_1bar': oi_change_1bar,
            'oi_change_5bar': oi_change_5bar,
            'oi_momentum': oi_momentum
        }
    
    def _build_vix_features(self, vix: Optional[float]) -> Dict[str, float]:
        """Build VIX features"""
        
        if vix is None:
            vix = 15.0  # Default neutral VIX
        
        # Normalize VIX to ~[0, 1] range
        vix_norm = vix / 30.0
        
        # VIX regime (high if > 20)
        vix_high_regime = 1.0 if vix > 20 else 0.0
        
        return {
            'vix_norm': vix_norm,
            'vix_high_regime': vix_high_regime
        }
    
    def _build_options_features(self, options_data: Optional[Dict[str, float]]) -> Dict[str, float]:
        """Build options-specific features"""
        
        if not options_data:
            return {
                'pcr': 1.0,
                'iv_rank': 0.5,
                'max_pain_dist': 0.0,
                'gamma_exposure': 0.0
            }
        
        # Put-Call Ratio (normalized around 1.0)
        pcr = options_data.get('pcr', 1.0)
        pcr_norm = (pcr - 1.0) / 0.5  # Scale: 0.5 = -1, 1.5 = +1
        
        # IV Rank (already in [0, 100], normalize to [0, 1])
        iv_rank = options_data.get('iv_rank', 50) / 100
        
        # Max Pain distance (%)
        max_pain_dist = options_data.get('max_pain_dist_pct', 0.0) / 5.0  # ±5% range
        
        # Gamma exposure (normalized)
        gamma_exposure = options_data.get('gamma_exposure', 0.0) / 1e9
        
        return {
            'pcr': pcr_norm,
            'iv_rank': iv_rank,
            'max_pain_dist': max_pain_dist,
            'gamma_exposure': gamma_exposure
        }
    
    def _build_sentiment_features(self, sentiment_data: Optional[Dict[str, float]]) -> Dict[str, float]:
        """Build sentiment features (ultra-short-term)"""
        
        if not sentiment_data:
            return {
                'sentiment_5min': 0.0,
                'sentiment_momentum_15min': 0.0,
                'breaking_news': 0.0
            }
        
        sentiment_5min = sentiment_data.get('market_sentiment_5min', 0.0)
        sentiment_momentum = sentiment_data.get('sentiment_momentum_15min', 0.0)
        breaking_news = 1.0 if sentiment_data.get('breaking_news_flag', False) else 0.0
        
        return {
            'sentiment_5min': sentiment_5min,
            'sentiment_momentum_15min': sentiment_momentum,
            'breaking_news': breaking_news
        }
    
    def _combine_features(
        self,
        price_feats: Dict[str, np.ndarray],
        tech_feats: Dict[str, np.ndarray],
        time_feats: Dict[str, np.ndarray],
        oi_feats: Dict[str, float],
        vix_feats: Dict[str, float],
        options_feats: Dict[str, float],
        sentiment_feats: Dict[str, float]
    ) -> np.ndarray:
        """Combine all features into final state vector"""
        
        n = len(price_feats['normalized_close'])
        
        # Time series features (shape: lookback x features)
        ts_features = [
            # Price features
            price_feats['normalized_close'],
            price_feats['returns_1bar'],
            price_feats['returns_5bar'],
            price_feats['returns_15bar'],
            price_feats['volatility_ewma'],
            price_feats['intraday_high_prox'],
            price_feats['intraday_low_prox'],
            
            # Technical features
            tech_feats['macd'],
            tech_feats['macd_signal'],
            tech_feats['macd_histogram'],
            tech_feats['rsi'] / 100,  # Normalize to [0, 1]
            tech_feats['atr'],
            tech_feats['vwap_dist'],
            tech_feats['volume_surge'],
            
            # Time features (repeated for each timestep)
            time_feats['minutes_since_open_norm'],
            time_feats['minutes_to_close_norm'],
            time_feats['session_opening'],
            time_feats['session_morning'],
            time_feats['session_afternoon'],
            time_feats['session_closing']
        ]
        
        # Scalar features (repeated for each timestep)
        scalar_features = [
            oi_feats['oi_change_1bar'],
            oi_feats['oi_change_5bar'],
            oi_feats['oi_momentum'],
            vix_feats['vix_norm'],
            vix_feats['vix_high_regime'],
            options_feats['pcr'],
            options_feats['iv_rank'],
            options_feats['max_pain_dist'],
            options_feats['gamma_exposure'],
            sentiment_feats['sentiment_5min'],
            sentiment_feats['sentiment_momentum_15min'],
            sentiment_feats['breaking_news']
        ]
        
        # Repeat scalar features for each timestep
        for val in scalar_features:
            ts_features.append(np.full(n, val))
        
        # Stack into matrix: (lookback, num_features)
        state_matrix = np.column_stack(ts_features).astype(np.float32)
        
        # Store feature names for logging
        if not self.feature_names:
            self.feature_names = [
                'norm_close', 'ret_1', 'ret_5', 'ret_15', 'vol_ewma',
                'high_prox', 'low_prox', 'macd', 'macd_sig', 'macd_hist',
                'rsi', 'atr', 'vwap_dist', 'vol_surge',
                'min_since_open', 'min_to_close', 'sess_open', 'sess_morn',
                'sess_aft', 'sess_close', 'oi_1bar', 'oi_5bar', 'oi_mom',
                'vix', 'vix_high', 'pcr', 'iv_rank', 'max_pain', 'gamma',
                'sent_5m', 'sent_mom', 'breaking'
            ]
        
        return state_matrix
    
    def _create_empty_state(self) -> np.ndarray:
        """Create empty state when data insufficient"""
        return np.zeros((self.lookback, 32), dtype=np.float32)
    
    # ==================== INDICATOR CALCULATIONS ====================
    
    def _calculate_macd(
        self,
        prices: pd.Series,
        fast: int = 5,
        slow: int = 13,
        signal: int = 5
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD (intraday-optimized periods)"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 9) -> pd.Series:
        """Calculate RSI (faster period for intraday)"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 10) -> float:
        """Calculate ATR (Average True Range)"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]
        
        return atr if not np.isnan(atr) else df['close'].std()
    
    def _calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Calculate VWAP (Volume-Weighted Average Price)"""
        if 'volume' not in df.columns or df['volume'].sum() == 0:
            return df['close']
        
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        return vwap
    
    def _calculate_volume_surge(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volume surge vs rolling average"""
        if 'volume' not in df.columns:
            return pd.Series([1.0] * len(df), index=df.index)
        
        vol_ma = df['volume'].rolling(window=20).mean()
        volume_surge = df['volume'] / vol_ma
        return volume_surge