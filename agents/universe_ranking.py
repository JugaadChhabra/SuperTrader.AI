"""
Dynamic Universe Selection and Ranking System
Comprehensive index ranking for optimal futures trading universe selection
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from enum import Enum
import math
from utils.config import load_config

logger = logging.getLogger(__name__)

class VolatilityRegime(Enum):
    """Volatility regime classification"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"

class TrendStrength(Enum):
    """Trend strength classification"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"

@dataclass
class IndexMetrics:
    """Comprehensive metrics for index ranking"""
    symbol: str
    
    # Liquidity metrics
    avg_daily_volume: int           # Average daily volume (lots)
    avg_open_interest: int          # Average open interest (lots)
    bid_ask_spread_bps: float       # Bid-ask spread in basis points
    liquidity_score: float          # Overall liquidity score (0-1)
    
    # Volatility metrics
    realized_volatility: float      # 30-day realized volatility
    volatility_regime: VolatilityRegime  # Current volatility regime
    vol_percentile: float           # Volatility percentile (0-100)
    volatility_score: float         # Volatility attractiveness score (0-1)
    
    # Momentum and trend metrics
    momentum_1d: float              # 1-day momentum
    momentum_5d: float              # 5-day momentum
    momentum_20d: float             # 20-day momentum
    trend_strength: TrendStrength   # Overall trend strength
    momentum_score: float           # Momentum score (0-1)
    
    # Mean reversion metrics
    rsi_14: float                   # 14-period RSI
    mean_reversion_score: float     # Mean reversion opportunity score (0-1)
    
    # Market structure
    support_resistance_strength: float  # S&R level strength
    breakout_potential: float       # Breakout potential score
    
    # Risk metrics
    max_drawdown_30d: float         # Maximum 30-day drawdown
    var_95_1d: float                # 1-day 95% VaR
    risk_score: float               # Overall risk score (0-1, lower is better)
    
    # Final ranking
    composite_score: float          # Final composite ranking score
    rank: int                       # Final rank (1 = best)
    
    # Additional metadata
    last_updated: datetime
    data_quality: float             # Data quality score (0-1)

@dataclass
class RankingWeights:
    """Configurable weights for ranking factors"""
    liquidity_weight: float = 0.30
    volatility_weight: float = 0.25
    momentum_weight: float = 0.25
    mean_reversion_weight: float = 0.20
    
    def __post_init__(self):
        """Validate weights sum to 1.0"""
        total = self.liquidity_weight + self.volatility_weight + self.momentum_weight + self.mean_reversion_weight
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total}")

class UniverseRanker:
    """
    Dynamic index ranking system for futures trading universe selection
    """
    
    def __init__(self, config_path: str = None):
        """Initialize universe ranker with configuration"""
        self.config = load_config(config_path or "configs/risk.yaml")
        self.ranking_config = self.config.get('universe_ranking', {})
        
        # Ranking weights from config
        ranking_factors = self.ranking_config.get('ranking_factors', {})
        self.weights = RankingWeights(
            liquidity_weight=ranking_factors.get('liquidity_weight', 0.30),
            volatility_weight=ranking_factors.get('volatility_weight', 0.25),
            momentum_weight=ranking_factors.get('momentum_weight', 0.25),
            mean_reversion_weight=ranking_factors.get('mean_reversion_weight', 0.20)
        )
        
        # Index universe configuration
        self.index_universe = {
            'NIFTY': {
                'full_name': 'Nifty 50',
                'category': 'broad_market',
                'typical_volume': 500000,
                'typical_oi': 2000000,
                'lot_size': 75,
                'tick_size': 0.05,
                'typical_volatility': 0.15
            },
            'BANKNIFTY': {
                'full_name': 'Bank Nifty',
                'category': 'sectoral',
                'typical_volume': 300000,
                'typical_oi': 1500000,
                'lot_size': 15,
                'tick_size': 0.05,
                'typical_volatility': 0.25
            },
            'FINNIFTY': {
                'full_name': 'Fin Nifty',
                'category': 'sectoral',
                'typical_volume': 100000,
                'typical_oi': 800000,
                'lot_size': 25,
                'tick_size': 0.05,
                'typical_volatility': 0.20
            },
            'MIDCPNIFTY': {
                'full_name': 'Nifty Midcap 150',
                'category': 'mid_cap',
                'typical_volume': 50000,
                'typical_oi': 300000,
                'lot_size': 75,
                'tick_size': 0.05,
                'typical_volatility': 0.22
            },
            'CNXIT': {
                'full_name': 'Nifty IT',
                'category': 'sectoral',
                'typical_volume': 30000,
                'typical_oi': 200000,
                'lot_size': 50,
                'tick_size': 0.05,
                'typical_volatility': 0.28
            }
        }
        
        # Quality filters from config
        self.min_avg_daily_volume = self.ranking_config.get('min_avg_daily_volume', 50000)
        self.min_open_interest = self.ranking_config.get('min_open_interest', 100000)
        self.max_bid_ask_spread_pct = self.ranking_config.get('max_bid_ask_spread_pct', 0.02)
        self.min_liquidity_score = self.ranking_config.get('min_liquidity_score', 0.6)
        self.max_volatility_threshold = self.ranking_config.get('max_volatility_threshold', 0.40)
        
        logger.info("Universe ranker initialized")
    
    def rank_universe(
        self,
        market_data: Dict[str, Dict[str, Any]],
        max_selections: int = None
    ) -> List[IndexMetrics]:
        """
        Rank the entire index universe based on current market conditions
        
        Args:
            market_data: Dictionary with market data for each index
            max_selections: Maximum number of indices to select
            
        Returns:
            List of IndexMetrics sorted by rank (best first)
        """
        try:
            if max_selections is None:
                max_selections = self.ranking_config.get('max_indices_selected', 5)
            
            all_metrics = []
            
            # Calculate metrics for each index
            for symbol in self.index_universe.keys():
                if symbol not in market_data:
                    logger.warning(f"No market data available for {symbol}")
                    continue
                
                try:
                    metrics = self._calculate_index_metrics(symbol, market_data[symbol])
                    
                    # Apply quality filters
                    if self._passes_quality_filters(metrics):
                        all_metrics.append(metrics)
                    else:
                        logger.info(f"{symbol} filtered out due to quality criteria")
                        
                except Exception as e:
                    logger.error(f"Error calculating metrics for {symbol}: {e}")
                    continue
            
            # Sort by composite score (descending)
            all_metrics.sort(key=lambda x: x.composite_score, reverse=True)
            
            # Assign ranks
            for i, metrics in enumerate(all_metrics):
                metrics.rank = i + 1
            
            # Return top selections
            selected_indices = all_metrics[:max_selections]
            
            logger.info(f"Ranked {len(all_metrics)} indices, selected top {len(selected_indices)}")
            
            return selected_indices
            
        except Exception as e:
            logger.error(f"Error in universe ranking: {e}")
            return []
    
    def _calculate_index_metrics(self, symbol: str, market_data: Dict[str, Any]) -> IndexMetrics:
        """Calculate comprehensive metrics for a single index"""
        try:
            # Extract basic data
            price_data = market_data.get('price_data', pd.DataFrame())
            current_price = market_data.get('current_price', 0)
            volume_data = market_data.get('volume_data', [])
            oi_data = market_data.get('open_interest_data', [])
            bid_ask_data = market_data.get('bid_ask_data', {})
            
            # Calculate individual metric components
            liquidity_metrics = self._calculate_liquidity_metrics(
                symbol, volume_data, oi_data, bid_ask_data
            )
            
            volatility_metrics = self._calculate_volatility_metrics(
                symbol, price_data, current_price
            )
            
            momentum_metrics = self._calculate_momentum_metrics(
                symbol, price_data, current_price
            )
            
            mean_reversion_metrics = self._calculate_mean_reversion_metrics(
                symbol, price_data, current_price
            )
            
            risk_metrics = self._calculate_risk_metrics(
                symbol, price_data, volatility_metrics['realized_volatility']
            )
            
            # Calculate composite score
            composite_score = self._calculate_composite_score(
                liquidity_metrics, volatility_metrics, 
                momentum_metrics, mean_reversion_metrics
            )
            
            # Create IndexMetrics object
            return IndexMetrics(
                symbol=symbol,
                
                # Liquidity
                avg_daily_volume=liquidity_metrics['avg_daily_volume'],
                avg_open_interest=liquidity_metrics['avg_open_interest'],
                bid_ask_spread_bps=liquidity_metrics['bid_ask_spread_bps'],
                liquidity_score=liquidity_metrics['liquidity_score'],
                
                # Volatility
                realized_volatility=volatility_metrics['realized_volatility'],
                volatility_regime=volatility_metrics['volatility_regime'],
                vol_percentile=volatility_metrics['vol_percentile'],
                volatility_score=volatility_metrics['volatility_score'],
                
                # Momentum
                momentum_1d=momentum_metrics['momentum_1d'],
                momentum_5d=momentum_metrics['momentum_5d'],
                momentum_20d=momentum_metrics['momentum_20d'],
                trend_strength=momentum_metrics['trend_strength'],
                momentum_score=momentum_metrics['momentum_score'],
                
                # Mean reversion
                rsi_14=mean_reversion_metrics['rsi_14'],
                mean_reversion_score=mean_reversion_metrics['mean_reversion_score'],
                
                # Market structure (placeholder)
                support_resistance_strength=0.5,
                breakout_potential=0.5,
                
                # Risk
                max_drawdown_30d=risk_metrics['max_drawdown_30d'],
                var_95_1d=risk_metrics['var_95_1d'],
                risk_score=risk_metrics['risk_score'],
                
                # Final ranking
                composite_score=composite_score,
                rank=0,  # Will be set after sorting
                
                # Metadata
                last_updated=datetime.now(),
                data_quality=self._assess_data_quality(price_data, volume_data)
            )
            
        except Exception as e:
            logger.error(f"Error calculating metrics for {symbol}: {e}")
            raise
    
    def _calculate_liquidity_metrics(
        self, 
        symbol: str, 
        volume_data: List[float], 
        oi_data: List[float], 
        bid_ask_data: Dict[str, float]
    ) -> Dict[str, Any]:
        """Calculate liquidity-related metrics"""
        try:
            # Average daily volume (last 30 days or available data)
            if volume_data:
                avg_daily_volume = int(np.mean(volume_data[-30:]))
            else:
                # Use typical volume from config as fallback
                avg_daily_volume = self.index_universe[symbol]['typical_volume']
            
            # Average open interest
            if oi_data:
                avg_open_interest = int(np.mean(oi_data[-30:]))
            else:
                avg_open_interest = self.index_universe[symbol]['typical_oi']
            
            # Bid-ask spread
            if bid_ask_data and 'bid' in bid_ask_data and 'ask' in bid_ask_data:
                bid = bid_ask_data['bid']
                ask = bid_ask_data['ask']
                mid_price = (bid + ask) / 2
                spread_bps = ((ask - bid) / mid_price) * 10000 if mid_price > 0 else 100
            else:
                # Estimate typical spread for the index
                typical_spreads = {'NIFTY': 5, 'BANKNIFTY': 8, 'FINNIFTY': 10}
                spread_bps = typical_spreads.get(symbol, 15)
            
            # Calculate liquidity score (0-1, higher is better)
            # Normalize against universe
            max_volume = max(self.index_universe[idx]['typical_volume'] for idx in self.index_universe)
            max_oi = max(self.index_universe[idx]['typical_oi'] for idx in self.index_universe)
            
            volume_score = min(avg_daily_volume / max_volume, 1.0)
            oi_score = min(avg_open_interest / max_oi, 1.0)
            spread_score = max(0, 1.0 - (spread_bps / 50))  # Penalize spreads > 50 bps
            
            liquidity_score = (volume_score * 0.4 + oi_score * 0.4 + spread_score * 0.2)
            
            return {
                'avg_daily_volume': avg_daily_volume,
                'avg_open_interest': avg_open_interest,
                'bid_ask_spread_bps': spread_bps,
                'liquidity_score': liquidity_score
            }
            
        except Exception as e:
            logger.error(f"Error calculating liquidity metrics for {symbol}: {e}")
            return {
                'avg_daily_volume': 0,
                'avg_open_interest': 0,
                'bid_ask_spread_bps': 100.0,
                'liquidity_score': 0.0
            }
    
    def _calculate_volatility_metrics(
        self, 
        symbol: str, 
        price_data: pd.DataFrame, 
        current_price: float
    ) -> Dict[str, Any]:
        """Calculate volatility-related metrics"""
        try:
            if price_data.empty or len(price_data) < 10:
                # Use typical volatility as fallback
                typical_vol = self.index_universe[symbol]['typical_volatility']
                return {
                    'realized_volatility': typical_vol,
                    'volatility_regime': VolatilityRegime.NORMAL,
                    'vol_percentile': 50.0,
                    'volatility_score': 0.7
                }
            
            # Calculate realized volatility (30-day)
            if 'close' in price_data.columns:
                returns = price_data['close'].pct_change().dropna()
                realized_vol = returns.std() * np.sqrt(252)  # Annualized
            else:
                typical_vol = self.index_universe[symbol]['typical_volatility']
                realized_vol = typical_vol
            
            # Determine volatility regime
            typical_vol = self.index_universe[symbol]['typical_volatility']
            vol_ratio = realized_vol / typical_vol
            
            if vol_ratio < 0.7:
                regime = VolatilityRegime.LOW
            elif vol_ratio < 1.3:
                regime = VolatilityRegime.NORMAL
            elif vol_ratio < 2.0:
                regime = VolatilityRegime.HIGH
            else:
                regime = VolatilityRegime.EXTREME
            
            # Calculate volatility percentile (simplified)
            vol_percentile = min(vol_ratio * 50, 95)
            
            # Volatility attractiveness score
            # Prefer normal to slightly high volatility for trading opportunities
            vol_preference = self.ranking_config.get('vol_regime_preference', 'normal')
            
            if vol_preference == 'normal':
                if regime == VolatilityRegime.NORMAL:
                    vol_score = 1.0
                elif regime == VolatilityRegime.LOW:
                    vol_score = 0.8
                elif regime == VolatilityRegime.HIGH:
                    vol_score = 0.9
                else:  # EXTREME
                    vol_score = 0.3
            else:
                vol_score = 0.7  # Default score
            
            return {
                'realized_volatility': realized_vol,
                'volatility_regime': regime,
                'vol_percentile': vol_percentile,
                'volatility_score': vol_score
            }
            
        except Exception as e:
            logger.error(f"Error calculating volatility metrics for {symbol}: {e}")
            return {
                'realized_volatility': 0.20,
                'volatility_regime': VolatilityRegime.NORMAL,
                'vol_percentile': 50.0,
                'volatility_score': 0.5
            }
    
    def _calculate_momentum_metrics(
        self, 
        symbol: str, 
        price_data: pd.DataFrame, 
        current_price: float
    ) -> Dict[str, Any]:
        """Calculate momentum and trend strength metrics"""
        try:
            if price_data.empty or 'close' not in price_data.columns:
                return {
                    'momentum_1d': 0.0,
                    'momentum_5d': 0.0,
                    'momentum_20d': 0.0,
                    'trend_strength': TrendStrength.WEAK,
                    'momentum_score': 0.5
                }
            
            prices = price_data['close']
            latest_price = current_price if current_price > 0 else prices.iloc[-1]
            
            # Calculate momentum over different periods
            momentum_1d = 0.0
            momentum_5d = 0.0
            momentum_20d = 0.0
            
            if len(prices) >= 2:
                momentum_1d = (latest_price / prices.iloc[-2] - 1) * 100
            
            if len(prices) >= 6:
                momentum_5d = (latest_price / prices.iloc[-6] - 1) * 100
            
            if len(prices) >= 21:
                momentum_20d = (latest_price / prices.iloc[-21] - 1) * 100
            
            # Determine trend strength
            momentum_values = [abs(momentum_1d), abs(momentum_5d), abs(momentum_20d)]
            avg_momentum = np.mean([m for m in momentum_values if m != 0])
            
            if avg_momentum >= 3.0:
                trend_strength = TrendStrength.VERY_STRONG
                momentum_score = 1.0
            elif avg_momentum >= 2.0:
                trend_strength = TrendStrength.STRONG
                momentum_score = 0.9
            elif avg_momentum >= 1.0:
                trend_strength = TrendStrength.MODERATE
                momentum_score = 0.7
            else:
                trend_strength = TrendStrength.WEAK
                momentum_score = 0.4
            
            return {
                'momentum_1d': momentum_1d,
                'momentum_5d': momentum_5d,
                'momentum_20d': momentum_20d,
                'trend_strength': trend_strength,
                'momentum_score': momentum_score
            }
            
        except Exception as e:
            logger.error(f"Error calculating momentum metrics for {symbol}: {e}")
            return {
                'momentum_1d': 0.0,
                'momentum_5d': 0.0,
                'momentum_20d': 0.0,
                'trend_strength': TrendStrength.WEAK,
                'momentum_score': 0.5
            }
    
    def _calculate_mean_reversion_metrics(
        self, 
        symbol: str, 
        price_data: pd.DataFrame, 
        current_price: float
    ) -> Dict[str, Any]:
        """Calculate mean reversion opportunity metrics"""
        try:
            if price_data.empty or 'close' not in price_data.columns or len(price_data) < 14:
                return {
                    'rsi_14': 50.0,
                    'mean_reversion_score': 0.5
                }
            
            # Calculate RSI (14-period)
            prices = price_data['close']
            delta = prices.diff()
            
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            current_rsi = rsi.iloc[-1] if not rsi.empty else 50.0
            
            # Mean reversion score based on RSI extremes
            if current_rsi >= 80 or current_rsi <= 20:
                # Extreme readings suggest mean reversion opportunity
                mean_reversion_score = 0.9
            elif current_rsi >= 70 or current_rsi <= 30:
                # Strong readings
                mean_reversion_score = 0.7
            elif current_rsi >= 60 or current_rsi <= 40:
                # Moderate readings
                mean_reversion_score = 0.5
            else:
                # Neutral zone
                mean_reversion_score = 0.3
            
            return {
                'rsi_14': current_rsi,
                'mean_reversion_score': mean_reversion_score
            }
            
        except Exception as e:
            logger.error(f"Error calculating mean reversion metrics for {symbol}: {e}")
            return {
                'rsi_14': 50.0,
                'mean_reversion_score': 0.5
            }
    
    def _calculate_risk_metrics(
        self, 
        symbol: str, 
        price_data: pd.DataFrame, 
        volatility: float
    ) -> Dict[str, Any]:
        """Calculate risk-related metrics"""
        try:
            if price_data.empty or 'close' not in price_data.columns:
                return {
                    'max_drawdown_30d': 0.05,
                    'var_95_1d': 0.02,
                    'risk_score': 0.5
                }
            
            prices = price_data['close']
            
            # Calculate maximum drawdown (last 30 days)
            if len(prices) >= 30:
                recent_prices = prices.tail(30)
                peak = recent_prices.expanding().max()
                drawdown = (recent_prices - peak) / peak
                max_drawdown = abs(drawdown.min())
            else:
                max_drawdown = volatility * 0.3  # Estimate based on volatility
            
            # Calculate 1-day 95% VaR
            if len(prices) >= 30:
                returns = prices.pct_change().dropna()
                var_95_1d = abs(np.percentile(returns, 5))  # 5th percentile (left tail)
            else:
                var_95_1d = volatility / np.sqrt(252) * 1.65  # Normal approximation
            
            # Risk score (lower is better, so invert)
            # Normalize risk metrics
            max_acceptable_dd = 0.15  # 15% max drawdown
            max_acceptable_var = 0.05  # 5% daily VaR
            
            dd_score = max(0, 1 - (max_drawdown / max_acceptable_dd))
            var_score = max(0, 1 - (var_95_1d / max_acceptable_var))
            
            risk_score = (dd_score + var_score) / 2
            
            return {
                'max_drawdown_30d': max_drawdown,
                'var_95_1d': var_95_1d,
                'risk_score': risk_score
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics for {symbol}: {e}")
            return {
                'max_drawdown_30d': 0.05,
                'var_95_1d': 0.02,
                'risk_score': 0.5
            }
    
    def _calculate_composite_score(
        self,
        liquidity_metrics: Dict[str, Any],
        volatility_metrics: Dict[str, Any],
        momentum_metrics: Dict[str, Any],
        mean_reversion_metrics: Dict[str, Any]
    ) -> float:
        """Calculate final composite ranking score"""
        try:
            # Extract individual scores
            liquidity_score = liquidity_metrics['liquidity_score']
            volatility_score = volatility_metrics['volatility_score']
            momentum_score = momentum_metrics['momentum_score']
            mean_reversion_score = mean_reversion_metrics['mean_reversion_score']
            
            # Weighted composite score
            composite_score = (
                self.weights.liquidity_weight * liquidity_score +
                self.weights.volatility_weight * volatility_score +
                self.weights.momentum_weight * momentum_score +
                self.weights.mean_reversion_weight * mean_reversion_score
            )
            
            return min(max(composite_score, 0.0), 1.0)  # Ensure 0-1 range
            
        except Exception as e:
            logger.error(f"Error calculating composite score: {e}")
            return 0.5
    
    def _passes_quality_filters(self, metrics: IndexMetrics) -> bool:
        """Check if index passes minimum quality filters"""
        try:
            # Volume filter
            if metrics.avg_daily_volume < self.min_avg_daily_volume:
                return False
            
            # Open interest filter
            if metrics.avg_open_interest < self.min_open_interest:
                return False
            
            # Spread filter
            if metrics.bid_ask_spread_bps > self.max_bid_ask_spread_pct * 10000:
                return False
            
            # Liquidity score filter
            if metrics.liquidity_score < self.min_liquidity_score:
                return False
            
            # Volatility filter
            if metrics.realized_volatility > self.max_volatility_threshold:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error in quality filters: {e}")
            return False
    
    def _assess_data_quality(self, price_data: pd.DataFrame, volume_data: List[float]) -> float:
        """Assess quality of available data"""
        try:
            quality_factors = []
            
            # Price data completeness
            if not price_data.empty:
                quality_factors.append(min(len(price_data) / 30, 1.0))  # Want at least 30 days
            else:
                quality_factors.append(0.3)  # Poor quality without price data
            
            # Volume data availability
            if volume_data:
                quality_factors.append(min(len(volume_data) / 30, 1.0))
            else:
                quality_factors.append(0.5)  # Moderate quality without volume
            
            return np.mean(quality_factors)
            
        except Exception as e:
            logger.error(f"Error assessing data quality: {e}")
            return 0.5
    
    def get_ranking_summary(self, ranked_indices: List[IndexMetrics]) -> str:
        """Get formatted ranking summary"""
        summary = "📊 Universe Ranking Summary\n"
        summary += "=" * 50 + "\n"
        
        for i, metrics in enumerate(ranked_indices):
            summary += f"{i+1}. {metrics.symbol} (Score: {metrics.composite_score:.3f})\n"
            summary += f"   Liquidity: {metrics.liquidity_score:.2f} | "
            summary += f"Volatility: {metrics.volatility_score:.2f} | "
            summary += f"Momentum: {metrics.momentum_score:.2f}\n"
            summary += f"   Volume: {metrics.avg_daily_volume:,} lots | "
            summary += f"Spread: {metrics.bid_ask_spread_bps:.1f} bps\n\n"
        
        return summary

# Convenience functions
def rank_indices_for_trading(
    market_data: Dict[str, Dict[str, Any]],
    max_selections: int = 5,
    config_path: str = None
) -> List[IndexMetrics]:
    """
    Standalone function to rank indices for trading
    """
    ranker = UniverseRanker(config_path)
    return ranker.rank_universe(market_data, max_selections)

def get_top_k_indices(
    market_data: Dict[str, Dict[str, Any]], 
    k: int = 3
) -> List[str]:
    """
    Get top K index symbols for trading
    """
    ranked_metrics = rank_indices_for_trading(market_data, k)
    return [metrics.symbol for metrics in ranked_metrics]

def is_index_tradeable(
    symbol: str,
    market_data: Dict[str, Any],
    config_path: str = None
) -> bool:
    """
    Check if a specific index meets tradability criteria
    """
    ranker = UniverseRanker(config_path)
    
    try:
        metrics = ranker._calculate_index_metrics(symbol, market_data)
        return ranker._passes_quality_filters(metrics)
    except Exception as e:
        logger.error(f"Error checking tradability for {symbol}: {e}")
        return False