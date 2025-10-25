"""
Volatility-Based Position Sizing System
Implementation of Oxford paper formula with futures-specific considerations
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import math
import logging
from utils.config import load_config

logger = logging.getLogger(__name__)

@dataclass
class PositionSizingResult:
    """Result of position sizing calculation"""
    symbol: str
    recommended_quantity: int  # In units (not lots)
    recommended_lots: int      # In standard lots
    notional_value: float      # Total notional exposure
    risk_amount: float         # Amount at risk
    volatility_target: float   # Target volatility used
    actual_volatility: float   # Current realized volatility
    vol_scaling_factor: float  # Volatility adjustment factor
    rl_signal_strength: float  # RL signal (-1 to +1)
    leverage_used: float       # Effective leverage
    kelly_fraction: float      # Kelly criterion fraction
    margin_required: float     # Estimated margin requirement
    confidence_score: float    # Confidence in sizing (0-1)
    warnings: List[str]        # Any sizing warnings

@dataclass
class IndexVolatilityProfile:
    """Volatility profile for each index"""
    index_name: str
    current_volatility: float     # Current realized volatility
    long_term_volatility: float   # Long-term average volatility
    volatility_regime: str        # 'low', 'normal', 'high', 'extreme'
    vol_percentile: float         # Percentile in historical distribution
    adjustment_factor: float      # Factor to adjust position size
    confidence: float            # Confidence in volatility estimate

class VolatilityPositionSizer:
    """
    Advanced position sizing using volatility targeting with futures-specific considerations
    Based on Oxford paper methodology with enhancements for Indian futures
    """
    
    def __init__(self, config_path: str = None):
        """Initialize position sizer with configuration"""
        self.config = load_config(config_path or "configs/risk.yaml")
        self.position_config = self.config.get('position_sizing', {})
        
        # Index-specific configurations with Indian futures specifications
        self.index_configs = {
            'NIFTY': {
                'lot_size': 75,
                'tick_size': 0.05,
                'typical_volatility': 0.15,
                'vol_floor': 0.08,          # Minimum volatility assumption
                'vol_ceiling': 0.40,        # Maximum volatility cap
                'leverage_cap': 10,         # Max leverage for NIFTY
                'liquidity_score': 1.0,     # Highest liquidity
                'margin_multiplier': 0.10,  # ~10% margin requirement
                'contract_multiplier': 1    # Standard multiplier
            },
            'BANKNIFTY': {
                'lot_size': 15,
                'tick_size': 0.05,
                'typical_volatility': 0.25,
                'vol_floor': 0.12,
                'vol_ceiling': 0.60,
                'leverage_cap': 8,          # Slightly lower due to higher volatility
                'liquidity_score': 0.95,    # Very high liquidity
                'margin_multiplier': 0.12,  # Higher margin due to volatility
                'contract_multiplier': 1
            },
            'FINNIFTY': {
                'lot_size': 25,
                'tick_size': 0.05,
                'typical_volatility': 0.20,
                'vol_floor': 0.10,
                'vol_ceiling': 0.50,
                'leverage_cap': 8,
                'liquidity_score': 0.85,    # Good liquidity
                'margin_multiplier': 0.11,
                'contract_multiplier': 1
            }
        }
        
        # Position sizing parameters
        self.base_volatility_target = self.position_config.get('base_volatility_target', 0.20)
        self.max_kelly_fraction = self.position_config.get('max_kelly_fraction', 0.25)
        self.min_kelly_fraction = self.position_config.get('min_kelly_fraction', 0.02)
        self.confidence_threshold = self.position_config.get('confidence_threshold', 0.70)
        
        logger.info("Volatility position sizer initialized")
    
    def compute_position_size(
        self,
        symbol: str,
        current_price: float,
        rl_signal: float,  # -1 to +1
        account_balance: float,
        available_margin: float,
        win_rate: float = 0.55,
        avg_win_loss_ratio: float = 1.2,
        current_positions: Dict[str, Any] = None,
        market_data: Dict[str, Any] = None
    ) -> PositionSizingResult:
        """
        Compute optimal position size using volatility targeting
        
        Args:
            symbol: Futures symbol (e.g., 'NIFTY25OCT')
            current_price: Current price of the instrument
            rl_signal: RL strategy signal strength (-1 to +1)
            account_balance: Total account balance
            available_margin: Available margin for new positions
            win_rate: Historical win rate (0-1)
            avg_win_loss_ratio: Average win/loss ratio
            current_positions: Current portfolio positions
            market_data: Additional market data for volatility calculation
            
        Returns:
            PositionSizingResult with recommended position size
        """
        try:
            # Extract index information
            index_name = self._extract_index_name(symbol)
            index_config = self.index_configs.get(index_name, self.index_configs['NIFTY'])
            
            warnings = []
            
            # 1. Calculate current volatility
            vol_profile = self._calculate_volatility_profile(symbol, current_price, market_data)
            
            # 2. Calculate Kelly fraction
            kelly_fraction = self._calculate_kelly_fraction(win_rate, avg_win_loss_ratio)
            
            # 3. Apply volatility scaling (Oxford paper methodology)
            vol_scaling_factor = self._calculate_volatility_scaling(vol_profile, index_config)
            
            # 4. Apply RL signal scaling
            signal_scaling = abs(rl_signal)  # Use absolute value for position size
            
            # 5. Calculate base position size
            base_position_value = account_balance * kelly_fraction * signal_scaling * vol_scaling_factor
            
            # 6. Apply futures-specific adjustments
            adjusted_position_value = self._apply_futures_adjustments(
                base_position_value, index_config, available_margin, warnings
            )
            
            # 7. Convert to lots and handle lot size constraints
            quantity, lots, notional_value = self._convert_to_lots(
                adjusted_position_value, current_price, index_config
            )
            
            # 8. Final validations and risk checks
            final_result = self._validate_and_finalize(
                symbol=symbol,
                quantity=quantity,
                lots=lots,
                current_price=current_price,
                notional_value=notional_value,
                account_balance=account_balance,
                available_margin=available_margin,
                vol_profile=vol_profile,
                kelly_fraction=kelly_fraction,
                vol_scaling_factor=vol_scaling_factor,
                rl_signal=rl_signal,
                index_config=index_config,
                warnings=warnings
            )
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error in position sizing for {symbol}: {e}")
            return self._create_error_result(symbol, str(e))
    
    def _calculate_volatility_profile(
        self, 
        symbol: str, 
        current_price: float, 
        market_data: Dict[str, Any] = None
    ) -> IndexVolatilityProfile:
        """Calculate comprehensive volatility profile"""
        try:
            index_name = self._extract_index_name(symbol)
            index_config = self.index_configs.get(index_name, {})
            
            # In production, this would use historical data
            # For now, using configurable parameters with market regime detection
            
            if market_data and 'price_history' in market_data:
                # Calculate realized volatility from price history
                prices = np.array(market_data['price_history'])
                returns = np.diff(np.log(prices))
                current_vol = np.std(returns) * np.sqrt(252)  # Annualized
            else:
                # Use typical volatility with some randomness for simulation
                typical_vol = index_config.get('typical_volatility', 0.20)
                # Add some market regime simulation
                vol_multiplier = np.random.uniform(0.8, 1.5)  # Market regime factor
                current_vol = typical_vol * vol_multiplier
            
            # Determine volatility regime
            long_term_vol = index_config.get('typical_volatility', 0.20)
            vol_ratio = current_vol / long_term_vol
            
            if vol_ratio < 0.7:
                regime = 'low'
                adjustment_factor = 1.2  # Increase position size in low vol
            elif vol_ratio < 1.3:
                regime = 'normal'
                adjustment_factor = 1.0
            elif vol_ratio < 2.0:
                regime = 'high'
                adjustment_factor = 0.8  # Reduce position size in high vol
            else:
                regime = 'extreme'
                adjustment_factor = 0.5  # Significantly reduce in extreme vol
            
            # Apply volatility floors and ceilings
            vol_floor = index_config.get('vol_floor', 0.08)
            vol_ceiling = index_config.get('vol_ceiling', 0.60)
            current_vol = np.clip(current_vol, vol_floor, vol_ceiling)
            
            return IndexVolatilityProfile(
                index_name=index_name,
                current_volatility=current_vol,
                long_term_volatility=long_term_vol,
                volatility_regime=regime,
                vol_percentile=min(vol_ratio * 50, 95),  # Approximate percentile
                adjustment_factor=adjustment_factor,
                confidence=0.8 if market_data else 0.6  # Higher confidence with real data
            )
            
        except Exception as e:
            logger.error(f"Error calculating volatility profile: {e}")
            # Return default profile
            return IndexVolatilityProfile(
                index_name=index_name,
                current_volatility=0.20,
                long_term_volatility=0.20,
                volatility_regime='normal',
                vol_percentile=50,
                adjustment_factor=1.0,
                confidence=0.5
            )
    
    def _calculate_kelly_fraction(self, win_rate: float, avg_win_loss_ratio: float) -> float:
        """Calculate Kelly criterion fraction"""
        try:
            # Kelly formula: f = (bp - q) / b
            # where b = odds received (avg_win_loss_ratio)
            #       p = probability of winning (win_rate)
            #       q = probability of losing (1 - win_rate)
            
            if win_rate <= 0 or win_rate >= 1 or avg_win_loss_ratio <= 0:
                return self.min_kelly_fraction
            
            kelly = (win_rate * avg_win_loss_ratio - (1 - win_rate)) / avg_win_loss_ratio
            
            # Apply conservative caps for futures trading
            kelly = np.clip(kelly, self.min_kelly_fraction, self.max_kelly_fraction)
            
            # Additional safety: reduce Kelly if win rate or ratio is questionable
            if win_rate < 0.45 or avg_win_loss_ratio < 1.1:
                kelly *= 0.5  # Halve Kelly for poor historical performance
            
            return kelly
            
        except Exception as e:
            logger.error(f"Error calculating Kelly fraction: {e}")
            return self.min_kelly_fraction
    
    def _calculate_volatility_scaling(
        self, 
        vol_profile: IndexVolatilityProfile, 
        index_config: Dict[str, Any]
    ) -> float:
        """Calculate volatility scaling factor (Oxford paper methodology)"""
        try:
            # Oxford paper: scale inversely with volatility to maintain constant risk
            # Scaling factor = target_volatility / current_volatility
            
            target_vol = self.base_volatility_target
            current_vol = vol_profile.current_volatility
            
            # Base scaling
            vol_scaling = target_vol / current_vol
            
            # Apply volatility regime adjustment
            vol_scaling *= vol_profile.adjustment_factor
            
            # Apply confidence adjustment
            confidence_factor = 0.5 + (vol_profile.confidence * 0.5)
            vol_scaling *= confidence_factor
            
            # Apply caps to prevent extreme positions
            min_scaling = 0.2  # Minimum 20% of base position
            max_scaling = 3.0  # Maximum 300% of base position
            vol_scaling = np.clip(vol_scaling, min_scaling, max_scaling)
            
            return vol_scaling
            
        except Exception as e:
            logger.error(f"Error calculating volatility scaling: {e}")
            return 1.0
    
    def _apply_futures_adjustments(
        self,
        base_position_value: float,
        index_config: Dict[str, Any],
        available_margin: float,
        warnings: List[str]
    ) -> float:
        """Apply futures-specific adjustments"""
        try:
            # 1. Leverage constraint
            leverage_cap = index_config.get('leverage_cap', 5)
            max_position_by_leverage = available_margin * leverage_cap
            
            if base_position_value > max_position_by_leverage:
                warnings.append(f"Position capped by leverage limit ({leverage_cap}x)")
                base_position_value = max_position_by_leverage
            
            # 2. Margin utilization constraint
            max_margin_usage = self.position_config.get('max_margin_utilization', 0.60)
            margin_multiplier = index_config.get('margin_multiplier', 0.10)
            
            # Estimate required margin
            required_margin = base_position_value * margin_multiplier
            max_allowed_margin = available_margin * max_margin_usage
            
            if required_margin > max_allowed_margin:
                warnings.append(f"Position reduced due to margin utilization limit")
                base_position_value = max_allowed_margin / margin_multiplier
            
            # 3. Liquidity adjustment
            liquidity_score = index_config.get('liquidity_score', 1.0)
            if liquidity_score < 0.9:
                liquidity_factor = 0.7 + (liquidity_score * 0.3)  # Scale down for lower liquidity
                base_position_value *= liquidity_factor
                warnings.append(f"Position reduced due to liquidity constraints")
            
            # 4. Minimum position threshold
            min_position_value = self.position_config.get('min_position_value', 10000)
            if base_position_value < min_position_value:
                if base_position_value > min_position_value * 0.7:
                    base_position_value = min_position_value
                    warnings.append("Position increased to minimum threshold")
                else:
                    base_position_value = 0
                    warnings.append("Position too small - recommended to skip")
            
            return base_position_value
            
        except Exception as e:
            logger.error(f"Error applying futures adjustments: {e}")
            return base_position_value
    
    def _convert_to_lots(
        self,
        position_value: float,
        current_price: float,
        index_config: Dict[str, Any]
    ) -> Tuple[int, int, float]:
        """Convert position value to lots and calculate actual quantities"""
        try:
            lot_size = index_config.get('lot_size', 25)
            
            # Calculate raw quantity
            raw_quantity = position_value / current_price
            
            # Round to lot size multiples
            lots = max(1, round(raw_quantity / lot_size))
            actual_quantity = lots * lot_size
            actual_notional = actual_quantity * current_price
            
            return actual_quantity, lots, actual_notional
            
        except Exception as e:
            logger.error(f"Error converting to lots: {e}")
            return 0, 0, 0.0
    
    def _validate_and_finalize(
        self,
        symbol: str,
        quantity: int,
        lots: int,
        current_price: float,
        notional_value: float,
        account_balance: float,
        available_margin: float,
        vol_profile: IndexVolatilityProfile,
        kelly_fraction: float,
        vol_scaling_factor: float,
        rl_signal: float,
        index_config: Dict[str, Any],
        warnings: List[str]
    ) -> PositionSizingResult:
        """Final validation and result compilation"""
        try:
            # Calculate risk metrics
            margin_required = notional_value * index_config.get('margin_multiplier', 0.10)
            leverage_used = notional_value / account_balance if account_balance > 0 else 0
            
            # Estimate risk amount (potential loss)
            stop_loss_points = index_config.get('stop_loss_points', 50)  # Will be configurable
            risk_amount = quantity * stop_loss_points * index_config.get('tick_size', 0.05)
            
            # Calculate confidence score
            confidence_factors = [
                vol_profile.confidence,
                min(kelly_fraction / self.max_kelly_fraction, 1.0),
                1.0 if abs(rl_signal) > 0.3 else abs(rl_signal) / 0.3,
                1.0 if margin_required < available_margin * 0.5 else 0.5
            ]
            confidence_score = np.mean(confidence_factors)
            
            # Final validations
            if margin_required > available_margin:
                warnings.append("Insufficient margin - position rejected")
                quantity = lots = 0
                notional_value = margin_required = risk_amount = 0
            
            if confidence_score < self.confidence_threshold:
                warnings.append(f"Low confidence ({confidence_score:.2f}) - consider reducing position")
            
            return PositionSizingResult(
                symbol=symbol,
                recommended_quantity=quantity,
                recommended_lots=lots,
                notional_value=notional_value,
                risk_amount=risk_amount,
                volatility_target=self.base_volatility_target,
                actual_volatility=vol_profile.current_volatility,
                vol_scaling_factor=vol_scaling_factor,
                rl_signal_strength=rl_signal,
                leverage_used=leverage_used,
                kelly_fraction=kelly_fraction,
                margin_required=margin_required,
                confidence_score=confidence_score,
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"Error in final validation: {e}")
            return self._create_error_result(symbol, str(e))
    
    def _create_error_result(self, symbol: str, error_msg: str) -> PositionSizingResult:
        """Create error result"""
        return PositionSizingResult(
            symbol=symbol,
            recommended_quantity=0,
            recommended_lots=0,
            notional_value=0.0,
            risk_amount=0.0,
            volatility_target=self.base_volatility_target,
            actual_volatility=0.20,
            vol_scaling_factor=1.0,
            rl_signal_strength=0.0,
            leverage_used=0.0,
            kelly_fraction=0.0,
            margin_required=0.0,
            confidence_score=0.0,
            warnings=[f"Error in position sizing: {error_msg}"]
        )
    
    def _extract_index_name(self, symbol: str) -> str:
        """Extract index name from futures symbol"""
        symbol_upper = symbol.upper()
        
        if 'BANKNIFTY' in symbol_upper or 'BANK NIFTY' in symbol_upper:
            return 'BANKNIFTY'
        elif 'FINNIFTY' in symbol_upper or 'FIN NIFTY' in symbol_upper:
            return 'FINNIFTY'
        elif 'NIFTY' in symbol_upper:
            return 'NIFTY'
        else:
            # Default to NIFTY for unknown symbols
            return 'NIFTY'
    
    def get_sizing_summary(self, result: PositionSizingResult) -> str:
        """Get formatted position sizing summary"""
        status = "✅ APPROVED" if result.recommended_lots > 0 else "❌ REJECTED"
        
        summary = f"{status} - {result.symbol}\n"
        summary += f"Recommended: {result.recommended_lots} lots ({result.recommended_quantity} units)\n"
        summary += f"Notional: ₹{result.notional_value:,.0f} | Margin: ₹{result.margin_required:,.0f}\n"
        summary += f"Leverage: {result.leverage_used:.1f}x | Risk: ₹{result.risk_amount:,.0f}\n"
        summary += f"Volatility: {result.actual_volatility:.1%} | Kelly: {result.kelly_fraction:.2%}\n"
        summary += f"RL Signal: {result.rl_signal_strength:+.2f} | Confidence: {result.confidence_score:.1%}\n"
        
        if result.warnings:
            summary += "\nWarnings:\n"
            for warning in result.warnings:
                summary += f"  ⚠️ {warning}\n"
        
        return summary

# Convenience functions for execution agent integration
def calculate_position_size(
    symbol: str,
    current_price: float,
    rl_signal: float,
    account_balance: float,
    available_margin: float,
    win_rate: float = 0.55,
    avg_win_loss_ratio: float = 1.2,
    config_path: str = None
) -> PositionSizingResult:
    """
    Standalone function for position size calculation
    """
    sizer = VolatilityPositionSizer(config_path)
    return sizer.compute_position_size(
        symbol=symbol,
        current_price=current_price,
        rl_signal=rl_signal,
        account_balance=account_balance,
        available_margin=available_margin,
        win_rate=win_rate,
        avg_win_loss_ratio=avg_win_loss_ratio
    )

def get_recommended_lots(sizing_result: PositionSizingResult) -> int:
    """Get recommended number of lots"""
    return sizing_result.recommended_lots

def is_position_viable(sizing_result: PositionSizingResult) -> bool:
    """Check if position is viable"""
    return (
        sizing_result.recommended_lots > 0 and
        sizing_result.confidence_score >= 0.5 and
        not any("rejected" in warning.lower() for warning in sizing_result.warnings)
    )