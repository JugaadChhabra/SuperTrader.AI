"""
Pre-Trade Risk Validation System
Comprehensive risk checks before order execution for futures trading
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
import pandas as pd
from utils.config import load_config

logger = logging.getLogger(__name__)

class RiskCheckStatus(Enum):
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"

@dataclass
class RiskCheckResult:
    """Individual risk check result"""
    check_name: str
    status: RiskCheckStatus
    message: str
    current_value: Optional[float] = None
    limit_value: Optional[float] = None
    breach_severity: str = "low"  # low, medium, high, critical

@dataclass
class PreTradeRiskReport:
    """Complete pre-trade risk assessment"""
    overall_status: RiskCheckStatus
    individual_checks: List[RiskCheckResult]
    total_checks: int
    passed_checks: int
    failed_checks: int
    warning_checks: int
    recommendation: str
    risk_score: float  # 0-100, higher = riskier

class PreTradeRiskValidator:
    """
    Comprehensive pre-trade risk validation system for futures trading
    """
    
    def __init__(self, risk_config_path: str = None):
        """Initialize risk validator with configuration"""
        self.risk_config = load_config(risk_config_path or "configs/risk.yaml")
        self.futures_config = self.risk_config.get('futures', {})
        
        # Index-specific configurations
        self.index_configs = {
            'NIFTY': {
                'lot_size': 75,
                'tick_size': 0.05,
                'typical_volatility': 0.15,
                'stop_loss_points': 50,
                'take_profit_points': 100,
                'max_leverage': 10
            },
            'BANKNIFTY': {
                'lot_size': 15,
                'tick_size': 0.05,
                'typical_volatility': 0.25,
                'stop_loss_points': 200,
                'take_profit_points': 400,
                'max_leverage': 8
            },
            'FINNIFTY': {
                'lot_size': 25,
                'tick_size': 0.05,
                'typical_volatility': 0.20,
                'stop_loss_points': 100,
                'take_profit_points': 200,
                'max_leverage': 8
            }
        }
        
        logger.info("Pre-trade risk validator initialized")
    
    def validate_order(
        self,
        symbol: str,
        side: str,  # 'buy' or 'sell'
        quantity: int,
        price: float,
        order_type: str,
        current_positions: Dict[str, Any],
        available_margin: float,
        account_balance: float
    ) -> PreTradeRiskReport:
        """
        Comprehensive pre-trade risk validation
        
        Returns complete risk assessment with pass/fail for each check
        """
        checks = []
        
        # 1. Index Exposure Check
        exposure_check = self._check_index_exposure(
            symbol, side, quantity, price, current_positions
        )
        checks.append(exposure_check)
        
        # 2. Margin Adequacy Check
        margin_check = self._check_margin_adequacy(
            symbol, quantity, price, available_margin
        )
        checks.append(margin_check)
        
        # 3. Position Limits Check
        position_check = self._check_position_limits(
            symbol, side, quantity, current_positions
        )
        checks.append(position_check)
        
        # 4. Expiry Proximity Check
        expiry_check = self._check_expiry_proximity(symbol)
        checks.append(expiry_check)
        
        # 5. Leverage Check
        leverage_check = self._check_leverage_limits(
            symbol, quantity, price, account_balance
        )
        checks.append(leverage_check)
        
        # 6. Correlation Check
        correlation_check = self._check_correlation_limits(
            symbol, side, quantity, price, current_positions
        )
        checks.append(correlation_check)
        
        # 7. Lot Size Validation
        lot_check = self._check_lot_size_compliance(symbol, quantity)
        checks.append(lot_check)
        
        # 8. Market Hours Check
        market_check = self._check_market_hours()
        checks.append(market_check)
        
        # 9. Volatility Regime Check
        vol_check = self._check_volatility_regime(symbol, price)
        checks.append(vol_check)
        
        # Calculate overall status and risk score
        return self._compile_risk_report(checks)
    
    def _check_index_exposure(
        self, 
        symbol: str, 
        side: str, 
        quantity: int, 
        price: float, 
        current_positions: Dict[str, Any]
    ) -> RiskCheckResult:
        """Check maximum exposure per index (30% limit)"""
        try:
            index_name = self._extract_index_name(symbol)
            
            # Calculate current exposure for this index
            current_exposure = 0
            total_portfolio_value = 0
            
            for pos_symbol, position in current_positions.items():
                pos_value = abs(position.get('quantity', 0)) * position.get('price', 0)
                total_portfolio_value += pos_value
                
                if self._extract_index_name(pos_symbol) == index_name:
                    current_exposure += pos_value
            
            # Calculate new trade value
            new_trade_value = quantity * price
            new_total_exposure = current_exposure + new_trade_value
            
            # Get exposure limit from config
            max_exposure_pct = self.risk_config.get('max_index_exposure', 0.30)
            max_exposure_value = total_portfolio_value * max_exposure_pct
            
            exposure_ratio = new_total_exposure / max(total_portfolio_value, 1)
            
            if exposure_ratio <= max_exposure_pct:
                return RiskCheckResult(
                    check_name="Index Exposure",
                    status=RiskCheckStatus.PASS,
                    message=f"{index_name} exposure within limits",
                    current_value=exposure_ratio * 100,
                    limit_value=max_exposure_pct * 100
                )
            else:
                return RiskCheckResult(
                    check_name="Index Exposure",
                    status=RiskCheckStatus.FAIL,
                    message=f"{index_name} exposure exceeds {max_exposure_pct*100:.1f}% limit",
                    current_value=exposure_ratio * 100,
                    limit_value=max_exposure_pct * 100,
                    breach_severity="high"
                )
                
        except Exception as e:
            logger.error(f"Error in index exposure check: {e}")
            return RiskCheckResult(
                check_name="Index Exposure",
                status=RiskCheckStatus.FAIL,
                message=f"Error calculating exposure: {str(e)}",
                breach_severity="critical"
            )
    
    def _check_margin_adequacy(
        self, 
        symbol: str, 
        quantity: int, 
        price: float, 
        available_margin: float
    ) -> RiskCheckResult:
        """Check if adequate margin is available"""
        try:
            index_name = self._extract_index_name(symbol)
            index_config = self.index_configs.get(index_name, {})
            
            # Calculate required margin (approximate)
            lot_size = index_config.get('lot_size', 25)
            lots = quantity // lot_size
            
            # Margin requirement varies by broker and volatility
            # Using conservative estimate
            margin_per_lot = price * lot_size * 0.10  # ~10% margin requirement
            total_margin_required = lots * margin_per_lot
            
            # Keep buffer for price movements
            margin_buffer = self.risk_config.get('margin_buffer', 0.20)
            required_with_buffer = total_margin_required * (1 + margin_buffer)
            
            # Check utilization limits
            max_utilization = self.risk_config.get('max_margin_utilization', 0.60)
            max_allowed_usage = available_margin * max_utilization
            
            if required_with_buffer <= max_allowed_usage:
                utilization_pct = (required_with_buffer / available_margin) * 100
                return RiskCheckResult(
                    check_name="Margin Adequacy",
                    status=RiskCheckStatus.PASS,
                    message=f"Sufficient margin available",
                    current_value=utilization_pct,
                    limit_value=max_utilization * 100
                )
            else:
                utilization_pct = (required_with_buffer / available_margin) * 100
                return RiskCheckResult(
                    check_name="Margin Adequacy",
                    status=RiskCheckStatus.FAIL,
                    message=f"Insufficient margin - requires ₹{required_with_buffer:.0f}, available ₹{available_margin:.0f}",
                    current_value=utilization_pct,
                    limit_value=max_utilization * 100,
                    breach_severity="critical"
                )
                
        except Exception as e:
            logger.error(f"Error in margin adequacy check: {e}")
            return RiskCheckResult(
                check_name="Margin Adequacy",
                status=RiskCheckStatus.FAIL,
                message=f"Error calculating margin: {str(e)}",
                breach_severity="critical"
            )
    
    def _check_position_limits(
        self, 
        symbol: str, 
        side: str, 
        quantity: int, 
        current_positions: Dict[str, Any]
    ) -> RiskCheckResult:
        """Check exchange position limits"""
        try:
            index_name = self._extract_index_name(symbol)
            
            # Get position limits from config
            position_limits = self.risk_config.get('position_limits', {})
            max_lots_per_symbol = position_limits.get(index_name, 3600)  # Default SEBI limit
            
            # Calculate current and new positions
            current_qty = current_positions.get(symbol, {}).get('quantity', 0)
            new_total_qty = abs(current_qty + (quantity if side == 'buy' else -quantity))
            
            # Convert to lots
            lot_size = self.index_configs.get(index_name, {}).get('lot_size', 25)
            lots = new_total_qty // lot_size
            
            if lots <= max_lots_per_symbol:
                return RiskCheckResult(
                    check_name="Position Limits",
                    status=RiskCheckStatus.PASS,
                    message=f"Position within exchange limits",
                    current_value=lots,
                    limit_value=max_lots_per_symbol
                )
            else:
                return RiskCheckResult(
                    check_name="Position Limits",
                    status=RiskCheckStatus.FAIL,
                    message=f"Position exceeds exchange limit of {max_lots_per_symbol} lots",
                    current_value=lots,
                    limit_value=max_lots_per_symbol,
                    breach_severity="critical"
                )
                
        except Exception as e:
            logger.error(f"Error in position limits check: {e}")
            return RiskCheckResult(
                check_name="Position Limits",
                status=RiskCheckStatus.FAIL,
                message=f"Error checking position limits: {str(e)}",
                breach_severity="medium"
            )
    
    def _check_expiry_proximity(self, symbol: str) -> RiskCheckResult:
        """Check expiry proximity - no new positions 2 days before expiry"""
        try:
            # Extract expiry date from symbol (assuming format like NIFTY25OCT24800CE)
            expiry_date = self._extract_expiry_date(symbol)
            
            if expiry_date is None:
                return RiskCheckResult(
                    check_name="Expiry Proximity",
                    status=RiskCheckStatus.WARNING,
                    message="Could not determine expiry date",
                    breach_severity="low"
                )
            
            # Check if within 2 days of expiry
            days_to_expiry = (expiry_date - datetime.now().date()).days
            min_days_buffer = self.risk_config.get('min_days_to_expiry', 2)
            
            if days_to_expiry >= min_days_buffer:
                return RiskCheckResult(
                    check_name="Expiry Proximity",
                    status=RiskCheckStatus.PASS,
                    message=f"{days_to_expiry} days to expiry",
                    current_value=days_to_expiry,
                    limit_value=min_days_buffer
                )
            else:
                severity = "critical" if days_to_expiry <= 0 else "high"
                return RiskCheckResult(
                    check_name="Expiry Proximity",
                    status=RiskCheckStatus.FAIL,
                    message=f"Too close to expiry - only {days_to_expiry} days remaining",
                    current_value=days_to_expiry,
                    limit_value=min_days_buffer,
                    breach_severity=severity
                )
                
        except Exception as e:
            logger.error(f"Error in expiry proximity check: {e}")
            return RiskCheckResult(
                check_name="Expiry Proximity",
                status=RiskCheckStatus.WARNING,
                message=f"Error checking expiry: {str(e)}",
                breach_severity="medium"
            )
    
    def _check_leverage_limits(
        self, 
        symbol: str, 
        quantity: int, 
        price: float, 
        account_balance: float
    ) -> RiskCheckResult:
        """Check leverage limits"""
        try:
            index_name = self._extract_index_name(symbol)
            index_config = self.index_configs.get(index_name, {})
            
            # Calculate notional exposure
            notional_exposure = quantity * price
            
            # Calculate effective leverage
            effective_leverage = notional_exposure / max(account_balance, 1)
            
            # Get max leverage from config
            max_leverage = index_config.get('max_leverage', 5)
            
            if effective_leverage <= max_leverage:
                return RiskCheckResult(
                    check_name="Leverage Limits",
                    status=RiskCheckStatus.PASS,
                    message=f"Leverage within limits",
                    current_value=effective_leverage,
                    limit_value=max_leverage
                )
            else:
                return RiskCheckResult(
                    check_name="Leverage Limits",
                    status=RiskCheckStatus.FAIL,
                    message=f"Leverage {effective_leverage:.1f}x exceeds limit of {max_leverage}x",
                    current_value=effective_leverage,
                    limit_value=max_leverage,
                    breach_severity="high"
                )
                
        except Exception as e:
            logger.error(f"Error in leverage check: {e}")
            return RiskCheckResult(
                check_name="Leverage Limits",
                status=RiskCheckStatus.FAIL,
                message=f"Error calculating leverage: {str(e)}",
                breach_severity="medium"
            )
    
    def _check_correlation_limits(
        self, 
        symbol: str, 
        side: str, 
        quantity: int, 
        price: float, 
        current_positions: Dict[str, Any]
    ) -> RiskCheckResult:
        """Check correlation limits to avoid overexposure to correlated indices"""
        try:
            index_name = self._extract_index_name(symbol)
            
            # Define correlation groups
            correlation_groups = {
                'broad_market': ['NIFTY', 'SENSEX'],
                'banking': ['BANKNIFTY', 'FINNIFTY'],
                'sectoral': ['CNXPHARMA', 'CNXIT', 'CNXAUTO']
            }
            
            # Find correlation group for this symbol
            symbol_group = None
            for group, indices in correlation_groups.items():
                if index_name in indices:
                    symbol_group = group
                    break
            
            if symbol_group is None:
                return RiskCheckResult(
                    check_name="Correlation Limits",
                    status=RiskCheckStatus.PASS,
                    message="Index not in correlation group",
                    breach_severity="low"
                )
            
            # Calculate exposure in correlation group
            group_exposure = 0
            total_exposure = 0
            
            for pos_symbol, position in current_positions.items():
                pos_index = self._extract_index_name(pos_symbol)
                pos_value = abs(position.get('quantity', 0)) * position.get('price', 0)
                total_exposure += pos_value
                
                if pos_index in correlation_groups[symbol_group]:
                    group_exposure += pos_value
            
            # Add new trade
            new_trade_value = quantity * price
            new_group_exposure = group_exposure + new_trade_value
            new_total_exposure = total_exposure + new_trade_value
            
            # Check correlation limit
            max_correlation_exposure = self.risk_config.get('max_correlation_exposure', 0.50)
            correlation_ratio = new_group_exposure / max(new_total_exposure, 1)
            
            if correlation_ratio <= max_correlation_exposure:
                return RiskCheckResult(
                    check_name="Correlation Limits",
                    status=RiskCheckStatus.PASS,
                    message=f"{symbol_group} correlation within limits",
                    current_value=correlation_ratio * 100,
                    limit_value=max_correlation_exposure * 100
                )
            else:
                return RiskCheckResult(
                    check_name="Correlation Limits",
                    status=RiskCheckStatus.FAIL,
                    message=f"{symbol_group} correlation exceeds {max_correlation_exposure*100:.1f}% limit",
                    current_value=correlation_ratio * 100,
                    limit_value=max_correlation_exposure * 100,
                    breach_severity="medium"
                )
                
        except Exception as e:
            logger.error(f"Error in correlation check: {e}")
            return RiskCheckResult(
                check_name="Correlation Limits",
                status=RiskCheckStatus.WARNING,
                message=f"Error checking correlation: {str(e)}",
                breach_severity="low"
            )
    
    def _check_lot_size_compliance(self, symbol: str, quantity: int) -> RiskCheckResult:
        """Check lot size compliance"""
        try:
            index_name = self._extract_index_name(symbol)
            index_config = self.index_configs.get(index_name, {})
            lot_size = index_config.get('lot_size', 25)
            
            if quantity % lot_size == 0:
                lots = quantity // lot_size
                return RiskCheckResult(
                    check_name="Lot Size Compliance",
                    status=RiskCheckStatus.PASS,
                    message=f"Quantity matches lot size ({lots} lots)",
                    current_value=lots,
                    limit_value=lot_size
                )
            else:
                return RiskCheckResult(
                    check_name="Lot Size Compliance",
                    status=RiskCheckStatus.FAIL,
                    message=f"Quantity {quantity} not multiple of lot size {lot_size}",
                    current_value=quantity,
                    limit_value=lot_size,
                    breach_severity="medium"
                )
                
        except Exception as e:
            logger.error(f"Error in lot size check: {e}")
            return RiskCheckResult(
                check_name="Lot Size Compliance",
                status=RiskCheckStatus.FAIL,
                message=f"Error checking lot size: {str(e)}",
                breach_severity="low"
            )
    
    def _check_market_hours(self) -> RiskCheckResult:
        """Check if market is open"""
        try:
            now = datetime.now()
            
            # Market hours: 9:15 AM to 3:30 PM IST on weekdays
            market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
            market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
            
            # Check if weekend
            if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
                return RiskCheckResult(
                    check_name="Market Hours",
                    status=RiskCheckStatus.FAIL,
                    message="Market closed - Weekend",
                    breach_severity="medium"
                )
            
            # Check if within market hours
            if market_open <= now <= market_close:
                return RiskCheckResult(
                    check_name="Market Hours",
                    status=RiskCheckStatus.PASS,
                    message="Market is open"
                )
            else:
                return RiskCheckResult(
                    check_name="Market Hours",
                    status=RiskCheckStatus.FAIL,
                    message="Market closed - Outside trading hours",
                    breach_severity="medium"
                )
                
        except Exception as e:
            logger.error(f"Error in market hours check: {e}")
            return RiskCheckResult(
                check_name="Market Hours",
                status=RiskCheckStatus.WARNING,
                message=f"Error checking market hours: {str(e)}",
                breach_severity="low"
            )
    
    def _check_volatility_regime(self, symbol: str, price: float) -> RiskCheckResult:
        """Check volatility regime"""
        try:
            index_name = self._extract_index_name(symbol)
            index_config = self.index_configs.get(index_name, {})
            
            # This would typically use historical volatility calculation
            # For now, using placeholder logic
            typical_vol = index_config.get('typical_volatility', 0.20)
            
            # In production, calculate actual volatility and compare
            # current_vol = calculate_realized_volatility(symbol, days=30)
            
            # Placeholder: assume normal volatility regime
            vol_regime = "normal"  # Could be "low", "normal", "high", "extreme"
            
            if vol_regime in ["low", "normal"]:
                return RiskCheckResult(
                    check_name="Volatility Regime",
                    status=RiskCheckStatus.PASS,
                    message=f"Normal volatility regime for {index_name}"
                )
            elif vol_regime == "high":
                return RiskCheckResult(
                    check_name="Volatility Regime",
                    status=RiskCheckStatus.WARNING,
                    message=f"High volatility regime for {index_name} - use caution",
                    breach_severity="medium"
                )
            else:  # extreme
                return RiskCheckResult(
                    check_name="Volatility Regime",
                    status=RiskCheckStatus.FAIL,
                    message=f"Extreme volatility regime for {index_name} - avoid new positions",
                    breach_severity="high"
                )
                
        except Exception as e:
            logger.error(f"Error in volatility regime check: {e}")
            return RiskCheckResult(
                check_name="Volatility Regime",
                status=RiskCheckStatus.WARNING,
                message=f"Error checking volatility: {str(e)}",
                breach_severity="low"
            )
    
    def _compile_risk_report(self, checks: List[RiskCheckResult]) -> PreTradeRiskReport:
        """Compile individual checks into overall risk report"""
        total_checks = len(checks)
        passed_checks = len([c for c in checks if c.status == RiskCheckStatus.PASS])
        failed_checks = len([c for c in checks if c.status == RiskCheckStatus.FAIL])
        warning_checks = len([c for c in checks if c.status == RiskCheckStatus.WARNING])
        
        # Determine overall status
        if failed_checks > 0:
            overall_status = RiskCheckStatus.FAIL
        elif warning_checks > 0:
            overall_status = RiskCheckStatus.WARNING
        else:
            overall_status = RiskCheckStatus.PASS
        
        # Calculate risk score (0-100, higher = riskier)
        risk_score = 0
        for check in checks:
            if check.status == RiskCheckStatus.FAIL:
                if check.breach_severity == "critical":
                    risk_score += 25
                elif check.breach_severity == "high":
                    risk_score += 15
                elif check.breach_severity == "medium":
                    risk_score += 10
                else:
                    risk_score += 5
            elif check.status == RiskCheckStatus.WARNING:
                risk_score += 5
        
        risk_score = min(risk_score, 100)
        
        # Generate recommendation
        if overall_status == RiskCheckStatus.PASS:
            recommendation = "Order approved - All risk checks passed"
        elif overall_status == RiskCheckStatus.WARNING:
            recommendation = "Proceed with caution - Some warnings detected"
        else:
            critical_failures = [c for c in checks if c.status == RiskCheckStatus.FAIL and c.breach_severity == "critical"]
            if critical_failures:
                recommendation = "Order REJECTED - Critical risk violations detected"
            else:
                recommendation = "Order NOT RECOMMENDED - Risk violations detected"
        
        return PreTradeRiskReport(
            overall_status=overall_status,
            individual_checks=checks,
            total_checks=total_checks,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            warning_checks=warning_checks,
            recommendation=recommendation,
            risk_score=risk_score
        )
    
    def _extract_index_name(self, symbol: str) -> str:
        """Extract index name from futures symbol"""
        # Handle different symbol formats
        symbol_upper = symbol.upper()
        
        if 'NIFTY' in symbol_upper and 'BANK' not in symbol_upper:
            return 'NIFTY'
        elif 'BANKNIFTY' in symbol_upper or 'BANK NIFTY' in symbol_upper:
            return 'BANKNIFTY'
        elif 'FINNIFTY' in symbol_upper or 'FIN NIFTY' in symbol_upper:
            return 'FINNIFTY'
        else:
            # Extract first part before numbers/month
            import re
            match = re.match(r'^([A-Z]+)', symbol_upper)
            return match.group(1) if match else symbol_upper
    
    def _extract_expiry_date(self, symbol: str) -> Optional[datetime.date]:
        """Extract expiry date from futures symbol"""
        try:
            # This would need to be implemented based on your symbol format
            # Placeholder implementation
            import re
            
            # Look for date patterns in symbol
            # Example: NIFTY25OCT24800CE -> October 25, 2024
            date_match = re.search(r'(\d{1,2})([A-Z]{3})(\d{2})', symbol)
            
            if date_match:
                day = int(date_match.group(1))
                month_str = date_match.group(2)
                year = 2000 + int(date_match.group(3))
                
                month_map = {
                    'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4,
                    'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8,
                    'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
                }
                
                month = month_map.get(month_str)
                if month:
                    return datetime(year, month, day).date()
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting expiry date from {symbol}: {e}")
            return None

# Convenience functions for execution agent integration
def validate_pre_trade_risk(
    symbol: str,
    side: str,
    quantity: int,
    price: float,
    current_positions: Dict[str, Any],
    available_margin: float,
    account_balance: float,
    risk_config_path: str = None
) -> PreTradeRiskReport:
    """
    Standalone function for pre-trade risk validation
    """
    validator = PreTradeRiskValidator(risk_config_path)
    return validator.validate_order(
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=price,
        current_positions=current_positions,
        available_margin=available_margin,
        account_balance=account_balance
    )

def is_order_approved(risk_report: PreTradeRiskReport) -> bool:
    """Check if order is approved based on risk report"""
    return risk_report.overall_status == RiskCheckStatus.PASS

def get_risk_summary(risk_report: PreTradeRiskReport) -> str:
    """Get formatted risk summary"""
    status_emoji = {
        RiskCheckStatus.PASS: "✅",
        RiskCheckStatus.WARNING: "⚠️",
        RiskCheckStatus.FAIL: "❌"
    }
    
    summary = f"{status_emoji[risk_report.overall_status]} {risk_report.recommendation}\n"
    summary += f"Risk Score: {risk_report.risk_score:.1f}/100\n"
    summary += f"Checks: {risk_report.passed_checks}✅ {risk_report.warning_checks}⚠️ {risk_report.failed_checks}❌\n"
    
    # Show failed checks
    failed_checks = [c for c in risk_report.individual_checks if c.status == RiskCheckStatus.FAIL]
    if failed_checks:
        summary += "\nFailed Checks:\n"
        for check in failed_checks:
            summary += f"  • {check.check_name}: {check.message}\n"
    
    return summary