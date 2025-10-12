"""
Futures Contract Management Module

Handles futures-specific operations including:
- Contract rollover logic
- Continuous contract construction
- Basis calculations
- Futures symbol generation
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List

import pandas as pd
import numpy as np

from .constants import (
    MONTH_CODES,
    TRADING_DAYS_PER_YEAR,
    DEFAULT_RISK_FREE_RATE,
    PRICE_COLUMNS
)

logger = logging.getLogger(__name__)


def handle_contract_rollover(current_contract: pd.DataFrame, next_contract: pd.DataFrame,
                           rollover_date: str, method: str = 'ratio') -> pd.DataFrame:
    """Handle contract rollover for continuous futures series.
    
    Args:
        current_contract: DataFrame for expiring contract
        next_contract: DataFrame for new contract  
        rollover_date: Date to perform rollover (YYYY-MM-DD)
        method: 'ratio' or 'difference' adjustment method
        
    Returns:
        Adjusted DataFrame for seamless transition
    """
    if current_contract.empty or next_contract.empty:
        logger.error("Empty contract data for rollover")
        return pd.DataFrame()
    
    rollover_dt = pd.to_datetime(rollover_date)
    
    # Find the rollover point data
    current_before = current_contract[current_contract.index <= rollover_dt]
    next_after = next_contract[next_contract.index >= rollover_dt]
    
    if current_before.empty or next_after.empty:
        logger.error("Insufficient data around rollover date %s", rollover_date)
        return current_contract
    
    # Get prices at rollover point
    current_price = current_before.iloc[-1]['close']
    next_price = next_after.iloc[0]['close'] 
    
    if method == 'ratio':
        # Ratio adjustment (multiplicative)
        adjustment_factor = current_price / next_price
        adjusted_next = next_after.copy()
        
        for col in PRICE_COLUMNS:
            if col in adjusted_next.columns:
                adjusted_next[col] = adjusted_next[col] * adjustment_factor
                
        logger.info("Applied ratio adjustment: factor=%.4f", adjustment_factor)
        
    elif method == 'difference':
        # Difference adjustment (additive)
        adjustment_diff = current_price - next_price
        adjusted_next = next_after.copy()
        
        for col in PRICE_COLUMNS:
            if col in adjusted_next.columns:
                adjusted_next[col] = adjusted_next[col] + adjustment_diff
                
        logger.info("Applied difference adjustment: diff=%.2f", adjustment_diff)
        
    else:
        logger.error("Unknown rollover method: %s", method)
        return current_contract
    
    # Combine the series
    continuous_series = pd.concat([current_before, adjusted_next])
    continuous_series = continuous_series.sort_index()
    
    # Add rollover metadata
    continuous_series.attrs = {
        'rollover_date': rollover_date,
        'rollover_method': method,
        'adjustment_factor': adjustment_factor if method == 'ratio' else adjustment_diff
    }
    
    return continuous_series


def construct_continuous_contract(contracts_data: Dict[str, pd.DataFrame], 
                                rollover_schedule: List[str],
                                method: str = 'ratio') -> pd.DataFrame:
    """Construct continuous futures contract from multiple expiry series.
    
    Args:
        contracts_data: Dict mapping contract_month -> DataFrame
        rollover_schedule: List of rollover dates in chronological order
        method: 'ratio' or 'difference' adjustment method
        
    Returns:
        Continuous contract DataFrame
    """
    if not contracts_data or not rollover_schedule:
        logger.error("Insufficient data for continuous contract construction")
        return pd.DataFrame()
    
    # Sort contracts by rollover dates
    contract_names = list(contracts_data.keys())
    if len(contract_names) < 2:
        logger.warning("Need at least 2 contracts for continuous series")
        return list(contracts_data.values())[0]
    
    # Start with the first contract
    continuous_series = contracts_data[contract_names[0]].copy()
    
    # Apply rollovers sequentially
    for i, rollover_date in enumerate(rollover_schedule):
        if i + 1 >= len(contract_names):
            break
            
        current_contract = continuous_series
        next_contract = contracts_data[contract_names[i + 1]]
        
        continuous_series = handle_contract_rollover(
            current_contract, next_contract, rollover_date, method
        )
        
        logger.info("Rolled over to %s on %s", contract_names[i + 1], rollover_date)
    
    # Add continuous contract metadata
    continuous_series.attrs = {
        'contract_type': 'continuous',
        'rollover_method': method,
        'num_rollovers': len(rollover_schedule),
        'contracts_used': contract_names
    }
    
    return continuous_series


def compute_basis(futures_price: pd.Series, spot_price: pd.Series, 
                 days_to_expiry: pd.Series) -> Dict[str, pd.Series]:
    """Compute futures basis and related metrics.
    
    Args:
        futures_price: Futures closing prices
        spot_price: Spot index closing prices  
        days_to_expiry: Days remaining to contract expiry
        
    Returns:
        Dict with basis metrics: 'basis', 'basis_pct', 'annualized_basis'
    """
    # Absolute basis (futures - spot)
    basis = futures_price - spot_price
    
    # Percentage basis ((futures - spot) / spot * 100)
    basis_pct = (basis / spot_price) * 100
    
    # Annualized basis (extrapolate to annual terms)
    annualized_basis = basis_pct * (TRADING_DAYS_PER_YEAR / days_to_expiry.replace(0, 1))  # Avoid div by zero
    
    # Theoretical fair value (cost of carry model)
    # This is simplified - in practice would include risk-free rate and dividends
    theoretical_basis = spot_price * (DEFAULT_RISK_FREE_RATE / 365) * days_to_expiry
    basis_deviation = basis - theoretical_basis
    
    return {
        'basis': basis,
        'basis_pct': basis_pct,
        'annualized_basis': annualized_basis,
        'theoretical_basis': theoretical_basis,
        'basis_deviation': basis_deviation
    }


def generate_futures_symbol(index_symbol: str, contract_month: str, 
                           market_config: Dict[str, Any]) -> str:
    """Generate NSE futures symbol based on index and contract month.
    
    Args:
        index_symbol: Index name (e.g., 'NIFTY')
        contract_month: Contract month ('current', 'next', 'far')
        market_config: Market configuration dict
        
    Returns:
        NSE futures symbol (e.g., 'NIFTY25JANFUT')
    """
    # Get current date to determine contract expiry months
    current_date = datetime.now()
    
    # Map contract month to actual month
    if contract_month == 'current':
        expiry_date = current_date
    elif contract_month == 'next':
        expiry_date = current_date + timedelta(days=30)
    elif contract_month == 'far':
        expiry_date = current_date + timedelta(days=60)
    else:
        # Assume it's already a month name or date
        expiry_date = current_date
    
    # Get month codes from config or use defaults
    config_month_codes = market_config.get('contract_naming', {}).get('month_codes', {})
    month_codes_map = {f"{i:02d}": month for i, month in MONTH_CODES.items()}
    month_codes_map.update(config_month_codes)
    
    year = str(expiry_date.year)[-2:]  # Last 2 digits of year
    month_num = expiry_date.strftime("%m")
    month_code = month_codes_map.get(month_num, "JAN")
    
    # NSE naming convention: {INDEX}{YY}{MON}FUT
    futures_symbol = f"{index_symbol}{year}{month_code}FUT"
    
    return futures_symbol