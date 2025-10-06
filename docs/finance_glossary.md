# 🧮 Index Futures Finance Reference

> **Developer Reference**: Tick value ↔ bp costs ↔ notional mappings for NSE Index Futures

---

## **Quick Reference Table**

| Index | Lot Size | Point Value | Tick Size | Margin (MIS) | Notional @ 24k |
|-------|----------|-------------|-----------|--------------|----------------|
| **NIFTY** | 50 | ₹50/point | ₹0.05 | ₹60,000 | ₹12,00,000 |
| **BANKNIFTY** | 25 | ₹25/point | ₹0.05 | ₹75,000 | ₹6,00,000 |
| **FINNIFTY** | 40 | ₹40/point | ₹0.05 | ₹50,000 | ₹9,60,000 |
| **MIDCPNIFTY** | 50 | ₹50/point | ₹0.05 | ₹65,000 | ₹12,00,000 |
| **CNXPHARMA** | 30 | ₹30/point | ₹0.05 | ₹70,000 | ₹7,20,000 |
| **CNXIT** | 50 | ₹50/point | ₹0.05 | ₹60,000 | ₹12,00,000 |

---

## **Point Value to Rupee Conversion**

```python
# From configs/market.yaml
POINT_VALUES = {
    'NIFTY': 50.0,      # ₹50 per index point
    'BANKNIFTY': 25.0,  # ₹25 per index point  
    'FINNIFTY': 40.0,   # ₹40 per index point
    'MIDCPNIFTY': 50.0, # ₹50 per index point
    'CNXPHARMA': 30.0,  # ₹30 per index point
    'CNXIT': 50.0       # ₹50 per index point
}

# P&L Calculation: point_diff * point_value
# Example: NIFTY moves 100 points = 100 * 50 = ₹5,000 P&L per lot
```

## **Margin Requirements (SPAN + Exposure)**

```python
# From configs/rl.yaml - MIS margin requirements
MARGIN_REQUIREMENTS = {
    'NIFTY': 60000,      # ₹60k per lot (MIS)
    'BANKNIFTY': 75000,  # ₹75k per lot (MIS)
    'FINNIFTY': 50000    # ₹50k per lot (MIS)
}

# NRML = 2x MIS margin (overnight positions)
# SPAN ≈ 70% of total, Exposure ≈ 30% of total
```

## **Transaction Cost Structure**

```python
# NSE F&O statutory charges
def calculate_transaction_costs(turnover):
    brokerage = max(20, turnover * 0.0001)  # ₹20 or 0.01%
    stt = turnover * 0.0001                 # 0.01% on sell side  
    exchange = turnover * 0.00002           # 0.002%
    sebi = (turnover / 10000000) * 10       # ₹10 per crore
    gst = (brokerage + exchange) * 0.18     # 18% GST
    
    return brokerage + stt + exchange + sebi + gst
```

## **Lot Size to Notional Exposure**

```python
# From configs/market.yaml
LOT_SIZES = {
    'NIFTY': 50, 'BANKNIFTY': 25, 'FINNIFTY': 40,
    'MIDCPNIFTY': 50, 'CNXPHARMA': 30, 'CNXIT': 50
}

# Notional = index_level * lot_size * lots
# Example: NIFTY @ 24,000 with 2 lots = 24,000 × 50 × 2 = ₹24,00,000
```

## **Rollover Mechanics**

```python
# Implemented in agents/data_agent.py
def handle_contract_rollover(current_price, next_price, method='ratio'):
    """Adjust prices for continuous series"""
    if method == 'ratio':
        adjustment_factor = current_price / next_price
        return adjustment_factor
    else:  # difference
        return current_price - next_price

# Rollover occurs 5-7 days before monthly expiry (last Thursday)
```

## **Basis Calculation** 

```python
# Implemented in agents/data_agent.py  
def compute_basis(futures_price, spot_price, days_to_expiry):
    """Calculate futures basis vs spot"""
    basis = futures_price - spot_price
    basis_pct = (basis / spot_price) * 100
    
    # Theoretical fair value (cost of carry)
    carry_cost = spot_price * (0.06 / 365) * days_to_expiry  # 6% risk-free rate
    
    return basis, basis_pct, carry_cost
```

---

## **Code Integration Points**

### **Existing Functions:**
- **Data Agent**: `compute_basis()`, `handle_contract_rollover()` in `agents/data_agent.py`
- **Loaders**: `calculate_futures_margin()`, `calculate_notional_value()` in `data/loaders.py`  
- **Market Config**: All static specs in `configs/market.yaml`

### **Usage Example:**
```python
# Position P&L
pnl = (current_price - entry_price) * point_value * lots

# Position sizing  
max_lots = available_margin // margin_per_lot

# Risk calculation
notional_exposure = index_level * lot_size * lots
```

---

**Reference**: NSE Contract Specifications | **Updated**: October 2025  