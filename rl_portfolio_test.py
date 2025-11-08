#!/usr/bin/env python3
"""
RL Strategy Portfolio Manager

Uses the actual RL Strategy Agent for AI-driven trading decisions.
This shows real rule-based AI trading with enhanced Phase 4 signals.
"""

import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.portfolio_simulator import PortfolioSimulator, ActionType
from agents.rl_strategy_agent import (
    sample_action, 
    build_state_representation,
    extract_phase4_signals
)

class AITradeTracker:
    """Enhanced trade tracking with RL Strategy Agent"""
    
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.current_cash = initial_capital
        self.positions: Dict[str, Dict] = {}
        self.completed_trades: List[Dict] = []
        self.trade_sequence = 0
        
        print("🤖 RL Strategy Agent initialized for AI trading decisions")
    
    def _create_realistic_market_state(self, symbol: str, price: float, timestamp: datetime) -> Dict[str, Any]:
        """Create realistic market features for the RL Strategy Agent"""
        
        time_of_day = timestamp.hour + timestamp.minute / 60.0
        
        # Create realistic technical features based on symbol characteristics
        volatility_base = {
            "RELIANCE": 0.18, "TCS": 0.15, "HDFCBANK": 0.20, "INFY": 0.16,
            "ITC": 0.12, "ASIANPAINT": 0.14, "LT": 0.22, "WIPRO": 0.25
        }
        
        # RSI simulation based on symbol trends
        rsi_bias = {
            "TCS": -5, "INFY": -3, "ITC": +8, "ASIANPAINT": +5, 
            "RELIANCE": +2, "HDFCBANK": 0, "LT": -2, "WIPRO": +3
        }
        
        # Technical features
        base_rsi = 50 + rsi_bias.get(symbol, 0) + np.random.normal(0, 10)
        rsi = np.clip(base_rsi, 20, 80)
        
        macd = np.random.normal(rsi_bias.get(symbol, 0) * 0.5, 2.0)
        volatility = volatility_base.get(symbol, 0.15) * np.random.uniform(0.8, 1.2)
        
        price_features = {
            'close': price,
            'open': price * np.random.uniform(0.998, 1.002),
            'high': price * np.random.uniform(1.002, 1.008),
            'low': price * np.random.uniform(0.992, 0.998)
        }
        
        tech_features = {
            'close': price,
            'rsi': rsi,
            'macd': macd,
            'volatility_20': volatility,
            'volume_ratio': np.random.uniform(0.8, 1.5),
            'intraday_range_pct': abs(price_features['high'] - price_features['low']) / price * 100,
            'vwap_dist': np.random.uniform(-0.3, 0.3)
        }
        
        # Enhanced Phase 4 features
        phase4_features = {
            'pcr_percentile': np.random.uniform(20, 80),
            'momentum_confluence_score': np.random.randint(0, 4),
            'market_regime_strong_trend': np.random.choice([True, False]),
            'market_risk_score': np.random.uniform(0.3, 0.8),
            'setup_quality_score': np.random.uniform(0.4, 0.9),
            'volatility_regime_high': volatility > 0.20,
            'volatility_regime_low': volatility < 0.12,
            'options_flow_bias': np.random.uniform(-0.5, 0.5)
        }
        
        # Add PCR directional signals
        pcr_pct = phase4_features['pcr_percentile']
        phase4_features['pcr_bullish'] = pcr_pct > 75  # High PCR = oversold = bullish
        phase4_features['pcr_bearish'] = pcr_pct < 25  # Low PCR = overbought = bearish
        
        # Other required features
        sentiment_features = {
            'market_sentiment_5min': np.random.uniform(-0.2, 0.2),
            'sentiment_momentum_15min': np.random.uniform(-0.1, 0.1)
        }
        
        oi_features = {
            'oi_momentum': np.random.uniform(-30, 30),
            'oi_change_5bar': np.random.uniform(-500, 500)
        }
        
        basis_features = {
            'basis_pct': np.random.uniform(-0.05, 0.05)
        }
        
        time_features = {
            'minutes_to_close': max(0, (15.25 - time_of_day) * 60),
            'session_phase': self._get_session_phase(time_of_day)
        }
        
        vix = np.random.uniform(12, 28)
        
        return {
            'price_feats': price_features,
            'tech_feats': tech_features,
            'senti_feats': sentiment_features,
            'oi': oi_features,
            'basis': basis_features,
            'time_feats': time_features,
            'vix': vix,
            'options_feats': {},
            'phase4_features': phase4_features
        }
    
    def _get_session_phase(self, time_of_day: float) -> str:
        """Determine market session phase"""
        if time_of_day < 9.5:
            return 'opening_range'
        elif time_of_day < 12.0:
            return 'morning'
        elif time_of_day < 15.0:
            return 'afternoon'
        else:
            return 'closing'
    
    def get_ai_decision(self, symbol: str, price: float, timestamp: datetime, has_position: bool = False) -> Dict:
        """Get AI trading decision using RL Strategy Agent"""
        
        try:
            # Create realistic market state
            market_data = self._create_realistic_market_state(symbol, price, timestamp)
            
            # Build state representation
            state_vector = build_state_representation(
                price_feats=market_data['price_feats'],
                tech_feats=market_data['tech_feats'],
                senti_feats=market_data['senti_feats'],
                basis=market_data['basis'],
                oi=market_data['oi'],
                vix=market_data['vix'],
                time_feats=market_data['time_feats'],
                options_feats=market_data['options_feats']
            )
            
            # Get RL Strategy decision with Phase 4 features
            time_remaining = market_data['time_feats']['minutes_to_close']
            rl_decision = sample_action(
                state=state_vector,
                mode='eval',
                time_remaining=time_remaining,
                phase4_features=market_data['phase4_features']
            )
            
            # Extract decision details
            raw_action = rl_decision['action']  # -1, 0, 1
            
            # Convert to portfolio action based on position status
            if raw_action > 0.3 and not has_position:
                portfolio_action = ActionType.BUY
                decision_type = "BUY"
            elif raw_action < -0.3 and has_position:
                portfolio_action = ActionType.SELL
                decision_type = "SELL"
            else:
                portfolio_action = ActionType.HOLD
                decision_type = "HOLD"
            
            # Create detailed reason
            rsi = market_data['tech_feats']['rsi']
            macd = market_data['tech_feats']['macd']
            phase4 = market_data['phase4_features']
            
            reason_details = []
            reason_details.append(f"RL Signal: {raw_action:+.2f}")
            reason_details.append(f"RSI: {rsi:.1f}")
            reason_details.append(f"MACD: {macd:+.2f}")
            
            if phase4['pcr_bullish']:
                reason_details.append("PCR Bullish")
            elif phase4['pcr_bearish']:
                reason_details.append("PCR Bearish")
            
            if phase4['momentum_confluence_score'] > 2:
                reason_details.append(f"Confluence: {phase4['momentum_confluence_score']}")
            
            if phase4['high_quality_setup']:
                reason_details.append("High Quality Setup")
            
            reason = f"AI {decision_type}: " + " | ".join(reason_details)
            
            return {
                'action': portfolio_action,
                'reason': reason,
                'ai_details': {
                    'rl_signal': raw_action,
                    'decision_confidence': rl_decision.get('confidence', 0.5),
                    'market_features': {
                        'rsi': rsi,
                        'macd': macd,
                        'volatility': market_data['tech_feats']['volatility_20'],
                        'time_to_close': time_remaining
                    },
                    'phase4_signals': {
                        'pcr_sentiment': rl_decision.get('pcr_sentiment', 'neutral'),
                        'confluence_score': phase4['momentum_confluence_score'],
                        'market_regime': rl_decision.get('market_regime', 'consolidating'),
                        'risk_level': rl_decision.get('risk_level', 'normal'),
                        'setup_quality': rl_decision.get('setup_quality', 'standard')
                    },
                    'raw_rl_decision': rl_decision
                }
            }
            
        except Exception as e:
            return {
                'action': ActionType.HOLD,
                'reason': f"AI Error: {str(e)[:50]}",
                'ai_details': {'error': str(e)}
            }
    
    def execute_trade(self, simulator, symbol: str, action: ActionType, price: float, reason: str, ai_details: Dict) -> Dict:
        """Execute trade with AI details tracking"""
        
        result = simulator.execute(symbol=symbol, action=action, price=price, sentiment=0.5)
        
        if not result.success:
            return {"success": False, "error": result.error_message}
        
        self.trade_sequence += 1
        timestamp = datetime.now()
        
        if action == ActionType.BUY:
            return self._handle_buy(symbol, result, timestamp, reason, ai_details)
        elif action == ActionType.SELL:
            return self._handle_sell(symbol, result, timestamp, reason, ai_details)
        else:
            return {"success": False, "error": "Hold action not tracked"}
    
    def _handle_buy(self, symbol: str, result, timestamp: datetime, reason: str, ai_details: Dict) -> Dict:
        """Handle buy with AI tracking"""
        
        total_cost = result.quantity * result.price + result.transaction_costs
        self.current_cash = result.net_cost
        
        if symbol in self.positions:
            existing = self.positions[symbol]
            total_shares = existing['quantity'] + result.quantity
            total_cost_basis = existing['total_cost'] + total_cost
            avg_price = total_cost_basis / total_shares
            
            self.positions[symbol].update({
                'quantity': total_shares,
                'avg_entry_price': avg_price,
                'total_cost': total_cost_basis,
                'ai_details': ai_details
            })
        else:
            self.positions[symbol] = {
                'quantity': result.quantity,
                'entry_price': result.price,
                'avg_entry_price': result.price,
                'total_cost': total_cost,
                'entry_time': timestamp,
                'entry_reason': reason,
                'ai_details': ai_details
            }
        
        return {
            "success": True, "action": "BUY", "symbol": symbol,
            "quantity": result.quantity, "price": result.price,
            "cost": total_cost, "cash_balance": self.current_cash,
            "ai_details": ai_details
        }
    
    def _handle_sell(self, symbol: str, result, timestamp: datetime, reason: str, ai_details: Dict) -> Dict:
        """Handle sell with AI tracking"""
        
        if symbol not in self.positions:
            return {"success": False, "error": f"No position in {symbol} to sell"}
        
        position = self.positions[symbol]
        shares_sold = abs(result.quantity)
        
        # Calculate P&L
        proceeds = shares_sold * result.price - result.transaction_costs
        cost_basis = (position['total_cost'] / position['quantity']) * shares_sold
        realized_pnl = proceeds - cost_basis
        return_pct = (result.price / position['avg_entry_price'] - 1) * 100
        holding_days = (timestamp - position['entry_time']).days
        
        self.current_cash += proceeds
        
        # Create trade record with AI details
        trade_record = {
            'trade_id': f"RL{self.trade_sequence:03d}",
            'symbol': symbol,
            'quantity': shares_sold,
            'entry_date': position['entry_time'].strftime('%Y-%m-%d %H:%M:%S'),
            'entry_price': position['avg_entry_price'],
            'exit_date': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'exit_price': result.price,
            'entry_value': cost_basis,
            'exit_value': proceeds,
            'gross_pnl': shares_sold * (result.price - position['avg_entry_price']),
            'transaction_costs': result.transaction_costs + (cost_basis * 0.001),
            'net_pnl': realized_pnl,
            'return_percent': return_pct,
            'holding_days': holding_days,
            'cash_balance': self.current_cash,
            'entry_reason': position['entry_reason'],
            'exit_reason': reason,
            # AI-specific fields
            'entry_ai_signal': position.get('ai_details', {}),
            'exit_ai_signal': ai_details,
            'ai_model_used': 'RL_Strategy_Agent_Phase4',
            'ai_signal_strength': ai_details.get('rl_signal', 0.0),
            'ai_confidence': ai_details.get('decision_confidence', 0.0)
        }
        
        self.completed_trades.append(trade_record)
        
        # Update position
        remaining_shares = position['quantity'] - shares_sold
        if remaining_shares <= 0:
            del self.positions[symbol]
        else:
            remaining_cost = position['total_cost'] - cost_basis
            self.positions[symbol].update({
                'quantity': remaining_shares,
                'total_cost': remaining_cost
            })
        
        return {
            "success": True, "action": "SELL", "symbol": symbol,
            "quantity": shares_sold, "price": result.price,
            "pnl": realized_pnl, "return_pct": return_pct,
            "cash_balance": self.current_cash, "ai_details": ai_details
        }
    
    def get_portfolio_summary(self, current_prices: Dict[str, float]) -> Dict:
        """Get portfolio summary"""
        
        total_invested = sum(pos['total_cost'] for pos in self.positions.values())
        total_market_value = sum(
            pos['quantity'] * current_prices.get(symbol, pos['avg_entry_price'])
            for symbol, pos in self.positions.items()
        )
        
        unrealized_pnl = total_market_value - total_invested
        total_realized_pnl = sum(trade['net_pnl'] for trade in self.completed_trades)
        total_portfolio_value = self.current_cash + total_market_value
        total_return = total_portfolio_value - self.initial_capital
        
        return {
            'initial_capital': self.initial_capital,
            'current_cash': self.current_cash,
            'invested_value': total_invested,
            'market_value': total_market_value,
            'portfolio_value': total_portfolio_value,
            'unrealized_pnl': unrealized_pnl,
            'realized_pnl': total_realized_pnl,
            'total_pnl': unrealized_pnl + total_realized_pnl,
            'total_return_pct': (total_return / self.initial_capital) * 100,
            'completed_trades': len(self.completed_trades),
            'open_positions': len(self.positions)
        }


def run_rl_strategy_portfolio_test():
    """Run RL Strategy driven portfolio test"""
    
    print("🚀 SuperTrader.AI - RL Strategy Portfolio Manager")
    print("=" * 55)
    print("🤖 Using: Enhanced RL Strategy Agent with Phase 4 Signals")
    print("=" * 55)
    
    # Initialize
    initial_capital = 1_000_000.0
    simulator = PortfolioSimulator(initial_capital=initial_capital)
    tracker = AITradeTracker(initial_capital)
    
    # Indian equity universe
    stocks = {
        "RELIANCE": 2850.0,
        "TCS": 4200.0,
        "HDFCBANK": 1650.0,
        "INFY": 1820.0,
        "ITC": 485.0,
        "ASIANPAINT": 2950.0,
        "LT": 3650.0,
        "WIPRO": 290.0
    }
    
    print(f"💰 Initial Capital: ₹{initial_capital:,.0f}")
    print(f"📊 Stock Universe: {list(stocks.keys())}")
    print()
    
    # Phase 1: RL Strategy driven portfolio construction  
    print("🤖 Phase 1: RL Strategy Portfolio Construction")
    print("-" * 45)
    
    constructed_positions = 0
    
    for symbol, price in stocks.items():
        ai_decision = tracker.get_ai_decision(symbol, price, datetime.now(), has_position=False)
        
        if ai_decision['action'] == ActionType.BUY:
            result = tracker.execute_trade(
                simulator, symbol, ActionType.BUY, price,
                ai_decision['reason'], ai_decision['ai_details']
            )
            
            if result['success']:
                constructed_positions += 1
                print(f"🤖 RL BOUGHT {symbol}: {result['quantity']} shares @ ₹{price:.2f}")
                print(f"   📊 {ai_decision['reason']}")
                
                # Show detailed AI analysis
                ai_details = ai_decision['ai_details']
                print(f"   🧠 Signal Strength: {ai_details['rl_signal']:+.3f}")
                print(f"   📈 Market: RSI {ai_details['market_features']['rsi']:.1f}, MACD {ai_details['market_features']['macd']:+.2f}")
                
                if 'phase4_signals' in ai_details:
                    p4 = ai_details['phase4_signals']
                    print(f"   🎯 Phase 4: {p4['pcr_sentiment']}, Confluence {p4['confluence_score']}, {p4['risk_level']} risk")
                
                print(f"   💰 Cost: ₹{result['cost']:,.0f} | Cash: ₹{result['cash_balance']:,.0f}")
                print()
        else:
            print(f"⚪ RL SKIPPED {symbol} @ ₹{price:.2f}")
            print(f"   📊 {ai_decision['reason']}")
            print()
    
    print(f"✅ Portfolio Construction Complete: {constructed_positions} positions built")
    print()
    
    # Phase 2: Time progression and rebalancing
    print("📊 Phase 2: Market Evolution & RL Rebalancing (6 weeks later)")
    print("-" * 60)
    
    future_time = datetime.now() + timedelta(days=42)
    
    # Simulate realistic market movements
    new_prices = {}
    for symbol, base_price in stocks.items():
        # Create market-like movements
        if symbol in ["TCS", "INFY"]:  # IT pressure
            change = np.random.uniform(-0.12, -0.02)  # -12% to -2%
        elif symbol in ["ITC", "ASIANPAINT"]:  # Consumer strength
            change = np.random.uniform(0.08, 0.25)   # +8% to +25%
        elif symbol == "RELIANCE":  # Energy volatility
            change = np.random.uniform(-0.05, 0.15)  # -5% to +15%
        else:
            change = np.random.uniform(-0.08, 0.12)  # -8% to +12%
        
        new_prices[symbol] = base_price * (1 + change)
    
    print("📈 Market Price Evolution:")
    for symbol, new_price in new_prices.items():
        old_price = stocks[symbol]
        change_pct = (new_price / old_price - 1) * 100
        trend = "📈" if change_pct > 0 else "📉"
        print(f"   {trend} {symbol:10}: ₹{old_price:6.0f} → ₹{new_price:6.0f} ({change_pct:+5.1f}%)")
    print()
    
    # RL Strategy rebalancing decisions
    print("🤖 RL Strategy Rebalancing Analysis:")
    print("-" * 40)
    
    trades_executed = 0
    
    for symbol in list(tracker.positions.keys()):
        if symbol in new_prices:
            ai_decision = tracker.get_ai_decision(symbol, new_prices[symbol], future_time, has_position=True)
            
            if ai_decision['action'] == ActionType.SELL:
                result = tracker.execute_trade(
                    simulator, symbol, ActionType.SELL, new_prices[symbol],
                    ai_decision['reason'], ai_decision['ai_details']
                )
                
                if result['success']:
                    trades_executed += 1
                    print(f"🤖 RL SOLD {symbol}: {result['quantity']} shares @ ₹{new_prices[symbol]:.2f}")
                    print(f"   📊 {ai_decision['reason']}")
                    print(f"   💰 P&L: ₹{result['pnl']:+,.0f} ({result['return_pct']:+.1f}%)")
                    
                    # Show AI exit analysis
                    ai_details = ai_decision['ai_details']
                    print(f"   🎯 Exit Signal: {ai_details['rl_signal']:+.3f}")
                    print(f"   💰 Cash: ₹{result['cash_balance']:,.0f}")
                    print()
            else:
                print(f"🤖 RL HOLDS {symbol} @ ₹{new_prices[symbol]:.2f}")
                print(f"   📊 {ai_decision['reason'][:60]}...")
                print()
    
    print(f"✅ Rebalancing Complete: {trades_executed} trades executed")
    print()
    
    # Generate comprehensive reports
    output_dir = Path("rl_strategy_output")
    output_dir.mkdir(exist_ok=True)
    
    portfolio_summary = tracker.get_portfolio_summary(new_prices)
    
    # Save trades CSV
    if tracker.completed_trades:
        trades_df = pd.DataFrame(tracker.completed_trades)
        csv_file = output_dir / "rl_strategy_trades.csv"
        trades_df.to_csv(csv_file, index=False)
        print(f"📄 RL Strategy trades saved to: {csv_file}")
    
    # Save comprehensive JSON report
    report_data = {
        'portfolio_summary': portfolio_summary,
        'completed_trades': tracker.completed_trades,
        'ai_model_info': {
            'model_type': 'RL_Strategy_Agent_Phase4',
            'features_used': ['RSI', 'MACD', 'PCR', 'Momentum_Confluence', 'Market_Regime', 'Options_Flow'],
            'decision_logic': 'Enhanced rule-based with Phase 4 signals',
            'time_constraints': 'Intraday position sizing and risk management'
        },
        'current_positions': {
            symbol: {
                'quantity': pos['quantity'],
                'entry_price': pos['avg_entry_price'],
                'current_price': new_prices.get(symbol, pos['avg_entry_price']),
                'market_value': pos['quantity'] * new_prices.get(symbol, pos['avg_entry_price']),
                'unrealized_pnl': pos['quantity'] * (new_prices.get(symbol, pos['avg_entry_price']) - pos['avg_entry_price']),
                'days_held': (datetime.now() - pos['entry_time']).days,
                'entry_ai_analysis': pos.get('ai_details', {})
            }
            for symbol, pos in tracker.positions.items()
        },
        'session_info': {
            'initial_capital': initial_capital,
            'generated_at': datetime.now().isoformat(),
            'stocks_analyzed': list(stocks.keys()),
            'positions_built': constructed_positions,
            'trades_executed': trades_executed
        }
    }
    
    json_file = output_dir / "rl_strategy_portfolio_report.json"
    with open(json_file, 'w') as f:
        json.dump(report_data, f, indent=2, default=str)
    print(f"📄 Comprehensive report saved to: {json_file}")
    
    # Display final results
    print("\n" + "=" * 55)
    print("📋 RL STRATEGY PORTFOLIO PERFORMANCE")
    print("=" * 55)
    
    print(f"💰 Initial Capital:      ₹{portfolio_summary['initial_capital']:,.0f}")
    print(f"💰 Portfolio Value:      ₹{portfolio_summary['portfolio_value']:,.0f}")
    print(f"📊 Total Return:         ₹{portfolio_summary['total_pnl']:+,.0f} ({portfolio_summary['total_return_pct']:+.2f}%)")
    print(f"🏦 Cash Balance:         ₹{portfolio_summary['current_cash']:,.0f}")
    print(f"📈 Market Value:         ₹{portfolio_summary['market_value']:,.0f}")
    print(f"💹 Realized P&L:         ₹{portfolio_summary['realized_pnl']:+,.0f}")
    print(f"📊 Unrealized P&L:       ₹{portfolio_summary['unrealized_pnl']:+,.0f}")
    
    if tracker.completed_trades:
        print(f"\n🤖 RL STRATEGY PERFORMANCE:")
        print(f"   Total Trades:         {len(tracker.completed_trades)}")
        
        winning_trades = [t for t in tracker.completed_trades if t['net_pnl'] > 0]
        win_rate = (len(winning_trades) / len(tracker.completed_trades)) * 100
        
        print(f"   Winning Trades:       {len(winning_trades)}")
        print(f"   Win Rate:             {win_rate:.1f}%")
        print(f"   Average P&L:          ₹{portfolio_summary['realized_pnl'] / len(tracker.completed_trades):+,.0f}")
        
        # AI-specific metrics
        avg_signal_strength = np.mean([t.get('ai_signal_strength', 0) for t in tracker.completed_trades])
        avg_confidence = np.mean([t.get('ai_confidence', 0) for t in tracker.completed_trades])
        
        print(f"   Avg Signal Strength:  {avg_signal_strength:+.3f}")
        print(f"   Avg AI Confidence:    {avg_confidence:.3f}")
        
        best_trade = max(tracker.completed_trades, key=lambda x: x['net_pnl'])
        worst_trade = min(tracker.completed_trades, key=lambda x: x['net_pnl'])
        
        print(f"   Best Trade:           {best_trade['symbol']} (₹{best_trade['net_pnl']:+,.0f})")
        print(f"   Worst Trade:          {worst_trade['symbol']} (₹{worst_trade['net_pnl']:+,.0f})")
        
        print(f"\n📊 INDIVIDUAL TRADE RESULTS:")
        for trade in tracker.completed_trades:
            signal_str = f"AI:{trade.get('ai_signal_strength', 0):+.2f}" if 'ai_signal_strength' in trade else ""
            print(f"   {trade['symbol']:10} | "
                  f"₹{trade['entry_price']:6.0f} → ₹{trade['exit_price']:6.0f} | "
                  f"₹{trade['net_pnl']:+8,.0f} ({trade['return_percent']:+5.1f}%) | "
                  f"{trade['holding_days']} days {signal_str}")
    
    # Current holdings
    if tracker.positions:
        print(f"\n📊 CURRENT HOLDINGS:")
        for symbol, pos in tracker.positions.items():
            current_price = new_prices.get(symbol, pos['avg_entry_price'])
            market_value = pos['quantity'] * current_price
            unrealized_pnl = market_value - pos['total_cost']
            unrealized_pct = (current_price / pos['avg_entry_price'] - 1) * 100
            
            print(f"   {symbol:10} | {pos['quantity']:3d} shares | "
                  f"₹{pos['avg_entry_price']:6.0f} → ₹{current_price:6.0f} | "
                  f"₹{unrealized_pnl:+8,.0f} ({unrealized_pct:+5.1f}%)")
    
    print("=" * 55)
    
    return report_data


if __name__ == "__main__":
    run_rl_strategy_portfolio_test()