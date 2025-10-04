"""
Pre-Market Analysis Module - PRODUCTION
Analyze overnight global cues before market open (9:00-9:15 AM)
Generate intraday bias and opening strategy
"""

import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class PreMarketAnalyzer:
    """
    Pre-market analysis (9:00-9:15 AM)
    - Fetch overnight global cues
    - Analyze NSE pre-open session
    - Generate intraday bias
    - Set opening range strategy
    """
    
    def __init__(self, config: Dict[str, any]):
        self.config = config
        self.cache = {}
        self.last_analysis = None
        
        logger.info("PreMarketAnalyzer initialized")
    
    def run_premarket_analysis(self) -> Dict[str, any]:
        """
        Main pre-market analysis workflow
        Run between 9:00-9:15 AM
        
        Returns:
            Complete pre-market analysis with bias and strategy
        """
        logger.info("="*80)
        logger.info("PRE-MARKET ANALYSIS")
        logger.info("="*80)
        
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'global_cues': {},
            'preopen_data': {},
            'sentiment': {},
            'bias': 'NEUTRAL',
            'opening_strategy': 'WAIT',
            'confidence': 0.0
        }
        
        try:
            # 1. Fetch overnight global markets
            logger.info("\n1️⃣ Fetching global market cues...")
            analysis['global_cues'] = self._fetch_global_cues()
            
            # 2. Analyze pre-open session (9:00-9:08 AM)
            logger.info("\n2️⃣ Analyzing NSE pre-open session...")
            analysis['preopen_data'] = self._analyze_preopen_session()
            
            # 3. Fetch overnight news sentiment
            logger.info("\n3️⃣ Analyzing overnight news...")
            analysis['sentiment'] = self._analyze_overnight_sentiment()
            
            # 4. Generate intraday bias
            logger.info("\n4️⃣ Generating intraday bias...")
            bias_result = self._generate_intraday_bias(
                analysis['global_cues'],
                analysis['preopen_data'],
                analysis['sentiment']
            )
            analysis['bias'] = bias_result['bias']
            analysis['confidence'] = bias_result['confidence']
            
            # 5. Set opening strategy
            logger.info("\n5️⃣ Setting opening range strategy...")
            analysis['opening_strategy'] = self._set_opening_strategy(
                analysis['bias'],
                analysis['confidence'],
                analysis['global_cues']
            )
            
            # Cache for later use
            self.last_analysis = analysis
            
            # Log summary
            self._log_premarket_summary(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Pre-market analysis failed: {e}")
            return analysis
    
    def _fetch_global_cues(self) -> Dict[str, any]:
        """
        Fetch overnight global market data
        - US market close (S&P, Nasdaq, Dow)
        - Asian markets (Nikkei, Hang Seng)
        - SGX Nifty
        - Commodities (Crude, Gold)
        - Forex (Dollar Index, USD/INR)
        """
        global_cues = {}
        
        # === US MARKETS (Previous close) ===
        try:
            us_markets = self._fetch_us_markets()
            global_cues['us_markets'] = us_markets
        except Exception as e:
            logger.error(f"Failed to fetch US markets: {e}")
            global_cues['us_markets'] = {'sp500': 0, 'nasdaq': 0, 'dow': 0}
        
        # === ASIAN MARKETS (Current) ===
        try:
            asian_markets = self._fetch_asian_markets()
            global_cues['asian_markets'] = asian_markets
        except Exception as e:
            logger.error(f"Failed to fetch Asian markets: {e}")
            global_cues['asian_markets'] = {'nikkei': 0, 'hang_seng': 0}
        
        # === SGX NIFTY (Real-time) ===
        try:
            sgx_nifty = self._fetch_sgx_nifty()
            global_cues['sgx_nifty'] = sgx_nifty
        except Exception as e:
            logger.error(f"Failed to fetch SGX Nifty: {e}")
            global_cues['sgx_nifty'] = {'price': 0, 'change_pct': 0}
        
        # === COMMODITIES ===
        try:
            commodities = self._fetch_commodities()
            global_cues['commodities'] = commodities
        except Exception as e:
            logger.error(f"Failed to fetch commodities: {e}")
            global_cues['commodities'] = {'crude_oil': 0, 'gold': 0}
        
        # === FOREX ===
        try:
            forex = self._fetch_forex()
            global_cues['forex'] = forex
        except Exception as e:
            logger.error(f"Failed to fetch forex: {e}")
            global_cues['forex'] = {'dollar_index': 0, 'usdinr': 0}
        
        return global_cues
    
    def _fetch_us_markets(self) -> Dict[str, float]:
        """Fetch US market close data"""
        # TODO: Use real API (Yahoo Finance, Alpha Vantage, etc.)
        # For now, placeholder implementation
        
        # Example with Yahoo Finance (requires yfinance package)
        # import yfinance as yf
        # sp500 = yf.Ticker("^GSPC")
        # data = sp500.history(period="1d")
        # change_pct = ((data['Close'][-1] - data['Close'][-2]) / data['Close'][-2]) * 100
        
        # Placeholder
        us_markets = {
            'sp500_change_pct': 0.5,  # +0.5%
            'nasdaq_change_pct': 0.7,
            'dow_change_pct': 0.3,
            'sentiment': 'bullish' if 0.5 > 0 else 'bearish'
        }
        
        logger.info(f"  US Markets: S&P {us_markets['sp500_change_pct']:+.2f}%, "
                   f"Nasdaq {us_markets['nasdaq_change_pct']:+.2f}%, "
                   f"Dow {us_markets['dow_change_pct']:+.2f}%")
        
        return us_markets
    
    def _fetch_asian_markets(self) -> Dict[str, float]:
        """Fetch Asian market data"""
        # TODO: Real API integration
        
        asian_markets = {
            'nikkei_change_pct': 0.3,
            'hang_seng_change_pct': -0.2,
            'sentiment': 'mixed'
        }
        
        logger.info(f"  Asian Markets: Nikkei {asian_markets['nikkei_change_pct']:+.2f}%, "
                   f"Hang Seng {asian_markets['hang_seng_change_pct']:+.2f}%")
        
        return asian_markets
    
    def _fetch_sgx_nifty(self) -> Dict[str, float]:
        """
        Fetch SGX Nifty (Singapore Nifty futures)
        Leading indicator for Indian markets
        """
        # TODO: Real-time SGX Nifty data
        # SGX Nifty often indicates gap-up/gap-down in NSE
        
        sgx_nifty = {
            'price': 22050,
            'change_pct': 0.4,
            'vs_nse_spot': 20,  # Premium/discount to NSE spot
            'sentiment': 'bullish'
        }
        
        logger.info(f"  SGX Nifty: ₹{sgx_nifty['price']:.0f} ({sgx_nifty['change_pct']:+.2f}%)")
        
        return sgx_nifty
    
    def _fetch_commodities(self) -> Dict[str, float]:
        """Fetch commodity prices"""
        # TODO: Real commodity data
        
        commodities = {
            'crude_oil_change_pct': -1.2,
            'gold_change_pct': 0.5,
            'sentiment': 'mixed'
        }
        
        logger.info(f"  Commodities: Crude {commodities['crude_oil_change_pct']:+.2f}%, "
                   f"Gold {commodities['gold_change_pct']:+.2f}%")
        
        return commodities
    
    def _fetch_forex(self) -> Dict[str, float]:
        """Fetch forex data"""
        # TODO: Real forex data
        
        forex = {
            'dollar_index': 104.5,
            'dollar_index_change_pct': 0.2,
            'usdinr': 83.15,
            'usdinr_change_pct': 0.1,
            'sentiment': 'neutral'
        }
        
        logger.info(f"  Forex: DXY {forex['dollar_index']:.2f} ({forex['dollar_index_change_pct']:+.2f}%), "
                   f"USD/INR {forex['usdinr']:.2f}")
        
        return forex
    
    def _analyze_preopen_session(self) -> Dict[str, any]:
        """
        Analyze NSE pre-open session (9:00-9:08 AM)
        Indicators: Pre-open price, volume, order imbalance
        """
        # TODO: Real NSE pre-open data
        # NSE provides pre-open auction data at 9:08 AM
        
        preopen_data = {
            'preopen_price': 22030,
            'preopen_change_pct': 0.3,
            'order_imbalance': 'buy_heavy',  # More buy orders
            'institutional_activity': 'buying',
            'delivery_percentage': 65,  # % of delivery trades (higher = stronger)
            'sentiment': 'bullish'
        }
        
        logger.info(f"  Pre-open: ₹{preopen_data['preopen_price']:.0f} ({preopen_data['preopen_change_pct']:+.2f}%)")
        logger.info(f"  Order Imbalance: {preopen_data['order_imbalance']}")
        logger.info(f"  Institutional: {preopen_data['institutional_activity']}")
        
        return preopen_data
    
    def _analyze_overnight_sentiment(self) -> Dict[str, float]:
        """Analyze overnight news sentiment"""
        # TODO: Integrate with sentiment pipeline
        
        sentiment = {
            'overnight_sentiment': 0.3,  # Slightly bullish
            'breaking_news': False,
            'high_impact_count': 0
        }
        
        logger.info(f"  Overnight Sentiment: {sentiment['overnight_sentiment']:+.2f}")
        
        return sentiment
    
    def _generate_intraday_bias(
        self,
        global_cues: Dict,
        preopen_data: Dict,
        sentiment: Dict
    ) -> Dict[str, any]:
        """
        Generate intraday bias from all cues
        
        Returns:
            {'bias': 'BULLISH' | 'BEARISH' | 'NEUTRAL', 'confidence': float}
        """
        # Scoring system
        bullish_score = 0
        bearish_score = 0
        
        # === US MARKETS (Weight: 30%) ===
        us_avg_change = np.mean([
            global_cues['us_markets']['sp500_change_pct'],
            global_cues['us_markets']['nasdaq_change_pct'],
            global_cues['us_markets']['dow_change_pct']
        ])
        
        if us_avg_change > 0.5:
            bullish_score += 3
        elif us_avg_change < -0.5:
            bearish_score += 3
        else:
            bullish_score += 1
            bearish_score += 1
        
        # === SGX NIFTY (Weight: 40% - strongest indicator) ===
        sgx_change = global_cues['sgx_nifty']['change_pct']
        
        if sgx_change > 0.5:
            bullish_score += 4
        elif sgx_change < -0.5:
            bearish_score += 4
        else:
            bullish_score += 2
            bearish_score += 2
        
        # === PRE-OPEN SESSION (Weight: 20%) ===
        if preopen_data['order_imbalance'] == 'buy_heavy':
            bullish_score += 2
        elif preopen_data['order_imbalance'] == 'sell_heavy':
            bearish_score += 2
        
        # === SENTIMENT (Weight: 10%) ===
        sentiment_score = sentiment['overnight_sentiment']
        if sentiment_score > 0.3:
            bullish_score += 1
        elif sentiment_score < -0.3:
            bearish_score += 1
        
        # === DETERMINE BIAS ===
        total_score = bullish_score + bearish_score
        confidence = abs(bullish_score - bearish_score) / total_score if total_score > 0 else 0
        
        if bullish_score > bearish_score and confidence > 0.2:
            bias = 'BULLISH'
        elif bearish_score > bullish_score and confidence > 0.2:
            bias = 'BEARISH'
        else:
            bias = 'NEUTRAL'
        
        logger.info(f"  Bias: {bias} (Confidence: {confidence:.2f})")
        logger.info(f"  Score: Bullish={bullish_score}, Bearish={bearish_score}")
        
        return {'bias': bias, 'confidence': confidence}
    
    def _set_opening_strategy(
        self,
        bias: str,
        confidence: float,
        global_cues: Dict
    ) -> str:
        """
        Set opening range trading strategy
        
        Strategies:
        - IMMEDIATE_ENTRY: High confidence, enter at market open
        - WAIT_30MIN: Low confidence, wait for opening range (9:15-9:45)
        - FADE_OPENING: Contrarian play if extreme gap
        """
        # Check for extreme gap (>1% SGX move)
        sgx_change = abs(global_cues['sgx_nifty']['change_pct'])
        
        if sgx_change > 1.5:
            # Extreme gap - consider fading
            strategy = 'FADE_OPENING'
            logger.info(f"  Strategy: {strategy} (Extreme gap: {sgx_change:.2f}%)")
        
        elif confidence > 0.6:
            # High confidence - immediate entry
            strategy = 'IMMEDIATE_ENTRY'
            logger.info(f"  Strategy: {strategy} (High confidence: {confidence:.2f})")
        
        elif confidence > 0.3:
            # Medium confidence - wait for confirmation
            strategy = 'WAIT_15MIN'
            logger.info(f"  Strategy: {strategy} (Medium confidence)")
        
        else:
            # Low confidence - wait for opening range
            strategy = 'WAIT_30MIN'
            logger.info(f"  Strategy: {strategy} (Low confidence)")
        
        return strategy
    
    def _log_premarket_summary(self, analysis: Dict):
        """Log comprehensive pre-market summary"""
        logger.info("\n" + "="*80)
        logger.info("PRE-MARKET SUMMARY")
        logger.info("="*80)
        
        logger.info(f"\n🌍 GLOBAL CUES:")
        logger.info(f"  US Markets: {analysis['global_cues']['us_markets']['sentiment'].upper()}")
        logger.info(f"  SGX Nifty: ₹{analysis['global_cues']['sgx_nifty']['price']:.0f} ({analysis['global_cues']['sgx_nifty']['change_pct']:+.2f}%)")
        
        logger.info(f"\n📊 PRE-OPEN SESSION:")
        logger.info(f"  Price: ₹{analysis['preopen_data']['preopen_price']:.0f}")
        logger.info(f"  Order Imbalance: {analysis['preopen_data']['order_imbalance'].upper()}")
        
        logger.info(f"\n🎯 INTRADAY BIAS:")
        logger.info(f"  Bias: {analysis['bias']}")
        logger.info(f"  Confidence: {analysis['confidence']:.1%}")
        
        logger.info(f"\n⚡ OPENING STRATEGY:")
        logger.info(f"  Strategy: {analysis['opening_strategy']}")
        
        if analysis['opening_strategy'] == 'IMMEDIATE_ENTRY':
            logger.info(f"  → Enter at market open (9:15 AM)")
        elif analysis['opening_strategy'] == 'WAIT_15MIN':
            logger.info(f"  → Wait 15 mins for confirmation (9:30 AM)")
        elif analysis['opening_strategy'] == 'WAIT_30MIN':
            logger.info(f"  → Wait for opening range completion (9:45 AM)")
        elif analysis['opening_strategy'] == 'FADE_OPENING':
            logger.info(f"  → Fade the opening gap (contrarian)")
        
        logger.info("\n" + "="*80)
    
    def get_last_analysis(self) -> Optional[Dict]:
        """Get cached pre-market analysis"""
        return self.last_analysis
    
    def should_trade_at_open(self) -> bool:
        """
        Quick check: Should we trade immediately at open?
        Based on last pre-market analysis
        """
        if not self.last_analysis:
            return False
        
        strategy = self.last_analysis.get('opening_strategy', 'WAIT_30MIN')
        
        return strategy == 'IMMEDIATE_ENTRY'


# ==================== UTILITY FUNCTIONS ====================

def is_premarket_time() -> bool:
    """Check if current time is pre-market (9:00-9:15 AM)"""
    now = datetime.now()
    return (now.hour == 9 and 0 <= now.minute < 15)


def wait_until_premarket():
    """Wait until pre-market time"""
    while not is_premarket_time():
        logger.info("Waiting for pre-market time (9:00 AM)...")
        import time
        time.sleep(60)  # Check every minute
    
    logger.info("Pre-market time reached!")


# ==================== MAIN ====================

if __name__ == "__main__":
    # Test pre-market analysis
    config = {
        'use_sgx_nifty': True,
        'use_global_cues': True
    }
    
    analyzer = PreMarketAnalyzer(config)
    analysis = analyzer.run_premarket_analysis()
    
    print(f"\nBias: {analysis['bias']}")
    print(f"Strategy: {analysis['opening_strategy']}")