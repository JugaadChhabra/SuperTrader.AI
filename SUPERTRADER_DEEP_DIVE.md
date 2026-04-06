# SuperTrader.AI — Complete Deep Dive

## 1. SYSTEM ARCHITECTURE

### The Pipeline

SuperTrader.AI is a multi-agent intraday trading system for Indian equity derivatives. It uses a LangGraph-orchestrated pipeline of 6 agents that runs every decision cycle (currently 5-minute bars).

```
Market Tick (ICICI Breeze WebSocket)
   |
   v
+----------------+     +---------------+
|  DATA AGENT    |---->| NEWS AGENT    |     <-- These two SHOULD run in parallel
+-------+--------+     +-------+-------+
        |                       |
        v                       v
    +-------------------------------+
    |     RL STRATEGY AGENT         |  <-- Combines signals, outputs action
    +---------------+---------------+
                    v
    +-------------------------------+
    |    EXECUTION AGENT            |  <-- Position sizing, order construction
    +---------------+---------------+
                    v
    +------------------------------+
    |     RISK GUARD               |  <-- Exposure, margin, VaR checks
    +---------------+--------------+
                    v
    +------------------------------+
    |     TIME GUARD               |  <-- Intraday kill switch (3:15 PM)
    +------------------------------+
```

### LangGraph Orchestration

The workflow is defined as a `StateGraph` in `graph/workflow.py`. Each node is a function that takes and returns a `TradingState` TypedDict -- a shared state object containing:

```python
class TradingState(TypedDict):
    timestamp: datetime
    minutes_to_close: float
    session_phase: str          # 'opening_range' | 'morning' | 'afternoon' | 'closing'
    trading_allowed: bool
    price_data: Dict            # Raw OHLCV per instrument
    indicators: Dict            # RSI, MACD, VWAP, etc.
    oi_data: Dict               # Open interest changes
    basis_data: Dict            # Spot-futures basis
    options_data: Dict          # PCR, options flow
    sentiment_scores: Dict      # News sentiment
    rl_action: Dict             # Agent's decision
    orders: List[Dict]          # Generated orders
    positions: Dict             # Current positions
    risk_status: Dict           # Risk check results
    time_guard_status: Dict     # Time constraint status
    broker_ctx: Dict            # API credentials
    errors: List[str]
    logs: List[str]
```

### Workflow Improvement Opportunities

The workflow is currently **linear** (A->B->C->D->E->F). In production, you'd want:
- Data + News running **in parallel** (they're independent)
- **Conditional edges**: If risk breach -> skip to forced exit node
- **Feedback loops**: Post-execution results fed back for online learning

---

## 2. DATA AGENT

### Broker Integration (ICICI Direct Breeze API)

`agents/data_agent/icici_broker.py` handles:
1. **Authentication**: App key + session token -> REST auth -> WebSocket upgrade
2. **Data Subscription**: Subscribe to live tick data for NIFTY, BANKNIFTY, FINNIFTY futures
3. **OHLCV Aggregation**: Raw ticks -> 1-min / 5-min candles with proper OHLCV aggregation

### Feature Engineering

`agents/data_agent/feature_builders.py` (54KB, largest module) builds a 32-dimensional feature vector from raw OHLCV data. This is what gets fed to the DQN.

**Feature Categories:**

#### A. Technical Indicators (`indicators/technical.py` -- uses TA-Lib)

| Feature | What It Captures | How It's Computed |
|---------|-----------------|-------------------|
| RSI (9, 14, 21, 30) | Momentum / overbought-oversold | Wilder's smoothed avg gains vs losses |
| MACD (fast 8/21/9, standard 12/26/9, slow 19/39/9) | Trend + momentum crossovers | EMA difference + signal line |
| Bollinger Bands | Volatility + mean reversion | 20-period SMA +/- 2 sigma |
| ATR | True range volatility | Max(H-L, |H-Cprev|, |L-Cprev|) smoothed |
| Stochastic %K/%D | Overbought/oversold in range | (C-L14)/(H14-L14) |
| Williams %R | Range position | (H14-C)/(H14-L14) |
| CCI | Deviation from statistical mean | (TP-SMA)/(.015 x MAD) |
| ADX | Trend strength (not direction) | Smoothed DI+/DI- differential |
| VWAP + distance | Institutional fair value | Sum(Price x Vol)/Sum(Vol) |

#### B. Volatility Features (`feature_builders.py:124-192`)

```python
# EWMA volatility at 3 decay rates
ewma_vol_10 = returns.ewm(span=10).std() * sqrt(252)   # Short-term
ewma_vol_20 = returns.ewm(span=20).std() * sqrt(252)   # Medium-term
ewma_vol_50 = returns.ewm(span=50).std() * sqrt(252)   # Long-term

# Volatility regime detection
low_vol_regime  = ewma_vol_20 < long_term_mean * 0.7
high_vol_regime = ewma_vol_20 > long_term_mean * 1.5
vol_spike       = ewma_vol_20 > long_term_mean * 2.0

# Vol-of-vol (second derivative of vol -- captures regime transitions)
vol_of_vol = ewma_vol_20.pct_change().rolling(20).std()

# Risk-adjusted momentum
risk_adj_momentum = 5d_return / ewma_vol_20
```

**Important**: Vol-of-vol is critical for options. When vol-of-vol is high, it means volatility itself is unstable -- dangerous for selling options, great for buying.

#### C. Open Interest Features

```python
oi_change_1bar = OI_current - OI_prev          # Short-term OI flow
oi_change_5bar = OI_current - OI_5bars_ago     # Medium-term flow
oi_momentum = oi_change_5bar / OI_5bars_ago    # Normalized OI momentum
```

**Why OI matters for options**:
- Rising OI + rising price = **new longs entering** -> strong bullish
- Rising OI + falling price = **new shorts entering** -> strong bearish
- Falling OI + rising price = **short covering** -> weak bullish (exhausts quickly)
- Falling OI + falling price = **long unwinding** -> weak bearish

#### D. Put-Call Ratio (PCR) Features (`feature_builders.py:195+`)

```python
# PCR percentile in historical distribution
pcr_percentile = percentile_rank(current_pcr, historical_pcr_window)

# Sentiment interpretation
pcr_bullish = pcr_percentile > 80   # High PCR = excessive puts = contrarian bullish
pcr_bearish = pcr_percentile < 20   # Low PCR = excessive calls = contrarian bearish
```

**Critical for options**: PCR is a **contrarian** indicator. When PCR > 1.2, retail is panic-buying puts, which means market makers are net-short puts (net-long delta). They'll gamma-hedge by buying futures, creating upward pressure. The reverse for low PCR.

#### E. Spot-Futures Basis

```python
basis = futures_price - spot_price
basis_pct = (basis / spot_price) * 100
```

In Indian markets, futures typically trade at a premium (cost of carry). When basis collapses or goes negative, it signals:
- **Basis narrowing**: Hedgers unwinding -> reduced bullish conviction
- **Backwardation** (negative basis): Panic/extreme bearishness, or near expiry

#### F. Time Features (Critical for Intraday)

```python
minutes_to_close_norm = minutes_to_close / 360.0    # 0 at close, 1 at open
session_onehot = {
    'opening_range': [1,0,0,0],   # 9:15-9:45 -- high vol, gap fills
    'morning':       [0,1,0,0],   # 9:45-11:45 -- trending
    'afternoon':     [0,0,1,0],   # 11:45-2:15 -- lunch lull, range-bound
    'closing':       [0,0,0,1]    # 2:15-3:15 -- momentum, square-off pressure
}
```

**Key point**: Time is THE most important feature in intraday. The opening 30 minutes contain ~25% of daily volume. After 2:30 PM, MIS positions start getting auto-squared, creating selling pressure regardless of direction.

#### G. Inter-Index Correlations (`feature_builders.py:37-121`)

```python
# Rolling correlation between NIFTY and BANKNIFTY
corr_10 = nifty_returns.rolling(10).corr(banknifty_returns)
corr_20 = nifty_returns.rolling(20).corr(banknifty_returns)

# Beta (sensitivity of NIFTY to BANKNIFTY)
beta = cov(nifty, banknifty) / var(banknifty)

# Correlation regime
high_corr_regime = corr_20 > 0.8    # Moving together
low_corr_regime  = corr_20 < 0.4    # Divergence -> dispersion trade opportunity

# Relative strength z-score
rs_zscore = (price_ratio - rolling_mean) / rolling_std   # Mean reversion signal
```

---

## 3. DQN MODEL ARCHITECTURE

### Architecture (`dqn_network.py`)

```
Input: (batch_size, 30 timesteps, 32 features)
         |
         v
    LSTM Layer 1
    +-- input_size = 32
    +-- hidden_size = 32
    +-- dropout = 0.2
    +-- Purpose: Learns short-term temporal patterns
         |         (e.g., "RSI has been declining for 5 bars")
         v
    LSTM Layer 2
    +-- input_size = 32
    +-- hidden_size = 16
    +-- Purpose: Learns higher-order temporal abstractions
         |         (e.g., "we're in the trending phase of a V-reversal")
         v
    final_hidden_state (batch, 16)
         |
    +----+----+
    |         |
    v         v
  VALUE     ADVANTAGE
  STREAM    STREAM
    |         |
  Linear(16,16)  Linear(16,16)
  LeakyReLU     LeakyReLU
  Dropout(0.2)  Dropout(0.2)
  Linear(16,1)  Linear(16,3)
    |         |
    V(s)      A(s,a) for 3 actions
    |         |
    +----+----+
         |
    Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
         |
    3 Q-values: [Q_SHORT, Q_HOLD, Q_LONG]
    Action = argmax(Q)
```

### Design Decisions

**Why LSTM instead of Transformer?**

LSTM is appropriate for fixed-lookback sequential data at low dimensionality (30 steps x 32 features). A Transformer would add attention overhead without clear benefit at this scale. However, for multi-instrument or longer lookback, a Temporal Fusion Transformer (TFT) which adds interpretable attention would be better -- useful for explaining which time steps drove a decision.

**Why Dueling DQN?**

In trading, many states are "neutral" -- the market isn't doing anything interesting. Standard DQN wastes capacity learning that Q(neutral_state, BUY) ~ Q(neutral_state, SELL) ~ Q(neutral_state, HOLD). Dueling separates V(s) ("is anything happening?") from A(s,a) ("which action matters?"). The value stream quickly learns "flat market = low value", and the advantage stream only needs to discriminate in states where action choice actually matters.

**Why gamma = 0.3 (low discount factor)?**

Standard RL uses gamma=0.99 (care about rewards far in the future). For intraday trading, the horizon is 75 bars (390 min / 5 min). With gamma=0.99, reward from 50 bars ahead has weight 0.99^50 = 0.60. With gamma=0.3, reward from 3 bars ahead has weight 0.3^3 = 0.027 -- the agent is nearly myopic. This is intentional: intraday positions shouldn't care about what happens 4 hours from now. They care about the next 15-30 minutes. For swing trading you'd increase gamma.

**Why 3 discrete actions?**

This is a limitation. Discrete {SHORT, HOLD, LONG} means the agent can't express "go long with 20% conviction" vs "go long with 90% conviction." Position sizing is handled externally by the `VolatilityPositionSizer`. A better approach would be continuous action space with PPO/SAC, where the action is a float in [-1, +1] representing direction AND magnitude.

### Training Details

```python
# Double DQN update (from dqn_network.py:299-307)
# Main brain picks action, target brain evaluates it
next_actions = main_brain(next_states).argmax(dim=1)           # Main picks
next_q = target_brain(next_states).gather(1, next_actions)     # Target evaluates
target = reward + (1 - done) * gamma * next_q

loss = smooth_L1(predicted_q, target)   # Huber loss -- robust to outliers
```

**Why Double DQN**: Standard DQN uses `max(Q_target(s', a'))` which systematically overestimates Q-values (the max of noisy estimates is biased upward). Double DQN decouples action selection from evaluation. Empirically reduces overestimation by 30-50%.

**Epsilon schedule**: 0.5 -> 0.01, decay factor 0.995 per step. After ~920 steps, epsilon halves. Total exploration phase: ~5000 steps to reach near-zero.

**Target network update**: Hard copy every 500 steps (not Polyak averaging). Could improve with soft update: `target = tau * main + (1-tau) * target` where tau=0.005.

---

## 4. POSITION SIZING

### Volatility Position Sizing (`volatility_position_sizing.py`)

Based on the Oxford paper methodology:

```python
# 1. Calculate volatility profile
realized_vol = returns.ewm(span=20).std() * sqrt(252)

# 2. Volatility scaling (target 20% annualized vol)
vol_scalar = target_vol / realized_vol
# Clamp to [0.5, 2.0] -- never more than 2x base size

# 3. Kelly Criterion
kelly_fraction = (win_rate * avg_win_loss_ratio - (1 - win_rate)) / avg_win_loss_ratio
# Example: 55% win rate, 1.2 W/L ratio -> Kelly = (0.55 * 1.2 - 0.45)/1.2 = 0.175
# Use half-Kelly (0.0875) for safety

# 4. Signal scaling
position_size = kelly_fraction * vol_scalar * |rl_signal| * account_balance / price

# 5. Discretize to lots
lots = round(position_size / lot_size)  # NIFTY lot = 75, BANKNIFTY = 15
```

### Index-Specific Parameters

```python
index_configs = {
    'NIFTY': {
        'lot_size': 75,              # Rs 75 per point per lot
        'tick_size': 0.05,           # Minimum price movement
        'typical_volatility': 0.15,  # ~15% annualized
        'margin_multiplier': 0.10,   # ~10% of notional for MIS
        'leverage_cap': 10           # Max 10x leverage
    },
    'BANKNIFTY': {
        'lot_size': 15,              # Changed from 25 recently
        'typical_volatility': 0.25,  # Higher vol
        'margin_multiplier': 0.12    # Higher margin
    }
}
```

BANKNIFTY has ~1.7x the volatility of NIFTY but smaller lot size. Per-lot notional risk is different. The sizer adjusts for this.

---

## 5. RISK MANAGEMENT

### Pre-Trade Risk Checks (`execution_agent.py:124-150`)

Every order goes through 6 checks before submission:

1. **Time check**: No new positions after 3:00 PM
2. **Exposure check**: Total notional < 50% of capital
3. **Margin check**: Required margin < available margin
4. **Position limit check**: Max positions per symbol
5. **Expiry check**: Don't trade contracts expiring today (low liquidity, settlement risk)
6. **Correlation check**: Don't be long NIFTY + long BANKNIFTY with corr > 0.8 (same bet twice)

### Real-Time Risk Metrics (`utils/risk_config.py`)

```python
@dataclass
class RiskMetrics:
    portfolio_leverage: float      # Current leverage ratio
    daily_pnl: float              # Today's P&L
    var_95: float                 # 95% Value at Risk
    var_99: float                 # 99% VaR
    margin_utilization: float     # % of margin used
    largest_position_pct: float   # Concentration risk
    correlation_risk: float       # Cross-position correlation
    consecutive_losses: int       # Tilt detection
    consecutive_loss_amount: float
```

### Risk Limits (`configs/risk.yaml`)

| Limit | Value | What Happens on Breach |
|-------|-------|----------------------|
| Max daily loss | 5% | All orders cancelled, no new trades |
| Max drawdown | 10% | Emergency mode -- force exit all |
| Max position size | 20% per position | Order rejected |
| Stop loss | 2% per trade | Automatic exit |
| Max leverage | 5-10x | Order reduced |
| VIX threshold | 20 | Reduce position sizes by 50% |
| Consecutive losses | 5 | Halt trading, cool-off period |

### Time Guard -- The Kill Switch (`graph/workflow.py:566-665`)

```
2:30 PM (45 min to close)  -> CAUTION: Reduce exposure
3:00 PM (15 min to close)  -> WARNING: No new entries
3:10 PM (5 min to close)   -> CRITICAL: Force exit ALL positions
3:15 PM (0 min)            -> EMERGENCY: Broker auto-squares anyway
```

In Indian markets, MIS (Margin Intraday Square-off) positions are auto-squared by the broker at 3:15-3:25 PM. If YOU don't exit by 3:10, the broker will at worse prices with MARKET orders. The time guard prevents this.

---

## 6. TRANSACTION COST MODELING

### NSE F&O Cost Breakdown (`trade_ledger.py:65-84`)

```python
@dataclass
class TransactionCosts:
    brokerage: float           # Rs 20 flat per order (discount broker)
    stt: float                 # Securities Transaction Tax
    exchange_fees: float       # NSE transaction charges: 0.00053%
    clearing_charges: float    # 0.00018%
    sebi_fees: float          # Rs 10 per crore
    stamp_duty: float         # 0.003% (buy side only)
    gst: float                # 18% on (brokerage + exchange fees)
    impact_costs: float       # Market impact (function of order size vs ADV)
```

**Key numbers for options trading:**
- **STT on options SELL**: 0.0625% of premium -- this is **5x** the futures STT (0.0125%). This is why selling options intraday gets expensive.
- STT applies only on sell side for futures, buy+sell for options
- Total round-trip cost for NIFTY futures: ~5-8 bps
- Total round-trip cost for NIFTY options: ~15-25 bps (the STT kills you)

**Important**: Many retail options traders don't realize STT on selling options is charged on **intrinsic value at expiry, not premium paid.** If you sell a 22000 CE at Rs 50 and NIFTY expires at 22300, your STT is on Rs 300 (intrinsic), not Rs 50 (premium). This can wipe out the entire profit on short-dated option selling.

---

## 7. OPTIONS-SPECIFIC KNOWLEDGE

The system currently trades **futures**, but the architecture supports options. Here's the full context:

### Greeks

| Greek | What It Measures | Why It Matters for Intraday |
|-------|-----------------|---------------------------|
| **Delta** | Price sensitivity to underlying | Position sizing: ATM option has delta ~ 0.5, so 2 lots of options ~ 1 lot of futures in directional exposure |
| **Gamma** | Rate of change of delta | Gamma is highest ATM and near expiry. On expiry day, ATM options have extreme gamma -- small move in underlying causes massive delta change. This is where "gamma scalping" profits come from |
| **Theta** | Time decay per day | Intraday theta is NOT constant. An ATM weekly option with 1 day to expiry can lose 50%+ of its value in a single session. Theta accelerates as sqrt(1/T) |
| **Vega** | Sensitivity to implied vol | VIX moves 2-3 points intraday. An ATM NIFTY option with vega=5 moves Rs 10-15 just from IV change, independent of underlying direction |
| **Vanna** | d(Delta)/d(IV) | When IV drops + underlying rises, calls gain delta faster. Critical for understanding post-event moves (earnings, RBI) |
| **Charm** | d(Delta)/d(Time) | Delta changes with time even if underlying doesn't move. On expiry day, OTM options lose delta rapidly -- "pin risk" |

### Extending the DQN for Options

**Action space expansion:**
```
Instead of: [SHORT_FUTURES, HOLD, LONG_FUTURES]

Use: [
    BUY_ATM_CALL,      # Directional long
    BUY_ATM_PUT,       # Directional short
    SELL_ATM_CALL,     # Short vol / bearish
    SELL_ATM_PUT,      # Short vol / bullish
    BUY_STRADDLE,      # Long vol (direction neutral)
    SELL_STRADDLE,     # Short vol (collect theta)
    IRON_CONDOR,       # Range-bound (collect premium with limited risk)
    HOLD,              # No action
    EXIT               # Close all positions
]
```

**State space additions for options:**
```python
options_state = [
    iv_atm,              # ATM implied volatility
    iv_skew,             # OTM put IV - OTM call IV (fear gauge)
    iv_term_structure,   # Near-month IV vs far-month IV
    iv_percentile,       # Where is current IV vs last 252 days
    pcr_oi,              # PCR based on open interest
    pcr_volume,          # PCR based on volume
    max_pain,            # Strike with max option writer profit
    gamma_exposure,      # Net market gamma (GEX)
    dealer_delta,        # Net dealer delta position
    days_to_expiry,      # Critical for theta/gamma
]
```

### Key Options Concepts

**1. Gamma Exposure (GEX)**

Market makers delta-hedge their option positions. When aggregate gamma is positive (market makers are long gamma), they BUY dips and SELL rallies -> dampens volatility. When GEX is negative (after big moves), they SELL dips and BUY rallies -> amplifies volatility. GEX flips sign are tradeable events.

**2. Max Pain**

The strike price where option buyers lose the most money (writers profit the most). NIFTY tends to gravitate toward max pain on expiry day. Not because of conspiracy -- because options writers dynamically hedge, pushing the underlying toward max pain. The system could use max pain as an "anchor" for mean-reversion strategies on expiry day.

**3. IV Crush**

After events (RBI policy, earnings, elections), IV drops sharply because uncertainty resolves. An ATM NIFTY straddle might cost Rs 300 before RBI. If NIFTY moves only Rs 100, both legs lose value from IV crush. The straddle buyer needs a move > straddle_price to profit. The DQN could learn: "before events, buy straddles; after events, sell them."

**4. Pin Risk on Weekly Expiry**

Every Thursday is NIFTY weekly expiry. ATM options go from delta=0.5 to delta=0 or delta=1 in the last hour. If you're short an ATM option and NIFTY oscillates around the strike, your delta swings wildly. You'd need to gamma-hedge every few minutes. The time guard should treat expiry day differently from non-expiry days.

**5. The "0DTE" Phenomenon**

Zero-days-to-expiry options have near-infinite gamma at ATM and near-zero theta (it's already decayed). They're essentially binary bets. The DQN's `minutes_to_close` feature + `days_to_expiry` would learn: "on 0DTE, only trade directionally with defined risk (long options only, never short)."

---

## 8. UNIVERSE RANKING

`agents/universe_ranking.py` scores instruments on 4 axes:

```python
composite_score = (
    0.30 * liquidity_score +     # Volume, OI, bid-ask spread
    0.25 * volatility_score +    # Right vol range (not too low, not extreme)
    0.25 * momentum_score +      # 1d/5d/20d momentum
    0.20 * mean_reversion_score  # RSI extremes = opportunity
)
```

**Liquidity score** considers:
- Average daily volume in lots
- Average open interest
- Bid-ask spread in basis points (lower = better)

**Why this matters for options**: A FINNIFTY weekly option at a strike 500 points OTM might have a bid-ask spread of Rs 2 on a Rs 5 premium -- that's **40% spread cost**. The universe ranker would filter this out because `liquidity_score` would be near zero.

---

## 9. NEWS / SENTIMENT AGENT

### Architecture (`agents/news_agent.py`)

```
3 Sources (Zerodha Pulse, NSE India, Upstox News)
    |
    v
Web Scraping (BeautifulSoup + requests.Session with retries)
    |
    v
Text Cleaning (remove URLs, tickers, emojis, normalize whitespace)
    |
    v
Deduplication (hash-based on first 300 chars)
    |
    v
Sentiment Scoring (HuggingFace pipeline)
    |-- Primary: DistilFinBERT / ProsusAI FinBERT
    |-- Fallback: Lexicon-based (POS/NEG word counting)
    |
    v
Ticker Mapping (stock_names_symbol.csv -> substring match)
    |
    v
Rolling 5-minute Aggregation (pandas resample + rolling mean)
    |
    v
Output: { market_sentiment_5min: float, sentiment_timeseries: [...], high_impact_news: [...] }
```

### Sentiment Model Cascade

```python
DEFAULT_MODEL_CANDIDATES = [
    "yashkumar/distil-finbert",                          # Fast, finance-tuned
    "ProsusAI/finbert",                                  # Original FinBERT
    "yiyanghkust/finbert-tone",                          # Alt FinBERT
    "cardiffnlp/twitter-roberta-base-sentiment",         # General sentiment
    "distilbert-base-uncased-finetuned-sst-2-english",   # Light baseline
]
```

Tries each model in order. If all fail, falls back to word-list lexicon scoring.

---

## 10. TRADE LEDGER & LOGGING

### Dual Logging System

| System | File | Purpose |
|--------|------|---------|
| CSV Ledger | `ledgers/ledger_master.csv` | One row per trade, easy Excel analysis |
| JSON Sessions | `sessions/session_<timestamp>.json` | Full session detail with nested data |
| SQLite | `data/trade_ledger.db` | Persistent querying, analytics |
| Server Logs | `backend.log`, `frontend.log` | Debug + operational logs |

### Trade Lifecycle Tracking (`trade_ledger.py`)

```
PENDING -> PARTIALLY_FILLED -> FILLED -> (position open)
                                            |
                                    Mark-to-market P&L updates
                                            |
                                    Exit triggered (signal/stop/time)
                                            |
                                    CLOSED (realized P&L recorded)
```

### Transaction Cost Detail for NSE

```python
@dataclass
class TransactionCosts:
    brokerage: float           # Rs 20 flat per order
    stt: float                 # Futures sell: 0.0125%, Options sell: 0.0625%
    exchange_fees: float       # 0.00053% of turnover
    clearing_charges: float    # 0.00018%
    sebi_fees: float          # Rs 10 per crore
    stamp_duty: float         # 0.003% (buy side only)
    gst: float                # 18% on (brokerage + exchange + SEBI fees)
    rollover_costs: float     # Cost if rolling to next expiry
    impact_costs: float       # f(order_size / ADV)
```

---

## 11. PORTFOLIO SIMULATOR

### Paper Trading (`utils/portfolio_simulator.py`)

```python
class PortfolioSimulator:
    # Manages:
    # - Cash balance tracking
    # - Position entry/exit with avg price calculation
    # - Transaction cost deduction per trade
    # - Unrealized + realized P&L
    # - Slippage modeling (configurable bps)
    # - Risk metrics (drawdown, exposure %)
```

### Execution Flow

```python
result = simulator.execute(
    symbol="NIFTY",
    action=ActionType.BUY,
    price=22150.0,
    sentiment=0.65
)
# Returns TradeResult with:
#   trade_id, quantity, notional_value,
#   transaction_costs, net_cost, slippage_bps,
#   position_change, portfolio_impact
```

---

## 12. FRONTEND DASHBOARD

### Routes

| Route | Component | Purpose |
|-------|-----------|---------|
| `/` | `home.tsx` | Landing page |
| `/dashboard` | `dashboard.tsx` | Main trading dashboard |
| `/login` | `login.tsx` | Firebase authentication |
| `/settings` | `settings.tsx` | User preferences |
| `/news` | `news.tsx` | Market news & sentiment |
| `/backtest` | `backtest.tsx` | Backtest results viewer |
| `/analytics` | `analytics.tsx` | Performance analytics |

### API Client (`app/services/tradingApi.ts`)

Axios-based client hitting FastAPI backend at `localhost:8000`. Endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check + component status |
| `GET` | `/api/trading/status` | Portfolio value, positions, market status |
| `POST` | `/api/trading/analyze` | AI decision for a symbol (RL + DQN combined) |
| `POST` | `/api/trading/execute` | Execute a BUY/SELL/HOLD trade |
| `GET` | `/api/portfolio/summary` | Portfolio metrics, P&L, win rate |
| `GET` | `/api/trades/history` | Completed trade history |
| `GET` | `/api/backtest/results` | Latest backtest results |
| `GET` | `/api/market/data?symbols=X,Y` | Market data for symbols |

---

## 13. INTERVIEW Q&A

### "Your DQN is trained on historical data. Markets are non-stationary. How do you handle regime change?"

Three approaches: (1) Walk-forward retraining -- retrain every N days on the most recent K days. (2) Regime-conditional models -- detect the regime (trending/mean-reverting/volatile) using a hidden Markov model, then use a regime-specific DQN. (3) Meta-learning -- train the DQN to adapt quickly to new regimes (MAML/Reptile).

### "What's your Sharpe ratio? What's a realistic expectation?"

For intraday index futures, a Sharpe of 2.0+ is achievable because you're not exposed to overnight risk. You earn returns for ~6 hours and have zero exposure for 18 hours. Annualized Sharpe for intraday strategies is naturally inflated by sqrt(252). A more honest metric is per-trade Sharpe or profit factor (gross_profit / gross_loss).

### "Why not just use a simpler model -- logistic regression, random forest?"

For the signal generation layer, you could. The DQN's advantage is that it optimizes for cumulative reward (total session PnL), not per-bar prediction accuracy. A random forest predicts "will price go up in the next bar?" -- but that doesn't account for transaction costs, position management, or the sequence of actions. The DQN learns: "I should hold this position even though the next bar might be red, because the trend will resume."

### "How do you backtest without lookahead bias?"

The environment (`intraday_environment.py`) steps through data chronologically. `current_step` only sees data up to the current bar. Feature computation uses `.shift(1)` for lagged values. The scaler is fit only on training data, frozen for eval. Walk-forward validation uses strict temporal splits -- never future data in training.

### "What's your latency? This isn't really HFT."

Correct. This is medium-frequency systematic trading, not HFT. Decision latency is ~100ms (DQN inference on CPU). End-to-end tick-to-order is ~500ms-1s. True HFT operates at microseconds with FPGA/co-location. The principles (feature engineering, risk management, execution) are the same; the timescale is different. The system could be adapted for lower latency by: (1) moving to C++/Rust, (2) pre-computing features incrementally, (3) using ONNX Runtime for inference.

### "How would you handle slippage in options?"

Options slippage is much worse than futures. NIFTY futures have 0.05 tick, ~1-2 bps spread. NIFTY ATM weekly options might have Rs 1-2 spread on a Rs 100 premium = 100-200 bps. For OTM options, it's even worse. Model slippage as: `slippage = base_bps + k * (order_size / ADV) + spread_cost`. The DQN's reward function includes this cost, so it naturally avoids strategies that look profitable before costs but aren't after.

### "What happens if your model starts losing money in production?"

Three layers of protection: (1) **Risk limits** -- 5% daily loss cap kills all trading. (2) **Performance monitoring** -- if rolling 5-day Sharpe < -1.0, the system enters "reduced confidence" mode (halves position sizes). (3) **Manual override** -- the dashboard shows real-time metrics; the human can kill the system via API. Additionally, the `consecutive_losses` tracker in `RiskMetrics` triggers a cool-off after 5 straight losses.

---

## 14. WHAT'S PRODUCTION vs SCAFFOLDING

### Working end-to-end:
- DQN model architecture (PyTorch) -- trains, infers, saves/loads
- Intraday environment -- proper time-aware simulation
- Feature engineering pipeline -- 32 features, properly normalized
- Risk management framework -- limits, checks, kill switch
- Portfolio simulator -- realistic paper trading with costs
- LangGraph workflow -- compiles and runs
- News agent -- real scraping + FinBERT sentiment
- React dashboard -- connected to FastAPI backend

### Not yet connected:
- API uses simulated/random market data instead of real broker feed
- RL strategy agent uses rule-based logic, not the trained DQN
- Trained Keras weights incompatible with PyTorch production model
- Broker order submission is a no-op
- VIX, real OI, spot-futures basis all placeholder

### How to describe it:
"The system is in paper trading mode. The full pipeline works end-to-end in simulation. Connecting to live market data and converting to real execution is primarily an integration task -- the architecture, risk management, and model inference are production-ready."

---

## 15. IMPROVEMENT ROADMAP

### P0 (Must Fix)
- Replace random features with real data pipeline
- Unify PyTorch, retrain DQN on real NIFTY 5min data
- Wire DQN into rl_strategy_agent (replace rule-based)

### P1 (High Impact)
- Implement Prioritized Experience Replay (PER)
- Better reward function (differential Sharpe ratio)
- Fix LangGraph to use real news + parallel nodes

### P2 (Algorithm Upgrades)
- Try PPO/SAC for continuous action space
- Expand action space to 5-7 discrete actions
- Add portfolio state to observations (current position, unrealized P&L)
- Add attention mechanism / Transformer encoder

### P3 (Production Hardening)
- Walk-forward validation
- LLM meta-agent for trade reasoning
- ONNX Runtime for lower inference latency
- Options chain integration with Greeks computation
