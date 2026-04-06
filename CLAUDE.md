# CLAUDE.md - SuperTrader.AI

## Project Overview

**SuperTrader.AI** is a production-grade intraday AI trading system for Indian equity markets (NSE/BSE). It combines Deep Q-Network (DQN) reinforcement learning with multi-agent orchestration, real-time risk management, and a modern React dashboard. Currently in **paper trading** mode.

**Trading scope**: Intraday only. All positions are squared off by 3:15 PM IST. Market hours: 9:15 AM - 3:30 PM IST.

---

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Backend** | Python 3.8+, FastAPI, Uvicorn |
| **Frontend** | React 19, TypeScript, React Router 7, Tailwind CSS 4, Vite |
| **ML/AI** | PyTorch (DQN), TensorFlow/Keras (checkpoints), scikit-learn, HuggingFace Transformers |
| **Orchestration** | LangGraph (multi-agent workflow) |
| **Broker** | ICICI Direct API (live data) |
| **Auth** | Firebase |
| **3D/UI** | Three.js, Framer Motion |
| **Data** | Pandas, NumPy, SQLite, CSV/JSON |
| **LFS** | `.weights.h5`, `.pkl`, `.joblib` files tracked via Git LFS |

---

## Project Structure

```
SuperTrader.AI/
├── api/                         # FastAPI backend
│   └── main.py                  # Entry point (port 8000)
│
├── Frontend/                    # React + TypeScript frontend
│   ├── app/
│   │   ├── routes/              # 9 routes (dashboard, home, login, settings, news, etc.)
│   │   ├── components/          # 24 React components
│   │   ├── contexts/            # AuthContext (Firebase)
│   │   └── services/            # tradingApi.ts (axios API client)
│   ├── package.json
│   └── vite.config.ts
│
├── agents/                      # Core AI trading agents
│   ├── data_agent/              # Market data fetching & feature engineering
│   │   ├── icici_broker.py      # ICICI Direct API integration
│   │   ├── futures_manager.py   # Futures contract management
│   │   ├── feature_builders.py  # Feature engineering (largest module, 54KB)
│   │   ├── validators.py        # Data validation
│   │   └── constants.py         # API endpoints & mappings
│   ├── execution_agent.py       # Trade execution + risk management (2524 lines)
│   ├── rl_strategy_agent.py     # DQN policy & state representation (469 lines)
│   ├── news_agent.py            # Sentiment analysis via web scraping (395 lines)
│   ├── trade_ledger.py          # Trade lifecycle tracking (1290 lines)
│   ├── enhanced_trade_ledger.py # Session ledger with dual CSV/JSON logging
│   ├── universe_ranking.py      # Tradeable asset scoring (759 lines)
│   └── volatility_position_sizing.py  # Risk-adjusted position sizing (543 lines)
│
├── models/                      # ML model definitions
│   ├── dqn_network.py           # Dueling DQN with LSTM (PyTorch)
│   ├── training.py              # Episode-based training loop
│   └── intraday_environment.py  # Market simulator environment
│
├── utils/                       # Infrastructure utilities
│   ├── risk_config.py           # Risk limit management (749 lines)
│   ├── position_manager.py      # Position lifecycle (734 lines)
│   ├── portfolio_simulator.py   # Paper trading simulator (689 lines)
│   ├── production.py            # Production deployment utils (678 lines)
│   ├── logging.py               # Structured logging (575 lines)
│   ├── performance.py           # Performance metrics (494 lines)
│   ├── monitoring.py            # Real-time monitoring (486 lines)
│   ├── premarket_analyzer.py    # Pre-market analysis (468 lines)
│   └── scaler_manager.py        # Feature scaler loading
│
├── indicators/                  # Technical indicator library
│   ├── technical.py             # RSI, MACD, Bollinger, etc.
│   └── features.py              # Feature computation pipeline
│
├── graph/                       # LangGraph workflow definitions
│
├── configs/                     # Configuration (YAML + Python)
│   ├── config.py                # Main config (dataclasses)
│   ├── risk.yaml                # Risk limits
│   ├── market.yaml              # Market settings & symbols
│   ├── rl.yaml                  # RL hyperparameters
│   └── agents.yaml              # Agent-specific settings
│
├── notebooks/                   # Jupyter training notebooks (5 notebooks, ~34MB)
│   ├── intraday_trading_dqn_training.ipynb   # Main DQN training
│   ├── intraday_entry_exit_tf.ipynb          # TensorFlow entry/exit
│   ├── intraday_entry_exit_lstm.ipynb        # LSTM variant
│   ├── intraday_entry_exit_gru.ipynb         # GRU variant
│   ├── intraday_entry_exit_neuralprophet.ipynb  # NeuralProphet
│   └── models/                  # Trained weights & scalers
│       ├── intraday_dqn/        # 150 checkpoints (episode_10 → episode_150)
│       └── feature_scaler.pkl   # Fitted scaler (Git LFS)
│
├── data/                        # Trading data
│   ├── NIFTY_historical_data_5min.csv        # NIFTY 50 5-min OHLCV
│   ├── NSE_AllStocks_historical_data_5min.json  # All NSE stocks
│   ├── stock_names_symbol.csv   # Symbol mapping for news
│   └── processed_data/          # Feature-engineered datasets
│
├── scripts/                     # Helper scripts
├── tests/                       # Test suite
│
├── start.sh                     # Start full stack (backend + frontend)
├── stop.sh                      # Stop all servers
├── requirements.txt             # Python dependencies
│
├── *.py (root)                  # Test/demo scripts
│   ├── run_simple_test.py       # Basic portfolio test
│   ├── run_portfolio_demo.py    # Multi-asset demo
│   ├── rl_portfolio_test.py     # RL strategy test
│   ├── dqn_portfolio_test.py    # DQN-specific test
│   ├── enhanced_portfolio_test.py  # Enhanced features test
│   ├── test_agentic_portfolio_manager.py  # Agentic workflow test
│   ├── test.py / quick_test.py  # General/sanity tests
│   └── main.py                  # Main orchestration entry
│
└── *_output/                    # Test output directories
    ├── simple_test_output/
    ├── test_outputs/
    ├── rl_strategy_output/
    ├── enhanced_output/
    ├── demo_outputs/ledgers/
    └── validation_output/
```

---

## Architecture & Data Flow

```
Live Market Data (ICICI Direct API)
    │
    ▼
Data Agent ──→ Feature Engineering (OHLCV + 32 technical features)
    │              └── RSI, MACD, Bollinger, PCR, momentum, volatility regime
    ▼
RL Strategy Agent (DQN)
    │   ├── Dueling DQN with LSTM layers
    │   ├── 3 actions: BUY/LONG, SELL/SHORT, HOLD
    │   └── Reward: 0.7×return + 0.3×(1-risk)
    ▼
News Agent ──→ Sentiment Analysis
    │   ├── Web scraping (Zerodha, NSE, Upstox)
    │   └── HuggingFace sentiment pipeline
    ▼
Execution Agent
    │   ├── Pre-trade risk checks
    │   ├── Volatility-adjusted position sizing
    │   ├── Order generation & execution
    │   └── Auto square-off at 3:15 PM
    ▼
Portfolio Simulator (Paper Trading)
    │   ├── Cash & position management
    │   ├── Transaction cost modeling (brokerage, STT, slippage)
    │   └── P&L tracking (realized + unrealized)
    ▼
FastAPI Backend (port 8000) ──→ React Dashboard (port 5173)
```

---

## DQN Model Architecture

```
Input: 32 features × 60 timestep lookback
    ↓
LSTM Layer 1 (input=32, hidden=32, dropout=0.2)
    ↓
LSTM Layer 2 (input=32, hidden=16)
    ↓
┌─────────────────────┬──────────────────────────┐
│ Value Stream         │ Advantage Stream          │
│ Dense(16)→LeakyReLU │ Dense(16)→LeakyReLU       │
│ Dense(1) = V(s)     │ Dense(3) = A(s,a)         │
└─────────────────────┴──────────────────────────┘
    ↓
Q(s,a) = V(s) + A(s,a) - mean(A)   →   3 actions (SHORT, HOLD, LONG)
```

**Training**: 150 episodes, lr=0.001, gamma=0.95, epsilon decay 0.1→0.01, replay buffer 10K.

---

## Key Configuration

### Risk Limits (`configs/risk.yaml`)
- Max daily loss: **5%**
- Max drawdown: **10%**
- Max position size: **20%** of portfolio
- Stop loss: **2%**
- Commission: **0.1%**
- Slippage: **5 bps**

### Trading Config (`configs/config.py`)
- Initial capital: **₹1,00,000** (configurable, API uses ₹10,00,000)
- Position sizing: **20%** of portfolio per trade
- Session length: **390 minutes** (9:15 AM - 3:30 PM)
- Data refresh: **60 seconds**
- News sentiment weight: **0.1**
- Decision timeout: **5 seconds**

### Performance Targets
- Sharpe ratio: **1.5**
- Win rate: **55%**
- Max drawdown: **10%**

---

## API Endpoints

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

**CORS**: Allows `localhost:3000` and `localhost:5173`.

---

## Quick Start

```bash
# Full stack (backend + frontend)
./start.sh

# Or manually:
# Terminal 1 - Backend
source venv/bin/activate
python api/main.py          # http://localhost:8000

# Terminal 2 - Frontend
cd Frontend && npm run dev  # http://localhost:5173
```

**Dashboard**: http://localhost:5173/dashboard
**API Docs**: http://localhost:8000/docs

---

## Common Commands

```bash
# Install Python deps
pip install -r requirements.txt

# Install frontend deps
cd Frontend && npm install

# Run tests
python run_simple_test.py        # Basic portfolio test
python enhanced_portfolio_test.py # Full feature test
python dqn_portfolio_test.py     # DQN model test

# Stop servers
./stop.sh
```

---

## Data & Logging

### Data Files
- `data/NIFTY_historical_data_5min.csv` - NIFTY 50 index 5-min candles
- `data/NSE_AllStocks_historical_data_5min.json` - All NSE stocks 5-min data
- `data/stock_names_symbol.csv` - Symbol↔name mapping for news agent

### Logging (Dual System)
- **CSV Ledger**: `ledgers/ledger_master.csv` - one row per trade
- **JSON Sessions**: `sessions/session_<timestamp>.json` - full session details
- **SQLite**: `data/trade_ledger.db` - persistent trade database
- **Server Logs**: `backend.log`, `frontend.log`

---

## Git LFS

Tracked in `.gitattributes`:
```
*.weights.h5   # Keras/TF model checkpoints (~162KB each, 15+ files)
*.pkl          # Pickle files (feature scaler)
*.joblib       # Joblib serialized objects
```

---

## Environment Variables

Stored in `.env` (gitignored). Required for:
- ICICI Direct API credentials (app key, session token)
- Firebase config (frontend auth)

---

## Important Patterns & Notes

1. **AI Decision Combining**: RL signal weighted 30%, DQN signal weighted 70% → combined score determines action
2. **State Representation**: `build_state_representation()` in `agents/rl_strategy_agent.py` creates the feature vector fed to models
3. **Feature Engineering**: `agents/data_agent/feature_builders.py` is the most complex module — handles Phase 4 features (PCR, momentum confluence, market regime, options flow)
4. **Intraday Constraint**: All logic enforces square-off before 3:15 PM. The `minutes_to_close` feature directly influences trading decisions
5. **Paper Trading**: `utils/portfolio_simulator.py` simulates realistic execution with transaction costs
6. **No live execution yet** - broker integration exists but system runs in paper mode
7. **Notebooks are large** (~34MB total) - contain training outputs, plots, and model artifacts
8. **Root-level .py files** are test/demo scripts, not part of the core system

---

## Frontend Routes

| Route | Component | Purpose |
|-------|-----------|---------|
| `/` | `home.tsx` | Landing page |
| `/dashboard` | `dashboard.tsx` | Main trading dashboard |
| `/login` | `login.tsx` | Firebase authentication |
| `/settings` | `settings.tsx` | User preferences |
| `/news` | `news.tsx` | Market news & sentiment |
| `/backtest` | `backtest.tsx` | Backtest results viewer |
| `/analytics` | `analytics.tsx` | Performance analytics |

---

## Broker Integration (ICICI Direct)

Located in `agents/data_agent/`:
- `icici_broker.py` handles auth (app key + session token) and WebSocket data streaming
- `futures_manager.py` manages futures contract lifecycle
- Connection flow: Auth → WebSocket → Subscribe → Tick updates → Feature generation → RL inference

---

## Code Style & Conventions

- Python: Standard Python with type hints, dataclasses for config
- Frontend: TypeScript strict mode, functional React components with hooks
- Config: YAML for tunable parameters, Python dataclasses for structured config
- Logging: Structured logging with emoji prefixes for visual parsing
- Naming: snake_case (Python), camelCase (TypeScript)
