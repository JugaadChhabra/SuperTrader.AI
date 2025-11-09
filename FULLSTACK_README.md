# SuperTrader.AI - Full Stack Setup

## Backend (FastAPI)

### Installation
```bash
# Install Python dependencies
pip install fastapi uvicorn[standard] pydantic

# Or install all requirements
pip install -r requirements.txt
```

### Running the Backend
```bash
# From the project root directory
python api/main.py

# Or using uvicorn directly
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

The backend will start on: **http://localhost:8000**

### API Endpoints

- **GET** `/api/health` - Health check with system status
- **GET** `/api/trading/status` - Current trading status
- **POST** `/api/trading/analyze` - Get AI decision for a symbol
- **GET** `/api/portfolio/summary` - Portfolio summary with P&L
- **GET** `/api/trades/history` - Completed trades history
- **POST** `/api/trading/execute` - Execute a trade
- **GET** `/api/market/data` - Get market data for symbols

### API Documentation
Once the backend is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Frontend (React Router)

### Installation
```bash
cd Frontend
npm install
```

### Running the Frontend
```bash
# Development mode
npm run dev

# Build for production
npm run build

# Start production server
npm start
```

The frontend will start on: **http://localhost:5173** (dev) or **http://localhost:3000** (prod)

## Full Stack Startup

### Option 1: Manual (Two Terminals)

**Terminal 1 - Backend:**
```bash
python api/main.py
```

**Terminal 2 - Frontend:**
```bash
cd Frontend
npm run dev
```

### Option 2: Using Screen (Background)

```bash
# Start backend in background
screen -dmS backend python api/main.py

# Start frontend in background
screen -dmS frontend bash -c "cd Frontend && npm run dev"

# View running sessions
screen -ls

# Attach to a session
screen -r backend  # or frontend

# Detach from session: Ctrl+A then D
```

## Important Dashboard Metrics

The new intraday trading dashboard displays:

### Critical Metrics (Top Cards)
- **Portfolio Value** - Total portfolio worth
- **Total P&L** - Combined realized + unrealized P&L
- **Realized P&L** - Profits/losses from closed trades
- **Unrealized P&L** - Current profits/losses on open positions
- **Win Rate** - Percentage of winning trades
- **Cash Balance** - Available cash for trading

### Market Status Banner
- Live market status (OPEN/CLOSED)
- Time until market close countdown
- Updates in real-time

### Open Positions Table
- Symbol, quantity, entry/current price
- Unrealized P&L and return percentage
- Real-time price updates

### Recent Trades Table
- Last 5 completed trades
- Entry/exit times and prices
- P&L and return percentage for each trade

### AI Trading Status
- Models initialization status
- DQN model availability
- Total trades and open positions count

## Intraday Trading Rules

⚠️ **CRITICAL**: This is an **INTRADAY** trading system
- Market hours: **9:30 AM - 3:15 PM**
- All positions MUST be squared off before **3:15 PM**
- No overnight positions allowed
- Real-time P&L tracking throughout the day

## Troubleshooting

### Backend Issues
- Check if port 8000 is available: `lsof -i :8000`
- Verify Python dependencies are installed
- Check if DQN model files exist in `models/` directory

### Frontend Issues
- Check if port 5173/3000 is available: `lsof -i :5173`
- Clear node_modules and reinstall: `rm -rf node_modules && npm install`
- Check CORS settings in backend (api/main.py)

### API Connection Issues
- Verify backend is running: `curl http://localhost:8000/api/health`
- Check browser console for CORS errors
- Ensure API_BASE_URL in tradingApi.ts matches backend URL

## Testing the System

1. **Start Backend**: `python api/main.py`
2. **Test Health**: Open http://localhost:8000/api/health
3. **Check API Docs**: Open http://localhost:8000/docs
4. **Start Frontend**: `cd Frontend && npm run dev`
5. **Open Dashboard**: Navigate to http://localhost:5173/dashboard

## Production Deployment

### Backend
```bash
# Use gunicorn for production
pip install gunicorn
gunicorn api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Frontend
```bash
cd Frontend
npm run build
npm start
```

## Environment Variables

Create a `.env` file in the project root:

```env
# Backend
BACKEND_PORT=8000
BACKEND_HOST=0.0.0.0

# Frontend
VITE_API_BASE_URL=http://localhost:8000/api

# Trading
INITIAL_CAPITAL=1000000
MAX_POSITIONS=10
RISK_PER_TRADE=0.02
```

## Key Features

✅ Real-time intraday trading dashboard
✅ AI-powered trading decisions (RL + DQN fusion)
✅ Automatic EOD square-off
✅ Live P&L tracking
✅ Win rate and trade statistics
✅ Market status with countdown
✅ RESTful API with FastAPI
✅ Modern React frontend with React Router
✅ Responsive design for mobile/desktop

## Support

For issues or questions:
1. Check the logs in backend terminal
2. Check browser console for frontend errors
3. Review API documentation at /docs
4. Verify all dependencies are installed
