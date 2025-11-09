# 🚀 SuperTrader.AI - Quick Reference

## Start Everything
```bash
./start.sh
```

## Stop Everything
```bash
./stop.sh
```

## Access Points

| Service | URL | Description |
|---------|-----|-------------|
| **Dashboard** | http://localhost:5173/dashboard | Main trading dashboard |
| **API Docs** | http://localhost:8000/docs | Swagger UI documentation |
| **Health Check** | http://localhost:8000/api/health | Backend status |
| **Frontend Dev** | http://localhost:5173 | Frontend development server |

## Important Metrics on Dashboard

### 🎯 Top Priority Metrics
1. **Total P&L** - Your overall profit/loss (realized + unrealized)
2. **Win Rate** - Success rate of your trades
3. **Market Status** - Live countdown to market close (3:15 PM)
4. **Realized P&L** - Actual money made from closed trades
5. **Open Positions** - Current holdings and their performance
6. **Cash Balance** - Available money for new trades

### 🕐 Intraday Rules
- Market Hours: **9:30 AM - 3:15 PM**
- All positions MUST close before **3:15 PM**
- No overnight holdings
- Dashboard shows countdown timer

## API Endpoints

```bash
# Get portfolio summary
curl http://localhost:8000/api/portfolio/summary

# Get AI decision for a stock
curl -X POST http://localhost:8000/api/trading/analyze \
  -H "Content-Type: application/json" \
  -d '{"symbol": "RELIANCE"}'

# Get trade history
curl http://localhost:8000/api/trades/history

# Get trading status
curl http://localhost:8000/api/trading/status
```

## Logs

```bash
# View backend logs
tail -f backend.log

# View frontend logs
tail -f frontend.log

# Check if servers are running
lsof -i :8000  # Backend
lsof -i :5173  # Frontend
```

## Troubleshooting

### Port Already in Use
```bash
# Kill process on port 8000 (Backend)
kill -9 $(lsof -ti :8000)

# Kill process on port 5173 (Frontend)
kill -9 $(lsof -ti :5173)
```

### Backend Won't Start
```bash
source venv/bin/activate
pip install -r requirements.txt
python api/main.py
```

### Frontend Won't Start
```bash
cd Frontend
npm install
npm run dev
```

### CORS Errors
- Make sure backend is running on port 8000
- Check browser console for specific error
- Verify CORS settings in `api/main.py`

## File Structure

```
SuperTrader.AI/
├── api/
│   └── main.py              ← FastAPI backend (NEW)
├── Frontend/
│   └── app/
│       ├── services/
│       │   └── tradingApi.ts       ← API client (NEW)
│       ├── components/dashboard/
│       │   ├── IntradayTradingDashboard.tsx  ← Dashboard (NEW)
│       │   └── IntradayTradingDashboard.css  ← Styles (NEW)
│       └── routes/
│           └── dashboard.tsx       ← Updated route
├── start.sh                 ← Start script (NEW)
├── stop.sh                  ← Stop script (NEW)
├── SETUP_COMPLETE.md        ← Full documentation (NEW)
└── FULLSTACK_README.md      ← Technical guide (NEW)
```

## Tech Stack

**Backend:**
- FastAPI (REST API)
- PyTorch (DQN Model)
- Pandas/NumPy (Data)
- Uvicorn (Server)

**Frontend:**
- React 19
- TypeScript
- React Router 7
- Tailwind CSS
- Vite

## Next Steps

### To Use:
1. Run `./start.sh`
2. Open http://localhost:5173/dashboard
3. View real-time metrics
4. Check AI trading decisions

### To Develop:
1. **Add real market data** - Replace simulated data
2. **Add authentication** - Protect dashboard
3. **Add WebSocket** - Real-time price updates
4. **Add charts** - Visualize P&L trends
5. **Add broker integration** - Execute real trades

## Success Criteria ✅

- [x] Backend API running
- [x] Frontend dashboard created
- [x] Real-time data refresh (10s)
- [x] Market status with countdown
- [x] All key metrics displayed
- [x] Professional styling
- [x] Mobile responsive
- [x] Error handling
- [x] Documentation complete

---

**Status:** ✅ COMPLETE AND READY TO USE

**Support:** Check SETUP_COMPLETE.md for detailed documentation
