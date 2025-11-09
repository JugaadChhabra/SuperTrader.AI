# 🚀 How to Run SuperTrader.AI

## Quick Start (2 Steps)

### Step 1: Start Backend
```bash
cd /Users/muaazshaikh/SuperTrader.AI
source venv/bin/activate
python api/main.py
```

**Backend will run on:** http://localhost:8000

---

### Step 2: Start Frontend (New Terminal)
```bash
cd /Users/muaazshaikh/SuperTrader.AI/Frontend
npm run dev
```

**Frontend will run on:** http://localhost:5173

---

## Open Dashboard
Go to: **http://localhost:5173/dashboard**

---

## Stop Servers

**Stop Backend:** Press `Ctrl+C` in backend terminal

**Stop Frontend:** Press `Ctrl+C` in frontend terminal

---

## One-Line Commands

### Backend
```bash
cd /Users/muaazshaikh/SuperTrader.AI && source venv/bin/activate && python api/main.py
```

### Frontend
```bash
cd /Users/muaazshaikh/SuperTrader.AI/Frontend && npm run dev
```

---

## Automated Scripts (Alternative)

### Start Everything
```bash
./start.sh
```

### Stop Everything
```bash
./stop.sh
```

---

## Check if Running

**Backend:**
```bash
curl http://localhost:8000/api/health
```

**Frontend:**
```bash
lsof -i :5173
```

---

## Troubleshooting

**Port already in use?**
```bash
# Kill backend on port 8000
kill -9 $(lsof -ti :8000)

# Kill frontend on port 5173
kill -9 $(lsof -ti :5173)
```

**Missing dependencies?**
```bash
# Backend
pip install -r requirements.txt

# Frontend
cd Frontend && npm install
```

---

## That's it! 🎉

Backend + Frontend = Full Trading Dashboard with AI
