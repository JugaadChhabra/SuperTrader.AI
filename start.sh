#!/bin/bash

# SuperTrader.AI - Quick Start Script
# This script starts both backend and frontend servers

set -e  # Exit on error

echo "🚀 Starting SuperTrader.AI Full Stack..."
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "${YELLOW}⚠️  Virtual environment not found. Creating one...${NC}"
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Check if FastAPI is installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "${YELLOW}⚠️  FastAPI not installed. Installing dependencies...${NC}"
    pip install fastapi "uvicorn[standard]" pydantic
fi

echo "${GREEN}✓${NC} Python virtual environment activated"

# Start backend
echo ""
echo "${BLUE}Starting Backend Server...${NC}"
python api/main.py > backend.log 2>&1 &
BACKEND_PID=$!
echo "${GREEN}✓${NC} Backend started (PID: $BACKEND_PID)"
echo "   Log: backend.log"
echo "   URL: http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"

# Wait for backend to initialize
echo ""
echo "Waiting for backend to initialize..."
sleep 5

# Test backend health
if curl -s http://localhost:8000/api/health > /dev/null; then
    echo "${GREEN}✓${NC} Backend is healthy"
else
    echo "${YELLOW}⚠️  Backend health check failed. Check backend.log${NC}"
fi

# Check if frontend node_modules exist
if [ ! -d "Frontend/node_modules" ]; then
    echo ""
    echo "${YELLOW}⚠️  Frontend dependencies not installed. Installing...${NC}"
    cd Frontend
    npm install
    cd ..
fi

# Start frontend
echo ""
echo "${BLUE}Starting Frontend Server...${NC}"
cd Frontend
npm run dev > ../frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..
echo "${GREEN}✓${NC} Frontend started (PID: $FRONTEND_PID)"
echo "   Log: frontend.log"
echo "   URL: http://localhost:5173"

echo ""
echo "═══════════════════════════════════════════════════════"
echo "${GREEN}✅ SuperTrader.AI is running!${NC}"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "📊 Dashboard: http://localhost:5173/dashboard"
echo "🔌 API Docs:  http://localhost:8000/docs"
echo "💚 Health:    http://localhost:8000/api/health"
echo ""
echo "Process IDs:"
echo "  Backend:  $BACKEND_PID"
echo "  Frontend: $FRONTEND_PID"
echo ""
echo "To stop servers:"
echo "  kill $BACKEND_PID $FRONTEND_PID"
echo "  or run: ./stop.sh"
echo ""
echo "View logs:"
echo "  Backend:  tail -f backend.log"
echo "  Frontend: tail -f frontend.log"
echo "═══════════════════════════════════════════════════════"

# Save PIDs for stop script
echo "$BACKEND_PID" > .backend.pid
echo "$FRONTEND_PID" > .frontend.pid

echo ""
echo "Press Ctrl+C to view menu options..."
