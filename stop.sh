#!/bin/bash

# SuperTrader.AI - Stop Script
# This script stops both backend and frontend servers

echo "🛑 Stopping SuperTrader.AI servers..."

# Kill backend
if [ -f ".backend.pid" ]; then
    BACKEND_PID=$(cat .backend.pid)
    if kill -0 $BACKEND_PID 2>/dev/null; then
        kill $BACKEND_PID
        echo "✓ Backend stopped (PID: $BACKEND_PID)"
    else
        echo "⚠️  Backend process not found"
    fi
    rm .backend.pid
fi

# Kill frontend
if [ -f ".frontend.pid" ]; then
    FRONTEND_PID=$(cat .frontend.pid)
    if kill -0 $FRONTEND_PID 2>/dev/null; then
        kill $FRONTEND_PID
        echo "✓ Frontend stopped (PID: $FRONTEND_PID)"
    else
        echo "⚠️  Frontend process not found"
    fi
    rm .frontend.pid
fi

# Kill any remaining processes on ports 8000 and 5173
echo ""
echo "Checking for remaining processes..."

# Port 8000 (Backend)
PORT_8000_PID=$(lsof -ti :8000)
if [ ! -z "$PORT_8000_PID" ]; then
    kill -9 $PORT_8000_PID 2>/dev/null
    echo "✓ Killed process on port 8000"
fi

# Port 5173 (Frontend)
PORT_5173_PID=$(lsof -ti :5173)
if [ ! -z "$PORT_5173_PID" ]; then
    kill -9 $PORT_5173_PID 2>/dev/null
    echo "✓ Killed process on port 5173"
fi

echo ""
echo "✅ All servers stopped"
