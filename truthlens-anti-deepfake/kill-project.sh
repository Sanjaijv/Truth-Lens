#!/bin/bash

# Define common ports
FRONTEND_PORT=3000
BACKEND_PORT=5001

kill_port() {
    local port=$1
    local name=$2
    local pid=$(lsof -t -i :$port)

    if [ -n "$pid" ]; then
        echo "Stopping $name on port $port (PID: $pid)..."
        kill -9 $pid
        echo "$name stopped."
    else
        echo "No process found running on port $port ($name)."
    fi
}

echo "--- Forensic Video Validation Tool: Process Killer ---"
kill_port $FRONTEND_PORT "Frontend (Next.js)"
kill_port $BACKEND_PORT "Backend (Express)"
echo "------------------------------------------------------"
echo "Done."
