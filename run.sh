#!/bin/bash
# FPL Analytics App - Run Script

cd "$(dirname "$0")"

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt --break-system-packages -q

# Run the server
echo "Starting FPL Analytics server on http://localhost:8000"
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
