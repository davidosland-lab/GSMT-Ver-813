#!/bin/bash

# Railway deployment startup script for GSMT Ver 7.0
echo "🚀 Starting GSMT Ver 7.0 deployment..."

# Check Python installation
echo "📋 Checking Python environment..."
python3 --version || python --version || {
    echo "❌ Python not found!"
    exit 1
}

# Check pip installation  
echo "📦 Checking pip installation..."
python3 -m pip --version || python -m pip --version || {
    echo "❌ pip not found!"
    exit 1
}

# Install dependencies
echo "⬇️ Installing dependencies..."
python3 -m pip install -r requirements.txt || python -m pip install -r requirements.txt || {
    echo "❌ Failed to install dependencies!"
    exit 1
}

# Get port from environment or default to 8000
PORT=${PORT:-8000}
echo "🌐 Starting server on port $PORT..."

# Start the application
exec python3 app.py || exec python app.py