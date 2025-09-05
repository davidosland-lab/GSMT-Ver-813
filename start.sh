#!/bin/bash
# Global Stock Market Tracker - Startup Script

echo "🚀 Starting Global Stock Market Tracker..."

# Check if PM2 is available
if command -v pm2 &> /dev/null; then
    echo "📦 Using PM2 for process management..."
    pm2 start ecosystem.config.json
    pm2 logs global-market-tracker --nostream
else
    # Check if supervisor is available
    if command -v supervisord &> /dev/null; then
        echo "📦 Using Supervisor for process management..."
        supervisord -c supervisord.conf
        sleep 2
        supervisorctl -c supervisord.conf status
    else
        echo "🔧 No process manager found, starting directly..."
        echo "💡 Consider installing PM2 (npm install -g pm2) or supervisor (pip install supervisor)"
        python3 app.py
    fi
fi
