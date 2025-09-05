#!/bin/bash
# Global Stock Market Tracker - Stop Script

echo "🛑 Stopping Global Stock Market Tracker..."

# Try to stop PM2 first
if command -v pm2 &> /dev/null; then
    pm2 stop global-market-tracker 2>/dev/null || true
    pm2 delete global-market-tracker 2>/dev/null || true
fi

# Try to stop supervisor
if [ -f supervisord.pid ]; then
    supervisorctl -c supervisord.conf stop all 2>/dev/null || true
    supervisorctl -c supervisord.conf shutdown 2>/dev/null || true
fi

# Kill any remaining processes
pkill -f "python.*app.py" 2>/dev/null || true

echo "✅ Application stopped"
