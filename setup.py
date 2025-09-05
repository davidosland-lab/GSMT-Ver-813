#!/usr/bin/env python3
"""
Setup script for Global Stock Market Tracker - Local Deployment
Installs dependencies and configures the application
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def install_requirements():
    """Install Python dependencies"""
    print("📦 Installing Python dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False
    return True

def create_pm2_config():
    """Create PM2 ecosystem configuration for service management"""
    print("🔧 Creating PM2 ecosystem configuration...")
    
    config = {
        "apps": [{
            "name": "global-market-tracker",
            "script": "app.py",
            "interpreter": "python3",
            "cwd": str(Path.cwd()),
            "env": {
                "PORT": "8000",
                "PYTHONPATH": str(Path.cwd())
            },
            "instances": 1,
            "exec_mode": "fork",
            "watch": False,
            "autorestart": True,
            "max_restarts": 10,
            "min_uptime": "10s",
            "log_file": "./logs/app.log",
            "error_file": "./logs/error.log",
            "out_file": "./logs/out.log",
            "log_date_format": "YYYY-MM-DD HH:mm:ss Z"
        }]
    }
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Write ecosystem file
    with open("ecosystem.config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✅ PM2 configuration created")
    return True

def create_supervisor_config():
    """Create Supervisor configuration as alternative to PM2"""
    print("🔧 Creating Supervisor configuration...")
    
    config = f"""[supervisord]
nodaemon=false
logfile={Path.cwd()}/logs/supervisord.log
pidfile={Path.cwd()}/supervisord.pid
childlogdir={Path.cwd()}/logs

[unix_http_server]
file={Path.cwd()}/supervisor.sock

[supervisorctl]
serverurl=unix://{Path.cwd()}/supervisor.sock

[rpcinterface:supervisor]
supervisor.rpcinterface_factory = supervisor.rpcinterface:make_main_rpcinterface

[program:global-market-tracker]
command=python3 app.py
directory={Path.cwd()}
user={os.getenv('USER', 'user')}
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile={Path.cwd()}/logs/app.log
environment=PORT="8000",PYTHONPATH="{Path.cwd()}"
"""
    
    with open("supervisord.conf", "w") as f:
        f.write(config)
    
    print("✅ Supervisor configuration created")
    return True

def create_startup_scripts():
    """Create convenient startup scripts"""
    print("📝 Creating startup scripts...")
    
    # Create start.sh script
    start_script = """#!/bin/bash
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
"""
    
    with open("start.sh", "w") as f:
        f.write(start_script)
    os.chmod("start.sh", 0o755)
    
    # Create stop.sh script
    stop_script = """#!/bin/bash
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
"""
    
    with open("stop.sh", "w") as f:
        f.write(stop_script)
    os.chmod("stop.sh", 0o755)
    
    print("✅ Startup scripts created")
    return True

def create_development_scripts():
    """Create development helper scripts"""
    print("🛠️ Creating development scripts...")
    
    # Development server script
    dev_script = """#!/bin/bash
# Development server with auto-reload

echo "🔧 Starting development server..."
echo "📱 Frontend: http://localhost:8000/"
echo "📚 API Docs: http://localhost:8000/api/docs"
echo ""

python3 -c "
import uvicorn
from app import app

if __name__ == '__main__':
    uvicorn.run(
        'app:app',
        host='0.0.0.0',
        port=8000,
        reload=True,
        log_level='info'
    )
"
"""
    
    with open("dev.sh", "w") as f:
        f.write(dev_script)
    os.chmod("dev.sh", 0o755)
    
    # Health check script
    health_script = """#!/bin/bash
# Health check script

echo "🩺 Checking application health..."

response=$(curl -s -w "%{http_code}" http://localhost:8000/api/health -o /tmp/health_response.json)

if [ "$response" = "200" ]; then
    echo "✅ Application is healthy"
    cat /tmp/health_response.json | python3 -m json.tool
else
    echo "❌ Application health check failed (HTTP $response)"
    if [ -f /tmp/health_response.json ]; then
        cat /tmp/health_response.json
    fi
fi
"""
    
    with open("health-check.sh", "w") as f:
        f.write(health_script)
    os.chmod("health-check.sh", 0o755)
    
    print("✅ Development scripts created")
    return True

def main():
    """Main setup function"""
    print("🌍 Global Stock Market Tracker - Local Deployment Setup")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    
    print(f"🐍 Python version: {sys.version}")
    
    # Install dependencies
    if not install_requirements():
        return False
    
    # Create service configurations
    create_pm2_config()
    create_supervisor_config()
    
    # Create startup scripts
    create_startup_scripts()
    create_development_scripts()
    
    print("\n" + "=" * 60)
    print("✅ Setup completed successfully!")
    print("\n📋 Quick Start:")
    print("   • Development: ./dev.sh")
    print("   • Production:  ./start.sh")
    print("   • Stop:        ./stop.sh")
    print("   • Health:      ./health-check.sh")
    print("\n🌐 Access URLs:")
    print("   • Frontend:    http://localhost:8000/")
    print("   • API Docs:    http://localhost:8000/api/docs")
    print("   • API Health:  http://localhost:8000/api/health")
    print("\n🔧 Process Management:")
    print("   • PM2:         pm2 start ecosystem.config.json")
    print("   • Supervisor:  supervisord -c supervisord.conf")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)