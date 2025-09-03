#!/usr/bin/env python3
"""
GSMT Ver 7.0 - Railway Startup Script
Alternative startup method that bypasses Procfile issues
"""

import os
import sys
import subprocess
import time

def log(message):
    """Simple logging function"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def install_dependencies():
    """Install required packages"""
    log("🔧 Installing dependencies...")
    
    packages = ["fastapi==0.104.1", "uvicorn[standard]==0.24.0"]
    
    for package in packages:
        try:
            log(f"Installing {package}...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", package
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                log(f"❌ Failed to install {package}: {result.stderr}")
                return False
            else:
                log(f"✅ Successfully installed {package}")
        except Exception as e:
            log(f"❌ Exception installing {package}: {e}")
            return False
    
    return True

def start_server():
    """Start the FastAPI server"""
    log("🚀 Starting GSMT API server...")
    
    try:
        import uvicorn
        from app import app
        
        port = int(os.environ.get("PORT", 8000))
        log(f"🌐 Starting server on port {port}")
        
        # Start the server
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info",
            workers=1
        )
        
    except ImportError as e:
        log(f"❌ Import error: {e}")
        return False
    except Exception as e:
        log(f"❌ Server startup error: {e}")
        return False
    
    return True

def main():
    """Main startup function"""
    log("🎯 GSMT Ver 7.0 - Railway Startup")
    log(f"🐍 Python version: {sys.version}")
    log(f"📁 Working directory: {os.getcwd()}")
    log(f"🌐 PORT environment: {os.environ.get('PORT', 'not set')}")
    
    # Check if dependencies are already installed
    try:
        import fastapi
        import uvicorn
        log("✅ Dependencies already available")
    except ImportError:
        log("📦 Installing dependencies...")
        if not install_dependencies():
            log("❌ Failed to install dependencies")
            sys.exit(1)
    
    # Start the server
    if not start_server():
        log("❌ Failed to start server")
        sys.exit(1)

if __name__ == "__main__":
    main()