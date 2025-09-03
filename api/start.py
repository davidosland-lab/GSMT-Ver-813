#!/usr/bin/env python3
"""
GSMT Ver 6.0 API Server Startup Script
Run this script to start the FastAPI backend server
"""

import uvicorn
import os
import sys

def main():
    """Start the FastAPI server"""
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    print("🚀 Starting GSMT Ver 6.0 API Server")
    print(f"📡 Server will run on http://{host}:{port}")
    print("📊 Features: FastAPI, Percentage Analysis, Multi-source Data")
    print("🌐 API Documentation: http://localhost:8000/docs")
    print("=" * 60)
    
    try:
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            log_level="info",
            reload=False,  # Set to True for development
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()