#!/usr/bin/env python3
"""
Simple Python environment test for Railway deployment
"""
import sys
import os

print(f"🐍 Python version: {sys.version}")
print(f"📍 Python executable: {sys.executable}")
print(f"📁 Current directory: {os.getcwd()}")
print(f"🔧 PATH: {os.environ.get('PATH', 'Not set')}")

# Test imports
try:
    import uvicorn
    print("✅ uvicorn imported successfully")
    print(f"📦 uvicorn version: {uvicorn.__version__}")
except ImportError as e:
    print(f"❌ uvicorn import failed: {e}")

try:
    import fastapi
    print("✅ fastapi imported successfully")
    print(f"📦 fastapi version: {fastapi.__version__}")
except ImportError as e:
    print(f"❌ fastapi import failed: {e}")

print("🎯 Python environment test completed")