#!/usr/bin/env python3
"""
Enhanced Stock Analysis - Local Mirror Startup Script
====================================================

This script starts the local mirror system with proper configuration and monitoring.

Usage:
    python start_local_mirror.py [options]
    
Options:
    --host HOST         Host to bind to (default: 127.0.0.1)
    --port PORT         Port to bind to (default: 8000)
    --config CONFIG     Path to configuration file
    --debug            Enable debug mode
    --reload           Enable auto-reload for development
    --workers WORKERS  Number of worker processes (default: 1)

Author: Local Mirror System
Version: 1.0.0
Date: 2025-09-16
"""

import os
import sys
import argparse
import logging
import signal
import asyncio
from pathlib import Path
import json
import subprocess
import time

# Add the project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

def setup_logging(debug: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(PROJECT_ROOT / 'logs' / 'startup.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are installed."""
    logger = logging.getLogger(__name__)
    
    required_packages = [
        'fastapi',
        'uvicorn',
        'pandas',
        'yfinance',
        'nltk',
        'spacy',
        'transformers',
        'torch'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"✗ {package} is NOT installed")
    
    if missing_packages:
        logger.error("Missing required packages. Please run: pip install -r requirements.txt")
        return False
    
    return True

def download_nlp_models():
    """Download required NLP models if not already present."""
    logger = logging.getLogger(__name__)
    
    try:
        # Download NLTK data
        import nltk
        nltk_data = ['punkt', 'stopwords', 'vader_lexicon', 'wordnet']
        
        for data in nltk_data:
            try:
                nltk.data.find(f'tokenizers/{data}')
                logger.info(f"✓ NLTK {data} already downloaded")
            except LookupError:
                logger.info(f"Downloading NLTK {data}...")
                nltk.download(data, quiet=True)
                logger.info(f"✓ NLTK {data} downloaded")
        
        # Download spaCy model
        import spacy
        try:
            spacy.load("en_core_web_sm")
            logger.info("✓ spaCy en_core_web_sm model already available")
        except OSError:
            logger.info("Downloading spaCy en_core_web_sm model...")
            subprocess.run([
                sys.executable, "-m", "spacy", "download", "en_core_web_sm"
            ], check=True, capture_output=True)
            logger.info("✓ spaCy en_core_web_sm model downloaded")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to download NLP models: {e}")
        return False

def create_directory_structure():
    """Create necessary directory structure."""
    logger = logging.getLogger(__name__)
    
    directories = [
        'data',
        'data/documents',
        'data/cache',
        'data/analysis',
        'data/models',
        'logs',
        'config',
        'static/css',
        'static/js',
        'static/images'
    ]
    
    for directory in directories:
        dir_path = PROJECT_ROOT / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Directory created/verified: {directory}")

def validate_configuration(config_path: Path):
    """Validate configuration file."""
    logger = logging.getLogger(__name__)
    
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        required_keys = ['app_name', 'server', 'database', 'storage']
        for key in required_keys:
            if key not in config:
                logger.error(f"Missing required configuration key: {key}")
                return False
        
        logger.info(f"✓ Configuration file validated: {config_path}")
        return True
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in configuration file: {e}")
        return False

def cleanup_old_processes():
    """Clean up any old processes that might be running."""
    logger = logging.getLogger(__name__)
    
    try:
        # This is a simple approach - in production you might want more sophisticated process management
        import psutil
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info['cmdline']
                if cmdline and any('local_mirror' in cmd for cmd in cmdline):
                    if proc.info['pid'] != os.getpid():
                        logger.info(f"Terminating old process: {proc.info['pid']}")
                        proc.terminate()
                        proc.wait(timeout=5)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
    
    except ImportError:
        logger.info("psutil not available, skipping process cleanup")

def start_application(host: str, port: int, config_path: Path, debug: bool, reload: bool, workers: int):
    """Start the FastAPI application using uvicorn."""
    logger = logging.getLogger(__name__)
    
    # Import the application factory
    try:
        from src.api.main_app import create_app
        app = create_app(str(config_path))
        logger.info("✓ Application imported successfully")
    except ImportError as e:
        logger.error(f"Failed to import application: {e}")
        return False
    
    # Configure uvicorn
    import uvicorn
    
    uvicorn_config = {
        "app": app,
        "host": host,
        "port": port,
        "reload": reload,
        "debug": debug,
        "access_log": True,
        "log_level": "debug" if debug else "info"
    }
    
    if not reload and workers > 1:
        uvicorn_config["workers"] = workers
    
    logger.info(f"Starting Enhanced Stock Analysis - Local Mirror")
    logger.info(f"Server will be available at: http://{host}:{port}")
    logger.info(f"API documentation: http://{host}:{port}/api/docs")
    logger.info(f"Configuration: {config_path}")
    
    try:
        # Run the server
        uvicorn.run(**uvicorn_config)
        return True
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        return True
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        return False

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger = logging.getLogger(__name__)
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    sys.exit(0)

def main():
    """Main startup function."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Enhanced Stock Analysis - Local Mirror Startup Script"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--skip-checks", action="store_true", help="Skip dependency checks")
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(args.debug)
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("=" * 70)
    logger.info("Enhanced Stock Analysis - Local Mirror Starting Up")
    logger.info("=" * 70)
    
    # Determine configuration file path
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = PROJECT_ROOT / "config" / "local_config.json"
    
    try:
        # Pre-flight checks
        logger.info("Running pre-flight checks...")
        
        # Create directory structure
        create_directory_structure()
        
        # Check dependencies
        if not args.skip_checks:
            logger.info("Checking dependencies...")
            if not check_dependencies():
                logger.error("Dependency check failed. Exiting.")
                sys.exit(1)
            
            # Download NLP models
            logger.info("Checking NLP models...")
            if not download_nlp_models():
                logger.error("NLP model download failed. Exiting.")
                sys.exit(1)
        else:
            logger.info("Skipping dependency checks (--skip-checks enabled)")
        
        # Validate configuration
        if not validate_configuration(config_path):
            logger.error("Configuration validation failed. Exiting.")
            sys.exit(1)
        
        # Clean up old processes
        cleanup_old_processes()
        
        logger.info("✓ All pre-flight checks passed")
        logger.info("-" * 70)
        
        # Start the application
        success = start_application(
            host=args.host,
            port=args.port,
            config_path=config_path,
            debug=args.debug,
            reload=args.reload,
            workers=args.workers
        )
        
        if success:
            logger.info("Application started successfully")
        else:
            logger.error("Application failed to start")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Startup interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error during startup: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()