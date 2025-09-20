#!/usr/bin/env python3
"""
Local Deployment Startup Script
==============================

Starts the Enhanced Stock Analysis system in local deployment mode with:
- Reduced prediction timeframes (5-15s vs 30-60s)
- Local document analysis caching
- Enhanced prediction model with offline capability
- Background processing for continuous operation

This integrates with the existing system while providing the local deployment
enhancements that were previously implemented.
"""

import os
import sys
import json
import asyncio
import logging
import subprocess
from datetime import datetime
from pathlib import Path
import signal

# Import local deployment components
from local_mirror_database import initialize_local_mirror_database
from enhanced_local_predictor import EnhancedLocalPredictor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/local_deployment.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class LocalDeploymentManager:
    """Manages the local deployment model integration."""
    
    def __init__(self, config_path: str = "local_deployment_config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self.is_running = False
        self.processes = {}
        
        # Create necessary directories
        self._create_directories()
        
        # Initialize database
        self.db = initialize_local_mirror_database(config_path)
        
        logger.info("🚀 Local Deployment Manager initialized")
    
    def _load_config(self) -> dict:
        """Load local deployment configuration."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            sys.exit(1)
    
    def _create_directories(self):
        """Create necessary directory structure."""
        directories = [
            "data", "data/documents", "data/models", "data/cache", "data/analysis",
            "logs", "static/css", "static/js", "static/images"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        logger.info("📁 Directory structure created")
    
    async def start_local_deployment(self):
        """Start the local deployment model."""
        logger.info("=" * 60)
        logger.info("🚀 Starting Enhanced Stock Analysis - Local Deployment Mode")
        logger.info("=" * 60)
        
        try:
            # Step 1: Initialize enhanced predictor
            logger.info("1️⃣ Initializing Enhanced Local Predictor...")
            self.predictor = EnhancedLocalPredictor(self.config_path)
            
            # Step 2: Update existing app.py to use enhanced predictor
            await self._integrate_enhanced_predictor()
            
            # Step 3: Start the main application with enhanced mode
            await self._start_enhanced_application()
            
            # Step 4: Start background services
            await self._start_background_services()
            
            self.is_running = True
            logger.info("✅ Local deployment started successfully!")
            logger.info(f"🌐 Enhanced system available at: http://{self.config['server']['host']}:{self.config['server']['port']}")
            logger.info("📊 Performance monitoring enabled")
            logger.info("⚡ Reduced prediction timeframes active (5-15s)")
            
            # Keep running until stopped
            await self._monitor_system()
            
        except Exception as e:
            logger.error(f"❌ Failed to start local deployment: {e}")
            await self.stop_local_deployment()
            return False
        
        return True
    
    async def _integrate_enhanced_predictor(self):
        """Integrate enhanced predictor with existing app.py."""
        logger.info("🔗 Integrating enhanced predictor with existing system...")
        
        try:
            # Read the existing app.py
            with open('app.py', 'r') as f:
                app_content = f.read()
            
            # Check if enhanced predictor is already integrated
            if 'enhanced_local_predictor' in app_content.lower():
                logger.info("✅ Enhanced predictor already integrated")
                return
            
            # Add import for enhanced predictor at the top
            enhanced_import = """
# Enhanced Local Predictor Import (Local Deployment Mode)
try:
    from enhanced_local_predictor import enhanced_prediction_with_local_mirror
    ENHANCED_MODE_AVAILABLE = True
    print("🚀 Enhanced Local Predictor loaded - Reduced timeframes active")
except ImportError as e:
    print(f"⚠️ Enhanced Local Predictor not available: {e}")
    ENHANCED_MODE_AVAILABLE = False
"""
            
            # Find a good place to insert the import (after existing imports)
            import_insertion_point = app_content.find('from unified_super_predictor import UnifiedSuperPredictor')
            if import_insertion_point != -1:
                # Insert after the UnifiedSuperPredictor import
                next_newline = app_content.find('\n', import_insertion_point)
                app_content = (app_content[:next_newline + 1] + 
                             enhanced_import + 
                             app_content[next_newline + 1:])
            
            # Create backup
            backup_path = f'app_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.py'
            with open(backup_path, 'w') as f:
                f.write(app_content)
            
            logger.info(f"📋 Created backup: {backup_path}")
            
            # Write the enhanced version (we'll modify the predict endpoint later via API)
            with open('app_enhanced_integration.py', 'w') as f:
                f.write(app_content)
            
            logger.info("✅ Enhanced predictor integration prepared")
            
        except Exception as e:
            logger.error(f"Failed to integrate enhanced predictor: {e}")
            raise
    
    async def _start_enhanced_application(self):
        """Start the main FastAPI application with enhanced features."""
        logger.info("🌐 Starting enhanced FastAPI application...")
        
        try:
            # Use the existing supervisor configuration but with enhanced mode
            # The enhanced predictor will be called through API integration
            
            # Check if supervisor is already running
            supervisor_status = subprocess.run(
                ['supervisorctl', '-c', 'supervisord.conf', 'status'],
                capture_output=True, text=True
            )
            
            if supervisor_status.returncode == 0:
                logger.info("📊 Supervisor is running, restarting webapp...")
                subprocess.run([
                    'supervisorctl', '-c', 'supervisord.conf', 'restart', 'webapp'
                ], check=True)
            else:
                logger.info("🔧 Starting supervisor for enhanced mode...")
                subprocess.run([
                    'supervisord', '-c', 'supervisord.conf'
                ], check=True)
            
            # Wait a moment for startup
            await asyncio.sleep(3)
            
            logger.info("✅ Enhanced FastAPI application started")
            
        except Exception as e:
            logger.error(f"Failed to start enhanced application: {e}")
            raise
    
    async def _start_background_services(self):
        """Start background services for local deployment."""
        logger.info("🔄 Starting background services...")
        
        # Background service for database cleanup
        asyncio.create_task(self._background_database_cleanup())
        
        # Background service for performance monitoring
        asyncio.create_task(self._background_performance_monitoring())
        
        # Background service for cache management
        asyncio.create_task(self._background_cache_management())
        
        logger.info("✅ Background services started")
    
    async def _background_database_cleanup(self):
        """Background task for database maintenance."""
        while self.is_running:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                logger.info("🧹 Running database cleanup...")
                self.db.cleanup_old_data(days=365)
                
            except Exception as e:
                logger.error(f"Database cleanup error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _background_performance_monitoring(self):
        """Background task for performance monitoring."""
        while self.is_running:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Get performance metrics
                metrics = await self.predictor.get_performance_metrics()
                
                if metrics:
                    logger.info(f"📊 Performance: {metrics['total_predictions']} predictions, "
                              f"avg time: {metrics['avg_processing_time']}s, "
                              f"avg confidence: {metrics['avg_confidence']}")
                
                # Log database stats
                db_stats = self.db.get_database_stats()
                logger.info(f"💾 Database: {db_stats}")
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(600)  # Wait 10 minutes on error
    
    async def _background_cache_management(self):
        """Background task for cache management."""
        while self.is_running:
            try:
                await asyncio.sleep(1800)  # Run every 30 minutes
                
                logger.info("🗄️ Managing cache...")
                
                # Clean expired cache entries
                with sqlite3.connect(self.db.db_path) as conn:
                    cursor = conn.execute("DELETE FROM cache_entries WHERE expiry_timestamp < ?", 
                                        (datetime.now(),))
                    removed = cursor.rowcount
                    conn.commit()
                    
                if removed > 0:
                    logger.info(f"🧹 Removed {removed} expired cache entries")
                
            except Exception as e:
                logger.error(f"Cache management error: {e}")
                await asyncio.sleep(900)  # Wait 15 minutes on error
    
    async def _monitor_system(self):
        """Monitor system status and keep it running."""
        logger.info("👀 System monitoring active...")
        
        try:
            while self.is_running:
                await asyncio.sleep(60)  # Check every minute
                
                # Check if supervisor is still running
                supervisor_status = subprocess.run(
                    ['supervisorctl', '-c', 'supervisord.conf', 'status', 'webapp'],
                    capture_output=True, text=True
                )
                
                if supervisor_status.returncode != 0 or 'RUNNING' not in supervisor_status.stdout:
                    logger.warning("⚠️ Main application not running, attempting restart...")
                    subprocess.run([
                        'supervisorctl', '-c', 'supervisord.conf', 'restart', 'webapp'
                    ])
                
        except KeyboardInterrupt:
            logger.info("👋 Shutdown requested by user")
            await self.stop_local_deployment()
    
    async def stop_local_deployment(self):
        """Stop the local deployment model."""
        logger.info("🛑 Stopping Enhanced Stock Analysis - Local Deployment Mode...")
        
        self.is_running = False
        
        try:
            # Stop supervisor
            subprocess.run([
                'supervisorctl', '-c', 'supervisord.conf', 'stop', 'webapp'
            ], timeout=10)
            
            # Stop supervisord
            subprocess.run([
                'supervisorctl', '-c', 'supervisord.conf', 'shutdown'
            ], timeout=10)
            
            logger.info("✅ Local deployment stopped successfully")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

# Enhanced API integration function
async def add_enhanced_prediction_endpoint():
    """
    Add enhanced prediction endpoint to the existing FastAPI app.
    This can be called to integrate the enhanced predictor with the running app.
    """
    try:
        # This would typically be integrated directly into app.py
        # For now, we'll create a complementary endpoint
        
        from fastapi import FastAPI
        import uvicorn
        
        enhanced_app = FastAPI(title="Enhanced Local Predictor API")
        
        @enhanced_app.post("/api/enhanced-predict/{symbol}")
        async def enhanced_predict_endpoint(symbol: str, timeframe: str = "5d"):
            """Enhanced prediction endpoint with reduced timeframes."""
            try:
                from enhanced_local_predictor import enhanced_prediction_with_local_mirror
                result = await enhanced_prediction_with_local_mirror(symbol, timeframe)
                return result
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        @enhanced_app.get("/api/enhanced-performance")
        async def enhanced_performance_endpoint():
            """Get enhanced predictor performance metrics."""
            try:
                predictor = EnhancedLocalPredictor()
                metrics = await predictor.get_performance_metrics()
                return {"success": True, "metrics": metrics}
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        # Run on a different port to complement main app
        uvicorn.run(enhanced_app, host="127.0.0.1", port=8002)
        
    except Exception as e:
        logger.error(f"Failed to add enhanced endpoints: {e}")

# Signal handlers
def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}, initiating shutdown...")
    sys.exit(0)

def main():
    """Main function to start local deployment."""
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("🚀 Enhanced Stock Analysis - Local Deployment Mode")
    print("=" * 60)
    
    # Create deployment manager
    manager = LocalDeploymentManager()
    
    # Start local deployment
    try:
        asyncio.run(manager.start_local_deployment())
    except KeyboardInterrupt:
        print("\n👋 Shutdown requested")
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()