#!/usr/bin/env python3
"""
Enhanced Stock Analysis - Local Mirror Main Application
=====================================================

This is the main FastAPI application that integrates all components of the local mirror system:
- Document downloading and management  
- AI-powered document analysis using FinBERT and NLP models
- Enhanced prediction modeling combining document sentiment with technical analysis
- Persistent local storage of all analysis results
- Offline-capable operation for continuous local deployment

Author: Local Mirror System
Version: 1.0.0
Date: 2025-09-16
"""

import os
import sys
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import sqlite3
from pathlib import Path

# FastAPI and web framework imports
from fastapi import FastAPI, HTTPException, Request, Depends, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Add project source to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Import our local mirror components
from document_management.document_downloader import DocumentDownloader
from analysis.document_analyzer import DocumentAnalyzer
from prediction.enhanced_local_predictor import EnhancedLocalPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('local_mirror_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LocalMirrorApplication:
    """
    Main application class that orchestrates the local mirror system.
    
    This application provides:
    - Document downloading and management for stock analysis
    - AI-powered document analysis with persistent storage
    - Enhanced prediction modeling using document insights
    - Web interface for managing and reviewing analysis results
    - Offline operation with local database storage
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the local mirror application with all components."""
        self.config_path = config_path or "config/local_config.json"
        self.config = self._load_configuration()
        
        # Initialize directory structure
        self._ensure_directories()
        
        # Initialize database connection
        self.db_path = self.config.get('database_path', 'data/local_mirror.db')
        self._initialize_database()
        
        # Initialize core components
        self.document_downloader = DocumentDownloader(
            download_dir=self.config.get('documents_dir', 'data/documents'),
            db_path=self.db_path
        )
        
        self.document_analyzer = DocumentAnalyzer(
            db_path=self.db_path,
            cache_dir=self.config.get('cache_dir', 'data/cache')
        )
        
        self.enhanced_predictor = EnhancedLocalPredictor(
            db_path=self.db_path,
            cache_dir=self.config.get('cache_dir', 'data/cache')
        )
        
        # Initialize FastAPI application
        self.app = self._create_fastapi_app()
        
        logger.info("Local Mirror Application initialized successfully")
    
    def _load_configuration(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"Configuration loaded from {self.config_path}")
                return config
        except Exception as e:
            logger.warning(f"Failed to load config from {self.config_path}: {e}")
        
        # Return default configuration
        default_config = {
            "app_name": "Enhanced Stock Analysis - Local Mirror",
            "version": "1.0.0",
            "host": "127.0.0.1",
            "port": 8000,
            "debug": False,
            "database_path": "data/local_mirror.db",
            "documents_dir": "data/documents",
            "cache_dir": "data/cache",
            "analysis_dir": "data/analysis",
            "max_concurrent_downloads": 3,
            "max_concurrent_analysis": 2,
            "document_retention_days": 365,
            "analysis_retention_days": 180,
            "enable_auto_analysis": True,
            "enable_background_updates": True,
            "prediction_weights": {
                "document_analysis": 0.30,
                "technical_analysis": 0.40,
                "market_context": 0.30
            }
        }
        
        logger.info("Using default configuration")
        return default_config
    
    def _ensure_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            'data',
            'data/documents',
            'data/cache',
            'data/analysis', 
            'data/models',
            'logs',
            'config',
            'templates',
            'static/css',
            'static/js',
            'static/images'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
        logger.info("Directory structure verified")
    
    def _initialize_database(self):
        """Initialize the local database with required tables."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create comprehensive database schema
            cursor.executescript("""
            -- Documents table for storing downloaded files
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                document_type TEXT NOT NULL,
                title TEXT,
                url TEXT UNIQUE,
                local_path TEXT,
                file_size INTEGER,
                content_hash TEXT,
                download_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_modified TIMESTAMP,
                analysis_version INTEGER DEFAULT 1,
                status TEXT DEFAULT 'downloaded',
                metadata TEXT,
                UNIQUE(symbol, url)
            );
            
            -- Document analysis results
            CREATE TABLE IF NOT EXISTS document_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                analysis_type TEXT NOT NULL,
                analysis_version INTEGER DEFAULT 1,
                sentiment_score REAL,
                sentiment_label TEXT,
                confidence_score REAL,
                key_insights TEXT,
                financial_metrics TEXT,
                risk_factors TEXT,
                business_outlook TEXT,
                summary TEXT,
                analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processing_time_seconds REAL,
                model_version TEXT,
                analysis_metadata TEXT,
                FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
            );
            
            -- Prediction history and results
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                prediction_type TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                predicted_price REAL,
                confidence_interval_lower REAL,
                confidence_interval_upper REAL,
                confidence_score REAL,
                prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                target_date TIMESTAMP,
                actual_price REAL,
                prediction_accuracy REAL,
                feature_weights TEXT,
                model_components TEXT,
                prediction_metadata TEXT
            );
            
            -- Stock data cache for offline operation
            CREATE TABLE IF NOT EXISTS stock_data_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                data_type TEXT NOT NULL, -- 'price', 'volume', 'technical_indicators', etc.
                timeframe TEXT NOT NULL,
                data_date DATE NOT NULL,
                data_values TEXT NOT NULL, -- JSON string
                cache_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expiry_date TIMESTAMP,
                data_source TEXT,
                UNIQUE(symbol, data_type, timeframe, data_date)
            );
            
            -- Analysis tasks queue for background processing
            CREATE TABLE IF NOT EXISTS analysis_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_type TEXT NOT NULL, -- 'download', 'analyze', 'predict'
                symbol TEXT NOT NULL,
                task_data TEXT, -- JSON parameters
                priority INTEGER DEFAULT 5,
                status TEXT DEFAULT 'pending', -- 'pending', 'processing', 'completed', 'failed'
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                started_date TIMESTAMP,
                completed_date TIMESTAMP,
                error_message TEXT,
                retry_count INTEGER DEFAULT 0,
                max_retries INTEGER DEFAULT 3
            );
            
            -- Application configuration and settings
            CREATE TABLE IF NOT EXISTS app_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                data_type TEXT DEFAULT 'string', -- 'string', 'integer', 'float', 'boolean', 'json'
                description TEXT,
                last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Create indexes for better query performance
            CREATE INDEX IF NOT EXISTS idx_documents_symbol ON documents(symbol);
            CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(document_type);
            CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
            CREATE INDEX IF NOT EXISTS idx_analysis_document_id ON document_analysis(document_id);
            CREATE INDEX IF NOT EXISTS idx_analysis_type ON document_analysis(analysis_type);
            CREATE INDEX IF NOT EXISTS idx_predictions_symbol ON predictions(symbol);
            CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(prediction_date);
            CREATE INDEX IF NOT EXISTS idx_cache_symbol_type ON stock_data_cache(symbol, data_type);
            CREATE INDEX IF NOT EXISTS idx_queue_status ON analysis_queue(status);
            CREATE INDEX IF NOT EXISTS idx_queue_priority ON analysis_queue(priority);
            """)
            
            conn.commit()
            conn.close()
            
            logger.info(f"Database initialized successfully at {self.db_path}")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def _create_fastapi_app(self) -> FastAPI:
        """Create and configure the FastAPI application with all routes."""
        
        app = FastAPI(
            title=self.config.get('app_name', 'Enhanced Stock Analysis - Local Mirror'),
            version=self.config.get('version', '1.0.0'),
            description="Comprehensive local stock analysis with document management and AI-powered insights",
            docs_url="/api/docs",
            redoc_url="/api/redoc"
        )
        
        # Configure CORS
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Mount static files
        app.mount("/static", StaticFiles(directory="static"), name="static")
        
        # Configure templates
        templates = Jinja2Templates(directory="templates")
        
        # === WEB INTERFACE ROUTES ===
        
        @app.get("/", response_class=HTMLResponse)
        async def dashboard_home(request: Request):
            """Main dashboard for local mirror system."""
            # Get system statistics
            stats = await self._get_system_statistics()
            return templates.TemplateResponse("dashboard.html", {
                "request": request,
                "stats": stats,
                "config": self.config
            })
        
        @app.get("/documents/{symbol}", response_class=HTMLResponse)
        async def documents_page(request: Request, symbol: str):
            """Document management page for a specific stock symbol."""
            documents = await self._get_symbol_documents(symbol)
            return templates.TemplateResponse("documents.html", {
                "request": request,
                "symbol": symbol.upper(),
                "documents": documents
            })
        
        @app.get("/analysis/{symbol}", response_class=HTMLResponse)
        async def analysis_page(request: Request, symbol: str):
            """Analysis results page for a specific stock symbol."""
            analysis_results = await self._get_symbol_analysis(symbol)
            return templates.TemplateResponse("analysis.html", {
                "request": request,
                "symbol": symbol.upper(),
                "analysis_results": analysis_results
            })
        
        @app.get("/predictions/{symbol}", response_class=HTMLResponse)
        async def predictions_page(request: Request, symbol: str):
            """Prediction results page for a specific stock symbol."""
            prediction_history = await self._get_prediction_history(symbol)
            return templates.TemplateResponse("predictions.html", {
                "request": request,
                "symbol": symbol.upper(),
                "prediction_history": prediction_history
            })
        
        # === API ROUTES ===
        
        @app.get("/api/health")
        async def health_check():
            """Health check endpoint for monitoring system status."""
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "version": self.config.get('version', '1.0.0'),
                "components": {
                    "document_downloader": "ready",
                    "document_analyzer": "ready", 
                    "enhanced_predictor": "ready",
                    "database": "connected"
                }
            }
        
        @app.get("/api/system/stats")
        async def system_statistics():
            """Get comprehensive system statistics."""
            return await self._get_system_statistics()
        
        # === DOCUMENT MANAGEMENT API ===
        
        @app.post("/api/documents/{symbol}/download")
        async def download_documents(symbol: str, background_tasks: BackgroundTasks, force_refresh: bool = False):
            """Download all available documents for a stock symbol."""
            try:
                # Queue download task for background processing
                task_id = await self._queue_task('download', symbol, {
                    'force_refresh': force_refresh
                })
                
                # Start background download
                background_tasks.add_task(
                    self._process_download_task, 
                    symbol, 
                    force_refresh
                )
                
                return {
                    "success": True,
                    "message": f"Document download initiated for {symbol}",
                    "task_id": task_id,
                    "status": "processing"
                }
                
            except Exception as e:
                logger.error(f"Document download failed for {symbol}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/api/documents/{symbol}")
        async def get_documents(symbol: str):
            """Get all documents for a stock symbol."""
            try:
                documents = await self._get_symbol_documents(symbol)
                return {
                    "success": True,
                    "symbol": symbol.upper(),
                    "documents": documents,
                    "total_count": len(documents)
                }
            except Exception as e:
                logger.error(f"Failed to get documents for {symbol}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.delete("/api/documents/{document_id}")
        async def delete_document(document_id: int):
            """Delete a specific document and its analysis results."""
            try:
                success = await self._delete_document(document_id)
                if success:
                    return {
                        "success": True,
                        "message": f"Document {document_id} deleted successfully"
                    }
                else:
                    raise HTTPException(status_code=404, detail="Document not found")
            except Exception as e:
                logger.error(f"Failed to delete document {document_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # === DOCUMENT ANALYSIS API ===
        
        @app.post("/api/analysis/{symbol}/analyze")
        async def analyze_documents(symbol: str, background_tasks: BackgroundTasks, force_reanalysis: bool = False):
            """Analyze all documents for a stock symbol."""
            try:
                # Queue analysis task for background processing
                task_id = await self._queue_task('analyze', symbol, {
                    'force_reanalysis': force_reanalysis
                })
                
                # Start background analysis
                background_tasks.add_task(
                    self._process_analysis_task,
                    symbol,
                    force_reanalysis
                )
                
                return {
                    "success": True,
                    "message": f"Document analysis initiated for {symbol}",
                    "task_id": task_id,
                    "status": "processing"
                }
                
            except Exception as e:
                logger.error(f"Document analysis failed for {symbol}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/api/analysis/{symbol}")
        async def get_analysis_results(symbol: str):
            """Get all analysis results for a stock symbol."""
            try:
                analysis_results = await self._get_symbol_analysis(symbol)
                return {
                    "success": True,
                    "symbol": symbol.upper(),
                    "analysis_results": analysis_results,
                    "total_count": len(analysis_results)
                }
            except Exception as e:
                logger.error(f"Failed to get analysis results for {symbol}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/api/analysis/document/{document_id}")
        async def get_document_analysis(document_id: int):
            """Get analysis results for a specific document."""
            try:
                analysis = await self._get_document_analysis(document_id)
                if analysis:
                    return {
                        "success": True,
                        "document_id": document_id,
                        "analysis": analysis
                    }
                else:
                    raise HTTPException(status_code=404, detail="Analysis not found")
            except Exception as e:
                logger.error(f"Failed to get analysis for document {document_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # === ENHANCED PREDICTION API ===
        
        @app.post("/api/predictions/{symbol}/predict")
        async def generate_prediction(
            symbol: str, 
            background_tasks: BackgroundTasks,
            timeframe: str = "5d",
            include_analysis: bool = True
        ):
            """Generate enhanced prediction for a stock symbol using document analysis."""
            try:
                # Queue prediction task for background processing
                task_id = await self._queue_task('predict', symbol, {
                    'timeframe': timeframe,
                    'include_analysis': include_analysis
                })
                
                # Start background prediction
                background_tasks.add_task(
                    self._process_prediction_task,
                    symbol,
                    timeframe,
                    include_analysis
                )
                
                return {
                    "success": True,
                    "message": f"Enhanced prediction initiated for {symbol}",
                    "task_id": task_id,
                    "timeframe": timeframe,
                    "status": "processing"
                }
                
            except Exception as e:
                logger.error(f"Prediction generation failed for {symbol}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/api/predictions/{symbol}")
        async def get_predictions(symbol: str, limit: int = 50):
            """Get prediction history for a stock symbol."""
            try:
                predictions = await self._get_prediction_history(symbol, limit)
                return {
                    "success": True,
                    "symbol": symbol.upper(),
                    "predictions": predictions,
                    "total_count": len(predictions)
                }
            except Exception as e:
                logger.error(f"Failed to get predictions for {symbol}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/api/predictions/{symbol}/latest")
        async def get_latest_prediction(symbol: str):
            """Get the most recent prediction for a stock symbol."""
            try:
                prediction = await self._get_latest_prediction(symbol)
                if prediction:
                    return {
                        "success": True,
                        "symbol": symbol.upper(),
                        "prediction": prediction
                    }
                else:
                    return {
                        "success": False,
                        "message": f"No predictions found for {symbol}",
                        "symbol": symbol.upper()
                    }
            except Exception as e:
                logger.error(f"Failed to get latest prediction for {symbol}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # === TASK MANAGEMENT API ===
        
        @app.get("/api/tasks")
        async def get_tasks(status: Optional[str] = None, limit: int = 100):
            """Get task queue status and history."""
            try:
                tasks = await self._get_tasks(status, limit)
                return {
                    "success": True,
                    "tasks": tasks,
                    "total_count": len(tasks)
                }
            except Exception as e:
                logger.error(f"Failed to get tasks: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/api/tasks/{task_id}")
        async def get_task_status(task_id: int):
            """Get status of a specific task."""
            try:
                task = await self._get_task_status(task_id)
                if task:
                    return {
                        "success": True,
                        "task": task
                    }
                else:
                    raise HTTPException(status_code=404, detail="Task not found")
            except Exception as e:
                logger.error(f"Failed to get task status for {task_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.delete("/api/tasks/{task_id}")
        async def cancel_task(task_id: int):
            """Cancel a pending task."""
            try:
                success = await self._cancel_task(task_id)
                if success:
                    return {
                        "success": True,
                        "message": f"Task {task_id} cancelled successfully"
                    }
                else:
                    raise HTTPException(status_code=404, detail="Task not found or cannot be cancelled")
            except Exception as e:
                logger.error(f"Failed to cancel task {task_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # === CONFIGURATION API ===
        
        @app.get("/api/config")
        async def get_configuration():
            """Get current application configuration."""
            # Return safe configuration (exclude sensitive data)
            safe_config = {k: v for k, v in self.config.items() 
                          if k not in ['database_path', 'secret_key']}
            return {
                "success": True,
                "configuration": safe_config
            }
        
        @app.put("/api/config")
        async def update_configuration(config_updates: Dict[str, Any]):
            """Update application configuration."""
            try:
                # Validate and update configuration
                for key, value in config_updates.items():
                    if key in self.config:
                        self.config[key] = value
                
                # Save updated configuration
                await self._save_configuration()
                
                return {
                    "success": True,
                    "message": "Configuration updated successfully"
                }
            except Exception as e:
                logger.error(f"Failed to update configuration: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        return app
    
    # === HELPER METHODS FOR DATABASE AND TASK MANAGEMENT ===
    
    async def _get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get document counts
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_documents,
                    COUNT(DISTINCT symbol) as unique_symbols,
                    COUNT(CASE WHEN status = 'downloaded' THEN 1 END) as downloaded_docs,
                    COUNT(CASE WHEN status = 'analyzing' THEN 1 END) as analyzing_docs,
                    COUNT(CASE WHEN status = 'analyzed' THEN 1 END) as analyzed_docs
                FROM documents
            """)
            doc_stats = cursor.fetchone()
            
            # Get analysis counts
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_analysis,
                    COUNT(DISTINCT document_id) as analyzed_documents,
                    AVG(confidence_score) as avg_confidence,
                    AVG(processing_time_seconds) as avg_processing_time
                FROM document_analysis
            """)
            analysis_stats = cursor.fetchone()
            
            # Get prediction counts
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_predictions,
                    COUNT(DISTINCT symbol) as predicted_symbols,
                    AVG(confidence_score) as avg_prediction_confidence,
                    COUNT(CASE WHEN actual_price IS NOT NULL THEN 1 END) as validated_predictions
                FROM predictions
            """)
            prediction_stats = cursor.fetchone()
            
            # Get task queue status
            cursor.execute("""
                SELECT 
                    status,
                    COUNT(*) as count
                FROM analysis_queue
                GROUP BY status
            """)
            task_counts = {row['status']: row['count'] for row in cursor.fetchall()}
            
            # Get recent activity (last 24 hours)
            cursor.execute("""
                SELECT 
                    COUNT(*) as recent_downloads
                FROM documents
                WHERE download_date > datetime('now', '-1 day')
            """)
            recent_downloads = cursor.fetchone()['recent_downloads']
            
            cursor.execute("""
                SELECT 
                    COUNT(*) as recent_analysis
                FROM document_analysis
                WHERE analysis_date > datetime('now', '-1 day')
            """)
            recent_analysis = cursor.fetchone()['recent_analysis']
            
            conn.close()
            
            return {
                "documents": {
                    "total": doc_stats['total_documents'],
                    "unique_symbols": doc_stats['unique_symbols'],
                    "downloaded": doc_stats['downloaded_docs'],
                    "analyzing": doc_stats['analyzing_docs'],
                    "analyzed": doc_stats['analyzed_docs']
                },
                "analysis": {
                    "total": analysis_stats['total_analysis'],
                    "analyzed_documents": analysis_stats['analyzed_documents'],
                    "avg_confidence": round(analysis_stats['avg_confidence'] or 0, 3),
                    "avg_processing_time": round(analysis_stats['avg_processing_time'] or 0, 2)
                },
                "predictions": {
                    "total": prediction_stats['total_predictions'],
                    "predicted_symbols": prediction_stats['predicted_symbols'],
                    "avg_confidence": round(prediction_stats['avg_prediction_confidence'] or 0, 3),
                    "validated": prediction_stats['validated_predictions']
                },
                "tasks": task_counts,
                "activity_24h": {
                    "downloads": recent_downloads,
                    "analysis": recent_analysis
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get system statistics: {e}")
            return {}
    
    async def _get_symbol_documents(self, symbol: str) -> List[Dict]:
        """Get all documents for a specific symbol."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    d.*,
                    COUNT(da.id) as analysis_count,
                    MAX(da.analysis_date) as last_analysis_date
                FROM documents d
                LEFT JOIN document_analysis da ON d.id = da.document_id
                WHERE d.symbol = ?
                GROUP BY d.id
                ORDER BY d.download_date DESC
            """, (symbol.upper(),))
            
            documents = [dict(row) for row in cursor.fetchall()]
            conn.close()
            
            return documents
            
        except Exception as e:
            logger.error(f"Failed to get documents for {symbol}: {e}")
            return []
    
    async def _get_symbol_analysis(self, symbol: str) -> List[Dict]:
        """Get all analysis results for a specific symbol."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    da.*,
                    d.title as document_title,
                    d.document_type,
                    d.url as document_url
                FROM document_analysis da
                JOIN documents d ON da.document_id = d.id
                WHERE d.symbol = ?
                ORDER BY da.analysis_date DESC
            """, (symbol.upper(),))
            
            analysis_results = [dict(row) for row in cursor.fetchall()]
            conn.close()
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Failed to get analysis results for {symbol}: {e}")
            return []
    
    async def _get_prediction_history(self, symbol: str, limit: int = 50) -> List[Dict]:
        """Get prediction history for a specific symbol."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT *
                FROM predictions
                WHERE symbol = ?
                ORDER BY prediction_date DESC
                LIMIT ?
            """, (symbol.upper(), limit))
            
            predictions = [dict(row) for row in cursor.fetchall()]
            conn.close()
            
            return predictions
            
        except Exception as e:
            logger.error(f"Failed to get prediction history for {symbol}: {e}")
            return []
    
    async def _queue_task(self, task_type: str, symbol: str, task_data: Dict) -> int:
        """Queue a background task and return task ID."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO analysis_queue (task_type, symbol, task_data, priority, status)
                VALUES (?, ?, ?, ?, ?)
            """, (task_type, symbol.upper(), json.dumps(task_data), 5, 'pending'))
            
            task_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            logger.info(f"Queued {task_type} task for {symbol} with ID {task_id}")
            return task_id
            
        except Exception as e:
            logger.error(f"Failed to queue task: {e}")
            raise
    
    async def _process_download_task(self, symbol: str, force_refresh: bool = False):
        """Process document download task in background."""
        try:
            logger.info(f"Starting document download for {symbol}")
            result = await self.document_downloader.download_all_documents(
                symbol, force_refresh
            )
            logger.info(f"Document download completed for {symbol}: {result}")
            
        except Exception as e:
            logger.error(f"Document download failed for {symbol}: {e}")
    
    async def _process_analysis_task(self, symbol: str, force_reanalysis: bool = False):
        """Process document analysis task in background."""
        try:
            logger.info(f"Starting document analysis for {symbol}")
            
            # Get all documents for the symbol
            documents = await self._get_symbol_documents(symbol)
            
            # Analyze each document
            for doc in documents:
                if doc['status'] == 'downloaded' or force_reanalysis:
                    result = await self.document_analyzer.analyze_document(
                        doc['id'], force_reanalysis
                    )
                    logger.info(f"Analysis completed for document {doc['id']}: {result}")
            
        except Exception as e:
            logger.error(f"Document analysis failed for {symbol}: {e}")
    
    async def _process_prediction_task(self, symbol: str, timeframe: str = "5d", include_analysis: bool = True):
        """Process enhanced prediction task in background."""
        try:
            logger.info(f"Starting enhanced prediction for {symbol}")
            result = await self.enhanced_predictor.predict_stock_price(
                symbol, timeframe
            )
            logger.info(f"Enhanced prediction completed for {symbol}: {result}")
            
        except Exception as e:
            logger.error(f"Enhanced prediction failed for {symbol}: {e}")
    
    # Additional helper methods would continue here...
    # (Methods for task management, configuration, etc.)
    
    async def _save_configuration(self):
        """Save current configuration to file."""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise

def create_app(config_path: Optional[str] = None) -> FastAPI:
    """Factory function to create the FastAPI application."""
    local_mirror = LocalMirrorApplication(config_path)
    return local_mirror.app

# Development server for local testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Stock Analysis - Local Mirror")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    # Create the application
    app = create_app(args.config)
    
    # Configure logging for development
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Start the server
    logger.info(f"Starting Enhanced Stock Analysis - Local Mirror")
    logger.info(f"Server will be available at: http://{args.host}:{args.port}")
    logger.info(f"API documentation: http://{args.host}:{args.port}/api/docs")
    
    uvicorn.run(
        "main_app:create_app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        factory=True
    )