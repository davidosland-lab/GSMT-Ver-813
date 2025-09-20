#!/usr/bin/env python3
"""
Local Mirror Database Schema and Initialization
==============================================

Creates and manages the SQLite database for the local deployment model.
This database stores documents, analysis results, and prediction history
to reduce prediction timeframes through persistent local storage.
"""

import sqlite3
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

class LocalMirrorDatabase:
    def __init__(self, db_path: str = "local_mirror.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._init_database()
    
    def _init_database(self):
        """Initialize database with comprehensive schema for local deployment."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON")
                
                # Documents table - stores all downloaded documents
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        document_type TEXT NOT NULL,
                        title TEXT,
                        url TEXT UNIQUE,
                        local_path TEXT,
                        content_hash TEXT UNIQUE,
                        file_size INTEGER,
                        download_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        analysis_version INTEGER DEFAULT 0,
                        is_active BOOLEAN DEFAULT TRUE,
                        metadata TEXT  -- JSON field for additional metadata
                    )
                """)
                
                # Document analysis results - stores AI analysis of documents
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS document_analysis (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        document_id INTEGER NOT NULL,
                        analysis_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        sentiment_score REAL,
                        sentiment_label TEXT,
                        confidence_score REAL,
                        key_insights TEXT,
                        financial_metrics TEXT,  -- JSON field
                        risk_factors TEXT,       -- JSON field
                        business_outlook TEXT,
                        summary TEXT,
                        entity_extraction TEXT,  -- JSON field for NER results
                        model_version TEXT,
                        processing_time_seconds REAL,
                        FOREIGN KEY (document_id) REFERENCES documents(id)
                    )
                """)
                
                # Enhanced predictions with local model results
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        prediction_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        current_price REAL NOT NULL,
                        predicted_price REAL NOT NULL,
                        confidence_score REAL,
                        direction TEXT,  -- UP/DOWN
                        expected_change_percent REAL,
                        probability_up REAL,
                        confidence_interval_lower REAL,
                        confidence_interval_upper REAL,
                        technical_score REAL,
                        document_sentiment_score REAL,
                        market_condition_score REAL,
                        document_count INTEGER DEFAULT 0,
                        feature_weights TEXT,  -- JSON field for model weights
                        model_components TEXT,  -- JSON field for ensemble details
                        model_version TEXT,
                        processing_time_seconds REAL,
                        supporting_evidence TEXT  -- JSON field
                    )
                """)
                
                # System monitoring and performance tracking
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS system_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        metric_type TEXT NOT NULL,  -- prediction, download, analysis
                        symbol TEXT,
                        operation_duration_seconds REAL,
                        memory_usage_mb REAL,
                        cpu_usage_percent REAL,
                        success BOOLEAN DEFAULT TRUE,
                        error_message TEXT,
                        additional_data TEXT  -- JSON field
                    )
                """)
                
                # Cache management for fast repeated operations
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS cache_entries (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        cache_key TEXT UNIQUE NOT NULL,
                        cache_value TEXT NOT NULL,  -- JSON serialized data
                        expiry_timestamp DATETIME,
                        created_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        access_count INTEGER DEFAULT 0,
                        last_access DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes for performance
                indexes = [
                    "CREATE INDEX IF NOT EXISTS idx_documents_symbol ON documents(symbol)",
                    "CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(document_type)",
                    "CREATE INDEX IF NOT EXISTS idx_documents_timestamp ON documents(download_timestamp)",
                    "CREATE INDEX IF NOT EXISTS idx_analysis_document ON document_analysis(document_id)",
                    "CREATE INDEX IF NOT EXISTS idx_analysis_timestamp ON document_analysis(analysis_timestamp)",
                    "CREATE INDEX IF NOT EXISTS idx_predictions_symbol ON predictions(symbol)",
                    "CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(prediction_timestamp)",
                    "CREATE INDEX IF NOT EXISTS idx_predictions_timeframe ON predictions(timeframe)",
                    "CREATE INDEX IF NOT EXISTS idx_metrics_type ON system_metrics(metric_type)",
                    "CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON system_metrics(timestamp)",
                    "CREATE INDEX IF NOT EXISTS idx_cache_key ON cache_entries(cache_key)",
                    "CREATE INDEX IF NOT EXISTS idx_cache_expiry ON cache_entries(expiry_timestamp)"
                ]
                
                for index in indexes:
                    conn.execute(index)
                
                conn.commit()
                self.logger.info("✅ Local mirror database initialized successfully")
                
        except sqlite3.Error as e:
            self.logger.error(f"Database initialization failed: {e}")
            raise
    
    def store_document(self, symbol: str, document_type: str, title: str, 
                      url: str, local_path: str, content_hash: str, 
                      file_size: int, metadata: Dict = None) -> int:
        """Store document metadata in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    INSERT INTO documents 
                    (symbol, document_type, title, url, local_path, content_hash, file_size, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (symbol, document_type, title, url, local_path, content_hash, 
                     file_size, json.dumps(metadata or {})))
                
                doc_id = cursor.lastrowid
                conn.commit()
                self.logger.info(f"Document stored: {symbol} - {title} (ID: {doc_id})")
                return doc_id
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to store document: {e}")
            return None
    
    def store_analysis_result(self, document_id: int, analysis_result: Dict) -> int:
        """Store document analysis results."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    INSERT INTO document_analysis 
                    (document_id, sentiment_score, sentiment_label, confidence_score,
                     key_insights, financial_metrics, risk_factors, business_outlook,
                     summary, entity_extraction, model_version, processing_time_seconds)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    document_id,
                    analysis_result.get('sentiment_score'),
                    analysis_result.get('sentiment_label'),
                    analysis_result.get('confidence_score'),
                    analysis_result.get('key_insights'),
                    json.dumps(analysis_result.get('financial_metrics', {})),
                    json.dumps(analysis_result.get('risk_factors', [])),
                    analysis_result.get('business_outlook'),
                    analysis_result.get('summary'),
                    json.dumps(analysis_result.get('entities', [])),
                    analysis_result.get('model_version'),
                    analysis_result.get('processing_time', 0)
                ))
                
                analysis_id = cursor.lastrowid
                conn.commit()
                self.logger.info(f"Analysis result stored for document {document_id} (ID: {analysis_id})")
                return analysis_id
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to store analysis result: {e}")
            return None
    
    def store_prediction(self, prediction_data: Dict) -> int:
        """Store enhanced prediction results."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    INSERT INTO predictions 
                    (symbol, timeframe, current_price, predicted_price, confidence_score,
                     direction, expected_change_percent, probability_up, 
                     confidence_interval_lower, confidence_interval_upper,
                     technical_score, document_sentiment_score, market_condition_score,
                     document_count, feature_weights, model_components, model_version,
                     processing_time_seconds, supporting_evidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    prediction_data['symbol'],
                    prediction_data['timeframe'],
                    prediction_data['current_price'],
                    prediction_data['predicted_price'],
                    prediction_data['confidence_score'],
                    prediction_data['direction'],
                    prediction_data['expected_change_percent'],
                    prediction_data['probability_up'],
                    prediction_data['confidence_interval_lower'],
                    prediction_data['confidence_interval_upper'],
                    prediction_data.get('technical_score', 0),
                    prediction_data.get('document_sentiment_score', 0),
                    prediction_data.get('market_condition_score', 0),
                    prediction_data.get('document_count', 0),
                    json.dumps(prediction_data.get('feature_weights', {})),
                    json.dumps(prediction_data.get('model_components', {})),
                    prediction_data.get('model_version'),
                    prediction_data.get('processing_time', 0),
                    json.dumps(prediction_data.get('supporting_evidence', {}))
                ))
                
                prediction_id = cursor.lastrowid
                conn.commit()
                self.logger.info(f"Prediction stored: {prediction_data['symbol']} (ID: {prediction_id})")
                return prediction_id
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to store prediction: {e}")
            return None
    
    def get_recent_analysis(self, symbol: str, days: int = 30) -> List[Dict]:
        """Get recent analysis results for a symbol."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT da.*, d.title, d.document_type, d.url
                    FROM document_analysis da
                    JOIN documents d ON da.document_id = d.id
                    WHERE d.symbol = ? AND da.analysis_timestamp >= ?
                    ORDER BY da.analysis_timestamp DESC
                """, (symbol, datetime.now() - timedelta(days=days)))
                
                columns = [desc[0] for desc in cursor.description]
                results = [dict(zip(columns, row)) for row in cursor.fetchall()]
                return results
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to get recent analysis: {e}")
            return []
    
    def get_prediction_history(self, symbol: str, days: int = 30) -> List[Dict]:
        """Get prediction history for a symbol."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM predictions
                    WHERE symbol = ? AND prediction_timestamp >= ?
                    ORDER BY prediction_timestamp DESC
                """, (symbol, datetime.now() - timedelta(days=days)))
                
                columns = [desc[0] for desc in cursor.description]
                results = [dict(zip(columns, row)) for row in cursor.fetchall()]
                return results
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to get prediction history: {e}")
            return []
    
    def cleanup_old_data(self, days: int = 365):
        """Clean up old data to maintain performance."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cutoff_date = datetime.now() - timedelta(days=days)
                
                # Clean old cache entries
                conn.execute("DELETE FROM cache_entries WHERE expiry_timestamp < ?", (datetime.now(),))
                
                # Clean old system metrics
                conn.execute("DELETE FROM system_metrics WHERE timestamp < ?", (cutoff_date,))
                
                # Clean old predictions (keep recent ones)
                conn.execute("DELETE FROM predictions WHERE prediction_timestamp < ?", (cutoff_date,))
                
                conn.commit()
                self.logger.info(f"Cleaned up data older than {days} days")
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to cleanup old data: {e}")
    
    def get_database_stats(self) -> Dict:
        """Get database statistics for monitoring."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                stats = {}
                
                # Count records in each table
                tables = ['documents', 'document_analysis', 'predictions', 'system_metrics', 'cache_entries']
                for table in tables:
                    cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                    stats[f"{table}_count"] = cursor.fetchone()[0]
                
                # Database file size
                db_file = Path(self.db_path)
                if db_file.exists():
                    stats['database_size_mb'] = db_file.stat().st_size / (1024 * 1024)
                
                return stats
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to get database stats: {e}")
            return {}

def initialize_local_mirror_database(config_path: str = "local_deployment_config.json"):
    """Initialize the local mirror database from configuration."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        db_path = config['database']['path']
        db = LocalMirrorDatabase(db_path)
        
        print(f"✅ Local mirror database initialized at: {db_path}")
        print(f"📊 Database stats: {db.get_database_stats()}")
        
        return db
        
    except Exception as e:
        print(f"❌ Failed to initialize database: {e}")
        return None

if __name__ == "__main__":
    # Initialize database when run directly
    db = initialize_local_mirror_database()