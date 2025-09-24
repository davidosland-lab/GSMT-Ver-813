#!/usr/bin/env python3
"""
🎯 Phase 4 Accuracy Validation & Tracking System
===============================================

Real-time accuracy validation for Phase 4 prediction models:
- Live prediction accuracy tracking against market outcomes
- Historical accuracy analysis and trending
- Model performance comparison (Phase 3 vs Phase 4)
- GNN relationship validation
- Prediction confidence calibration analysis

Features:
- Real-time market data comparison
- Rolling accuracy windows (1d, 7d, 30d)
- Direction accuracy, price accuracy, confidence calibration
- Prediction outcome database with SQLite storage
- Performance degradation alerts
- Model accuracy benchmarking
"""

import sqlite3
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, asdict
import yfinance as yf
import asyncio
import json
from pathlib import Path
from enum import Enum
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionOutcome(Enum):
    """Possible outcomes for prediction validation."""
    CORRECT_DIRECTION = "correct_direction"
    WRONG_DIRECTION = "wrong_direction"
    ACCURATE_PRICE = "accurate_price"  # Within 5% of actual
    INACCURATE_PRICE = "inaccurate_price"
    PENDING = "pending"  # Not yet validated

@dataclass
class PredictionRecord:
    """Record of a Phase 4 prediction for accuracy tracking."""
    prediction_id: str
    timestamp: datetime
    symbol: str
    model_type: str  # 'phase4-gnn', 'phase4-multimodal', 'phase3-extended'
    predicted_price: float
    current_price: float
    confidence_score: float
    timeframe: str
    prediction_horizon_end: datetime
    
    # Actual outcome (filled when validation occurs)
    actual_price: Optional[float] = None
    direction_accuracy: Optional[bool] = None
    price_accuracy_percent: Optional[float] = None
    outcome: Optional[PredictionOutcome] = None
    validated_at: Optional[datetime] = None
    
    # Additional Phase 4 specific metrics
    node_importance: Optional[float] = None
    graph_centrality: Optional[float] = None
    sector_influence: Optional[float] = None
    systemic_risk_score: Optional[float] = None

class Phase4AccuracyTracker:
    """
    Comprehensive accuracy tracking and validation system for Phase 4 models.
    """
    
    def __init__(self, db_path: str = "data/phase4_accuracy.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Performance tracking
        self.accuracy_cache = {}
        self.last_cache_update = {}
        self.cache_ttl = 300  # 5 minutes
        
        logger.info("🎯 Phase 4 Accuracy Tracker initialized")
    
    def _init_database(self):
        """Initialize SQLite database for tracking predictions."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    prediction_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    symbol TEXT,
                    model_type TEXT,
                    predicted_price REAL,
                    current_price REAL,
                    confidence_score REAL,
                    timeframe TEXT,
                    prediction_horizon_end TEXT,
                    actual_price REAL,
                    direction_accuracy BOOLEAN,
                    price_accuracy_percent REAL,
                    outcome TEXT,
                    validated_at TEXT,
                    node_importance REAL,
                    graph_centrality REAL,
                    sector_influence REAL,
                    systemic_risk_score REAL
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_symbol_timestamp 
                ON predictions(symbol, timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_model_outcome 
                ON predictions(model_type, outcome)
            """)
    
    async def record_prediction(self, prediction_data: Dict[str, Any], 
                              model_type: str, timeframe: str = "5d") -> str:
        """
        Record a new Phase 4 prediction for future accuracy validation.
        
        Args:
            prediction_data: Raw prediction response from Phase 4 API
            model_type: Type of model ('phase4-gnn', 'phase4-multimodal', 'phase3-extended')
            timeframe: Prediction timeframe
            
        Returns:
            Prediction ID for tracking
        """
        try:
            # Generate unique prediction ID
            prediction_id = f"{model_type}_{prediction_data.get('symbol', 'unknown')}_{int(datetime.now().timestamp())}"
            
            # Parse prediction data based on model type
            if model_type.startswith('phase4'):
                # Phase 4 format
                symbol = prediction_data.get('symbol', '')
                predicted_price = prediction_data.get('predicted_price', 0)
                confidence_score = prediction_data.get('confidence_score', 0)
                current_price = predicted_price * 0.95  # Estimate if not available
                
                # Phase 4 specific metrics
                gnn_insights = prediction_data.get('gnn_insights', {})
                node_importance = gnn_insights.get('node_importance')
                graph_centrality = gnn_insights.get('graph_centrality')
                sector_influence = gnn_insights.get('sector_influence')
                systemic_risk_score = gnn_insights.get('systemic_risk_score')
                
            else:
                # Phase 3 format
                prediction = prediction_data.get('prediction', {})
                symbol = prediction_data.get('symbol', '')
                predicted_price = prediction.get('predicted_price', 0)
                current_price = prediction.get('current_price', 0)
                confidence_score = prediction.get('confidence_score', 0)
                
                # Phase 3 doesn't have GNN metrics
                node_importance = None
                graph_centrality = None
                sector_influence = None
                systemic_risk_score = None
            
            # Calculate prediction horizon end date
            now = datetime.now(timezone.utc)
            if timeframe == "5d":
                horizon_end = now + timedelta(days=5)
            elif timeframe == "1d":
                horizon_end = now + timedelta(days=1)
            elif timeframe == "30d":
                horizon_end = now + timedelta(days=30)
            else:
                horizon_end = now + timedelta(days=5)  # Default
            
            # Create prediction record
            record = PredictionRecord(
                prediction_id=prediction_id,
                timestamp=now,
                symbol=symbol,
                model_type=model_type,
                predicted_price=predicted_price,
                current_price=current_price,
                confidence_score=confidence_score,
                timeframe=timeframe,
                prediction_horizon_end=horizon_end,
                node_importance=node_importance,
                graph_centrality=graph_centrality,
                sector_influence=sector_influence,
                systemic_risk_score=systemic_risk_score
            )
            
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO predictions VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                """, (
                    record.prediction_id,
                    record.timestamp.isoformat(),
                    record.symbol,
                    record.model_type,
                    record.predicted_price,
                    record.current_price,
                    record.confidence_score,
                    record.timeframe,
                    record.prediction_horizon_end.isoformat(),
                    record.actual_price,
                    record.direction_accuracy,
                    record.price_accuracy_percent,
                    record.outcome.value if record.outcome else None,
                    record.validated_at.isoformat() if record.validated_at else None,
                    record.node_importance,
                    record.graph_centrality,
                    record.sector_influence,
                    record.systemic_risk_score
                ))
            
            logger.info(f"📝 Recorded {model_type} prediction {prediction_id} for {symbol}: ${predicted_price:.2f}")
            return prediction_id
            
        except Exception as e:
            logger.error(f"Error recording prediction: {e}")
            raise
    
    async def validate_pending_predictions(self) -> Dict[str, Any]:
        """
        Validate all pending predictions that have reached their horizon end date.
        
        Returns:
            Validation results and statistics
        """
        try:
            now = datetime.now(timezone.utc)
            validated_count = 0
            validation_results = {}
            
            # Get pending predictions past their horizon
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM predictions 
                    WHERE outcome IS NULL OR outcome = 'pending'
                    AND prediction_horizon_end <= ?
                """, (now.isoformat(),))
                
                pending_predictions = cursor.fetchall()
            
            for row in pending_predictions:
                try:
                    # Parse row data
                    prediction_id = row[0]
                    symbol = row[2]
                    predicted_price = row[4]
                    current_price = row[5]
                    confidence_score = row[6]
                    
                    # Get actual current price from market
                    actual_price = await self._get_current_market_price(symbol)
                    
                    if actual_price is not None:
                        # Calculate accuracy metrics
                        predicted_direction = "up" if predicted_price > current_price else "down"
                        actual_direction = "up" if actual_price > current_price else "down"
                        direction_accuracy = predicted_direction == actual_direction
                        
                        price_accuracy_percent = 1 - abs(predicted_price - actual_price) / actual_price
                        
                        # Determine outcome
                        if direction_accuracy and price_accuracy_percent > 0.95:  # Within 5%
                            outcome = PredictionOutcome.ACCURATE_PRICE
                        elif direction_accuracy:
                            outcome = PredictionOutcome.CORRECT_DIRECTION
                        else:
                            outcome = PredictionOutcome.WRONG_DIRECTION
                        
                        # Update database record
                        with sqlite3.connect(self.db_path) as conn:
                            conn.execute("""
                                UPDATE predictions SET
                                    actual_price = ?,
                                    direction_accuracy = ?,
                                    price_accuracy_percent = ?,
                                    outcome = ?,
                                    validated_at = ?
                                WHERE prediction_id = ?
                            """, (
                                actual_price,
                                direction_accuracy,
                                price_accuracy_percent,
                                outcome.value,
                                now.isoformat(),
                                prediction_id
                            ))
                        
                        validation_results[prediction_id] = {
                            'symbol': symbol,
                            'predicted_price': predicted_price,
                            'actual_price': actual_price,
                            'direction_accuracy': direction_accuracy,
                            'price_accuracy_percent': price_accuracy_percent,
                            'outcome': outcome.value
                        }
                        
                        validated_count += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to validate prediction {prediction_id}: {e}")
            
            # Clear accuracy cache to force refresh
            self.accuracy_cache.clear()
            
            logger.info(f"✅ Validated {validated_count} predictions")
            
            return {
                'validated_count': validated_count,
                'results': validation_results,
                'timestamp': now.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error validating predictions: {e}")
            raise
    
    async def get_accuracy_metrics(self, model_type: str = None, 
                                 days: int = 30, symbol: str = None) -> Dict[str, Any]:
        """
        Get comprehensive accuracy metrics for Phase 4 models.
        
        Args:
            model_type: Specific model to analyze ('phase4-gnn', 'phase3-extended', etc.)
            days: Number of days to analyze
            symbol: Specific symbol to analyze
            
        Returns:
            Comprehensive accuracy metrics
        """
        try:
            # Check cache first
            cache_key = f"{model_type}_{days}_{symbol}"
            if (cache_key in self.accuracy_cache and 
                cache_key in self.last_cache_update and
                (datetime.now() - self.last_cache_update[cache_key]).seconds < self.cache_ttl):
                return self.accuracy_cache[cache_key]
            
            # Build query conditions
            where_conditions = ["outcome IS NOT NULL", "outcome != 'pending'"]
            params = []
            
            if model_type:
                where_conditions.append("model_type = ?")
                params.append(model_type)
            
            if symbol:
                where_conditions.append("symbol = ?")
                params.append(symbol)
            
            if days > 0:
                cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
                where_conditions.append("timestamp >= ?")
                params.append(cutoff_date)
            
            where_clause = " AND ".join(where_conditions)
            
            # Get prediction data
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(f"""
                    SELECT * FROM predictions 
                    WHERE {where_clause}
                    ORDER BY timestamp DESC
                """, conn, params=params)
            
            if df.empty:
                return {
                    'total_predictions': 0,
                    'accuracy_metrics': {},
                    'message': 'No predictions found for the specified criteria'
                }
            
            # Calculate accuracy metrics
            total_predictions = len(df)
            direction_accuracy = df['direction_accuracy'].mean() if 'direction_accuracy' in df.columns else 0
            
            # Price accuracy (within 5% considered accurate)
            price_accurate = (df['price_accuracy_percent'] > 0.95).sum() if 'price_accuracy_percent' in df.columns else 0
            price_accuracy_rate = price_accurate / total_predictions if total_predictions > 0 else 0
            
            # Average price accuracy percentage
            avg_price_accuracy = df['price_accuracy_percent'].mean() if 'price_accuracy_percent' in df.columns else 0
            
            # Confidence calibration
            avg_confidence = df['confidence_score'].mean() if 'confidence_score' in df.columns else 0
            
            # Outcome distribution
            outcome_counts = df['outcome'].value_counts().to_dict() if 'outcome' in df.columns else {}
            
            # Model-specific metrics for Phase 4
            phase4_metrics = {}
            if model_type and model_type.startswith('phase4'):
                phase4_df = df[df['node_importance'].notna()]
                if not phase4_df.empty:
                    phase4_metrics = {
                        'avg_node_importance': phase4_df['node_importance'].mean(),
                        'avg_graph_centrality': phase4_df['graph_centrality'].mean(),
                        'avg_sector_influence': phase4_df['sector_influence'].mean(),
                        'avg_systemic_risk': phase4_df['systemic_risk_score'].mean(),
                        'gnn_predictions_count': len(phase4_df)
                    }
            
            # Recent performance trend (last 7 days vs previous 7 days)
            recent_trend = {}
            if days >= 14:
                now = datetime.now(timezone.utc)
                recent_cutoff = (now - timedelta(days=7)).isoformat()
                previous_cutoff = (now - timedelta(days=14)).isoformat()
                
                recent_df = df[df['timestamp'] >= recent_cutoff]
                previous_df = df[(df['timestamp'] >= previous_cutoff) & (df['timestamp'] < recent_cutoff)]
                
                if not recent_df.empty and not previous_df.empty:
                    recent_accuracy = recent_df['direction_accuracy'].mean()
                    previous_accuracy = previous_df['direction_accuracy'].mean()
                    
                    recent_trend = {
                        'recent_7d_accuracy': recent_accuracy,
                        'previous_7d_accuracy': previous_accuracy,
                        'trend_change': recent_accuracy - previous_accuracy,
                        'trend_direction': 'improving' if recent_accuracy > previous_accuracy else 'declining'
                    }
            
            metrics = {
                'total_predictions': total_predictions,
                'accuracy_metrics': {
                    'direction_accuracy': round(direction_accuracy, 4),
                    'direction_accuracy_percent': round(direction_accuracy * 100, 2),
                    'price_accuracy_rate': round(price_accuracy_rate, 4),
                    'price_accuracy_percent': round(price_accuracy_rate * 100, 2),
                    'average_price_accuracy': round(avg_price_accuracy, 4),
                    'average_confidence': round(avg_confidence, 4),
                    'confidence_calibration': 'well_calibrated' if abs(direction_accuracy - avg_confidence) < 0.1 else 'poorly_calibrated'
                },
                'outcome_distribution': outcome_counts,
                'phase4_specific': phase4_metrics,
                'performance_trend': recent_trend,
                'analysis_period': {
                    'days': days,
                    'model_type': model_type,
                    'symbol': symbol,
                    'generated_at': datetime.now(timezone.utc).isoformat()
                }
            }
            
            # Cache results
            self.accuracy_cache[cache_key] = metrics
            self.last_cache_update[cache_key] = datetime.now()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating accuracy metrics: {e}")
            raise
    
    async def compare_model_accuracy(self, symbols: List[str] = None, 
                                   days: int = 30) -> Dict[str, Any]:
        """
        Compare accuracy between Phase 3 and Phase 4 models.
        
        Args:
            symbols: List of symbols to analyze
            days: Analysis period in days
            
        Returns:
            Comparative accuracy analysis
        """
        try:
            models = ['phase3-extended', 'phase4-gnn', 'phase4-multimodal']
            comparison_results = {}
            
            for model in models:
                try:
                    metrics = await self.get_accuracy_metrics(model_type=model, days=days)
                    comparison_results[model] = metrics
                except Exception as e:
                    logger.warning(f"Could not get metrics for {model}: {e}")
                    comparison_results[model] = {'error': str(e)}
            
            # Calculate comparative insights
            valid_models = {k: v for k, v in comparison_results.items() 
                          if 'accuracy_metrics' in v and v['total_predictions'] > 0}
            
            insights = {}
            if len(valid_models) > 1:
                # Find best performing model
                best_direction_accuracy = max(valid_models.items(), 
                                            key=lambda x: x[1]['accuracy_metrics']['direction_accuracy'])
                
                best_price_accuracy = max(valid_models.items(),
                                        key=lambda x: x[1]['accuracy_metrics']['price_accuracy_rate'])
                
                insights = {
                    'best_direction_model': {
                        'model': best_direction_accuracy[0],
                        'accuracy': best_direction_accuracy[1]['accuracy_metrics']['direction_accuracy']
                    },
                    'best_price_model': {
                        'model': best_price_accuracy[0],
                        'accuracy': best_price_accuracy[1]['accuracy_metrics']['price_accuracy_rate']
                    },
                    'recommendation': self._generate_model_recommendation(valid_models)
                }
            
            return {
                'model_comparison': comparison_results,
                'comparative_insights': insights,
                'analysis_summary': {
                    'models_analyzed': list(comparison_results.keys()),
                    'analysis_period_days': days,
                    'generated_at': datetime.now(timezone.utc).isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error comparing model accuracy: {e}")
            raise
    
    def _generate_model_recommendation(self, valid_models: Dict[str, Any]) -> str:
        """Generate recommendation based on model performance comparison."""
        try:
            phase3_accuracy = 0
            phase4_accuracy = 0
            
            for model, metrics in valid_models.items():
                accuracy = metrics['accuracy_metrics']['direction_accuracy']
                if model.startswith('phase3'):
                    phase3_accuracy = max(phase3_accuracy, accuracy)
                elif model.startswith('phase4'):
                    phase4_accuracy = max(phase4_accuracy, accuracy)
            
            if phase4_accuracy > phase3_accuracy + 0.05:  # 5% better
                return "Phase 4 models show significant improvement - recommended for production"
            elif phase4_accuracy > phase3_accuracy:
                return "Phase 4 models show marginal improvement - suitable for experimental use"
            elif abs(phase4_accuracy - phase3_accuracy) < 0.02:  # Within 2%
                return "Both models perform similarly - Phase 3 recommended for stability"
            else:
                return "Phase 3 models currently outperform Phase 4 - stick with Phase 3"
                
        except Exception as e:
            return f"Could not generate recommendation: {e}"
    
    async def _get_current_market_price(self, symbol: str) -> Optional[float]:
        """Get current market price for accuracy validation."""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
            
            # Fallback to info
            info = ticker.info
            if 'currentPrice' in info:
                return float(info['currentPrice'])
            elif 'regularMarketPrice' in info:
                return float(info['regularMarketPrice'])
                
        except Exception as e:
            logger.warning(f"Could not fetch current price for {symbol}: {e}")
        
        return None

# Convenience functions for API integration
async def record_phase4_prediction(prediction_data: Dict[str, Any], 
                                 model_type: str, timeframe: str = "5d") -> str:
    """Record a Phase 4 prediction for accuracy tracking."""
    tracker = Phase4AccuracyTracker()
    return await tracker.record_prediction(prediction_data, model_type, timeframe)

async def get_phase4_accuracy_report(model_type: str = None, days: int = 30) -> Dict[str, Any]:
    """Get Phase 4 accuracy report."""
    tracker = Phase4AccuracyTracker()
    return await tracker.get_accuracy_metrics(model_type, days)

async def validate_all_pending_predictions() -> Dict[str, Any]:
    """Validate all pending predictions."""
    tracker = Phase4AccuracyTracker()
    return await tracker.validate_pending_predictions()

async def compare_all_model_accuracy(days: int = 30) -> Dict[str, Any]:
    """Compare accuracy between all available models."""
    tracker = Phase4AccuracyTracker()
    return await tracker.compare_model_accuracy(days=days)