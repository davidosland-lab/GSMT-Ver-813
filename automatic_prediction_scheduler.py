#!/usr/bin/env python3
"""
🤖 Automatic Prediction Scheduler - Continuous Self-Learning System
================================================================

Runs automatic predictions throughout trading periods to enable continuous learning:
- Market hours detection and scheduling
- Automatic predictions for key symbols
- Real-time outcome tracking
- Performance-based learning adaptation
- Portfolio-wide prediction cycles

Target: Continuous self-improvement through automated prediction cycles
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import pytz
import json
import os
import sqlite3
from collections import defaultdict, deque
import numpy as np

# Import Phase 3 components
from phase3_extended_unified_predictor import ExtendedUnifiedSuperPredictor, ExtendedConfig
from phase3_realtime_performance_monitoring import RealtimePerformanceMonitor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketStatus(Enum):
    """Market status enumeration."""
    OPEN = "open"
    CLOSED = "closed"
    PRE_MARKET = "pre_market"
    AFTER_HOURS = "after_hours"
    WEEKEND = "weekend"
    HOLIDAY = "holiday"

@dataclass
class PredictionSchedule:
    """Configuration for automatic prediction scheduling."""
    symbol: str
    market: str
    intervals: List[str] = field(default_factory=lambda: ['1h', '4h', '1d'])  # Prediction intervals
    priority: int = 1  # 1=high, 2=medium, 3=low
    enabled: bool = True
    last_prediction: Optional[datetime] = None
    consecutive_errors: int = 0
    performance_score: float = 0.7  # 0-1 score based on recent accuracy

@dataclass
class MarketSession:
    """Market session configuration."""
    market_name: str
    timezone: str
    open_time: str  # "09:30"
    close_time: str  # "16:00"
    pre_market_start: str  # "04:00"
    after_hours_end: str  # "20:00"
    trading_days: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])  # Mon-Fri

class AutomaticPredictionScheduler:
    """
    Automatic Prediction Scheduler for continuous self-learning.
    
    Features:
    - Market hours-aware scheduling
    - Multi-symbol prediction cycles
    - Performance-based adaptation
    - Learning outcome tracking
    - Error handling and recovery
    """
    
    def __init__(self, config: Dict = None):
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.config = config or {}
        self.db_path = self.config.get('scheduler_db', 'scheduler.db')
        self.max_concurrent_predictions = self.config.get('max_concurrent', 3)
        self.prediction_timeout = self.config.get('prediction_timeout', 30)  # seconds
        
        # Core components
        self.predictor = ExtendedUnifiedSuperPredictor()
        self.performance_monitor = RealtimePerformanceMonitor()
        
        # Scheduling state
        self.active_schedules: Dict[str, PredictionSchedule] = {}
        self.market_sessions = self._initialize_market_sessions()
        self.prediction_queue = asyncio.Queue()
        self.running = False
        self.worker_tasks: List[asyncio.Task] = []
        
        # Performance tracking
        self.success_count = 0
        self.error_count = 0
        self.prediction_history = deque(maxlen=1000)
        self.last_outcome_check = datetime.now()
        
        # Learning metrics
        self.learning_metrics = {
            'total_auto_predictions': 0,
            'successful_predictions': 0,
            'learning_improvements': 0,
            'avg_accuracy_trend': [],
            'model_adaptations': 0
        }
        
        # Initialize database and default schedules
        self._initialize_database()
        self._load_default_schedules()
        
        self.logger.info("🤖 Automatic Prediction Scheduler initialized")
    
    def _initialize_market_sessions(self) -> Dict[str, MarketSession]:
        """Initialize market trading sessions."""
        return {
            'ASX': MarketSession(
                market_name='ASX',
                timezone='Australia/Sydney',
                open_time='10:00',
                close_time='16:00',
                pre_market_start='07:00',
                after_hours_end='18:00'
            ),
            'NYSE': MarketSession(
                market_name='NYSE',
                timezone='US/Eastern',
                open_time='09:30',
                close_time='16:00',
                pre_market_start='04:00',
                after_hours_end='20:00'
            ),
            'NASDAQ': MarketSession(
                market_name='NASDAQ',
                timezone='US/Eastern',
                open_time='09:30',
                close_time='16:00',
                pre_market_start='04:00',
                after_hours_end='20:00'
            ),
            'LSE': MarketSession(
                market_name='LSE',
                timezone='Europe/London',
                open_time='08:00',
                close_time='16:30',
                pre_market_start='07:00',
                after_hours_end='17:30'
            )
        }
    
    def _initialize_database(self):
        """Initialize SQLite database for scheduler persistence."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Prediction schedules table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS prediction_schedules (
                        symbol TEXT PRIMARY KEY,
                        market TEXT NOT NULL,
                        intervals TEXT NOT NULL,
                        priority INTEGER DEFAULT 1,
                        enabled INTEGER DEFAULT 1,
                        last_prediction TEXT,
                        consecutive_errors INTEGER DEFAULT 0,
                        performance_score REAL DEFAULT 0.7,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Automatic predictions history
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS auto_predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        prediction_type TEXT NOT NULL,
                        predicted_value REAL NOT NULL,
                        confidence REAL NOT NULL,
                        timestamp TEXT NOT NULL,
                        market_status TEXT NOT NULL,
                        outcome_recorded INTEGER DEFAULT 0,
                        actual_value REAL,
                        accuracy_score REAL,
                        learning_impact REAL
                    )
                """)
                
                # Scheduler metrics
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS scheduler_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        metric_date TEXT NOT NULL,
                        total_predictions INTEGER DEFAULT 0,
                        successful_predictions INTEGER DEFAULT 0,
                        avg_accuracy REAL DEFAULT 0.0,
                        learning_improvements INTEGER DEFAULT 0,
                        market_coverage REAL DEFAULT 0.0,
                        uptime_percentage REAL DEFAULT 0.0
                    )
                """)
                
                conn.commit()
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
    
    def _load_default_schedules(self):
        """Load default prediction schedules for key symbols."""
        default_symbols = {
            # ASX - Australian market
            'CBA.AX': {'market': 'ASX', 'priority': 1, 'intervals': ['1h', '4h', '1d']},
            'BHP.AX': {'market': 'ASX', 'priority': 1, 'intervals': ['1h', '4h', '1d']}, 
            'WBC.AX': {'market': 'ASX', 'priority': 2, 'intervals': ['4h', '1d']},
            'ANZ.AX': {'market': 'ASX', 'priority': 2, 'intervals': ['4h', '1d']},
            '^AORD': {'market': 'ASX', 'priority': 1, 'intervals': ['1h', '4h', '1d']},
            
            # US Markets
            'AAPL': {'market': 'NASDAQ', 'priority': 1, 'intervals': ['1h', '4h', '1d']},
            'MSFT': {'market': 'NASDAQ', 'priority': 1, 'intervals': ['1h', '4h', '1d']},
            'GOOGL': {'market': 'NASDAQ', 'priority': 2, 'intervals': ['4h', '1d']},
            'TSLA': {'market': 'NASDAQ', 'priority': 2, 'intervals': ['4h', '1d']},
            '^GSPC': {'market': 'NYSE', 'priority': 1, 'intervals': ['1h', '4h', '1d']},
            '^DJI': {'market': 'NYSE', 'priority': 2, 'intervals': ['4h', '1d']},
            
            # UK Market
            '^FTSE': {'market': 'LSE', 'priority': 1, 'intervals': ['4h', '1d']},
        }
        
        for symbol, config in default_symbols.items():
            schedule = PredictionSchedule(
                symbol=symbol,
                market=config['market'],
                intervals=config['intervals'],
                priority=config['priority']
            )
            self.active_schedules[symbol] = schedule
        
        self.logger.info(f"📅 Loaded {len(default_symbols)} default prediction schedules")
    
    def get_market_status(self, market: str, current_time: datetime = None) -> MarketStatus:
        """Determine current market status."""
        if current_time is None:
            current_time = datetime.now()
        
        if market not in self.market_sessions:
            return MarketStatus.CLOSED
        
        session = self.market_sessions[market]
        market_tz = pytz.timezone(session.timezone)
        local_time = current_time.astimezone(market_tz)
        
        # Check if it's a weekend
        if local_time.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return MarketStatus.WEEKEND
        
        # Parse time strings
        open_time = datetime.strptime(session.open_time, '%H:%M').time()
        close_time = datetime.strptime(session.close_time, '%H:%M').time()
        pre_market_start = datetime.strptime(session.pre_market_start, '%H:%M').time()
        after_hours_end = datetime.strptime(session.after_hours_end, '%H:%M').time()
        
        current_time_only = local_time.time()
        
        # Determine status
        if pre_market_start <= current_time_only < open_time:
            return MarketStatus.PRE_MARKET
        elif open_time <= current_time_only < close_time:
            return MarketStatus.OPEN
        elif close_time <= current_time_only < after_hours_end:
            return MarketStatus.AFTER_HOURS
        else:
            return MarketStatus.CLOSED
    
    def should_make_prediction(self, schedule: PredictionSchedule) -> Tuple[bool, str]:
        """Determine if a prediction should be made for the given schedule."""
        if not schedule.enabled:
            return False, "Schedule disabled"
        
        # Check market status
        market_status = self.get_market_status(schedule.market)
        
        # Only predict during market hours or pre/after market for high priority
        if market_status == MarketStatus.WEEKEND:
            return False, "Weekend - no trading"
        
        if market_status == MarketStatus.CLOSED:
            return False, "Market closed"
        
        if market_status in [MarketStatus.PRE_MARKET, MarketStatus.AFTER_HOURS]:
            if schedule.priority > 1:  # Only high priority symbols
                return False, "Outside main trading hours for medium/low priority"
        
        # Check if enough time has passed since last prediction
        now = datetime.now()
        if schedule.last_prediction:
            min_interval = self._get_min_interval_minutes(schedule.intervals)
            time_since_last = (now - schedule.last_prediction).total_seconds() / 60
            
            if time_since_last < min_interval:
                return False, f"Too soon - {min_interval - time_since_last:.1f}min remaining"
        
        # Check consecutive errors
        if schedule.consecutive_errors >= 5:
            return False, "Too many consecutive errors - temporary suspension"
        
        return True, f"Ready for prediction - Market: {market_status.value}"
    
    def _get_min_interval_minutes(self, intervals: List[str]) -> int:
        """Get minimum interval in minutes from interval list."""
        interval_map = {
            '15m': 15, '30m': 30, '1h': 60, '2h': 120, 
            '4h': 240, '1d': 1440, '1w': 10080
        }
        
        min_minutes = float('inf')
        for interval in intervals:
            if interval in interval_map:
                min_minutes = min(min_minutes, interval_map[interval])
        
        return int(min_minutes) if min_minutes != float('inf') else 60
    
    async def make_automatic_prediction(self, schedule: PredictionSchedule) -> Dict[str, Any]:
        """Make an automatic prediction for the given schedule."""
        try:
            start_time = datetime.now()
            
            # Determine best time horizon based on market status and intervals
            market_status = self.get_market_status(schedule.market)
            time_horizon = self._select_optimal_horizon(schedule.intervals, market_status)
            
            self.logger.info(f"🔮 Making automatic prediction: {schedule.symbol} ({time_horizon}) - Market: {market_status.value}")
            
            # Make prediction using Phase 3 Extended system
            prediction_result = await self.predictor.predict(
                symbol=schedule.symbol,
                time_horizon=time_horizon,
                include_all_domains=True,
                enable_rl_optimization=True,
                include_risk_management=True
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Extract key metrics from prediction
            predicted_return = prediction_result.expected_return
            confidence = prediction_result.confidence
            
            # Record prediction for learning
            await self._record_automatic_prediction(
                schedule=schedule,
                prediction_result=prediction_result,
                time_horizon=time_horizon,
                market_status=market_status,
                processing_time=processing_time
            )
            
            # Update schedule
            schedule.last_prediction = start_time
            schedule.consecutive_errors = 0
            
            # Update learning metrics
            self.learning_metrics['total_auto_predictions'] += 1
            self.success_count += 1
            
            self.logger.info(f"✅ Automatic prediction completed: {schedule.symbol} -> {predicted_return:+.2f}% (conf: {confidence:.1%}) in {processing_time:.1f}s")
            
            return {
                'success': True,
                'symbol': schedule.symbol,
                'prediction': predicted_return,
                'confidence': confidence,
                'time_horizon': time_horizon,
                'market_status': market_status.value,
                'processing_time': processing_time
            }
            
        except Exception as e:
            # Handle prediction error
            schedule.consecutive_errors += 1
            self.error_count += 1
            
            self.logger.error(f"❌ Automatic prediction failed for {schedule.symbol}: {e}")
            
            return {
                'success': False,
                'symbol': schedule.symbol,
                'error': str(e),
                'consecutive_errors': schedule.consecutive_errors
            }
    
    def _select_optimal_horizon(self, intervals: List[str], market_status: MarketStatus) -> str:
        """Select optimal prediction horizon based on market status and available intervals."""
        
        # During main trading hours, prefer shorter horizons for better learning
        if market_status == MarketStatus.OPEN:
            if '1h' in intervals:
                return '1h'
            elif '4h' in intervals:
                return '4h'
        
        # During pre/after market, prefer medium horizons
        elif market_status in [MarketStatus.PRE_MARKET, MarketStatus.AFTER_HOURS]:
            if '4h' in intervals:
                return '4h'
            elif '1d' in intervals:
                return '1d'
        
        # Default to daily predictions
        return '1d' if '1d' in intervals else intervals[0]
    
    async def _record_automatic_prediction(self, schedule: PredictionSchedule, 
                                         prediction_result: Any, time_horizon: str,
                                         market_status: MarketStatus, processing_time: float):
        """Record automatic prediction in database and performance monitor."""
        
        try:
            # Record in performance monitoring system
            await self.performance_monitor.record_prediction(
                model_name="auto_extended_unified",
                symbol=schedule.symbol,
                prediction=prediction_result.expected_return,
                confidence=prediction_result.confidence,
                regime=getattr(prediction_result, 'market_regime', 'unknown'),
                timeframe=time_horizon
            )
            
            # Record in scheduler database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO auto_predictions 
                    (symbol, prediction_type, predicted_value, confidence, timestamp, market_status)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    schedule.symbol,
                    time_horizon,
                    prediction_result.expected_return,
                    prediction_result.confidence,
                    datetime.now().isoformat(),
                    market_status.value
                ))
                conn.commit()
            
            # Add to recent history for analysis
            self.prediction_history.append({
                'symbol': schedule.symbol,
                'prediction': prediction_result.expected_return,
                'confidence': prediction_result.confidence,
                'timestamp': datetime.now(),
                'market_status': market_status.value,
                'processing_time': processing_time
            })
            
        except Exception as e:
            self.logger.warning(f"Failed to record automatic prediction: {e}")
    
    async def check_prediction_outcomes(self):
        """Check outcomes of previous predictions and record for learning."""
        
        try:
            # This would be run periodically (e.g., every hour) to check if
            # enough time has passed to evaluate predictions
            
            now = datetime.now()
            cutoff_time = now - timedelta(hours=24)  # Check predictions from last 24h
            
            # In a real implementation, this would:
            # 1. Fetch actual market data for symbols with pending predictions
            # 2. Calculate actual returns vs predicted returns
            # 3. Record outcomes in performance monitoring system
            # 4. Trigger RL weight updates based on performance
            
            self.logger.info("🔍 Checking automatic prediction outcomes...")
            
            # Mock implementation - in reality would fetch real market data
            outcomes_checked = 0
            
            # Here you would integrate with your market data service
            # to fetch actual price changes and compare with predictions
            
            self.last_outcome_check = now
            
            if outcomes_checked > 0:
                self.logger.info(f"📊 Processed {outcomes_checked} prediction outcomes for learning")
            
        except Exception as e:
            self.logger.error(f"Error checking prediction outcomes: {e}")
    
    async def prediction_worker(self, worker_id: int):
        """Worker task that processes prediction queue."""
        
        self.logger.info(f"🏃 Prediction worker {worker_id} started")
        
        while self.running:
            try:
                # Get next prediction task from queue
                schedule = await asyncio.wait_for(
                    self.prediction_queue.get(), 
                    timeout=10.0
                )
                
                # Make the prediction
                result = await self.make_automatic_prediction(schedule)
                
                # Mark task as done
                self.prediction_queue.task_done()
                
            except asyncio.TimeoutError:
                # No tasks in queue, continue
                continue
            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(5)
    
    async def scheduler_loop(self):
        """Main scheduler loop that evaluates and queues predictions."""
        
        self.logger.info("📅 Scheduler loop started")
        
        while self.running:
            try:
                scheduled_count = 0
                
                # Check each active schedule
                for symbol, schedule in self.active_schedules.items():
                    should_predict, reason = self.should_make_prediction(schedule)
                    
                    if should_predict:
                        # Add to prediction queue
                        await self.prediction_queue.put(schedule)
                        scheduled_count += 1
                        
                        self.logger.debug(f"📋 Queued prediction: {symbol} - {reason}")
                
                if scheduled_count > 0:
                    self.logger.info(f"📅 Scheduled {scheduled_count} predictions")
                
                # Check prediction outcomes periodically
                if (datetime.now() - self.last_outcome_check).total_seconds() > 3600:  # Every hour
                    await self.check_prediction_outcomes()
                
                # Sleep before next scheduling cycle
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Scheduler loop error: {e}")
                await asyncio.sleep(30)
    
    async def start(self):
        """Start the automatic prediction scheduler."""
        
        if self.running:
            self.logger.warning("Scheduler already running")
            return
        
        self.running = True
        self.logger.info("🚀 Starting Automatic Prediction Scheduler")
        
        # Start worker tasks
        for i in range(self.max_concurrent_predictions):
            task = asyncio.create_task(self.prediction_worker(i))
            self.worker_tasks.append(task)
        
        # Start scheduler loop
        scheduler_task = asyncio.create_task(self.scheduler_loop())
        self.worker_tasks.append(scheduler_task)
        
        self.logger.info(f"✅ Scheduler started with {self.max_concurrent_predictions} workers")
    
    async def stop(self):
        """Stop the automatic prediction scheduler."""
        
        if not self.running:
            return
        
        self.logger.info("⏹️ Stopping Automatic Prediction Scheduler")
        self.running = False
        
        # Cancel all worker tasks
        for task in self.worker_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.worker_tasks:
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        self.worker_tasks.clear()
        self.logger.info("✅ Scheduler stopped")
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get current scheduler status and metrics."""
        
        total_predictions = self.success_count + self.error_count
        success_rate = (self.success_count / total_predictions * 100) if total_predictions > 0 else 0
        
        active_markets = set(schedule.market for schedule in self.active_schedules.values() if schedule.enabled)
        
        # Count predictions by market status
        recent_predictions = list(self.prediction_history)[-50:]  # Last 50 predictions
        status_counts = defaultdict(int)
        for pred in recent_predictions:
            status_counts[pred['market_status']] += 1
        
        return {
            'running': self.running,
            'active_schedules': len([s for s in self.active_schedules.values() if s.enabled]),
            'total_schedules': len(self.active_schedules),
            'active_markets': list(active_markets),
            'queue_size': self.prediction_queue.qsize(),
            'worker_count': len(self.worker_tasks),
            'performance': {
                'total_predictions': total_predictions,
                'successful_predictions': self.success_count,
                'error_count': self.error_count,
                'success_rate': success_rate,
                'recent_predictions': len(recent_predictions)
            },
            'learning_metrics': self.learning_metrics,
            'market_coverage': status_counts
        }
    
    def add_symbol_schedule(self, symbol: str, market: str, intervals: List[str], priority: int = 2):
        """Add a new symbol to automatic prediction schedule."""
        
        schedule = PredictionSchedule(
            symbol=symbol,
            market=market,
            intervals=intervals,
            priority=priority
        )
        
        self.active_schedules[symbol] = schedule
        self.logger.info(f"➕ Added prediction schedule: {symbol} ({market}) - Intervals: {intervals}")
        
        # Persist to database
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO prediction_schedules 
                    (symbol, market, intervals, priority, enabled)
                    VALUES (?, ?, ?, ?, 1)
                """, (symbol, market, json.dumps(intervals), priority))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to persist schedule for {symbol}: {e}")
    
    def remove_symbol_schedule(self, symbol: str):
        """Remove a symbol from automatic prediction schedule."""
        
        if symbol in self.active_schedules:
            del self.active_schedules[symbol]
            self.logger.info(f"➖ Removed prediction schedule: {symbol}")
            
            # Remove from database
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("DELETE FROM prediction_schedules WHERE symbol = ?", (symbol,))
                    conn.commit()
            except Exception as e:
                self.logger.error(f"Failed to remove schedule for {symbol}: {e}")


# Global scheduler instance
_scheduler_instance: Optional[AutomaticPredictionScheduler] = None

def get_scheduler() -> AutomaticPredictionScheduler:
    """Get or create the global scheduler instance."""
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = AutomaticPredictionScheduler()
    return _scheduler_instance

async def start_automatic_predictions():
    """Start the automatic prediction system."""
    scheduler = get_scheduler()
    await scheduler.start()

async def stop_automatic_predictions():
    """Stop the automatic prediction system."""
    scheduler = get_scheduler()
    await scheduler.stop()

if __name__ == "__main__":
    async def main():
        scheduler = AutomaticPredictionScheduler()
        
        try:
            await scheduler.start()
            
            # Run for demonstration
            print("🤖 Automatic Prediction Scheduler running...")
            print("📊 Status:", scheduler.get_scheduler_status())
            
            # Let it run for a bit
            await asyncio.sleep(300)  # 5 minutes
            
        finally:
            await scheduler.stop()
    
    asyncio.run(main())