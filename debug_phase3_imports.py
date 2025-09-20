#!/usr/bin/env python3
"""
Debug Phase 3 Import Issues
===========================
"""

import logging
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_import_issues():
    """Debug what's causing the import issues."""
    
    # Test P3_001 import with detailed error handling
    logger.info("🔍 Testing P3_001 import...")
    try:
        from phase3_multi_timeframe_architecture import MultiTimeframeArchitecture
        logger.info("✅ P3_001 import successful")
        
        # Try instantiation
        logger.info("🔍 Testing P3_001 instantiation...")
        architecture = MultiTimeframeArchitecture()
        logger.info("✅ P3_001 instantiation successful")
        
    except Exception as e:
        logger.error(f"❌ P3_001 error: {type(e).__name__}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
    
    # Test P3_002 import with detailed error handling
    logger.info("🔍 Testing P3_002 import...")
    try:
        from phase3_bayesian_ensemble_framework import BayesianEnsembleFramework
        logger.info("✅ P3_002 import successful")
        
        # Try instantiation
        logger.info("🔍 Testing P3_002 instantiation...")
        framework = BayesianEnsembleFramework()
        logger.info("✅ P3_002 instantiation successful")
        
    except Exception as e:
        logger.error(f"❌ P3_002 error: {type(e).__name__}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
    
    # Test P3_003 import with detailed error handling
    logger.info("🔍 Testing P3_003 import...")
    try:
        from phase3_market_regime_detection import MarketRegimeDetection
        logger.info("✅ P3_003 import successful")
        
        # Try instantiation
        logger.info("🔍 Testing P3_003 instantiation...")
        detector = MarketRegimeDetection()
        logger.info("✅ P3_003 instantiation successful")
        
    except Exception as e:
        logger.error(f"❌ P3_003 error: {type(e).__name__}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
    
    # Test P3_004 import with detailed error handling
    logger.info("🔍 Testing P3_004 import...")
    try:
        from phase3_realtime_performance_monitoring import RealtimePerformanceMonitor
        logger.info("✅ P3_004 import successful")
        
        # Try instantiation
        logger.info("🔍 Testing P3_004 instantiation...")
        monitor = RealtimePerformanceMonitor()
        logger.info("✅ P3_004 instantiation successful")
        
    except Exception as e:
        logger.error(f"❌ P3_004 error: {type(e).__name__}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    debug_import_issues()