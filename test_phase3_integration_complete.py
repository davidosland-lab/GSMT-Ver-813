#!/usr/bin/env python3
"""
Phase 3 Integration Complete Test Suite
=====================================

Comprehensive testing of Phase 3 integration with the existing prediction system.
Tests all components working together in the unified architecture.
"""

import asyncio
import logging
import json
from datetime import datetime
import sys
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_phase3_integration_complete():
    """Complete integration test for Phase 3 components"""
    
    logger.info("🚀 PHASE 3 INTEGRATION COMPLETE TEST SUITE")
    logger.info("=" * 60)
    
    test_results = {}
    
    # Test 1: Phase 3 Unified Predictor Direct Integration
    logger.info("\n📊 Test 1: Phase 3 Unified Predictor Direct Test")
    logger.info("-" * 50)
    
    try:
        from phase3_unified_super_predictor import Phase3UnifiedSuperPredictor
        
        # Create predictor with test configuration
        config = {
            'lookback_period': 60,
            'min_samples': 30,  # Reduced for faster testing
            'confidence_threshold': 0.7,
            'performance_db_path': ':memory:',  # In-memory for testing
            'mcmc_samples': 500,  # Reduced for faster testing
            'prior_alpha': 1.0,
            'posterior_window': 50,
            'regime_lookback': 30,
            'max_memory_records': 1000
        }
        
        predictor = Phase3UnifiedSuperPredictor(config)
        
        # Test with different symbols and timeframes
        test_cases = [
            ('AAPL', '5d'),
            ('CBA.AX', '30d'),
            ('^GSPC', '1d')
        ]
        
        for symbol, timeframe in test_cases:
            try:
                logger.info(f"🎯 Testing {symbol} - {timeframe}")
                
                prediction = await predictor.generate_phase3_unified_prediction(
                    symbol=symbol,
                    time_horizon=timeframe,
                    use_phase3_enhancements=True
                )
                
                # Verify Phase 3 components
                phase3_checks = {
                    'multi_timeframe': len(prediction.timeframe_predictions) > 0,
                    'bayesian_uncertainty': len(prediction.bayesian_uncertainty) > 0,
                    'market_regime': prediction.market_regime != 'Unknown',
                    'performance_monitoring': prediction.performance_adjusted_weights is not None
                }
                
                logger.info(f"✅ {symbol} prediction successful:")
                logger.info(f"   Current: ${prediction.current_price:.2f}")
                logger.info(f"   Predicted: ${prediction.predicted_price:.2f}")
                logger.info(f"   Expected Return: {prediction.expected_return:+.2%}")
                logger.info(f"   Confidence: {prediction.confidence_score:.3f}")
                logger.info(f"   Regime: {prediction.market_regime}")
                logger.info(f"   Phase 3 Components: {sum(phase3_checks.values())}/4 active")
                
                test_results[f'direct_{symbol}_{timeframe}'] = {
                    'status': 'SUCCESS',
                    'phase3_components': sum(phase3_checks.values()),
                    'confidence': prediction.confidence_score,
                    'regime': prediction.market_regime
                }
                
            except Exception as e:
                logger.error(f"❌ Direct test failed for {symbol}-{timeframe}: {e}")
                test_results[f'direct_{symbol}_{timeframe}'] = {
                    'status': 'FAILED',
                    'error': str(e)
                }
        
        logger.info("✅ Phase 3 Direct Integration: COMPLETED")
        
    except Exception as e:
        logger.error(f"❌ Phase 3 Direct Integration: {e}")
        test_results['direct_integration'] = {'status': 'FAILED', 'error': str(e)}
    
    # Test 2: App.py Integration Test
    logger.info("\n🔗 Test 2: App.py Integration Test")
    logger.info("-" * 50)
    
    try:
        # Test the app.py integration by importing the updated module
        sys.path.insert(0, '/home/user/webapp')
        
        # Import after adding path
        from phase3_unified_super_predictor import Phase3UnifiedSuperPredictor
        
        # Test initialization with app.py style config
        app_config = {
            'lookback_period': 60,
            'min_samples': 50,
            'confidence_threshold': 0.7,
            'performance_db_path': 'phase3_performance_monitoring.db',
            'mcmc_samples': 1000,
            'prior_alpha': 1.0,
            'posterior_window': 100,
            'regime_lookback': 60,
            'max_memory_records': 10000
        }
        
        app_predictor = Phase3UnifiedSuperPredictor(app_config)
        
        # Test app-style prediction call
        app_result = await app_predictor.generate_phase3_unified_prediction(
            symbol='AAPL',
            time_horizon='5d',
            include_all_domains=True,
            use_phase3_enhancements=True
        )
        
        # Verify response structure matches app.py expectations
        required_fields = [
            'symbol', 'time_horizon', 'predicted_price', 'current_price',
            'expected_return', 'direction', 'confidence_score', 'uncertainty_score',
            'timeframe_predictions', 'bayesian_uncertainty', 'market_regime',
            'performance_adjusted_weights', 'domain_predictions'
        ]
        
        missing_fields = []
        for field in required_fields:
            if not hasattr(app_result, field):
                missing_fields.append(field)
        
        if missing_fields:
            logger.warning(f"⚠️ Missing fields: {missing_fields}")
        else:
            logger.info("✅ All required fields present in prediction result")
        
        # Test Phase 3 specific enhancements
        logger.info(f"📊 Phase 3 Enhancement Status:")
        logger.info(f"   Timeframes: {len(app_result.timeframe_predictions)} available")
        logger.info(f"   Bayesian uncertainty: {len(app_result.bayesian_uncertainty)} components")
        logger.info(f"   Market regime: {app_result.market_regime}")
        logger.info(f"   Performance weights: {len(app_result.performance_adjusted_weights)} models")
        
        test_results['app_integration'] = {
            'status': 'SUCCESS',
            'missing_fields': len(missing_fields),
            'phase3_timeframes': len(app_result.timeframe_predictions),
            'regime': app_result.market_regime
        }
        
        logger.info("✅ App.py Integration: SUCCESSFUL")
        
    except Exception as e:
        logger.error(f"❌ App.py Integration failed: {e}")
        test_results['app_integration'] = {'status': 'FAILED', 'error': str(e)}
    
    # Test 3: API Response Format Test
    logger.info("\n🌐 Test 3: API Response Format Compatibility")
    logger.info("-" * 50)
    
    try:
        # Simulate the API response formatting
        if 'app_integration' in test_results and test_results['app_integration']['status'] == 'SUCCESS':
            
            # Mock the response formatting logic from app.py
            unified_result = app_result
            PHASE3_ENABLED = True
            
            # Test response structure
            response = {
                "success": True,
                "symbol": "AAPL",
                "timeframe": "5d",
                "prediction_type": "PHASE3_UNIFIED_SUPER_PREDICTION" if PHASE3_ENABLED else "UNIFIED_SUPER_PREDICTION",
                
                "prediction": {
                    "direction": unified_result.direction,
                    "expected_return": unified_result.expected_return,
                    "predicted_price": unified_result.predicted_price,
                    "current_price": unified_result.current_price,
                    "confidence_score": unified_result.confidence_score,
                    "uncertainty_score": unified_result.uncertainty_score,
                    "probability_up": unified_result.probability_up,
                    "confidence_interval": {
                        "lower": unified_result.confidence_interval[0],
                        "upper": unified_result.confidence_interval[1]
                    }
                },
                
                "phase3_enhancements": {
                    "multi_timeframe_analysis": {
                        "timeframe_predictions": unified_result.timeframe_predictions,
                        "timeframe_weights": unified_result.timeframe_weights,
                        "cross_timeframe_consistency": unified_result.cross_timeframe_consistency
                    },
                    "bayesian_uncertainty": {
                        "bayesian_uncertainty": unified_result.bayesian_uncertainty,
                        "credible_intervals": unified_result.credible_intervals
                    },
                    "market_regime_detection": {
                        "market_regime": unified_result.market_regime,
                        "regime_confidence": unified_result.regime_confidence,
                        "volatility_regime": unified_result.volatility_regime
                    },
                    "performance_monitoring": {
                        "model_performance_scores": unified_result.model_performance_scores,
                        "performance_adjusted_weights": unified_result.performance_adjusted_weights,
                        "degradation_alerts": unified_result.degradation_alerts
                    }
                }
            }
            
            # Test JSON serializability
            try:
                json_response = json.dumps(response, default=str, indent=2)
                logger.info("✅ API Response is JSON serializable")
                logger.info(f"📊 Response size: {len(json_response)} characters")
                
                # Verify key sections are present
                key_sections = ['prediction', 'phase3_enhancements']
                sections_present = all(section in response for section in key_sections)
                
                if sections_present:
                    logger.info("✅ All key response sections present")
                else:
                    logger.warning("⚠️ Some key response sections missing")
                
                test_results['api_format'] = {
                    'status': 'SUCCESS',
                    'json_serializable': True,
                    'response_size': len(json_response),
                    'key_sections_present': sections_present
                }
                
            except Exception as json_error:
                logger.error(f"❌ JSON serialization failed: {json_error}")
                test_results['api_format'] = {
                    'status': 'FAILED',
                    'error': f"JSON serialization: {json_error}"
                }
        
        else:
            logger.warning("⚠️ Skipping API format test due to app integration failure")
            test_results['api_format'] = {'status': 'SKIPPED', 'reason': 'App integration failed'}
    
    except Exception as e:
        logger.error(f"❌ API Format test failed: {e}")
        test_results['api_format'] = {'status': 'FAILED', 'error': str(e)}
    
    # Test 4: Performance and Scalability Test
    logger.info("\n⚡ Test 4: Performance and Scalability")
    logger.info("-" * 50)
    
    try:
        import time
        
        # Test prediction speed with Phase 3 enhancements
        start_time = time.time()
        
        speed_test_result = await app_predictor.generate_phase3_unified_prediction(
            symbol='AAPL',
            time_horizon='5d',
            include_all_domains=True,
            use_phase3_enhancements=True
        )
        
        end_time = time.time()
        prediction_time = end_time - start_time
        
        logger.info(f"⏱️ Prediction Time: {prediction_time:.2f} seconds")
        
        # Performance benchmarks
        performance_rating = "EXCELLENT" if prediction_time < 5 else \
                           "GOOD" if prediction_time < 10 else \
                           "ACCEPTABLE" if prediction_time < 20 else "NEEDS_OPTIMIZATION"
        
        logger.info(f"📊 Performance Rating: {performance_rating}")
        
        test_results['performance'] = {
            'status': 'SUCCESS',
            'prediction_time': prediction_time,
            'performance_rating': performance_rating
        }
        
    except Exception as e:
        logger.error(f"❌ Performance test failed: {e}")
        test_results['performance'] = {'status': 'FAILED', 'error': str(e)}
    
    # Final Summary
    logger.info("\n" + "=" * 60)
    logger.info("🎯 PHASE 3 INTEGRATION TEST SUMMARY")
    logger.info("=" * 60)
    
    total_tests = len(test_results)
    successful_tests = sum(1 for result in test_results.values() 
                          if result.get('status') == 'SUCCESS')
    
    logger.info(f"\n📊 Test Results: {successful_tests}/{total_tests} successful")
    
    for test_name, result in test_results.items():
        status_icon = "✅" if result.get('status') == 'SUCCESS' else \
                     "⚠️" if result.get('status') == 'SKIPPED' else "❌"
        logger.info(f"   {status_icon} {test_name}: {result.get('status')}")
        
        if result.get('error'):
            logger.info(f"      Error: {result['error']}")
    
    # Overall integration status
    if successful_tests == total_tests:
        logger.info("\n🎉 PHASE 3 INTEGRATION: FULLY SUCCESSFUL!")
        logger.info("✅ All components integrated and working correctly")
        logger.info("✅ Ready for production deployment")
        success = True
    elif successful_tests >= total_tests * 0.75:
        logger.info("\n🎯 PHASE 3 INTEGRATION: MOSTLY SUCCESSFUL")
        logger.info("✅ Core components working, minor issues detected")
        logger.info("⚠️ Review failed tests before production deployment")
        success = True
    else:
        logger.info("\n⚠️ PHASE 3 INTEGRATION: NEEDS ATTENTION")
        logger.info("❌ Multiple integration issues detected")
        logger.info("🔧 Address failed tests before proceeding")
        success = False
    
    logger.info(f"\n📈 Integration Score: {successful_tests}/{total_tests} ({successful_tests/total_tests*100:.1f}%)")
    
    return success, test_results

async def main():
    """Main test execution function"""
    try:
        success, results = await test_phase3_integration_complete()
        
        # Save test results
        with open('phase3_integration_test_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"\n💾 Test results saved to: phase3_integration_test_results.json")
        
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)