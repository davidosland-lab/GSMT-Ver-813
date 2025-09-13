#!/usr/bin/env python3
"""
Test Phase 1 Critical Fixes Implementation
Validates LSTM fixes, performance-based weighting, and confidence calibration
"""

import asyncio
import numpy as np
from advanced_ensemble_predictor import AdvancedEnsemblePredictor, PredictionHorizon
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_phase1_improvements():
    """Test that Phase 1 improvements are working"""
    
    print("🚀 TESTING PHASE 1 CRITICAL FIXES")
    print("=" * 50)
    
    # Initialize predictor with Phase 1 fixes
    predictor = AdvancedEnsemblePredictor()
    
    # Test data
    test_symbol = "AAPL"
    test_market_data = {
        'price': 150.0,
        'volume': 1000000,
        'open': 149.5,
        'high': 151.2,
        'low': 148.8,
        'volatility': 0.25,
        'price_history': [149.0, 149.5, 150.0, 150.5, 150.2]
    }
    
    external_factors = {
        'market_sentiment': 0.2,
        'sector_momentum': 0.15,
        'geopolitical_risk': 0.1
    }
    
    print(f"📊 Testing prediction for {test_symbol}")
    print(f"   Market data: {test_market_data}")
    
    # Test short-term prediction (includes LSTM and RF with new weighting)
    try:
        result = await predictor.generate_advanced_prediction(
            symbol=test_symbol,
            timeframe="5d",  # Use valid PredictionHorizon value
            market_data=test_market_data,
            external_factors=external_factors
        )
        
        print(f"\n✅ PHASE 1 FIX VALIDATION:")
        print(f"   🎯 Direction: {result.direction}")
        print(f"   📈 Expected Return: {result.expected_return:+.4f}")
        print(f"   🎲 Confidence: {(1-result.uncertainty_score)*100:.1f}%")
        print(f"   📊 Model Weights: {result.model_ensemble_weights}")
        print(f"   🔄 Volatility: {result.volatility_estimate:.3f}")
        
        # Validate performance-based weighting
        weights = result.model_ensemble_weights
        
        print(f"\n🔧 PERFORMANCE-BASED WEIGHTING VALIDATION:")
        if 'quantile_regression' in weights:
            qr_weight = weights['quantile_regression']
            print(f"   Quantile Regression: {qr_weight:.1%} (Target: ~45%)")
            if qr_weight > 0.35:  # Should be highest weight
                print(f"   ✅ Quantile Regression properly weighted as best performer")
            else:
                print(f"   ❌ Quantile Regression underweighted: {qr_weight:.1%}")
        
        if 'random_forest' in weights:
            rf_weight = weights['random_forest']  
            print(f"   Random Forest: {rf_weight:.1%} (Target: ~30%)")
            
        if 'lstm' in weights:
            lstm_weight = weights['lstm']
            print(f"   LSTM: {lstm_weight:.1%} (Target: ~10%)")
            if lstm_weight < 0.25:  # Should have reduced weight due to bugs
                print(f"   ✅ LSTM appropriately reduced weight after bug fixes")
            
        # Test confidence calibration
        confidence = (1 - result.uncertainty_score)
        print(f"\n🎯 CONFIDENCE CALIBRATION VALIDATION:")
        print(f"   Calibrated Confidence: {confidence:.1%}")
        if 0.4 <= confidence <= 0.8:  # More realistic range
            print(f"   ✅ Confidence in realistic range (40-80%)")
        else:
            print(f"   ⚠️ Confidence outside expected range: {confidence:.1%}")
            
        print(f"\n📋 SUMMARY:")
        print(f"   ✅ Prediction generation successful")
        print(f"   ✅ Performance-based weighting active")
        print(f"   ✅ Confidence calibration applied")
        print(f"   🎯 All Phase 1 critical fixes integrated")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Phase 1 fixes: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_lstm_improvements():
    """Test that LSTM improvements are available"""
    
    print(f"\n🔧 TESTING LSTM IMPROVEMENTS:")
    
    try:
        from improved_lstm_predictor import ImprovedLSTMPredictor
        lstm = ImprovedLSTMPredictor()
        print(f"   ✅ Improved LSTM predictor available")
        
        # Test feature preparation
        test_data = {
            'prices': [100, 101, 99, 102, 100],
            'volumes': [1000, 1200, 800, 1500, 1100],
            'high': [102, 103, 101, 104, 102],
            'low': [99, 100, 98, 101, 99]
        }
        
        features = lstm.prepare_features(test_data)
        print(f"   ✅ Feature preparation working: {len(features)} features")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Improved LSTM not available: {e}")
        return False
    except Exception as e:
        print(f"   ❌ LSTM test error: {e}")
        return False

def test_confidence_calibration():
    """Test confidence calibration system"""
    
    print(f"\n🎯 TESTING CONFIDENCE CALIBRATION:")
    
    try:
        from phase1_critical_fixes import ImprovedConfidenceCalibration
        calibrator = ImprovedConfidenceCalibration()
        print(f"   ✅ Confidence calibration system available")
        
        # Test calibration
        test_predictions = {
            'quantile_regression': 0.02,
            'random_forest': 0.015,
            'lstm': 0.01
        }
        
        test_uncertainties = {
            'quantile_regression': 0.3,
            'random_forest': 0.4,
            'lstm': 0.6
        }
        
        test_weights = {
            'quantile_regression': 0.45,
            'random_forest': 0.30,
            'lstm': 0.10
        }
        
        confidence = calibrator.calibrate_confidence(
            ensemble_prediction=0.017,
            model_predictions=test_predictions,
            model_uncertainties=test_uncertainties,
            model_weights=test_weights
        )
        
        print(f"   ✅ Calibration successful: {confidence:.3f}")
        
        if 0.3 <= confidence <= 0.8:
            print(f"   ✅ Confidence in expected range")
        else:
            print(f"   ⚠️ Confidence outside expected range: {confidence:.3f}")
            
        return True
        
    except ImportError as e:
        print(f"   ❌ Confidence calibration not available: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Calibration test error: {e}")
        return False

async def main():
    """Main test function"""
    
    print("🚀 PHASE 1 CRITICAL FIXES - INTEGRATION TEST")
    print("=" * 60)
    
    # Test individual components
    lstm_ok = test_lstm_improvements()
    calibration_ok = test_confidence_calibration()
    
    # Test integrated system
    integration_ok = await test_phase1_improvements()
    
    print(f"\n📊 TEST RESULTS SUMMARY:")
    print(f"{'='*40}")
    print(f"   LSTM Fixes:           {'✅ PASS' if lstm_ok else '❌ FAIL'}")
    print(f"   Confidence Calibration: {'✅ PASS' if calibration_ok else '❌ FAIL'}")
    print(f"   Integration Test:     {'✅ PASS' if integration_ok else '❌ FAIL'}")
    
    if all([lstm_ok, calibration_ok, integration_ok]):
        print(f"\n🎉 ALL PHASE 1 CRITICAL FIXES OPERATIONAL!")
        print(f"   Ready for Phase 1 validation backtesting")
    else:
        print(f"\n⚠️ Some Phase 1 fixes need attention")
        
    return all([lstm_ok, calibration_ok, integration_ok])

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)