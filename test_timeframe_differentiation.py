#!/usr/bin/env python3
"""
Test Timeframe Differentiation - Validate that different timeframes produce different results
"""

import asyncio
import json
from datetime import datetime
from optimized_prediction_system import OptimizedMarketPredictor, OptimizedPredictionRequest

async def test_timeframe_differentiation():
    """Test that different timeframes produce meaningfully different predictions"""
    
    print("🧪 Testing Timeframe Differentiation in Market Predictions")
    print("=" * 70)
    
    predictor = OptimizedMarketPredictor()
    
    # Test parameters
    test_symbol = "^AORD"
    timeframes = ["1d", "5d", "30d", "90d"]
    
    print(f"📊 Testing symbol: {test_symbol}")
    print(f"⏰ Timeframes: {', '.join(timeframes)}")
    print()
    
    # Store predictions for comparison
    predictions = {}
    
    # Generate predictions for each timeframe
    for timeframe in timeframes:
        print(f"🔍 Testing timeframe: {timeframe}")
        
        request = OptimizedPredictionRequest(
            symbol=test_symbol,
            timeframe=timeframe,
            include_factors=True,
            include_news_intelligence=True
        )
        
        try:
            start_time = datetime.now()
            response = await predictor.generate_fast_prediction(request)
            end_time = datetime.now()
            
            if response.success:
                prediction = response.prediction
                predictions[timeframe] = prediction
                
                processing_time = (end_time - start_time).total_seconds()
                
                print(f"  ✅ Success in {processing_time:.2f}s")
                print(f"  📈 Direction: {prediction['direction'].upper()}")
                print(f"  📊 Expected Change: {prediction['expected_change_percent']:+.2f}%")
                print(f"  🎯 Confidence: {prediction['confidence_score']:.1%}")
                print(f"  ⚠️  Risk Level: {prediction['risk_level']}")
                
                # Show timeframe-specific factors
                if 'market_factors' in prediction:
                    factors = prediction['market_factors']
                    print(f"  🔧 Timeframe Multiplier: {factors.get('timeframe_multiplier', 'N/A')}")
                    global_vol = factors.get('global_volatility', 'N/A')
                    if isinstance(global_vol, (int, float)):
                        print(f"  🌍 Global Volatility: {global_vol:.1f}%")
                    else:
                        print(f"  🌍 Global Volatility: {global_vol}")
                
                print()
                
            else:
                print(f"  ❌ Failed to generate prediction for {timeframe}")
                print()
                
        except Exception as e:
            print(f"  💥 Error generating prediction for {timeframe}: {e}")
            print()
    
    # Analyze differences between timeframes
    print("🔍 ANALYSIS: Timeframe Differentiation Results")
    print("=" * 50)
    
    if len(predictions) < 2:
        print("❌ Insufficient predictions generated for comparison")
        return
    
    # Compare key metrics across timeframes
    print("\n📊 Comparative Analysis:")
    print(f"{'Timeframe':<10} {'Direction':<10} {'Change%':<10} {'Confidence':<12} {'Risk':<8} {'Multiplier':<12}")
    print("-" * 70)
    
    changes = []
    confidences = []
    multipliers = []
    
    for tf in timeframes:
        if tf in predictions:
            pred = predictions[tf]
            change = pred.get('expected_change_percent', 0)
            confidence = pred.get('confidence_score', 0)
            direction = pred.get('direction', 'N/A')
            risk = pred.get('risk_level', 'N/A')
            multiplier = pred.get('market_factors', {}).get('timeframe_multiplier', 'N/A')
            
            changes.append(abs(change))
            confidences.append(confidence)
            if isinstance(multiplier, (int, float)):
                multipliers.append(multiplier)
            
            print(f"{tf:<10} {direction:<10} {change:+7.2f}% {confidence:>10.1%} {risk:<8} {multiplier!s:<12}")
    
    print()
    
    # Statistical analysis
    if len(changes) > 1:
        change_variance = max(changes) - min(changes)
        confidence_variance = max(confidences) - min(confidences)
        
        print(f"📈 Expected Change Variance: {change_variance:.2f}% (Range: {min(changes):.2f}% - {max(changes):.2f}%)")
        print(f"🎯 Confidence Variance: {confidence_variance:.1%} (Range: {min(confidences):.1%} - {max(confidences):.1%})")
        
        if multipliers and len(multipliers) > 1:
            multiplier_variance = max(multipliers) - min(multipliers)
            print(f"🔧 Timeframe Multiplier Variance: {multiplier_variance:.1f}x (Range: {min(multipliers):.1f}x - {max(multipliers):.1f}x)")
    
    # Validation checks
    print("\n✅ VALIDATION RESULTS:")
    
    success_criteria = []
    
    # Check 1: Different expected changes
    if len(changes) > 1 and change_variance > 0.5:  # At least 0.5% difference
        success_criteria.append("✅ Expected changes vary significantly across timeframes")
    else:
        success_criteria.append("❌ Expected changes are too similar across timeframes")
    
    # Check 2: Different confidence levels
    if len(confidences) > 1 and confidence_variance > 0.05:  # At least 5% difference
        success_criteria.append("✅ Confidence levels vary across timeframes")
    else:
        success_criteria.append("❌ Confidence levels are too similar across timeframes")
    
    # Check 3: Different timeframe multipliers
    if multipliers and len(set(multipliers)) > 1:
        success_criteria.append("✅ Timeframe multipliers are properly differentiated")
    else:
        success_criteria.append("❌ Timeframe multipliers are not differentiated")
    
    # Check 4: Geopolitical data integration
    geopolitical_detected = False
    for tf, pred in predictions.items():
        if 'geopolitical_assessment' in pred and pred['geopolitical_assessment']:
            geopolitical_detected = True
            break
    
    if geopolitical_detected:
        success_criteria.append("✅ Geopolitical events are being integrated")
    else:
        success_criteria.append("❌ No geopolitical event integration detected")
    
    # Print results
    for criterion in success_criteria:
        print(f"  {criterion}")
    
    # Overall assessment
    passed_checks = sum(1 for c in success_criteria if c.startswith('✅'))
    total_checks = len(success_criteria)
    
    print(f"\n🏆 OVERALL ASSESSMENT: {passed_checks}/{total_checks} checks passed")
    
    if passed_checks >= 3:
        print("🎉 SUCCESS: Timeframe differentiation is working properly!")
    elif passed_checks >= 2:
        print("⚠️  PARTIAL SUCCESS: Some timeframe differentiation detected, but improvements needed")
    else:
        print("❌ FAILURE: Timeframe differentiation is not working properly")
    
    # Show geopolitical assessment if available
    print("\n🌍 GEOPOLITICAL ASSESSMENT:")
    for tf, pred in predictions.items():
        if 'geopolitical_assessment' in pred and pred['geopolitical_assessment']:
            geo_assessment = pred['geopolitical_assessment']
            print(f"  {tf}: Global volatility {geo_assessment.get('global_volatility_score', 'N/A'):.1f}% ({geo_assessment.get('risk_level', 'unknown')})")
            if geo_assessment.get('active_conflicts', 0) > 0:
                print(f"      Active conflicts: {geo_assessment.get('active_conflicts', 0)}")
            break
    else:
        print("  No geopolitical assessment data available")
    
    return passed_checks >= 3

if __name__ == "__main__":
    success = asyncio.run(test_timeframe_differentiation())
    exit(0 if success else 1)