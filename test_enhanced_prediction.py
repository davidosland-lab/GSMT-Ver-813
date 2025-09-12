#!/usr/bin/env python3
"""
Test script for enhanced LLM prediction system with Tier 1 factors
Validates integration of superannuation flows, options analysis, and social sentiment
"""

import asyncio
import json
import time
from datetime import datetime
from market_prediction_llm import (
    prediction_service,
    PredictionRequest,
    PredictionTimeframe
)

async def test_enhanced_prediction():
    """Test the enhanced prediction system with Tier 1 factors"""
    
    print("🧠 Testing Enhanced LLM Prediction System with Tier 1 Factors")
    print("=" * 70)
    
    # Test configuration
    test_symbol = "^AORD"  # Australian All Ordinaries
    test_timeframes = ["1d", "5d", "30d"]
    
    for timeframe in test_timeframes:
        print(f"\n📊 Testing {test_symbol} prediction for {timeframe} timeframe...")
        
        try:
            # Create prediction request
            request = PredictionRequest(
                symbol=test_symbol,
                timeframe=timeframe,
                include_factors=True,
                include_news_intelligence=True,
                news_lookback_hours=48
            )
            
            # Measure prediction time
            start_time = time.time()
            
            # Generate enhanced prediction
            response = await prediction_service.get_market_prediction(request)
            
            prediction_time = time.time() - start_time
            
            # Validate response
            if response.success:
                print(f"✅ Prediction generated successfully in {prediction_time:.2f}s")
                
                # Display prediction results
                prediction = response.prediction
                print(f"   🎯 Direction: {prediction['direction'].upper()}")
                print(f"   📈 Expected Change: {prediction['expected_change_percent']:+.2f}%")
                print(f"   🎲 Confidence: {prediction['confidence_score']:.1%}")
                print(f"   ⚠️  Risk Level: {prediction['risk_level'].upper()}")
                
                # Display Tier 1 factors
                if response.tier1_factors:
                    print(f"\n   🎯 Tier 1 Factors ({len(response.tier1_factors)} factors):")
                    
                    # Group factors by category
                    super_factors = {k: v for k, v in response.tier1_factors.items() if k.startswith('super_')}
                    options_factors = {k: v for k, v in response.tier1_factors.items() if k.startswith('options_')}
                    social_factors = {k: v for k, v in response.tier1_factors.items() if k.startswith('social_')}
                    
                    if super_factors:
                        print("      🏦 Super Fund Flows:")
                        for factor, value in super_factors.items():
                            signal = "🟢" if value > 0.1 else "🔴" if value < -0.1 else "🟡"
                            print(f"         {signal} {factor.replace('super_', '').replace('_', ' ').title()}: {value:+.3f}")
                    
                    if options_factors:
                        print("      📊 Options Positioning:")
                        for factor, value in options_factors.items():
                            signal = "🟢" if value > 0.1 else "🔴" if value < -0.1 else "🟡"
                            print(f"         {signal} {factor.replace('options_', '').replace('_', ' ').title()}: {value:+.3f}")
                    
                    if social_factors:
                        print("      🗣️ Social Sentiment:")
                        for factor, value in social_factors.items():
                            signal = "🟢" if value > 0.1 else "🔴" if value < -0.1 else "🟡"
                            print(f"         {signal} {factor.replace('social_', '').replace('_', ' ').title()}: {value:+.3f}")
                
                # Display factor attribution
                if response.factor_attribution:
                    attribution = response.factor_attribution
                    overall = attribution.get('overall_signal', {})
                    
                    print(f"\n   📈 Factor Attribution Analysis:")
                    print(f"      Overall Direction: {overall.get('direction', 'unknown').upper()}")
                    print(f"      Signal Strength: {overall.get('signal_strength', 0):.3f}")
                    print(f"      Consensus: {attribution.get('factor_consensus', {}).get('consensus_strength', 0):.1%}")
                
                # Display model info
                model_info = response.model_info
                print(f"\n   🤖 Model Performance:")
                print(f"      Model Type: {model_info.get('model_type')}")
                print(f"      Version: {model_info.get('version')}")
                print(f"      Enhanced Accuracy: {model_info.get('accuracy_metrics', {}).get('enhanced_accuracy', 0):.1%}")
                print(f"      Tier 1 Contribution: {model_info.get('accuracy_metrics', {}).get('tier1_contribution', 'Unknown')}")
                
                # Display reasoning excerpt
                reasoning = prediction.get('reasoning', '')
                if len(reasoning) > 200:
                    reasoning = reasoning[:200] + "..."
                print(f"\n   💭 Key Reasoning: {reasoning}")
                
            else:
                print(f"❌ Prediction failed")
                
        except Exception as e:
            print(f"❌ Error testing {timeframe}: {e}")
    
    print(f"\n🎯 Enhanced Prediction System Test Complete")
    print("=" * 70)

async def test_factor_systems_individually():
    """Test each Tier 1 factor system individually"""
    
    print("\n🔧 Testing Individual Tier 1 Factor Systems")
    print("=" * 50)
    
    try:
        from super_fund_flow_analyzer import super_fund_analyzer
        print("\n🏦 Testing Super Fund Flow Analyzer...")
        super_factors = await super_fund_analyzer.get_market_prediction_factors()
        print(f"   ✅ Generated {len(super_factors)} super fund factors")
        for k, v in list(super_factors.items())[:3]:  # Show first 3
            print(f"      • {k}: {v:+.3f}")
        
    except Exception as e:
        print(f"   ❌ Super fund analyzer error: {e}")
    
    try:
        from asx_options_analyzer import asx_options_analyzer
        print("\n📊 Testing ASX Options Analyzer...")
        options_factors = await asx_options_analyzer.get_market_prediction_factors(['XJO', 'CBA'])
        print(f"   ✅ Generated {len(options_factors)} options factors")
        for k, v in list(options_factors.items())[:3]:  # Show first 3
            print(f"      • {k}: {v:+.3f}")
        
    except Exception as e:
        print(f"   ❌ Options analyzer error: {e}")
    
    try:
        from social_sentiment_tracker import social_sentiment_analyzer
        print("\n🗣️ Testing Social Sentiment Tracker...")
        social_factors = await social_sentiment_analyzer.get_market_prediction_factors(24)
        print(f"   ✅ Generated {len(social_factors)} social factors")
        for k, v in list(social_factors.items())[:3]:  # Show first 3
            print(f"      • {k}: {v:+.3f}")
        
    except Exception as e:
        print(f"   ❌ Social sentiment analyzer error: {e}")

if __name__ == "__main__":
    print("🚀 Starting Enhanced Prediction System Integration Test")
    print(f"⏰ Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run the comprehensive test
    asyncio.run(test_factor_systems_individually())
    asyncio.run(test_enhanced_prediction())
    
    print(f"\n✅ All tests completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")