#!/usr/bin/env python3
"""
Test script for the enhanced LLM prediction system with Tier 1 factors
"""

import asyncio
import json
import sys
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the enhanced prediction system
from market_prediction_llm import (
    MarketPredictionService,
    PredictionRequest,
    PredictionTimeframe
)

async def test_enhanced_prediction():
    """Test the enhanced prediction system with Tier 1 factors"""
    
    print("🧪 Testing Enhanced LLM Market Prediction System with Tier 1 Factors")
    print("=" * 80)
    
    # Initialize the service
    service = MarketPredictionService()
    
    # Create test request for Australian All Ordinaries
    request = PredictionRequest(
        symbol="^AORD",
        timeframe="5d",
        include_factors=True,
        include_news_intelligence=True,
        news_lookback_hours=48
    )
    
    print(f"📊 Testing prediction for: {request.symbol}")
    print(f"⏰ Timeframe: {request.timeframe}")
    print(f"🔍 Include factors: {request.include_factors}")
    print(f"📰 Include news intelligence: {request.include_news_intelligence}")
    print()
    
    try:
        # Generate enhanced prediction
        start_time = datetime.now()
        response = await service.get_market_prediction(request)
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        
        print(f"✅ Prediction generated successfully in {processing_time:.2f} seconds")
        print()
        
        # Display results
        if response.success:
            pred = response.prediction
            print("📈 PREDICTION RESULTS:")
            print(f"  Symbol: {pred['symbol']}")
            print(f"  Direction: {pred['direction']}")
            print(f"  Expected Change: {pred['expected_change_percent']:.2f}%")
            print(f"  Confidence: {pred['confidence_score']:.1%}")
            print(f"  Risk Level: {pred['risk_level']}")
            print(f"  Timeframe: {pred['timeframe']}")
            print()
            
            # Display Tier 1 factors
            if response.tier1_factors:
                print("🎯 TIER 1 FACTORS:")
                for category in ['super', 'options', 'social']:
                    category_factors = {k: v for k, v in response.tier1_factors.items() if k.startswith(category)}
                    if category_factors:
                        print(f"  {category.title()} Factors:")
                        for factor, value in category_factors.items():
                            print(f"    • {factor.replace('_', ' ').title()}: {value:+.3f}")
                print()
            
            # Display factor attribution
            if response.factor_attribution:
                attr = response.factor_attribution
                overall = attr.get('overall_signal', {})
                print("📊 FACTOR ATTRIBUTION:")
                print(f"  Overall Signal: {overall.get('direction', 'N/A').upper()}")
                print(f"  Bullishness Score: {overall.get('bullishness_score', 0):+.3f}")
                print(f"  Signal Confidence: {overall.get('confidence', 0):.1%}")
                
                consensus = attr.get('factor_consensus', {})
                print(f"  Factor Consensus: {consensus.get('consensus_strength', 0):.1%}")
                print(f"  Aligned Factors: {consensus.get('aligned_factors', 0)}/{consensus.get('total_factors', 0)}")
                print()
            
            # Display model information
            model_info = response.model_info
            print("🤖 MODEL INFORMATION:")
            print(f"  Type: {model_info.get('model_type', 'N/A')}")
            print(f"  Version: {model_info.get('version', 'N/A')}")
            
            accuracy = model_info.get('accuracy_metrics', {})
            if accuracy:
                print("  Enhanced Accuracy Metrics:")
                print(f"    • Baseline: {accuracy.get('baseline_accuracy', 0):.1%}")
                print(f"    • Enhanced: {accuracy.get('enhanced_accuracy', 0):.1%}")
                print(f"    • Directional: {accuracy.get('directional_accuracy', 0):.1%}")
                print(f"    • Tier 1 Contribution: {accuracy.get('tier1_contribution', 'N/A')}")
            
            factor_categories = model_info.get('factor_categories', [])
            if factor_categories:
                print(f"  Active Factor Categories: {', '.join(factor_categories)}")
            
            print()
            print("🎉 Enhanced prediction system working correctly!")
            
        else:
            print("❌ Prediction failed")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

async def test_individual_factor_systems():
    """Test individual factor systems"""
    
    print("\n🔬 Testing Individual Factor Systems")
    print("=" * 50)
    
    try:
        # Test super fund analyzer
        from super_fund_flow_analyzer import super_fund_analyzer
        print("📊 Testing Super Fund Flow Analyzer...")
        super_factors = await super_fund_analyzer.get_market_prediction_factors()
        print(f"  ✅ Collected {len(super_factors)} super fund factors")
        
        # Test options analyzer  
        from asx_options_analyzer import OptionsFlowAnalyzer
        options_analyzer = OptionsFlowAnalyzer()
        print("📈 Testing ASX Options Analyzer...")
        options_factors = await options_analyzer.get_market_prediction_factors(['XJO', 'CBA'])
        print(f"  ✅ Collected {len(options_factors)} options factors")
        
        # Test social sentiment analyzer
        from social_sentiment_tracker import SocialSentimentAnalyzer
        social_analyzer = SocialSentimentAnalyzer()
        print("🗣️ Testing Social Sentiment Analyzer...")
        social_factors = await social_analyzer.get_market_prediction_factors(24)
        print(f"  ✅ Collected {len(social_factors)} social factors")
        
        total_factors = len(super_factors) + len(options_factors) + len(social_factors)
        print(f"\n🎯 Total Tier 1 factors available: {total_factors}")
        
    except Exception as e:
        print(f"❌ Error testing factor systems: {e}")

if __name__ == "__main__":
    print("🚀 Starting Enhanced Market Prediction Tests")
    print()
    
    # Run tests
    asyncio.run(test_individual_factor_systems())
    asyncio.run(test_enhanced_prediction())
    
    print("\n✅ Test suite completed!")