#!/usr/bin/env python3
"""
Performance Validation Test - Compare original vs optimized prediction systems
"""

import asyncio
import time
import statistics
from market_prediction_llm import prediction_service, PredictionRequest
from optimized_prediction_system import optimized_prediction_service, OptimizedPredictionRequest

async def benchmark_systems():
    """Compare performance of original vs optimized systems"""
    
    print("🏆 PERFORMANCE VALIDATION TEST")
    print("=" * 60)
    
    # Test parameters
    test_cases = [
        {"symbol": "^AORD", "timeframe": "1d"},
        {"symbol": "^AORD", "timeframe": "5d"},
    ]
    
    original_times = []
    optimized_times = []
    
    print("\n📊 ORIGINAL SYSTEM PERFORMANCE:")
    for i, test_case in enumerate(test_cases, 1):
        try:
            print(f"\n🧪 Test {i}: {test_case['symbol']} ({test_case['timeframe']})")
            
            # Test original system
            start_time = time.time()
            
            request = PredictionRequest(
                symbol=test_case['symbol'],
                timeframe=test_case['timeframe'],
                include_factors=True,
                include_news_intelligence=True
            )
            
            response = await prediction_service.get_market_prediction(request)
            original_time = time.time() - start_time
            original_times.append(original_time)
            
            print(f"  ⏱️ Original: {original_time:.2f}s - {response.prediction.get('direction', 'N/A').upper()}")
            
        except Exception as e:
            print(f"  ❌ Original system failed: {e}")
            original_times.append(60)  # Penalty for failure
    
    print(f"\n📈 OPTIMIZED SYSTEM PERFORMANCE:")
    for i, test_case in enumerate(test_cases, 1):
        try:
            print(f"\n🧪 Test {i}: {test_case['symbol']} ({test_case['timeframe']})")
            
            # Test optimized system (first call)
            start_time = time.time()
            
            opt_request = OptimizedPredictionRequest(
                symbol=test_case['symbol'],
                timeframe=test_case['timeframe'],
                include_factors=True,
                include_news_intelligence=True
            )
            
            opt_response = await optimized_prediction_service.generate_fast_prediction(opt_request)
            optimized_time = time.time() - start_time
            optimized_times.append(optimized_time)
            
            grade = opt_response.processing_metrics.get('performance_grade', 'UNKNOWN')
            print(f"  ⚡ Optimized: {optimized_time:.2f}s ({grade}) - {opt_response.prediction.get('direction', 'N/A').upper()}")
            
        except Exception as e:
            print(f"  ❌ Optimized system failed: {e}")
            optimized_times.append(60)  # Penalty for failure
    
    # Performance comparison
    print(f"\n🎯 PERFORMANCE COMPARISON:")
    print(f"{'Metric':<25} {'Original':<15} {'Optimized':<15} {'Improvement'}")
    print("-" * 70)
    
    if original_times and optimized_times:
        avg_original = statistics.mean(original_times)
        avg_optimized = statistics.mean(optimized_times)
        improvement_factor = avg_original / avg_optimized if avg_optimized > 0 else 0
        
        print(f"{'Average Time':<25} {avg_original:.2f}s{'':<8} {avg_optimized:.2f}s{'':<8} {improvement_factor:.1f}x faster")
        print(f"{'Best Time':<25} {min(original_times):.2f}s{'':<8} {min(optimized_times):.2f}s{'':<8} {min(original_times)/min(optimized_times):.1f}x faster")
        print(f"{'Worst Time':<25} {max(original_times):.2f}s{'':<8} {max(optimized_times):.2f}s{'':<8} {max(original_times)/max(optimized_times):.1f}x faster")
        
        # Performance grade
        if avg_optimized < 1.0:
            grade = "🏆 EXCELLENT"
        elif avg_optimized < 3.0:
            grade = "🥈 GOOD"
        elif avg_optimized < 8.0:
            grade = "🥉 ACCEPTABLE"
        else:
            grade = "❌ NEEDS_WORK"
        
        print(f"\n🏅 OVERALL PERFORMANCE GRADE: {grade}")
        print(f"💰 EFFICIENCY GAIN: {improvement_factor:.0f}x performance improvement")
        
        # Cost savings estimate
        cost_savings = ((avg_original - avg_optimized) / avg_original) * 100
        print(f"💵 RESOURCE SAVINGS: ~{cost_savings:.0f}% reduction in processing time")
        
    return {
        'original_times': original_times,
        'optimized_times': optimized_times,
        'improvement_factor': improvement_factor if 'improvement_factor' in locals() else 0
    }

if __name__ == "__main__":
    print("🚀 Starting Performance Validation")
    results = asyncio.run(benchmark_systems())
    print(f"\n✅ Validation completed! Optimization delivers {results['improvement_factor']:.0f}x performance boost!")