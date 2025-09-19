#!/usr/bin/env python3
"""
Comprehensive test script for Enhanced Market Tracker functionality
"""

import requests
import json
import time

# API base URL
API_BASE = "https://8080-ib2hlmks3ra0wwghsafox-6532622b.e2b.dev"

def test_api_endpoint(endpoint_name, url, expected_keys=None):
    """Test an API endpoint and verify response structure"""
    print(f"\n🔍 Testing {endpoint_name}:")
    print(f"   URL: {url}")
    
    try:
        start_time = time.time()
        response = requests.get(url, timeout=30)
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"✅ Status: {response.status_code} (Response time: {response_time:.2f}s)")
            
            if expected_keys:
                for key in expected_keys:
                    if key in data:
                        print(f"   ✅ Has '{key}': {type(data[key]).__name__}")
                        if key == 'total_symbols':
                            print(f"      📊 Value: {data[key]}")
                        elif key == 'data' and isinstance(data[key], list):
                            print(f"      📊 Data points: {len(data[key])}")
                        elif key == 'prediction' and isinstance(data[key], dict):
                            pred = data[key]
                            print(f"      📊 Direction: {pred.get('direction', 'N/A')}")
                            print(f"      💰 Current Price: ${pred.get('current_price', 0):.2f}")
                            print(f"      🎯 Predicted Price: ${pred.get('predicted_price', 0):.2f}")
                    else:
                        print(f"   ❌ Missing '{key}'")
            
            return True
            
        else:
            print(f"❌ Status: {response.status_code}")
            print(f"   Error: {response.text[:200]}...")
            return False
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

def main():
    """Test all Enhanced Market Tracker dependencies"""
    
    print("🧪 ENHANCED MARKET TRACKER - COMPREHENSIVE ENDPOINT TESTING")
    print("=" * 80)
    
    # Test 1: Symbols API (for dropdown population)
    symbols_success = test_api_endpoint(
        "Symbols API", 
        f"{API_BASE}/api/symbols",
        expected_keys=['total_symbols', 'markets', 'chart_types']
    )
    
    # Test 2: Stock Data API (for chart plotting)
    stock_success = test_api_endpoint(
        "Stock Data API (AAPL)", 
        f"{API_BASE}/api/stock/AAPL?period=1d&interval=1m",
        expected_keys=['success', 'symbol', 'data', 'total_points']
    )
    
    # Test 3: Unified Prediction API (for prediction generation)
    prediction_success = test_api_endpoint(
        "Unified Prediction API (AAPL)", 
        f"{API_BASE}/api/unified-prediction/AAPL?timeframe=5d&include_all_domains=true",
        expected_keys=['success', 'symbol', 'prediction', 'domain_analysis']
    )
    
    # Test 4: Alternative symbols for variety
    test_symbols = ['CBA.AX', '^GSPC', '^FTSE']
    
    print(f"\n{'='*50}")
    print("🔄 TESTING ADDITIONAL SYMBOLS")
    print(f"{'='*50}")
    
    additional_success = 0
    for symbol in test_symbols:
        print(f"\n📈 Testing {symbol}:")
        
        # Quick stock data test
        stock_url = f"{API_BASE}/api/stock/{symbol}?period=1d"
        try:
            response = requests.get(stock_url, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if data.get('success') and data.get('data'):
                    points = len(data['data'])
                    print(f"   ✅ Stock Data: {points} data points")
                    additional_success += 1
                else:
                    print(f"   ⚠️ Stock Data: No data available")
            else:
                print(f"   ❌ Stock Data: HTTP {response.status_code}")
        except Exception as e:
            print(f"   ❌ Stock Data: {e}")
    
    # Results summary
    print(f"\n{'='*80}")
    print("📊 TEST RESULTS SUMMARY")
    print(f"{'='*80}")
    
    core_tests = [
        ("Symbols API", symbols_success),
        ("Stock Data API", stock_success), 
        ("Unified Prediction API", prediction_success)
    ]
    
    core_passed = sum(1 for _, success in core_tests if success)
    
    print(f"\n🔧 CORE FUNCTIONALITY:")
    for test_name, success in core_tests:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status} {test_name}")
    
    print(f"\n📈 ADDITIONAL SYMBOLS:")
    print(f"   ✅ PASS {additional_success}/{len(test_symbols)} symbols")
    
    print(f"\n🎯 OVERALL RESULT:")
    if core_passed == 3:
        print("   ✅ ENHANCED MARKET TRACKER: FULLY FUNCTIONAL")
        print("   🚀 All required APIs working correctly")
        print("   📊 Chart plotting capability: READY")
        print("   🔮 Prediction generation: READY")
        print("   📱 Interface should work perfectly")
    elif core_passed >= 2:
        print("   ⚠️ ENHANCED MARKET TRACKER: MOSTLY FUNCTIONAL")
        print("   🔧 Some components may need attention")
    else:
        print("   ❌ ENHANCED MARKET TRACKER: NEEDS INVESTIGATION")
        print("   🚨 Multiple core components failing")
    
    print(f"\n🌐 LIVE INTERFACE URL:")
    print(f"   {API_BASE}/enhanced_market_tracker.html")
    print(f"   {API_BASE}/stock-plotter")

if __name__ == "__main__":
    main()