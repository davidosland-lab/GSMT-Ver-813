#!/usr/bin/env python3
"""
Test script to verify:
1. CBA prediction functionality is working
2. Candlestick plotting data availability
3. Enhanced Market Tracker API endpoints
"""

import requests
import json
import time

# Test configuration
BASE_URL = "https://8080-ib2hlmks3ra0wwghsafox-6532622b.e2b.dev"

def test_cba_prediction():
    """Test CBA prediction endpoint"""
    print("🧪 TESTING CBA PREDICTION FUNCTIONALITY")
    print("=" * 60)
    
    try:
        # Test CBA prediction
        print("📊 Testing CBA.AX prediction...")
        start_time = time.time()
        
        response = requests.get(f"{BASE_URL}/api/unified-prediction/CBA.AX?horizon=5d", timeout=120)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                prediction = data['prediction']
                print(f"✅ CBA Prediction Successful!")
                print(f"   Symbol: CBA.AX")
                print(f"   Current Price: ${prediction['current_price']:.2f}")
                print(f"   Predicted Price: ${prediction['predicted_price']:.2f}")
                print(f"   Direction: {prediction['direction']}")
                print(f"   Confidence: {prediction['confidence_score']*100:.1f}%")
                print(f"   Expected Return: {prediction['expected_return']*100:.2f}%")
                print(f"   Processing Time: {processing_time:.2f}s")
                return True
            else:
                print(f"❌ Prediction not successful: {data}")
                return False
        else:
            print(f"❌ HTTP Error {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ CBA Prediction Test Failed: {e}")
        return False

def test_candlestick_data():
    """Test stock data endpoints for candlestick plotting"""
    print("\n🕯️ TESTING CANDLESTICK DATA AVAILABILITY")
    print("=" * 60)
    
    symbols = ['CBA.AX', 'AAPL', '^GSPC', '^FTSE']
    intervals = ['1m', '5m', '1h', '1d']
    
    success_count = 0
    total_tests = len(symbols) * len(intervals)
    
    for symbol in symbols:
        print(f"\n📈 Testing {symbol}:")
        for interval in intervals:
            try:
                response = requests.get(
                    f"{BASE_URL}/api/stock/{symbol}?period=1d&interval={interval}", 
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('success') and data.get('data'):
                        ohlc_data = data['data']
                        data_points = len(ohlc_data)
                        
                        # Check OHLC data quality
                        if data_points > 0:
                            sample = ohlc_data[0]
                            has_ohlc = all(key in sample for key in ['open', 'high', 'low', 'close'])
                            
                            if has_ohlc:
                                print(f"   ✅ {interval}: {data_points} points, OHLC complete")
                                success_count += 1
                            else:
                                print(f"   ❌ {interval}: Missing OHLC data")
                        else:
                            print(f"   ❌ {interval}: No data points")
                    else:
                        print(f"   ❌ {interval}: Invalid response format")
                else:
                    print(f"   ❌ {interval}: HTTP {response.status_code}")
                    
            except Exception as e:
                print(f"   ❌ {interval}: Error - {e}")
    
    success_rate = (success_count / total_tests) * 100
    print(f"\n📊 Candlestick Data Summary:")
    print(f"   ✅ Success: {success_count}/{total_tests} ({success_rate:.1f}%)")
    
    return success_rate > 80

def test_enhanced_market_tracker():
    """Test Enhanced Market Tracker main page"""
    print("\n🎯 TESTING ENHANCED MARKET TRACKER INTERFACE")
    print("=" * 60)
    
    try:
        # Test main page
        response = requests.get(f"{BASE_URL}/enhanced_market_tracker.html", timeout=10)
        
        if response.status_code == 200:
            content = response.text
            
            # Check for key elements
            checks = {
                "Candlestick Button": "data-view=\"candlestick\"" in content,
                "Time Interval Dropdown": "intervalSelect" in content,
                "Chart Container": "mainChart" in content,
                "setChartView Function": "setChartView" in content,
                "changeTimeInterval Function": "changeTimeInterval" in content,
                "initializeCandlestickChart Function": "initializeCandlestickChart" in content
            }
            
            print("🔍 Interface Component Checks:")
            all_passed = True
            for check_name, passed in checks.items():
                status = "✅" if passed else "❌"
                print(f"   {status} {check_name}: {'Present' if passed else 'Missing'}")
                if not passed:
                    all_passed = False
            
            return all_passed
            
        else:
            print(f"❌ Could not load interface: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Interface test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 ENHANCED MARKET TRACKER - FUNCTIONALITY TESTING")
    print("=" * 80)
    
    results = {}
    
    # Test CBA prediction
    results['cba_prediction'] = test_cba_prediction()
    
    # Test candlestick data
    results['candlestick_data'] = test_candlestick_data()
    
    # Test interface
    results['interface'] = test_enhanced_market_tracker()
    
    # Final summary
    print("\n" + "=" * 80)
    print("📋 FINAL TEST RESULTS")
    print("=" * 80)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status}: {test_name.replace('_', ' ').title()}")
        if not passed:
            all_passed = False
    
    print(f"\n🎯 OVERALL STATUS: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    if results.get('cba_prediction'):
        print("✅ CBA Prediction: Working correctly")
    
    if results.get('candlestick_data'):
        print("✅ Candlestick Data: Available and complete")
    else:
        print("❌ Candlestick Data: Issues detected")
        
    if results.get('interface'):
        print("✅ Interface: All components present")
    else:
        print("❌ Interface: Missing components detected")
    
    print(f"\n🌐 Access URL: {BASE_URL}/enhanced_market_tracker.html")

if __name__ == "__main__":
    main()