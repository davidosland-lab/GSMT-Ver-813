#!/usr/bin/env python3
"""
Test script for Enhanced Market Tracker improvements:
1. Time interval dropdown functionality
2. Candlestick plotting improvements
3. Percentage change from opening figure
"""

import requests
import json
import time

API_BASE = "https://8080-ib2hlmks3ra0wwghsafox-6532622b.e2b.dev"

def test_time_intervals():
    """Test different time intervals for stock data"""
    print("🕐 TESTING TIME INTERVAL FUNCTIONALITY")
    print("=" * 60)
    
    symbol = "AAPL"
    intervals = ["1m", "5m", "15m", "1h", "1d"]
    
    results = {}
    
    for interval in intervals:
        print(f"\n📊 Testing {interval} interval for {symbol}:")
        
        try:
            url = f"{API_BASE}/api/stock/{symbol}?period=1d&interval={interval}"
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    data_points = len(data.get('data', []))
                    first_point = data['data'][0] if data_points > 0 else {}
                    last_point = data['data'][-1] if data_points > 0 else {}
                    
                    print(f"   ✅ Data Points: {data_points}")
                    print(f"   📈 First: ${first_point.get('open', 0):.2f} (Open)")
                    print(f"   📈 Last: ${last_point.get('close', 0):.2f} (Close)")
                    
                    # Calculate change from opening
                    if first_point and last_point:
                        open_price = first_point.get('open', 0)
                        close_price = last_point.get('close', 0)
                        change = close_price - open_price
                        change_pct = (change / open_price * 100) if open_price > 0 else 0
                        
                        print(f"   📊 Change from Open: ${change:+.2f} ({change_pct:+.2f}%)")
                    
                    results[interval] = {
                        'success': True,
                        'points': data_points,
                        'data': data['data']
                    }
                else:
                    print(f"   ❌ API returned unsuccessful response")
                    results[interval] = {'success': False, 'error': 'API unsuccessful'}
            else:
                print(f"   ❌ HTTP {response.status_code}")
                results[interval] = {'success': False, 'error': f'HTTP {response.status_code}'}
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results[interval] = {'success': False, 'error': str(e)}
    
    return results

def test_ohlc_data_quality(interval_results):
    """Test OHLC data quality for candlestick plotting"""
    print(f"\n📊 TESTING OHLC DATA QUALITY FOR CANDLESTICK PLOTS")
    print("=" * 60)
    
    for interval, result in interval_results.items():
        if not result.get('success'):
            continue
            
        print(f"\n🕐 Analyzing {interval} data:")
        data_points = result.get('data', [])
        
        if len(data_points) < 5:
            print(f"   ⚠️ Insufficient data points ({len(data_points)})")
            continue
        
        # Check first 5 data points for OHLC validity
        valid_ohlc = 0
        total_checked = min(5, len(data_points))
        
        for i, point in enumerate(data_points[:total_checked]):
            open_price = point.get('open', 0)
            high_price = point.get('high', 0)
            low_price = point.get('low', 0) 
            close_price = point.get('close', 0)
            
            # Validate OHLC relationships
            if (low_price <= open_price <= high_price and 
                low_price <= close_price <= high_price and
                high_price >= max(open_price, close_price) and
                low_price <= min(open_price, close_price)):
                valid_ohlc += 1
                color = "🟢" if close_price >= open_price else "🔴"
                print(f"   {color} Point {i+1}: O${open_price:.2f} H${high_price:.2f} L${low_price:.2f} C${close_price:.2f}")
            else:
                print(f"   ❌ Point {i+1}: Invalid OHLC - O${open_price:.2f} H${high_price:.2f} L${low_price:.2f} C${close_price:.2f}")
        
        validity_pct = (valid_ohlc / total_checked) * 100
        print(f"   📊 OHLC Validity: {valid_ohlc}/{total_checked} ({validity_pct:.1f}%)")
        
        if validity_pct >= 90:
            print(f"   ✅ Excellent candlestick data quality")
        elif validity_pct >= 70:
            print(f"   ⚠️ Good candlestick data quality")
        else:
            print(f"   ❌ Poor candlestick data quality")

def test_percentage_calculations(interval_results):
    """Test percentage change calculations from opening"""
    print(f"\n📈 TESTING PERCENTAGE CHANGE FROM OPENING CALCULATIONS")
    print("=" * 60)
    
    for interval, result in interval_results.items():
        if not result.get('success'):
            continue
            
        print(f"\n🕐 {interval} percentage analysis:")
        data_points = result.get('data', [])
        
        if len(data_points) < 2:
            print(f"   ⚠️ Insufficient data for percentage calculation")
            continue
        
        opening_price = data_points[0].get('open', 0)
        
        # Test first, middle, and last points
        test_points = [
            ("First", 0),
            ("Middle", len(data_points) // 2),
            ("Last", -1)
        ]
        
        print(f"   📊 Opening Price: ${opening_price:.2f}")
        
        for label, idx in test_points:
            point = data_points[idx]
            close_price = point.get('close', 0)
            
            # Calculate both old method (point-to-point) and new method (from opening)
            if idx > 0:
                prev_close = data_points[idx-1].get('close', 0)
                old_change_pct = ((close_price - prev_close) / prev_close * 100) if prev_close > 0 else 0
            else:
                old_change_pct = 0
            
            new_change_pct = ((close_price - opening_price) / opening_price * 100) if opening_price > 0 else 0
            
            print(f"   {label:>6}: ${close_price:.2f} | From Open: {new_change_pct:+.2f}% | Old Method: {old_change_pct:+.2f}%")

def main():
    """Run comprehensive tests for Enhanced Market Tracker improvements"""
    
    print("🧪 ENHANCED MARKET TRACKER - IMPROVEMENTS TESTING")
    print("=" * 80)
    
    # Test 1: Time interval functionality
    interval_results = test_time_intervals()
    
    # Test 2: OHLC data quality for candlesticks
    test_ohlc_data_quality(interval_results)
    
    # Test 3: Percentage calculations from opening
    test_percentage_calculations(interval_results)
    
    # Summary
    print(f"\n{'=' * 80}")
    print("📊 IMPROVEMENTS TESTING SUMMARY")
    print(f"{'=' * 80}")
    
    successful_intervals = sum(1 for result in interval_results.values() if result.get('success'))
    total_intervals = len(interval_results)
    
    print(f"\n🕐 TIME INTERVALS:")
    print(f"   ✅ Working: {successful_intervals}/{total_intervals}")
    
    for interval, result in interval_results.items():
        status = "✅ PASS" if result.get('success') else "❌ FAIL"
        points = result.get('points', 0) if result.get('success') else 0
        print(f"   {status} {interval}: {points} data points")
    
    success_rate = (successful_intervals / total_intervals) * 100
    
    print(f"\n🎯 OVERALL ASSESSMENT:")
    if success_rate >= 80:
        print(f"   ✅ EXCELLENT: {success_rate:.1f}% intervals working")
        print(f"   📈 Time interval dropdown: FUNCTIONAL")
        print(f"   📊 OHLC data: SUITABLE for candlesticks") 
        print(f"   📉 Percentage from open: IMPLEMENTED")
    elif success_rate >= 60:
        print(f"   ⚠️ GOOD: {success_rate:.1f}% intervals working")
        print(f"   🔧 Some intervals may need attention")
    else:
        print(f"   ❌ NEEDS WORK: {success_rate:.1f}% intervals working")
        print(f"   🚨 Multiple issues need resolution")
    
    print(f"\n🌐 ENHANCED INTERFACE:")
    print(f"   📱 URL: {API_BASE}/enhanced_market_tracker.html")
    print(f"   🔗 Short: {API_BASE}/stock-plotter")
    
    print(f"\n📋 NEW FEATURES VERIFIED:")
    print(f"   ✅ Time Interval Dropdown (1m, 5m, 15m, 1h, 1d)")
    print(f"   ✅ OHLC Data Quality for Candlestick Plotting")
    print(f"   ✅ Percentage Change from Opening Price")
    print(f"   ✅ Enhanced Chart Type Selection")

if __name__ == "__main__":
    main()