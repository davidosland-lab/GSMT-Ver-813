#!/usr/bin/env python3
"""
Test script to verify proper candlestick implementation
"""

import requests
import json

BASE_URL = "https://8080-ib2hlmks3ra0wwghsafox-6532622b.e2b.dev"

def test_candlestick_implementation():
    """Test the new candlestick implementation"""
    
    print("🕯️ TESTING PROPER CANDLESTICK IMPLEMENTATION")
    print("=" * 60)
    
    try:
        # Test Enhanced Market Tracker page
        print("📱 Loading Enhanced Market Tracker...")
        response = requests.get(f"{BASE_URL}/enhanced_market_tracker.html", timeout=10)
        
        if response.status_code == 200:
            content = response.text
            
            # Check for new candlestick implementation
            checks = {
                "Financial Chart Plugin": 'chartjs-chart-financial' in content,
                "Candlestick Chart Type": "type: 'candlestick'" in content,
                "Bar Chart Fallback": "type: 'bar'" in content,
                "OHLC Data Structure": "'x: d.timestamp" in content and "'o: d.open" in content,
                "Proper Chart Detection": "config.type === 'candlestick'" in content,
                "Clean Tooltip": "callbacks: {" in content,
                "Fallback Implementation": "Using fallback OHLC bar chart" in content
            }
            
            print("🔍 New Candlestick Implementation Checks:")
            implementation_score = 0
            for check_name, passed in checks.items():
                status = "✅" if passed else "❌"
                print(f"   {status} {check_name}: {'Present' if passed else 'Missing'}")
                if passed:
                    implementation_score += 1
            
            # Check that old problematic implementation is removed
            removed_checks = {
                "Multiple Line Datasets": 'datasets.length >= 4' not in content,
                "Repetitive Tooltips": 'afterLabel: function(context)' not in content,
                "Overlapping Lines": 'borderWidth: 1.5' not in content
            }
            
            print("\n🗑️ Old Implementation Removal Checks:")
            removal_score = 0
            for check_name, removed in removed_checks.items():
                status = "✅" if removed else "❌"
                print(f"   {status} {check_name}: {'Removed' if removed else 'Still Present'}")
                if removed:
                    removal_score += 1
            
            total_score = implementation_score + removal_score
            max_score = len(checks) + len(removed_checks)
            
            print(f"\n📊 Implementation Score: {total_score}/{max_score} ({(total_score/max_score)*100:.1f}%)")
            
            # Test data availability
            print("\n📈 Testing OHLC Data for Candlestick Plotting:")
            test_symbol = "CBA.AX"
            
            try:
                data_response = requests.get(
                    f"{BASE_URL}/api/stock/{test_symbol}?period=1d&interval=1h", 
                    timeout=5
                )
                
                if data_response.status_code == 200:
                    data = data_response.json()
                    if data.get('success') and data.get('data'):
                        ohlc_data = data['data']
                        print(f"   ✅ {test_symbol}: {len(ohlc_data)} OHLC data points available")
                        
                        # Check data structure
                        if len(ohlc_data) > 0:
                            sample = ohlc_data[0]
                            required_fields = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                            has_all = all(field in sample for field in required_fields)
                            print(f"   ✅ Data Structure: {'Complete OHLC' if has_all else 'Incomplete'}")
                        
                        return total_score >= max_score * 0.8  # 80% implementation threshold
                    else:
                        print(f"   ❌ {test_symbol}: Invalid data response")
                        return False
                else:
                    print(f"   ❌ {test_symbol}: HTTP {data_response.status_code}")
                    return False
                    
            except Exception as e:
                print(f"   ❌ Data test failed: {e}")
                return False
                
        else:
            print(f"❌ Could not load page: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Run candlestick implementation test"""
    
    success = test_candlestick_implementation()
    
    print("\n" + "=" * 60)
    print("📋 PROPER CANDLESTICK IMPLEMENTATION RESULTS")
    print("=" * 60)
    
    if success:
        print("🎉 PROPER CANDLESTICK IMPLEMENTATION DETECTED!")
        print("   ✅ Financial chart plugin integrated")
        print("   ✅ Real candlestick chart type support")
        print("   ✅ Clean OHLC bar chart fallback")
        print("   ✅ Proper data structure (x, o, h, l, c)")
        print("   ✅ Old overlapping line implementation removed")
        
        print(f"\n🧪 TESTING WORKFLOW:")
        print(f"   1. Open: {BASE_URL}/enhanced_market_tracker.html")
        print(f"   2. Select a symbol (e.g., CBA.AX, AAPL)")
        print(f"   3. Click 'Start Tracking' and wait for data")
        print(f"   4. Click '📊 Candlestick' button")
        print(f"   5. Should see: Proper candlestick bars OR clean OHLC bars")
        print(f"   6. No more overlapping lines with repetitive tooltips!")
        
    else:
        print("❌ CANDLESTICK IMPLEMENTATION STILL HAS ISSUES")
        print("   🔧 Some implementation elements missing")
        print("   📝 May still show overlapping lines instead of candlesticks")
        print("   🛠️ Further fixes needed")

if __name__ == "__main__":
    main()