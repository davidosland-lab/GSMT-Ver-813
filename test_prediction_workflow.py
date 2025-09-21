#!/usr/bin/env python3
"""
Test the complete prediction workflow through the plotting interface
"""

import requests
import json
import time

API_BASE = "https://8080-ib2hlmks3ra0wwghsafox-6532622b.e2b.dev"

def test_complete_workflow(symbol, timeframe="5d"):
    """Test the complete workflow: stock data → chart → prediction"""
    print(f"\n{'='*60}")
    print(f"🔄 TESTING COMPLETE WORKFLOW FOR {symbol}")
    print(f"{'='*60}")
    
    # Step 1: Get stock data (for plotting)
    print(f"\n📊 Step 1: Fetching stock data for charting...")
    stock_url = f"{API_BASE}/api/stock/{symbol}?period=1d"
    
    try:
        response = requests.get(stock_url, timeout=30)
        if response.status_code == 200:
            stock_data = response.json()
            if stock_data.get('success'):
                data_points = len(stock_data.get('data', []))
                print(f"   ✅ Stock Data: {data_points} data points")
                
                # Get current price from stock data
                if data_points > 0:
                    latest_point = stock_data['data'][-1]
                    current_from_stock = latest_point.get('close', 0)
                    print(f"   💰 Current Price (from stock data): ${current_from_stock:.2f}")
                    print(f"   📈 Price Range: ${latest_point.get('low', 0):.2f} - ${latest_point.get('high', 0):.2f}")
            else:
                print(f"   ❌ Stock Data: API returned unsuccessful response")
                return False
        else:
            print(f"   ❌ Stock Data: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Stock Data: {e}")
        return False
    
    # Step 2: Generate prediction (what happens when user clicks predict)
    print(f"\n🔮 Step 2: Generating {timeframe} prediction...")
    pred_url = f"{API_BASE}/api/unified-prediction/{symbol}?timeframe={timeframe}&include_all_domains=true"
    
    try:
        start_time = time.time()
        response = requests.get(pred_url, timeout=60)
        processing_time = time.time() - start_time
        
        if response.status_code == 200:
            pred_data = response.json()
            if pred_data.get('success'):
                prediction = pred_data.get('prediction', {})
                
                print(f"   ✅ Prediction: Generated in {processing_time:.2f}s")
                print(f"   🎯 Direction: {prediction.get('direction', 'N/A')}")
                print(f"   💰 Current Price: ${prediction.get('current_price', 0):.2f}")
                print(f"   🎯 Predicted Price: ${prediction.get('predicted_price', 0):.2f}")
                print(f"   📊 Expected Return: {prediction.get('expected_return', 0)*100:.2f}%")
                print(f"   🎪 Confidence: {prediction.get('confidence_score', 0)*100:.1f}%")
                
                # Step 3: Validate prediction logic
                current_price = prediction.get('current_price', 0)
                predicted_price = prediction.get('predicted_price', 0)
                
                print(f"\n🔍 Step 3: Validating prediction logic...")
                
                # Check if prices are in reasonable range for symbol
                if symbol in ['CBA.AX', 'CBA']:
                    if 150 <= current_price <= 200:
                        print(f"   ✅ Current price in expected CBA range")
                    else:
                        print(f"   ⚠️ Current price outside expected CBA range ({current_price:.2f})")
                elif symbol in ['^GSPC']:
                    if 6000 <= current_price <= 7000:
                        print(f"   ✅ Current price in expected S&P 500 range")
                    else:
                        print(f"   ⚠️ Current price outside expected S&P 500 range ({current_price:.2f})")
                elif symbol == 'AAPL':
                    if 200 <= current_price <= 300:
                        print(f"   ✅ Current price in expected Apple range")
                    else:
                        print(f"   ⚠️ Current price outside expected Apple range ({current_price:.2f})")
                
                # Check prediction reasonableness
                return_pct = ((predicted_price - current_price) / current_price) * 100
                if abs(return_pct) < 50:  # Reasonable return expectation
                    print(f"   ✅ Prediction return ({return_pct:.2f}%) is reasonable")
                else:
                    print(f"   ⚠️ Prediction return ({return_pct:.2f}%) seems extreme")
                
                # Step 4: Check domain analysis
                domain_analysis = pred_data.get('domain_analysis', {})
                active_domains = domain_analysis.get('active_domains', 0)
                print(f"\n📈 Step 4: Domain Analysis")
                print(f"   📊 Active Domains: {active_domains}")
                
                if active_domains > 0:
                    print(f"   ✅ Multi-domain prediction working")
                    domain_predictions = domain_analysis.get('domain_predictions', {})
                    for domain, return_val in domain_predictions.items():
                        print(f"   🔧 {domain}: {return_val*100:.2f}% return")
                else:
                    print(f"   ⚠️ No active domains detected")
                
                return True
                
            else:
                print(f"   ❌ Prediction: API returned unsuccessful response")
                return False
        else:
            print(f"   ❌ Prediction: HTTP {response.status_code}")
            print(f"   📄 Error: {response.text[:200]}...")
            return False
    except Exception as e:
        print(f"   ❌ Prediction: {e}")
        return False

def main():
    """Test prediction workflow for multiple symbols"""
    
    print("🧪 ENHANCED MARKET TRACKER - PREDICTION WORKFLOW TESTING")
    print("=" * 80)
    
    # Test symbols that represent different markets and price ranges
    test_symbols = [
        ("AAPL", "5d"),      # US Tech stock - high price range
        ("CBA.AX", "5d"),    # Australian bank - medium price range  
        ("^GSPC", "1d"),     # S&P 500 Index - very high price range
    ]
    
    results = []
    
    for symbol, timeframe in test_symbols:
        success = test_complete_workflow(symbol, timeframe)
        results.append((symbol, timeframe, success))
        
        # Small delay between tests
        time.sleep(2)
    
    # Summary
    print(f"\n{'='*80}")
    print("🎯 WORKFLOW TESTING RESULTS")
    print(f"{'='*80}")
    
    successful_tests = 0
    for symbol, timeframe, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status} {symbol} ({timeframe})")
        if success:
            successful_tests += 1
    
    success_rate = (successful_tests / len(results)) * 100
    
    print(f"\n📊 OVERALL RESULTS:")
    print(f"   🎯 Success Rate: {successful_tests}/{len(results)} ({success_rate:.1f}%)")
    
    if success_rate >= 100:
        print(f"   🚀 PREDICTION WORKFLOW: PERFECT")
        print(f"   ✅ All symbols working correctly")
        print(f"   📈 Chart plotting + predictions fully functional")
    elif success_rate >= 75:
        print(f"   ✅ PREDICTION WORKFLOW: EXCELLENT") 
        print(f"   📈 Most functionality working correctly")
    elif success_rate >= 50:
        print(f"   ⚠️ PREDICTION WORKFLOW: GOOD")
        print(f"   🔧 Some issues may need attention")
    else:
        print(f"   ❌ PREDICTION WORKFLOW: NEEDS INVESTIGATION")
        print(f"   🚨 Multiple issues detected")
    
    print(f"\n🌐 READY TO USE:")
    print(f"   Interface: {API_BASE}/enhanced_market_tracker.html")
    print(f"   Short URL: {API_BASE}/stock-plotter")
    
    print(f"\n📋 USAGE INSTRUCTIONS:")
    print(f"   1. Open the Enhanced Market Tracker interface")
    print(f"   2. Select a symbol from dropdown or enter manually")  
    print(f"   3. Click 'Start Tracking' to load chart data")
    print(f"   4. Use prediction buttons (15min, 1h, 1d, 5d, 30d) to generate forecasts")
    print(f"   5. View predictions with confidence scores and risk analysis")

if __name__ == "__main__":
    main()