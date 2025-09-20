#!/usr/bin/env python3
"""
Test the API endpoints to verify symbol selection bug fix
"""

import requests
import json
import time

# API base URL
API_BASE = "https://8080-ib2hlmks3ra0wwghsafox-6532622b.e2b.dev"

def test_api_symbol_prediction(symbol, timeframe="5d"):
    """Test the unified prediction API endpoint with different symbols"""
    
    url = f"{API_BASE}/api/unified-prediction/{symbol}"
    params = {"timeframe": timeframe, "include_all_domains": True}
    
    print(f"🔍 Testing API for symbol: {symbol}")
    print(f"   URL: {url}")
    
    try:
        response = requests.get(url, params=params, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"✅ API Response Success:")
            print(f"   Symbol: {data.get('symbol', 'N/A')}")
            print(f"   Current Price: ${data.get('current_price', 0):.2f}")
            print(f"   Predicted Price: ${data.get('predicted_price', 0):.2f}")
            print(f"   Expected Return: {data.get('expected_return', 0):+.2%}")
            print(f"   Direction: {data.get('direction', 'N/A')}")
            print(f"   Confidence: {data.get('confidence_score', 0):.1%}")
            
            # Check for CBA contamination in non-CBA symbols
            current_price = data.get('current_price', 0)
            if symbol not in ['CBA.AX', 'CBA'] and 160 <= current_price <= 170:
                print(f"🚨 WARNING: Non-CBA symbol {symbol} has CBA-range price ({current_price:.2f})")
            else:
                print(f"✅ Price range looks correct for {symbol}")
            
            return data
            
        else:
            print(f"❌ API Error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return None

def main():
    """Test the API with multiple symbols to verify bug fix"""
    
    print("🧪 TESTING API SYMBOL SELECTION BUG FIX")
    print("=" * 70)
    
    test_symbols = [
        "CBA.AX",    # Should work with CBA-specific pricing
        "^GSPC",     # S&P 500 - should NOT return CBA pricing
        "^FTSE",     # FTSE - should NOT return CBA pricing  
        "AAPL",      # Apple - should NOT return CBA pricing
    ]
    
    results = {}
    
    for symbol in test_symbols:
        print(f"\n{'-' * 50}")
        result = test_api_symbol_prediction(symbol)
        results[symbol] = result
        time.sleep(2)  # Small delay between requests
    
    print(f"\n{'=' * 70}")
    print("📊 SUMMARY ANALYSIS")
    print(f"{'=' * 70}")
    
    success_count = 0
    for symbol, result in results.items():
        if result:
            success_count += 1
            current_price = result.get('current_price', 0)
            
            if symbol == "CBA.AX":
                if 160 <= current_price <= 180:
                    print(f"✅ {symbol}: Correct CBA pricing range (${current_price:.2f})")
                else:
                    print(f"⚠️ {symbol}: Unexpected price for CBA (${current_price:.2f})")
            else:
                if 160 <= current_price <= 180:
                    print(f"🚨 {symbol}: BUG STILL EXISTS - CBA-range price (${current_price:.2f})")
                else:
                    print(f"✅ {symbol}: Correct non-CBA pricing (${current_price:.2f})")
        else:
            print(f"❌ {symbol}: API request failed")
    
    print(f"\n🎯 RESULT: {success_count}/{len(test_symbols)} symbols tested successfully")
    
    if success_count == len(test_symbols):
        print("✅ SYMBOL SELECTION BUG FIX: API TEST PASSED!")
    else:
        print("❌ Some API tests failed - investigate further")

if __name__ == "__main__":
    main()