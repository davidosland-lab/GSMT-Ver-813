#!/usr/bin/env python3
"""
Test script to verify the direction logic fix for CBA prediction
"""

import requests
import json
import time

# Test configuration
BASE_URL = "https://8080-ib2hlmks3ra0wwghsafox-6532622b.e2b.dev"

def test_prediction_consistency(symbol):
    """Test that predicted price and direction are consistent"""
    print(f"🧪 Testing {symbol} prediction consistency...")
    
    try:
        # Get prediction
        response = requests.get(f"{BASE_URL}/api/unified-prediction/{symbol}?horizon=5d", timeout=120)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                prediction = data['prediction']
                
                current_price = prediction['current_price']
                predicted_price = prediction['predicted_price']
                direction = prediction['direction']
                expected_return = prediction['expected_return']
                
                # Calculate actual return from prices
                actual_return_from_prices = (predicted_price - current_price) / current_price
                
                # Determine expected direction based on price difference
                expected_direction = "UP" if predicted_price > current_price else "DOWN"
                
                # Check consistency
                direction_consistent = direction == expected_direction
                return_consistent = abs(expected_return - actual_return_from_prices) < 0.001  # Small tolerance for floating point
                
                print(f"   Symbol: {symbol}")
                print(f"   Current Price: ${current_price:.2f}")
                print(f"   Predicted Price: ${predicted_price:.2f}")
                print(f"   Price Difference: ${predicted_price - current_price:.2f}")
                print(f"   Expected Return: {expected_return*100:.2f}%")
                print(f"   Actual Return from Prices: {actual_return_from_prices*100:.2f}%")
                print(f"   Direction: {direction}")
                print(f"   Expected Direction: {expected_direction}")
                print(f"   ✅ Direction Consistent: {direction_consistent}")
                print(f"   ✅ Return Consistent: {return_consistent}")
                
                if direction_consistent and return_consistent:
                    print(f"   🎯 SUCCESS: All values are consistent!")
                    return True
                else:
                    print(f"   ❌ FAILURE: Inconsistency detected!")
                    return False
            else:
                print(f"   ❌ Prediction not successful: {data}")
                return False
        else:
            print(f"   ❌ HTTP Error {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Test Failed: {e}")
        return False

def main():
    """Test multiple symbols to verify the fix"""
    print("🚀 DIRECTION LOGIC FIX VERIFICATION")
    print("=" * 60)
    
    # Test symbols
    symbols = ['CBA.AX', 'AAPL', '^GSPC', '^FTSE', 'ANZ.AX']
    
    results = {}
    
    for symbol in symbols:
        print(f"\n{'='*20} {symbol} {'='*20}")
        results[symbol] = test_prediction_consistency(symbol)
        
        # Add delay between requests to avoid overwhelming the server
        if symbol != symbols[-1]:  # Don't delay after the last symbol
            print(f"   ⏳ Waiting 5 seconds before next test...")
            time.sleep(5)
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 DIRECTION LOGIC FIX VERIFICATION SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for symbol, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status}: {symbol}")
        if not passed:
            all_passed = False
    
    print(f"\n🎯 OVERALL STATUS: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n🎉 DIRECTION LOGIC FIX SUCCESSFUL!")
        print("   ✅ Predicted price and direction are now consistent")
        print("   ✅ Expected return matches actual price difference")
        print("   ✅ All symbols show correct logic")
    else:
        print("\n⚠️ DIRECTION LOGIC STILL HAS ISSUES!")
        print("   ❌ Some symbols show inconsistent values")
        print("   🔧 Additional fixes may be needed")

if __name__ == "__main__":
    main()