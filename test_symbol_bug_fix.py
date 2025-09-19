#!/usr/bin/env python3
"""
Test script to verify the symbol selection bug fix
"""

import asyncio
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, '/home/user/webapp')

from unified_super_predictor import unified_super_predictor

async def test_symbol_predictions():
    """Test predictions for different symbols to verify bug fix"""
    
    test_symbols = [
        "CBA.AX",    # Original working symbol
        "^GSPC",     # S&P 500 - should NOT return CBA data
        "^FTSE",     # FTSE - should NOT return CBA data
        "AAPL"       # Apple - should NOT return CBA data
    ]
    
    print("🧪 TESTING SYMBOL SELECTION BUG FIX")
    print("=" * 60)
    
    for symbol in test_symbols:
        try:
            print(f"\n🔍 Testing symbol: {symbol}")
            print("-" * 40)
            
            result = await unified_super_predictor.generate_unified_prediction(
                symbol=symbol,
                time_horizon="5d",
                include_all_domains=True
            )
            
            print(f"✅ Symbol: {result.symbol}")
            print(f"   Current Price: ${result.current_price:.2f}")
            print(f"   Predicted Price: ${result.predicted_price:.2f}")
            print(f"   Expected Return: {result.expected_return:+.2%}")
            print(f"   Direction: {result.direction}")
            print(f"   Confidence: {result.confidence_score:.1%}")
            
            # Check if we're getting CBA-range prices for non-CBA symbols
            if symbol != "CBA.AX" and symbol != "CBA":
                if 160 <= result.current_price <= 170:
                    print(f"🚨 WARNING: Non-CBA symbol {symbol} has CBA-range current price ({result.current_price:.2f})")
                if 160 <= result.predicted_price <= 170:
                    print(f"🚨 WARNING: Non-CBA symbol {symbol} has CBA-range predicted price ({result.predicted_price:.2f})")
            
        except Exception as e:
            print(f"❌ Error testing {symbol}: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Symbol selection bug fix test completed!")

if __name__ == "__main__":
    asyncio.run(test_symbol_predictions())