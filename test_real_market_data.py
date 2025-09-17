#!/usr/bin/env python3
"""
Test Real Market Data Integration
Verify that we're getting actual market data instead of simulated data
"""

import asyncio
import sys
import os
sys.path.append('/home/user/webapp')

from real_market_data_service import real_market_aggregator, YahooFinanceRealAPI
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_real_market_apis():
    """Test real market data APIs"""
    
    print("🧪 Testing Real Market Data APIs...")
    print("=" * 60)
    
    # Test symbols
    test_symbols = [
        '^GSPC',   # S&P 500
        '^AORD',   # ASX All Ordinaries
        'AAPL',    # Apple
        '^IXIC',   # NASDAQ
    ]
    
    for symbol in test_symbols:
        print(f"\n📊 Testing REAL market data for {symbol}...")
        print("-" * 40)
        
        try:
            # Test aggregator (tries multiple APIs)
            market_data = await real_market_aggregator.get_real_market_data(symbol)
            
            if market_data and market_data.data_points:
                latest = market_data.data_points[-1]  # Most recent data point
                oldest = market_data.data_points[0]   # Oldest data point
                
                print(f"✅ SUCCESS: Got {len(market_data.data_points)} REAL data points")
                print(f"   Sources used: {', '.join(market_data.sources_used)}")
                print(f"   Time range: {oldest.timestamp} to {latest.timestamp}")
                print(f"   Latest price: ${latest.close:.2f}")
                print(f"   Latest volume: {latest.volume:,}")
                print(f"   OHLC validation: O={latest.open:.2f}, H={latest.high:.2f}, L={latest.low:.2f}, C={latest.close:.2f}")
                
                # Validate OHLC relationships
                if latest.high >= max(latest.open, latest.close) and latest.low <= min(latest.open, latest.close):
                    print("   ✅ OHLC relationships are valid")
                else:
                    print("   ❌ OHLC relationships are invalid!")
                
                # Check if data is recent (within last 48 hours)
                from datetime import datetime, timezone, timedelta
                now = datetime.now(timezone.utc)
                if (now - latest.timestamp).total_seconds() < 48 * 3600:
                    print("   ✅ Data is recent (within 48 hours)")
                else:
                    print(f"   ⚠️ Data is old: {now - latest.timestamp} ago")
                    
            else:
                print("❌ FAILED: No real market data available")
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
    
    print(f"\n" + "=" * 60)
    print("🔍 Individual API Tests...")
    
    # Test Yahoo Finance directly
    print(f"\n📡 Testing Yahoo Finance API directly...")
    yahoo_api = YahooFinanceRealAPI()
    
    try:
        yahoo_data = await yahoo_api.get_real_data('^GSPC', period='1d', interval='5m')
        if yahoo_data:
            print(f"✅ Yahoo Finance: Got {len(yahoo_data)} data points for ^GSPC")
            latest = yahoo_data[-1]
            print(f"   Latest S&P 500: ${latest.close:.2f} at {latest.timestamp}")
        else:
            print("❌ Yahoo Finance: No data returned")
    except Exception as e:
        print(f"❌ Yahoo Finance error: {e}")

async def compare_real_vs_simulated():
    """Compare real market data vs our previous simulated data"""
    
    print(f"\n" + "=" * 60)
    print("📈 Real vs Simulated Data Comparison")
    print("=" * 60)
    
    # Get real ASX data
    asx_real_data = await real_market_aggregator.get_real_market_data('^AORD')
    
    if asx_real_data and asx_real_data.data_points:
        latest_real = asx_real_data.data_points[-1]
        print(f"\n🇦🇺 ASX All Ordinaries (^AORD)")
        print(f"   REAL market price: ${latest_real.close:.2f}")
        print(f"   Previous simulated: ~$8,250.00")
        print(f"   Difference: {((latest_real.close - 8250) / 8250 * 100):+.2f}%")
        print(f"   Source: {asx_real_data.sources_used[0]}")
        print(f"   Data timestamp: {latest_real.timestamp}")
        
        if abs(latest_real.close - 8250) / 8250 < 0.10:  # Within 10%
            print("   ✅ Simulated data was reasonably accurate")
        else:
            print("   ⚠️ Significant difference from simulated data")
    else:
        print("❌ Could not get real ASX data for comparison")

if __name__ == "__main__":
    print("🌐 REAL MARKET DATA TEST")
    print("Testing integration with actual market APIs...")
    print("NO SIMULATED DATA - Real market feeds only")
    
    asyncio.run(test_real_market_apis())
    asyncio.run(compare_real_vs_simulated())