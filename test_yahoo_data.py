#!/usr/bin/env python3
"""
Test Yahoo Finance data provider directly
"""

import asyncio
import sys
import os
sys.path.append('/home/user/webapp')

from multi_source_data_service import YahooFinanceProvider, multi_source_aggregator
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_yahoo_provider():
    """Test Yahoo Finance provider directly"""
    
    print("🧪 Testing Yahoo Finance Provider...")
    
    # Test Yahoo directly
    yahoo = YahooFinanceProvider()
    print(f"Yahoo configured: {yahoo.is_configured()}")
    
    # Test a few symbols
    symbols_to_test = ['^GSPC', '^IXIC', '^DJI']
    
    for symbol in symbols_to_test:
        print(f"\n📊 Testing {symbol}...")
        try:
            data_points = await yahoo.get_intraday_data(symbol)
            if data_points:
                print(f"✅ Got {len(data_points)} data points for {symbol}")
                if len(data_points) > 0:
                    latest = data_points[-1]
                    print(f"   Latest: {latest.timestamp} | Close: ${latest.close:.2f} | Volume: {latest.volume:,}")
            else:
                print(f"❌ No data for {symbol}")
        except Exception as e:
            print(f"❌ Error for {symbol}: {e}")
    
    print(f"\n🔧 Testing Multi-Source Aggregator...")
    print(f"Providers initialized: {len(multi_source_aggregator.providers)}")
    for provider in multi_source_aggregator.providers:
        print(f"  - {provider.name}")
    
    # Test aggregator
    test_symbol = '^GSPC'
    print(f"\n📈 Testing aggregator with {test_symbol}...")
    try:
        market_data = await multi_source_aggregator.get_live_data(test_symbol)
        if market_data:
            print(f"✅ Aggregator success: {len(market_data.data_points)} points from {market_data.sources_used}")
        else:
            print("❌ Aggregator returned no data")
    except Exception as e:
        print(f"❌ Aggregator error: {e}")

if __name__ == "__main__":
    asyncio.run(test_yahoo_provider())