#!/usr/bin/env python3
"""
Test the future data fix to ensure no future timestamps are returned
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
import pytz
from real_market_data_service import RealMarketDataAggregator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_future_data_fix():
    """Test that the fix prevents future data from being returned"""
    
    print("🧪 Testing future data fix for ^AORD...")
    
    # Get current time
    now_utc = datetime.now(timezone.utc)
    aest = pytz.timezone('Australia/Sydney')
    now_aest = now_utc.astimezone(aest)
    
    print(f"\n🕐 Current Time:")
    print(f"   UTC: {now_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"   AEST: {now_aest.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    # Get ^AORD data with the fix
    aggregator = RealMarketDataAggregator()
    symbol = "^AORD"
    
    # Clear any existing cache to test fresh data
    aggregator.cache.clear()
    
    market_data = await aggregator.get_real_market_data(symbol)
    
    if not market_data or not market_data.data_points:
        print(f"❌ No data received for {symbol}")
        return
    
    print(f"\n📊 Received {len(market_data.data_points)} data points for {symbol}")
    
    # Check for future data
    future_points = []
    current_points = []
    
    for point in market_data.data_points:
        if point.timestamp > now_utc:
            future_points.append(point)
        else:
            current_points.append(point)
    
    print(f"\n🎯 Future Data Check:")
    print(f"   Current/Past points: {len(current_points)}")
    print(f"   🚨 Future points: {len(future_points)}")
    
    if future_points:
        print(f"\n❌ FIX FAILED - Future data still present:")
        for i, point in enumerate(future_points[:5]):
            aest_time = point.timestamp.astimezone(aest)
            time_diff = point.timestamp - now_utc
            print(f"   {i+1}. {aest_time.strftime('%Y-%m-%d %H:%M:%S %Z')}: "
                  f"O={point.open:.2f} H={point.high:.2f} L={point.low:.2f} C={point.close:.2f} "
                  f"[+{time_diff.total_seconds()/3600:.1f}h future]")
    else:
        print(f"   ✅ SUCCESS - No future data detected!")
        print(f"   All {len(current_points)} data points are current or past")
    
    # Show the most recent data point
    if current_points:
        latest_point = max(current_points, key=lambda x: x.timestamp)
        aest_time = latest_point.timestamp.astimezone(aest)
        time_ago = now_utc - latest_point.timestamp
        
        print(f"\n📈 Most Recent Data Point:")
        print(f"   Time: {aest_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"   OHLC: O={latest_point.open:.2f} H={latest_point.high:.2f} L={latest_point.low:.2f} C={latest_point.close:.2f}")
        print(f"   Age: {time_ago.total_seconds()/60:.0f} minutes ago")
        
        # Check if this looks reasonable for real-time data
        if time_ago.total_seconds() > 3600:  # More than 1 hour old
            print(f"   ⚠️ Data seems quite old for real-time feed")
        else:
            print(f"   ✅ Data age is reasonable for real-time feed")
    
    # Test the cache behavior
    print(f"\n🔄 Testing Cache Behavior:")
    
    # Get data again to test cache
    market_data_cached = await aggregator.get_real_market_data(symbol)
    
    if market_data_cached:
        cached_future_points = [p for p in market_data_cached.data_points if p.timestamp > now_utc]
        
        print(f"   Cached data points: {len(market_data_cached.data_points)}")
        print(f"   Cached future points: {len(cached_future_points)}")
        
        if len(cached_future_points) == 0:
            print(f"   ✅ Cache also contains no future data")
        else:
            print(f"   ❌ Cache still contains future data")
    
    print(f"\n✅ Future data fix test complete.")

if __name__ == "__main__":
    asyncio.run(test_future_data_fix())