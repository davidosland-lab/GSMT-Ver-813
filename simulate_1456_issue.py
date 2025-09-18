#!/usr/bin/env python3
"""
Simulate the 14:56 AEST issue by checking data behavior at that specific time
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
import pytz
from real_market_data_service import RealMarketDataAggregator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def simulate_1456_issue():
    """Simulate what happens around 14:56 AEST"""
    
    print("🔍 Simulating the 14:56 AEST issue with ^AORD...")
    
    # Get ^AORD data
    aggregator = RealMarketDataAggregator()
    symbol = "^AORD"
    market_data = await aggregator.get_real_market_data(symbol)
    
    if not market_data or not market_data.data_points:
        print(f"❌ No data received for {symbol}")
        return
    
    # Convert to AEST for analysis
    aest = pytz.timezone('Australia/Sydney')
    
    # Sort data points by timestamp
    sorted_points = sorted(market_data.data_points, key=lambda x: x.timestamp)
    
    print(f"📊 Analyzing {len(sorted_points)} data points for 14:56 AEST behavior...")
    
    # Simulate the current time being 14:56 AEST (key insight: this is when issue appears)
    # Create a test time at 14:56 AEST today
    now_aest = datetime.now(aest)
    test_time_1456 = now_aest.replace(hour=14, minute=56, second=0, microsecond=0)
    test_time_1456_utc = test_time_1456.astimezone(timezone.utc)
    
    print(f"\n🕐 Simulation: Current time is 14:56 AEST")
    print(f"   Test AEST: {test_time_1456.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"   Test UTC: {test_time_1456_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    # Analyze what data would be considered "future" at 14:56 AEST
    past_at_1456 = []
    future_at_1456 = []
    
    for point in sorted_points:
        aest_time = point.timestamp.astimezone(aest)
        
        if point.timestamp <= test_time_1456_utc:
            past_at_1456.append((point, aest_time))
        else:
            future_at_1456.append((point, aest_time))
    
    print(f"\n📈 Data Analysis at 14:56 AEST simulation:")
    print(f"   Past/current data: {len(past_at_1456)}")
    print(f"   Future data (would be filtered): {len(future_at_1456)}")
    
    # Look for the pattern: data after 14:56 that appears as separate segment
    if future_at_1456:
        print(f"\n🚨 Future data that would be filtered at 14:56 AEST:")
        
        # Group future data by time periods
        immediate_future = []  # Within 1 hour
        later_future = []      # More than 1 hour
        
        for point, aest_time in future_at_1456:
            time_diff = point.timestamp - test_time_1456_utc
            if time_diff <= timedelta(hours=1):
                immediate_future.append((point, aest_time, time_diff))
            else:
                later_future.append((point, aest_time, time_diff))
        
        print(f"   Immediate future (< 1h): {len(immediate_future)}")
        print(f"   Later future (> 1h): {len(later_future)}")
        
        # Show immediate future data (this might be the "separate segment")
        if immediate_future:
            print(f"\n🎯 Immediate future data (potential separate segment):")
            for i, (point, aest_time, time_diff) in enumerate(immediate_future[:10]):
                minutes_future = time_diff.total_seconds() / 60
                print(f"   {i+1}. {aest_time.strftime('%H:%M AEST')}: "
                      f"O={point.open:.2f} H={point.high:.2f} L={point.low:.2f} C={point.close:.2f} "
                      f"[+{minutes_future:.0f}min future]")
    
    # Check for the specific behavior: what if filtering is inconsistent?
    print(f"\n🔍 Checking for inconsistent filtering behavior...")
    
    # Simulate frontend filtering at 14:56 AEST
    # The frontend does: timestampStr <= currentAESTStr
    current_aest_str = test_time_1456.strftime('%Y-%m-%d %H:%M:%S')
    
    frontend_filtered = []
    frontend_future = []
    
    for point in sorted_points:
        aest_time = point.timestamp.astimezone(aest)
        timestamp_str = aest_time.strftime('%Y-%m-%d %H:%M:%S')
        
        if timestamp_str <= current_aest_str:
            frontend_filtered.append((point, aest_time, timestamp_str))
        else:
            frontend_future.append((point, aest_time, timestamp_str))
    
    print(f"   Frontend would show: {len(frontend_filtered)} points")
    print(f"   Frontend would hide: {len(frontend_future)} points")
    
    # The key insight: if backend provides future data but frontend filtering is inconsistent
    if future_at_1456 and len(frontend_future) != len(future_at_1456):
        print(f"\n⚠️ FILTERING MISMATCH DETECTED!")
        print(f"   Backend future count: {len(future_at_1456)}")
        print(f"   Frontend future count: {len(frontend_future)}")
        print(f"   This mismatch could cause the separate segment issue!")
    
    # Check for timezone-based issues around 14:56
    print(f"\n🌏 Timezone Analysis at 14:56 AEST:")
    aest_offset = test_time_1456.utcoffset().total_seconds() / 3600
    print(f"   AEST UTC offset: +{aest_offset:.0f} hours")
    
    # Check if any data points have timestamps that would be problematic
    problematic_points = []
    for point in sorted_points:
        aest_time = point.timestamp.astimezone(aest)
        
        # Check if this point is within the "problem window"
        if (aest_time.hour == 14 and aest_time.minute >= 56) or (aest_time.hour == 15):
            problematic_points.append((point, aest_time))
    
    if problematic_points:
        print(f"   Points in problem window (14:56-15:59 AEST): {len(problematic_points)}")
        print(f"   These points might cause the separate segment visualization")
        
        # Show a few examples
        for i, (point, aest_time) in enumerate(problematic_points[:3]):
            is_future_at_1456 = point.timestamp > test_time_1456_utc
            status = "FUTURE" if is_future_at_1456 else "PAST"
            print(f"   Example {i+1}: {aest_time.strftime('%H:%M AEST')} - {status} at 14:56")
    
    print(f"\n✅ 14:56 AEST issue simulation complete.")

if __name__ == "__main__":
    asyncio.run(simulate_1456_issue())