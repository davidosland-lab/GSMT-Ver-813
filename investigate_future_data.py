#!/usr/bin/env python3
"""
Investigate future data issue in ^AORD plotting
Focus on identifying data beyond current timestamp
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
import pytz
from real_market_data_service import RealMarketDataAggregator

# Set up detailed logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def investigate_future_data():
    """Investigate ^AORD future data plotting issue"""
    
    print("🔍 Investigating ^AORD future data plotting issue...")
    
    # Get current time in multiple timezones
    now_utc = datetime.now(timezone.utc)
    aest = pytz.timezone('Australia/Sydney')
    now_aest = now_utc.astimezone(aest)
    
    print(f"\n🕐 Current Time Analysis:")
    print(f"   UTC Now: {now_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"   AEST Now: {now_aest.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    # Get ^AORD data
    aggregator = RealMarketDataAggregator()
    symbol = "^AORD"
    market_data = await aggregator.get_real_market_data(symbol)
    
    if not market_data or not market_data.data_points:
        print(f"❌ No data received for {symbol}")
        return
    
    print(f"\n📊 Received {len(market_data.data_points)} data points for {symbol}")
    
    # Sort data points by timestamp
    sorted_points = sorted(market_data.data_points, key=lambda x: x.timestamp)
    
    print(f"\n🔍 Analyzing timestamps for future data...")
    
    future_points = []
    current_points = []
    past_points = []
    
    # Analyze each point relative to current time
    for point in sorted_points:
        aest_time = point.timestamp.astimezone(aest)
        
        # Check if this is future data
        if point.timestamp > now_utc:
            future_points.append((point, aest_time))
        elif point.timestamp > now_utc - timedelta(hours=1):  # Last hour
            current_points.append((point, aest_time))
        else:
            past_points.append((point, aest_time))
    
    print(f"\n📈 Timestamp Distribution:")
    print(f"   Past data points: {len(past_points)}")
    print(f"   Current/recent (last hour): {len(current_points)}")
    print(f"   🚨 FUTURE data points: {len(future_points)}")
    
    if future_points:
        print(f"\n🚨 FUTURE DATA DETECTED - This is the problem!")
        print(f"   Future points that shouldn't be plotted: {len(future_points)}")
        
        # Show the future data points
        print(f"\n🔍 Future Data Points (shouldn't exist):")
        for i, (point, aest_time) in enumerate(future_points[:10]):  # Show first 10
            time_diff = point.timestamp - now_utc
            print(f"   {i+1}. {aest_time.strftime('%Y-%m-%d %H:%M:%S %Z')}: "
                  f"O={point.open:.2f} H={point.high:.2f} L={point.low:.2f} C={point.close:.2f} "
                  f"[+{time_diff.total_seconds()/3600:.1f}h in future]")
        
        # Check if these future points cluster around 14:56 AEST
        future_14_56_points = [
            (point, aest_time) for point, aest_time in future_points 
            if aest_time.hour == 14 and aest_time.minute >= 56
        ]
        
        if future_14_56_points:
            print(f"\n🎯 CONFIRMED: Future data around 14:56 AEST timeframe!")
            print(f"   Points around 14:56 AEST: {len(future_14_56_points)}")
    else:
        print(f"   ✅ No future data detected")
    
    # Check the most recent legitimate data
    if current_points or past_points:
        all_legitimate = current_points + past_points
        latest_legitimate = max(all_legitimate, key=lambda x: x[0].timestamp)
        point, aest_time = latest_legitimate
        
        print(f"\n✅ Latest Legitimate Data:")
        print(f"   Time: {aest_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"   OHLC: O={point.open:.2f} H={point.high:.2f} L={point.low:.2f} C={point.close:.2f}")
        
        # Check gap between legitimate data and future data
        if future_points:
            earliest_future = min(future_points, key=lambda x: x[0].timestamp)
            future_point, future_aest = earliest_future
            
            gap = future_point.timestamp - latest_legitimate[0].timestamp
            print(f"\n⚠️ GAP ANALYSIS:")
            print(f"   Latest legitimate: {aest_time.strftime('%H:%M:%S %Z')}")
            print(f"   Earliest future: {future_aest.strftime('%H:%M:%S %Z')}")
            print(f"   Gap duration: {gap.total_seconds()/3600:.1f} hours")
            print(f"   This gap causes the separate plot segment!")
    
    # Check Yahoo Finance data timestamp issues
    print(f"\n🌐 Yahoo Finance Data Analysis:")
    print(f"   Data source: {market_data.sources_used}")
    print(f"   Last updated: {market_data.last_updated}")
    
    # Look for timestamp patterns that might indicate timezone issues
    utc_hours = [point.timestamp.hour for point in sorted_points[-20:]]  # Last 20 points
    aest_hours = [point.timestamp.astimezone(aest).hour for point in sorted_points[-20:]]
    
    print(f"   Recent UTC hours: {set(utc_hours)}")
    print(f"   Recent AEST hours: {set(aest_hours)}")
    
    # Check for timezone conversion issues
    if future_points:
        future_utc_hours = [point.timestamp.hour for point, _ in future_points[:5]]
        future_aest_hours = [aest_time.hour for _, aest_time in future_points[:5]]
        print(f"   Future UTC hours: {set(future_utc_hours)}")
        print(f"   Future AEST hours: {set(future_aest_hours)}")
    
    print(f"\n🎯 CONCLUSION:")
    if future_points:
        print(f"   🚨 PROBLEM CONFIRMED: Future data is being plotted")
        print(f"   🔧 SOLUTION NEEDED: Filter out timestamps > current time")
        print(f"   📊 This creates the separate segment you observed")
    else:
        print(f"   ✅ No future data detected - issue might be elsewhere")
    
    print(f"\n✅ Future data investigation complete.")

if __name__ == "__main__":
    asyncio.run(investigate_future_data())