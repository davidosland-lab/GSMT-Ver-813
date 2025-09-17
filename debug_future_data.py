#!/usr/bin/env python3
"""
Debug script to investigate ^AORD future data plotting issue
Focus on timestamp analysis and future data detection
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
import pytz
from real_market_data_service import RealMarketDataAggregator

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def investigate_future_data():
    """Investigate ^AORD future data plotting issue"""
    
    print("🔍 Investigating ^AORD future data plotting issue...")
    
    # Get current time in multiple timezones
    now_utc = datetime.now(timezone.utc)
    aest = pytz.timezone('Australia/Sydney')
    now_aest = now_utc.astimezone(aest)
    
    print(f"\n🕐 Current Time Analysis:")
    print(f"   UTC:  {now_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"   AEST: {now_aest.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    # Create aggregator and get data
    aggregator = RealMarketDataAggregator()
    symbol = "^AORD"
    
    print(f"\n📡 Fetching ^AORD data...")
    market_data = await aggregator.get_real_market_data(symbol)
    
    if not market_data or not market_data.data_points:
        print(f"❌ No data received for {symbol}")
        return
    
    print(f"📊 Received {len(market_data.data_points)} data points")
    
    # Sort data points by timestamp
    sorted_points = sorted(market_data.data_points, key=lambda x: x.timestamp)
    
    # Analyze timestamps vs current time
    print(f"\n🔍 Timestamp Analysis vs Current Time:")
    
    past_points = []
    current_points = []
    future_points = []
    
    # Allow small buffer for real-time data (5 minutes)
    current_buffer = timedelta(minutes=5)
    
    for point in sorted_points:
        time_diff = point.timestamp - now_utc
        
        if time_diff < -current_buffer:
            past_points.append(point)
        elif abs(time_diff) <= current_buffer:
            current_points.append(point)
        else:
            future_points.append(point)
    
    print(f"   📈 Past data points: {len(past_points)}")
    print(f"   🕐 Current data points (±5min): {len(current_points)}")  
    print(f"   🚨 FUTURE data points: {len(future_points)}")
    
    if future_points:
        print(f"\n🚨 FUTURE DATA DETECTED - This is the problem!")
        print(f"   Future points detected: {len(future_points)}")
        
        # Show the first few future points
        print(f"\n📅 Sample Future Data Points:")
        for i, point in enumerate(future_points[:10]):
            aest_time = point.timestamp.astimezone(aest)
            time_diff = point.timestamp - now_utc
            
            # Check if it's a flat-line point
            is_flat = (point.open == point.high == point.low == point.close)
            
            print(f"   {i+1}. {aest_time.strftime('%Y-%m-%d %H:%M:%S %Z')} - Future by {time_diff} - O={point.open:.2f} H={point.high:.2f} L={point.low:.2f} C={point.close:.2f} {'[FLAT]' if is_flat else '[NORMAL]'}")
    
    # Check the timeline around current time
    print(f"\n⏰ Timeline Analysis Around Current Time:")
    
    # Get points within 2 hours of current time
    timeline_window = timedelta(hours=2)
    timeline_points = [
        p for p in sorted_points 
        if abs(p.timestamp - now_utc) <= timeline_window
    ]
    
    timeline_points.sort(key=lambda x: x.timestamp)
    
    print(f"   Points within ±2 hours of current time: {len(timeline_points)}")
    
    for i, point in enumerate(timeline_points):
        aest_time = point.timestamp.astimezone(aest)
        time_diff = point.timestamp - now_utc
        status = "FUTURE" if time_diff > current_buffer else ("CURRENT" if abs(time_diff) <= current_buffer else "PAST")
        is_flat = (point.open == point.high == point.low == point.close)
        
        print(f"   {aest_time.strftime('%H:%M:%S')} ({time_diff.total_seconds()/60:+5.1f}min) [{status:7}] O={point.open:.2f} H={point.high:.2f} L={point.low:.2f} C={point.close:.2f} {'[FLAT]' if is_flat else ''}")
    
    # Check if there's a gap between past and future data
    if past_points and future_points:
        latest_past = max(past_points, key=lambda x: x.timestamp)
        earliest_future = min(future_points, key=lambda x: x.timestamp)
        
        gap_duration = earliest_future.timestamp - latest_past.timestamp
        
        print(f"\n⚠️ DATA GAP ANALYSIS:")
        print(f"   Latest past data: {latest_past.timestamp.astimezone(aest).strftime('%H:%M:%S %Z')}")
        print(f"   Earliest future data: {earliest_future.timestamp.astimezone(aest).strftime('%H:%M:%S %Z')}")
        print(f"   Gap duration: {gap_duration}")
        print(f"   This gap + future data creates the separate plot segment!")
    
    # Check ASX market hours
    print(f"\n📊 ASX Market Context:")
    asx_open_hour = 10  # 10:00 AEST
    asx_close_hour = 16  # 16:00 AEST (4:00 PM)
    
    current_hour = now_aest.hour
    
    if asx_open_hour <= current_hour < asx_close_hour:
        market_status = "OPEN"
    elif current_hour < asx_open_hour:
        market_status = "PRE-MARKET"
    else:
        market_status = "CLOSED"
    
    print(f"   Current AEST time: {now_aest.strftime('%H:%M:%S')}")
    print(f"   ASX Market Status: {market_status}")
    print(f"   Market Hours: {asx_open_hour}:00 - {asx_close_hour}:00 AEST")
    
    # Look for the source of future data
    print(f"\n🔍 Data Source Analysis:")
    if market_data.sources_used:
        print(f"   Data source: {', '.join(market_data.sources_used)}")
    
    if future_points:
        # Check if future points follow a pattern
        future_intervals = []
        for i in range(1, min(10, len(future_points))):
            interval = future_points[i].timestamp - future_points[i-1].timestamp
            future_intervals.append(interval.total_seconds() / 60)  # minutes
        
        if future_intervals:
            avg_interval = sum(future_intervals) / len(future_intervals)
            print(f"   Future data interval: {avg_interval:.1f} minutes (avg)")
            print(f"   This suggests regular future timestamp generation")
    
    print(f"\n🎯 CONCLUSION:")
    if future_points:
        print(f"   🚨 ROOT CAUSE: Yahoo Finance is returning future timestamps")
        print(f"   📊 Solution needed: Filter out timestamps beyond current time")
        print(f"   🔧 The flat-line segment is future data that shouldn't be plotted")
    else:
        print(f"   ✅ No future data detected - issue might be elsewhere")
    
    print(f"\n✅ Future data investigation complete.")

if __name__ == "__main__":
    asyncio.run(investigate_future_data())