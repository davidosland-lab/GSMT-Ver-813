#!/usr/bin/env python3
"""
Debug script to investigate ^AORD flat-line issue at 14:56 AEST
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
import pytz
from real_market_data_service import RealMarketDataAggregator

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def investigate_aord_flatline():
    """Investigate ^AORD data around 14:56 AEST timeframe"""
    
    print("🔍 Investigating ^AORD flat-line issue at 14:56 AEST...")
    
    # Create aggregator
    aggregator = RealMarketDataAggregator()
    
    # Get ^AORD data
    symbol = "^AORD"
    market_data = await aggregator.get_real_market_data(symbol)
    
    if not market_data or not market_data.data_points:
        print(f"❌ No data received for {symbol}")
        return
    
    print(f"\n📊 Received {len(market_data.data_points)} data points for {symbol}")
    print(f"📅 Sources used: {market_data.sources_used}")
    print(f"🕐 Last updated: {market_data.last_updated}")
    
    # Convert to AEST for analysis
    aest = pytz.timezone('Australia/Sydney')
    
    print("\n🔍 Analyzing data points around 14:56 AEST timeframe...")
    
    # Sort data points by timestamp
    sorted_points = sorted(market_data.data_points, key=lambda x: x.timestamp)
    
    # Look for the problematic timeframe (14:30-16:00 AEST)
    problem_start = None
    problem_end = None
    
    identical_ohlc_count = 0
    flat_line_sequences = []
    current_sequence = []
    
    for i, point in enumerate(sorted_points):
        # Convert to AEST
        aest_time = point.timestamp.astimezone(aest)
        time_str = aest_time.strftime('%Y-%m-%d %H:%M:%S %Z')
        
        # Check for identical OHLC (flat-line indicator)
        is_flat = (point.open == point.high == point.low == point.close)
        
        # Check for suspicious patterns
        ohlc_range = point.high - point.low
        close_open_diff = abs(point.close - point.open)
        
        # Check if we're in the problematic timeframe (14:00-16:00 AEST)
        if 14 <= aest_time.hour < 16:
            status = "🔴 PROBLEM TIMEFRAME"
            if is_flat:
                identical_ohlc_count += 1
                current_sequence.append(point)
                status += " + FLAT-LINE"
            elif current_sequence:
                # End of a flat-line sequence
                if len(current_sequence) > 1:
                    flat_line_sequences.append(current_sequence.copy())
                current_sequence = []
        else:
            status = "✅ Normal"
            if current_sequence:
                if len(current_sequence) > 1:
                    flat_line_sequences.append(current_sequence.copy())
                current_sequence = []
        
        print(f"{time_str}: O={point.open:7.2f} H={point.high:7.2f} L={point.low:7.2f} C={point.close:7.2f} V={point.volume:8d} Range={ohlc_range:6.3f} {status}")
    
    # Check for final sequence
    if current_sequence and len(current_sequence) > 1:
        flat_line_sequences.append(current_sequence)
    
    print(f"\n📈 Data Quality Analysis:")
    print(f"   Total data points: {len(sorted_points)}")
    print(f"   Identical OHLC points in problem timeframe: {identical_ohlc_count}")
    print(f"   Flat-line sequences detected: {len(flat_line_sequences)}")
    
    if flat_line_sequences:
        print(f"\n🚨 Flat-line sequences found:")
        for i, sequence in enumerate(flat_line_sequences):
            start_time = sequence[0].timestamp.astimezone(aest)
            end_time = sequence[-1].timestamp.astimezone(aest)
            print(f"   Sequence {i+1}: {start_time.strftime('%H:%M:%S')} - {end_time.strftime('%H:%M:%S')} AEST ({len(sequence)} points)")
            print(f"   Value: {sequence[0].close:.2f} (all OHLC identical)")
    
    # Check for data gaps
    print(f"\n⏱️ Checking for data gaps...")
    for i in range(1, len(sorted_points)):
        prev_point = sorted_points[i-1]
        curr_point = sorted_points[i]
        
        time_diff = curr_point.timestamp - prev_point.timestamp
        if time_diff > timedelta(minutes=10):  # Gap larger than expected interval
            prev_aest = prev_point.timestamp.astimezone(aest)
            curr_aest = curr_point.timestamp.astimezone(aest)
            print(f"   Gap detected: {prev_aest.strftime('%H:%M:%S')} -> {curr_aest.strftime('%H:%M:%S')} AEST ({time_diff})")
    
    # Check for recent data (last hour)
    now_aest = datetime.now(aest)
    recent_cutoff = now_aest - timedelta(hours=1)
    recent_points = [p for p in sorted_points if p.timestamp.astimezone(aest) >= recent_cutoff]
    
    print(f"\n🕐 Recent data (last hour): {len(recent_points)} points")
    if recent_points:
        latest_point = max(recent_points, key=lambda x: x.timestamp)
        latest_aest = latest_point.timestamp.astimezone(aest)
        print(f"   Latest data: {latest_aest.strftime('%H:%M:%S %Z')} - O={latest_point.open:.2f} H={latest_point.high:.2f} L={latest_point.low:.2f} C={latest_point.close:.2f}")
    
    print("\n🔚 Investigation complete.")

if __name__ == "__main__":
    asyncio.run(investigate_aord_flatline())