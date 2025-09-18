#!/usr/bin/env python3
"""
Debug the actual API data being sent to frontend
"""

import asyncio
import logging
import json
from datetime import datetime, timezone, timedelta
import pytz
import aiohttp

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def debug_api_data():
    """Debug the actual API data being sent to frontend"""
    
    print("🔍 Debugging API data sent to frontend...")
    
    # Get current time
    now_utc = datetime.now(timezone.utc)
    aest = pytz.timezone('Australia/Sydney')
    now_aest = now_utc.astimezone(aest)
    
    print(f"\n🕐 Current Time:")
    print(f"   UTC: {now_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"   AEST: {now_aest.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    # Test the actual API endpoint that the frontend calls
    base_url = "https://8000-ibk6l44w5p7m34y5uzj0a-6532622b.e2b.dev"
    
    # Test ^AORD specifically
    symbols = ["^AORD"]
    
    for symbol in symbols:
        print(f"\n🌐 Testing API for {symbol}...")
        
        try:
            # Call the analyze endpoint (same as frontend)
            analyze_url = f"{base_url}/api/analyze"
            payload = {
                "symbols": [symbol],
                "timeframe": "48h"
            }
            
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(analyze_url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if symbol in data and 'data' in data[symbol]:
                            points = data[symbol]['data']
                            print(f"   📊 Received {len(points)} data points")
                            
                            # Analyze timestamps
                            future_points = []
                            current_points = []
                            
                            for point in points:
                                timestamp_str = point.get('timestamp', '')
                                
                                # Parse the timestamp (remove AEST if present)
                                clean_timestamp = timestamp_str.replace(' AEST', '')
                                
                                try:
                                    # Parse as AEST timestamp
                                    point_dt = datetime.strptime(clean_timestamp, '%Y-%m-%d %H:%M:%S')
                                    # Assume it's AEST and convert to UTC for comparison
                                    point_aest = aest.localize(point_dt)
                                    point_utc = point_aest.astimezone(timezone.utc)
                                    
                                    # Check if this is future data
                                    if point_utc > now_utc:
                                        time_diff = point_utc - now_utc
                                        future_points.append({
                                            'original': timestamp_str,
                                            'parsed_aest': point_aest,
                                            'parsed_utc': point_utc,
                                            'future_by': time_diff,
                                            'data': point
                                        })
                                    else:
                                        current_points.append({
                                            'original': timestamp_str,
                                            'parsed_aest': point_aest,
                                            'parsed_utc': point_utc,
                                            'data': point
                                        })
                                        
                                except ValueError as e:
                                    print(f"   ❌ Error parsing timestamp '{timestamp_str}': {e}")
                            
                            print(f"\n📈 Timestamp Analysis for {symbol}:")
                            print(f"   Current/Past points: {len(current_points)}")
                            print(f"   🚨 Future points: {len(future_points)}")
                            
                            if future_points:
                                print(f"\n🚨 FUTURE DATA DETECTED IN API RESPONSE:")
                                for i, fp in enumerate(future_points[:5]):  # Show first 5
                                    print(f"   {i+1}. Original: '{fp['original']}'")
                                    print(f"      AEST: {fp['parsed_aest'].strftime('%Y-%m-%d %H:%M:%S %Z')}")
                                    print(f"      UTC: {fp['parsed_utc'].strftime('%Y-%m-%d %H:%M:%S %Z')}")
                                    print(f"      Future by: {fp['future_by'].total_seconds()/3600:.1f} hours")
                                    
                                    data_point = fp['data']
                                    print(f"      OHLC: O={data_point.get('open', 'N/A')} H={data_point.get('high', 'N/A')} L={data_point.get('low', 'N/A')} C={data_point.get('close', 'N/A')}")
                                    print()
                                
                                # Check if future points cluster around 14:56
                                future_14_56 = [
                                    fp for fp in future_points 
                                    if fp['parsed_aest'].hour == 14 and fp['parsed_aest'].minute >= 56
                                ]
                                
                                if future_14_56:
                                    print(f"   🎯 Future points around 14:56 AEST: {len(future_14_56)}")
                                    print(f"   This confirms the user's observation!")
                            
                            # Show the most recent legitimate data
                            if current_points:
                                latest = max(current_points, key=lambda x: x['parsed_utc'])
                                print(f"\n✅ Latest legitimate data:")
                                print(f"   Time: {latest['parsed_aest'].strftime('%Y-%m-%d %H:%M:%S %Z')}")
                                data_point = latest['data']
                                print(f"   OHLC: O={data_point.get('open')} H={data_point.get('high')} L={data_point.get('low')} C={data_point.get('close')}")
                        else:
                            print(f"   ❌ No data found for {symbol}")
                    else:
                        print(f"   ❌ API Error: {response.status}")
                        
        except Exception as e:
            print(f"   ❌ Exception: {e}")
    
    print(f"\n✅ API data debugging complete.")

if __name__ == "__main__":
    asyncio.run(debug_api_data())