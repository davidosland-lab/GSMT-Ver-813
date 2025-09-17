#!/usr/bin/env python3
"""
Debug API response structure for ^AORD data
"""

import asyncio
import aiohttp
import json
from datetime import datetime
import pytz

async def debug_api_structure():
    """Debug the actual API response structure"""
    
    print("🔍 Debugging API response structure for ^AORD...")
    
    try:
        base_url = "https://8000-ibk6l44w5p7m34y5uzj0a-6532622b.e2b.dev"
        api_url = f"{base_url}/api/analyze"
        
        request_data = {
            "symbols": ["^AORD"],
            "hours_back": 48,
            "hours_forward": 24
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, json=request_data) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    print(f"📊 API Response Structure:")
                    print(f"   Top-level keys: {list(data.keys())}")
                    
                    if "data" in data:
                        print(f"\n📈 Data section:")
                        print(f"   Data keys: {list(data['data'].keys())}")
                        
                        if "^AORD" in data['data']:
                            aord_data = data['data']["^AORD"]
                            print(f"\n🇦🇺 ^AORD section:")
                            print(f"   ^AORD data type: {type(aord_data)}")
                            
                            # Handle if it's a list instead of dict
                            if isinstance(aord_data, list):
                                points = aord_data
                                print(f"   ^AORD is a list of points")
                            elif isinstance(aord_data, dict) and "points" in aord_data:
                                points = aord_data["points"]
                                print(f"   ^AORD keys: {list(aord_data.keys())}")
                            else:
                                print(f"   ^AORD structure: {aord_data}")
                                points = []
                                print(f"\n📊 Points data:")
                                print(f"   Number of points: {len(points)}")
                                
                                if points:
                                    print(f"   First point structure: {list(points[0].keys())}")
                                    print(f"   First point sample: {points[0]}")
                                    
                                    # Get current AEST time
                                    aest = pytz.timezone('Australia/Sydney')
                                    now_aest = datetime.now(aest)
                                    current_aest_str = now_aest.strftime('%Y-%m-%d %H:%M:%S')
                                    print(f"\n🕐 Current AEST: {current_aest_str}")
                                    
                                    # Analyze timestamps
                                    print(f"\n🔍 Timestamp Analysis:")
                                    
                                    future_points = []
                                    past_points = []
                                    
                                    for point in points:
                                        timestamp = point.get("timestamp", "")
                                        timestamp_clean = timestamp.replace(' AEST', '') if timestamp else ""
                                        
                                        # Compare with current time (using same logic as frontend)
                                        if timestamp_clean > current_aest_str:
                                            future_points.append(point)
                                        else:
                                            past_points.append(point)
                                    
                                    print(f"   Past/Present points: {len(past_points)}")
                                    print(f"   Future points: {len(future_points)}")
                                    
                                    if future_points:
                                        print(f"\n🚨 FUTURE POINTS DETECTED:")
                                        print(f"   Number of future points: {len(future_points)}")
                                        
                                        # Show some future points
                                        print(f"\n   Sample future points:")
                                        for i, fp in enumerate(future_points[:5]):
                                            close_val = fp.get('close', 'N/A')
                                            timestamp = fp.get('timestamp', 'N/A')
                                            market_open = fp.get('market_open', False)
                                            print(f"     {i+1}. {timestamp} | Close: {close_val} | Market Open: {market_open}")
                                        
                                        # Check if future points have different characteristics
                                        future_closes = [fp.get('close', 0) for fp in future_points if fp.get('close') is not None]
                                        future_opens = [fp.get('open', 0) for fp in future_points if fp.get('open') is not None]
                                        
                                        # Check for flat-line pattern in future data
                                        flat_future_count = 0
                                        for fp in future_points:
                                            o, h, l, c = fp.get('open', 0), fp.get('high', 0), fp.get('low', 0), fp.get('close', 0)
                                            if o == h == l == c and o > 0:
                                                flat_future_count += 1
                                        
                                        print(f"\n   Future data analysis:")
                                        print(f"     Flat-line future points: {flat_future_count}/{len(future_points)}")
                                        if future_closes:
                                            print(f"     Future close values range: {min(future_closes):.2f} - {max(future_closes):.2f}")
                                        
                                        print(f"\n🎯 ROOT CAUSE IDENTIFIED:")
                                        print(f"   ✅ The API IS returning future timestamps")
                                        print(f"   🔧 Frontend filtering should remove these")
                                        print(f"   📊 These future points create the separate flat-line segment")
                                    
                                    else:
                                        print(f"\n✅ No future points detected in this response")
                                        print(f"🤔 The issue might be intermittent or timing-dependent")
                                    
                                    # Show the timeline around current time
                                    print(f"\n⏰ Timeline around current time:")
                                    
                                    # Sort points by timestamp
                                    sorted_points = sorted(points, key=lambda x: x.get('timestamp', ''))
                                    
                                    # Find points near current time
                                    current_time_index = -1
                                    for i, point in enumerate(sorted_points):
                                        timestamp_clean = point.get('timestamp', '').replace(' AEST', '')
                                        if timestamp_clean > current_aest_str:
                                            current_time_index = i
                                            break
                                    
                                    if current_time_index > 0:
                                        # Show a few points before and after current time
                                        start_idx = max(0, current_time_index - 3)
                                        end_idx = min(len(sorted_points), current_time_index + 3)
                                        
                                        for i in range(start_idx, end_idx):
                                            point = sorted_points[i]
                                            timestamp = point.get('timestamp', 'N/A')
                                            close_val = point.get('close', 'N/A')
                                            timestamp_clean = timestamp.replace(' AEST', '') if timestamp else ""
                                            
                                            status = "FUTURE" if timestamp_clean > current_aest_str else "PAST"
                                            marker = " <- NOW" if i == current_time_index else ""
                                            
                                            print(f"     {timestamp} | Close: {close_val} | [{status}]{marker}")
                                
                else:
                    print(f"❌ API Error: {response.status}")
                    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n✅ API structure debug complete.")

if __name__ == "__main__":
    asyncio.run(debug_api_structure())