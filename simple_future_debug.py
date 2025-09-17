#!/usr/bin/env python3
"""
Simple debug to find future data in ^AORD
"""

import asyncio
import aiohttp
import json
from datetime import datetime
import pytz

async def simple_debug():
    print("🔍 Simple future data debug...")
    
    try:
        async with aiohttp.ClientSession() as session:
            url = "https://8000-ibk6l44w5p7m34y5uzj0a-6532622b.e2b.dev/api/analyze"
            data = {"symbols": ["^AORD"], "hours_back": 48, "hours_forward": 24}
            
            async with session.post(url, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    # Get current AEST time
                    aest = pytz.timezone('Australia/Sydney')
                    now_aest = datetime.now(aest)
                    current_str = now_aest.strftime('%Y-%m-%d %H:%M:%S')
                    print(f"Current AEST: {current_str}")
                    
                    if "data" in result and "^AORD" in result["data"]:
                        points = result["data"]["^AORD"]
                        print(f"Total points: {len(points)}")
                        
                        future_count = 0
                        
                        # Check last 20 points for future data
                        last_points = points[-20:] if len(points) >= 20 else points
                        
                        print(f"\nLast 20 points:")
                        for i, point in enumerate(last_points):
                            ts = point.get("timestamp", "")
                            close_val = point.get("close", 0)
                            
                            # Clean timestamp for comparison
                            ts_clean = ts.replace(' AEST', '') if ts else ""
                            is_future = ts_clean > current_str
                            
                            if is_future:
                                future_count += 1
                            
                            status = "FUTURE" if is_future else "PAST"
                            print(f"{i+1:2d}. {ts} | {close_val:.2f} | {status}")
                        
                        print(f"\nFuture points in last 20: {future_count}")
                        
                        if future_count > 0:
                            print("🚨 FOUND THE ISSUE: Future data detected!")
                        else:
                            print("✅ No future data in this sample")
                        
                else:
                    print(f"API Error: {response.status}")
                    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(simple_debug())