#!/usr/bin/env python3
import requests
import json
from datetime import datetime
import pytz

def debug_timestamps():
    # Get market data
    response = requests.post('http://localhost:8000/api/analyze', json={
        'symbols': ['^N225'], 
        'chart_type': 'percentage',
        'interval_minutes': 5,
        'time_period': '24h'
    })
    
    data = response.json()
    points = data['data']['^N225']
    
    print(f"Total points received: {len(points)}")
    
    # Show current times
    now_utc = datetime.now(pytz.UTC)
    aest = pytz.timezone('Australia/Sydney')
    now_aest = now_utc.astimezone(aest)
    
    print(f"Current UTC: {now_utc}")
    print(f"Current AEST: {now_aest}")
    print(f"Current naive datetime: {datetime.now()}")
    print()
    
    # Show sample timestamps
    print("Sample timestamps from data:")
    for i in range(min(10, len(points))):
        point = points[i]
        timestamp_str = point['timestamp']
        timestamp_clean = timestamp_str.replace(' AEST', '')
        
        try:
            point_time = datetime.strptime(timestamp_clean, '%Y-%m-%d %H:%M:%S')
            print(f"  {i}: '{timestamp_str}' -> parsed: {point_time}")
            print(f"      Clean: '{timestamp_clean}'")
            print(f"      vs now(): {point_time} <= {datetime.now()} = {point_time <= datetime.now()}")
        except Exception as e:
            print(f"  {i}: '{timestamp_str}' -> ERROR: {e}")
        print()
    
    print("Last few timestamps:")
    for i in range(max(0, len(points)-5), len(points)):
        point = points[i]
        timestamp_str = point['timestamp']
        timestamp_clean = timestamp_str.replace(' AEST', '')
        
        try:
            point_time = datetime.strptime(timestamp_clean, '%Y-%m-%d %H:%M:%S')
            print(f"  {i}: '{timestamp_str}' -> parsed: {point_time}")
        except Exception as e:
            print(f"  {i}: '{timestamp_str}' -> ERROR: {e}")

if __name__ == "__main__":
    debug_timestamps()