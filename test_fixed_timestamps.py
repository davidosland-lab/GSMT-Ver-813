#!/usr/bin/env python3
import requests
from datetime import datetime, timezone, timedelta

def test_fixed_timestamps():
    # Get market data
    response = requests.post('http://localhost:8000/api/analyze', json={
        'symbols': ['^N225'], 
        'chart_type': 'percentage',
        'interval_minutes': 5,
        'time_period': '24h'
    })
    
    data = response.json()
    points = data['data']['^N225']
    
    # Simulate the fixed JavaScript logic
    now_utc = datetime.now(timezone.utc)
    aest_offset_hours = 10
    now_aest = now_utc + timedelta(hours=aest_offset_hours)
    
    # Format current AEST time to match timestamp format
    current_aest_str = now_aest.strftime('%Y-%m-%d %H:%M:%S')
    
    print(f"Current UTC: {now_utc}")
    print(f"Current AEST: {current_aest_str} AEST") 
    print(f"Total points: {len(points)}")
    print()
    
    # Filter valid points
    valid_points = []
    for point in points:
        timestamp_str = point['timestamp'].replace(' AEST', '')
        is_valid = timestamp_str <= current_aest_str
        if is_valid:
            valid_points.append(point)
    
    print(f"Valid points (past/present): {len(valid_points)}")
    
    if valid_points:
        print(f"First valid point: {valid_points[0]['timestamp']}")
        print(f"Last valid point: {valid_points[-1]['timestamp']}")
        
        # Test market close line logic
        market_close_times = {
            'Japan': {'hour': 16, 'minute': 30}  # 16:30 AEST
        }
        
        target_close = market_close_times['Japan']
        target_minutes = target_close['hour'] * 60 + target_close['minute']
        
        print(f"\nLooking for Japan close at {target_close['hour']:02d}:{target_close['minute']:02d} AEST")
        
        closest_index = -1
        min_time_diff = float('inf')
        
        for i, point in enumerate(valid_points):
            timestamp_str = point['timestamp'].replace(' AEST', '')
            point_time = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            point_minutes = point_time.hour * 60 + point_time.minute
            
            time_diff = abs(point_minutes - target_minutes)
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_index = i
        
        if closest_index >= 0:
            print(f"Closest match: {valid_points[closest_index]['timestamp']}")
            print(f"Index: {closest_index}, Time diff: {min_time_diff} minutes")
            
            # Show context
            print("\nContext:")
            start = max(0, closest_index - 2)
            end = min(len(valid_points), closest_index + 3)
            for j in range(start, end):
                marker = " --> " if j == closest_index else "     "
                print(f"{marker}{j}: {valid_points[j]['timestamp']}")

if __name__ == "__main__":
    test_fixed_timestamps()