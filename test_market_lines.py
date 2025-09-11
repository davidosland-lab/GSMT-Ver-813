#!/usr/bin/env python3
import requests
import json
from datetime import datetime

def test_market_line_positioning():
    # Get market data
    response = requests.post('http://localhost:8000/api/analyze', json={
        'symbols': ['^N225', '^AXJO'], 
        'chart_type': 'percentage',
        'interval_minutes': 5,
        'time_period': '24h'
    })
    
    data = response.json()
    
    symbol_infos = {
        '^N225': {'market': 'Japan'},
        '^AXJO': {'market': 'Australia'}
    }
    
    market_close_times = {
        'Japan': {'hour': 16, 'minute': 30},      # 16:30 AEST 
        'Australia': {'hour': 16, 'minute': 0}    # 16:00 AEST
    }
    
    for symbol, points in data['data'].items():
        if symbol not in symbol_infos:
            continue
            
        symbol_info = symbol_infos[symbol]
        market = symbol_info['market']
        
        print(f"=== {symbol} ({market}) ===")
        print(f"Total points: {len(points)}")
        
        # Filter valid points (no future data)
        now = datetime.now()
        valid_points = []
        for point in points:
            timestamp_str = point['timestamp'].replace(' AEST', '')
            point_time = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            if point_time <= now:
                valid_points.append(point)
        
        print(f"Valid points (past/present): {len(valid_points)}")
        
        if market in market_close_times:
            target_close = market_close_times[market]
            target_minutes = target_close['hour'] * 60 + target_close['minute']
            
            print(f"Looking for close time: {target_close['hour']:02d}:{target_close['minute']:02d} AEST")
            
            closest_index = -1
            min_time_diff = float('inf')
            closest_point = None
            
            for i, point in enumerate(valid_points):
                timestamp_str = point['timestamp'].replace(' AEST', '')
                point_time = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                point_minutes = point_time.hour * 60 + point_time.minute
                
                time_diff = abs(point_minutes - target_minutes)
                
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_index = i
                    closest_point = point
            
            if closest_point:
                print(f"Closest match: {closest_point['timestamp']}")
                print(f"Index: {closest_index}, Time diff: {min_time_diff} minutes")
                
                # Show some surrounding timestamps for context
                print("Context (nearby timestamps):")
                start_idx = max(0, closest_index - 2)
                end_idx = min(len(valid_points), closest_index + 3)
                for j in range(start_idx, end_idx):
                    marker = " --> " if j == closest_index else "     "
                    print(f"{marker}{j}: {valid_points[j]['timestamp']}")
        
        print()

if __name__ == "__main__":
    test_market_line_positioning()