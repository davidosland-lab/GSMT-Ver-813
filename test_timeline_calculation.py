#!/usr/bin/env python3

def test_timeline_calculation():
    """Test the timeline calculation logic"""
    
    # Market close times
    market_close_times = {
        'Japan': {'hour': 16, 'minute': 30},      # 16:30 AEST 
        'Australia': {'hour': 16, 'minute': 0}    # 16:00 AEST
    }
    
    # Timeline settings
    timeline_start_hour = 10
    timeline_start_minute = 0
    interval_minutes = 5
    
    timeline_start_total_minutes = timeline_start_hour * 60 + timeline_start_minute
    
    for market, target_close in market_close_times.items():
        target_total_minutes = target_close['hour'] * 60 + target_close['minute']
        minutes_from_start = target_total_minutes - timeline_start_total_minutes
        expected_index = minutes_from_start // interval_minutes
        
        print(f"{market} Market Close:")
        print(f"  Target time: {target_close['hour']:02d}:{target_close['minute']:02d} AEST")
        print(f"  Timeline start: {timeline_start_hour:02d}:{timeline_start_minute:02d} AEST") 
        print(f"  Minutes from start: {minutes_from_start}")
        print(f"  Expected index: {expected_index}")
        
        # What timestamp would this be?
        expected_minutes_total = timeline_start_total_minutes + (expected_index * interval_minutes)
        expected_hour = expected_minutes_total // 60
        expected_minute = expected_minutes_total % 60
        print(f"  Calculated timestamp: {expected_hour:02d}:{expected_minute:02d} AEST")
        print()

if __name__ == "__main__":
    test_timeline_calculation()