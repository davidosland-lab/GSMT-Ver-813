#!/usr/bin/env python3
"""
Debug the timestamp filtering issue in the frontend
"""

from datetime import datetime, timezone
import pytz

def debug_timestamp_issue():
    """Debug the frontend timestamp calculation issue"""
    
    print("🔍 Debugging frontend timestamp filtering issue...")
    
    # Current time
    now_utc = datetime.now(timezone.utc)
    aest = pytz.timezone('Australia/Sydney')
    now_aest_correct = now_utc.astimezone(aest)
    
    print(f"\n🕐 Correct Time Calculation:")
    print(f"   UTC Now: {now_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"   AEST Now (correct): {now_aest_correct.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    # Simulate the INCORRECT frontend calculation (from app.js lines 832-834)
    aest_offset_hours = 10  # AEST is UTC+10
    now_aest_frontend = datetime.utcfromtimestamp(
        (now_utc.timestamp() + (aest_offset_hours * 60 * 60))
    )
    
    # Format as the frontend does (lines 837-842)
    current_aest_str_frontend = (
        f"{now_aest_frontend.year}-" +
        f"{now_aest_frontend.month:02d}-" +
        f"{now_aest_frontend.day:02d} " +
        f"{now_aest_frontend.hour:02d}:" +
        f"{now_aest_frontend.minute:02d}:" +
        f"{now_aest_frontend.second:02d}"
    )
    
    print(f"\n❌ Frontend's INCORRECT Calculation:")
    print(f"   AEST Frontend: {now_aest_frontend.strftime('%Y-%m-%d %H:%M:%S')} (NO TIMEZONE)")
    print(f"   Frontend String: '{current_aest_str_frontend}'")
    
    # Check for DST issues
    from datetime import timedelta
    is_dst = now_aest_correct.dst() != timedelta(0)
    actual_offset = now_aest_correct.utcoffset().total_seconds() / 3600
    
    print(f"\n🌏 AEST Timezone Analysis:")
    print(f"   DST Active: {is_dst}")
    print(f"   Actual UTC Offset: +{actual_offset:.0f} hours")
    print(f"   Frontend assumes: +10 hours (ALWAYS)")
    
    if is_dst:
        print(f"   🚨 PROBLEM: Frontend ignores DST - should be +{actual_offset:.0f}, not +10!")
    
    # Show the impact
    time_diff = abs((now_aest_frontend - now_aest_correct.replace(tzinfo=None)).total_seconds())
    print(f"\n⚠️ Time Difference Impact:")
    print(f"   Frontend vs Correct: {time_diff/3600:.1f} hours difference")
    
    # Simulate a data point timestamp around the problematic time (14:56 AEST)
    test_time_aest = now_aest_correct.replace(hour=14, minute=56, second=0, microsecond=0)
    test_time_str = test_time_aest.strftime('%Y-%m-%d %H:%M:%S')
    
    print(f"\n🧪 Test Case: Data point at 14:56 AEST")
    print(f"   Test timestamp: '{test_time_str}' (no AEST suffix)")
    print(f"   Frontend filter string: '{current_aest_str_frontend}'")
    print(f"   String comparison: '{test_time_str}' <= '{current_aest_str_frontend}' = {test_time_str <= current_aest_str_frontend}")
    
    # The issue: if DST affects the comparison or if there's timezone confusion
    if is_dst and actual_offset != 10:
        print(f"   🚨 ISSUE: DST causes incorrect filtering!")
        print(f"   Data that should be filtered out appears as 'future' data")
    
    # Check what happens with different offset
    if actual_offset != 10:
        correct_frontend = datetime.utcfromtimestamp(
            (now_utc.timestamp() + (actual_offset * 60 * 60))
        )
        correct_str = (
            f"{correct_frontend.year}-" +
            f"{correct_frontend.month:02d}-" +
            f"{correct_frontend.day:02d} " +
            f"{correct_frontend.hour:02d}:" +
            f"{correct_frontend.minute:02d}:" +
            f"{correct_frontend.second:02d}"
        )
        print(f"\n✅ Corrected Frontend Calculation:")
        print(f"   Should be: '{correct_str}'")
        print(f"   Test comparison: '{test_time_str}' <= '{correct_str}' = {test_time_str <= correct_str}")

if __name__ == "__main__":
    debug_timestamp_issue()