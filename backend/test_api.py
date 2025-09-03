#!/usr/bin/env python3
"""
Test script for Global Markets Tracker Backend API
Run this to verify your deployed backend is working correctly
"""

import requests
import json
from datetime import datetime, timedelta
import sys

def test_api(base_url):
    """Test all API endpoints"""
    
    print(f"🧪 Testing Global Markets Tracker API at: {base_url}")
    print("=" * 60)
    
    # Test 1: Health Check
    print("\n1️⃣ Testing Health Check...")
    try:
        response = requests.get(f"{base_url}/api/health", timeout=10)
        if response.status_code == 200:
            print("✅ Health check passed")
            data = response.json()
            print(f"   Status: {data.get('status')}")
            print(f"   Timestamp: {data.get('timestamp')}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False
    
    # Test 2: Symbols List
    print("\n2️⃣ Testing Symbols Endpoint...")
    try:
        response = requests.get(f"{base_url}/api/symbols", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Symbols endpoint working")
            print(f"   Available symbols: {data.get('count')}")
            # Show first few symbols
            symbols = list(data.get('symbols', {}).items())[:5]
            for symbol, name in symbols:
                print(f"   {symbol}: {name}")
        else:
            print(f"❌ Symbols endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Symbols endpoint error: {e}")
    
    # Test 3: Single Stock Data (ASX 200)
    print("\n3️⃣ Testing Single Stock Data (ASX 200)...")
    test_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')  # Yesterday
    try:
        response = requests.get(
            f"{base_url}/api/stock/^AXJO", 
            params={'interval': '5m', 'date': test_date},
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            print(f"✅ ASX 200 data retrieved")
            print(f"   Symbol: {data.get('symbol')}")
            print(f"   Name: {data.get('name')}")
            print(f"   Data points: {data.get('count')}")
            print(f"   Date: {data.get('date')}")
            print(f"   Interval: {data.get('interval')}")
            
            # Show sample data point
            if data.get('data') and len(data['data']) > 0:
                sample = data['data'][0]
                print(f"   Sample data: {sample.get('time')} | Close: {sample.get('close')}")
        else:
            print(f"❌ ASX 200 data failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ ASX 200 data error: {e}")
    
    # Test 4: US Stock Data (S&P 500)
    print("\n4️⃣ Testing US Stock Data (S&P 500)...")
    try:
        response = requests.get(
            f"{base_url}/api/stock/^GSPC", 
            params={'interval': '1h', 'date': test_date},
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            print(f"✅ S&P 500 data retrieved")
            print(f"   Data points: {data.get('count')}")
            if data.get('data') and len(data['data']) > 0:
                sample = data['data'][0]
                print(f"   Sample data: {sample.get('time')} | Close: {sample.get('close')}")
        else:
            print(f"❌ S&P 500 data failed: {response.status_code}")
    except Exception as e:
        print(f"❌ S&P 500 data error: {e}")
    
    # Test 5: Bulk Data Request
    print("\n5️⃣ Testing Bulk Data Request...")
    try:
        payload = {
            "symbols": ["^AXJO", "^GSPC", "^N225"],
            "interval": "1h",
            "date": test_date
        }
        response = requests.post(
            f"{base_url}/api/bulk",
            json=payload,
            timeout=60
        )
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Bulk data request successful")
            print(f"   Requested symbols: {data.get('requested_symbols')}")
            results = data.get('results', {})
            for symbol, info in results.items():
                count = len(info.get('data', []))
                print(f"   {symbol}: {count} data points")
        else:
            print(f"❌ Bulk data request failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Bulk data request error: {e}")
    
    # Test 6: Error Handling
    print("\n6️⃣ Testing Error Handling...")
    try:
        response = requests.get(f"{base_url}/api/stock/INVALID_SYMBOL", timeout=10)
        print(f"✅ Error handling test completed (status: {response.status_code})")
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 API Testing Complete!")
    print("\nNext steps:")
    print("1. Update your frontend with this backend URL")
    print("2. Test the frontend with 'YFinance (live data)' option")
    print("3. Check browser console for any CORS errors")
    
    return True

def main():
    """Main test function"""
    if len(sys.argv) > 1:
        base_url = sys.argv[1].rstrip('/')
    else:
        print("Usage: python test_api.py <API_BASE_URL>")
        print("Example: python test_api.py https://your-app.railway.app")
        print("\nOr test locally: python test_api.py http://localhost:5000")
        sys.exit(1)
    
    test_api(base_url)

if __name__ == "__main__":
    main()