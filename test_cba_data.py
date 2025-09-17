#!/usr/bin/env python3
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def test_cba_data():
    print('🔍 Testing direct yfinance CBA.AX data retrieval...')
    ticker = yf.Ticker('CBA.AX')
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    hist = ticker.history(start=start_date, end=end_date)

    if not hist.empty:
        latest_price = hist['Close'].iloc[-1]
        min_price = hist['Close'].min()
        max_price = hist['Close'].max()
        print(f'✅ CBA.AX Latest Close Price: ${latest_price:.2f}')
        print(f'📊 Price Range (90 days): ${min_price:.2f} - ${max_price:.2f}')
        print(f'📈 Sample recent prices:')
        recent_prices = hist['Close'].tail()
        for date, price in recent_prices.items():
            print(f'  {date.strftime("%Y-%m-%d")}: ${price:.2f}')
        return True, latest_price, min_price, max_price
    else:
        print('❌ No data retrieved for CBA.AX')
        return False, None, None, None

if __name__ == "__main__":
    test_cba_data()