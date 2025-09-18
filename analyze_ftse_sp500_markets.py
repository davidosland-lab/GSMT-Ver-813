#!/usr/bin/env python3
"""
Analyze ^FTSE and ^GSPC market characteristics for prediction development
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
import pytz
import numpy as np
import pandas as pd
from real_market_data_service import RealMarketDataAggregator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def analyze_markets():
    """Analyze ^FTSE and ^GSPC market characteristics"""
    
    print("📊 Analyzing ^FTSE and ^GSPC market characteristics for prediction development...")
    
    # Market configurations
    markets = {
        '^FTSE': {
            'name': 'FTSE 100',
            'timezone': 'Europe/London', 
            'market_open_utc': 8,  # 08:00 UTC (09:00 CET/08:00 GMT)
            'market_close_utc': 16, # 16:00 UTC (17:00 CET/16:00 GMT)
            'currency': 'GBP'
        },
        '^GSPC': {
            'name': 'S&P 500',
            'timezone': 'America/New_York',
            'market_open_utc': 14,  # 14:30 UTC (09:30 EST/EDT)
            'market_close_utc': 21, # 21:00 UTC (16:00 EST/EDT) 
            'currency': 'USD'
        }
    }
    
    aggregator = RealMarketDataAggregator()
    
    market_analysis = {}
    
    for symbol, config in markets.items():
        print(f"\n🔍 Analyzing {symbol} ({config['name']})...")
        
        try:
            # Get market data
            market_data = await aggregator.get_real_market_data(symbol)
            
            if not market_data or not market_data.data_points:
                print(f"   ❌ No data available for {symbol}")
                continue
            
            # Convert to DataFrame for analysis
            data_points = []
            for point in market_data.data_points:
                tz = pytz.timezone(config['timezone'])
                local_time = point.timestamp.astimezone(tz)
                
                data_points.append({
                    'timestamp': point.timestamp,
                    'local_time': local_time,
                    'open': point.open,
                    'high': point.high,
                    'low': point.low,
                    'close': point.close,
                    'volume': point.volume,
                    'hour': local_time.hour,
                    'minute': local_time.minute
                })
            
            df = pd.DataFrame(data_points)
            
            # Calculate key statistics
            df['price_range'] = df['high'] - df['low']
            df['price_change'] = df['close'] - df['open']
            df['price_change_pct'] = (df['price_change'] / df['open']) * 100
            df['volatility'] = df['price_range'] / df['open'] * 100
            
            # Analysis results
            analysis = {
                'symbol': symbol,
                'name': config['name'],
                'data_points': len(df),
                'date_range': {
                    'start': df['timestamp'].min(),
                    'end': df['timestamp'].max()
                },
                'price_stats': {
                    'current_price': df['close'].iloc[-1],
                    'min_price': df['low'].min(),
                    'max_price': df['high'].max(),
                    'avg_price': df['close'].mean(),
                    'price_std': df['close'].std()
                },
                'volatility_stats': {
                    'avg_volatility': df['volatility'].mean(),
                    'max_volatility': df['volatility'].max(),
                    'volatility_std': df['volatility'].std()
                },
                'volume_stats': {
                    'avg_volume': df['volume'].mean(),
                    'max_volume': df['volume'].max(),
                    'zero_volume_pct': (df['volume'] == 0).sum() / len(df) * 100
                },
                'market_hours': {
                    'active_hours': df['hour'].value_counts().head(10).to_dict(),
                    'trading_start': config['market_open_utc'],
                    'trading_end': config['market_close_utc']
                },
                'prediction_factors': {
                    'trend_strength': abs(df['price_change_pct'].mean()),
                    'mean_reversion': df['price_change_pct'].std(),
                    'momentum': df['price_change_pct'].rolling(5).mean().std(),
                    'data_quality': (df['volume'] > 0).sum() / len(df)
                }
            }
            
            market_analysis[symbol] = analysis
            
            # Print analysis
            print(f"   📈 Data Points: {analysis['data_points']}")
            print(f"   💰 Current Price: {analysis['price_stats']['current_price']:.2f} {config['currency']}")
            print(f"   📊 Average Volatility: {analysis['volatility_stats']['avg_volatility']:.3f}%")
            print(f"   🔄 Trend Strength: {analysis['prediction_factors']['trend_strength']:.3f}%")
            print(f"   📉 Mean Reversion: {analysis['prediction_factors']['mean_reversion']:.3f}")
            print(f"   🎯 Data Quality: {analysis['prediction_factors']['data_quality']:.1%}")
            
        except Exception as e:
            print(f"   ❌ Error analyzing {symbol}: {e}")
    
    # Compare markets for prediction suitability
    print(f"\n🎯 Multi-Market Prediction Analysis:")
    
    if len(market_analysis) >= 2:
        ftse = market_analysis.get('^FTSE', {})
        sp500 = market_analysis.get('^GSPC', {})
        
        if ftse and sp500:
            print(f"\n📊 Comparative Analysis:")
            print(f"   FTSE Volatility: {ftse.get('volatility_stats', {}).get('avg_volatility', 0):.3f}%")
            print(f"   S&P 500 Volatility: {sp500.get('volatility_stats', {}).get('avg_volatility', 0):.3f}%")
            
            print(f"\n   FTSE Data Quality: {ftse.get('prediction_factors', {}).get('data_quality', 0):.1%}")
            print(f"   S&P 500 Data Quality: {sp500.get('prediction_factors', {}).get('data_quality', 0):.1%}")
            
            # Determine prediction strategy
            ftse_vol = ftse.get('volatility_stats', {}).get('avg_volatility', 0)
            sp500_vol = sp500.get('volatility_stats', {}).get('avg_volatility', 0)
            
            print(f"\n🤖 Prediction Strategy Recommendations:")
            
            if ftse_vol > sp500_vol:
                print(f"   • FTSE shows higher volatility - suitable for short-term predictions")
                print(f"   • S&P 500 shows lower volatility - suitable for trend following")
            else:
                print(f"   • S&P 500 shows higher volatility - suitable for short-term predictions") 
                print(f"   • FTSE shows lower volatility - suitable for trend following")
            
            # Market timing analysis
            print(f"\n🕐 Market Timing Analysis:")
            print(f"   • FTSE trading: 08:00-16:00 UTC (London)")
            print(f"   • S&P 500 trading: 14:30-21:00 UTC (New York)")
            print(f"   • Overlap period: 14:30-16:00 UTC (1.5 hours)")
            print(f"   • Sequential coverage: FTSE → Overlap → S&P 500")
    
    # Prediction model recommendations
    print(f"\n🧠 Prediction Model Recommendations:")
    
    for symbol, analysis in market_analysis.items():
        volatility = analysis.get('volatility_stats', {}).get('avg_volatility', 0)
        data_quality = analysis.get('prediction_factors', {}).get('data_quality', 0)
        
        print(f"\n   {symbol} ({analysis.get('name', 'Unknown')}):")
        
        if data_quality > 0.8:
            print(f"     ✅ High data quality - suitable for advanced ML models")
        elif data_quality > 0.5:
            print(f"     ⚠️ Moderate data quality - use robust models with filtering")
        else:
            print(f"     ❌ Low data quality - use simple models or improve data")
        
        if volatility > 1.0:
            print(f"     📈 High volatility - use LSTM/RNN for pattern recognition")
        elif volatility > 0.5:
            print(f"     📊 Moderate volatility - use ensemble methods")
        else:
            print(f"     📉 Low volatility - use linear regression or ARIMA")
    
    return market_analysis

if __name__ == "__main__":
    asyncio.run(analyze_markets())