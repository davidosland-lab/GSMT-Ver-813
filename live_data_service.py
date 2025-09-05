"""
Live Market Data Service
Integrates with multiple data providers for real-time market data
"""

import aiohttp
import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
import random
import time

logger = logging.getLogger(__name__)

@dataclass
class LiveDataPoint:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    symbol: str

class DataCache:
    """Simple in-memory cache for API responses"""
    def __init__(self, cache_minutes: int = 5):
        self.cache = {}
        self.cache_duration = timedelta(minutes=cache_minutes)
    
    def get(self, key: str) -> Optional[any]:
        if key in self.cache:
            data, timestamp = self.cache[key]
            if datetime.now() - timestamp < self.cache_duration:
                return data
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, data: any):
        self.cache[key] = (data, datetime.now())
    
    def clear(self):
        self.cache.clear()

class RateLimiter:
    """Rate limiter for API calls"""
    def __init__(self, max_calls_per_minute: int = 5):
        self.max_calls = max_calls_per_minute
        self.calls = []
    
    async def wait_if_needed(self):
        now = datetime.now()
        # Remove calls older than 1 minute
        self.calls = [call_time for call_time in self.calls if now - call_time < timedelta(minutes=1)]
        
        if len(self.calls) >= self.max_calls:
            # Wait until we can make another call
            oldest_call = min(self.calls)
            wait_time = 60 - (now - oldest_call).seconds
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting {wait_time} seconds")
                await asyncio.sleep(wait_time)
        
        self.calls.append(now)

class LiveDataService:
    """Main service for fetching live market data"""
    
    def __init__(self):
        self.cache = DataCache(cache_minutes=int(os.getenv('DATA_CACHE_MINUTES', 5)))
        self.rate_limiter = RateLimiter(max_calls_per_minute=int(os.getenv('MAX_API_CALLS_PER_MINUTE', 4)))
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.twelve_data_key = os.getenv('TWELVE_DATA_API_KEY')
        self.use_yahoo = os.getenv('USE_YAHOO_FINANCE', 'true').lower() == 'true'
        self.fallback_to_demo = os.getenv('FALLBACK_TO_DEMO', 'true').lower() == 'true'
        
        # Symbol mapping for different providers
        self.symbol_mapping = {
            # Yahoo Finance symbols
            '^FTSE': 'UKX.L',  # FTSE 100
            '^GSPC': '^GSPC',   # S&P 500
            '^IXIC': '^IXIC',   # NASDAQ
            '^DJI': '^DJI',     # Dow Jones
            '^N225': '^N225',   # Nikkei 225
            '^HSI': '^HSI',     # Hang Seng
            '^AXJO': '^AXJO',   # ASX 200
            '^GDAXI': '^GDAXI', # DAX
            '^FCHI': '^FCHI',   # CAC 40
        }
    
    async def get_yahoo_finance_data(self, symbol: str) -> Optional[List[LiveDataPoint]]:
        """Fetch intraday data from Yahoo Finance"""
        try:
            # Use yfinance-like API approach
            yahoo_symbol = self.symbol_mapping.get(symbol, symbol)
            
            # Yahoo Finance API endpoint (unofficial)
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}"
            params = {
                'interval': '1h',  # 1-hour intervals
                'range': '1d',     # 1 day
                'includePrePost': 'false'
            }
            
            async with aiohttp.ClientSession() as session:
                await self.rate_limiter.wait_if_needed()
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_yahoo_response(data, symbol)
                    else:
                        logger.warning(f"Yahoo Finance API error for {symbol}: {response.status}")
                        return None
        
        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance data for {symbol}: {e}")
            return None
    
    def _parse_yahoo_response(self, data: dict, symbol: str) -> List[LiveDataPoint]:
        """Parse Yahoo Finance API response"""
        try:
            result = data['chart']['result'][0]
            timestamps = result['timestamp']
            indicators = result['indicators']['quote'][0]
            
            data_points = []
            for i, ts in enumerate(timestamps):
                timestamp = datetime.fromtimestamp(ts, tz=timezone.utc)
                
                # Skip if any price data is None
                if any(v is None for v in [
                    indicators['open'][i], 
                    indicators['high'][i], 
                    indicators['low'][i], 
                    indicators['close'][i]
                ]):
                    continue
                
                data_points.append(LiveDataPoint(
                    timestamp=timestamp,
                    open=float(indicators['open'][i]),
                    high=float(indicators['high'][i]),
                    low=float(indicators['low'][i]),
                    close=float(indicators['close'][i]),
                    volume=int(indicators['volume'][i] or 0),
                    symbol=symbol
                ))
            
            return data_points
        
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Error parsing Yahoo Finance response for {symbol}: {e}")
            return []
    
    async def get_alpha_vantage_data(self, symbol: str) -> Optional[List[LiveDataPoint]]:
        """Fetch intraday data from Alpha Vantage"""
        if not self.alpha_vantage_key or self.alpha_vantage_key == 'demo':
            return None
        
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'TIME_SERIES_INTRADAY',
                'symbol': symbol,
                'interval': '60min',
                'apikey': self.alpha_vantage_key,
                'outputsize': 'compact'
            }
            
            async with aiohttp.ClientSession() as session:
                await self.rate_limiter.wait_if_needed()
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_alpha_vantage_response(data, symbol)
                    else:
                        logger.warning(f"Alpha Vantage API error for {symbol}: {response.status}")
                        return None
        
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage data for {symbol}: {e}")
            return None
    
    def _parse_alpha_vantage_response(self, data: dict, symbol: str) -> List[LiveDataPoint]:
        """Parse Alpha Vantage API response"""
        try:
            time_series = data['Time Series (60min)']
            
            data_points = []
            for timestamp_str, values in time_series.items():
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
                
                data_points.append(LiveDataPoint(
                    timestamp=timestamp,
                    open=float(values['1. open']),
                    high=float(values['2. high']),
                    low=float(values['3. low']),
                    close=float(values['4. close']),
                    volume=int(values['5. volume']),
                    symbol=symbol
                ))
            
            return sorted(data_points, key=lambda x: x.timestamp)
        
        except (KeyError, ValueError) as e:
            logger.error(f"Error parsing Alpha Vantage response for {symbol}: {e}")
            return []
    
    async def get_live_data_for_symbol(self, symbol: str) -> Optional[List[LiveDataPoint]]:
        """Get live data for a single symbol, trying multiple providers"""
        
        # Check cache first
        cache_key = f"live_data_{symbol}"
        cached_data = self.cache.get(cache_key)
        if cached_data:
            return cached_data
        
        # Try Yahoo Finance first (most reliable for international indices)
        if self.use_yahoo:
            data = await self.get_yahoo_finance_data(symbol)
            if data:
                self.cache.set(cache_key, data)
                return data
        
        # Try Alpha Vantage as backup
        if self.alpha_vantage_key and self.alpha_vantage_key != 'demo':
            data = await self.get_alpha_vantage_data(symbol)
            if data:
                self.cache.set(cache_key, data)
                return data
        
        # Return None if no data available
        logger.warning(f"No live data available for {symbol}")
        return None
    
    async def get_live_data_batch(self, symbols: List[str]) -> Dict[str, Optional[List[LiveDataPoint]]]:
        """Get live data for multiple symbols"""
        tasks = [self.get_live_data_for_symbol(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        data_dict = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"Error fetching data for {symbol}: {result}")
                data_dict[symbol] = None
            else:
                data_dict[symbol] = result
        
        return data_dict
    
    def generate_demo_data_for_symbol(self, symbol: str, base_price: float) -> List[LiveDataPoint]:
        """Generate demo data as fallback"""
        data_points = []
        base_time = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        current_price = base_price
        
        for hour in range(24):
            timestamp = base_time + timedelta(hours=hour)
            
            # Simple random walk
            change = random.gauss(0, 0.02)  # 2% volatility
            current_price *= (1 + change)
            
            high_var = abs(random.gauss(0, 0.005))
            low_var = abs(random.gauss(0, 0.005))
            
            high = current_price * (1 + high_var)
            low = current_price * (1 - low_var)
            open_price = low + (high - low) * random.uniform(0.2, 0.8)
            
            data_points.append(LiveDataPoint(
                timestamp=timestamp,
                open=open_price,
                high=high,
                low=low,
                close=current_price,
                volume=random.randint(1000000, 5000000),
                symbol=symbol
            ))
        
        return data_points

# Global instance
live_data_service = LiveDataService()