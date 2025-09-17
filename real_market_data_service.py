#!/usr/bin/env python3
"""
Real Market Data Service
Fetches actual live market data from multiple free APIs
NO SIMULATED DATA - Real market feeds only
"""

import aiohttp
import asyncio
import json
import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

@dataclass
class RealMarketDataPoint:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    symbol: str
    source: str

@dataclass
class RealMarketData:
    symbol: str
    data_points: List[RealMarketDataPoint]
    sources_used: List[str]
    last_updated: datetime

class YahooFinanceRealAPI:
    """Yahoo Finance API with improved error handling and multiple endpoints"""
    
    def __init__(self):
        self.base_urls = [
            "https://query1.finance.yahoo.com",
            "https://query2.finance.yahoo.com",
            "https://finance.yahoo.com"
        ]
        
    async def get_real_data(self, symbol: str, period: str = "1d", interval: str = "5m") -> Optional[List[RealMarketDataPoint]]:
        """Get real market data from Yahoo Finance"""
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0'
        }
        
        # Convert common symbols to Yahoo format
        yahoo_symbol = self._convert_symbol(symbol)
        
        for base_url in self.base_urls:
            try:
                # Try the v8 chart API first
                url = f"{base_url}/v8/finance/chart/{yahoo_symbol}"
                params = {
                    'period1': int((datetime.now() - timedelta(days=2)).timestamp()),
                    'period2': int(datetime.now().timestamp()),
                    'interval': interval,
                    'includePrePost': 'false',
                    'events': 'div%2Csplit'
                }
                
                timeout = aiohttp.ClientTimeout(total=30)
                async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            result = await self._parse_yahoo_response(data, symbol)
                            if result and len(result) > 0:
                                logger.info(f"✅ Yahoo Finance: Got {len(result)} real data points for {symbol}")
                                return result
                        else:
                            logger.warning(f"Yahoo Finance {response.status} for {symbol} at {base_url}")
                            
            except Exception as e:
                logger.warning(f"Yahoo Finance error for {symbol} at {base_url}: {e}")
                continue
        
        logger.error(f"❌ All Yahoo Finance endpoints failed for {symbol}")
        return None
    
    def _convert_symbol(self, symbol: str) -> str:
        """Convert symbols to Yahoo Finance format"""
        symbol_mapping = {
            # Keep original symbols
            '^GSPC': '^GSPC', '^IXIC': '^IXIC', '^DJI': '^DJI',
            '^FTSE': '^FTSE', '^GDAXI': '^GDAXI', '^N225': '^N225',
            '^AORD': '^AORD', '^AXJO': '^AXJO',
            # US Stocks
            'AAPL': 'AAPL', 'GOOGL': 'GOOGL', 'MSFT': 'MSFT',
            'AMZN': 'AMZN', 'TSLA': 'TSLA', 'META': 'META', 'NVDA': 'NVDA'
        }
        return symbol_mapping.get(symbol, symbol)
    
    async def _parse_yahoo_response(self, data: dict, symbol: str) -> List[RealMarketDataPoint]:
        """Parse Yahoo Finance API response"""
        try:
            if 'chart' not in data or 'result' not in data['chart']:
                logger.error(f"Invalid Yahoo response structure for {symbol}")
                return []
                
            result = data['chart']['result'][0]
            
            if 'timestamp' not in result or 'indicators' not in result:
                logger.error(f"Missing timestamp or indicators in Yahoo response for {symbol}")
                return []
                
            timestamps = result['timestamp']
            indicators = result['indicators']['quote'][0]
            
            data_points = []
            for i, ts in enumerate(timestamps):
                try:
                    # Check if we have valid OHLC data
                    open_price = indicators.get('open', [None])[i]
                    high_price = indicators.get('high', [None])[i]
                    low_price = indicators.get('low', [None])[i]
                    close_price = indicators.get('close', [None])[i]
                    volume = indicators.get('volume', [0])[i] or 0
                    
                    # Skip if any OHLC value is None
                    if any(v is None for v in [open_price, high_price, low_price, close_price]):
                        continue
                    
                    timestamp = datetime.fromtimestamp(ts, tz=timezone.utc)
                    
                    data_points.append(RealMarketDataPoint(
                        timestamp=timestamp,
                        open=float(open_price),
                        high=float(high_price),
                        low=float(low_price),
                        close=float(close_price),
                        volume=int(volume),
                        symbol=symbol,
                        source="Yahoo Finance"
                    ))
                    
                except (ValueError, TypeError, IndexError) as e:
                    logger.warning(f"Skipping invalid data point {i} for {symbol}: {e}")
                    continue
            
            return data_points
            
        except Exception as e:
            logger.error(f"Error parsing Yahoo Finance data for {symbol}: {e}")
            return []

class FinnhubRealAPI:
    """Finnhub.io free tier API for real market data"""
    
    def __init__(self):
        self.api_key = os.getenv('FINNHUB_API_KEY', 'demo')  # Free tier
        self.base_url = "https://finnhub.io/api/v1"
        
    async def get_real_data(self, symbol: str) -> Optional[List[RealMarketDataPoint]]:
        """Get real market data from Finnhub"""
        
        if self.api_key == 'demo':
            logger.info(f"Using Finnhub demo key for {symbol}")
        
        try:
            # Convert to Finnhub format
            finnhub_symbol = self._convert_symbol(symbol)
            
            # Get current timestamp and 2 days ago
            end_time = int(datetime.now().timestamp())
            start_time = int((datetime.now() - timedelta(days=2)).timestamp())
            
            url = f"{self.base_url}/stock/candle"
            params = {
                'symbol': finnhub_symbol,
                'resolution': '5',  # 5-minute intervals
                'from': start_time,
                'to': end_time,
                'token': self.api_key
            }
            
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        result = await self._parse_finnhub_response(data, symbol)
                        if result and len(result) > 0:
                            logger.info(f"✅ Finnhub: Got {len(result)} real data points for {symbol}")
                            return result
                    else:
                        logger.warning(f"Finnhub {response.status} for {symbol}")
                        
        except Exception as e:
            logger.warning(f"Finnhub error for {symbol}: {e}")
            
        return None
    
    def _convert_symbol(self, symbol: str) -> str:
        """Convert symbols to Finnhub format"""
        # Finnhub uses different symbol formats
        symbol_mapping = {
            '^GSPC': 'SPY',  # Use SPY ETF as proxy for S&P 500
            '^IXIC': 'QQQ',  # Use QQQ ETF as proxy for NASDAQ
            '^DJI': 'DIA',   # Use DIA ETF as proxy for Dow Jones
            # Individual stocks stay the same
            'AAPL': 'AAPL', 'GOOGL': 'GOOGL', 'MSFT': 'MSFT',
            'AMZN': 'AMZN', 'TSLA': 'TSLA', 'META': 'META', 'NVDA': 'NVDA'
        }
        return symbol_mapping.get(symbol, symbol)
    
    async def _parse_finnhub_response(self, data: dict, symbol: str) -> List[RealMarketDataPoint]:
        """Parse Finnhub API response"""
        try:
            if data.get('s') != 'ok':
                logger.warning(f"Finnhub API error for {symbol}: {data}")
                return []
            
            timestamps = data.get('t', [])
            opens = data.get('o', [])
            highs = data.get('h', [])
            lows = data.get('l', [])
            closes = data.get('c', [])
            volumes = data.get('v', [])
            
            data_points = []
            for i in range(len(timestamps)):
                try:
                    timestamp = datetime.fromtimestamp(timestamps[i], tz=timezone.utc)
                    
                    data_points.append(RealMarketDataPoint(
                        timestamp=timestamp,
                        open=float(opens[i]),
                        high=float(highs[i]),
                        low=float(lows[i]),
                        close=float(closes[i]),
                        volume=int(volumes[i]),
                        symbol=symbol,
                        source="Finnhub"
                    ))
                    
                except (ValueError, TypeError, IndexError) as e:
                    logger.warning(f"Skipping invalid Finnhub data point {i} for {symbol}: {e}")
                    continue
            
            return data_points
            
        except Exception as e:
            logger.error(f"Error parsing Finnhub data for {symbol}: {e}")
            return []

class AlphaVantageRealAPI:
    """Alpha Vantage free tier API for real market data"""
    
    def __init__(self):
        self.api_key = os.getenv('ALPHA_VANTAGE_API_KEY', 'demo')
        self.base_url = "https://www.alphavantage.co/query"
        
    async def get_real_data(self, symbol: str) -> Optional[List[RealMarketDataPoint]]:
        """Get real market data from Alpha Vantage"""
        
        if self.api_key == 'demo':
            logger.info(f"Using Alpha Vantage demo key for {symbol}")
            return None  # Demo key doesn't work well
        
        try:
            # Use intraday 5min data
            params = {
                'function': 'TIME_SERIES_INTRADAY',
                'symbol': symbol,
                'interval': '5min',
                'apikey': self.api_key,
                'outputsize': 'compact'
            }
            
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        result = await self._parse_alphavantage_response(data, symbol)
                        if result and len(result) > 0:
                            logger.info(f"✅ Alpha Vantage: Got {len(result)} real data points for {symbol}")
                            return result
                    else:
                        logger.warning(f"Alpha Vantage {response.status} for {symbol}")
                        
        except Exception as e:
            logger.warning(f"Alpha Vantage error for {symbol}: {e}")
            
        return None
    
    async def _parse_alphavantage_response(self, data: dict, symbol: str) -> List[RealMarketDataPoint]:
        """Parse Alpha Vantage API response"""
        try:
            time_series_key = "Time Series (5min)"
            if time_series_key not in data:
                logger.warning(f"Alpha Vantage: No time series data for {symbol}")
                return []
            
            time_series = data[time_series_key]
            data_points = []
            
            for timestamp_str, ohlcv in time_series.items():
                try:
                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                    
                    data_points.append(RealMarketDataPoint(
                        timestamp=timestamp,
                        open=float(ohlcv['1. open']),
                        high=float(ohlcv['2. high']),
                        low=float(ohlcv['3. low']),
                        close=float(ohlcv['4. close']),
                        volume=int(ohlcv['5. volume']),
                        symbol=symbol,
                        source="Alpha Vantage"
                    ))
                    
                except (ValueError, TypeError, KeyError) as e:
                    logger.warning(f"Skipping invalid Alpha Vantage data point for {symbol}: {e}")
                    continue
            
            # Sort by timestamp (newest first)
            data_points.sort(key=lambda x: x.timestamp, reverse=True)
            return data_points
            
        except Exception as e:
            logger.error(f"Error parsing Alpha Vantage data for {symbol}: {e}")
            return []

class RealMarketDataAggregator:
    """Aggregates real market data from multiple APIs"""
    
    def __init__(self):
        self.yahoo_api = YahooFinanceRealAPI()
        self.finnhub_api = FinnhubRealAPI()
        self.alphavantage_api = AlphaVantageRealAPI()
        self.cache: Dict[str, Tuple[RealMarketData, datetime]] = {}
        self.cache_duration = timedelta(minutes=2)  # Short cache for real data
        
    async def get_real_market_data(self, symbol: str) -> Optional[RealMarketData]:
        """Get real market data from multiple sources with priority order"""
        
        # Check cache first
        cache_key = f"real_{symbol}"
        if cache_key in self.cache:
            cached_data, cached_time = self.cache[cache_key]
            if datetime.now() - cached_time < self.cache_duration:
                logger.info(f"📋 Returning cached real data for {symbol}")
                return cached_data
        
        logger.info(f"🌐 Fetching real market data for {symbol}...")
        
        # Try APIs in priority order
        apis = [
            ("Yahoo Finance", self.yahoo_api),
            ("Finnhub", self.finnhub_api),
            ("Alpha Vantage", self.alphavantage_api),
        ]
        
        for api_name, api in apis:
            try:
                logger.info(f"🔍 Trying {api_name} for {symbol}...")
                data_points = await api.get_real_data(symbol)
                
                if data_points and len(data_points) > 0:
                    # Filter to recent data only (last 24-48 hours)
                    cutoff = datetime.now(timezone.utc) - timedelta(hours=48)
                    recent_points = [p for p in data_points if p.timestamp >= cutoff]
                    
                    if len(recent_points) >= 10:  # Need minimum data points
                        market_data = RealMarketData(
                            symbol=symbol,
                            data_points=recent_points,
                            sources_used=[api_name],
                            last_updated=datetime.now()
                        )
                        
                        # Cache the result
                        self.cache[cache_key] = (market_data, datetime.now())
                        
                        logger.info(f"✅ Got {len(recent_points)} real data points for {symbol} from {api_name}")
                        return market_data
                    else:
                        logger.warning(f"⚠️ {api_name}: Only got {len(recent_points)} recent points for {symbol}")
                
            except Exception as e:
                logger.error(f"❌ {api_name} failed for {symbol}: {e}")
                continue
        
        logger.error(f"❌ All real data APIs failed for {symbol}")
        return None

# Global instance
real_market_aggregator = RealMarketDataAggregator()