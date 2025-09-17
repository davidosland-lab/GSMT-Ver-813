#!/usr/bin/env python3
"""
Live Market Data Simulator
Generates realistic OHLCV market data that behaves like live data
"""

import asyncio
import random
import math
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass 
class SimulatedDataPoint:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    symbol: str

class LiveMarketSimulator:
    """Generates realistic live market data with proper market behavior"""
    
    def __init__(self):
        # Base prices for major indices - Updated to realistic current market values (Sept 2024)
        self.base_prices = {
            # === MAJOR INDICES (Updated to current levels) ===
            '^GSPC': 5800.0,    # S&P 500 (currently around 5800-5900)
            '^IXIC': 18200.0,   # NASDAQ Composite (currently around 18000-18500) 
            '^DJI': 42000.0,    # Dow Jones (currently around 41000-42500)
            '^FTSE': 8200.0,    # FTSE 100 (currently around 8100-8300)
            '^GDAXI': 18900.0,  # DAX (currently around 18500-19200)
            '^N225': 38500.0,   # Nikkei 225 (currently around 38000-39500)
            '^AORD': 8250.0,    # ASX All Ordinaries (currently around 8200-8400) - CORRECTED
            '^RUT': 2200.0,     # Russell 2000 (currently around 2150-2250)
            '^VIX': 15.0,       # VIX Volatility Index (typically 12-25)
            
            # === AUSTRALIAN MARKET ===
            '^AXJO': 8250.0,    # ASX 200 (currently around 8200-8400)
            '^AORD': 8250.0,    # ASX All Ordinaries (currently around 8200-8400)
            
            # === MAJOR TECH STOCKS (Updated to current levels) ===
            'AAPL': 225.0,      # Apple (currently around 220-235)
            'GOOGL': 165.0,     # Alphabet/Google (currently around 160-170)
            'MSFT': 420.0,      # Microsoft (currently around 415-430)
            'AMZN': 185.0,      # Amazon (currently around 180-190)
            'TSLA': 240.0,      # Tesla (currently around 230-250)
            'META': 500.0,      # Meta (currently around 490-520)
            'NVDA': 135.0,      # NVIDIA (currently around 130-140 post-split)
            'NFLX': 700.0,      # Netflix (currently around 690-720)
            
            # === MAJOR FINANCIALS ===
            'JPM': 220.0,       # JPMorgan Chase (currently around 215-225)
            'V': 280.0,         # Visa (currently around 275-285)
            'MA': 480.0,        # Mastercard (currently around 470-490)
            'BAC': 42.0,        # Bank of America (currently around 40-44)
            
            # === COMMODITIES & CRYPTO PROXIES ===
            'GLD': 190.0,       # Gold ETF (currently around 185-195)
            'BTC-USD': 65000.0, # Bitcoin (highly volatile, around 60k-70k)
        }
        
        # Current prices (will evolve over time)
        self.current_prices = self.base_prices.copy()
        
        # Market state (trending up, down, or sideways)
        self.market_trend = 0.0  # -1 to 1, where 1 is strong uptrend
        self.volatility = 0.005   # Base volatility (0.5% - more realistic)
        
        # Time tracking
        self.last_update = datetime.now(timezone.utc)
    
    def _get_market_hours_multiplier(self, timestamp: datetime) -> float:
        """Adjust volatility based on market hours (higher during market open)"""
        hour = timestamp.hour
        
        # US market hours (9:30 AM - 4:00 PM EST = 14:30 - 21:00 UTC)
        if 14 <= hour <= 21:
            return 1.5  # Higher volatility during US market hours
        # European market hours (8:00 AM - 4:30 PM GMT = 8:00 - 16:30 UTC)
        elif 8 <= hour <= 16:
            return 1.2  # Moderate volatility during European hours
        # Asian market hours (overlap periods)
        elif 0 <= hour <= 6:
            return 1.1  # Lower volatility during Asian hours
        else:
            return 0.5  # Very low volatility during off-hours
    
    def _generate_realistic_ohlc(self, symbol: str, base_price: float, time_interval_minutes: int = 5) -> SimulatedDataPoint:
        """Generate realistic OHLC data for a time interval"""
        
        now = datetime.now(timezone.utc)
        
        # Market hours multiplier
        hours_multiplier = self._get_market_hours_multiplier(now)
        
        # Evolve market trend slowly
        self.market_trend += random.gauss(0, 0.001)
        self.market_trend = max(-1, min(1, self.market_trend))
        
        # Calculate price change (more conservative)
        trend_component = self.market_trend * 0.0002  # Reduced trend influence
        random_component = random.gauss(0, self.volatility * hours_multiplier * 0.5)  # Reduced random volatility
        total_change = (trend_component + random_component) * (time_interval_minutes / 5.0)
        
        # Current price for this symbol
        current_price = self.current_prices[symbol]
        
        # Generate OHLC with realistic relationships
        open_price = current_price
        
        # Price movement during the interval
        price_range = abs(total_change) * current_price
        direction = 1 if total_change > 0 else -1
        
        # Generate intraday high and low (more conservative)
        high_move = random.uniform(0.1, 0.3) * price_range * hours_multiplier
        low_move = random.uniform(0.1, 0.3) * price_range * hours_multiplier
        
        if direction > 0:  # Bullish candle
            close_price = open_price + (price_range * random.uniform(0.2, 0.6))
            high_price = max(open_price, close_price) + high_move
            low_price = min(open_price, close_price) - low_move * 0.3
        else:  # Bearish candle
            close_price = open_price - (price_range * random.uniform(0.2, 0.6))
            high_price = max(open_price, close_price) + high_move * 0.3
            low_price = min(open_price, close_price) - low_move
        
        # Ensure prices don't go negative or too extreme
        min_price = current_price * 0.95  # Max 5% drop per interval
        max_price = current_price * 1.05  # Max 5% rise per interval
        
        close_price = max(min_price, min(max_price, close_price))
        high_price = max(open_price, close_price, min(max_price, high_price))
        low_price = min(open_price, close_price, max(min_price, low_price))
        
        # Ensure OHLC relationships are correct
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)
        
        # Update current price
        self.current_prices[symbol] = close_price
        
        # Generate realistic volume
        base_volume = 1000000 if symbol.startswith('^') else 10000000  # Lower volume for indices
        volume_multiplier = hours_multiplier * random.uniform(0.5, 2.0)
        volume = int(base_volume * volume_multiplier)
        
        return SimulatedDataPoint(
            timestamp=now,
            open=round(open_price, 2),
            high=round(high_price, 2),
            low=round(low_price, 2),
            close=round(close_price, 2),
            volume=volume,
            symbol=symbol
        )
    
    def generate_historical_data(self, symbol: str, hours_back: int = 24, interval_minutes: int = 5) -> List[SimulatedDataPoint]:
        """Generate historical data for the past N hours"""
        
        if symbol not in self.base_prices:
            return []
        
        data_points = []
        current_time = datetime.now(timezone.utc)
        
        # Save current state
        original_price = self.current_prices[symbol] 
        original_trend = self.market_trend
        
        # Start from base price for historical generation
        self.current_prices[symbol] = self.base_prices[symbol]
        
        # Generate data points going backwards in time, then reverse
        num_points = (hours_back * 60) // interval_minutes
        
        for i in range(num_points):
            timestamp = current_time - timedelta(minutes=(num_points - i) * interval_minutes)
            
            # Temporarily set market trend for historical consistency
            old_trend = self.market_trend
            self.market_trend = math.sin(i * 0.1) * 0.3  # Gentle wave pattern
            
            point = self._generate_realistic_ohlc(symbol, self.current_prices[symbol], interval_minutes)
            point.timestamp = timestamp
            data_points.append(point)
            
            self.market_trend = original_trend
        
        # Restore original state
        self.current_prices[symbol] = original_price
        self.market_trend = original_trend
        
        return data_points
    
    async def get_live_data(self, symbol: str, interval_minutes: int = 5) -> Optional[List[SimulatedDataPoint]]:
        """Get live data for a symbol (generates realistic data)"""
        
        if symbol not in self.base_prices:
            logger.warning(f"Symbol {symbol} not supported in simulator")
            return None
        
        try:
            # Generate recent historical data (last 24 hours)
            historical_data = self.generate_historical_data(symbol, hours_back=24, interval_minutes=interval_minutes)
            
            # Add current live data point
            current_point = self._generate_realistic_ohlc(symbol, self.current_prices[symbol], interval_minutes)
            historical_data.append(current_point)
            
            logger.info(f"✅ Generated {len(historical_data)} simulated data points for {symbol}")
            return historical_data
            
        except Exception as e:
            logger.error(f"Error generating simulated data for {symbol}: {e}")
            return None

# Global simulator instance
live_simulator = LiveMarketSimulator()