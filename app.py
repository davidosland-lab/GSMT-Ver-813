"""
Global Stock Market Tracker - Local Deployment
24-Hour UTC Timeline Focus for Global Stock Indices with Live Data
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime, timedelta, timezone
from enum import Enum
import pytz
import os
import logging
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import multi-source live data service
from multi_source_data_service import multi_source_aggregator, LiveDataPoint, MarketData

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Global Stock Market Tracker",
    description="24-Hour UTC Timeline for Global Stock Indices with Live Data",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Configuration - NO DEMO DATA
LIVE_DATA_ENABLED = os.getenv('LIVE_DATA_ENABLED', 'true').lower() == 'true'
REQUIRE_LIVE_DATA = os.getenv('REQUIRE_LIVE_DATA', 'true').lower() == 'true'

# CORS middleware - allow local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local deployment
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Enums - Simplified for 24-hour focus
class TimePeriod(str, Enum):
    HOUR_24 = "24h"

class ChartType(str, Enum):
    PERCENTAGE = "percentage"
    PRICE = "price"

# Pydantic models
class MarketDataPoint(BaseModel):
    timestamp: str
    timestamp_ms: int
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: int
    percentage_change: Optional[float] = None
    market_open: Optional[bool] = None

class SymbolInfo(BaseModel):
    symbol: str
    name: str
    market: str
    category: str
    currency: str = "USD"

class AnalysisRequest(BaseModel):
    symbols: List[str] = Field(..., min_items=1, max_items=20)
    chart_type: ChartType = ChartType.PERCENTAGE

# Removed CandlestickRequest - focusing on 24h timeline only

class AnalysisResponse(BaseModel):
    success: bool
    data: Dict[str, List[MarketDataPoint]]
    metadata: Dict[str, SymbolInfo]
    chart_type: str
    timestamp: str
    total_symbols: int
    successful_symbols: int
    market_hours: Dict[str, Dict[str, int]]

# Comprehensive symbols database
SYMBOLS_DB = {
    # US Indices
    "^GSPC": SymbolInfo(symbol="^GSPC", name="S&P 500", market="US", category="Index"),
    "^IXIC": SymbolInfo(symbol="^IXIC", name="NASDAQ Composite", market="US", category="Index"),
    "^DJI": SymbolInfo(symbol="^DJI", name="Dow Jones Industrial Average", market="US", category="Index"),
    "^RUT": SymbolInfo(symbol="^RUT", name="Russell 2000", market="US", category="Index"),
    
    # US Tech Stocks
    "AAPL": SymbolInfo(symbol="AAPL", name="Apple Inc.", market="US", category="Technology"),
    "GOOGL": SymbolInfo(symbol="GOOGL", name="Alphabet Inc.", market="US", category="Technology"),
    "MSFT": SymbolInfo(symbol="MSFT", name="Microsoft Corporation", market="US", category="Technology"),
    "AMZN": SymbolInfo(symbol="AMZN", name="Amazon.com Inc.", market="US", category="Technology"),
    "TSLA": SymbolInfo(symbol="TSLA", name="Tesla Inc.", market="US", category="Automotive"),
    "META": SymbolInfo(symbol="META", name="Meta Platforms Inc.", market="US", category="Technology"),
    "NVDA": SymbolInfo(symbol="NVDA", name="NVIDIA Corporation", market="US", category="Technology"),
    "NFLX": SymbolInfo(symbol="NFLX", name="Netflix Inc.", market="US", category="Technology"),
    
    # US Finance
    "JPM": SymbolInfo(symbol="JPM", name="JPMorgan Chase & Co.", market="US", category="Finance"),
    "V": SymbolInfo(symbol="V", name="Visa Inc.", market="US", category="Finance"),
    "MA": SymbolInfo(symbol="MA", name="Mastercard Inc.", market="US", category="Finance"),
    "BAC": SymbolInfo(symbol="BAC", name="Bank of America Corp.", market="US", category="Finance"),
    "WFC": SymbolInfo(symbol="WFC", name="Wells Fargo & Co.", market="US", category="Finance"),
    
    # US Healthcare
    "JNJ": SymbolInfo(symbol="JNJ", name="Johnson & Johnson", market="US", category="Healthcare"),
    "UNH": SymbolInfo(symbol="UNH", name="UnitedHealth Group Inc.", market="US", category="Healthcare"),
    "PFE": SymbolInfo(symbol="PFE", name="Pfizer Inc.", market="US", category="Healthcare"),
    "ABBV": SymbolInfo(symbol="ABBV", name="AbbVie Inc.", market="US", category="Healthcare"),
    
    # Australian Markets
    "^AXJO": SymbolInfo(symbol="^AXJO", name="ASX 200", market="Australia", category="Index", currency="AUD"),
    "CBA.AX": SymbolInfo(symbol="CBA.AX", name="Commonwealth Bank of Australia", market="Australia", category="Finance", currency="AUD"),
    "WBC.AX": SymbolInfo(symbol="WBC.AX", name="Westpac Banking Corporation", market="Australia", category="Finance", currency="AUD"),
    "ANZ.AX": SymbolInfo(symbol="ANZ.AX", name="Australia and New Zealand Banking Group", market="Australia", category="Finance", currency="AUD"),
    "NAB.AX": SymbolInfo(symbol="NAB.AX", name="National Australia Bank", market="Australia", category="Finance", currency="AUD"),
    "BHP.AX": SymbolInfo(symbol="BHP.AX", name="BHP Group Limited", market="Australia", category="Mining", currency="AUD"),
    "RIO.AX": SymbolInfo(symbol="RIO.AX", name="Rio Tinto Limited", market="Australia", category="Mining", currency="AUD"),
    "FMG.AX": SymbolInfo(symbol="FMG.AX", name="Fortescue Metals Group", market="Australia", category="Mining", currency="AUD"),
    "CSL.AX": SymbolInfo(symbol="CSL.AX", name="CSL Limited", market="Australia", category="Healthcare", currency="AUD"),
    "WES.AX": SymbolInfo(symbol="WES.AX", name="Wesfarmers Limited", market="Australia", category="Retail", currency="AUD"),
    "TLS.AX": SymbolInfo(symbol="TLS.AX", name="Telstra Corporation", market="Australia", category="Telecommunications", currency="AUD"),
    "WOW.AX": SymbolInfo(symbol="WOW.AX", name="Woolworths Group", market="Australia", category="Retail", currency="AUD"),
    
    # Asian Indices
    "^N225": SymbolInfo(symbol="^N225", name="Nikkei 225", market="Japan", category="Index", currency="JPY"),
    "^HSI": SymbolInfo(symbol="^HSI", name="Hang Seng Index", market="Hong Kong", category="Index", currency="HKD"),
    "000001.SS": SymbolInfo(symbol="000001.SS", name="Shanghai Composite", market="China", category="Index", currency="CNY"),
    "^KS11": SymbolInfo(symbol="^KS11", name="KOSPI Composite Index", market="South Korea", category="Index", currency="KRW"),
    "^TWII": SymbolInfo(symbol="^TWII", name="Taiwan Weighted Index", market="Taiwan", category="Index", currency="TWD"),
    
    # European Indices
    "^FTSE": SymbolInfo(symbol="^FTSE", name="FTSE 100", market="UK", category="Index", currency="GBP"),
    "^GDAXI": SymbolInfo(symbol="^GDAXI", name="DAX Performance Index", market="Germany", category="Index", currency="EUR"),
    "^FCHI": SymbolInfo(symbol="^FCHI", name="CAC 40", market="France", category="Index", currency="EUR"),
    "^AEX": SymbolInfo(symbol="^AEX", name="AEX Index", market="Netherlands", category="Index", currency="EUR"),
    "^IBEX": SymbolInfo(symbol="^IBEX", name="IBEX 35", market="Spain", category="Index", currency="EUR"),
    
    # Commodities
    "GC=F": SymbolInfo(symbol="GC=F", name="Gold Futures", market="Global", category="Commodities", currency="USD"),
    "CL=F": SymbolInfo(symbol="CL=F", name="Crude Oil Futures", market="Global", category="Commodities", currency="USD"),
    "SI=F": SymbolInfo(symbol="SI=F", name="Silver Futures", market="Global", category="Commodities", currency="USD"),
    
    # Cryptocurrencies
    "BTC-USD": SymbolInfo(symbol="BTC-USD", name="Bitcoin", market="Global", category="Cryptocurrency", currency="USD"),
    "ETH-USD": SymbolInfo(symbol="ETH-USD", name="Ethereum", market="Global", category="Cryptocurrency", currency="USD"),
}

# 24-hour period configuration
PERIOD_CONFIG = {
    TimePeriod.HOUR_24: {"hours": 24, "description": "24 Hours"}
}

# Removed candlestick intervals - focusing on 24h timeline

def get_dynamic_market_hours():
    """Get market hours adjusted for current daylight saving time"""
    now = datetime.now(timezone.utc)
    
    # Check if UK is in BST (British Summer Time) - last Sunday in March to last Sunday in October
    uk_tz = pytz.timezone('Europe/London')
    uk_time = now.astimezone(uk_tz)
    is_bst = uk_time.dst() != timedelta(0)
    
    # Check if US is in EDT (Eastern Daylight Time) - 2nd Sunday in March to 1st Sunday in November  
    us_tz = pytz.timezone('America/New_York')
    us_time = now.astimezone(us_tz)
    is_edt = us_time.dst() != timedelta(0)
    
    return {
        "Japan": {"open": 0, "close": 6},           # 00:00-06:25 UTC (JST 09:00-15:25)
        "Hong Kong": {"open": 1, "close": 8},      # 01:30-08:00 UTC (HKT 09:30-16:00)
        "China": {"open": 1, "close": 7},          # 01:30-07:00 UTC (CST 09:30-15:00)
        "Australia": {"open": 0, "close": 6},      # 00:00-06:00 UTC (AEST 10:00-16:00)
        "South Korea": {"open": 0, "close": 6},    # 00:00-06:30 UTC (KST 09:00-15:30)
        "India": {"open": 3, "close": 10},         # 03:45-10:00 UTC (IST 09:15-15:30)
        "UK": {"open": 7 if is_bst else 8, "close": 16 if is_bst else 17},  # Dynamic BST/GMT
        "Germany": {"open": 6 if is_bst else 7, "close": 15 if is_bst else 16},  # Dynamic CEST/CET
        "France": {"open": 6 if is_bst else 7, "close": 15 if is_bst else 16},   # Dynamic CEST/CET
        "Netherlands": {"open": 6 if is_bst else 7, "close": 15 if is_bst else 16}, # Dynamic CEST/CET
        "Spain": {"open": 6 if is_bst else 7, "close": 15 if is_bst else 16},    # Dynamic CEST/CET
        "US": {"open": 13 if is_edt else 14, "close": 21 if is_edt else 22},     # Dynamic EDT/EST
        "Canada": {"open": 13 if is_edt else 14, "close": 21 if is_edt else 22}, # Dynamic EDT/EST
        "Brazil": {"open": 13, "close": 20},       # 13:00-20:00 UTC (BRT 10:00-17:00)
        "Global": {"open": 0, "close": 23}         # 24/7 for commodities and crypto
    }

# Use dynamic market hours
MARKET_HOURS = get_dynamic_market_hours()

async def generate_24h_market_data_live(symbols: List[str], chart_type: ChartType = ChartType.PERCENTAGE) -> Dict[str, List[MarketDataPoint]]:
    """Generate 24-hour market data using multi-source live data aggregator - NO DEMO DATA"""
    result = {}
    
    if not LIVE_DATA_ENABLED:
        raise HTTPException(status_code=503, detail="Live data is disabled")
    
    try:
        # Fetch live data from multiple sources for all symbols
        for symbol in symbols:
            if symbol not in SYMBOLS_DB:
                logger.warning(f"Symbol {symbol} not in database, skipping")
                continue
                
            market = SYMBOLS_DB[symbol].market
            
            # Get live data from multi-source aggregator
            market_data = await multi_source_aggregator.get_live_data(symbol)
            
            if market_data and market_data.data_points:
                # Convert live data to our format with market hours logic
                data_points = convert_live_data_to_24h_format(market_data.data_points, symbol, market, chart_type)
                result[symbol] = data_points
                logger.info(f"✅ Generated {len(data_points)} data points for {symbol} from sources: {', '.join(market_data.sources_used)}")
            else:
                logger.error(f"❌ No live data available for {symbol} from any provider")
                if REQUIRE_LIVE_DATA:
                    continue  # Skip symbols without live data
                else:
                    raise HTTPException(status_code=503, detail=f"No live data available for {symbol}")
        
        if not result:
            raise HTTPException(status_code=503, detail="No live data available for any requested symbols")
            
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching multi-source live data: {e}")
        raise HTTPException(status_code=500, detail=f"Live data service error: {str(e)}")

def convert_live_data_to_24h_format(live_points: List[LiveDataPoint], symbol: str, market: str, chart_type: ChartType) -> List[MarketDataPoint]:
    """Convert live data points to rolling 24-hour format with complete session coverage"""
    utc_now = datetime.now(timezone.utc)
    
    # Create rolling 24-hour window (last 24 hours from current time)
    end_time = utc_now
    start_time = end_time - timedelta(hours=24)
    
    logger.info(f"📅 Rolling 24h window for {symbol}: {start_time.strftime('%H:%M')} to {end_time.strftime('%H:%M')} UTC")
    
    # Get market hours configuration
    market_hours = MARKET_HOURS.get(market, {"open": 0, "close": 23})
    
    # Filter and sort live data points within our 24-hour window
    if not live_points:
        logger.warning(f"No live data points available for {symbol}")
        return []
    
    # Sort all points by timestamp
    sorted_points = sorted(live_points, key=lambda x: x.timestamp)
    
    # Find the most recent market session for base price calculation
    base_price = None
    most_recent_market_open = None
    
    # Look for market open price in the last 24 hours - handle overnight markets
    for point in reversed(sorted_points):
        if point.timestamp >= start_time and is_market_open_at_time(point.timestamp, market_hours):
            hour = point.timestamp.hour
            
            # Handle overnight markets (like Australia: 23:00-06:00)
            if market_hours["open"] > market_hours["close"]:
                # For overnight markets, open could be 23 (previous day) or early hours
                is_market_open_time = (hour >= market_hours["open"] or hour <= 1)
            else:
                # Regular markets
                is_market_open_time = market_hours["open"] <= hour <= (market_hours["open"] + 1)
            
            if is_market_open_time:
                most_recent_market_open = point.open
                break
    
    # Use most recent market open, or fallback to first available price
    base_price = most_recent_market_open if most_recent_market_open else sorted_points[0].close
    
    logger.info(f"📊 Base price for {symbol}: {base_price}")
    
    # Create 5-minute interval lookup for better precision
    live_data_lookup = {}
    for point in sorted_points:
        if start_time <= point.timestamp <= end_time:
            # Use exact timestamp for better matching
            live_data_lookup[point.timestamp] = point
    
    logger.info(f"📈 Found {len(live_data_lookup)} data points for {symbol} in 24h window")
    
    # Generate data points for each hour in the rolling 24-hour window
    data_points = []
    
    for hour_offset in range(24):
        current_hour_start = start_time + timedelta(hours=hour_offset)
        current_hour_end = current_hour_start + timedelta(hours=1)
        
        # Check if market should be open during this hour
        is_market_open_hour = is_market_open_at_time(current_hour_start, market_hours)
        
        # Find the best data point within this hour
        best_point = None
        best_timestamp = current_hour_start
        
        # Look for data points within this hour
        hour_points = []
        for ts, point in live_data_lookup.items():
            if current_hour_start <= ts < current_hour_end:
                hour_points.append((ts, point))
        
        # Special case: if this is the market close hour and we have no data in the hour,
        # look for the most recent data within the last 2 hours
        if is_market_open_hour and not hour_points:
            market_close_hour = market_hours.get("close", 16)
            if current_hour_start.hour == market_close_hour:
                # Look for data in previous hours up to market close
                search_start = current_hour_start - timedelta(hours=2)
                recent_points = []
                for ts, point in live_data_lookup.items():
                    if search_start <= ts <= current_hour_end:
                        recent_points.append((ts, point))
                
                if recent_points:
                    # Take the most recent point as the market close data
                    recent_points.sort(key=lambda x: x[0])
                    best_timestamp, best_point = recent_points[-1]
                    logger.info(f"📊 Using recent data for market close hour {current_hour_start.hour}: {best_timestamp}")
        
        elif hour_points:
            # Sort by timestamp and take the latest point in the hour (most current data)
            hour_points.sort(key=lambda x: x[0])
            best_timestamp, best_point = hour_points[-1]
        
        if is_market_open_hour and best_point:
            # Market is open and we have live data
            if chart_type == ChartType.PERCENTAGE:
                percentage_change = ((best_point.close - base_price) / base_price) * 100
            else:
                percentage_change = best_point.close
            
            data_points.append(MarketDataPoint(
                timestamp=current_hour_start.strftime('%Y-%m-%d %H:%M:%S UTC'),
                timestamp_ms=int(current_hour_start.timestamp() * 1000),
                open=best_point.open,
                high=best_point.high,
                low=best_point.low,
                close=best_point.close,
                volume=best_point.volume,
                percentage_change=round(percentage_change, 3),
                market_open=True
            ))
        else:
            # Market is closed or no data available
            data_points.append(MarketDataPoint(
                timestamp=current_hour_start.strftime('%Y-%m-%d %H:%M:%S UTC'),
                timestamp_ms=int(current_hour_start.timestamp() * 1000),
                open=None,
                high=None,
                low=None,
                close=None,
                volume=0,
                percentage_change=None,
                market_open=False
            ))
    
    logger.info(f"✅ Generated {len(data_points)} hourly data points for {symbol}")
    market_open_count = sum(1 for p in data_points if p.market_open)
    logger.info(f"📊 {market_open_count} points with market data, {24-market_open_count} points market closed")
    
    return data_points
def round_to_nearest_5min(dt: datetime) -> datetime:
    """Round datetime to nearest 5-minute interval"""
    minute = dt.minute
    rounded_minute = 5 * round(minute / 5)
    if rounded_minute == 60:
        dt = dt.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    else:
        dt = dt.replace(minute=rounded_minute, second=0, microsecond=0)
    return dt

def is_market_open_at_time(timestamp: datetime, market_hours: dict) -> bool:
    """Check if market is open at specific timestamp"""
    hour = timestamp.hour
    minute = timestamp.minute
    
    # Handle markets that span midnight (like Australia)
    if market_hours["open"] > market_hours["close"]:
        # Market spans midnight
        if hour > market_hours["open"] or hour < market_hours["close"]:
            return True
        elif hour == market_hours["open"]:
            return minute >= 30  # Markets typically open at :30
        elif hour == market_hours["close"]:
            return minute <= 30  # Include the full closing hour
    else:
        # Normal market hours - include the full closing hour
        if market_hours["open"] < hour < market_hours["close"]:
            return True
        elif hour == market_hours["open"]:
            return minute >= 30  # Markets typically open at :30
        elif hour == market_hours["close"]:
            return True  # Include all minutes in the closing hour
        elif hour == market_hours["close"] + 1 and minute == 0:
            return True  # Include the exact close time (e.g., 21:00 for US markets)
    
    return False

def find_best_data_point_for_hour(target_hour: datetime, data_lookup: dict) -> Optional[LiveDataPoint]:
    """Find the best data point for a given hour, preferring later times in the hour"""
    if not data_lookup:
        return None
    
    # Look for data points within the target hour
    hour_points = []
    for timestamp, point in data_lookup.items():
        if timestamp.hour == target_hour.hour and timestamp.date() == target_hour.date():
            hour_points.append(point)
    
    if hour_points:
        # Return the latest point in the hour
        return max(hour_points, key=lambda p: p.timestamp)
    
    # If no data in the exact hour, find the nearest point
    nearest_time = min(data_lookup.keys(), key=lambda x: abs((x - target_hour).total_seconds()))
    if abs((nearest_time - target_hour).total_seconds()) < 3600:  # Within 1 hour
        return data_lookup[nearest_time]
    
    return None

def find_market_close_data(current_time: datetime, data_lookup: dict, market_hours: dict) -> Optional[LiveDataPoint]:
    """Find the most recent market close data point"""
    if not data_lookup:
        return None
    
    # Look for data points near market close time
    close_hour = market_hours["close"]
    
    # Find the most recent close time
    recent_close_points = []
    for timestamp, point in data_lookup.items():
        # Check if this is a market close time (close hour or slightly after)
        if (timestamp.hour == close_hour and timestamp.minute >= 30) or \
           (timestamp.hour == close_hour + 1 and timestamp.minute <= 30):
            recent_close_points.append(point)
    
    if recent_close_points:
        # Return the latest close point
        return max(recent_close_points, key=lambda p: p.timestamp)
    
    return None

# ALL DEMO DATA FUNCTIONS REMOVED - USING MULTI-SOURCE LIVE DATA ONLY

def is_market_open_at_hour(hour: int, market: str) -> bool:
    """Check if market is open at given UTC hour"""
    market_hours = MARKET_HOURS.get(market, {"open": 0, "close": 23})
    
    # Handle overnight markets (like Australia)
    if market_hours["open"] > market_hours["close"]:
        return hour >= market_hours["open"] or hour <= market_hours["close"]
    else:
        return market_hours["open"] <= hour <= market_hours["close"]

# API Routes
@app.get("/api")
@app.get("/api/")
async def api_root():
    """API root endpoint"""
    return {
        "name": "Global Stock Market Tracker",
        "version": "1.0.0",
        "description": "24-Hour UTC Timeline for Global Stock Indices",
        "status": "healthy",
        "deployment": "local",
        "features": [
            "24-Hour UTC Timeline",
            "Global Stock Indices Selection",
            "Market Session Tracking",
            "Real-time Market Hours Display",
            "Cross-timezone Market Analysis"
        ],
        "endpoints": {
            "health": "/api/health",
            "symbols": "/api/symbols", 
            "search": "/api/search/{query}",
            "analyze": "/api/analyze",
            "market-hours": "/api/market-hours",
            "docs": "/api/docs"
        },
        "supported_indices": len(SYMBOLS_DB),
        "markets_covered": list(set(info.market for info in SYMBOLS_DB.values())),
        "chart_types": ["percentage", "price"],
        "key_features": [
            "24-hour UTC timeline focus",
            "Global stock indices from all major markets",
            "Real-time market session indicators",
            "Opening and closing time visualization",
            "Cross-market correlation analysis"
        ]
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint with multi-source live data status"""
    utc_now = datetime.now(timezone.utc)
    
    # Check multi-source data aggregator status
    live_data_status = "enabled" if LIVE_DATA_ENABLED else "disabled"
    data_providers = []
    
    if LIVE_DATA_ENABLED:
        # Get active providers from multi-source aggregator
        for provider in multi_source_aggregator.providers:
            if provider.is_configured():
                data_providers.append(f"{provider.name} (configured)")
            else:
                data_providers.append(f"{provider.name} (not configured)")
        
        if not data_providers:
            data_providers.append("No providers configured")
    else:
        data_providers.append("Live data disabled")
    
    return {
        "status": "healthy",
        "version": "2.0.0",
        "service": "Global Stock Market Tracker - Multi-Source Live Data",
        "timestamp": utc_now.isoformat(),
        "utc_time": utc_now.strftime('%Y-%m-%d %H:%M:%S UTC'),
        "deployment": "local",
        "supported_symbols": len(SYMBOLS_DB),
        "markets": list(set(info.market for info in SYMBOLS_DB.values())),
        "active_markets": get_currently_open_markets(utc_now.hour),
        "live_data_status": live_data_status,
        "total_providers": len(multi_source_aggregator.providers),
        "data_providers": data_providers,
        "demo_data_removed": True,
        "require_live_data": REQUIRE_LIVE_DATA,
        "features": [
            "24-hour UTC timeline tracking",
            "Multi-source live data aggregation",
            "Global market session indicators",
            "Real-time market hours display", 
            "Multi-index selection and analysis",
            "NO demo data fallbacks"
        ]
    }

def get_currently_open_markets(utc_hour: int) -> List[str]:
    """Get list of markets currently open at given UTC hour"""
    open_markets = []
    for market, hours in MARKET_HOURS.items():
        if is_market_open_at_hour(utc_hour, market):
            open_markets.append(market)
    return open_markets

@app.get("/api/symbols")
async def get_symbols():
    """Get all supported symbols organized by market and category"""
    symbols_by_market = {}
    
    for symbol, info in SYMBOLS_DB.items():
        market = info.market
        if market not in symbols_by_market:
            symbols_by_market[market] = []
        
        symbol_data = info.dict()
        # Add market hours info
        market_hours = MARKET_HOURS.get(market, {"open": 0, "close": 23})
        symbol_data["market_hours_utc"] = f"{market_hours['open']:02d}:00-{market_hours['close']:02d}:00"
        
        symbols_by_market[market].append(symbol_data)
    
    return {
        "total_symbols": len(SYMBOLS_DB),
        "markets": symbols_by_market,
        "chart_types": [c.value for c in ChartType],
        "market_hours_utc": MARKET_HOURS,
        "timeline": "24-hour UTC focus"
    }

@app.get("/api/live-status")
async def get_live_status():
    """Get real-time market status and refresh information"""
    utc_now = datetime.now(timezone.utc)
    
    # Calculate when the next refresh will happen
    next_refresh = utc_now.replace(second=0, microsecond=0)
    minutes_to_next = 5 - (next_refresh.minute % 5)
    if minutes_to_next == 5:
        minutes_to_next = 0
    next_refresh += timedelta(minutes=minutes_to_next)
    
    # Check which markets are currently open
    currently_open_markets = []
    for market, hours in MARKET_HOURS.items():
        if is_market_open_at_hour(utc_now.hour, market):
            # Check more precisely if market is actually open right now
            if is_market_open_at_time(utc_now, hours):
                currently_open_markets.append({
                    "market": market,
                    "hours_utc": f"{hours['open']:02d}:30-{hours['close']:02d}:30",
                    "status": "open"
                })
    
    return {
        "current_time_utc": utc_now.isoformat(),
        "display_time": utc_now.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "next_refresh_utc": next_refresh.isoformat(),
        "minutes_to_next_refresh": minutes_to_next,
        "refresh_interval_minutes": 5,
        "currently_open_markets": currently_open_markets,
        "rolling_window_hours": 24,
        "data_granularity": "5-minute intervals",
        "last_data_update": utc_now.isoformat()
    }

@app.get("/api/search/{query}")
async def search_symbols(query: str, limit: int = Query(default=20, ge=1, le=50)):
    """Search symbols by name, symbol, market, or category"""
    query_lower = query.lower()
    results = []
    
    for symbol, info in SYMBOLS_DB.items():
        if (query_lower in symbol.lower() or 
            query_lower in info.name.lower() or
            query_lower in info.market.lower() or
            query_lower in info.category.lower()):
            
            market_hours = MARKET_HOURS.get(info.market, {"open": 0, "close": 23})
            results.append({
                "symbol": symbol,
                "name": info.name,
                "market": info.market,
                "category": info.category,
                "currency": info.currency,
                "market_hours_utc": f"{market_hours['open']:02d}:00-{market_hours['close']:02d}:00"
            })
    
    return {
        "query": query,
        "results": results[:limit],
        "total_found": len(results)
    }

@app.get("/api/market-hours")
async def get_market_hours():
    """Get current market hours and status across all markets"""
    utc_now = datetime.now(timezone.utc)
    current_hour = utc_now.hour
    
    market_status = {}
    for market, hours in MARKET_HOURS.items():
        is_open = is_market_open_at_hour(current_hour, market)
        
        # Calculate next open/close time
        if hours["open"] > hours["close"]:  # Overnight market
            if current_hour >= hours["open"] or current_hour <= hours["close"]:
                next_close = hours["close"] if current_hour >= hours["open"] else hours["close"]
                next_event = "closes"
                next_time = f"{next_close:02d}:00 UTC"
            else:
                next_event = "opens"
                next_time = f"{hours['open']:02d}:00 UTC"
        else:  # Regular market
            if hours["open"] <= current_hour <= hours["close"]:
                next_event = "closes"
                next_time = f"{hours['close']:02d}:00 UTC"
            else:
                next_event = "opens"
                next_time = f"{hours['open']:02d}:00 UTC"
        
        market_status[market] = {
            "is_open": is_open,
            "hours_utc": f"{hours['open']:02d}:00-{hours['close']:02d}:00",
            "next_event": next_event,
            "next_time": next_time
        }
    
    return {
        "current_utc_time": utc_now.strftime('%Y-%m-%d %H:%M:%S UTC'),
        "current_utc_hour": current_hour,
        "markets": market_status,
        "currently_open": [market for market, status in market_status.items() if status["is_open"]]
    }

@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_symbols(request: AnalysisRequest):
    """Analyze selected symbols with 24-hour UTC timeline using live data when available"""
    
    # Validate symbols
    invalid_symbols = [s for s in request.symbols if s not in SYMBOLS_DB]
    if invalid_symbols:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported symbols: {', '.join(invalid_symbols)}"
        )
    
    # Generate 24-hour data for all symbols (now with live data integration)
    try:
        symbol_data = await generate_24h_market_data_live(request.symbols, request.chart_type)
        symbol_metadata = {symbol: SYMBOLS_DB[symbol] for symbol in request.symbols if symbol in SYMBOLS_DB}
        
        # Add data source information
        data_source = "live" if LIVE_DATA_ENABLED else "demo"
        
        return AnalysisResponse(
            success=True,
            data=symbol_data,
            metadata=symbol_metadata,
            chart_type=request.chart_type.value,
            timestamp=datetime.now(timezone.utc).isoformat(),
            total_symbols=len(request.symbols),
            successful_symbols=len(symbol_data),
            market_hours=MARKET_HOURS
        )
    except Exception as e:
        logger.error(f"Failed to generate market data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate market data: {str(e)}")

# Removed candlestick endpoint - focusing on 24h timeline only

@app.get("/api/data-status")
async def get_data_status():
    """Get current multi-source data provider status and configuration"""
    
    # Get provider status from multi-source aggregator
    provider_status = {}
    for provider in multi_source_aggregator.providers:
        provider_status[provider.name.lower().replace(' ', '_')] = {
            "enabled": True,
            "description": f"{provider.name} data provider",
            "status": "configured" if provider.is_configured() else "not configured"
        }
    
    return {
        "live_data_enabled": LIVE_DATA_ENABLED,
        "require_live_data": REQUIRE_LIVE_DATA,
        "demo_data_removed": True,
        "total_providers": len(multi_source_aggregator.providers),
        "data_providers": provider_status,
        "cache_info": {
            "cache_duration_minutes": int(os.getenv('DATA_CACHE_MINUTES', 3)),
            "rate_limit_per_minute": int(os.getenv('MAX_API_CALLS_PER_MINUTE', 10))
        },
        "setup_instructions": {
            "alpha_vantage": "Get free API key at https://www.alphavantage.co/support/#api-key",
            "twelve_data": "Get free API key at https://twelvedata.com/",
            "finnhub": "Get free API key at https://finnhub.io/",
            "environment_file": "Add API keys to .env file (ALPHA_VANTAGE_API_KEY, TWELVE_DATA_API_KEY, FINNHUB_API_KEY)"
        }
    }

@app.get("/api/suggested-indices")
async def get_suggested_indices():
    """Get suggested global indices for 24-hour timeline analysis"""
    
    # Suggested indices across different time zones for 24h coverage
    suggested = {
        "asian_pacific": [
            {"symbol": "^N225", "name": "Nikkei 225", "market": "Japan", "hours": "00:00-06:00 UTC"},
            {"symbol": "^HSI", "name": "Hang Seng Index", "market": "Hong Kong", "hours": "01:00-08:00 UTC"},
            {"symbol": "^AXJO", "name": "ASX 200", "market": "Australia", "hours": "23:00-06:00 UTC"}
        ],
        "european": [
            {"symbol": "^FTSE", "name": "FTSE 100", "market": "UK", "hours": "08:00-16:00 UTC"},
            {"symbol": "^GDAXI", "name": "DAX", "market": "Germany", "hours": "08:00-17:00 UTC"},
            {"symbol": "^FCHI", "name": "CAC 40", "market": "France", "hours": "08:00-17:00 UTC"}
        ],
        "americas": [
            {"symbol": "^GSPC", "name": "S&P 500", "market": "US", "hours": "14:30-21:00 UTC"},
            {"symbol": "^IXIC", "name": "NASDAQ", "market": "US", "hours": "14:30-21:00 UTC"},
            {"symbol": "^DJI", "name": "Dow Jones", "market": "US", "hours": "14:30-21:00 UTC"}
        ]
    }
    
    return {
        "suggested_indices": suggested,
        "total_coverage": "24-hour global market flow",
        "recommendation": "Select at least one index from each region for complete 24h coverage"
    }

# Application startup event
@app.on_event("startup")
async def startup_event():
    """Application startup with live data status"""
    utc_now = datetime.now(timezone.utc)
    logger.info("🚀 Global Stock Market Tracker v2.0 - Live Data Integration")
    logger.info(f"📊 Loaded {len(SYMBOLS_DB)} symbols across {len(set(info.market for info in SYMBOLS_DB.values()))} markets")
    logger.info(f"🕐 Current UTC Time: {utc_now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    
    # Multi-source live data status
    if LIVE_DATA_ENABLED:
        logger.info("📶 Multi-Source Live Data: ENABLED")
        logger.info(f"   • Total Providers: {len(multi_source_aggregator.providers)}")
        
        for provider in multi_source_aggregator.providers:
            status = "✅ CONFIGURED" if provider.is_configured() else "❌ NOT CONFIGURED"
            logger.info(f"   • {provider.name}: {status}")
        
        logger.info("   • Demo Data Fallback: COMPLETELY REMOVED")
        logger.info(f"   • Require Live Data: {REQUIRE_LIVE_DATA}")
    else:
        logger.error("📶 Live Data: DISABLED - Service will not function without live data")
    
    logger.info("⏰ Focus: 24-Hour UTC Timeline Only")
    logger.info("🌍 Market Coverage:")
    for market, hours in MARKET_HOURS.items():
        status = "🟢 OPEN" if is_market_open_at_hour(utc_now.hour, market) else "🔴 CLOSED"
        logger.info(f"   • {market}: {hours['open']:02d}:00-{hours['close']:02d}:00 UTC {status}")
    
    logger.info("✅ Ready for local deployment with live data integration")
    logger.info("🌐 Frontend served at: http://localhost:8000/")
    logger.info("📚 API docs at: http://localhost:8000/api/docs")
    logger.info("📊 Data status at: http://localhost:8000/api/data-status")
    
    logger.info(f"🔗 Multi-source data providers: {len(multi_source_aggregator.providers)} configured")
    logger.info("")
    logger.info("💡 To configure additional providers, add API keys to .env:")
    logger.info("   ALPHA_VANTAGE_API_KEY=your_key")
    logger.info("   TWELVE_DATA_API_KEY=your_key") 
    logger.info("   FINNHUB_API_KEY=your_key")

# Mount static files AFTER all API routes are defined
if os.path.exists("frontend"):
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on http://0.0.0.0:{port}")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port, 
        log_level="info",
        reload=False
    )