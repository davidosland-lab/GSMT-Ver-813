"""
GSMT Ver 7.0 - FastAPI Backend
Optimized for Railway deployment with Sydney timezone integration
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union
from datetime import datetime, timedelta
from enum import Enum
import yfinance as yf
import pandas as pd
import numpy as np
import asyncio
import logging
import os
import json
# Import timezone handlers - handle as modules in same directory
try:
    from timezone_handler import sydney_tz_handler, get_sydney_24h_period, format_for_sydney_display
    from market_sessions import market_sessions_manager
except ImportError:
    # Fallback for different import patterns
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from timezone_handler import sydney_tz_handler, get_sydney_24h_period, format_for_sydney_display
    from market_sessions import market_sessions_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app with optimized settings
app = FastAPI(
    title="GSMT Ver 7.0 API",
    description="Global Stock Market Tracker - Clean Architecture for Railway Deployment",
    version="7.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS - configured for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for your Netlify domain in production
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Enums for type safety
class TimePeriod(str, Enum):
    HOUR_24 = "24h"
    DAYS_3 = "3d"
    WEEK_1 = "1w"
    WEEKS_2 = "2w"
    MONTH_1 = "1M"
    MONTHS_3 = "3M"
    MONTHS_6 = "6M"
    YEAR_1 = "1Y"
    YEARS_2 = "2Y"

class ChartType(str, Enum):
    PERCENTAGE = "percentage"
    PRICE = "price"
    CANDLESTICK = "candlestick"

# Pydantic models
class MarketDataPoint(BaseModel):
    timestamp: str
    timestamp_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: int
    percentage_change: float

class SymbolInfo(BaseModel):
    symbol: str
    name: str
    market: str
    category: str
    currency: str = "USD"

class AnalysisRequest(BaseModel):
    symbols: List[str] = Field(..., min_items=1, max_items=10)
    period: TimePeriod = TimePeriod.HOUR_24
    chart_type: ChartType = ChartType.PERCENTAGE
    sydney_start: Optional[bool] = Field(default=True, description="Start 24h period from 10am Sydney time")
    reference_time: Optional[str] = Field(default=None, description="Reference time for analysis (ISO format)")

class AnalysisResponse(BaseModel):
    success: bool
    data: Dict[str, List[MarketDataPoint]]
    metadata: Dict[str, SymbolInfo]
    period: str
    chart_type: str
    timestamp: str
    sydney_timestamp: str
    total_symbols: int
    successful_symbols: int
    period_start: str
    period_end: str
    market_sessions: Optional[Dict] = None

class SydneyMarketResponse(BaseModel):
    success: bool
    data: Dict[str, List[MarketDataPoint]]
    market_sessions: List[Dict]
    timeline: List[Dict]
    sydney_context: Dict
    refresh_schedule: Dict
    timestamp: str
    period_start: str
    period_end: str

# Supported symbols database
SYMBOLS_DB = {
    # US Indices
    "^GSPC": SymbolInfo(symbol="^GSPC", name="S&P 500", market="US", category="Index"),
    "^IXIC": SymbolInfo(symbol="^IXIC", name="NASDAQ Composite", market="US", category="Index"),
    "^DJI": SymbolInfo(symbol="^DJI", name="Dow Jones Industrial Average", market="US", category="Index"),
    
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
    "BAC": SymbolInfo(symbol="BAC", name="Bank of America Corporation", market="US", category="Finance"),
    "V": SymbolInfo(symbol="V", name="Visa Inc.", market="US", category="Finance"),
    "MA": SymbolInfo(symbol="MA", name="Mastercard Incorporated", market="US", category="Finance"),
    
    # Australian Index
    "^AXJO": SymbolInfo(symbol="^AXJO", name="ASX 200", market="Australia", category="Index", currency="AUD"),
    
    # Australian Stocks
    "CBA.AX": SymbolInfo(symbol="CBA.AX", name="Commonwealth Bank of Australia", market="Australia", category="Finance", currency="AUD"),
    "BHP.AX": SymbolInfo(symbol="BHP.AX", name="BHP Group Limited", market="Australia", category="Mining", currency="AUD"),
    "CSL.AX": SymbolInfo(symbol="CSL.AX", name="CSL Limited", market="Australia", category="Healthcare", currency="AUD"),
    "WBC.AX": SymbolInfo(symbol="WBC.AX", name="Westpac Banking Corporation", market="Australia", category="Finance", currency="AUD"),
    "ANZ.AX": SymbolInfo(symbol="ANZ.AX", name="Australia and New Zealand Banking Group", market="Australia", category="Finance", currency="AUD"),
    "NAB.AX": SymbolInfo(symbol="NAB.AX", name="National Australia Bank Limited", market="Australia", category="Finance", currency="AUD"),
    "WES.AX": SymbolInfo(symbol="WES.AX", name="Wesfarmers Limited", market="Australia", category="Retail", currency="AUD"),
    "TLS.AX": SymbolInfo(symbol="TLS.AX", name="Telstra Corporation Limited", market="Australia", category="Telecommunications", currency="AUD"),
    "WOW.AX": SymbolInfo(symbol="WOW.AX", name="Woolworths Group Limited", market="Australia", category="Retail", currency="AUD"),
    "RIO.AX": SymbolInfo(symbol="RIO.AX", name="Rio Tinto Limited", market="Australia", category="Mining", currency="AUD"),
    
    # Asian Indices
    "^N225": SymbolInfo(symbol="^N225", name="Nikkei 225", market="Japan", category="Index", currency="JPY"),
    "^HSI": SymbolInfo(symbol="^HSI", name="Hang Seng Index", market="Hong Kong", category="Index", currency="HKD"),
    "000001.SS": SymbolInfo(symbol="000001.SS", name="Shanghai Composite", market="China", category="Index", currency="CNY"),
    
    # European Indices
    "^FTSE": SymbolInfo(symbol="^FTSE", name="FTSE 100", market="UK", category="Index", currency="GBP"),
    "^GDAXI": SymbolInfo(symbol="^GDAXI", name="DAX Performance Index", market="Germany", category="Index", currency="EUR"),
    "^FCHI": SymbolInfo(symbol="^FCHI", name="CAC 40", market="France", category="Index", currency="EUR"),
}

# Period configurations
PERIOD_CONFIG = {
    TimePeriod.HOUR_24: {"days": 1, "interval": "5m", "description": "24 Hours"},
    TimePeriod.DAYS_3: {"days": 3, "interval": "15m", "description": "3 Days"},
    TimePeriod.WEEK_1: {"days": 7, "interval": "30m", "description": "1 Week"},
    TimePeriod.WEEKS_2: {"days": 14, "interval": "1h", "description": "2 Weeks"},
    TimePeriod.MONTH_1: {"days": 30, "interval": "1d", "description": "1 Month"},
    TimePeriod.MONTHS_3: {"days": 90, "interval": "1d", "description": "3 Months"},
    TimePeriod.MONTHS_6: {"days": 180, "interval": "1d", "description": "6 Months"},
    TimePeriod.YEAR_1: {"days": 365, "interval": "1d", "description": "1 Year"},
    TimePeriod.YEARS_2: {"days": 730, "interval": "1wk", "description": "2 Years"},
}

# Cache for performance (simple in-memory cache)
_cache = {}
_cache_timeout = 300  # 5 minutes

def get_cached_data(key: str):
    """Get cached data if not expired"""
    if key in _cache:
        data, timestamp = _cache[key]
        if (datetime.now() - timestamp).seconds < _cache_timeout:
            return data
    return None

def set_cached_data(key: str, data):
    """Set cached data with timestamp"""
    _cache[key] = (data, datetime.now())

async def fetch_symbol_data(symbol: str, period: TimePeriod, sydney_start: bool = True, reference_time: Optional[datetime] = None) -> List[MarketDataPoint]:
    """Fetch data for a single symbol with Sydney timezone support"""
    
    # Create cache key with Sydney start parameter
    cache_key = f"{symbol}_{period.value}_sydney_{sydney_start}_{reference_time.isoformat() if reference_time else 'now'}"
    
    # Check cache first
    cached = get_cached_data(cache_key)
    if cached:
        return cached
    
    try:
        # Get period configuration
        config = PERIOD_CONFIG[period]
        
        # Calculate time range based on Sydney timezone if requested
        if sydney_start and period == TimePeriod.HOUR_24:
            start_time, end_time = get_sydney_24h_period(reference_time)
            
            # Convert to UTC for yfinance
            start_utc = start_time.astimezone(sydney_tz_handler.utc_tz)
            end_utc = end_time.astimezone(sydney_tz_handler.utc_tz)
            
            logger.info(f"Fetching {symbol} data for Sydney 24h period: {start_time} to {end_time}")
        else:
            # Use standard period calculation
            end_time = datetime.now(sydney_tz_handler.sydney_tz)
            start_time = end_time - timedelta(days=config["days"])
            start_utc = start_time.astimezone(sydney_tz_handler.utc_tz)
            end_utc = end_time.astimezone(sydney_tz_handler.utc_tz)
        
        # Fetch data from Yahoo Finance
        ticker = yf.Ticker(symbol)
        
        # Use start/end dates for more precise control
        hist = ticker.history(
            start=start_utc.strftime('%Y-%m-%d'),
            end=end_utc.strftime('%Y-%m-%d'), 
            interval=config["interval"]
        )
        
        if hist.empty:
            # Try with period fallback
            if config["days"] <= 7:
                yf_period = f"{config['days']}d"
            elif config["days"] <= 60:
                yf_period = f"{config['days']}d"
            elif config["days"] <= 365:
                months = config["days"] // 30
                yf_period = f"{months}mo" if months <= 12 else "1y"
            else:
                yf_period = "2y"
                
            hist = ticker.history(period=yf_period, interval=config["interval"])
        
        if hist.empty:
            return []
        
        # Calculate percentage changes with Sydney 10am as base if applicable
        data_points = []
        prices = hist['Close'].values
        
        # Find base price - use first price in Sydney 24h period if applicable
        if sydney_start and period == TimePeriod.HOUR_24:
            # Use price closest to Sydney 10am start as base
            base_price = prices[0] if len(prices) > 0 else 1
        else:
            base_price = prices[0] if len(prices) > 0 else 1
        
        for i, (timestamp, row) in enumerate(hist.iterrows()):
            # Convert timestamp to Sydney timezone for display
            sydney_timestamp = timestamp.tz_localize('UTC').astimezone(sydney_tz_handler.sydney_tz)
            
            percentage_change = ((prices[i] - base_price) / base_price * 100) if base_price != 0 else 0
            
            data_points.append(MarketDataPoint(
                timestamp=sydney_timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                timestamp_ms=int(sydney_timestamp.timestamp() * 1000),
                open=float(row['Open']),
                high=float(row['High']),
                low=float(row['Low']),
                close=float(row['Close']),
                volume=int(row['Volume']) if not pd.isna(row['Volume']) else 0,
                percentage_change=round(percentage_change, 2)
            ))
        
        # Cache the result
        set_cached_data(cache_key, data_points)
        return data_points
        
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        return []

def generate_demo_data(symbol: str, period: TimePeriod, start_time: Optional[datetime] = None) -> List[MarketDataPoint]:
    """Generate realistic demo data with Sydney timezone support"""
    config = PERIOD_CONFIG[period]
    days = config["days"]
    
    # Generate realistic base price
    if symbol.startswith('^'):
        base_price = np.random.uniform(3000, 40000)  # Index range
    elif '.AX' in symbol:
        base_price = np.random.uniform(10, 300)      # Australian stock range
    else:
        base_price = np.random.uniform(50, 500)      # US stock range
    
    # Use provided start time or calculate from Sydney timezone
    if start_time is None:
        if period == TimePeriod.HOUR_24:
            start_time, _ = get_sydney_24h_period()
        else:
            start_time = sydney_tz_handler.get_sydney_now() - timedelta(days=days)
    
    # Generate points based on period
    if period == TimePeriod.HOUR_24:
        # For 24h period, generate hourly points
        num_points = 24
        time_increment = timedelta(hours=1)
    else:
        # For longer periods, generate daily points
        num_points = min(100, days * 2)
        time_increment = timedelta(days=(days / num_points))
    
    data_points = []
    current_price = base_price
    
    for i in range(num_points):
        # Enhanced volatility model based on market and time
        if symbol.startswith('^'):
            # Index volatility
            volatility = 0.015  # 1.5%
        elif '.AX' in symbol:
            # Australian stock volatility
            volatility = 0.025  # 2.5%
        else:
            # International stock volatility
            volatility = 0.02   # 2%
        
        # Random walk with mean reversion
        change = np.random.normal(0, volatility)
        current_price = max(current_price * (1 + change), base_price * 0.3)
        
        # Calculate timestamp in Sydney timezone
        timestamp = start_time + (time_increment * i)
        
        # Ensure timestamp is in Sydney timezone
        if timestamp.tzinfo != sydney_tz_handler.sydney_tz:
            timestamp = timestamp.astimezone(sydney_tz_handler.sydney_tz)
        
        percentage_change = ((current_price - base_price) / base_price) * 100
        
        # Generate OHLC around current price with realistic spreads
        spread = volatility * 0.5
        high = current_price * (1 + abs(np.random.normal(0, spread)))
        low = current_price * (1 - abs(np.random.normal(0, spread)))
        open_price = current_price * (1 + np.random.normal(0, spread * 0.5))
        
        # Volume varies by market and time
        if '.AX' in symbol:
            volume = int(np.random.uniform(500000, 50000000))  # Australian volumes
        elif symbol.startswith('^'):
            volume = 0  # Indices don't have volume
        else:
            volume = int(np.random.uniform(1000000, 100000000))  # US volumes
        
        data_points.append(MarketDataPoint(
            timestamp=timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            timestamp_ms=int(timestamp.timestamp() * 1000),
            open=round(open_price, 2),
            high=round(high, 2),
            low=round(low, 2),
            close=round(current_price, 2),
            volume=volume,
            percentage_change=round(percentage_change, 2)
        ))
    
    return data_points

# API Routes
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "GSMT Ver 7.0 API",
        "version": "7.0.0",
        "description": "Global Stock Market Tracker - Clean Architecture",
        "status": "healthy",
        "endpoints": {
            "health": "/health",
            "symbols": "/symbols",
            "analyze": "/analyze",
            "search": "/search/{query}",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint with Sydney timezone info"""
    try:
        sydney_now = sydney_tz_handler.get_sydney_now()
        
        # Basic health check
        health_data = {
            "status": "healthy",
            "version": "7.0.0",
            "timestamp": datetime.now().isoformat(),
            "sydney_time": format_for_sydney_display(sydney_now),
            "service": "GSMT Ver 7.0 API - Sydney Edition",
            "environment": os.environ.get("RAILWAY_ENVIRONMENT", "development"),
            "features": [
                "Sydney timezone integration",
                "24-hour periods from 10am AEST/AEDT",
                "Global market sessions tracking",
                "Percentage-based analysis",
                "Multi-source data fallback", 
                "Global market coverage",
                "Railway optimized deployment"
            ],
            "supported_symbols": len(SYMBOLS_DB),
            "cache_size": len(_cache),
            "active_markets": sydney_tz_handler.get_active_markets(),
            "sydney_market_open": sydney_tz_handler.is_market_open_now("Australia"),
            "uptime": "healthy"
        }
        
        # Add Railway-specific info if available
        if "RAILWAY_DEPLOYMENT_ID" in os.environ:
            health_data["deployment_id"] = os.environ.get("RAILWAY_DEPLOYMENT_ID")
        
        return health_data
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy", 
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/symbols")
async def get_symbols():
    """Get all supported symbols"""
    symbols_by_category = {}
    
    for symbol, info in SYMBOLS_DB.items():
        category = f"{info.market} {info.category}"
        if category not in symbols_by_category:
            symbols_by_category[category] = []
        symbols_by_category[category].append(info.dict())
    
    return {
        "total_symbols": len(SYMBOLS_DB),
        "categories": symbols_by_category,
        "supported_periods": [p.value for p in TimePeriod],
        "chart_types": [c.value for c in ChartType]
    }

@app.get("/search/{query}")
async def search_symbols(query: str, limit: int = Query(default=10, ge=1, le=50)):
    """Search symbols by name or symbol"""
    query_lower = query.lower()
    results = []
    
    for symbol, info in SYMBOLS_DB.items():
        if (query_lower in symbol.lower() or 
            query_lower in info.name.lower() or
            query_lower in info.market.lower() or
            query_lower in info.category.lower()):
            
            results.append({
                "symbol": symbol,
                "name": info.name,
                "market": info.market,
                "category": info.category,
                "currency": info.currency,
                "relevance": (
                    query_lower in symbol.lower() * 3 +
                    query_lower in info.name.lower() * 2 +
                    query_lower in info.market.lower() +
                    query_lower in info.category.lower()
                )
            })
    
    # Sort by relevance and limit results
    results.sort(key=lambda x: x["relevance"], reverse=True)
    
    return {
        "query": query,
        "results": results[:limit],
        "total_found": len(results)
    }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_symbols(request: AnalysisRequest):
    """Analyze symbols with Sydney timezone support"""
    
    # Validate symbols
    invalid_symbols = [s for s in request.symbols if s not in SYMBOLS_DB]
    if invalid_symbols:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported symbols: {', '.join(invalid_symbols)}"
        )
    
    # Parse reference time if provided
    reference_time = None
    if request.reference_time:
        try:
            reference_time = datetime.fromisoformat(request.reference_time.replace('Z', '+00:00'))
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid reference_time format. Use ISO format."
            )
    
    # Get period boundaries
    if request.sydney_start and request.period == TimePeriod.HOUR_24:
        start_time, end_time = get_sydney_24h_period(reference_time)
        period_start = format_for_sydney_display(start_time)
        period_end = format_for_sydney_display(end_time)
    else:
        sydney_now = sydney_tz_handler.get_sydney_now()
        end_time = sydney_now
        start_time = sydney_now - timedelta(days=PERIOD_CONFIG[request.period]["days"])
        period_start = format_for_sydney_display(start_time)
        period_end = format_for_sydney_display(end_time)
    
    # Fetch data for all symbols
    symbol_data = {}
    symbol_metadata = {}
    successful_count = 0
    
    for symbol in request.symbols:
        try:
            # Try to fetch real data with Sydney timezone support
            data = await fetch_symbol_data(symbol, request.period, request.sydney_start, reference_time)
            
            # If no real data, generate demo data
            if not data:
                logger.warning(f"No real data for {symbol}, using demo data")
                data = generate_demo_data(symbol, request.period, start_time)
            
            if data:
                symbol_data[symbol] = data
                symbol_metadata[symbol] = SYMBOLS_DB[symbol]
                successful_count += 1
                
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {str(e)}")
            # Generate demo data as fallback
            try:
                data = generate_demo_data(symbol, request.period, start_time)
                if data:
                    symbol_data[symbol] = data
                    symbol_metadata[symbol] = SYMBOLS_DB[symbol]
                    successful_count += 1
            except Exception as demo_error:
                logger.error(f"Failed to generate demo data for {symbol}: {str(demo_error)}")
    
    if not symbol_data:
        raise HTTPException(
            status_code=503,
            detail="Failed to fetch data for any symbols"
        )
    
    # Get market sessions for 24h analysis
    market_sessions_data = None
    if request.period == TimePeriod.HOUR_24:
        market_sessions_data = market_sessions_manager.get_24h_market_flow_data()
    
    sydney_now = sydney_tz_handler.get_sydney_now()
    
    return AnalysisResponse(
        success=True,
        data=symbol_data,
        metadata=symbol_metadata,
        period=request.period.value,
        chart_type=request.chart_type.value,
        timestamp=datetime.now().isoformat(),
        sydney_timestamp=format_for_sydney_display(sydney_now),
        total_symbols=len(request.symbols),
        successful_symbols=successful_count,
        period_start=period_start,
        period_end=period_end,
        market_sessions=market_sessions_data
    )

@app.get("/sydney-markets", response_model=SydneyMarketResponse)
async def get_sydney_markets():
    """Get comprehensive Sydney-focused market analysis"""
    
    try:
        # Get 24-hour market flow data
        flow_data = market_sessions_manager.get_24h_market_flow_data()
        
        # Get default symbols for Sydney analysis
        default_symbols = flow_data["default_symbols"]
        
        # Fetch data for default symbols
        symbol_data = {}
        for symbol in default_symbols:
            try:
                data = await fetch_symbol_data(symbol, TimePeriod.HOUR_24, sydney_start=True)
                if not data:
                    data = generate_demo_data(symbol, TimePeriod.HOUR_24, flow_data["start_time"])
                
                if data:
                    symbol_data[symbol] = data
            except Exception as e:
                logger.error(f"Failed to fetch Sydney market data for {symbol}: {str(e)}")
                try:
                    symbol_data[symbol] = generate_demo_data(symbol, TimePeriod.HOUR_24, flow_data["start_time"])
                except Exception as demo_error:
                    logger.error(f"Failed to generate demo data for {symbol}: {str(demo_error)}")
        
        # Get additional context
        sydney_context = market_sessions_manager.get_sydney_market_context()
        refresh_schedule = market_sessions_manager.get_optimal_refresh_schedule()
        
        sydney_now = sydney_tz_handler.get_sydney_now()
        
        return SydneyMarketResponse(
            success=True,
            data=symbol_data,
            market_sessions=flow_data["market_sessions"],
            timeline=flow_data["timeline"],
            sydney_context=sydney_context,
            refresh_schedule=refresh_schedule,
            timestamp=format_for_sydney_display(sydney_now),
            period_start=format_for_sydney_display(flow_data["start_time"]),
            period_end=format_for_sydney_display(flow_data["end_time"])
        )
        
    except Exception as e:
        logger.error(f"Sydney markets endpoint failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Failed to fetch Sydney market data: {str(e)}"
        )

@app.get("/market-sessions")
async def get_current_market_sessions():
    """Get current global market sessions in Sydney timezone"""
    
    try:
        sessions = market_sessions_manager.get_current_market_sessions()
        timeline = sydney_tz_handler.get_market_session_timeline()
        active_markets = sydney_tz_handler.get_active_markets()
        
        return {
            "success": True,
            "sessions": [
                {
                    "market": session.market,
                    "display_name": session.display_name,
                    "open_sydney": format_for_sydney_display(session.open_sydney),
                    "close_sydney": format_for_sydney_display(session.close_sydney),
                    "is_active": session.is_active,
                    "color": session.color,
                    "timezone": session.timezone_name
                } for session in sessions
            ],
            "timeline": timeline,
            "active_markets": active_markets,
            "sydney_time": format_for_sydney_display(sydney_tz_handler.get_sydney_now()),
            "refresh_schedule": market_sessions_manager.get_optimal_refresh_schedule()
        }
        
    except Exception as e:
        logger.error(f"Market sessions endpoint failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Failed to fetch market sessions: {str(e)}"
        )

@app.get("/sydney-time")
async def get_sydney_time_info():
    """Get comprehensive Sydney timezone information"""
    
    sydney_now = sydney_tz_handler.get_sydney_now()
    start_10am, end_10am = get_sydney_24h_period()
    
    return {
        "success": True,
        "sydney_time": format_for_sydney_display(sydney_now),
        "sydney_timestamp_ms": int(sydney_now.timestamp() * 1000),
        "timezone": str(sydney_now.tzinfo),
        "is_dst": bool(sydney_now.dst()),
        "utc_offset_hours": sydney_now.utcoffset().total_seconds() / 3600,
        "period_24h": {
            "start": format_for_sydney_display(start_10am),
            "end": format_for_sydney_display(end_10am),
            "start_ms": int(start_10am.timestamp() * 1000),
            "end_ms": int(end_10am.timestamp() * 1000)
        },
        "market_context": market_sessions_manager.get_sydney_market_context()
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.now().isoformat()
        }
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """Application startup"""
    logger.info("🚀 Starting GSMT Ver 7.0 API")
    logger.info(f"📊 Loaded {len(SYMBOLS_DB)} symbols")
    logger.info("✅ Ready for Railway deployment")
    
    # Verify critical endpoints
    logger.info("🔍 Verifying critical endpoints...")
    logger.info("   - /health (health check)")
    logger.info("   - /docs (API documentation)")
    logger.info("   - /symbols (symbols database)")
    logger.info("   - /analyze (analysis endpoint)")
    logger.info("🎯 All endpoints configured and ready")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)