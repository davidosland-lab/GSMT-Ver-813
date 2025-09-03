"""
GSMT Ver 7.0 - Simplified Railway Deployment
Single-file FastAPI application optimized for Railway
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from enum import Enum
import os
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="GSMT Ver 7.0 API",
    description="Global Stock Market Tracker - Railway Optimized",
    version="7.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Enums
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

class AnalysisResponse(BaseModel):
    success: bool
    data: Dict[str, List[MarketDataPoint]]
    metadata: Dict[str, SymbolInfo]
    period: str
    chart_type: str
    timestamp: str
    total_symbols: int
    successful_symbols: int

# Symbols database
SYMBOLS_DB = {
    # US Indices (Trading: 9:30 AM - 4:00 PM EST / 14:30-21:00 UTC)
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
    
    # US Finance
    "JPM": SymbolInfo(symbol="JPM", name="JPMorgan Chase & Co.", market="US", category="Finance"),
    "V": SymbolInfo(symbol="V", name="Visa Inc.", market="US", category="Finance"),
    
    # Australian Markets
    "^AXJO": SymbolInfo(symbol="^AXJO", name="ASX 200", market="Australia", category="Index", currency="AUD"),
    "CBA.AX": SymbolInfo(symbol="CBA.AX", name="Commonwealth Bank of Australia", market="Australia", category="Finance", currency="AUD"),
    "BHP.AX": SymbolInfo(symbol="BHP.AX", name="BHP Group Limited", market="Australia", category="Mining", currency="AUD"),
    "CSL.AX": SymbolInfo(symbol="CSL.AX", name="CSL Limited", market="Australia", category="Healthcare", currency="AUD"),
    
    # Asian Indices (Trading: 9:00 AM - 3:00 PM local / 00:00-06:00 UTC)
    "^N225": SymbolInfo(symbol="^N225", name="Nikkei 225", market="Japan", category="Index", currency="JPY"),
    "^HSI": SymbolInfo(symbol="^HSI", name="Hang Seng Index", market="Hong Kong", category="Index", currency="HKD"),
    
    # European Indices (Trading: 8:00 AM - 4:30 PM CET / 07:00-15:30 UTC)
    "^FTSE": SymbolInfo(symbol="^FTSE", name="FTSE 100", market="UK", category="Index", currency="GBP"),
    "^GDAXI": SymbolInfo(symbol="^GDAXI", name="DAX Performance Index", market="Germany", category="Index", currency="EUR"),
    "^FCHI": SymbolInfo(symbol="^FCHI", name="CAC 40", market="France", category="Index", currency="EUR"),
}

# Period configurations
PERIOD_CONFIG = {
    TimePeriod.HOUR_24: {"days": 1, "description": "24 Hours"},
    TimePeriod.DAYS_3: {"days": 3, "description": "3 Days"},
    TimePeriod.WEEK_1: {"days": 7, "description": "1 Week"},
    TimePeriod.WEEKS_2: {"days": 14, "description": "2 Weeks"},
    TimePeriod.MONTH_1: {"days": 30, "description": "1 Month"},
    TimePeriod.MONTHS_3: {"days": 90, "description": "3 Months"},
    TimePeriod.MONTHS_6: {"days": 180, "description": "6 Months"},
    TimePeriod.YEAR_1: {"days": 365, "description": "1 Year"},
    TimePeriod.YEARS_2: {"days": 730, "description": "2 Years"},
}

# Market trading hours (UTC)
MARKET_HOURS = {
    "Japan": {"open": 0, "close": 6},      # 00:00-06:00 UTC
    "Hong Kong": {"open": 1, "close": 8},  # 01:00-08:00 UTC  
    "UK": {"open": 8, "close": 16},        # 08:00-16:00 UTC
    "Germany": {"open": 7, "close": 15},   # 07:00-15:30 UTC
    "France": {"open": 7, "close": 15},    # 07:00-15:30 UTC
    "US": {"open": 14, "close": 21}        # 14:30-21:00 UTC
}

def generate_global_24h_data(symbols: List[str]) -> Dict[str, List[MarketDataPoint]]:
    """Generate 24-hour global market flow data"""
    result = {}
    
    # Create 24 hours of data points (every 30 minutes = 48 points)
    base_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    for symbol in symbols:
        market = SYMBOLS_DB[symbol].market
        data_points = []
        
        # Base price for the market
        if symbol.startswith('^'):
            base_price = np.random.uniform(3000, 40000)
        else:
            base_price = np.random.uniform(50, 500)
        
        current_price = base_price
        
        for i in range(48):  # 48 points for 24 hours (every 30 min)
            timestamp = base_time + timedelta(minutes=i * 30)
            hour_utc = timestamp.hour
            
            # Check if market is open for this hour
            market_hours = MARKET_HOURS.get(market, {"open": 0, "close": 23})
            is_market_open = market_hours["open"] <= hour_utc < market_hours["close"]
            
            if is_market_open:
                # Active trading - more volatility
                change = np.random.normal(0, 0.015)  # 1.5% volatility
                volume_multiplier = 1.0
            else:
                # Market closed - minimal movement
                change = np.random.normal(0, 0.002)  # 0.2% volatility
                volume_multiplier = 0.1
            
            current_price *= (1 + change)
            percentage_change = ((current_price - base_price) / base_price) * 100
            
            # Generate OHLC
            high = current_price * (1 + abs(np.random.normal(0, 0.005)))
            low = current_price * (1 - abs(np.random.normal(0, 0.005)))
            open_price = current_price * (1 + np.random.normal(0, 0.002))
            
            base_volume = 1000000 if symbol.startswith('^') else 500000
            volume = int(base_volume * volume_multiplier * np.random.uniform(0.5, 2.0))
            
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
        
        result[symbol] = data_points
    
    return result

def generate_demo_data(symbol: str, period: TimePeriod) -> List[MarketDataPoint]:
    """Generate realistic demo data"""
    config = PERIOD_CONFIG[period]
    days = config["days"]
    
    # Generate realistic base price
    if symbol.startswith('^'):
        base_price = np.random.uniform(3000, 40000)  # Index range
    elif '.AX' in symbol:
        base_price = np.random.uniform(10, 300)      # Australian stock range
    else:
        base_price = np.random.uniform(50, 500)      # US stock range
    
    # Generate points
    num_points = min(100, days * 2)
    data_points = []
    current_price = base_price
    
    start_time = datetime.now() - timedelta(days=days)
    
    for i in range(num_points):
        # Random walk with mean reversion
        change = np.random.normal(0, 0.02)  # 2% volatility
        current_price = max(current_price * (1 + change), base_price * 0.5)
        
        timestamp = start_time + timedelta(days=(days * i / num_points))
        percentage_change = ((current_price - base_price) / base_price) * 100
        
        # Generate OHLC around current price
        high = current_price * (1 + abs(np.random.normal(0, 0.01)))
        low = current_price * (1 - abs(np.random.normal(0, 0.01)))
        open_price = current_price * (1 + np.random.normal(0, 0.005))
        
        data_points.append(MarketDataPoint(
            timestamp=timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            timestamp_ms=int(timestamp.timestamp() * 1000),
            open=round(open_price, 2),
            high=round(high, 2),
            low=round(low, 2),
            close=round(current_price, 2),
            volume=int(np.random.uniform(100000, 10000000)),
            percentage_change=round(percentage_change, 2)
        ))
    
    return data_points

# Routes
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "GSMT Ver 7.0 API",
        "version": "7.0.0",
        "description": "Global Stock Market Tracker - Railway Optimized",
        "status": "healthy",
        "deployment": "railway",
        "endpoints": {
            "health": "/health",
            "symbols": "/symbols", 
            "analyze": "/analyze",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "7.0.0",
        "service": "GSMT Ver 7.0 API",
        "timestamp": datetime.now().isoformat(),
        "deployment": "railway",
        "supported_symbols": len(SYMBOLS_DB),
        "features": [
            "Demo data generation",
            "Percentage-based analysis",
            "Global market coverage",
            "Railway optimized"
        ]
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
    """Search symbols"""
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
                "currency": info.currency
            })
    
    return {
        "query": query,
        "results": results[:limit],
        "total_found": len(results)
    }

@app.get("/global-24h")
async def get_global_24h_flow():
    """Get 24-hour global market flow across Asia, Europe, US"""
    
    # Key global indices for 24-hour tracking
    global_indices = [
        "^N225",   # Nikkei 225 (Japan) - Asian open
        "^HSI",    # Hang Seng (Hong Kong) - Asian continuation  
        "^FTSE",   # FTSE 100 (UK) - European open
        "^GDAXI",  # DAX (Germany) - European main
        "^GSPC",   # S&P 500 (US) - US market
        "^IXIC"    # NASDAQ (US) - US tech
    ]
    
    # Generate 24-hour flow data
    symbol_data = generate_global_24h_data(global_indices)
    symbol_metadata = {symbol: SYMBOLS_DB[symbol] for symbol in global_indices}
    
    # Add market session information
    market_sessions = [
        {"name": "Asian Session", "start": "00:00", "end": "08:00", "markets": ["Japan", "Hong Kong"], "color": "#f59e0b"},
        {"name": "European Session", "start": "07:00", "end": "16:00", "markets": ["UK", "Germany", "France"], "color": "#10b981"},
        {"name": "US Session", "start": "14:30", "end": "21:00", "markets": ["US"], "color": "#3b82f6"}
    ]
    
    return {
        "success": True,
        "data": symbol_data,
        "metadata": symbol_metadata,
        "market_sessions": market_sessions,
        "period": "24h",
        "chart_type": "percentage",
        "timestamp": datetime.now().isoformat(),
        "total_symbols": len(global_indices),
        "successful_symbols": len(symbol_data),
        "description": "24-hour global market flow tracking across time zones"
    }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_symbols(request: AnalysisRequest):
    """Analyze symbols with demo data"""
    
    # Special handling for 24-hour global flow
    if request.period == TimePeriod.HOUR_24 and len(request.symbols) > 1:
        # Check if this looks like a global market request
        markets = set(SYMBOLS_DB[symbol].market for symbol in request.symbols if symbol in SYMBOLS_DB)
        if len(markets) > 1:  # Multiple markets = global tracking
            symbol_data = generate_global_24h_data(request.symbols)
            symbol_metadata = {symbol: SYMBOLS_DB[symbol] for symbol in request.symbols if symbol in SYMBOLS_DB}
            
            return AnalysisResponse(
                success=True,
                data=symbol_data,
                metadata=symbol_metadata,
                period=request.period.value,
                chart_type=request.chart_type.value,
                timestamp=datetime.now().isoformat(),
                total_symbols=len(request.symbols),
                successful_symbols=len(symbol_data)
            )
    
    # Validate symbols
    invalid_symbols = [s for s in request.symbols if s not in SYMBOLS_DB]
    if invalid_symbols:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported symbols: {', '.join(invalid_symbols)}"
        )
    
    # Generate demo data for all symbols
    symbol_data = {}
    symbol_metadata = {}
    
    for symbol in request.symbols:
        try:
            data = generate_demo_data(symbol, request.period)
            symbol_data[symbol] = data
            symbol_metadata[symbol] = SYMBOLS_DB[symbol]
        except Exception as e:
            logger.error(f"Failed to generate data for {symbol}: {str(e)}")
    
    return AnalysisResponse(
        success=True,
        data=symbol_data,
        metadata=symbol_metadata,
        period=request.period.value,
        chart_type=request.chart_type.value,
        timestamp=datetime.now().isoformat(),
        total_symbols=len(request.symbols),
        successful_symbols=len(symbol_data)
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """Application startup"""
    logger.info("🚀 GSMT Ver 7.0 API Starting")
    logger.info(f"📊 Loaded {len(SYMBOLS_DB)} symbols")
    logger.info("🎯 Railway deployment ready")
    logger.info("✅ Health check endpoint active at /health")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")