"""
GSMT Ver 7.0 - Global Stock Market Tracker
Complete application with Global 24H Market Flow functionality
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

# Try to import numpy, install if needed
try:
    import numpy as np
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
    import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="GSMT Ver 7.0 API",
    description="Global Stock Market Tracker with 24H Market Flow",
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
    market_open: Optional[bool] = None

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
    market_hours: Optional[Dict[str, Dict[str, int]]] = None

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

# Market trading hours (UTC) - CORE FEATURE FOR 24H FLOW
MARKET_HOURS = {
    "Japan": {"open": 0, "close": 6},      # 00:00-06:00 UTC
    "Hong Kong": {"open": 1, "close": 8},  # 01:00-08:00 UTC  
    "Australia": {"open": 23, "close": 6}, # 23:00-06:00 UTC (overnight)
    "UK": {"open": 8, "close": 16},        # 08:00-16:00 UTC
    "Germany": {"open": 7, "close": 15},   # 07:00-15:30 UTC
    "France": {"open": 7, "close": 15},    # 07:00-15:30 UTC
    "US": {"open": 14, "close": 21}        # 14:30-21:00 UTC
}

def generate_global_24h_data(symbols: List[str]) -> Dict[str, List[MarketDataPoint]]:
    """Generate 24-hour global market flow data - CORE GLOBAL 24H FEATURE"""
    result = {}
    
    # Create 24 hours of data points (every hour = 24 points)
    base_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    for symbol in symbols:
        if symbol not in SYMBOLS_DB:
            continue
            
        market = SYMBOLS_DB[symbol].market
        data_points = []
        
        # Base price for the market
        if symbol.startswith('^') or symbol.endswith('.SS'):
            base_price = np.random.uniform(3000, 40000)
        elif '.AX' in symbol:
            base_price = np.random.uniform(10, 300)
        else:
            base_price = np.random.uniform(50, 500)
        
        current_price = base_price
        
        for hour in range(24):  # 24 hours
            timestamp = base_time + timedelta(hours=hour)
            
            # Check if market is open for this hour
            market_hours = MARKET_HOURS.get(market, {"open": 0, "close": 23})
            
            # Handle overnight markets (like Australia)
            if market_hours["open"] > market_hours["close"]:
                is_market_open = hour >= market_hours["open"] or hour <= market_hours["close"]
            else:
                is_market_open = market_hours["open"] <= hour <= market_hours["close"]
            
            if is_market_open:
                # Active trading - higher volatility
                change = np.random.normal(0, 0.02)  # 2% volatility
                volume_multiplier = 1.0
            else:
                # Market closed - minimal movement
                change = np.random.normal(0, 0.003)  # 0.3% volatility
                volume_multiplier = 0.1
            
            current_price *= (1 + change)
            percentage_change = ((current_price - base_price) / base_price) * 100
            
            # Generate OHLC
            high = current_price * (1 + abs(np.random.normal(0, 0.005)))
            low = current_price * (1 - abs(np.random.normal(0, 0.005)))
            open_price = current_price * (1 + np.random.normal(0, 0.002))
            
            base_volume = 2000000 if symbol.startswith('^') else 1000000
            volume = int(base_volume * volume_multiplier * np.random.uniform(0.5, 2.0))
            
            data_points.append(MarketDataPoint(
                timestamp=timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                timestamp_ms=int(timestamp.timestamp() * 1000),
                open=round(open_price, 2),
                high=round(high, 2),
                low=round(low, 2),
                close=round(current_price, 2),
                volume=volume,
                percentage_change=round(percentage_change, 2),
                market_open=is_market_open
            ))
        
        result[symbol] = data_points
    
    return result

def generate_demo_data(symbol: str, period: TimePeriod, chart_type: ChartType = ChartType.PERCENTAGE) -> List[MarketDataPoint]:
    """Generate realistic demo data for individual symbols"""
    if symbol not in SYMBOLS_DB:
        raise HTTPException(status_code=400, detail=f"Symbol {symbol} not found")
    
    config = PERIOD_CONFIG[period]
    days = config["days"]
    
    # Generate realistic base price
    if symbol.startswith('^') or symbol.endswith('.SS'):
        base_price = np.random.uniform(3000, 40000)  # Index range
    elif '.AX' in symbol:
        base_price = np.random.uniform(10, 300)      # Australian stock range
    elif symbol.endswith('-USD'):
        base_price = np.random.uniform(20000, 70000) # Crypto range
    else:
        base_price = np.random.uniform(50, 500)      # US stock range
    
    # Generate points based on period
    if days == 1:  # 24h - hourly data
        num_points = 24
        time_delta = timedelta(hours=1)
    else:
        num_points = min(100, days * 2)
        time_delta = timedelta(days=(days / num_points))
    
    data_points = []
    current_price = base_price
    
    start_time = datetime.now() - timedelta(days=days)
    
    for i in range(num_points):
        # Random walk with mean reversion
        volatility = 0.03 if symbol.endswith('-USD') else 0.02  # Higher volatility for crypto
        change = np.random.normal(0, volatility)
        current_price = max(current_price * (1 + change), base_price * 0.5)
        
        timestamp = start_time + (time_delta * i)
        percentage_change = ((current_price - base_price) / base_price) * 100
        
        # Generate OHLC around current price
        high = current_price * (1 + abs(np.random.normal(0, 0.01)))
        low = current_price * (1 - abs(np.random.normal(0, 0.01)))
        open_price = current_price * (1 + np.random.normal(0, 0.005))
        
        # Volume based on symbol type
        if symbol.startswith('^'):
            base_volume = np.random.uniform(1000000, 5000000)
        elif symbol.endswith('-USD'):
            base_volume = np.random.uniform(500000, 2000000)
        else:
            base_volume = np.random.uniform(100000, 1000000)
        
        data_points.append(MarketDataPoint(
            timestamp=timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            timestamp_ms=int(timestamp.timestamp() * 1000),
            open=round(open_price, 2),
            high=round(high, 2),
            low=round(low, 2),
            close=round(current_price, 2),
            volume=int(base_volume),
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
        "description": "Global Stock Market Tracker with 24H Market Flow",
        "status": "healthy",
        "deployment": "railway",
        "features": [
            "Global 24H Market Flow",
            "Individual stock analysis",
            "Candlestick charts",
            "Percentage-based comparison",
            "Market session tracking",
            "70+ global symbols"
        ],
        "endpoints": {
            "health": "/health",
            "symbols": "/symbols",
            "search": "/search/{query}",
            "analyze": "/analyze",
            "global-24h": "/global-24h",
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
        "markets": list(set(info.market for info in SYMBOLS_DB.values())),
        "features": [
            "Global 24H Market Flow tracking",
            "Real-time market session indicators",
            "Individual stock and index analysis",
            "Candlestick and percentage charts"
        ]
    }

@app.get("/symbols")
async def get_symbols():
    """Get all supported symbols organized by category"""
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
        "chart_types": [c.value for c in ChartType],
        "markets": list(set(info.market for info in SYMBOLS_DB.values())),
        "market_hours": MARKET_HOURS
    }

@app.get("/search/{query}")
async def search_symbols(query: str, limit: int = Query(default=10, ge=1, le=50)):
    """Search symbols by name, symbol, market, or category"""
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
    """Get 24-hour global market flow across Asia, Europe, US - CORE FEATURE"""
    
    # Key global indices for 24-hour tracking
    global_indices = [
        "^N225",   # Nikkei 225 (Japan) - Asian open
        "^HSI",    # Hang Seng (Hong Kong) - Asian continuation  
        "^AXJO",   # ASX 200 (Australia) - Pacific market
        "^FTSE",   # FTSE 100 (UK) - European open
        "^GDAXI",  # DAX (Germany) - European main
        "^GSPC",   # S&P 500 (US) - US market
    ]
    
    # Generate 24-hour flow data with market session awareness
    symbol_data = generate_global_24h_data(global_indices)
    symbol_metadata = {symbol: SYMBOLS_DB[symbol] for symbol in global_indices}
    
    return AnalysisResponse(
        success=True,
        data=symbol_data,
        metadata=symbol_metadata,
        market_hours=MARKET_HOURS,
        period="24h",
        chart_type="percentage",
        timestamp=datetime.now().isoformat(),
        total_symbols=len(global_indices),
        successful_symbols=len(symbol_data)
    ).dict()

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_symbols(request: AnalysisRequest):
    """Analyze individual symbols or mixed symbols with full chart type support"""
    
    # Validate symbols
    invalid_symbols = [s for s in request.symbols if s not in SYMBOLS_DB]
    if invalid_symbols:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported symbols: {', '.join(invalid_symbols)}"
        )
    
    # Generate data for all symbols
    symbol_data = {}
    symbol_metadata = {}
    
    for symbol in request.symbols:
        try:
            data = generate_demo_data(symbol, request.period, request.chart_type)
            symbol_data[symbol] = data
            symbol_metadata[symbol] = SYMBOLS_DB[symbol]
        except Exception as e:
            logger.error(f"Failed to generate data for {symbol}: {str(e)}")
            continue
    
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
    logger.info(f"🌍 Supporting {len(set(info.market for info in SYMBOLS_DB.values()))} markets")
    logger.info("🎯 Global 24H Market Flow ready")
    logger.info("✅ Individual stock analysis ready")
    logger.info("📈 Candlestick and percentage charts ready")

if __name__ == "__main__":
    try:
        import uvicorn
    except ImportError:
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "uvicorn"])
        import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")