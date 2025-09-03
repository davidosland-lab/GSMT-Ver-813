"""
GSMT Ver 6.0 - Modern FastAPI Backend
Global Stock Market Tracker with Percentage Analysis

Modern API-first architecture with:
- FastAPI for high performance
- Pydantic for data validation
- Async operations
- Comprehensive error handling
- Built-in documentation
- Health monitoring
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
from typing import List, Dict, Optional, Union
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import pytz
import asyncio
import aiohttp
import logging
import os
from enum import Enum
import json
from functools import lru_cache
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# FastAPI app instance
app = FastAPI(
    title="GSMT Ver 6.0 API",
    description="Global Stock Market Tracker - Modern API for percentage-based financial analysis",
    version="6.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
AUSTRALIA_TZ = pytz.timezone('Australia/Sydney')
CACHE_TTL = 300  # 5 minutes cache

# Enums for better type safety
class MarketType(str, Enum):
    INDEX = "index"
    STOCK = "stock"

class TimeInterval(str, Enum):
    MIN_5 = "5m"
    MIN_15 = "15m"
    MIN_30 = "30m"
    HOUR_1 = "1h"
    DAY_1 = "1d"
    WEEK_1 = "1wk"

class TimePeriod(str, Enum):
    DAY_1 = "1d"
    DAYS_3 = "3d"
    WEEK_1 = "1w"
    WEEKS_2 = "2w"
    MONTH_1 = "1M"
    MONTHS_3 = "3M"
    MONTHS_6 = "6M"
    YEAR_1 = "1Y"
    YEARS_2 = "2Y"

# Pydantic models for request/response validation
class SymbolInfo(BaseModel):
    symbol: str
    name: str
    market: str
    type: MarketType
    sector: Optional[str] = None
    currency: Optional[str] = "USD"

class MarketDataPoint(BaseModel):
    timestamp: datetime
    timestamp_raw: int
    open: float
    high: float
    low: float
    close: float
    volume: int
    percentage_change: float
    data_source: str

class SymbolDataResponse(BaseModel):
    symbol: str
    success: bool
    data: List[MarketDataPoint]
    period: TimePeriod
    interval: TimeInterval
    points: int
    base_price: float
    current_change: float
    data_source: str
    error: Optional[str] = None

class BulkDataRequest(BaseModel):
    symbols: List[str]
    period: TimePeriod = TimePeriod.DAY_1
    interval: Optional[TimeInterval] = None

    @validator('symbols')
    def validate_symbols(cls, v):
        if not v or len(v) == 0:
            raise ValueError('At least one symbol is required')
        if len(v) > 50:
            raise ValueError('Maximum 50 symbols allowed')
        return v

class BulkDataResponse(BaseModel):
    results: Dict[str, SymbolDataResponse]
    success_rate: str
    total_symbols: int
    successful_fetches: int
    period: TimePeriod
    interval: TimeInterval
    timestamp: datetime

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime
    features: List[str]
    data_sources: int
    supported_symbols: int
    time_periods: List[str]
    uptime_seconds: float

class ValidationResponse(BaseModel):
    symbol: str
    valid: bool
    supported: bool
    format_valid: bool
    category: Optional[str] = None
    metadata: Optional[SymbolInfo] = None
    suggestions: List[str] = []

# Market data configuration
SUPPORTED_MARKETS = {
    "indices": {
        "^GSPC": SymbolInfo(symbol="^GSPC", name="S&P 500", market="usa", type=MarketType.INDEX, sector="Broad Market"),
        "^IXIC": SymbolInfo(symbol="^IXIC", name="NASDAQ Composite", market="usa", type=MarketType.INDEX, sector="Technology"),
        "^DJI": SymbolInfo(symbol="^DJI", name="Dow Jones Industrial Average", market="usa", type=MarketType.INDEX, sector="Industrials"),
        "^AXJO": SymbolInfo(symbol="^AXJO", name="ASX 200", market="australia", type=MarketType.INDEX, sector="Broad Market", currency="AUD"),
        "^N225": SymbolInfo(symbol="^N225", name="Nikkei 225", market="japan", type=MarketType.INDEX, sector="Broad Market", currency="JPY"),
        "^HSI": SymbolInfo(symbol="^HSI", name="Hang Seng Index", market="hong_kong", type=MarketType.INDEX, sector="Broad Market", currency="HKD"),
        "000001.SS": SymbolInfo(symbol="000001.SS", name="Shanghai Composite", market="china", type=MarketType.INDEX, sector="Broad Market", currency="CNY"),
        "^FTSE": SymbolInfo(symbol="^FTSE", name="FTSE 100", market="uk", type=MarketType.INDEX, sector="Broad Market", currency="GBP"),
        "^GDAXI": SymbolInfo(symbol="^GDAXI", name="DAX Performance Index", market="germany", type=MarketType.INDEX, sector="Broad Market", currency="EUR"),
        "^FCHI": SymbolInfo(symbol="^FCHI", name="CAC 40", market="france", type=MarketType.INDEX, sector="Broad Market", currency="EUR"),
    },
    "australian_stocks": {
        "CBA.AX": SymbolInfo(symbol="CBA.AX", name="Commonwealth Bank of Australia", market="australia", type=MarketType.STOCK, sector="Financial Services", currency="AUD"),
        "BHP.AX": SymbolInfo(symbol="BHP.AX", name="BHP Group Limited", market="australia", type=MarketType.STOCK, sector="Basic Materials", currency="AUD"),
        "CSL.AX": SymbolInfo(symbol="CSL.AX", name="CSL Limited", market="australia", type=MarketType.STOCK, sector="Healthcare", currency="AUD"),
        "WBC.AX": SymbolInfo(symbol="WBC.AX", name="Westpac Banking Corporation", market="australia", type=MarketType.STOCK, sector="Financial Services", currency="AUD"),
        "ANZ.AX": SymbolInfo(symbol="ANZ.AX", name="Australia and New Zealand Banking Group Limited", market="australia", type=MarketType.STOCK, sector="Financial Services", currency="AUD"),
        "NAB.AX": SymbolInfo(symbol="NAB.AX", name="National Australia Bank Limited", market="australia", type=MarketType.STOCK, sector="Financial Services", currency="AUD"),
        "WES.AX": SymbolInfo(symbol="WES.AX", name="Wesfarmers Limited", market="australia", type=MarketType.STOCK, sector="Consumer Defensive", currency="AUD"),
        "TLS.AX": SymbolInfo(symbol="TLS.AX", name="Telstra Corporation Limited", market="australia", type=MarketType.STOCK, sector="Communication Services", currency="AUD"),
        "WOW.AX": SymbolInfo(symbol="WOW.AX", name="Woolworths Group Limited", market="australia", type=MarketType.STOCK, sector="Consumer Defensive", currency="AUD"),
        "FMG.AX": SymbolInfo(symbol="FMG.AX", name="Fortescue Metals Group Ltd", market="australia", type=MarketType.STOCK, sector="Basic Materials", currency="AUD"),
        "MQG.AX": SymbolInfo(symbol="MQG.AX", name="Macquarie Group Limited", market="australia", type=MarketType.STOCK, sector="Financial Services", currency="AUD"),
        "COL.AX": SymbolInfo(symbol="COL.AX", name="Coles Group Limited", market="australia", type=MarketType.STOCK, sector="Consumer Defensive", currency="AUD"),
        "TCL.AX": SymbolInfo(symbol="TCL.AX", name="Transurban Group", market="australia", type=MarketType.STOCK, sector="Real Estate", currency="AUD"),
        "RIO.AX": SymbolInfo(symbol="RIO.AX", name="Rio Tinto Limited", market="australia", type=MarketType.STOCK, sector="Basic Materials", currency="AUD"),
        "STO.AX": SymbolInfo(symbol="STO.AX", name="Santos Limited", market="australia", type=MarketType.STOCK, sector="Energy", currency="AUD"),
    },
    "us_stocks": {
        "AAPL": SymbolInfo(symbol="AAPL", name="Apple Inc.", market="usa", type=MarketType.STOCK, sector="Technology"),
        "GOOGL": SymbolInfo(symbol="GOOGL", name="Alphabet Inc.", market="usa", type=MarketType.STOCK, sector="Technology"),
        "MSFT": SymbolInfo(symbol="MSFT", name="Microsoft Corporation", market="usa", type=MarketType.STOCK, sector="Technology"),
        "AMZN": SymbolInfo(symbol="AMZN", name="Amazon.com Inc.", market="usa", type=MarketType.STOCK, sector="Consumer Cyclical"),
        "TSLA": SymbolInfo(symbol="TSLA", name="Tesla Inc.", market="usa", type=MarketType.STOCK, sector="Consumer Cyclical"),
        "META": SymbolInfo(symbol="META", name="Meta Platforms Inc.", market="usa", type=MarketType.STOCK, sector="Technology"),
        "NVDA": SymbolInfo(symbol="NVDA", name="NVIDIA Corporation", market="usa", type=MarketType.STOCK, sector="Technology"),
        "NFLX": SymbolInfo(symbol="NFLX", name="Netflix Inc.", market="usa", type=MarketType.STOCK, sector="Communication Services"),
        "V": SymbolInfo(symbol="V", name="Visa Inc.", market="usa", type=MarketType.STOCK, sector="Financial Services"),
        "JPM": SymbolInfo(symbol="JPM", name="JPMorgan Chase & Co.", market="usa", type=MarketType.STOCK, sector="Financial Services"),
        "JNJ": SymbolInfo(symbol="JNJ", name="Johnson & Johnson", market="usa", type=MarketType.STOCK, sector="Healthcare"),
        "PG": SymbolInfo(symbol="PG", name="Procter & Gamble Company", market="usa", type=MarketType.STOCK, sector="Consumer Defensive"),
        "UNH": SymbolInfo(symbol="UNH", name="UnitedHealth Group Incorporated", market="usa", type=MarketType.STOCK, sector="Healthcare"),
        "HD": SymbolInfo(symbol="HD", name="Home Depot Inc.", market="usa", type=MarketType.STOCK, sector="Consumer Cyclical"),
    }
}

# Time period configurations
TIME_PERIODS = {
    TimePeriod.DAY_1: {"days": 1, "interval": TimeInterval.MIN_5, "description": "24 Hours (5min intervals)"},
    TimePeriod.DAYS_3: {"days": 3, "interval": TimeInterval.MIN_15, "description": "3 Days (15min intervals)"},
    TimePeriod.WEEK_1: {"days": 7, "interval": TimeInterval.MIN_30, "description": "1 Week (30min intervals)"},
    TimePeriod.WEEKS_2: {"days": 14, "interval": TimeInterval.HOUR_1, "description": "2 Weeks (1hr intervals)"},
    TimePeriod.MONTH_1: {"days": 30, "interval": TimeInterval.DAY_1, "description": "1 Month (daily)"},
    TimePeriod.MONTHS_3: {"days": 90, "interval": TimeInterval.DAY_1, "description": "3 Months (daily)"},
    TimePeriod.MONTHS_6: {"days": 180, "interval": TimeInterval.DAY_1, "description": "6 Months (daily)"},
    TimePeriod.YEAR_1: {"days": 365, "interval": TimeInterval.DAY_1, "description": "1 Year (daily)"},
    TimePeriod.YEARS_2: {"days": 730, "interval": TimeInterval.WEEK_1, "description": "2 Years (weekly)"},
}

# Application startup time for uptime calculation
app_start_time = datetime.now()

# Data source management
class DataSourceManager:
    """Manages multiple data sources with fallback logic"""
    
    def __init__(self):
        self.sources = [
            {"name": "yfinance", "priority": 1, "active": True},
            {"name": "demo", "priority": 999, "active": True}  # Always available fallback
        ]
    
    @lru_cache(maxsize=128)
    def get_symbol_data(self, symbol: str, period: TimePeriod, interval: TimeInterval) -> SymbolDataResponse:
        """Get symbol data with caching and fallback logic"""
        
        for source in sorted(self.sources, key=lambda x: x["priority"]):
            if not source["active"]:
                continue
                
            try:
                if source["name"] == "yfinance":
                    return self._fetch_yfinance_data(symbol, period, interval)
                elif source["name"] == "demo":
                    return self._generate_demo_data(symbol, period, interval)
                    
            except Exception as e:
                logger.warning(f"Data source {source['name']} failed for {symbol}: {str(e)}")
                continue
        
        # If all sources fail, raise an exception
        raise HTTPException(status_code=503, detail=f"All data sources failed for symbol {symbol}")
    
    def _fetch_yfinance_data(self, symbol: str, period: TimePeriod, interval: TimeInterval) -> SymbolDataResponse:
        """Fetch data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            period_config = TIME_PERIODS[period]
            
            # Convert period to yfinance format
            if period_config["days"] <= 7:
                yf_period = f"{period_config['days']}d"
            elif period_config["days"] <= 60:
                yf_period = f"{period_config['days']}d"
            elif period_config["days"] <= 365:
                yf_period = f"{period_config['days']//30}mo"
            else:
                yf_period = f"{period_config['days']//365}y"
            
            # Fetch historical data
            hist = ticker.history(period=yf_period, interval=interval.value)
            
            if hist.empty:
                raise ValueError("No data returned from Yahoo Finance")
            
            # Convert to our format and calculate percentage changes
            data_points = []
            prices = hist['Close'].values
            base_price = prices[0] if len(prices) > 0 else 0
            
            for i, (timestamp, row) in enumerate(hist.iterrows()):
                percentage_change = ((prices[i] - base_price) / base_price * 100) if base_price != 0 else 0
                
                data_points.append(MarketDataPoint(
                    timestamp=timestamp.to_pydatetime(),
                    timestamp_raw=int(timestamp.timestamp() * 1000),
                    open=float(row['Open']),
                    high=float(row['High']),
                    low=float(row['Low']),
                    close=float(row['Close']),
                    volume=int(row['Volume']) if not pd.isna(row['Volume']) else 0,
                    percentage_change=percentage_change,
                    data_source="yfinance"
                ))
            
            current_change = data_points[-1].percentage_change if data_points else 0
            
            return SymbolDataResponse(
                symbol=symbol,
                success=True,
                data=data_points,
                period=period,
                interval=interval,
                points=len(data_points),
                base_price=base_price,
                current_change=current_change,
                data_source="yfinance"
            )
            
        except Exception as e:
            logger.error(f"YFinance error for {symbol}: {str(e)}")
            raise e
    
    def _generate_demo_data(self, symbol: str, period: TimePeriod, interval: TimeInterval) -> SymbolDataResponse:
        """Generate realistic demo data for testing"""
        try:
            period_config = TIME_PERIODS[period]
            days = period_config["days"]
            
            # Generate realistic base price based on symbol type
            if symbol.startswith('^'):
                base_price = np.random.uniform(3000, 40000)  # Index range
            elif '.AX' in symbol:
                base_price = np.random.uniform(10, 300)      # Australian stock range
            else:
                base_price = np.random.uniform(50, 500)      # US stock range
            
            # Calculate number of points based on interval
            interval_minutes = {
                TimeInterval.MIN_5: 5,
                TimeInterval.MIN_15: 15,
                TimeInterval.MIN_30: 30,
                TimeInterval.HOUR_1: 60,
                TimeInterval.DAY_1: 1440,
                TimeInterval.WEEK_1: 10080
            }
            
            total_minutes = days * 24 * 60
            minutes_per_point = interval_minutes.get(interval, 5)
            num_points = min(1000, total_minutes // minutes_per_point)
            
            # Generate realistic price movement
            data_points = []
            current_price = base_price
            start_time = datetime.now(AUSTRALIA_TZ) - timedelta(days=days)
            
            for i in range(num_points):
                # Random walk with mean reversion
                change_percent = np.random.normal(0, 0.02)  # 2% volatility
                current_price = max(current_price * (1 + change_percent), base_price * 0.5)
                
                timestamp = start_time + timedelta(minutes=i * minutes_per_point)
                percentage_change = ((current_price - base_price) / base_price) * 100
                
                # Generate OHLC around current price
                high = current_price * (1 + abs(np.random.normal(0, 0.01)))
                low = current_price * (1 - abs(np.random.normal(0, 0.01)))
                open_price = current_price * (1 + np.random.normal(0, 0.005))
                
                data_points.append(MarketDataPoint(
                    timestamp=timestamp,
                    timestamp_raw=int(timestamp.timestamp() * 1000),
                    open=float(open_price),
                    high=float(high),
                    low=float(low),
                    close=float(current_price),
                    volume=int(np.random.uniform(100000, 10000000)),
                    percentage_change=percentage_change,
                    data_source="demo"
                ))
            
            current_change = data_points[-1].percentage_change if data_points else 0
            
            return SymbolDataResponse(
                symbol=symbol,
                success=True,
                data=data_points,
                period=period,
                interval=interval,
                points=len(data_points),
                base_price=base_price,
                current_change=current_change,
                data_source="demo"
            )
            
        except Exception as e:
            logger.error(f"Demo data generation error for {symbol}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to generate demo data for {symbol}")

# Initialize data source manager
data_manager = DataSourceManager()

# API Routes
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check endpoint"""
    uptime = (datetime.now() - app_start_time).total_seconds()
    total_symbols = sum(len(category) for category in SUPPORTED_MARKETS.values())
    
    return HealthResponse(
        status="healthy",
        version="6.0.0",
        timestamp=datetime.now(AUSTRALIA_TZ),
        features=[
            "Modern FastAPI architecture",
            "Percentage-based analysis",
            "Multi-source data fallback",
            "Real-time data processing",
            "Comprehensive error handling",
            "Automatic data validation",
            "Built-in API documentation"
        ],
        data_sources=len(data_manager.sources),
        supported_symbols=total_symbols,
        time_periods=[period.value for period in TimePeriod],
        uptime_seconds=uptime
    )

@app.get("/api/symbols")
async def get_symbols():
    """Get all supported symbols with metadata"""
    total_symbols = sum(len(category) for category in SUPPORTED_MARKETS.values())
    
    return {
        "markets": {
            category: {symbol: info.dict() for symbol, info in symbols.items()}
            for category, symbols in SUPPORTED_MARKETS.items()
        },
        "total_symbols": total_symbols,
        "time_periods": {period.value: config for period, config in TIME_PERIODS.items()},
        "features": {
            "percentage_change": True,
            "overlay_charts": True,
            "flexible_periods": True,
            "multi_source_fallback": True,
            "modern_architecture": True
        }
    }

@app.get("/api/periods")
async def get_time_periods():
    """Get available time periods"""
    return {
        "periods": {period.value: config for period, config in TIME_PERIODS.items()},
        "default": TimePeriod.DAY_1.value,
        "description": "Time periods from 24 hours to 2 years with optimized intervals"
    }

@app.get("/api/stock/{symbol}", response_model=SymbolDataResponse)
async def get_stock_data(
    symbol: str,
    period: TimePeriod = TimePeriod.DAY_1,
    interval: Optional[TimeInterval] = None
):
    """Get individual stock data with percentage change analysis"""
    
    # Use default interval if not specified
    if interval is None:
        interval = TIME_PERIODS[period]["interval"]
    
    # Validate symbol
    if not _is_symbol_supported(symbol):
        raise HTTPException(
            status_code=404,
            detail=f"Symbol {symbol} not supported. Use /api/symbols to see available symbols."
        )
    
    try:
        result = data_manager.get_symbol_data(symbol, period, interval)
        return result
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/bulk", response_model=BulkDataResponse)
async def get_bulk_data(request: BulkDataRequest):
    """Get bulk stock data with percentage change analysis"""
    
    # Use default interval if not specified
    interval = request.interval or TIME_PERIODS[request.period]["interval"]
    
    results = {}
    successful_fetches = 0
    
    # Process each symbol
    for symbol in request.symbols:
        try:
            result = data_manager.get_symbol_data(symbol, request.period, interval)
            results[symbol] = result
            if result.success:
                successful_fetches += 1
        except Exception as e:
            logger.error(f"Error processing symbol {symbol}: {str(e)}")
            results[symbol] = SymbolDataResponse(
                symbol=symbol,
                success=False,
                data=[],
                period=request.period,
                interval=interval,
                points=0,
                base_price=0,
                current_change=0,
                data_source="error",
                error=str(e)
            )
    
    success_rate = f"{(successful_fetches / len(request.symbols) * 100):.1f}%" if request.symbols else "0%"
    
    return BulkDataResponse(
        results=results,
        success_rate=success_rate,
        total_symbols=len(request.symbols),
        successful_fetches=successful_fetches,
        period=request.period,
        interval=interval,
        timestamp=datetime.now(AUSTRALIA_TZ)
    )

@app.get("/api/validate/{symbol}", response_model=ValidationResponse)
async def validate_symbol(symbol: str):
    """Validate symbol and check if supported"""
    
    # Check if symbol exists in any category
    symbol_found = False
    category = None
    metadata = None
    
    for cat_name, symbols in SUPPORTED_MARKETS.items():
        if symbol in symbols:
            symbol_found = True
            category = cat_name
            metadata = symbols[symbol]
            break
    
    # Basic format validation
    format_valid = _validate_symbol_format(symbol)
    
    # Generate suggestions if not found
    suggestions = []
    if not symbol_found:
        suggestions = _get_symbol_suggestions(symbol)
    
    return ValidationResponse(
        symbol=symbol,
        valid=symbol_found and format_valid,
        supported=symbol_found,
        format_valid=format_valid,
        category=category,
        metadata=metadata,
        suggestions=suggestions
    )

@app.get("/api/search/{query}")
async def search_symbols(query: str, limit: int = 20):
    """Search symbols by name, symbol, or sector"""
    query = query.lower().strip()
    results = []
    
    for category, symbols in SUPPORTED_MARKETS.items():
        for symbol_info in symbols.values():
            # Search in symbol, name, and sector
            searchable_text = f"{symbol_info.symbol} {symbol_info.name} {symbol_info.sector or ''}".lower()
            
            if query in searchable_text:
                results.append({
                    "symbol": symbol_info.symbol,
                    "name": symbol_info.name,
                    "category": category,
                    "market": symbol_info.market,
                    "type": symbol_info.type,
                    "sector": symbol_info.sector,
                    "relevance": searchable_text.count(query)
                })
    
    # Sort by relevance and limit results
    results.sort(key=lambda x: x["relevance"], reverse=True)
    
    return {
        "query": query,
        "results": results[:limit],
        "total_found": len(results)
    }

# Helper functions
def _is_symbol_supported(symbol: str) -> bool:
    """Check if symbol is supported"""
    for symbols in SUPPORTED_MARKETS.values():
        if symbol in symbols:
            return True
    return False

def _validate_symbol_format(symbol: str) -> bool:
    """Validate symbol format"""
    import re
    patterns = [
        r'^[A-Z]{1,5}$',           # US stocks: AAPL, GOOGL
        r'^[A-Z]{1,5}\.[A-Z]{1,3}$', # International: CBA.AX
        r'^\^[A-Z0-9]{1,10}$'      # Indices: ^GSPC, ^DJI
    ]
    return any(re.match(pattern, symbol) for pattern in patterns)

def _get_symbol_suggestions(symbol: str) -> List[str]:
    """Get symbol suggestions based on partial match"""
    suggestions = []
    symbol_lower = symbol.lower()
    
    for symbols in SUPPORTED_MARKETS.values():
        for sym_info in symbols.values():
            if (symbol_lower in sym_info.symbol.lower() or 
                symbol_lower in sym_info.name.lower()):
                suggestions.append(sym_info.symbol)
    
    return suggestions[:5]  # Limit to 5 suggestions

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now(AUSTRALIA_TZ).isoformat(),
            "path": str(request.url)
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
            "timestamp": datetime.now(AUSTRALIA_TZ).isoformat(),
            "path": str(request.url)
        }
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """Application startup tasks"""
    logger.info("🚀 Starting GSMT Ver 6.0 API Server")
    logger.info("📊 Features: Modern FastAPI, Percentage Analysis, Multi-source Fallback")
    logger.info(f"⏱️ Supported periods: {[p.value for p in TimePeriod]}")
    logger.info(f"🌐 Total symbols: {sum(len(category) for category in SUPPORTED_MARKETS.values())}")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown tasks"""
    logger.info("🛑 Shutting down GSMT Ver 6.0 API Server")

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=False  # Set to True for development
    )