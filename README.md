# Global Stock Market Tracker - Local Deployment with Live Data

A 24-hour UTC timeline tracker for global stock indices with **live market data integration**, designed for local deployment.

## 🎯 Project Overview

This application provides a comprehensive view of global stock market indices on a 24-hour UTC timeline with **live data integration**, allowing users to:

- **Live Market Data**: Real-time data from Yahoo Finance and Alpha Vantage APIs
- **Select Multiple Indices**: Choose from 48+ global stock indices across 13 markets
- **24-Hour UTC Timeline**: View all markets on a synchronized UTC timeline with actual trading hours
- **Market Session Tracking**: Real-time indicators showing which markets are open/closed
- **Cross-Market Analysis**: Compare live indices from different time zones simultaneously
- **Opening/Closing Visualization**: Market opening and closing times displayed on the timeline
- **Smart Fallbacks**: Graceful fallback to demo data when live data is unavailable

## 🌍 Supported Markets

- **Asian-Pacific**: Japan (Nikkei 225), Hong Kong (Hang Seng), Australia (ASX 200), South Korea (KOSPI), China (Shanghai Composite)
- **European**: UK (FTSE 100), Germany (DAX), France (CAC 40), Netherlands (AEX), Spain (IBEX 35)
- **Americas**: US (S&P 500, NASDAQ, Dow Jones), Canada, Brazil
- **Commodities**: Gold, Oil, Silver futures
- **Cryptocurrencies**: Bitcoin, Ethereum

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Internet connection for:
  - External CDN resources (Tailwind CSS, ECharts, etc.)
  - Live market data APIs (Yahoo Finance, Alpha Vantage)
- Optional: API keys for enhanced data access

### Installation

1. **Clone or download the project**
   ```bash
   # If you have the project files
   cd global-stock-market-tracker
   ```

2. **Configure live data (optional but recommended)**
   ```bash
   # Copy environment template
   cp .env.example .env
   
   # Edit .env file and add your API keys:
   # ALPHA_VANTAGE_API_KEY=your_key_here
   # Get free key at: https://www.alphavantage.co/support/#api-key
   ```

3. **Run the automated setup**
   ```bash
   python3 setup.py
   ```

4. **Start the application**
   ```bash
   # Development mode (with auto-reload)
   ./dev.sh
   
   # Or production mode (with process management)
   ./start.sh
   ```

5. **Access the application**
   - Frontend: http://localhost:8000/
   - API Documentation: http://localhost:8000/api/docs
   - API Health: http://localhost:8000/api/health
   - **Data Status: http://localhost:8000/api/data-status** 📊

### Manual Installation

If you prefer manual installation:

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install process manager (optional)
pip install supervisor

# Start the application
python3 app.py
```

## 📖 Usage Guide

### 1. Selecting Indices

- Use the search box to find specific indices
- Browse suggested indices organized by region:
  - **Asian-Pacific**: Nikkei 225, Hang Seng, ASX 200
  - **European**: FTSE 100, DAX, CAC 40
  - **Americas**: S&P 500, NASDAQ, Dow Jones

### 2. Chart Types

- **Percentage Change**: View relative performance (recommended for comparison)
- **Price Values**: View absolute price movements

### 3. 24-Hour Timeline Features

- **UTC Synchronization**: All data displayed in UTC time
- **Market Hours**: Vertical lines show opening/closing times
- **Current Time Indicator**: Yellow line shows current UTC time
- **Market Status**: Color-coded indicators for open/closed markets

### 4. Market Sessions

The application tracks these market sessions in UTC:

- **Asian Session**: 00:00-08:00 UTC
- **European Session**: 08:00-17:00 UTC  
- **American Session**: 14:00-21:00 UTC
- **Overlap Periods**: Times when multiple markets are open

## 🔧 Configuration

### Environment Variables

- `PORT`: Server port (default: 8000)
- `PYTHONPATH`: Python path (auto-configured)

### Process Management

#### Using Supervisor (Recommended)

```bash
# Start supervisor daemon
supervisord -c supervisord.conf

# Control the application
supervisorctl -c supervisord.conf status
supervisorctl -c supervisord.conf restart global-market-tracker
supervisorctl -c supervisord.conf stop global-market-tracker
```

#### Using PM2 (if available)

```bash
# Install PM2
npm install -g pm2

# Start application
pm2 start ecosystem.config.json

# Monitor
pm2 status
pm2 logs global-market-tracker --nostream
```

## 📶 Live Data Setup

### Quick Setup (Recommended)

The application works immediately with Yahoo Finance (no API key required):

```bash
# Start with Yahoo Finance (default)
python3 setup.py
./start.sh
```

### Enhanced Setup with Alpha Vantage

For more reliable data access, add an Alpha Vantage API key:

1. **Get free API key**: https://www.alphavantage.co/support/#api-key
2. **Configure environment**:
   ```bash
   echo "ALPHA_VANTAGE_API_KEY=your_actual_key_here" >> .env
   ```
3. **Restart application**: `./stop.sh && ./start.sh`

### Data Provider Status

| Provider | Status | Limits | Setup Required |
|----------|--------|--------|----------------|
| **Yahoo Finance** | 🟢 Active | Rate limited | None |
| **Alpha Vantage** | 🟡 Optional | 25 calls/day (free) | API key |
| **Demo Fallback** | 🟢 Always available | None | None |

### Verify Live Data

```bash
# Check data source status
curl http://localhost:8000/api/data-status

# Test live data fetch
curl -X POST http://localhost:8000/api/analyze \
  -d '{"symbols":["^FTSE"],"chart_type":"percentage"}' \
  -H "Content-Type: application/json"
```

## 📊 API Reference

### Core Endpoints

- `GET /api/health` - Health check with live data status
- `GET /api/data-status` - **Live data source status** 🆕
- `GET /api/symbols` - All available symbols organized by market
- `GET /api/suggested-indices` - Suggested indices by region
- `GET /api/market-hours` - Current market status and hours
- `POST /api/analyze` - **Live market data analysis** with 24h UTC timeline

### Example API Usage

```bash
# Check live data status
curl http://localhost:8000/api/data-status

# Get current market status  
curl http://localhost:8000/api/market-hours

# Analyze multiple indices with live data
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["^FTSE", "^GSPC", "^N225"],
    "chart_type": "percentage"
  }'

# Check application health and data sources
curl http://localhost:8000/api/health
```

## 🛠️ Development

### Development Mode

```bash
# Start with auto-reload
./dev.sh

# Or manually
python3 -c "
import uvicorn
uvicorn.run('app:app', host='0.0.0.0', port=8000, reload=True)
"
```

### Project Structure

```
global-stock-market-tracker/
├── app.py                    # Main FastAPI application
├── setup.py                  # Automated setup script
├── requirements.txt          # Python dependencies
├── frontend/                 # Frontend application
│   ├── index.html           # Main HTML file
│   ├── assets/
│   │   ├── app.js           # Frontend JavaScript
│   │   └── style.css        # Custom CSS
├── logs/                     # Application logs
├── supervisord.conf          # Supervisor configuration
├── ecosystem.config.json     # PM2 configuration (if used)
├── start.sh                  # Production startup script
├── stop.sh                   # Stop script
├── dev.sh                    # Development startup script
└── health-check.sh           # Health check script
```

### Adding New Indices

To add new indices, update the `SYMBOLS_DB` in `app.py`:

```python
SYMBOLS_DB = {
    # Add new indices here
    "NEW_SYMBOL": SymbolInfo(
        symbol="NEW_SYMBOL", 
        name="Index Name", 
        market="Market", 
        category="Index"
    ),
}
```

## 🌐 Deployment Features

### Local-First Design

- **No External Dependencies**: Runs completely locally
- **Self-Contained**: All necessary files included
- **Process Management**: Built-in supervisor/PM2 support
- **Auto-Setup**: One-command installation

### Performance Optimizations

- **Efficient Data Generation**: Optimized for 24-hour timeline
- **Minimal Resource Usage**: Lightweight Python backend
- **Fast Frontend**: Vanilla JavaScript with modern CDN libraries
- **Smart Caching**: Reduced redundant calculations

## 🔍 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Check what's using port 8000
   lsof -i :8000
   
   # Use different port
   PORT=8001 python3 app.py
   ```

2. **Permission Denied**
   ```bash
   # Make scripts executable
   chmod +x *.sh
   ```

3. **API Not Responding**
   ```bash
   # Check application status
   ./health-check.sh
   
   # Check logs
   tail -f logs/app.log
   ```

4. **Frontend Not Loading**
   - Ensure the `frontend/` directory exists
   - Check browser console for CDN loading errors
   - Verify all files are present

### Log Locations

- **Application Logs**: `logs/app.log`
- **Error Logs**: `logs/error.log`
- **Supervisor Logs**: `logs/supervisord.log`

## 📈 Features Comparison

| Feature | Original Project | This Version |
|---------|------------------|--------------|
| **Data Source** | Demo only | **Live market data** + Demo fallback |
| **Deployment** | Railway + Netlify | Local only |
| **Time Ranges** | Multiple (24h-10Y) | 24h UTC only |
| **Focus** | General analysis | **Live global market flow** |
| **Dependencies** | Cloud services | Self-contained + APIs |
| **Complexity** | High | Simplified |
| **Market Hours** | Basic | **Precise UTC timing** |
| **Data Providers** | None | Yahoo Finance + Alpha Vantage |
| **Real-time** | No | **Yes (during market hours)** |

## 🎮 Usage Examples

### Monitor Asian Markets Opening
Set up indices from Asian markets and watch them open in real-time during UTC 00:00-08:00.

### Track Market Handoffs
Monitor how markets hand off activity:
1. Asian markets (00:00-08:00 UTC)
2. European markets (08:00-17:00 UTC)  
3. American markets (14:00-21:00 UTC)

### Compare Cross-Market Performance
Select indices from different regions to see how global events affect various markets simultaneously.

## 📜 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

This is a self-contained local deployment project. To contribute:

1. Fork the repository
2. Make your changes
3. Test locally using `./dev.sh`
4. Submit a pull request

## 🎯 Original Project Intent

This project was converted from a cloud-deployed application to fulfill the original intent:
- **Map stock indices from across the world on a 24-hour timeline**
- **Allow users to select the indices they want to view**
- **Plot indices on a UTC timeline**
- **Transform opening and closing times into UTC and plot them**
- **Remove timeframe options to focus on 24-hour view only**

All original requirements have been implemented with a focus on local deployment and simplified operation.

---

**Built for local deployment • No cloud dependencies • 24-hour UTC focus • Real-time market tracking**