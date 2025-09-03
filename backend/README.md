# Global Markets Tracker - Backend API

A Python Flask backend that provides real-time stock market data using YFinance for the Global Markets Tracker frontend.

## Features

- **Real YFinance Data**: Fetches live stock market data from Yahoo Finance
- **Australian Market Focus**: Optimized for ASX indices and global markets
- **Intraday Support**: 5m, 15m, 30m, 1h intervals
- **CORS Enabled**: Ready for frontend integration
- **Bulk Data API**: Fetch multiple symbols efficiently
- **Error Handling**: Robust error handling and logging
- **Production Ready**: Configured for Heroku/Railway deployment

## API Endpoints

### Health Check
```
GET /api/health
```
Returns server status and uptime information.

### Available Symbols
```
GET /api/symbols
```
Returns list of all supported stock symbols.

### Single Stock Data
```
GET /api/stock/<symbol>?interval=5m&date=2024-01-15
```
**Parameters:**
- `symbol`: Stock symbol (e.g., `^AXJO`, `^GSPC`)
- `interval`: `5m`, `15m`, `30m`, `1h`, `1d` (default: `5m`)
- `date`: `YYYY-MM-DD` format (default: today)

**Response:**
```json
{
  "symbol": "^AXJO",
  "name": "ASX 200",
  "interval": "5m",
  "date": "2024-01-15",
  "data": [
    {
      "time": "2024-01-15T10:00:00",
      "open": 7500.0,
      "high": 7520.0,
      "low": 7495.0,
      "close": 7510.0,
      "volume": 1000000,
      "value": 7510.0
    }
  ],
  "count": 78
}
```

### Bulk Data
```
POST /api/bulk
```
**Request Body:**
```json
{
  "symbols": ["^AXJO", "^GSPC", "^N225"],
  "interval": "5m", 
  "date": "2024-01-15"
}
```

**Response:**
```json
{
  "results": {
    "^AXJO": {
      "name": "ASX 200",
      "data": [...]
    },
    "^GSPC": {
      "name": "S&P 500", 
      "data": [...]
    }
  },
  "interval": "5m",
  "date": "2024-01-15"
}
```

## Supported Indices

### 🇦🇺 Australian
- `^AXJO` - ASX 200
- `^AXKO` - ASX 300
- `^AFLI` - ASX 50
- `^AXMD` - ASX 100
- `^AXSO` - ASX Small Cap

### 🌏 Asian
- `^N225` - Nikkei 225 (Japan)
- `^HSI` - Hang Seng (Hong Kong)
- `000001.SS` - SSE Composite (China)
- `^NSEI` - NIFTY 50 (India)

### 🌍 European
- `^FTSE` - FTSE 100 (UK)
- `^GDAXI` - DAX (Germany)
- `^FCHI` - CAC 40 (France)

### 🌎 US
- `^GSPC` - S&P 500
- `^DJI` - Dow Jones
- `^IXIC` - NASDAQ

## Local Development

### Prerequisites
- Python 3.11+
- pip

### Setup
```bash
# Clone and navigate to backend
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\\Scripts\\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env

# Run development server
python app.py
```

The API will be available at `http://localhost:5000`

### Testing the API
```bash
# Health check
curl http://localhost:5000/api/health

# Get ASX 200 data
curl "http://localhost:5000/api/stock/^AXJO?interval=5m&date=2024-01-15"

# Get all symbols
curl http://localhost:5000/api/symbols
```

## Deployment

### Railway (Recommended - Free Tier)

1. **Fork/Clone this repo**
2. **Connect to Railway**: 
   - Go to [railway.app](https://railway.app)
   - Connect your GitHub account
   - Click "Deploy from GitHub repo"
   - Select your repo and choose `/backend` folder

3. **Railway will auto-deploy** using the `Procfile`

4. **Your API URL**: `https://your-app-name.railway.app`

### Heroku (Free tier discontinued but instructions for paid)

```bash
# Install Heroku CLI
# Create Heroku app
heroku create your-markets-api

# Set buildpack
heroku buildpacks:set heroku/python

# Deploy
git subtree push --prefix backend heroku main

# Check logs
heroku logs --tail
```

### Render (Free Tier Available)

1. Connect GitHub repo
2. Choose "Web Service"
3. Set:
   - **Root Directory**: `backend`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`

## Environment Variables

For production deployment, set these environment variables:

```
FLASK_ENV=production
FLASK_DEBUG=False
PORT=5000
```

## Frontend Integration

Update your frontend to use the real API:

```javascript
// In js/providers.js
async function fetchYFinanceSeries(symbol, interval, selectedDate) {
  const API_BASE = 'https://your-api-url.railway.app';
  
  const response = await fetch(
    `${API_BASE}/api/stock/${symbol}?interval=${interval}&date=${selectedDate}`
  );
  
  const json = await response.json();
  
  return json.data.map(item => ({
    time: new Date(item.time),
    open: item.open,
    high: item.high,
    low: item.low, 
    close: item.close,
    value: item.value
  }));
}
```

## Performance Notes

- **Rate Limiting**: YFinance has unofficial rate limits (~200 requests/hour)
- **Caching**: Consider implementing Redis cache for production
- **Bulk Requests**: Use `/api/bulk` endpoint for multiple symbols
- **Error Handling**: API gracefully handles YFinance failures

## License

This project is open source and available under the MIT License.