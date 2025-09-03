# GSMT Ver 6.0 - Deployment Guide

## Modern Web Application Deployment

This guide covers deploying the **completely rebuilt GSMT Ver 6.0** with modern architecture:
- **FastAPI Backend** (api/) - High-performance API with percentage analysis
- **Modern SPA Frontend** - Clean JavaScript application with state management
- **Professional deployment workflow** with CI/CD support

---

## 🚀 Quick Deployment Options

### Option 1: Frontend-Only (Recommended for Testing)
Deploy just the frontend with demo data:

1. **Netlify (Recommended)**:
   ```bash
   # Deploy directly from this folder
   netlify deploy --prod --dir .
   ```

2. **GitHub Pages**:
   - Push to GitHub repository
   - Enable GitHub Pages in repository settings
   - Select main branch as source

### Option 2: Full Stack Deployment
Deploy both frontend and backend:

#### Backend (FastAPI API):
1. **Railway** (Recommended):
   ```bash
   cd api/
   railway login
   railway link [your-project]
   railway up
   ```

2. **Heroku**:
   ```bash
   cd api/
   git init
   heroku create your-app-name
   git add .
   git commit -m "Deploy FastAPI backend"
   git push heroku main
   ```

3. **Docker** (Any platform):
   ```bash
   cd api/
   docker build -t gsmt-api .
   docker run -p 8000:8000 gsmt-api
   ```

#### Frontend Configuration:
1. Update API URL in settings or set `REACT_APP_API_URL` environment variable
2. Deploy frontend to Netlify/Vercel with backend URL configured

---

## 🏗️ Architecture Overview

```
GSMT Ver 6.0/
├── api/                     # FastAPI Backend
│   ├── main.py             # Complete API implementation
│   ├── requirements.txt    # Python dependencies
│   ├── Dockerfile         # Container configuration  
│   └── start.py           # Development server
├── js/
│   └── app.js             # Modern SPA application
├── index.html             # Main frontend entry
├── netlify.toml          # Frontend deployment config
├── railway.json          # Backend deployment config
└── DEPLOYMENT_V6.md      # This guide
```

---

## 🔧 Environment Configuration

### Backend Environment Variables
```bash
# Required
PORT=8000                    # Server port
HOST=0.0.0.0                # Server host

# Optional
LOG_LEVEL=info              # Logging level
CORS_ORIGINS=*              # CORS allowed origins
```

### Frontend Configuration
The application automatically detects:
- **Development**: Uses `http://localhost:8000`
- **Production**: Uses same domain or configured API URL

Manual configuration via Settings modal:
- Click gear icon in top right
- Set custom API Base URL
- Configure auto-refresh settings

---

## 📊 Features Verification

After deployment, verify these key features:

### ✅ Core Functionality
- [ ] API health check at `/health`
- [ ] Symbol search and selection
- [ ] Percentage-based analysis
- [ ] Multiple time periods (24hr - 2Y)
- [ ] Unified overlay charts
- [ ] Performance summary cards

### ✅ Advanced Features  
- [ ] Real-time data updates
- [ ] Auto-refresh capability
- [ ] Chart type switching (Percentage/Price/Candlestick)
- [ ] Fullscreen chart mode
- [ ] Toast notifications
- [ ] Responsive design

### ✅ Data Sources
- [ ] YFinance integration
- [ ] Demo data fallback
- [ ] Multi-source error handling
- [ ] Australian (.AX) stock support
- [ ] Global indices support

---

## 🛠️ Development Setup

### Backend Development
```bash
cd api/
pip install -r requirements.txt
python start.py
# API available at http://localhost:8000
# Documentation at http://localhost:8000/docs
```

### Frontend Development
```bash
# Serve frontend locally
python -m http.server 8080
# Frontend available at http://localhost:8080
```

### Local Full-Stack Testing
1. Start backend: `cd api/ && python start.py`
2. Start frontend: `python -m http.server 8080`
3. Access app at `http://localhost:8080`

---

## 🔍 Troubleshooting

### Common Issues

#### Frontend can't connect to API
- **Symptoms**: Red API status, "Connection failed"
- **Solutions**:
  1. Check API URL in settings
  2. Verify backend is running
  3. Check CORS configuration
  4. Test API health: `curl http://your-api/health`

#### No data loading
- **Symptoms**: Empty charts, "No data" messages
- **Solutions**:
  1. Verify selected symbols are supported
  2. Check API logs for errors
  3. Try different time periods
  4. Enable demo mode for testing

#### Charts not rendering
- **Symptoms**: Blank chart area
- **Solutions**:
  1. Check browser console for errors
  2. Verify ECharts library loaded
  3. Test with different chart types
  4. Try fullscreen mode

### Performance Optimization

#### Backend Performance
- Enable response caching with Redis
- Use connection pooling
- Implement rate limiting
- Monitor memory usage

#### Frontend Performance  
- Enable service worker caching
- Optimize chart rendering
- Implement virtual scrolling for large datasets
- Use CDN for static assets

---

## 📈 Monitoring & Analytics

### API Monitoring
- Health endpoint: `/health`
- Metrics endpoint: `/metrics` (if implemented)
- Log aggregation via platform tools

### Frontend Analytics
- User interaction tracking
- Performance monitoring
- Error reporting
- Usage analytics

---

## 🔐 Security Considerations

### API Security
- CORS properly configured
- Input validation with Pydantic
- Rate limiting implemented
- Error handling prevents data leaks

### Frontend Security
- CSP headers configured
- XSS protection enabled
- Secure API communication
- No sensitive data in localStorage

---

## 📚 API Documentation

Once deployed, comprehensive API documentation is available at:
- **Interactive Docs**: `https://your-api-domain/docs`
- **ReDoc**: `https://your-api-domain/redoc`

### Key Endpoints
- `GET /health` - System health check
- `GET /api/symbols` - Available symbols
- `GET /api/stock/{symbol}` - Individual stock data
- `POST /api/bulk` - Bulk symbol analysis
- `GET /api/search/{query}` - Symbol search

---

## 🎯 Next Steps

After successful deployment:

1. **Test all features** with real market data
2. **Configure monitoring** and alerting
3. **Set up CI/CD** for automated deployments
4. **Implement advanced features**:
   - User accounts and portfolios
   - Custom alerts and notifications  
   - Advanced technical indicators
   - Export functionality

---

## 💡 Support

For deployment issues or questions:
- Check the troubleshooting section above
- Review API documentation
- Test with demo data mode
- Verify all dependencies are installed

**GSMT Ver 6.0** represents a complete architectural rebuild focused on modern web standards, percentage-based analysis, and professional deployment workflows.