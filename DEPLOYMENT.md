# 🚀 GSMT Ver 6.0 - Complete Deployment Guide

## 📋 **Deployment Overview**

GSMT Ver 6.0 uses a **dual deployment strategy** with enhanced percentage analysis:
- **Backend (Python Flask)** → Railway (handles percentage calculations)
- **Frontend (HTML/CSS/JS)** → Netlify (displays normalized charts)

## 🎯 **Pre-Deployment Checklist**

### **✅ Files Ready for Deployment**
- [x] `app.py` - Enhanced backend with percentage change calculations
- [x] `requirements.txt` - Python dependencies
- [x] `Procfile` - Railway configuration
- [x] `runtime.txt` - Python version specification (3.11.6)
- [x] `index.html` - Percentage analysis frontend
- [x] `index-standalone.html` - Self-contained demo version
- [x] `.gitignore` - Git ignore rules
- [x] `README.md` - Complete documentation

### **✅ Prerequisites**
- [x] GitHub account
- [x] Railway account (free)
- [x] Netlify account (free)
- [x] Optional: Free API keys for enhanced reliability

### **✅ New Features in Ver 6.0**
- [x] Percentage change Y-axis (normalized comparison)
- [x] Unified overlay charts (mix indices with stocks)
- [x] Flexible time periods (24 hours to 2 years)
- [x] Enhanced time interval optimization
- [x] Mixed asset comparison capabilities

## 🗂️ **Step 1: GitHub Repository Setup**

### **Create New Repository**
1. **Go to GitHub**: https://github.com/new
2. **Repository Details**:
   - **Name**: `GSMT-Ver-6.0`
   - **Description**: `Global Stock Market Tracker v6.0 - Percentage Analysis & Unified Overlay Charts`
   - **Visibility**: Public (recommended for free deployments)
   - **Initialize**: ✅ Add README file

### **Upload Project Files**
```bash
# Clone your new repository
git clone https://github.com/yourusername/GSMT-Ver-6.0.git
cd GSMT-Ver-6.0

# Copy all project files to repository folder
# Then add and commit
git add .
git commit -m "GSMT Ver 6.0 - Percentage Analysis & Unified Overlay Charts

Features:
- Percentage change Y-axis for normalized comparison
- Unified overlay charts mixing indices and stocks
- Flexible time periods from 24 hours to 2 years
- Enhanced 5-step multi-source fallback
- Individual stock support (US & Australian)
- Optimized intervals per time period"

git push origin main
```

## 🚂 **Step 2: Railway Backend Deployment**

### **Deploy to Railway**
1. **Go to Railway**: https://railway.app
2. **Sign Up/Login**: Use GitHub account for easy integration
3. **Create New Project**: Click "New Project"
4. **Deploy from GitHub**: Select "Deploy from GitHub repo"
5. **Select Repository**: Choose your `GSMT-Ver-6.0` repository
6. **Auto-Detection**: Railway automatically detects Python app

### **Railway Configuration** (Automatic)
Railway will automatically use these files:
- **`requirements.txt`**: Install Python dependencies
- **`Procfile`**: Configure Gunicorn server
- **`runtime.txt`**: Use Python 3.11.6
- **`app.py`**: Main Flask application

### **Deployment Process**
```bash
# Railway will run these commands automatically:
pip install -r requirements.txt
gunicorn app:app --bind 0.0.0.0:$PORT --workers 3 --timeout 60
```

### **Get Railway URL**
After deployment (3-5 minutes):
1. **Go to Railway Dashboard**
2. **Select Your Project**
3. **Click on Service**
4. **Copy the generated URL**: `https://gsmt-production-abc123.up.railway.app`

### **Test Backend**
Test your Railway deployment:
```bash
# Health check
curl https://your-railway-url.railway.app/api/health

# Should return JSON with status: "healthy" and features list
```

### **Optional: Add API Keys**
For enhanced reliability, add environment variables in Railway:
1. **Go to Variables tab**
2. **Add**:
   - `ALPHA_VANTAGE_API_KEY`: Your Alpha Vantage key
   - `TWELVE_DATA_API_KEY`: Your Twelve Data key

## 🌐 **Step 3: Frontend Configuration**

### **Update API URL**
**CRITICAL**: Update the API URL in your frontend:

1. **Edit `index.html`**
2. **Find line 283** (approximately):
```javascript
const API_BASE = window.location.hostname === 'localhost' ? 
  'http://localhost:5000' : 
  'https://your-railway-url.railway.app';  // <-- CHANGE THIS
```

3. **Replace with your actual Railway URL**:
```javascript
const API_BASE = window.location.hostname === 'localhost' ? 
  'http://localhost:5000' : 
  'https://gsmt-production-abc123.up.railway.app';  // <-- YOUR ACTUAL URL
```

4. **Save and commit**:
```bash
git add index.html
git commit -m "Update API URL to Railway deployment"
git push origin main
```

## 🎯 **Step 4: Netlify Frontend Deployment**

### **Option A: Drag & Drop (Easiest)**
1. **Go to Netlify**: https://netlify.com
2. **Sign Up/Login**
3. **Drag & Drop**: Drag your updated `index.html` to Netlify deploy area
4. **Instant Deploy**: Site will be live immediately
5. **Custom Domain**: Set up custom domain if desired

### **Option B: GitHub Integration**
1. **Go to Netlify**: https://netlify.com
2. **New Site from Git**
3. **Connect to GitHub**
4. **Select Repository**: `GSMT-Ver-6.0`
5. **Build Settings**: 
   - **Build Command**: Leave empty (static site)
   - **Publish Directory**: `.` (root directory)
6. **Deploy**: Automatic deployment on git push

### **Get Netlify URL**
After deployment:
1. **Copy the generated URL**: `https://precious-stroopwafel-369123.netlify.app`
2. **Optional**: Set up custom domain

## ✅ **Step 5: Test Full Deployment**

### **Frontend Testing**
1. **Open your Netlify URL**
2. **Check Elements**:
   - ✅ Page loads without errors
   - ✅ "5-Source Fallback Active" status
   - ✅ All symbol categories populated
   - ✅ Time period selector shows 24hr-2Y options
   - ✅ Default selection shows percentage charts

### **Backend Integration Testing**
1. **Select Mixed Symbols**: Choose indices and stocks
2. **Test Time Periods**: Switch between 24hr, 1 week, 1 month
3. **Verify Percentage Mode**: Y-axis shows percentage change
4. **Check Overlay**: Multiple symbols on same normalized scale
5. **Test Data Sources**: Monitor success rate indicator

### **Feature Verification**
- ✅ **Percentage Y-Axis**: All charts show % change from start of period
- ✅ **Unified Overlay**: Mix ^GSPC, AAPL, CBA.AX on same chart
- ✅ **Time Periods**: 24hr (default) up to 2 years work correctly
- ✅ **Narrower Candlesticks**: 50% width, 8px max for single symbol view
- ✅ **Custom Symbols**: Add and validate new symbols
- ✅ **Mobile Responsive**: Works on phone/tablet screens

## 🐛 **Troubleshooting**

### **"Failed to initialize" Error**
**Cause**: Frontend can't connect to backend
**Solution**: 
1. Verify Railway URL is correct in `index.html` line 283
2. Check Railway deployment status
3. Test backend health endpoint directly

### **No Percentage Data**
**Cause**: Backend percentage calculation issues
**Solution**:
1. Check Railway logs for errors
2. Verify API keys if using enhanced sources
3. Test with different time periods

### **Charts Not Loading**
**Cause**: ECharts library or data format issues
**Solution**:
1. Check browser console for JavaScript errors
2. Verify symbols are selected
3. Try different time periods
4. Check network tab for API responses

### **Slow Performance**
**Cause**: API rate limits or large datasets
**Solution**:
1. Add free API keys for Alpha Vantage and Twelve Data
2. Reduce number of selected symbols
3. Use shorter time periods for testing

## 🚀 **Advanced Configuration**

### **Custom Domain Setup**
#### **Netlify Custom Domain**
1. Go to Site Settings → Domain Management
2. Add custom domain
3. Configure DNS settings
4. Enable HTTPS (automatic)

#### **Railway Custom Domain**
1. Go to Project Settings → Domains
2. Add custom domain
3. Configure DNS CNAME record

### **Environment Variables**
#### **Railway Environment Variables**
```bash
ALPHA_VANTAGE_API_KEY=your_key_here
TWELVE_DATA_API_KEY=your_key_here
FLASK_ENV=production
```

### **Performance Optimization**
#### **Backend Optimization**
- Enable API keys for higher rate limits
- Configure Redis caching (Railway addon)
- Set up monitoring (Railway built-in)

#### **Frontend Optimization**
- Enable Netlify CDN (automatic)
- Configure caching headers
- Set up form handling if needed

## 📊 **Monitoring & Analytics**

### **Railway Monitoring**
- **Metrics**: CPU, Memory, Network usage
- **Logs**: Real-time application logs
- **Health Checks**: Automatic uptime monitoring

### **Netlify Analytics**
- **Traffic**: Page views, unique visitors
- **Performance**: Load times, Core Web Vitals
- **Forms**: Contact form submissions (if added)

### **Application Monitoring**
- **Success Rate**: Built-in API success rate tracking
- **Data Sources**: Monitor which APIs are being used
- **Performance**: Chart rendering times and responsiveness

## 🎯 **Success Metrics**

After successful deployment, you should see:
- **✅ 99%+ Success Rate**: Multi-source fallback working
- **✅ Sub-3s Load Times**: Fast initial data loading
- **✅ Percentage Normalization**: All symbols comparable on same scale
- **✅ Mixed Asset Support**: Indices and stocks overlaid properly
- **✅ Flexible Time Periods**: 24hr to 2Y working smoothly
- **✅ Mobile Responsive**: Works on all device sizes

## 📞 **Support & Next Steps**

### **If You Need Help**
1. **Check Railway Logs**: Look for backend errors
2. **Check Browser Console**: Look for frontend errors
3. **Test API Endpoints**: Verify backend is responding
4. **GitHub Issues**: Create issue with error details

### **Enhancement Opportunities**
- Add more data sources to fallback chain
- Implement user portfolios and watchlists
- Add technical indicators and alerts
- Create mobile app using same backend API

---

**🎉 Congratulations!** Your GSMT Ver 6.0 with percentage analysis and unified overlay charts is now live and ready for comprehensive market analysis!