# 🚀 **Netlify Deployment with Railway Backend URL**

## 📋 **Complete Step-by-Step Process**

### **Phase 1: Deploy Backend to Railway.app FIRST**

#### **Step 1: Deploy to Railway**
```bash
1. Go to railway.app
2. Click "New Project" → "Deploy from GitHub repo"
3. Select: davidosland-lab/GSMT-Ver-813
4. Branch: genspark_ai_developer
5. Railway auto-detects Procfile and deploys
```

#### **Step 2: Add Environment Variables in Railway**
```bash
Railway Dashboard → Your Project → Variables → Add:

ALPHA_VANTAGE_API_KEY=your_actual_key
TWELVE_DATA_API_KEY=your_actual_key  
FINNHUB_API_KEY=your_actual_key
USE_YAHOO_FINANCE=true
USE_ALPHA_VANTAGE=true
USE_TWELVE_DATA=true
LIVE_DATA_ENABLED=true
REQUIRE_LIVE_DATA=true
```

#### **Step 3: Get Your Railway URL**
```bash
# Railway will assign a URL like:
https://gsmt-production-abc123.railway.app

# Copy this URL - you'll need it for Step 4
```

### **Phase 2: Update Frontend Code with Railway URL**

#### **Step 4: Update Frontend API Detection**

**File to Edit**: `/frontend/assets/app.js`
**Line to Change**: Line 33

```javascript
// FIND THIS (around line 31-33):
if (currentHost.includes('netlify.app')) {
    // Use Railway backend URL for production
    return 'https://your-railway-backend-url.railway.app/api';  // ← CHANGE THIS LINE
}

// REPLACE WITH YOUR ACTUAL RAILWAY URL:
if (currentHost.includes('netlify.app')) {
    // Use Railway backend URL for production  
    return 'https://gsmt-production-abc123.railway.app/api';  // ← YOUR ACTUAL URL
}
```

#### **Step 5: Commit the URL Change**
```bash
git add frontend/assets/app.js
git commit -m "feat: update production API URL for Railway backend"
git push origin genspark_ai_developer
```

### **Phase 3: Deploy Frontend to Netlify**

#### **Method A: GitHub Integration (Recommended)**
```bash
1. Go to netlify.com
2. Click "New site from Git"
3. Choose "GitHub" 
4. Select repository: davidosland-lab/GSMT-Ver-813
5. Branch to deploy: genspark_ai_developer
6. Build command: (leave empty)
7. Publish directory: frontend
8. Click "Deploy site"
```

#### **Method B: Manual Upload**
```bash
1. Download/zip the /frontend folder from GitHub
2. Go to netlify.com → "Sites"
3. Drag the frontend folder to the deploy area
4. Netlify will automatically deploy
```

## 🎯 **Deployment Configuration in Netlify**

### **Where Settings Are Applied:**

#### **Automatic Configuration (via netlify.toml)**
The `netlify.toml` file automatically configures:
```toml
[build]
  publish = "."  # Deploy from frontend directory

[[redirects]]
  from = "/*"
  to = "/index.html"  # SPA routing
  status = 200

# Asset caching, optimization, etc.
```

#### **No Additional Netlify Settings Needed**
- ✅ **Build Settings**: Auto-detected from netlify.toml
- ✅ **Redirects**: Configured in _redirects and netlify.toml  
- ✅ **API Proxy**: Not needed - direct calls to Railway
- ✅ **Environment Variables**: Not needed for frontend

## 🔧 **Alternative: Environment Variables Method**

If you prefer using Netlify environment variables instead of hardcoding the URL:

#### **Option B: Use Netlify Environment Variables**

**Step 1: Update Frontend Code**
```javascript
// In frontend/assets/app.js, change to:
if (currentHost.includes('netlify.app')) {
    // Use environment variable if available
    const apiUrl = window.VITE_API_URL || 'https://default-railway-url.railway.app/api';
    return apiUrl;
}
```

**Step 2: Add Environment Variable in Netlify**
```bash
Netlify Dashboard → Site Settings → Environment Variables → Add:

Key: VITE_API_URL  
Value: https://your-railway-url.railway.app/api
```

## ✅ **Verification Steps**

### **Test Your Deployment:**
```bash
1. Visit your Netlify URL: https://your-app.netlify.app
2. Open browser DevTools → Network tab
3. Select a market and analyze
4. Verify API calls go to Railway URL
5. Check for CORS errors (should be none)
```

### **Expected API Calls:**
```bash
✅ GET https://your-railway-url.railway.app/api/symbols
✅ GET https://your-railway-url.railway.app/api/market-hours  
✅ POST https://your-railway-url.railway.app/api/analyze
```

## 🚨 **Common Issues & Solutions**

### **Issue**: CORS Errors
**Solution**: CORS is already configured in Railway backend to allow all origins

### **Issue**: API calls fail
**Solution**: Check Railway backend is deployed and environment variables are set

### **Issue**: Frontend shows but no data
**Solution**: Verify the Railway URL is correct in frontend/assets/app.js

## 📝 **Summary: Exact Location to Enter Railway URL**

**File**: `frontend/assets/app.js`  
**Line**: ~33  
**Function**: `detectApiUrl()`  
**Change**: Replace `'https://your-railway-backend-url.railway.app/api'` with your actual Railway URL

**Then**: Commit, push, and deploy to Netlify

---

## 🎯 **Quick Reference**

1. **Deploy Railway** → Get URL
2. **Update** `frontend/assets/app.js` line 33  
3. **Commit** and **push** changes
4. **Deploy Netlify** → Works automatically

The Railway URL goes in the **frontend code**, not in Netlify's dashboard settings!