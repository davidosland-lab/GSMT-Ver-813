# GSMT Global 24H Timeline - Refined Implementation

## 🌍 Core Concept

A **24-hour continuous x-axis timeline** showing four core global market indices, each appearing only during their Sydney timezone trading windows. This creates a seamless global market flow visualization with Sydney time as the reference point.

## 📊 Timeline Structure

### **24-Hour X-Axis (Sydney Time)**
```
00:00 ────────────────────────────────── 24:00
  │                                        │
  └── Starting reference: 10:00 Sydney ───┘

Market Windows:
├─ 09:00-15:00: Nikkei 225 (Japan) ──────── 6 hours
├─ 10:00-16:00: ASX 200 (Australia) ────── 6 hours  
├─ 18:00-00:00: FTSE 100 (UK) ──────────── 6 hours
└─ 00:30-07:30: S&P 500 (US) ───────────── 7 hours
```

## 🎯 Four Core Indices Only

| Symbol | Index | Market | Sydney Window | Duration | Color |
|--------|-------|--------|---------------|----------|-------|
| `^N225` | Nikkei 225 | Japan | 09:00-15:00 | 6h | Blue |
| `^AXJO` | ASX 200 | Australia | 10:00-16:00 | 6h | Green |
| `^FTSE` | FTSE 100 | UK | 18:00-00:00 | 6h | Amber |
| `^GSPC` | S&P 500 | US | 00:30-07:30 | 7h | Red |

## ⚡ Key Features

### **Continuous 24-Hour Timeline**
- X-axis shows full 24-hour period starting from 10am Sydney
- Markets appear only during their trading windows
- **Intentional gaps** between trading sessions (no overnight data)
- Timeline repeats daily with same market sequence

### **Sydney Timezone Reference**  
- All times displayed in Sydney timezone (AEST/AEDT)
- Automatic daylight saving time handling
- 10am Sydney as daily reference point (project requirement)
- Global market windows converted to Sydney time equivalents

### **Simplified Display**
- **Percentage change only** (no candlesticks)
- **Four indices maximum** (core global markets)
- **Live data during trading hours**
- **Market window shading** for visual clarity

## 🚀 Implementation Files

### **Backend**
- `backend/timezone_handler.py` - Sydney timezone calculations
- `backend/market_sessions.py` - Global market window manager
- `backend/app.py` - Enhanced API with Sydney support
- `timeline-server.js` - Simplified Node.js server for testing

### **Frontend**
- `frontend/js/refined-app.js` - Simplified app for global timeline
- `frontend/js/global-timeline.js` - Timeline management utilities
- `frontend/index-refined.html` - Clean UI focused on 24H concept
- `frontend/js/timezone-utils.js` - Sydney timezone utilities

### **API Endpoints**
- `GET /global-timeline` - 24-hour timeline data for four indices
- `GET /health` - Server status with Sydney timezone info
- `GET /symbols` - Four core indices information
- `POST /analyze` - Redirects to global timeline (simplified)

## 🕒 Trading Windows in Sydney Time

### **Market Sequence Flow**
1. **00:30-07:30**: S&P 500 (US) - Night/Early morning Sydney
2. **09:00-15:00**: Nikkei 225 (Japan) - Morning/Afternoon Sydney  
3. **10:00-16:00**: ASX 200 (Australia) - Business hours Sydney
4. **18:00-00:00**: FTSE 100 (UK) - Evening Sydney

### **Visual Timeline**
```
Sydney Time:  00  03  06  09  12  15  18  21  00
             ├──┼──┼──┼──┼──┼──┼──┼──┼──┤
S&P 500:     ███████                     
Nikkei:            ████████████
ASX 200:              ████████████
FTSE:                              ████████
             └── 24-hour continuous timeline ──┘
```

## 📈 Data Characteristics

### **Real Market Windows**
- **Japan (Nikkei)**: Typically 9am-3pm JST = 9am-3pm Sydney (similar timezone)
- **Australia (ASX)**: 10am-4pm AEST = 10am-4pm Sydney (local market)
- **UK (FTSE)**: 8am-4pm GMT = 6pm-2am Sydney = 18:00-00:00 simplified
- **US (S&P)**: 9:30am-4pm EST = 12:30am-8am Sydney = 00:30-07:30 simplified

### **Chart Display**
- **Continuous x-axis**: 24 hours starting 10am Sydney
- **Data gaps**: Intentional - no data between trading sessions
- **Market overlaps**: Visual highlights when multiple markets trade
- **Live updates**: 15-minute intervals during active trading

## 🔧 Technical Implementation

### **Data Generation**
```javascript
// Market windows in Sydney time
const windows = {
  '^N225': { start: 9, end: 15 },     // 09:00-15:00
  '^AXJO': { start: 10, end: 16 },    // 10:00-16:00
  '^FTSE': { start: 18, end: 24 },    // 18:00-00:00
  '^GSPC': { start: 0.5, end: 7.5 }   // 00:30-07:30
};
```

### **Chart Configuration**
```javascript
xAxis: {
  type: 'time',
  min: sydney10am.getTime(),
  max: sydney10am.getTime() + (24 * 60 * 60 * 1000) // Full 24h
},
connectNulls: false // Show gaps between sessions
```

## 📋 User Experience

### **What Users See**
1. **24-hour timeline** starting from 10am Sydney (reference)
2. **Four colored lines** appearing during their respective trading windows  
3. **Market window shading** showing active trading periods
4. **Gaps between sessions** - no artificial connections
5. **Live Sydney time** in header with current market status

### **Interactive Features**
- **Load Timeline** button refreshes all four indices
- **Market status cards** show current trading state
- **Auto-refresh** during any active trading session
- **Fullscreen** chart mode for detailed analysis

## ✅ Final Status

**Implementation Complete**: The code now displays exactly what was requested:
- ✅ 24-hour x-axis timeline in Sydney timezone
- ✅ Markets appear only during 6-7 hour trading windows  
- ✅ Australian market at 10:00hrs Sydney time
- ✅ Nikkei starts at 09:00hrs Sydney time
- ✅ FTSE opens at 18:00hrs Sydney time  
- ✅ American market: 00:30hrs-07:30hrs Sydney time
- ✅ Four indices only (no other symbols)
- ✅ Candlesticks removed (percentage only)
- ✅ 10am Sydney as fundamental reference point

**Ready for production deployment** with the refined global 24-hour timeline concept.