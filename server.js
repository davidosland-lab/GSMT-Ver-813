/**
 * GSMT Ver 7.0 Sydney Edition - Test Server
 * Simple Express server for testing Sydney timezone integration
 */

const express = require('express');
const cors = require('cors');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;

// Enable CORS and JSON parsing
app.use(cors());
app.use(express.json());
app.use(express.static('frontend'));
app.use(express.static('.'));

// Sydney timezone utilities (Node.js version)
class SydneyTimezoneHandler {
    constructor() {
        this.sydneyTimezone = 'Australia/Sydney';
    }
    
    getSydneyNow() {
        return new Date(new Date().toLocaleString("en-US", {timeZone: this.sydneyTimezone}));
    }
    
    getSydney10amStart(referenceDate = null) {
        const sydneyDate = referenceDate ? 
            new Date(referenceDate.toLocaleString("en-US", {timeZone: this.sydneyTimezone})) : 
            this.getSydneyNow();
        
        // Set to 10:00 AM Sydney time
        const start = new Date(sydneyDate);
        start.setHours(10, 0, 0, 0);
        
        return start;
    }
    
    get24HourPeriod() {
        const start = this.getSydney10amStart();
        const end = new Date(start);
        end.setHours(start.getHours() + 24);
        
        return { start, end };
    }
    
    formatSydneyTime(date, showTimezone = true) {
        const sydneyTime = new Date(date.toLocaleString("en-US", {timeZone: this.sydneyTimezone}));
        const formatted = sydneyTime.toLocaleString('en-AU', {
            timeZone: this.sydneyTimezone,
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
            timeZoneName: showTimezone ? 'short' : undefined
        });
        return formatted;
    }
    
    getMarketStatus() {
        const sydneyNow = this.getSydneyNow();
        const marketHours = {
            'Australia': { open: 10, close: 16, active: false },
            'Japan': { open: 9, close: 15, active: false },
            'Hong Kong': { open: 9, close: 16, active: false },
            'UK': { open: 8, close: 16, active: false },
            'US': { open: 9, close: 16, active: false }
        };
        
        // Simulate market status (simplified)
        const sydneyHour = sydneyNow.getHours();
        marketHours['Australia'].active = sydneyHour >= 10 && sydneyHour <= 16;
        
        return marketHours;
    }
}

const sydneyHandler = new SydneyTimezoneHandler();

// Mock data generator for trading hours only
function generateMockMarketData(symbol, tradingHoursOnly = true) {
    const data = [];
    const basePrice = Math.random() * 20000 + 5000;
    let currentPrice = basePrice;
    
    // Get market for symbol
    const market = getMarketForSymbol(symbol);
    
    // Trading hours for each market (in their local time)
    const tradingSchedule = {
        'Australia': { start: 10, duration: 6, timezone: 'Australia/Sydney' },    // 10am-4pm AEST
        'Japan': { start: 9, duration: 6, timezone: 'Asia/Tokyo' },               // 9am-3pm JST
        'Hong Kong': { start: 9, duration: 7, timezone: 'Asia/Hong_Kong' },       // 9am-4pm HKT
        'China': { start: 9, duration: 6, timezone: 'Asia/Shanghai' },            // 9am-3pm CST
        'UK': { start: 8, duration: 8, timezone: 'Europe/London' },               // 8am-4pm GMT
        'Germany': { start: 9, duration: 8, timezone: 'Europe/Berlin' },          // 9am-5pm CET
        'France': { start: 9, duration: 8, timezone: 'Europe/Paris' },            // 9am-5pm CET
        'US': { start: 9, duration: 7, timezone: 'America/New_York' }             // 9:30am-4pm EST (simplified)
    };
    
    const schedule = tradingSchedule[market] || tradingSchedule['Australia'];
    
    if (tradingHoursOnly) {
        // Generate data for trading hours only
        const today = new Date();
        
        // Create trading session start time in market's timezone
        let sessionStart;
        try {
            const marketToday = new Date(today.toLocaleString("en-US", {timeZone: schedule.timezone}));
            sessionStart = new Date(marketToday);
            sessionStart.setHours(schedule.start, 0, 0, 0);
        } catch (error) {
            // Fallback to Sydney time
            sessionStart = new Date(today);
            sessionStart.setHours(schedule.start, 0, 0, 0);
        }
        
        // Generate minute-by-minute data for the trading session
        const intervalMinutes = 15; // 15-minute intervals
        const totalIntervals = (schedule.duration * 60) / intervalMinutes;
        
        for (let interval = 0; interval < totalIntervals; interval++) {
            const timestamp = new Date(sessionStart);
            timestamp.setMinutes(sessionStart.getMinutes() + (interval * intervalMinutes));
            
            // More realistic price movement during trading hours
            const volatility = 0.008; // 0.8% volatility per 15-min interval
            const change = (Math.random() - 0.5) * volatility;
            currentPrice *= (1 + change);
            
            // Ensure price doesn't drift too far from base
            const maxDrift = 0.15; // 15% max drift from base price
            if (Math.abs((currentPrice - basePrice) / basePrice) > maxDrift) {
                currentPrice = basePrice * (1 + (Math.random() - 0.5) * maxDrift);
            }
            
            const percentageChange = ((currentPrice - basePrice) / basePrice) * 100;
            
            // Higher volume at market open/close
            let volumeMultiplier = 1;
            if (interval < 4 || interval > totalIntervals - 4) { // First/last hour
                volumeMultiplier = 2;
            }
            
            data.push({
                timestamp: timestamp.toISOString(),
                timestamp_ms: timestamp.getTime(),
                open: Math.round(currentPrice * 0.998 * 100) / 100,
                high: Math.round(currentPrice * 1.005 * 100) / 100,
                low: Math.round(currentPrice * 0.995 * 100) / 100,
                close: Math.round(currentPrice * 100) / 100,
                volume: Math.floor(Math.random() * 5000000 * volumeMultiplier),
                percentage_change: Math.round(percentageChange * 100) / 100,
                market: market,
                trading_session: true
            });
        }
    } else {
        // Original 24-hour generation (fallback)
        const period = sydneyHandler.get24HourPeriod();
        
        for (let hour = 0; hour < 24; hour++) {
            const timestamp = new Date(period.start);
            timestamp.setHours(period.start.getHours() + hour);
            
            const change = (Math.random() - 0.5) * 0.03;
            currentPrice *= (1 + change);
            
            const percentageChange = ((currentPrice - basePrice) / basePrice) * 100;
            
            data.push({
                timestamp: timestamp.toISOString(),
                timestamp_ms: timestamp.getTime(),
                open: Math.round(currentPrice * 0.99 * 100) / 100,
                high: Math.round(currentPrice * 1.02 * 100) / 100,
                low: Math.round(currentPrice * 0.98 * 100) / 100,
                close: Math.round(currentPrice * 100) / 100,
                volume: Math.floor(Math.random() * 10000000),
                percentage_change: Math.round(percentageChange * 100) / 100,
                market: market,
                trading_session: false
            });
        }
    }
    
    return data;
}

// Helper function to get market for symbol
function getMarketForSymbol(symbol) {
    if (symbol === '^AXJO') return 'Australia';
    if (symbol === '^N225') return 'Japan';
    if (symbol === '^HSI') return 'Hong Kong';
    if (symbol === '^FTSE') return 'UK';
    if (symbol === '^GSPC' || symbol === '^IXIC') return 'US';
    if (symbol.includes('.AX')) return 'Australia';
    return 'US'; // Default
}

// API Routes

// Health check with Sydney timezone info
app.get('/health', (req, res) => {
    const sydneyNow = sydneyHandler.getSydneyNow();
    
    res.json({
        status: 'healthy',
        version: '7.0.0',
        timestamp: new Date().toISOString(),
        sydney_time: sydneyHandler.formatSydneyTime(sydneyNow),
        service: 'GSMT Ver 7.0 API - Sydney Edition (Test Server)',
        environment: 'development',
        features: [
            'Sydney timezone integration',
            '24-hour periods from 10am AEST/AEDT',
            'Global market sessions tracking',
            'Live data simulation',
            'Express.js test server'
        ],
        supported_symbols: 8,
        active_markets: sydneyHandler.getMarketStatus(),
        sydney_market_open: sydneyHandler.getMarketStatus()['Australia'].active
    });
});

// Symbols database
app.get('/symbols', (req, res) => {
    const symbols = {
        'Australian Indices': [
            { symbol: '^AXJO', name: 'ASX 200', market: 'Australia', category: 'Index' }
        ],
        'Asian Indices': [
            { symbol: '^N225', name: 'Nikkei 225', market: 'Japan', category: 'Index' },
            { symbol: '^HSI', name: 'Hang Seng', market: 'Hong Kong', category: 'Index' }
        ],
        'European Indices': [
            { symbol: '^FTSE', name: 'FTSE 100', market: 'UK', category: 'Index' }
        ],
        'US Indices': [
            { symbol: '^GSPC', name: 'S&P 500', market: 'US', category: 'Index' },
            { symbol: '^IXIC', name: 'NASDAQ', market: 'US', category: 'Index' }
        ]
    };
    
    res.json({
        total_symbols: 6,
        categories: symbols,
        supported_periods: ['24h', '3d', '1w', '1M'],
        chart_types: ['percentage', 'price', 'candlestick']
    });
});

// Search symbols
app.get('/search/:query', (req, res) => {
    const query = req.params.query.toLowerCase();
    const allSymbols = [
        { symbol: '^AXJO', name: 'ASX 200', market: 'Australia', category: 'Index' },
        { symbol: '^N225', name: 'Nikkei 225', market: 'Japan', category: 'Index' },
        { symbol: '^HSI', name: 'Hang Seng', market: 'Hong Kong', category: 'Index' },
        { symbol: '^FTSE', name: 'FTSE 100', market: 'UK', category: 'Index' },
        { symbol: '^GSPC', name: 'S&P 500', market: 'US', category: 'Index' },
        { symbol: '^IXIC', name: 'NASDAQ', market: 'US', category: 'Index' }
    ];
    
    const results = allSymbols.filter(s => 
        s.symbol.toLowerCase().includes(query) ||
        s.name.toLowerCase().includes(query) ||
        s.market.toLowerCase().includes(query)
    );
    
    res.json({
        query: req.params.query,
        results: results.slice(0, 8),
        total_found: results.length
    });
});

// Analysis endpoint with trading hours focus
app.post('/analyze', (req, res) => {
    const { symbols, period, chart_type, sydney_start = true, reference_time } = req.body;
    
    if (!symbols || symbols.length === 0) {
        return res.status(400).json({ error: 'No symbols provided' });
    }
    
    const sydneyNow = sydneyHandler.getSydneyNow();
    
    const data = {};
    const metadata = {};
    const marketSessions = [];
    
    symbols.forEach(symbol => {
        // Generate trading hours data only
        data[symbol] = generateMockMarketData(symbol, true);
        
        const market = getMarketForSymbol(symbol);
        metadata[symbol] = {
            symbol,
            name: symbol === '^AXJO' ? 'ASX 200' : 
                  symbol === '^N225' ? 'Nikkei 225' :
                  symbol === '^FTSE' ? 'FTSE 100' : 
                  symbol === '^GSPC' ? 'S&P 500' : symbol,
            market: market,
            category: 'Index'
        };
        
        // Add market session info
        if (!marketSessions.find(s => s.market === market)) {
            const tradingHours = {
                'Australia': { start: 10, duration: 6 },
                'Japan': { start: 9, duration: 6 },
                'UK': { start: 8, duration: 8 },
                'US': { start: 9, duration: 7 }
            };
            
            const hours = tradingHours[market] || { start: 10, duration: 6 };
            const sessionStart = new Date();
            sessionStart.setHours(hours.start, 0, 0, 0);
            const sessionEnd = new Date(sessionStart);
            sessionEnd.setHours(sessionStart.getHours() + hours.duration);
            
            marketSessions.push({
                market,
                display_name: market === 'Australia' ? '🇦🇺 Sydney' :
                             market === 'Japan' ? '🇯🇵 Tokyo' :
                             market === 'UK' ? '🇬🇧 London' :
                             market === 'US' ? '🇺🇸 New York' : market,
                trading_hours: `${hours.duration}h session`,
                session_start: sessionStart.toISOString(),
                session_end: sessionEnd.toISOString(),
                is_active: sydneyHandler.getMarketStatus()[market]?.active || false,
                color: market === 'Australia' ? '#10b981' : 
                       market === 'Japan' ? '#3b82f6' :
                       market === 'UK' ? '#f59e0b' : '#ef4444'
            });
        }
    });
    
    res.json({
        success: true,
        data,
        metadata,
        period,
        chart_type,
        timestamp: new Date().toISOString(),
        sydney_timestamp: sydneyHandler.formatSydneyTime(sydneyNow),
        total_symbols: symbols.length,
        successful_symbols: symbols.length,
        market_sessions: marketSessions,
        display_mode: 'trading_hours_only',
        note: 'Chart displays trading hours only (6-8h per market)'
    });
});

// Sydney markets endpoint with trading hours focus
app.get('/sydney-markets', (req, res) => {
    const defaultSymbols = ['^AXJO', '^N225', '^FTSE', '^GSPC'];
    const sydneyNow = sydneyHandler.getSydneyNow();
    
    const data = {};
    const marketSessions = [];
    
    defaultSymbols.forEach(symbol => {
        // Generate trading hours data only
        data[symbol] = generateMockMarketData(symbol, true);
        
        const market = getMarketForSymbol(symbol);
        
        // Add unique market sessions
        if (!marketSessions.find(s => s.market === market)) {
            const tradingHours = {
                'Australia': { start: 10, duration: 6, timezone: 'Australia/Sydney' },
                'Japan': { start: 9, duration: 6, timezone: 'Asia/Tokyo' },
                'UK': { start: 8, duration: 8, timezone: 'Europe/London' },
                'US': { start: 9, duration: 7, timezone: 'America/New_York' }
            };
            
            const hours = tradingHours[market] || tradingHours['Australia'];
            
            marketSessions.push({
                market,
                display_name: market === 'Australia' ? '🇦🇺 Sydney (ASX)' :
                             market === 'Japan' ? '🇯🇵 Tokyo (Nikkei)' :
                             market === 'UK' ? '🇬🇧 London (FTSE)' :
                             market === 'US' ? '🇺🇸 New York (S&P)' : market,
                trading_hours: `${hours.duration}h session (${hours.start}:00-${hours.start + hours.duration}:00 local)`,
                duration_hours: hours.duration,
                local_start: `${hours.start}:00`,
                local_end: `${hours.start + hours.duration}:00`,
                timezone: hours.timezone,
                is_active: sydneyHandler.getMarketStatus()[market]?.active || false,
                color: market === 'Australia' ? '#10b981' : 
                       market === 'Japan' ? '#3b82f6' :
                       market === 'UK' ? '#f59e0b' : '#ef4444'
            });
        }
    });
    
    res.json({
        success: true,
        data,
        market_sessions: marketSessions,
        sydney_context: {
            current_sydney_time: sydneyHandler.formatSydneyTime(sydneyNow),
            sydney_market_status: sydneyHandler.getMarketStatus()['Australia'].active,
            market_day_phase: sydneyNow.getHours() >= 10 && sydneyNow.getHours() <= 16 ? 
                'Sydney market hours (10am-4pm)' : 'After Sydney market hours',
            display_note: 'Charts show trading hours only - no overnight periods'
        },
        refresh_schedule: {
            primary_refresh: 180,  // 3 minutes during trading
            secondary_refresh: 300,
            chart_refresh: 180
        },
        timestamp: sydneyHandler.formatSydneyTime(sydneyNow),
        display_mode: 'trading_hours_only'
    });
});

// Market sessions endpoint
app.get('/market-sessions', (req, res) => {
    const sydneyNow = sydneyHandler.getSydneyNow();
    const marketStatus = sydneyHandler.getMarketStatus();
    
    const sessions = Object.entries(marketStatus).map(([market, status]) => ({
        market,
        display_name: market === 'Australia' ? '🇦🇺 Sydney' : 
                     market === 'Japan' ? '🇯🇵 Tokyo' :
                     market === 'UK' ? '🇬🇧 London' : 
                     market === 'US' ? '🇺🇸 New York' : market,
        is_active: status.active,
        color: market === 'Australia' ? '#10b981' : '#3b82f6'
    }));
    
    res.json({
        success: true,
        sessions,
        sydney_time: sydneyHandler.formatSydneyTime(sydneyNow),
        active_markets: marketStatus
    });
});

// Sydney time info endpoint
app.get('/sydney-time', (req, res) => {
    const sydneyNow = sydneyHandler.getSydneyNow();
    const period = sydneyHandler.get24HourPeriod();
    
    res.json({
        success: true,
        sydney_time: sydneyHandler.formatSydneyTime(sydneyNow),
        sydney_timestamp_ms: sydneyNow.getTime(),
        timezone: 'Australia/Sydney',
        period_24h: {
            start: sydneyHandler.formatSydneyTime(period.start),
            end: sydneyHandler.formatSydneyTime(period.end),
            start_ms: period.start.getTime(),
            end_ms: period.end.getTime()
        },
        market_context: {
            sydney_market_open: sydneyHandler.getMarketStatus()['Australia'].active
        }
    });
});

// Serve the frontend
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'frontend', 'index.html'));
});

// Start server
app.listen(PORT, () => {
    const sydneyNow = sydneyHandler.getSydneyNow();
    console.log(`🚀 GSMT Ver 7.0 Sydney Edition Test Server running on port ${PORT}`);
    console.log(`🕒 Sydney time: ${sydneyHandler.formatSydneyTime(sydneyNow)}`);
    console.log(`📊 24-hour period: ${sydneyHandler.formatSydneyTime(sydneyHandler.get24HourPeriod().start)} to ${sydneyHandler.formatSydneyTime(sydneyHandler.get24HourPeriod().end)}`);
    console.log(`🌐 Frontend: http://localhost:${PORT}`);
    console.log(`⚡ API Health: http://localhost:${PORT}/health`);
});

module.exports = app;