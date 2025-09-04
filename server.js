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

// Mock data generator
function generateMockMarketData(symbol, hours = 24) {
    const data = [];
    const basePrice = Math.random() * 20000 + 5000;
    let currentPrice = basePrice;
    
    const period = sydneyHandler.get24HourPeriod();
    
    for (let hour = 0; hour < hours; hour++) {
        const timestamp = new Date(period.start);
        timestamp.setHours(period.start.getHours() + hour);
        
        const change = (Math.random() - 0.5) * 0.03; // 3% max change
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
            percentage_change: Math.round(percentageChange * 100) / 100
        });
    }
    
    return data;
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

// Analysis endpoint with Sydney timezone support
app.post('/analyze', (req, res) => {
    const { symbols, period, chart_type, sydney_start = true, reference_time } = req.body;
    
    if (!symbols || symbols.length === 0) {
        return res.status(400).json({ error: 'No symbols provided' });
    }
    
    const sydneyNow = sydneyHandler.getSydneyNow();
    const periodData = sydneyHandler.get24HourPeriod();
    
    const data = {};
    const metadata = {};
    
    symbols.forEach(symbol => {
        data[symbol] = generateMockMarketData(symbol, period === '24h' ? 24 : 48);
        metadata[symbol] = {
            symbol,
            name: symbol === '^AXJO' ? 'ASX 200' : 
                  symbol === '^N225' ? 'Nikkei 225' :
                  symbol === '^FTSE' ? 'FTSE 100' : 
                  symbol === '^GSPC' ? 'S&P 500' : symbol,
            market: symbol === '^AXJO' ? 'Australia' : 
                   symbol === '^N225' ? 'Japan' :
                   symbol === '^FTSE' ? 'UK' : 
                   symbol === '^GSPC' ? 'US' : 'Unknown',
            category: 'Index'
        };
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
        period_start: sydneyHandler.formatSydneyTime(periodData.start),
        period_end: sydneyHandler.formatSydneyTime(periodData.end),
        market_sessions: sydney_start ? {
            start_time: periodData.start,
            end_time: periodData.end,
            market_sessions: [
                {
                    market: 'Australia',
                    display_name: '🇦🇺 Sydney',
                    open_sydney: periodData.start,
                    close_sydney: new Date(periodData.start.getTime() + 6 * 60 * 60 * 1000),
                    is_active: sydneyHandler.getMarketStatus()['Australia'].active,
                    color: '#10b981'
                }
            ]
        } : null
    });
});

// Sydney markets endpoint
app.get('/sydney-markets', (req, res) => {
    const defaultSymbols = ['^AXJO', '^N225', '^FTSE', '^GSPC'];
    const sydneyNow = sydneyHandler.getSydneyNow();
    const period = sydneyHandler.get24HourPeriod();
    
    const data = {};
    defaultSymbols.forEach(symbol => {
        data[symbol] = generateMockMarketData(symbol);
    });
    
    res.json({
        success: true,
        data,
        market_sessions: [
            {
                market: 'Australia',
                display_name: '🇦🇺 Sydney',
                open_sydney: period.start,
                close_sydney: new Date(period.start.getTime() + 6 * 60 * 60 * 1000),
                is_active: sydneyHandler.getMarketStatus()['Australia'].active,
                color: '#10b981'
            }
        ],
        timeline: Array.from({length: 24}, (_, hour) => ({
            hour,
            sydneyTime: sydneyHandler.formatSydneyTime(new Date(period.start.getTime() + hour * 60 * 60 * 1000), false),
            activeMarkets: hour >= 10 && hour <= 16 ? ['Australia'] : []
        })),
        sydney_context: {
            current_sydney_time: sydneyHandler.formatSydneyTime(sydneyNow),
            sydney_market_status: sydneyHandler.getMarketStatus()['Australia'].active,
            market_day_phase: sydneyNow.getHours() >= 10 && sydneyNow.getHours() <= 16 ? 
                'Sydney market hours' : 'After hours'
        },
        refresh_schedule: {
            primary_refresh: 300,
            secondary_refresh: 600,
            chart_refresh: 300
        },
        timestamp: sydneyHandler.formatSydneyTime(sydneyNow),
        period_start: sydneyHandler.formatSydneyTime(period.start),
        period_end: sydneyHandler.formatSydneyTime(period.end)
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