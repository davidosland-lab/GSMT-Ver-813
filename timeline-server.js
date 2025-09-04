/**
 * GSMT Global 24H Timeline Server
 * Simple server for testing the refined 24-hour Sydney timeline concept
 */

const express = require('express');
const cors = require('cors');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());
app.use(express.static('frontend'));
app.use(express.static('.'));

// Core indices configuration with Sydney trading windows
const CORE_INDICES = {
    '^N225': {
        name: 'Nikkei 225',
        market: 'Japan',
        sydney_open: 9,      // 09:00 Sydney time
        sydney_close: 15,    // 15:00 Sydney time  
        duration: 6,
        color: '#3b82f6'
    },
    '^AXJO': {
        name: 'ASX 200', 
        market: 'Australia',
        sydney_open: 10,     // 10:00 Sydney time (project reference point)
        sydney_close: 16,    // 16:00 Sydney time
        duration: 6,
        color: '#10b981'
    },
    '^FTSE': {
        name: 'FTSE 100',
        market: 'UK',
        sydney_open: 18,     // 18:00 Sydney time
        sydney_close: 24,    // 00:00 Sydney time (midnight)
        duration: 6,
        color: '#f59e0b'
    },
    '^GSPC': {
        name: 'S&P 500',
        market: 'US', 
        sydney_open: 0.5,    // 00:30 Sydney time (next day)
        sydney_close: 7.5,   // 07:30 Sydney time
        duration: 7,
        color: '#ef4444'
    }
};

function getSydneyNow() {
    return new Date(new Date().toLocaleString("en-US", {timeZone: 'Australia/Sydney'}));
}

function getSydney10amReference() {
    const sydney = getSydneyNow();
    const reference = new Date(sydney);
    reference.setHours(10, 0, 0, 0);
    return reference;
}

function formatSydneyTime(date) {
    try {
        return date.toLocaleString('en-AU', {
            timeZone: 'Australia/Sydney',
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
            timeZoneName: 'short'
        });
    } catch (error) {
        return date.toLocaleString();
    }
}

function generateGlobalTimelineData(symbol) {
    const config = CORE_INDICES[symbol];
    if (!config) return [];
    
    const data = [];
    const basePrice = Math.random() * 15000 + 5000;
    let currentPrice = basePrice;
    
    // Sydney 10am reference point
    const sydney10am = getSydney10amReference();
    
    // Generate data for 24-hour timeline
    for (let hour = 0; hour < 24; hour++) {
        const timestamp = new Date(sydney10am);
        timestamp.setHours(sydney10am.getHours() + hour);
        
        // Calculate Sydney hour (0-24)
        const sydneyHour = (10 + hour) % 24;
        
        // Check if market is trading at this Sydney time
        let isTrading = false;
        if (config.sydney_close > config.sydney_open) {
            // Same day trading (Japan, Australia, UK)
            isTrading = sydneyHour >= config.sydney_open && sydneyHour <= config.sydney_close;
        } else {
            // Cross-midnight trading (US: 00:30-07:30)
            isTrading = sydneyHour >= config.sydney_open || sydneyHour <= config.sydney_close;
        }
        
        if (isTrading) {
            // Generate price movement during trading
            const volatility = 0.012; // 1.2% per hour
            const change = (Math.random() - 0.5) * volatility;
            currentPrice *= (1 + change);
            
            // Prevent excessive drift
            const maxDrift = 0.15;
            if (Math.abs((currentPrice - basePrice) / basePrice) > maxDrift) {
                currentPrice = basePrice * (1 + (Math.random() - 0.5) * maxDrift);
            }
            
            const percentageChange = ((currentPrice - basePrice) / basePrice) * 100;
            
            data.push({
                timestamp: timestamp.toISOString(),
                timestamp_ms: timestamp.getTime(),
                sydney_hour: sydneyHour,
                close: Math.round(currentPrice * 100) / 100,
                percentage_change: Math.round(percentageChange * 100) / 100,
                market: config.market,
                trading: true
            });
        }
        // Note: No data points during non-trading hours - chart will show gaps
    }
    
    return data;
}

// Health endpoint with global timeline info
app.get('/health', (req, res) => {
    const sydneyNow = getSydneyNow();
    
    res.json({
        status: 'healthy',
        version: '7.0-refined',
        timestamp: new Date().toISOString(),
        sydney_time: formatSydneyTime(sydneyNow),
        service: 'GSMT Global 24H Timeline API',
        core_indices: Object.keys(CORE_INDICES),
        concept: '24-hour Sydney timezone timeline',
        market_windows: {
            nikkei: 'Japan: 09:00-15:00 Sydney',
            asx: 'Australia: 10:00-16:00 Sydney (reference)',
            ftse: 'UK: 18:00-00:00 Sydney',
            sp500: 'US: 00:30-07:30 Sydney'
        }
    });
});

// Core indices info
app.get('/symbols', (req, res) => {
    res.json({
        core_indices: Object.entries(CORE_INDICES).map(([symbol, config]) => ({
            symbol,
            name: config.name,
            market: config.market,
            sydney_window: `${formatHour(config.sydney_open)}-${formatHour(config.sydney_close)}`,
            duration_hours: config.duration,
            color: config.color
        })),
        timeline_concept: '24-hour Sydney timezone reference starting 10:00',
        x_axis: '24-hour continuous timeline in Sydney time',
        data_display: 'Markets appear only during their Sydney trading windows'
    });
});

function formatHour(hour) {
    const h = Math.floor(hour);
    const m = Math.round((hour % 1) * 60);
    const hourStr = h < 10 ? '0' + h : h.toString();
    const minStr = m === 0 ? '00' : (m < 10 ? '0' + m : m.toString());
    return `${hourStr}:${minStr}`;
}

// Global timeline endpoint
app.get('/global-timeline', (req, res) => {
    const sydneyNow = getSydneyNow();
    const currentSydneyHour = sydneyNow.getHours();
    
    const data = {};
    const activeMarkets = [];
    
    // Generate timeline data for each core index
    Object.keys(CORE_INDICES).forEach(symbol => {
        data[symbol] = generateGlobalTimelineData(symbol);
        
        // Check if market is currently active
        const config = CORE_INDICES[symbol];
        let isActive = false;
        
        if (config.sydney_close > config.sydney_open) {
            isActive = currentSydneyHour >= config.sydney_open && currentSydneyHour <= config.sydney_close;
        } else {
            // US market crosses midnight
            isActive = currentSydneyHour >= config.sydney_open || currentSydneyHour <= config.sydney_close;
        }
        
        if (isActive) {
            activeMarkets.push({
                symbol,
                name: config.name,
                market: config.market
            });
        }
    });
    
    res.json({
        success: true,
        data,
        timeline_reference: '24-hour Sydney timezone starting 10:00',
        current_sydney_time: formatSydneyTime(sydneyNow),
        current_sydney_hour: currentSydneyHour,
        active_markets: activeMarkets,
        market_windows: Object.entries(CORE_INDICES).map(([symbol, config]) => ({
            symbol,
            name: config.name,
            sydney_window: `${formatHour(config.sydney_open)}-${formatHour(config.sydney_close)}`,
            duration: `${config.duration}h`,
            color: config.color,
            is_active: activeMarkets.some(m => m.symbol === symbol)
        })),
        concept: {
            description: 'Four indices across 24-hour Sydney timeline',
            x_axis: 'Continuous 24-hour timeline starting 10am Sydney',
            data_points: 'Markets visible only during Sydney trading windows',
            gaps: 'No data shown outside trading windows (intentional)'
        }
    });
});

// Simple analyze endpoint
app.post('/analyze', (req, res) => {
    // Always return the four core indices
    res.redirect(307, '/global-timeline');
});

// Sydney markets endpoint (alias for global timeline)
app.get('/sydney-markets', (req, res) => {
    res.redirect(301, '/global-timeline');
});

// Serve refined frontend
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'frontend', 'index-refined.html'));
});

app.get('/refined', (req, res) => {
    res.sendFile(path.join(__dirname, 'frontend', 'index-refined.html'));
});

// Start server
app.listen(PORT, () => {
    const sydneyNow = getSydneyNow();
    console.log(`🚀 GSMT Global 24H Timeline Server running on port ${PORT}`);
    console.log(`🕒 Sydney time: ${formatSydneyTime(sydneyNow)}`);
    console.log(`📊 Timeline reference: 10am Sydney (${formatSydneyTime(getSydney10amReference())})`);
    console.log('📈 Core indices: Nikkei(09:00), ASX(10:00), FTSE(18:00), S&P(00:30)');
    console.log(`🌐 Refined Frontend: http://localhost:${PORT}/refined`);
    console.log(`⚡ Timeline API: http://localhost:${PORT}/global-timeline`);
});

module.exports = app;