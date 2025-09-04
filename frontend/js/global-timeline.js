/**
 * GSMT Ver 7.0 - Global 24H Timeline
 * Core implementation for 24-hour Sydney timezone market flow
 */

class GlobalTimelineManager {
    constructor() {
        // Core four indices only
        this.coreIndices = {
            '^N225': {
                name: 'Nikkei 225',
                market: 'Japan',
                sydney_open: 9,      // 09:00 Sydney time
                sydney_close: 15,    // 15:00 Sydney time
                duration: 6,
                color: '#3b82f6'     // Blue
            },
            '^AXJO': {
                name: 'ASX 200',
                market: 'Australia', 
                sydney_open: 10,     // 10:00 Sydney time
                sydney_close: 16,    // 16:00 Sydney time
                duration: 6,
                color: '#10b981'     // Green
            },
            '^FTSE': {
                name: 'FTSE 100',
                market: 'UK',
                sydney_open: 18,     // 18:00 Sydney time
                sydney_close: 24,    // 24:00 Sydney time (midnight)
                duration: 6,
                color: '#f59e0b'     // Amber
            },
            '^GSPC': {
                name: 'S&P 500',
                market: 'US',
                sydney_open: 0.5,    // 00:30 Sydney time (next day)
                sydney_close: 7.5,   // 07:30 Sydney time
                duration: 7,
                color: '#ef4444'     // Red
            }
        };
    }
    
    /**
     * Create 24-hour ECharts option for global timeline
     */
    createGlobalTimelineChart(chartData) {
        const series = [];
        const markAreas = [];
        
        // Create timeline starting from 10am Sydney (reference point)
        const sydney10am = new Date();
        sydney10am.setHours(10, 0, 0, 0);
        
        // Generate series for each core index
        Object.entries(this.coreIndices).forEach(([symbol, config]) => {
            if (chartData.has(symbol)) {
                const data = chartData.get(symbol);
                const timelineData = this.createIndexTimelineData(symbol, data, sydney10am);
                
                series.push({
                    name: config.name,
                    type: 'line',
                    data: timelineData,
                    smooth: true,
                    symbol: 'none',
                    lineStyle: { 
                        width: 3,
                        color: config.color
                    },
                    itemStyle: {
                        color: config.color
                    },
                    emphasis: {
                        lineStyle: { width: 4 }
                    },
                    connectNulls: false // Don't connect across non-trading periods
                });
                
                // Add market window
                this.addMarketWindow(markAreas, symbol, config, sydney10am);
            }
        });
        
        return {
            title: {
                text: 'Global Markets: 24-Hour Sydney Timeline',
                subtext: 'Nikkei(09:00) → ASX(10:00) → FTSE(18:00) → S&P(00:30+1)',
                left: 'center',
                textStyle: { fontSize: 20, fontWeight: 'bold', color: '#1f2937' },
                subtextStyle: { fontSize: 13, color: '#6b7280' }
            },
            tooltip: {
                trigger: 'axis',
                axisPointer: { type: 'cross', lineStyle: { color: '#cbd5e1' } },
                backgroundColor: 'rgba(255, 255, 255, 0.98)',
                borderColor: '#e5e7eb',
                textStyle: { color: '#374151' },
                formatter: (params) => {
                    const timestamp = params[0].axisValue;
                    const sydneyTime = new Date(timestamp);
                    const timeStr = sydneyTime.toLocaleTimeString('en-AU', {
                        hour: '2-digit',
                        minute: '2-digit', 
                        hour12: false
                    });
                    
                    let html = `<div style="font-weight: bold; margin-bottom: 8px; color: #1e40af; font-size: 14px;">Sydney: ${timeStr}</div>`;
                    
                    // Show active indices only
                    const activeParams = params.filter(p => p.value && p.value[1] !== null);
                    
                    if (activeParams.length === 0) {
                        html += `<div style="color: #6b7280; font-style: italic;">No markets trading</div>`;
                    } else {
                        activeParams.forEach(param => {
                            const value = param.value[1];
                            const color = param.color;
                            html += `<div style="display: flex; align-items: center; margin: 4px 0;">`;
                            html += `<span style="display: inline-block; width: 14px; height: 14px; background-color: ${color}; border-radius: 50%; margin-right: 12px;"></span>`;
                            html += `<span style="flex: 1; font-weight: 600;">${param.seriesName}: </span>`;
                            html += `<span style="font-weight: bold; font-size: 14px; color: ${value >= 0 ? '#059669' : '#dc2626'}">${value >= 0 ? '+' : ''}${value.toFixed(2)}%</span>`;
                            html += `</div>`;
                        });
                    }
                    
                    return html;
                }
            },
            legend: {
                top: 70,
                type: 'plain',
                textStyle: { fontSize: 13, fontWeight: '600' },
                itemGap: 25
            },
            grid: {
                left: '5%',
                right: '4%',
                bottom: '18%',
                top: '25%',
                containLabel: true
            },
            xAxis: {
                type: 'time',
                axisLine: { lineStyle: { color: '#9ca3af', width: 2 } },
                axisLabel: {
                    color: '#4b5563',
                    fontSize: 12,
                    fontWeight: '500',
                    formatter: (value) => {
                        const time = new Date(value);
                        const hour = time.getHours();
                        const minute = time.getMinutes();
                        
                        // Show hour markers every 3 hours for clarity
                        if (minute === 0 && hour % 3 === 0) {
                            return `${hour.toString().padStart(2, '0')}:00`;
                        } else if (minute === 0) {
                            return `${hour.toString().padStart(2, '0')}`;
                        }
                        return '';
                    },
                    interval: 0,
                    rotate: 0
                },
                axisTick: {
                    show: true,
                    lineStyle: { color: '#d1d5db' },
                    interval: (index, value) => {
                        const time = new Date(value);
                        return time.getMinutes() === 0; // Show ticks on the hour
                    }
                },
                splitLine: {
                    show: true,
                    lineStyle: { 
                        color: '#f3f4f6', 
                        type: 'solid', 
                        width: 1,
                        opacity: 0.6
                    },
                    interval: (index, value) => {
                        const time = new Date(value);
                        return time.getMinutes() === 0 && time.getHours() % 3 === 0; // Every 3 hours
                    }
                },
                // Full 24-hour range starting from 10am Sydney
                min: sydney10am.getTime(),
                max: sydney10am.getTime() + (24 * 60 * 60 * 1000)
            },
            yAxis: {
                type: 'value',
                name: 'Percentage Change (%)',
                nameLocation: 'middle',
                nameGap: 50,
                nameTextStyle: { color: '#4b5563', fontSize: 12 },
                axisLine: { lineStyle: { color: '#9ca3af' } },
                axisLabel: {
                    color: '#6b7280',
                    formatter: '{value}%',
                    fontSize: 11
                },
                splitLine: { lineStyle: { color: '#f3f4f6', opacity: 0.8 } }
            },
            series: series.concat(markAreas.length > 0 ? [{
                name: 'Trading Windows',
                type: 'line',
                data: [],
                markArea: {
                    silent: true,
                    itemStyle: { opacity: 0.08 },
                    label: {
                        show: true,
                        position: 'top',
                        fontSize: 11,
                        color: '#6b7280',
                        fontWeight: 'bold'
                    },
                    data: markAreas
                }
            }] : []),
            animation: true,
            animationDuration: 2500,
            animationEasing: 'cubicOut'
        };
    }
    
    /**
     * Create timeline data for a specific index
     */
    createIndexTimelineData(symbol, marketData, sydney10am) {
        const timelineData = [];
        const config = this.coreIndices[symbol];
        if (!config || !marketData || marketData.length === 0) return [];
        
        // Create 24-hour timeline (every hour for smooth display)
        for (let hour = 0; hour < 24; hour++) {
            const timestamp = new Date(sydney10am);
            timestamp.setHours(sydney10am.getHours() + hour);
            
            // Calculate Sydney hour (0-24 format)
            const sydneyHour = (10 + hour) % 24;
            
            // Check if this index is trading at this Sydney time
            let isTrading = false;
            if (config.sydney_close > config.sydney_open) {
                // Same day trading
                isTrading = sydneyHour >= config.sydney_open && sydneyHour <= config.sydney_close;
            } else {
                // Cross-midnight trading (US market)
                isTrading = sydneyHour >= config.sydney_open || sydneyHour <= config.sydney_close;
            }
            
            if (isTrading) {
                // Find corresponding data point from market data
                const progress = isTrading ? this.calculateTradingProgress(sydneyHour, config) : 0;
                const dataIndex = Math.floor(progress * (marketData.length - 1));
                const point = marketData[Math.min(dataIndex, marketData.length - 1)];
                
                timelineData.push([timestamp.getTime(), point.percentage_change]);
            } else {
                // No data outside trading window
                timelineData.push([timestamp.getTime(), null]);
            }
        }
        
        return timelineData;
    }
    
    /**
     * Calculate trading progress within session (0-1)
     */
    calculateTradingProgress(sydneyHour, config) {
        if (config.sydney_close > config.sydney_open) {
            // Same day session
            return (sydneyHour - config.sydney_open) / (config.sydney_close - config.sydney_open);
        } else {
            // Cross-midnight session (US)
            if (sydneyHour >= config.sydney_open) {
                return (sydneyHour - config.sydney_open) / ((24 - config.sydney_open) + config.sydney_close);
            } else {
                return ((24 - config.sydney_open) + sydneyHour) / ((24 - config.sydney_open) + config.sydney_close);
            }
        }
    }
    
    /**
     * Add market window visualization
     */
    addMarketWindow(markAreas, symbol, config, sydney10am) {
        let startTime, endTime;
        
        if (symbol === '^GSPC') {
            // US market: 00:30-07:30 (next day in timeline)
            startTime = sydney10am.getTime() + (14.5 * 60 * 60 * 1000); // 14.5h after 10am = 00:30 next day
            endTime = sydney10am.getTime() + (21.5 * 60 * 60 * 1000);   // 21.5h after 10am = 07:30 next day
        } else {
            // Other markets within same day
            const hoursFrom10am = config.sydney_open - 10;
            startTime = sydney10am.getTime() + (hoursFrom10am * 60 * 60 * 1000);
            endTime = sydney10am.getTime() + ((config.sydney_close - 10) * 60 * 60 * 1000);
        }
        
        markAreas.push([
            {
                xAxis: startTime,
                itemStyle: { 
                    color: config.color.replace(/rgb\(([^)]+)\)/, 'rgba($1, 0.08)'),
                    borderColor: config.color.replace(/rgb\(([^)]+)\)/, 'rgba($1, 0.25)'),
                    borderWidth: 2
                },
                label: {
                    show: true,
                    position: 'insideTop',
                    formatter: config.name,
                    color: config.color,
                    fontSize: 12,
                    fontWeight: 'bold'
                }
            },
            {
                xAxis: endTime,
                itemStyle: { 
                    color: config.color.replace(/rgb\(([^)]+)\)/, 'rgba($1, 0.08)'),
                    borderColor: config.color.replace(/rgb\(([^)]+)\)/, 'rgba($1, 0.25)'),
                    borderWidth: 2
                }
            }
        ]);
    }
    
    /**
     * Get market status for current Sydney time
     */
    getCurrentMarketStatus() {
        const sydneyNow = new Date(new Date().toLocaleString("en-US", {timeZone: 'Australia/Sydney'}));
        const currentHour = sydneyNow.getHours() + (sydneyNow.getMinutes() / 60);
        
        const status = {};
        
        Object.entries(this.coreIndices).forEach(([symbol, config]) => {
            let isActive = false;
            
            if (config.sydney_close > config.sydney_open) {
                // Same day trading
                isActive = currentHour >= config.sydney_open && currentHour <= config.sydney_close;
            } else {
                // Cross-midnight trading (US)
                isActive = currentHour >= config.sydney_open || currentHour <= config.sydney_close;
            }
            
            status[symbol] = {
                name: config.name,
                market: config.market,
                isActive,
                window: `${this.formatHour(config.sydney_open)}-${this.formatHour(config.sydney_close)}`,
                color: config.color
            };
        });
        
        return status;
    }
    
    /**
     * Format hour for display (handle decimals)
     */
    formatHour(hour) {
        const h = Math.floor(hour);
        const m = Math.round((hour % 1) * 60);
        
        if (m === 0) {
            return `${h.toString().padStart(2, '0')}:00`;
        } else {
            return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}`;
        }
    }
    
    /**
     * Get next market opening
     */
    getNextMarketOpening() {
        const sydneyNow = new Date(new Date().toLocaleString("en-US", {timeZone: 'Australia/Sydney'}));
        const currentHour = sydneyNow.getHours() + (sydneyNow.getMinutes() / 60);
        
        const openingTimes = Object.entries(this.coreIndices)
            .map(([symbol, config]) => ({
                symbol,
                name: config.name,
                hour: config.sydney_open,
                market: config.market
            }))
            .sort((a, b) => a.hour - b.hour);
        
        // Find next opening
        for (const opening of openingTimes) {
            if (currentHour < opening.hour) {
                const hoursUntil = opening.hour - currentHour;
                return {
                    market: opening.name,
                    symbol: opening.symbol,
                    hours_until: Math.round(hoursUntil * 100) / 100,
                    opens_at: this.formatHour(opening.hour)
                };
            }
        }
        
        // Next day - first market is US at 00:30
        const hoursUntil = (24 - currentHour) + 0.5;
        return {
            market: 'S&P 500',
            symbol: '^GSPC',
            hours_until: Math.round(hoursUntil * 100) / 100,
            opens_at: '00:30 (tomorrow)'
        };
    }
    
    /**
     * Generate mock data for a specific index in its Sydney trading window
     */
    generateMockTimelineData(symbol) {
        const config = this.coreIndices[symbol];
        if (!config) return [];
        
        const data = [];
        const basePrice = Math.random() * 15000 + 5000;
        let currentPrice = basePrice;
        
        // Generate 15-minute intervals during trading window
        const windowHours = config.duration;
        const intervals = windowHours * 4; // 4 intervals per hour
        
        for (let i = 0; i < intervals; i++) {
            // Create timestamp within trading window
            const intervalHours = (i / 4); // Hours from session start
            const sessionTime = config.sydney_open + intervalHours;
            
            // Create timestamp
            const timestamp = new Date();
            if (symbol === '^GSPC' && sessionTime < 10) {
                // US market - next day times
                timestamp.setDate(timestamp.getDate() + 1);
                timestamp.setHours(Math.floor(sessionTime), (sessionTime % 1) * 60, 0, 0);
            } else {
                timestamp.setHours(Math.floor(sessionTime), (sessionTime % 1) * 60, 0, 0);
            }
            
            // Price movement
            const volatility = 0.008; // 0.8% per 15min
            const change = (Math.random() - 0.5) * volatility;
            currentPrice *= (1 + change);
            
            const percentageChange = ((currentPrice - basePrice) / basePrice) * 100;
            
            data.push({
                timestamp: timestamp.toISOString(),
                timestamp_ms: timestamp.getTime(),
                close: currentPrice,
                percentage_change: percentageChange,
                sydney_hour: sessionTime,
                market: config.market,
                trading_session: true
            });
        }
        
        return data;
    }
}

// Export for global access
window.GlobalTimelineManager = GlobalTimelineManager;