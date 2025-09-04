/**
 * GSMT Ver 7.0 - Refined Global Timeline App
 * Simplified app focused on 24-hour Sydney timeline with four core indices
 */

class RefinedGSMTApp {
    constructor() {
        // Core application state
        this.state = {
            apiBaseUrl: this.detectApiUrl(),
            chartData: new Map(),
            chartInstance: null,
            isLoading: false,
            refreshTimer: null,
            currentSydneyTime: null
        };
        
        // Four core indices with Sydney trading windows
        this.coreIndices = {
            '^N225': { name: 'Nikkei 225', market: 'Japan', sydney_open: 9, sydney_close: 15, duration: 6, color: '#3b82f6' },
            '^AXJO': { name: 'ASX 200', market: 'Australia', sydney_open: 10, sydney_close: 16, duration: 6, color: '#10b981' },
            '^FTSE': { name: 'FTSE 100', market: 'UK', sydney_open: 18, sydney_close: 24, duration: 6, color: '#f59e0b' },
            '^GSPC': { name: 'S&P 500', market: 'US', sydney_open: 0.5, sydney_close: 7.5, duration: 7, color: '#ef4444' }
        };
        
        this.init();
    }
    
    /**
     * Initialize the refined application
     */
    async init() {
        try {
            console.log('🚀 Initializing GSMT Refined - Global 24H Timeline');
            
            this.setupEventListeners();
            this.initializeChart();
            this.initializeSydneyTime();
            
            // Check API and load data immediately
            await this.checkApiConnection();
            await this.loadGlobalTimelineData();
            
            // Setup auto-refresh
            this.setupAutoRefresh();
            
            console.log('✅ Global 24H Timeline ready');
            this.showToast('Global Timeline: Four indices across 24-hour Sydney timezone', 'success');
            
        } catch (error) {
            console.error('❌ Initialization failed:', error);
            this.showToast('Loading demo data...', 'warning');
            this.loadDemoData();
        }
    }
    
    /**
     * Detect API URL
     */
    detectApiUrl() {
        const saved = localStorage.getItem('gsmt-api-url');
        if (saved) return saved;
        
        if (window.location.hostname === 'localhost') {
            return 'http://localhost:3000';
        }
        
        return null; // Will use demo mode
    }
    
    /**
     * Setup event listeners
     */
    setupEventListeners() {
        document.getElementById('analyze-btn').addEventListener('click', () => this.loadGlobalTimelineData());
        document.getElementById('clear-btn').addEventListener('click', () => this.clearChart());
        document.getElementById('fullscreen-btn').addEventListener('click', () => this.toggleFullscreen());
        
        // Auto-refresh on page focus
        window.addEventListener('focus', () => {
            if (!this.state.isLoading) {
                this.loadGlobalTimelineData(true); // Silent refresh
            }
        });
    }
    
    /**
     * Initialize chart
     */
    initializeChart() {
        const chartContainer = document.getElementById('main-chart');
        this.state.chartInstance = echarts.init(chartContainer, 'light', {
            renderer: 'canvas',
            useDirtyRect: true
        });
        
        this.showEmptyChart();
    }
    
    /**
     * Initialize Sydney time display
     */
    initializeSydneyTime() {
        this.updateSydneyTimeDisplay();
        
        // Update every minute
        setInterval(() => {
            this.updateSydneyTimeDisplay();
        }, 60000);
    }
    
    /**
     * Update Sydney time in UI
     */
    updateSydneyTimeDisplay() {
        try {
            const sydneyNow = new Date(new Date().toLocaleString("en-US", {timeZone: 'Australia/Sydney'}));
            this.state.currentSydneyTime = sydneyNow;
            
            const timeElement = document.querySelector('.sydney-time-value');
            if (timeElement) {
                const formatted = sydneyNow.toLocaleString('en-AU', {
                    timeZone: 'Australia/Sydney',
                    hour: '2-digit',
                    minute: '2-digit',
                    timeZoneName: 'short'
                });
                timeElement.textContent = formatted;
            }
        } catch (error) {
            console.warn('Sydney time display error:', error);
        }
    }
    
    /**
     * Check API connection
     */
    async checkApiConnection() {
        if (!this.state.apiBaseUrl) {
            this.updateApiStatus('disconnected', 'Demo mode');
            return false;
        }
        
        try {
            const response = await fetch(`${this.state.apiBaseUrl}/health`, { timeout: 5000 });
            
            if (response.ok) {
                const data = await response.json();
                this.updateApiStatus('connected', `Live API v${data.version}`);
                return true;
            } else {
                throw new Error(`API returned ${response.status}`);
            }
        } catch (error) {
            console.warn('API connection failed:', error);
            this.updateApiStatus('error', 'Using demo data');
            return false;
        }
    }
    
    /**
     * Load global timeline data
     */
    async loadGlobalTimelineData(silent = false) {
        this.setLoading(true);
        
        try {
            let data;
            
            if (this.state.apiBaseUrl) {
                // Try live API
                const response = await fetch(`${this.state.apiBaseUrl}/sydney-markets`);
                if (response.ok) {
                    data = await response.json();
                } else {
                    throw new Error('API failed');
                }
            } else {
                // Demo data
                data = this.generateDemoData();
            }
            
            // Store chart data
            this.state.chartData.clear();
            Object.entries(data.data).forEach(([symbol, symbolData]) => {
                this.state.chartData.set(symbol, symbolData);
            });
            
            // Update chart
            this.updateGlobalTimelineChart();
            this.updateMarketSessionCards(data.market_sessions);
            
            if (!silent) {
                this.showToast('Global timeline updated - Four indices loaded', 'success');
            }
            
        } catch (error) {
            console.error('Failed to load data:', error);
            if (!silent) {
                this.showToast('Loading demo data...', 'warning');
            }
            this.loadDemoData();
        } finally {
            this.setLoading(false);
        }
    }
    
    /**
     * Update the global timeline chart
     */
    updateGlobalTimelineChart() {
        if (!this.state.chartInstance || this.state.chartData.size === 0) {
            this.showEmptyChart();
            return;
        }
        
        const option = this.createGlobalTimelineOption();
        this.state.chartInstance.setOption(option, true);
    }
    
    /**
     * Create global timeline chart option
     */
    createGlobalTimelineOption() {
        const series = [];
        const markAreas = [];
        
        // Sydney 10am reference point
        const sydney10am = new Date();
        sydney10am.setHours(10, 0, 0, 0);
        
        // Create series for each core index
        Object.entries(this.coreIndices).forEach(([symbol, config]) => {
            if (this.state.chartData.has(symbol)) {
                const marketData = this.state.chartData.get(symbol);
                const timelineData = this.createTimelineDataPoints(symbol, marketData, sydney10am);
                
                series.push({
                    name: config.name,
                    type: 'line',
                    data: timelineData,
                    smooth: true,
                    symbol: 'none',
                    lineStyle: { width: 3, color: config.color },
                    itemStyle: { color: config.color },
                    connectNulls: false
                });
                
                // Add market window shading
                this.addMarketWindow(markAreas, symbol, config, sydney10am);
            }
        });
        
        return {
            title: {
                text: '🌍 Global Markets: 24-Hour Sydney Timeline',
                subtext: 'Nikkei(09:00) → ASX(10:00) → FTSE(18:00) → S&P(00:30+1) • Live Data',
                left: 'center',
                textStyle: { fontSize: 20, fontWeight: 'bold', color: '#1f2937' },
                subtextStyle: { fontSize: 13, color: '#6b7280', lineHeight: 20 }
            },
            tooltip: {
                trigger: 'axis',
                axisPointer: { type: 'cross', lineStyle: { color: '#cbd5e1' } },
                backgroundColor: 'rgba(255, 255, 255, 0.98)',
                borderColor: '#d1d5db',
                borderWidth: 1,
                textStyle: { color: '#374151' },
                formatter: (params) => {
                    const timestamp = params[0].axisValue;
                    const sydneyTime = new Date(timestamp);
                    const timeStr = sydneyTime.toLocaleTimeString('en-AU', {
                        hour: '2-digit',
                        minute: '2-digit',
                        hour12: false,
                        timeZone: 'Australia/Sydney'
                    });
                    
                    let html = `<div style="font-weight: 700; margin-bottom: 8px; color: #1e40af; font-size: 14px;">Sydney: ${timeStr}</div>`;
                    
                    const activeParams = params.filter(p => p.value && p.value[1] !== null);
                    
                    if (activeParams.length === 0) {
                        html += `<div style="color: #6b7280; font-style: italic; margin-top: 4px;">No markets trading</div>`;
                    } else {
                        activeParams.forEach(param => {
                            const value = param.value[1];
                            const changeColor = value >= 0 ? '#059669' : '#dc2626';
                            html += `<div style="display: flex; align-items: center; margin: 6px 0;">`;
                            html += `<span style="display: inline-block; width: 14px; height: 14px; background-color: ${param.color}; border-radius: 50%; margin-right: 12px; border: 2px solid white; box-shadow: 0 0 4px rgba(0,0,0,0.2);"></span>`;
                            html += `<span style="flex: 1; font-weight: 600; font-size: 13px;">${param.seriesName}: </span>`;
                            html += `<span style="font-weight: bold; font-size: 14px; color: ${changeColor}">${value >= 0 ? '+' : ''}${value.toFixed(2)}%</span>`;
                            html += `</div>`;
                        });
                    }
                    
                    return html;
                }
            },
            legend: {
                top: 75,
                type: 'plain',
                textStyle: { fontSize: 13, fontWeight: '600' },
                itemGap: 30,
                itemWidth: 25,
                itemHeight: 14
            },
            grid: {
                left: '5%',
                right: '4%',
                bottom: '20%',
                top: '28%',
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
                        return `${hour.toString().padStart ? hour.toString().padStart(2, '0') : (hour < 10 ? '0' + hour : hour)}:00`;
                    },
                    interval: (index, value) => {
                        const time = new Date(value);
                        return time.getMinutes() === 0 && time.getHours() % 2 === 0; // Every 2 hours
                    },
                    rotate: 0
                },
                axisTick: {
                    show: true,
                    lineStyle: { color: '#d1d5db' },
                    interval: (index, value) => {
                        const time = new Date(value);
                        return time.getMinutes() === 0; // Every hour
                    }
                },
                splitLine: {
                    show: true,
                    lineStyle: {
                        color: '#f3f4f6',
                        type: 'solid',
                        width: 1,
                        opacity: 0.8
                    },
                    interval: (index, value) => {
                        const time = new Date(value);
                        return time.getMinutes() === 0 && time.getHours() % 6 === 0; // Every 6 hours
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
                nameTextStyle: { color: '#4b5563', fontSize: 12, fontWeight: '600' },
                axisLine: { lineStyle: { color: '#9ca3af' } },
                axisLabel: {
                    color: '#6b7280',
                    formatter: '{value}%',
                    fontSize: 11
                },
                splitLine: { lineStyle: { color: '#f3f4f6', opacity: 0.8 } }
            },
            series: series.concat([{
                name: 'Market Trading Windows',
                type: 'line',
                data: [],
                markArea: {
                    silent: true,
                    itemStyle: { opacity: 0.08 },
                    label: {
                        show: true,
                        position: 'insideTop',
                        fontSize: 11,
                        color: '#6b7280',
                        fontWeight: 'bold',
                        distance: 5
                    },
                    data: markAreas
                }
            }]),
            animation: true,
            animationDuration: 2000,
            animationEasing: 'cubicOut'
        };
    }
    
    /**
     * Create timeline data points for an index
     */
    createTimelineDataPoints(symbol, marketData, sydney10am) {
        const timelineData = [];
        const config = this.coreIndices[symbol];
        
        if (!config || !marketData) return [];
        
        // Create 24-hour timeline (hourly points)
        for (let hour = 0; hour < 24; hour++) {
            const timestamp = new Date(sydney10am);
            timestamp.setHours(sydney10am.getHours() + hour);
            
            // Calculate current Sydney hour (0-24)
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
            
            if (isTrading && marketData.length > 0) {
                // Map to corresponding market data point
                const progress = this.calculateTradingProgress(sydneyHour, config);
                const dataIndex = Math.floor(progress * (marketData.length - 1));
                const point = marketData[Math.min(dataIndex, marketData.length - 1)];
                
                timelineData.push([timestamp.getTime(), point.percentage_change]);
            } else {
                // Null data outside trading window
                timelineData.push([timestamp.getTime(), null]);
            }
        }
        
        return timelineData;
    }
    
    /**
     * Calculate trading progress (0-1) within session
     */
    calculateTradingProgress(sydneyHour, config) {
        if (config.sydney_close > config.sydney_open) {
            // Same day session
            return Math.max(0, Math.min(1, (sydneyHour - config.sydney_open) / (config.sydney_close - config.sydney_open)));
        } else {
            // Cross-midnight session (US)
            if (sydneyHour >= config.sydney_open) {
                const totalHours = (24 - config.sydney_open) + config.sydney_close;
                return (sydneyHour - config.sydney_open) / totalHours;
            } else {
                const totalHours = (24 - config.sydney_open) + config.sydney_close;
                return ((24 - config.sydney_open) + sydneyHour) / totalHours;
            }
        }
    }
    
    /**
     * Add market window shading
     */
    addMarketWindow(markAreas, symbol, config, sydney10am) {
        let startTime, endTime;
        
        if (symbol === '^GSPC') {
            // US market: 00:30-07:30 (spans midnight)
            startTime = sydney10am.getTime() + (14.5 * 60 * 60 * 1000); // 14.5h from 10am = 00:30 next day
            endTime = sydney10am.getTime() + (21.5 * 60 * 60 * 1000);   // 21.5h from 10am = 07:30 next day
        } else {
            // Other markets (same day)
            const hoursFrom10am = config.sydney_open - 10;
            startTime = sydney10am.getTime() + (hoursFrom10am * 60 * 60 * 1000);
            endTime = sydney10am.getTime() + ((config.sydney_close - 10) * 60 * 60 * 1000);
        }
        
        markAreas.push([
            {
                xAxis: startTime,
                itemStyle: {
                    color: config.color.replace('#', 'rgba(').replace(config.color, this.hexToRgba(config.color, 0.06)),
                    borderColor: this.hexToRgba(config.color, 0.25),
                    borderWidth: 2
                },
                label: {
                    show: true,
                    position: 'insideTop',
                    formatter: `${config.name}\n${this.formatHour(config.sydney_open)}-${this.formatHour(config.sydney_close)}`,
                    color: config.color,
                    fontSize: 11,
                    fontWeight: 'bold'
                }
            },
            {
                xAxis: endTime,
                itemStyle: {
                    color: this.hexToRgba(config.color, 0.06),
                    borderColor: this.hexToRgba(config.color, 0.25),
                    borderWidth: 2
                }
            }
        ]);
    }
    
    /**
     * Convert hex to rgba
     */
    hexToRgba(hex, alpha) {
        const r = parseInt(hex.slice(1, 3), 16);
        const g = parseInt(hex.slice(3, 5), 16);
        const b = parseInt(hex.slice(5, 7), 16);
        return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    }
    
    /**
     * Format hour (handle decimals)
     */
    formatHour(hour) {
        const h = Math.floor(hour);
        const m = Math.round((hour % 1) * 60);
        const hourStr = h < 10 ? '0' + h : h.toString();
        const minStr = m === 0 ? '00' : (m < 10 ? '0' + m : m.toString());
        return `${hourStr}:${minStr}`;
    }
    
    /**
     * Load demo data
     */
    loadDemoData() {
        this.state.chartData.clear();
        
        Object.keys(this.coreIndices).forEach(symbol => {
            const demoData = this.generateIndexDemoData(symbol);
            this.state.chartData.set(symbol, demoData);
        });
        
        this.updateGlobalTimelineChart();
        this.showDemoMarketSessions();
    }
    
    /**
     * Generate demo data for an index
     */
    generateIndexDemoData(symbol) {
        const config = this.coreIndices[symbol];
        const data = [];
        const basePrice = Math.random() * 15000 + 5000;
        let currentPrice = basePrice;
        
        // Generate points during trading window only
        const intervals = config.duration * 4; // 15-min intervals
        
        for (let i = 0; i < intervals; i++) {
            const volatility = 0.01;
            const change = (Math.random() - 0.5) * volatility;
            currentPrice *= (1 + change);
            
            const percentageChange = ((currentPrice - basePrice) / basePrice) * 100;
            
            data.push({
                timestamp: new Date().toISOString(),
                timestamp_ms: Date.now() + (i * 15 * 60 * 1000),
                percentage_change: percentageChange,
                close: currentPrice
            });
        }
        
        return data;
    }
    
    /**
     * Show empty chart
     */
    showEmptyChart() {
        if (!this.state.chartInstance) return;
        
        this.state.chartInstance.setOption({
            title: {
                text: 'Loading Global Timeline...',
                subtext: 'Fetching live market data',
                left: 'center',
                top: 'middle',
                textStyle: { fontSize: 16, color: '#6b7280' },
                subtextStyle: { fontSize: 12, color: '#9ca3af' }
            }
        });
    }
    
    /**
     * Update market session cards
     */
    updateMarketSessionCards(sessions) {
        const container = document.getElementById('market-sessions');
        if (!container || !sessions) return;
        
        container.classList.remove('hidden');
        
        const currentSydneyHour = this.state.currentSydneyTime ? 
            this.state.currentSydneyTime.getHours() : new Date().getHours();
        
        const grid = container.querySelector('.grid');
        const cards = Object.entries(this.coreIndices).map(([symbol, config]) => {
            const isActive = this.isMarketActive(currentSydneyHour, config);
            const statusClass = isActive ? 'border-emerald-500 bg-emerald-50' : 'border-gray-300 bg-gray-50';
            const statusText = isActive ? 'TRADING' : 'CLOSED';
            
            return `
                <div class="border-2 ${statusClass} rounded-lg p-4 text-center">
                    <div class="flex items-center justify-center mb-3">
                        <div class="w-3 h-3 rounded-full mr-2" style="background-color: ${config.color}"></div>
                        <span class="text-sm font-bold ${isActive ? 'text-emerald-700' : 'text-gray-600'}">${statusText}</span>
                    </div>
                    <div class="font-semibold text-base text-gray-900 mb-2">${config.name}</div>
                    <div class="text-sm text-gray-600 mb-1">
                        ${this.formatHour(config.sydney_open)} - ${this.formatHour(config.sydney_close)}
                    </div>
                    <div class="text-xs text-gray-500">
                        ${config.duration}h session • Sydney time
                    </div>
                </div>
            `;
        }).join('');
        
        grid.innerHTML = cards;
    }
    
    /**
     * Show demo market sessions
     */
    showDemoMarketSessions() {
        const demoSessions = Object.entries(this.coreIndices).map(([symbol, config]) => config);
        this.updateMarketSessionCards(demoSessions);
    }
    
    /**
     * Check if market is active at Sydney hour
     */
    isMarketActive(sydneyHour, config) {
        if (config.sydney_close > config.sydney_open) {
            return sydneyHour >= config.sydney_open && sydneyHour <= config.sydney_close;
        } else {
            // Cross-midnight (US)
            return sydneyHour >= config.sydney_open || sydneyHour <= config.sydney_close;
        }
    }
    
    /**
     * Setup auto-refresh
     */
    setupAutoRefresh() {
        // Refresh every 3 minutes during any trading session
        this.state.refreshTimer = setInterval(() => {
            const sydneyHour = this.state.currentSydneyTime ? 
                this.state.currentSydneyTime.getHours() : new Date().getHours();
            
            // Check if any market is active
            const anyActive = Object.values(this.coreIndices).some(config => 
                this.isMarketActive(sydneyHour, config)
            );
            
            if (anyActive) {
                console.log('🔄 Auto-refresh: Market active');
                this.loadGlobalTimelineData(true);
            }
        }, 180000); // 3 minutes
    }
    
    /**
     * Set loading state
     */
    setLoading(loading) {
        this.state.isLoading = loading;
        const button = document.getElementById('analyze-btn');
        const indicator = document.getElementById('loading-indicator');
        
        if (loading) {
            if (button) {
                button.disabled = true;
                button.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Loading Timeline...';
            }
            if (indicator) indicator.classList.remove('hidden');
        } else {
            if (button) {
                button.disabled = false;
                button.innerHTML = '<i class="fas fa-globe mr-2"></i>Refresh Timeline';
            }
            if (indicator) indicator.classList.add('hidden');
        }
    }
    
    /**
     * Update API status
     */
    updateApiStatus(status, message) {
        const statusElement = document.getElementById('api-status');
        if (!statusElement) return;
        
        const dot = statusElement.querySelector('div');
        const text = statusElement.querySelector('span');
        
        dot.className = 'w-2 h-2 rounded-full';
        
        switch (status) {
            case 'connected':
                dot.classList.add('bg-emerald-500');
                text.className = 'text-xs text-emerald-600';
                break;
            case 'error':
                dot.classList.add('bg-red-500');
                text.className = 'text-xs text-red-600';
                break;
            default:
                dot.classList.add('bg-gray-400');
                text.className = 'text-xs text-gray-500';
        }
        
        text.textContent = message;
    }
    
    /**
     * Show toast notification
     */
    showToast(message, type = 'info') {
        const container = document.getElementById('toast-container');
        if (!container) return;
        
        const colors = {
            success: 'bg-emerald-500',
            error: 'bg-red-500',
            warning: 'bg-yellow-500',
            info: 'bg-blue-500'
        };
        
        const toast = document.createElement('div');
        toast.className = `flex items-center p-4 rounded-lg shadow-lg text-white ${colors[type]} transform transition-all duration-300`;
        toast.innerHTML = `
            <span class="flex-1">${message}</span>
            <button onclick="this.parentElement.remove()" class="ml-3 text-white hover:text-gray-200">
                <i class="fas fa-times"></i>
            </button>
        `;
        
        container.appendChild(toast);
        setTimeout(() => toast.remove(), 4000);
    }
    
    /**
     * Clear chart
     */
    clearChart() {
        this.state.chartData.clear();
        this.showEmptyChart();
        this.showToast('Chart cleared', 'info');
    }
    
    /**
     * Toggle fullscreen
     */
    toggleFullscreen() {
        const chartContainer = document.getElementById('main-chart');
        
        if (!document.fullscreenElement) {
            chartContainer.requestFullscreen().then(() => {
                setTimeout(() => this.state.chartInstance?.resize(), 100);
            });
        } else {
            document.exitFullscreen();
        }
    }
    
    /**
     * Generate demo data
     */
    generateDemoData() {
        const demoData = {};
        
        Object.keys(this.coreIndices).forEach(symbol => {
            demoData[symbol] = this.generateIndexDemoData(symbol);
        });
        
        return {
            success: true,
            data: demoData,
            market_sessions: Object.entries(this.coreIndices).map(([symbol, config]) => config)
        };
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.refinedApp = new RefinedGSMTApp();
});

window.RefinedGSMTApp = RefinedGSMTApp;