/**
 * Global Stock Market Tracker - 24H UTC Timeline
 * Local deployment frontend application
 */

class GlobalMarketTracker {
    constructor() {
        this.apiBaseUrl = this.detectApiUrl();
        this.selectedIndices = new Set();
        this.allSymbols = new Map();
        this.chartInstance = null;
        this.marketHours = {};
        this.currentData = {};
        this.refreshInterval = null;
        
        this.init();
    }

    detectApiUrl() {
        // For local deployment, API runs on same host
        const baseUrl = window.location.origin;
        return `${baseUrl}/api`;
    }

    async init() {
        console.log('🚀 Initializing Global Market Tracker');
        
        try {
            // Setup event listeners
            this.setupEventListeners();
            
            // Initialize chart
            this.initializeChart();
            
            // Start UTC clock
            this.updateUTCClock();
            setInterval(() => this.updateUTCClock(), 1000);
            
            // Load initial data
            await this.loadSymbols();
            await this.loadMarketStatus();
            
            // Load suggested indices
            await this.loadSuggestedIndices();
            
            // Auto-refresh every 5 minutes
            this.startAutoRefresh();
            
            this.showToast('Global Market Tracker loaded successfully', 'success');
            
        } catch (error) {
            console.error('❌ Initialization failed:', error);
            this.showToast('Failed to connect to API. Please ensure the server is running.', 'error');
        }
    }

    setupEventListeners() {
        // Search functionality
        document.getElementById('index-search').addEventListener('input', 
            this.debounce(this.handleSearch.bind(this), 300));
        
        // Analyze button
        document.getElementById('analyze-btn').addEventListener('click', 
            this.analyzeSelectedIndices.bind(this));
        
        // Refresh button
        document.getElementById('refresh-btn').addEventListener('click', 
            this.refreshData.bind(this));
        
        // Chart type change
        document.getElementById('chart-type').addEventListener('change', 
            this.handleChartTypeChange.bind(this));
    }

    initializeChart() {
        const chartElement = document.getElementById('main-chart');
        this.chartInstance = echarts.init(chartElement);
        
        // Set initial empty state
        this.updateChart([]);
    }

    updateUTCClock() {
        const now = new Date();
        const utcString = now.toISOString().replace('T', ' ').substring(0, 19) + ' UTC';
        document.getElementById('current-utc').textContent = utcString;
    }

    async loadSymbols() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/symbols`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            
            const data = await response.json();
            
            // Store symbols by market
            for (const [market, symbols] of Object.entries(data.markets)) {
                symbols.forEach(symbol => {
                    this.allSymbols.set(symbol.symbol, {
                        ...symbol,
                        market: market
                    });
                });
            }
            
            this.marketHours = data.market_hours_utc;
            console.log(`📊 Loaded ${this.allSymbols.size} symbols from ${Object.keys(data.markets).length} markets`);
            
        } catch (error) {
            console.error('Failed to load symbols:', error);
            throw error;
        }
    }

    async loadMarketStatus() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/market-hours`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            
            const data = await response.json();
            
            this.updateMarketStatusDisplay(data);
            this.updateMarketHoursReference(data.markets);
            
        } catch (error) {
            console.error('Failed to load market status:', error);
        }
    }

    async loadSuggestedIndices() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/suggested-indices`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            
            const data = await response.json();
            this.displaySuggestedIndices(data.suggested_indices);
            
        } catch (error) {
            console.error('Failed to load suggested indices:', error);
        }
    }

    displaySuggestedIndices(suggestedIndices) {
        const container = document.getElementById('suggested-indices');
        container.innerHTML = '';
        
        for (const [region, indices] of Object.entries(suggestedIndices)) {
            const regionDiv = document.createElement('div');
            regionDiv.className = 'border rounded-lg p-4';
            
            const regionTitle = region.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
            regionDiv.innerHTML = `
                <h4 class="font-medium text-gray-900 mb-3">${regionTitle}</h4>
                <div class="space-y-2">
                    ${indices.map(index => `
                        <label class="flex items-center cursor-pointer hover:bg-gray-50 p-2 rounded">
                            <input type="checkbox" 
                                   value="${index.symbol}" 
                                   class="index-checkbox mr-3 h-4 w-4 text-primary-600"
                                   onchange="window.tracker.toggleIndex('${index.symbol}')">
                            <div class="flex-1 min-w-0">
                                <div class="flex items-center justify-between">
                                    <span class="font-medium text-sm">${index.name}</span>
                                    <span class="text-xs text-gray-500">${index.symbol}</span>
                                </div>
                                <div class="text-xs text-gray-500">${index.hours}</div>
                            </div>
                        </label>
                    `).join('')}
                </div>
            `;
            
            container.appendChild(regionDiv);
        }
    }

    toggleIndex(symbol) {
        if (this.selectedIndices.has(symbol)) {
            this.selectedIndices.delete(symbol);
        } else {
            this.selectedIndices.add(symbol);
        }
        
        this.updateSelectedIndicesDisplay();
        this.updateAnalyzeButton();
    }

    updateSelectedIndicesDisplay() {
        const container = document.getElementById('selected-indices');
        const listContainer = document.getElementById('selected-indices-list');
        
        if (this.selectedIndices.size === 0) {
            container.classList.add('hidden');
            return;
        }
        
        container.classList.remove('hidden');
        listContainer.innerHTML = '';
        
        this.selectedIndices.forEach(symbol => {
            const symbolInfo = this.allSymbols.get(symbol);
            if (!symbolInfo) return;
            
            const chip = document.createElement('div');
            chip.className = 'bg-primary-100 text-primary-800 px-3 py-1 rounded-full text-sm flex items-center';
            chip.innerHTML = `
                <span class="mr-2">${symbolInfo.name} (${symbol})</span>
                <button onclick="window.tracker.removeIndex('${symbol}')" 
                        class="text-primary-600 hover:text-primary-800">
                    <i class="fas fa-times"></i>
                </button>
            `;
            listContainer.appendChild(chip);
        });
    }

    removeIndex(symbol) {
        this.selectedIndices.delete(symbol);
        
        // Update checkbox
        const checkbox = document.querySelector(`input[value="${symbol}"]`);
        if (checkbox) checkbox.checked = false;
        
        this.updateSelectedIndicesDisplay();
        this.updateAnalyzeButton();
    }

    updateAnalyzeButton() {
        const button = document.getElementById('analyze-btn');
        button.disabled = this.selectedIndices.size === 0;
    }

    async analyzeSelectedIndices() {
        if (this.selectedIndices.size === 0) return;
        
        this.showLoading(true);
        
        try {
            const chartType = document.getElementById('chart-type').value;
            
            const response = await fetch(`${this.apiBaseUrl}/analyze`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    symbols: Array.from(this.selectedIndices),
                    chart_type: chartType
                })
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `HTTP ${response.status}`);
            }
            
            const data = await response.json();
            this.currentData = data;
            this.updateChart(data);
            
            this.showToast(`Loaded data for ${data.successful_symbols} indices`, 'success');
            
        } catch (error) {
            console.error('Analysis failed:', error);
            this.showToast(`Analysis failed: ${error.message}`, 'error');
        } finally {
            this.showLoading(false);
        }
    }

    updateChart(data) {
        const noDataMessage = document.getElementById('no-data-message');
        
        if (!data.data || Object.keys(data.data).length === 0) {
            noDataMessage.classList.remove('hidden');
            this.chartInstance.clear();
            return;
        }
        
        noDataMessage.classList.add('hidden');
        
        const chartType = document.getElementById('chart-type').value;
        const series = [];
        const xAxisData = [];
        const markLines = [];
        
        // Get time points from first symbol - show date and hours for rolling window
        const firstSymbol = Object.keys(data.data)[0];
        if (firstSymbol) {
            data.data[firstSymbol].forEach(point => {
                const timestamp = new Date(point.timestamp);
                const month = (timestamp.getUTCMonth() + 1).toString().padStart(2, '0');
                const day = timestamp.getUTCDate().toString().padStart(2, '0');
                const hour = timestamp.getUTCHours().toString().padStart(2, '0');
                
                // Format: MM/DD HH:00 for rolling window display
                xAxisData.push(`${month}/${day} ${hour}:00`);
            });
        }
        
        // Calculate data range for intelligent y-axis scaling
        let allValues = [];
        
        // Create series for each selected index
        Object.entries(data.data).forEach(([symbol, points]) => {
            const symbolInfo = data.metadata[symbol];
            
            // Handle null values - ECharts will create gaps in the line for null values
            const values = points.map(point => {
                if (!point.market_open || point.percentage_change === null || point.close === null) {
                    return null;  // This creates a gap in the line chart
                }
                const value = chartType === 'percentage' ? point.percentage_change : point.close;
                if (value !== null && !isNaN(value)) {
                    allValues.push(value);
                }
                return value;
            });
            
            series.push({
                name: symbolInfo.name,
                type: 'line',
                data: values,
                smooth: true,
                symbol: 'circle',
                symbolSize: 4,
                lineWidth: 2,
                connectNulls: false,  // Don't connect across null values (market closed periods)
                emphasis: {
                    focus: 'series'
                }
            });
            
            // Add market open/close markers - match new x-axis format
            const marketHours = this.marketHours[symbolInfo.market];
            if (marketHours && xAxisData.length > 0) {
                // Find matching x-axis labels for market hours
                const openHour = marketHours.open.toString().padStart(2, '0');
                const closeHour = marketHours.close.toString().padStart(2, '0');
                
                // Find x-axis entries that match market open/close hours
                const openLabel = xAxisData.find(label => label.endsWith(`${openHour}:00`));
                const closeLabel = xAxisData.find(label => label.endsWith(`${closeHour}:00`));
                
                if (openLabel) {
                    markLines.push({
                        name: `${symbolInfo.market} Open`,
                        xAxis: openLabel,
                        lineStyle: { color: '#10b981', type: 'dashed', width: 2 },
                        label: { 
                            show: true, 
                            position: 'end',
                            formatter: `${symbolInfo.market} Open`
                        }
                    });
                }
                
                if (closeLabel) {
                    markLines.push({
                        name: `${symbolInfo.market} Close`,
                        xAxis: closeLabel,
                        lineStyle: { color: '#ef4444', type: 'dashed', width: 2 },
                        label: { 
                            show: true, 
                            position: 'end',
                            formatter: `${symbolInfo.market} Close`
                        }
                    });
                }
            }
        });
        
        // Add current time indicator for rolling window - match new x-axis format
        const now = new Date();
        const currentMonth = (now.getUTCMonth() + 1).toString().padStart(2, '0');
        const currentDay = now.getUTCDate().toString().padStart(2, '0');
        const currentHour = now.getUTCHours().toString().padStart(2, '0');
        const currentTimeLabel = `${currentMonth}/${currentDay} ${currentHour}:00`;
        
        // Only add current time marker if it's within our data range
        if (xAxisData.includes(currentTimeLabel)) {
            markLines.push({
                name: 'Current Time',
                xAxis: currentTimeLabel,
                lineStyle: { color: '#f59e0b', width: 3 },
                label: { 
                    show: true, 
                    position: 'middle',
                    formatter: 'Now (UTC)'
                }
            });
        }
        
        const option = {
            title: {
                text: 'Rolling 24-Hour Market Timeline - Live Data',
                left: 'center',
                textStyle: { fontSize: 16, fontWeight: 'bold' }
            },
            tooltip: {
                trigger: 'axis',
                axisPointer: { type: 'cross' },
                formatter: function(params) {
                    let result = `<strong>${params[0].name} UTC</strong><br/>`;
                    let hasData = false;
                    
                    params.forEach(param => {
                        if (param.value !== null && param.value !== undefined) {
                            hasData = true;
                            const value = chartType === 'percentage' 
                                ? `${param.value.toFixed(2)}%`
                                : param.value.toFixed(2);
                            result += `${param.marker}${param.seriesName}: ${value}<br/>`;
                        } else {
                            result += `${param.marker}${param.seriesName}: Market Closed<br/>`;
                        }
                    });
                    
                    if (!hasData) {
                        result += '<em>All markets closed during this hour</em>';
                    }
                    
                    return result;
                }
            },
            legend: {
                top: 30,
                type: 'scroll'
            },
            grid: {
                left: '3%',
                right: '4%',
                bottom: '3%',
                top: 80,
                containLabel: true
            },
            xAxis: {
                type: 'category',
                data: xAxisData,
                name: 'UTC Time',
                nameLocation: 'middle',
                nameGap: 30
            },
            yAxis: this.getYAxisConfig(chartType, allValues),
            series: series.concat([{
                type: 'line',
                markLine: {
                    silent: true,
                    data: markLines
                }
            }])
        };
        
        this.chartInstance.setOption(option, true);
    }

    getYAxisConfig(chartType, allValues) {
        // Base configuration for y-axis
        const baseConfig = {
            type: 'value',
            axisLabel: {
                formatter: chartType === 'percentage' ? '{value}%' : '{value}'
            }
        };

        // If no data values, return basic config
        if (!allValues || allValues.length === 0) {
            return baseConfig;
        }

        // Calculate min and max values
        const minValue = Math.min(...allValues);
        const maxValue = Math.max(...allValues);
        const range = maxValue - minValue;

        if (chartType === 'percentage') {
            // For percentage charts, create a reasonable range around the data
            if (range === 0) {
                // If all values are the same, create a small range around that value
                const center = minValue;
                return {
                    ...baseConfig,
                    min: Math.max(center - 2, -10), // Don't go below -10%
                    max: Math.min(center + 2, 10)   // Don't go above +10%
                };
            } else {
                // Add padding to the range (20% on each side)
                const padding = range * 0.2;
                const suggestedMin = minValue - padding;
                const suggestedMax = maxValue + padding;
                
                // Round to nice values and ensure reasonable bounds
                const roundedMin = Math.max(Math.floor(suggestedMin), -20);
                const roundedMax = Math.min(Math.ceil(suggestedMax), 20);
                
                return {
                    ...baseConfig,
                    min: roundedMin,
                    max: roundedMax
                };
            }
        } else {
            // For price charts, let ECharts auto-scale but with some padding
            return {
                ...baseConfig,
                scale: true, // Enable smart scaling
                boundaryGap: ['5%', '5%'] // Add 5% padding on top and bottom
            };
        }
    }

    updateMarketStatusDisplay(statusData) {
        const container = document.getElementById('current-market-status');
        container.innerHTML = '';
        
        Object.entries(statusData.markets).forEach(([market, status]) => {
            const statusDiv = document.createElement('div');
            statusDiv.className = `flex items-center justify-between p-3 rounded-lg ${status.is_open ? 'bg-green-50 border-green-200' : 'bg-gray-50 border-gray-200'} border`;
            
            statusDiv.innerHTML = `
                <div class="flex items-center">
                    <div class="w-3 h-3 rounded-full mr-3 ${status.is_open ? 'bg-green-500' : 'bg-gray-400'}"></div>
                    <div>
                        <div class="font-medium text-gray-900">${market}</div>
                        <div class="text-sm text-gray-600">${status.hours_utc}</div>
                    </div>
                </div>
                <div class="text-right">
                    <div class="text-sm font-medium ${status.is_open ? 'text-green-600' : 'text-gray-600'}">
                        ${status.is_open ? 'OPEN' : 'CLOSED'}
                    </div>
                    <div class="text-xs text-gray-500">
                        ${status.next_event} ${status.next_time}
                    </div>
                </div>
            `;
            
            container.appendChild(statusDiv);
        });
        
        // Update market sessions indicator
        this.updateMarketSessionsIndicator(statusData.currently_open);
    }

    updateMarketSessionsIndicator(openMarkets) {
        const container = document.getElementById('market-status-indicators');
        container.innerHTML = openMarkets.map(market => 
            `<span class="market-open px-2 py-1 rounded-full text-xs font-medium">${market}</span>`
        ).join('');
        
        if (openMarkets.length === 0) {
            container.innerHTML = '<span class="market-closed px-2 py-1 rounded-full text-xs font-medium">All Markets Closed</span>';
        }
    }

    updateMarketHoursReference(markets) {
        const container = document.getElementById('market-hours-reference');
        container.innerHTML = '';
        
        Object.entries(markets).forEach(([market, status]) => {
            const div = document.createElement('div');
            div.className = 'flex justify-between items-center';
            div.innerHTML = `
                <span class="font-medium">${market}:</span>
                <span class="text-gray-600">${status.hours_utc}</span>
            `;
            container.appendChild(div);
        });
    }

    handleSearch(event) {
        const query = event.target.value.toLowerCase();
        if (!query) return;
        
        // Simple search implementation
        console.log(`Searching for: ${query}`);
        // Could implement live search results here
    }

    handleChartTypeChange() {
        if (this.currentData && Object.keys(this.currentData).length > 0) {
            this.updateChart(this.currentData);
        }
    }

    async refreshData() {
        await this.loadMarketStatus();
        
        if (this.selectedIndices.size > 0) {
            await this.analyzeSelectedIndices();
        }
        
        this.showToast('Data refreshed', 'success');
    }

    startAutoRefresh() {
        // Refresh every 5 minutes
        this.refreshInterval = setInterval(() => {
            this.refreshData();
        }, 5 * 60 * 1000);
    }

    showLoading(show) {
        const overlay = document.getElementById('loading-overlay');
        if (show) {
            overlay.classList.remove('hidden');
        } else {
            overlay.classList.add('hidden');
        }
    }

    showToast(message, type = 'info') {
        const container = document.getElementById('toast-container');
        const toast = document.createElement('div');
        
        const colors = {
            success: 'bg-green-500',
            error: 'bg-red-500',
            warning: 'bg-yellow-500',
            info: 'bg-blue-500'
        };
        
        toast.className = `${colors[type]} text-white px-4 py-3 rounded-lg shadow-lg transform transition-all duration-300 translate-x-full`;
        toast.innerHTML = `
            <div class="flex items-center">
                <span class="mr-2">${message}</span>
                <button onclick="this.parentElement.parentElement.remove()" class="text-white hover:text-gray-200">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;
        
        container.appendChild(toast);
        
        // Animate in
        setTimeout(() => toast.classList.remove('translate-x-full'), 100);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            if (toast.parentElement) {
                toast.classList.add('translate-x-full');
                setTimeout(() => toast.remove(), 300);
            }
        }, 5000);
    }

    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
}

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    window.tracker = new GlobalMarketTracker();
});