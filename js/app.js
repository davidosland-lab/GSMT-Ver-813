/**
 * GSMT Ver 6.0 - Modern Frontend Application
 * Global Stock Market Tracker with Percentage Analysis
 * 
 * Features:
 * - Modern state management
 * - Percentage-based analysis
 * - Unified overlay charts
 * - Real-time data updates
 * - Professional error handling
 */

class GSMTApplication {
    constructor() {
        // Application state
        this.state = {
            apiBaseUrl: this.getApiUrl(),
            selectedSymbols: new Set(),
            chartData: new Map(),
            chartInstance: null,
            settings: this.loadSettings(),
            isLoading: false,
            lastUpdateTime: null
        };
        
        // Available symbols cache
        this.symbolsCache = new Map();
        this.searchCache = new Map();
        
        // Auto-refresh timer
        this.refreshTimer = null;
        
        // Initialize application
        this.init();
    }
    
    /**
     * Initialize the application
     */
    async init() {
        console.log('🚀 Initializing GSMT Ver 6.0');
        
        try {
            // Setup event listeners
            this.setupEventListeners();
            
            // Initialize chart
            this.initializeChart();
            
            // Check API connection
            await this.checkApiConnection();
            
            // Load available symbols
            await this.loadSymbols();
            
            // Apply saved settings
            this.applySettings();
            
            console.log('✅ GSMT Ver 6.0 initialized successfully');
            this.showToast('GSMT Ver 6.0 ready for analysis', 'success');
            
        } catch (error) {
            console.error('❌ Initialization failed:', error);
            this.showToast('Failed to initialize application', 'error');
            this.updateApiStatus('error', 'Connection failed');
        }
    }
    
    /**
     * Setup all event listeners
     */
    setupEventListeners() {
        // Symbol search
        const symbolSearch = document.getElementById('symbol-search');
        symbolSearch.addEventListener('input', this.debounce(this.handleSymbolSearch.bind(this), 300));
        symbolSearch.addEventListener('keydown', this.handleSymbolKeydown.bind(this));
        
        // Analysis button
        document.getElementById('analyze-btn').addEventListener('click', this.handleAnalyze.bind(this));
        
        // Clear button
        document.getElementById('clear-btn').addEventListener('click', this.handleClear.bind(this));
        
        // Chart type change
        document.getElementById('chart-type').addEventListener('change', this.handleChartTypeChange.bind(this));
        
        // Time period change
        document.getElementById('time-period').addEventListener('change', this.handleTimePeriodChange.bind(this));
        
        // Settings modal
        document.getElementById('settings-btn').addEventListener('click', this.showSettings.bind(this));
        document.getElementById('close-settings').addEventListener('click', this.hideSettings.bind(this));
        document.getElementById('save-settings').addEventListener('click', this.saveSettings.bind(this));
        document.getElementById('cancel-settings').addEventListener('click', this.hideSettings.bind(this));
        
        // Fullscreen chart
        document.getElementById('fullscreen-btn').addEventListener('click', this.toggleFullscreen.bind(this));
        
        // Click outside to close dropdowns
        document.addEventListener('click', this.handleDocumentClick.bind(this));
        
        // Window resize
        window.addEventListener('resize', this.debounce(this.handleResize.bind(this), 250));
    }
    
    /**
     * Initialize the main chart
     */
    initializeChart() {
        const chartContainer = document.getElementById('main-chart');
        this.state.chartInstance = echarts.init(chartContainer, 'light', {
            renderer: 'canvas',
            useDirtyRect: true
        });
        
        this.state.chartInstance.setOption(this.getEmptyChartOption());
    }
    
    /**
     * Check API connection and update status
     */
    async checkApiConnection() {
        try {
            const response = await fetch(`${this.state.apiBaseUrl}/health`);
            const data = await response.json();
            
            if (response.ok) {
                this.updateApiStatus('connected', `v${data.version}`);
                return true;
            } else {
                throw new Error('API returned error status');
            }
        } catch (error) {
            console.error('API connection failed:', error);
            this.updateApiStatus('error', 'Connection failed');
            throw error;
        }
    }
    
    /**
     * Load available symbols from API
     */
    async loadSymbols() {
        try {
            const response = await fetch(`${this.state.apiBaseUrl}/api/symbols`);
            const data = await response.json();
            
            if (response.ok) {
                // Store symbols by category
                for (const [category, symbols] of Object.entries(data.markets)) {
                    for (const [symbol, info] of Object.entries(symbols)) {
                        this.symbolsCache.set(symbol, {
                            ...info,
                            category: category
                        });
                    }
                }
                
                console.log(`📊 Loaded ${this.symbolsCache.size} symbols from API`);
            } else {
                throw new Error('Failed to load symbols');
            }
        } catch (error) {
            console.error('Failed to load symbols:', error);
            // Load fallback symbols
            this.loadFallbackSymbols();
        }
    }
    
    /**
     * Load fallback symbols when API is unavailable
     */
    loadFallbackSymbols() {
        const fallbackSymbols = {
            '^GSPC': { symbol: '^GSPC', name: 'S&P 500', category: 'indices', market: 'usa' },
            '^IXIC': { symbol: '^IXIC', name: 'NASDAQ', category: 'indices', market: 'usa' },
            '^AXJO': { symbol: '^AXJO', name: 'ASX 200', category: 'indices', market: 'australia' },
            'AAPL': { symbol: 'AAPL', name: 'Apple Inc.', category: 'us_stocks', market: 'usa' },
            'CBA.AX': { symbol: 'CBA.AX', name: 'Commonwealth Bank', category: 'australian_stocks', market: 'australia' }
        };
        
        for (const [symbol, info] of Object.entries(fallbackSymbols)) {
            this.symbolsCache.set(symbol, info);
        }
        
        console.log('📊 Loaded fallback symbols');
    }
    
    /**
     * Handle symbol search with suggestions
     */
    async handleSymbolSearch(event) {
        const query = event.target.value.trim();
        const suggestionsContainer = document.getElementById('symbol-suggestions');
        
        if (query.length < 1) {
            suggestionsContainer.classList.add('hidden');
            return;
        }
        
        try {
            let suggestions = [];
            
            // Try API search first
            if (this.state.apiBaseUrl) {
                try {
                    const response = await fetch(`${this.state.apiBaseUrl}/api/search/${encodeURIComponent(query)}`);
                    if (response.ok) {
                        const data = await response.json();
                        suggestions = data.results.slice(0, 8);
                    }
                } catch (error) {
                    console.warn('API search failed, using local search');
                }
            }
            
            // Fallback to local search
            if (suggestions.length === 0) {
                suggestions = this.searchSymbolsLocally(query);
            }
            
            this.displaySuggestions(suggestions);
            
        } catch (error) {
            console.error('Search error:', error);
            suggestionsContainer.classList.add('hidden');
        }
    }
    
    /**
     * Search symbols locally
     */
    searchSymbolsLocally(query) {
        const results = [];
        const queryLower = query.toLowerCase();
        
        for (const [symbol, info] of this.symbolsCache.entries()) {
            if (symbol.toLowerCase().includes(queryLower) || 
                info.name.toLowerCase().includes(queryLower)) {
                results.push({
                    symbol: symbol,
                    name: info.name,
                    category: info.category,
                    market: info.market
                });
                
                if (results.length >= 8) break;
            }
        }
        
        return results;
    }
    
    /**
     * Display search suggestions
     */
    displaySuggestions(suggestions) {
        const container = document.getElementById('symbol-suggestions');
        
        if (suggestions.length === 0) {
            container.innerHTML = '<div class="p-3 text-sm text-gray-500">No symbols found</div>';
        } else {
            container.innerHTML = suggestions.map(suggestion => `
                <div class="suggestion-item p-3 hover:bg-gray-50 cursor-pointer border-b border-gray-100 last:border-b-0" 
                     data-symbol="${suggestion.symbol}">
                    <div class="flex justify-between items-center">
                        <div>
                            <div class="font-medium text-gray-900">${suggestion.symbol}</div>
                            <div class="text-sm text-gray-600">${suggestion.name}</div>
                        </div>
                        <div class="text-xs text-gray-500 capitalize">${suggestion.category?.replace('_', ' ')}</div>
                    </div>
                </div>
            `).join('');
            
            // Add click listeners to suggestions
            container.querySelectorAll('.suggestion-item').forEach(item => {
                item.addEventListener('click', this.handleSymbolSelection.bind(this));
            });
        }
        
        container.classList.remove('hidden');
    }
    
    /**
     * Handle symbol selection from suggestions
     */
    handleSymbolSelection(event) {
        const symbol = event.currentTarget.dataset.symbol;
        this.addSymbol(symbol);
        
        // Clear search and hide suggestions
        document.getElementById('symbol-search').value = '';
        document.getElementById('symbol-suggestions').classList.add('hidden');
    }
    
    /**
     * Handle keyboard navigation in symbol search
     */
    handleSymbolKeydown(event) {
        if (event.key === 'Enter') {
            const suggestions = document.querySelectorAll('.suggestion-item');
            if (suggestions.length > 0) {
                suggestions[0].click();
            }
        } else if (event.key === 'Escape') {
            document.getElementById('symbol-suggestions').classList.add('hidden');
        }
    }
    
    /**
     * Add symbol to selection
     */
    addSymbol(symbol) {
        if (this.state.selectedSymbols.has(symbol)) {
            this.showToast(`${symbol} is already selected`, 'warning');
            return;
        }
        
        if (this.state.selectedSymbols.size >= 10) {
            this.showToast('Maximum 10 symbols allowed', 'warning');
            return;
        }
        
        this.state.selectedSymbols.add(symbol);
        this.updateSelectedSymbolsDisplay();
        this.showToast(`Added ${symbol} to analysis`, 'success');
    }
    
    /**
     * Remove symbol from selection
     */
    removeSymbol(symbol) {
        this.state.selectedSymbols.delete(symbol);
        this.state.chartData.delete(symbol);
        this.updateSelectedSymbolsDisplay();
        this.updateChart();
        this.showToast(`Removed ${symbol} from analysis`, 'info');
    }
    
    /**
     * Update selected symbols display
     */
    updateSelectedSymbolsDisplay() {
        const container = document.getElementById('selected-symbols');
        const chipsContainer = document.getElementById('symbol-chips');
        const countElement = document.getElementById('symbol-count');
        
        if (this.state.selectedSymbols.size === 0) {
            container.classList.add('hidden');
            return;
        }
        
        container.classList.remove('hidden');
        countElement.textContent = `${this.state.selectedSymbols.size} symbol${this.state.selectedSymbols.size > 1 ? 's' : ''}`;
        
        chipsContainer.innerHTML = Array.from(this.state.selectedSymbols).map(symbol => {
            const info = this.symbolsCache.get(symbol) || { name: symbol };
            return `
                <div class="flex items-center bg-primary-100 text-primary-800 px-3 py-1 rounded-full text-sm">
                    <span class="font-medium">${symbol}</span>
                    <span class="ml-2 text-primary-600">•</span>
                    <span class="ml-1 text-primary-700">${info.name}</span>
                    <button class="ml-2 text-primary-600 hover:text-primary-800" onclick="app.removeSymbol('${symbol}')">
                        <i class="fas fa-times text-xs"></i>
                    </button>
                </div>
            `;
        }).join('');
    }
    
    /**
     * Handle analyze button click
     */
    async handleAnalyze() {
        if (this.state.selectedSymbols.size === 0) {
            this.showToast('Please select at least one symbol', 'warning');
            return;
        }
        
        this.setLoading(true);
        
        try {
            const period = document.getElementById('time-period').value;
            const symbols = Array.from(this.state.selectedSymbols);
            
            // Fetch data for all selected symbols
            const bulkRequest = {
                symbols: symbols,
                period: period
            };
            
            const response = await fetch(`${this.state.apiBaseUrl}/api/bulk`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(bulkRequest)
            });
            
            if (!response.ok) {
                throw new Error(`API error: ${response.status}`);
            }
            
            const data = await response.json();
            
            // Store chart data
            this.state.chartData.clear();
            for (const [symbol, symbolData] of Object.entries(data.results)) {
                if (symbolData.success) {
                    this.state.chartData.set(symbol, symbolData);
                }
            }
            
            // Update chart
            this.updateChart();
            
            // Update performance summary
            this.updatePerformanceSummary();
            
            // Update timestamp
            this.state.lastUpdateTime = new Date();
            
            this.showToast(`Analysis complete: ${data.success_rate} success rate`, 'success');
            
        } catch (error) {
            console.error('Analysis failed:', error);
            this.showToast('Analysis failed: ' + error.message, 'error');
        } finally {
            this.setLoading(false);
        }
    }
    
    /**
     * Update the main chart with current data
     */
    updateChart() {
        if (!this.state.chartInstance || this.state.chartData.size === 0) {
            this.state.chartInstance?.setOption(this.getEmptyChartOption());
            return;
        }
        
        const chartType = document.getElementById('chart-type').value;
        const option = this.generateChartOption(chartType);
        
        this.state.chartInstance.setOption(option, true);
    }
    
    /**
     * Generate chart option based on type
     */
    generateChartOption(chartType) {
        const series = [];
        const colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#f97316', '#06b6d4', '#84cc16'];
        let colorIndex = 0;
        
        for (const [symbol, data] of this.state.chartData.entries()) {
            if (chartType === 'percentage') {
                series.push({
                    name: symbol,
                    type: 'line',
                    data: data.data.map(point => [point.timestamp_raw, point.percentage_change]),
                    smooth: true,
                    symbol: 'none',
                    lineStyle: { width: 2 },
                    color: colors[colorIndex % colors.length]
                });
            } else if (chartType === 'price') {
                series.push({
                    name: symbol,
                    type: 'line',
                    data: data.data.map(point => [point.timestamp_raw, point.close]),
                    smooth: true,
                    symbol: 'none',
                    lineStyle: { width: 2 },
                    color: colors[colorIndex % colors.length]
                });
            } else if (chartType === 'candlestick') {
                // For candlestick, only show the first symbol to avoid cluttering
                if (colorIndex === 0) {
                    series.push({
                        name: symbol,
                        type: 'candlestick',
                        data: data.data.map(point => [
                            point.timestamp_raw,
                            point.open,
                            point.close,
                            point.low,
                            point.high
                        ]),
                        itemStyle: {
                            color: '#10b981',
                            color0: '#ef4444',
                            borderColor: '#10b981',
                            borderColor0: '#ef4444'
                        }
                    });
                }
            }
            colorIndex++;
        }
        
        return {
            title: {
                text: `${chartType === 'percentage' ? 'Percentage Change Analysis' : chartType === 'price' ? 'Price Analysis' : 'Candlestick Chart'}`,
                left: 'center',
                textStyle: { fontSize: 18, fontWeight: 'bold' }
            },
            tooltip: {
                trigger: 'axis',
                axisPointer: { type: 'cross', label: { backgroundColor: '#6a7985' } },
                formatter: function(params) {
                    if (!params || params.length === 0) return '';
                    
                    const time = new Date(params[0].value[0]).toLocaleString();
                    let content = `<div class="font-semibold">${time}</div>`;
                    
                    params.forEach(param => {
                        const value = param.value[1];
                        const color = param.color;
                        const symbol = param.seriesName;
                        
                        if (chartType === 'percentage') {
                            content += `<div style="color: ${color}">
                                <span class="font-medium">${symbol}:</span> 
                                <span class="font-bold">${value > 0 ? '+' : ''}${value.toFixed(2)}%</span>
                            </div>`;
                        } else {
                            content += `<div style="color: ${color}">
                                <span class="font-medium">${symbol}:</span> 
                                <span class="font-bold">$${value.toFixed(2)}</span>
                            </div>`;
                        }
                    });
                    
                    return content;
                }
            },
            legend: {
                top: 40,
                type: 'scroll'
            },
            grid: {
                left: '3%',
                right: '4%',
                bottom: '3%',
                top: '80px',
                containLabel: true
            },
            xAxis: {
                type: 'time',
                axisLine: { lineStyle: { color: '#ccc' } },
                axisLabel: { color: '#666' }
            },
            yAxis: {
                type: 'value',
                axisLine: { lineStyle: { color: '#ccc' } },
                axisLabel: { 
                    color: '#666',
                    formatter: chartType === 'percentage' ? '{value}%' : '${value}'
                },
                splitLine: { lineStyle: { color: '#eee' } }
            },
            series: series,
            animation: true,
            animationDuration: 1000
        };
    }
    
    /**
     * Get empty chart option
     */
    getEmptyChartOption() {
        return {
            title: {
                text: 'Select symbols to begin analysis',
                left: 'center',
                top: 'middle',
                textStyle: { fontSize: 18, color: '#9ca3af' }
            },
            grid: { show: false },
            xAxis: { show: false },
            yAxis: { show: false }
        };
    }
    
    /**
     * Update performance summary
     */
    updatePerformanceSummary() {
        const container = document.getElementById('performance-summary');
        const grid = document.getElementById('performance-grid');
        
        if (this.state.chartData.size === 0) {
            container.classList.add('hidden');
            return;
        }
        
        container.classList.remove('hidden');
        
        const performanceCards = Array.from(this.state.chartData.entries()).map(([symbol, data]) => {
            const latestChange = data.current_change;
            const isPositive = latestChange >= 0;
            const info = this.symbolsCache.get(symbol) || { name: symbol };
            
            return `
                <div class="bg-gray-50 rounded-lg p-4">
                    <div class="flex items-center justify-between mb-2">
                        <div class="font-medium text-gray-900">${symbol}</div>
                        <div class="text-xs text-gray-500 capitalize">${info.category?.replace('_', ' ') || 'Unknown'}</div>
                    </div>
                    <div class="text-sm text-gray-600 mb-2">${info.name}</div>
                    <div class="flex items-center">
                        <span class="text-2xl font-bold ${isPositive ? 'text-success-600' : 'text-danger-600'}">
                            ${isPositive ? '+' : ''}${latestChange.toFixed(2)}%
                        </span>
                        <i class="fas fa-arrow-${isPositive ? 'up' : 'down'} ml-2 ${isPositive ? 'text-success-500' : 'text-danger-500'}"></i>
                    </div>
                </div>
            `;
        }).join('');
        
        grid.innerHTML = performanceCards;
    }
    
    /**
     * Handle clear button
     */
    handleClear() {
        this.state.selectedSymbols.clear();
        this.state.chartData.clear();
        this.updateSelectedSymbolsDisplay();
        this.updateChart();
        document.getElementById('performance-summary').classList.add('hidden');
        this.showToast('Analysis cleared', 'info');
    }
    
    /**
     * Handle chart type change
     */
    handleChartTypeChange() {
        if (this.state.chartData.size > 0) {
            this.updateChart();
        }
    }
    
    /**
     * Handle time period change
     */
    handleTimePeriodChange() {
        if (this.state.selectedSymbols.size > 0) {
            this.showToast('Time period changed. Click Analyze to update data.', 'info');
        }
    }
    
    /**
     * Set loading state
     */
    setLoading(loading) {
        this.state.isLoading = loading;
        const indicator = document.getElementById('loading-indicator');
        const button = document.getElementById('analyze-btn');
        
        if (loading) {
            indicator.classList.remove('hidden');
            button.disabled = true;
            button.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Analyzing...';
        } else {
            indicator.classList.add('hidden');
            button.disabled = false;
            button.innerHTML = '<i class="fas fa-analytics mr-2"></i>Analyze';
        }
    }
    
    /**
     * Update API status indicator
     */
    updateApiStatus(status, message) {
        const statusElement = document.getElementById('api-status');
        const dot = statusElement.querySelector('div');
        const text = statusElement.querySelector('span');
        
        dot.className = 'w-2 h-2 rounded-full';
        
        switch (status) {
            case 'connected':
                dot.classList.add('bg-success-500');
                text.textContent = `Connected ${message}`;
                text.className = 'text-xs text-success-600';
                break;
            case 'error':
                dot.classList.add('bg-danger-500');
                text.textContent = message;
                text.className = 'text-xs text-danger-600';
                break;
            default:
                dot.classList.add('bg-gray-400', 'animate-pulse');
                text.textContent = message;
                text.className = 'text-xs text-gray-500';
        }
    }
    
    /**
     * Show settings modal
     */
    showSettings() {
        document.getElementById('settings-modal').classList.remove('hidden');
        document.getElementById('api-url').value = this.state.apiBaseUrl;
        document.getElementById('auto-refresh').checked = this.state.settings.autoRefresh;
        document.getElementById('refresh-interval').value = this.state.settings.refreshInterval;
    }
    
    /**
     * Hide settings modal
     */
    hideSettings() {
        document.getElementById('settings-modal').classList.add('hidden');
    }
    
    /**
     * Save settings
     */
    saveSettings() {
        const apiUrl = document.getElementById('api-url').value.trim();
        const autoRefresh = document.getElementById('auto-refresh').checked;
        const refreshInterval = parseInt(document.getElementById('refresh-interval').value);
        
        this.state.apiBaseUrl = apiUrl || this.getApiUrl();
        this.state.settings.autoRefresh = autoRefresh;
        this.state.settings.refreshInterval = refreshInterval;
        
        localStorage.setItem('gsmt-settings', JSON.stringify(this.state.settings));
        localStorage.setItem('gsmt-api-url', this.state.apiBaseUrl);
        
        this.applySettings();
        this.hideSettings();
        this.showToast('Settings saved successfully', 'success');
    }
    
    /**
     * Apply current settings
     */
    applySettings() {
        // Setup auto-refresh
        if (this.refreshTimer) {
            clearInterval(this.refreshTimer);
            this.refreshTimer = null;
        }
        
        if (this.state.settings.autoRefresh && this.state.selectedSymbols.size > 0) {
            this.refreshTimer = setInterval(() => {
                this.handleAnalyze();
            }, this.state.settings.refreshInterval * 1000);
        }
    }
    
    /**
     * Load settings from localStorage
     */
    loadSettings() {
        const defaultSettings = {
            autoRefresh: false,
            refreshInterval: 300
        };
        
        try {
            const saved = localStorage.getItem('gsmt-settings');
            return saved ? { ...defaultSettings, ...JSON.parse(saved) } : defaultSettings;
        } catch (error) {
            return defaultSettings;
        }
    }
    
    /**
     * Get API URL from environment or default
     */
    getApiUrl() {
        // Check for environment variable first
        if (window.REACT_APP_API_URL || process.env.REACT_APP_API_URL) {
            return window.REACT_APP_API_URL || process.env.REACT_APP_API_URL;
        }
        
        // Check localStorage
        const saved = localStorage.getItem('gsmt-api-url');
        if (saved) return saved;
        
        // Try to detect if we're in development or production
        if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
            return 'http://localhost:8000';
        }
        
        // For production, assume API is on the same domain
        return window.location.origin;
    }
    
    /**
     * Handle document clicks for closing dropdowns
     */
    handleDocumentClick(event) {
        if (!event.target.closest('#symbol-search') && !event.target.closest('#symbol-suggestions')) {
            document.getElementById('symbol-suggestions').classList.add('hidden');
        }
    }
    
    /**
     * Handle window resize
     */
    handleResize() {
        if (this.state.chartInstance) {
            this.state.chartInstance.resize();
        }
    }
    
    /**
     * Toggle fullscreen chart
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
     * Show toast notification
     */
    showToast(message, type = 'info') {
        const container = document.getElementById('toast-container');
        const id = Date.now();
        
        const colors = {
            success: 'bg-success-500',
            error: 'bg-danger-500',
            warning: 'bg-yellow-500',
            info: 'bg-primary-500'
        };
        
        const icons = {
            success: 'fa-check-circle',
            error: 'fa-exclamation-circle',
            warning: 'fa-exclamation-triangle',
            info: 'fa-info-circle'
        };
        
        const toast = document.createElement('div');
        toast.id = `toast-${id}`;
        toast.className = `flex items-center p-4 rounded-lg shadow-lg text-white ${colors[type]} transform transition-all duration-300 translate-x-full`;
        toast.innerHTML = `
            <i class="fas ${icons[type]} mr-3"></i>
            <span class="flex-1">${message}</span>
            <button onclick="this.parentElement.remove()" class="ml-3 text-white hover:text-gray-200">
                <i class="fas fa-times"></i>
            </button>
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
    
    /**
     * Debounce function for performance
     */
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
document.addEventListener('DOMContentLoaded', () => {
    window.app = new GSMTApplication();
});

// Export for use in HTML
window.GSMTApplication = GSMTApplication;