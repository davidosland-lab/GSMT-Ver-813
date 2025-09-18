/**
 * Mobile-Optimized Global Stock Market Tracker
 * Enhanced for mobile devices with graceful element handling
 */

class MobileGlobalMarketTracker {
    constructor() {
        this.apiBaseUrl = this.detectApiUrl();
        this.selectedIndices = new Set();
        this.allSymbols = new Map();
        this.chartInstance = null;
        this.marketHours = {};
        this.currentData = {};
        this.refreshInterval = null;
        
        // Calendar functionality
        this.selectedDate = new Date();
        this.isHistoricalMode = false;
        this.calendarVisible = false;
        this.currentMonth = new Date().getMonth();
        this.currentYear = new Date().getFullYear();
        
        this.init();
    }

    detectApiUrl() {
        // Production API URL detection for deployment
        const currentHost = window.location.hostname;
        const currentPort = window.location.port;
        
        // Production deployment on Netlify
        if (currentHost.includes('netlify.app')) {
            // Use Railway backend URL for production
            return 'https://gsmt-ver-813-production.up.railway.app/api';
        }
        
        // For sandbox environment, use the same port as current page
        if (currentHost.includes('e2b.dev')) {
            // Use the same port as the current page for the API
            return `https://${currentPort}-${currentHost.split('-').slice(1).join('-')}/api`;
        }
        
        // For localhost development - use same port as frontend
        if (currentHost === 'localhost' || currentHost === '127.0.0.1') {
            return `/api`;
        }
        
        // Use relative API path for same-server deployment
        return '/api';
    }

    async init() {
        console.log('🚀 Initializing Mobile Global Market Tracker');
        console.log('🔗 API URL detected:', this.apiBaseUrl);
        
        try {
            // Setup event listeners with safe element checking
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
            
            this.showToast('Mobile Market Tracker loaded successfully', 'success');
            
        } catch (error) {
            console.error('❌ Initialization failed:', error);
            this.showToast('Failed to connect to API. Please ensure the server is running.', 'error');
        }
    }

    setupEventListeners() {
        // Safe element selection with null checks
        const safeAddEventListener = (id, event, handler) => {
            const element = document.getElementById(id);
            if (element) {
                element.addEventListener(event, handler);
                console.log(`✅ Event listener added for ${id}`);
            } else {
                console.warn(`⚠️ Element ${id} not found, skipping event listener`);
            }
        };

        // Search functionality
        safeAddEventListener('index-search', 'input', 
            this.debounce(this.handleSearch.bind(this), 300));
        
        // Analyze button
        safeAddEventListener('analyze-btn', 'click', 
            this.analyzeSelectedIndices.bind(this));
        
        // Refresh button
        safeAddEventListener('refresh-btn', 'click', 
            this.refreshData.bind(this));
        
        // Chart type change
        safeAddEventListener('chart-type', 'change', 
            this.handleChartTypeChange.bind(this));
        
        // Plot mode change (combined vs individual)
        safeAddEventListener('plot-mode', 'change', 
            this.handlePlotModeChange.bind(this));

        // Region and market dropdown functionality
        safeAddEventListener('region-dropdown', 'change', 
            this.handleRegionChange.bind(this));
        
        safeAddEventListener('markets-dropdown', 'change', 
            this.handleMarketDropdownChange.bind(this));
        
        safeAddEventListener('add-market-btn', 'click', 
            this.addSelectedMarket.bind(this));

        // Preset buttons
        document.querySelectorAll('.preset-btn').forEach(btn => {
            btn.addEventListener('click', this.handlePresetSelection.bind(this));
        });
        
        console.log('✅ Mobile event listeners setup complete');
    }

    updateUTCClock() {
        const now = new Date();
        const utcString = now.toISOString().replace('T', ' ').substring(0, 19) + ' UTC';
        const clockElement = document.getElementById('current-utc');
        if (clockElement) {
            clockElement.textContent = utcString;
        }
    }

    async loadSymbols() {
        try {
            console.log('🔄 Loading symbols from:', `${this.apiBaseUrl}/symbols`);
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
            this.showToast('Failed to load market symbols', 'error');
        }
    }

    async loadMarketStatus() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/market-status`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            
            const data = await response.json();
            this.updateMarketStatusDisplay(data);
            
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
        // Store the indices data for dropdown population
        this.allIndicesData = suggestedIndices;
        console.log('📊 Loaded market data:', Object.keys(suggestedIndices));
        
        // Populate region dropdown
        this.populateRegionDropdown(suggestedIndices);
    }
    
    populateRegionDropdown(suggestedIndices) {
        console.log('🔽 Populating region dropdown with data:', Object.keys(suggestedIndices));
        const regionDropdown = document.getElementById('region-dropdown');
        if (!regionDropdown) {
            console.warn('❌ Region dropdown not found');
            return;
        }
        
        // Clear existing options except default
        regionDropdown.innerHTML = '<option value="">Choose a region...</option>';
        
        // Create region mappings based on suggested indices
        const regionMap = {
            'asia_pacific': '🌏 Asia Pacific',
            'europe_middle_east_africa': '🌍 Europe, Middle East & Africa', 
            'americas': '🌎 Americas',
            'major_global_stocks': '🏢 Major Global Stocks',
            'commodities_energy': '🥇 Commodities & Energy',
            'cryptocurrencies': '₿ Cryptocurrencies',
            'forex_majors': '💱 Forex Majors',
            'australian_stocks': '🇦🇺 Australian Stocks'
        };
        
        // Populate with available regions from suggested indices
        Object.keys(suggestedIndices).forEach(regionKey => {
            if (suggestedIndices[regionKey] && suggestedIndices[regionKey].length > 0) {
                const option = document.createElement('option');
                option.value = regionKey;
                option.textContent = regionMap[regionKey] || regionKey.replace(/_/g, ' ').toUpperCase();
                regionDropdown.appendChild(option);
            }
        });
        
        console.log('✅ Populated region dropdown with:', Object.keys(suggestedIndices));
    }

    handleRegionChange(event) {
        const selectedRegion = event.target.value;
        const marketsDropdown = document.getElementById('markets-dropdown');
        const addBtn = document.getElementById('add-market-btn');
        
        if (!marketsDropdown) return;
        
        if (!selectedRegion) {
            marketsDropdown.disabled = true;
            marketsDropdown.innerHTML = '<option value="">Select a region first...</option>';
            if (addBtn) addBtn.disabled = true;
            return;
        }

        // Populate markets dropdown based on selected region
        marketsDropdown.disabled = false;
        marketsDropdown.innerHTML = '<option value="">Choose a market...</option>';
        
        if (this.allIndicesData && this.allIndicesData[selectedRegion]) {
            this.allIndicesData[selectedRegion].forEach(index => {
                const option = document.createElement('option');
                option.value = index.symbol;
                option.textContent = `${index.name} (${index.symbol})`;
                option.dataset.marketInfo = JSON.stringify(index);
                marketsDropdown.appendChild(option);
            });
            
            console.log(`✅ Populated ${this.allIndicesData[selectedRegion].length} markets for region: ${selectedRegion}`);
        }
    }

    handleMarketDropdownChange(event) {
        const addBtn = document.getElementById('add-market-btn');
        if (addBtn) {
            addBtn.disabled = !event.target.value;
        }
    }

    addSelectedMarket() {
        const regionDropdown = document.getElementById('region-dropdown');
        const marketsDropdown = document.getElementById('markets-dropdown');
        
        if (!marketsDropdown || !marketsDropdown.value) return;

        const symbol = marketsDropdown.value;
        const marketInfo = JSON.parse(marketsDropdown.selectedOptions[0].dataset.marketInfo);
        
        // Add to selected indices
        if (!this.selectedIndices.has(symbol)) {
            this.selectedIndices.add(symbol);
            this.updateSelectedMarketsDisplay();
            this.showToast(`Added ${marketInfo.name}`, 'success');
        } else {
            this.showToast(`${marketInfo.name} already selected`, 'warning');
        }

        // Enable analyze button if markets are selected
        const analyzeBtn = document.getElementById('analyze-btn');
        if (analyzeBtn) {
            analyzeBtn.disabled = this.selectedIndices.size === 0;
        }
    }

    updateSelectedMarketsDisplay() {
        const container = document.getElementById('selected-markets');
        const counter = document.getElementById('selected-count');
        
        if (!container) return;
        
        if (counter) {
            counter.textContent = this.selectedIndices.size;
        }

        if (this.selectedIndices.size === 0) {
            container.innerHTML = '<p class="text-gray-500 text-sm text-center py-4">No markets selected yet</p>';
            return;
        }

        container.innerHTML = '';
        this.selectedIndices.forEach(symbol => {
            const symbolData = this.allSymbols.get(symbol) || { name: symbol, symbol: symbol };
            
            const marketCard = document.createElement('div');
            marketCard.className = 'flex items-center justify-between p-2 bg-gray-50 rounded-lg';
            marketCard.innerHTML = `
                <div class="flex-1">
                    <span class="font-medium text-sm">${symbolData.name || symbol}</span>
                    <div class="text-xs text-gray-500">${symbol}</div>
                </div>
                <button class="text-red-500 hover:text-red-700 p-1" onclick="window.mobileTracker.removeSelectedMarket('${symbol}')">
                    <i class="fas fa-times"></i>
                </button>
            `;
            container.appendChild(marketCard);
        });
    }

    removeSelectedMarket(symbol) {
        this.selectedIndices.delete(symbol);
        this.updateSelectedMarketsDisplay();
        
        const analyzeBtn = document.getElementById('analyze-btn');
        if (analyzeBtn) {
            analyzeBtn.disabled = this.selectedIndices.size === 0;
        }
        
        this.showToast('Market removed', 'info');
    }

    handlePresetSelection(event) {
        const preset = event.target.dataset.preset;
        console.log('🎯 Preset selection:', preset);
        
        // Clear current selection
        this.selectedIndices.clear();
        
        // Add preset markets based on type
        const presetMappings = {
            'asian-indices': ['^N225', '^HSI', '^AXJO', '^AORD', '^KS11'],
            'european-indices': ['^FTSE', '^GDAXI', '^FCHI', '^AEX', '^IBEX'],
            'us-indices': ['^GSPC', '^IXIC', '^DJI', '^RUT', '^VIX'],
            'australian-indices': ['^AXJO', '^AORD', 'CBA.AX', 'BHP.AX', 'CSL.AX'],
            'crypto-major': ['BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD'],
            'commodities': ['GC=F', 'CL=F', 'SI=F', 'NG=F']
        };
        
        const symbols = presetMappings[preset] || [];
        symbols.forEach(symbol => {
            if (this.allSymbols.has(symbol)) {
                this.selectedIndices.add(symbol);
            }
        });
        
        this.updateSelectedMarketsDisplay();
        
        // Enable analyze button
        const analyzeBtn = document.getElementById('analyze-btn');
        if (analyzeBtn) {
            analyzeBtn.disabled = this.selectedIndices.size === 0;
        }
        
        this.showToast(`${preset.replace('-', ' ')} preset loaded (${this.selectedIndices.size} markets)`, 'success');
    }

    handleSearch(event) {
        const query = event.target.value.toLowerCase();
        console.log('🔍 Search query:', query);
        
        // Implement search functionality here
        // For now, just log the search
        if (query.length > 2) {
            console.log('Searching for:', query);
        }
    }

    async analyzeSelectedIndices() {
        if (this.selectedIndices.size === 0) {
            this.showToast('Please select markets to analyze', 'warning');
            return;
        }

        console.log('📈 Analyzing selected indices:', [...this.selectedIndices]);
        
        try {
            await this.loadChartData();
            this.showToast('Chart updated with selected markets', 'success');
        } catch (error) {
            console.error('Analysis error:', error);
            this.showToast('Failed to load chart data', 'error');
        }
    }

    async refreshData() {
        this.showToast('Refreshing market data...', 'info');
        
        try {
            await this.loadSymbols();
            await this.loadMarketStatus();
            await this.loadSuggestedIndices();
            
            if (this.selectedIndices.size > 0) {
                await this.loadChartData();
            }
            
            this.showToast('Data refreshed successfully', 'success');
        } catch (error) {
            console.error('Refresh error:', error);
            this.showToast('Failed to refresh data', 'error');
        }
    }

    initializeChart() {
        const chartContainer = document.getElementById('main-chart');
        if (!chartContainer) {
            console.warn('Chart container not found');
            return;
        }

        this.chartInstance = echarts.init(chartContainer);
        
        // Initial empty chart
        const option = {
            title: { text: 'Select markets to view data', left: 'center' },
            grid: { left: '10%', right: '10%', top: '15%', bottom: '15%' },
            xAxis: { type: 'category', data: [] },
            yAxis: { type: 'value' },
            series: []
        };
        
        this.chartInstance.setOption(option);
        console.log('✅ Chart initialized');
    }

    async loadChartData() {
        if (!this.chartInstance || this.selectedIndices.size === 0) return;

        try {
            const symbols = [...this.selectedIndices].join(',');
            const response = await fetch(`${this.apiBaseUrl}/market-data?symbols=${symbols}&days=1`);
            
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            
            const data = await response.json();
            this.renderChart(data);
            
        } catch (error) {
            console.error('Failed to load chart data:', error);
            this.showToast('Failed to load chart data', 'error');
        }
    }

    renderChart(data) {
        if (!this.chartInstance || !data) return;

        // Process chart data and render
        const option = {
            title: { text: 'Market Data', left: 'center' },
            tooltip: { trigger: 'axis' },
            legend: { data: Object.keys(data), bottom: 0 },
            grid: { left: '3%', right: '4%', bottom: '20%', containLabel: true },
            xAxis: { type: 'category', data: data.timestamps || [] },
            yAxis: { type: 'value' },
            series: Object.entries(data).filter(([key]) => key !== 'timestamps').map(([symbol, values]) => ({
                name: symbol,
                type: 'line',
                data: values,
                smooth: true
            }))
        };
        
        this.chartInstance.setOption(option);
        console.log('✅ Chart rendered with data');
    }

    handleChartTypeChange(event) {
        console.log('Chart type changed:', event.target.value);
        // Implement chart type change
    }

    handlePlotModeChange(event) {
        console.log('Plot mode changed:', event.target.value);
        // Implement plot mode change
    }

    updateMarketStatusDisplay(data) {
        // Update market ticker if available
        if (window.mobileInterface && data.markets) {
            window.mobileInterface.updateTicker(data.markets);
        }
    }

    startAutoRefresh() {
        // Refresh every 5 minutes
        this.refreshInterval = setInterval(() => {
            this.refreshData();
        }, 5 * 60 * 1000);
        console.log('✅ Auto-refresh started (5 minutes)');
    }

    showToast(message, type = 'info') {
        if (window.mobileInterface) {
            window.mobileInterface.showToast(message, type);
        } else {
            console.log(`Toast (${type}):`, message);
        }
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

// Initialize mobile tracker when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Mobile DOM loaded, initializing tracker...');
    window.mobileTracker = new MobileGlobalMarketTracker();
});