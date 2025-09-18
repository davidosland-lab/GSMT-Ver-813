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
        const currentHref = window.location.href;
        
        console.log('🔍 URL Detection Debug:');
        console.log('  - hostname:', currentHost);
        console.log('  - port:', currentPort);
        console.log('  - href:', currentHref);
        
        // Production deployment on Netlify
        if (currentHost.includes('netlify.app')) {
            // Use Railway backend URL for production
            return 'https://gsmt-ver-813-production.up.railway.app/api';
        }
        
        // For sandbox environment, use the same port as current page
        if (currentHost.includes('e2b.dev')) {
            // Extract port from hostname if currentPort is empty
            let port = currentPort;
            if (!port || port === '') {
                const portMatch = currentHost.match(/^(\d+)-/);
                port = portMatch ? portMatch[1] : '8080';
                console.log('  - extracted port from hostname:', port);
            }
            
            // Rebuild the URL correctly
            const hostWithoutPort = currentHost.replace(/^\d+-/, '');
            const apiUrl = `https://${port}-${hostWithoutPort}/api`;
            console.log('  - generated API URL:', apiUrl);
            return apiUrl;
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
            
            // Add resize handler for mobile orientation changes
            this.setupResizeHandler();
            
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
            
            // Load demo chart data
            this.loadDemoChart();
            
            // Auto-select some popular markets for testing (optional)
            setTimeout(() => this.autoSelectTestMarkets(), 3000);
            
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
            console.warn('❌ Chart container not found');
            return;
        }

        // Debug container dimensions
        const containerRect = chartContainer.getBoundingClientRect();
        console.log('📏 Chart container dimensions:', {
            width: containerRect.width,
            height: containerRect.height,
            visible: containerRect.width > 0 && containerRect.height > 0
        });

        // Ensure container has minimum dimensions
        if (containerRect.width === 0 || containerRect.height === 0) {
            console.warn('⚠️ Chart container has zero dimensions, setting fallback size');
            chartContainer.style.width = '100%';
            chartContainer.style.height = '280px';
            chartContainer.style.minHeight = '280px';
        }

        // Initialize ECharts with explicit sizing
        this.chartInstance = echarts.init(chartContainer, null, {
            width: chartContainer.offsetWidth || 350,
            height: chartContainer.offsetHeight || 280
        });
        
        // Initial empty chart with better mobile styling
        const option = {
            title: { 
                text: 'Select Australian markets to view data', 
                left: 'center',
                textStyle: { fontSize: 14, color: '#333' }
            },
            grid: { 
                left: '8%', 
                right: '8%', 
                top: '20%', 
                bottom: '20%',
                containLabel: true
            },
            xAxis: { 
                type: 'category', 
                data: [],
                axisLabel: { fontSize: 10 }
            },
            yAxis: { 
                type: 'value',
                axisLabel: { fontSize: 10 }
            },
            series: [],
            backgroundColor: '#ffffff'
        };
        
        this.chartInstance.setOption(option);
        
        // Force resize after initialization
        setTimeout(() => {
            if (this.chartInstance) {
                this.chartInstance.resize();
                console.log('🔄 Chart resized after initialization');
            }
        }, 100);
        
        console.log('✅ Chart initialized with dimensions:', chartContainer.offsetWidth, 'x', chartContainer.offsetHeight);
    }

    setupResizeHandler() {
        // Handle window resize and orientation change for mobile
        let resizeTimeout;
        const handleResize = () => {
            clearTimeout(resizeTimeout);
            resizeTimeout = setTimeout(() => {
                if (this.chartInstance) {
                    console.log('📱 Handling resize/orientation change');
                    const container = document.getElementById('main-chart');
                    if (container) {
                        // Force container to refresh its dimensions
                        container.style.width = '100%';
                        container.style.height = '280px';
                        
                        // Resize ECharts instance
                        this.chartInstance.resize({
                            width: container.offsetWidth,
                            height: container.offsetHeight
                        });
                        
                        console.log('🔄 Chart resized to:', container.offsetWidth, 'x', container.offsetHeight);
                    }
                }
            }, 300);
        };
        
        // Listen to multiple resize events
        window.addEventListener('resize', handleResize);
        window.addEventListener('orientationchange', handleResize);
        
        // Also listen for visibility changes (tab switching)
        document.addEventListener('visibilitychange', () => {
            if (!document.hidden && this.chartInstance) {
                setTimeout(() => {
                    this.chartInstance.resize();
                    console.log('👁️ Chart resized after visibility change');
                }, 200);
            }
        });
    }

    async loadChartData() {
        if (!this.chartInstance || this.selectedIndices.size === 0) {
            console.log('⏭️ No chart instance or selected indices, skipping chart data load');
            return;
        }

        console.log('📈 Loading chart data for selected indices:', [...this.selectedIndices]);
        this.showLoadingChart();

        try {
            const symbols = [...this.selectedIndices];
            const chartType = document.getElementById('chart-type')?.value || 'percentage';
            
            // Use the correct analyze endpoint
            const requestBody = {
                symbols: symbols,
                chart_type: chartType,
                interval_minutes: 60,
                time_period: "24h"
            };
            
            console.log('📊 Requesting chart data:', requestBody);
            
            const response = await fetch(`${this.apiBaseUrl}/analyze`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestBody)
            });
            
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`HTTP ${response.status}: ${errorText}`);
            }
            
            const data = await response.json();
            console.log('✅ Chart data received:', Object.keys(data.data || {}));
            this.renderChart(data);
            
        } catch (error) {
            console.error('❌ Failed to load chart data:', error);
            this.showToast(`Failed to load chart data: ${error.message}`, 'error');
            this.showErrorChart();
        }
    }

    renderChart(apiResponse) {
        if (!this.chartInstance || !apiResponse || !apiResponse.data) {
            console.warn('⚠️ No chart instance or data available for rendering');
            return;
        }

        console.log('🎨 Rendering chart with data:', Object.keys(apiResponse.data));
        
        const chartData = apiResponse.data;
        const metadata = apiResponse.metadata || {};
        
        // Check if we have Australian markets to determine market hours
        const hasAustralianMarkets = Object.keys(chartData).some(symbol => 
            symbol === '^AORD' || symbol === 'CBA.AX' || symbol === '^AXJO' || symbol.endsWith('.AX')
        );
        
        // Extract timestamps and series data
        const timestamps = [];
        const seriesData = [];
        
        // Get all unique timestamps across all symbols
        const allTimestamps = new Set();
        Object.values(chartData).forEach(symbolData => {
            symbolData.forEach(point => allTimestamps.add(point.timestamp));
        });
        
        const sortedTimestamps = Array.from(allTimestamps).sort();
        
        // Create series for each symbol
        Object.entries(chartData).forEach(([symbol, points]) => {
            const values = sortedTimestamps.map(timestamp => {
                const point = points.find(p => p.timestamp === timestamp);
                return point ? parseFloat(point.value) : null;
            });
            
            const symbolName = metadata[symbol]?.name || symbol;
            
            const colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4'];
            const colorIndex = Object.keys(chartData).indexOf(symbol) % colors.length;
            
            seriesData.push({
                name: symbolName,
                type: 'line',
                data: values,
                smooth: true,
                connectNulls: false,
                symbol: 'circle',
                symbolSize: 4,
                lineStyle: { 
                    width: 3,
                    color: colors[colorIndex]
                },
                itemStyle: {
                    color: colors[colorIndex]
                },
                emphasis: {
                    focus: 'series',
                    lineStyle: { width: 4 }
                }
            });
        });
        
        // Format timestamps for display in AEST with proper market hours
        const displayTimestamps = this.formatAESTTimestamps(sortedTimestamps, hasAustralianMarkets);
        
        console.log('📊 Chart rendering details:', {
            seriesCount: seriesData.length,
            timestampCount: displayTimestamps.length,
            hasAustralianMarkets,
            chartType: apiResponse.chart_type
        });
        
        const option = {
            backgroundColor: '#ffffff',
            title: { 
                text: `${hasAustralianMarkets ? 'Australian' : 'Global'} Markets (${apiResponse.chart_type || 'Live'})`, 
                left: 'center',
                textStyle: { fontSize: 14, color: '#1f2937' }
            },
            tooltip: { 
                trigger: 'axis',
                axisPointer: { type: 'cross' },
                formatter: function(params) {
                    let result = `<strong>${params[0].axisValue}</strong><br/>`;
                    params.forEach(param => {
                        if (param.value !== null) {
                            const value = typeof param.value === 'number' ? param.value.toFixed(2) : param.value;
                            result += `${param.marker} ${param.seriesName}: ${value}<br/>`;
                        }
                    });
                    return result;
                }
            },
            legend: { 
                data: seriesData.map(s => s.name), 
                bottom: 10,
                type: 'scroll'
            },
            grid: { 
                left: '8%', 
                right: '8%', 
                bottom: hasAustralianMarkets ? '20%' : '15%', 
                top: '20%',
                containLabel: true,
                backgroundColor: 'transparent'
            },
            xAxis: { 
                type: 'category', 
                data: displayTimestamps,
                axisLabel: { 
                    interval: hasAustralianMarkets ? 0 : 'auto',
                    fontSize: 10,
                    rotate: hasAustralianMarkets ? 45 : 0
                },
                name: hasAustralianMarkets ? 'AEST Time' : 'Local Time',
                nameTextStyle: { fontSize: 9 }
            },
            yAxis: { 
                type: 'value',
                axisLabel: { fontSize: 10 }
            },
            series: seriesData,
            animation: true,
            animationDuration: 1000
        };
        
        this.chartInstance.setOption(option, true);
        
        // Force resize after rendering to ensure proper display
        setTimeout(() => {
            if (this.chartInstance) {
                this.chartInstance.resize();
                console.log('🔄 Chart resized after rendering');
            }
        }, 50);
        
        console.log('✅ Chart rendered successfully with', seriesData.length, 'series');
    }

    formatAESTTimestamps(timestamps, isAustralianFocused) {
        if (!timestamps || timestamps.length === 0) return [];
        
        if (isAustralianFocused) {
            // Generate Australian market hours (9:00 AM - 4:00 PM AEST)
            const marketHours = [];
            for (let hour = 9; hour <= 16; hour++) {
                marketHours.push(`${hour.toString().padStart(2, '0')}:00`);
            }
            
            // If we have fewer data points, use the market hours template
            if (timestamps.length <= marketHours.length) {
                return marketHours.slice(0, Math.max(timestamps.length, 1));
            }
        }
        
        // For mixed markets or many data points, format actual timestamps in AEST
        return timestamps.map(ts => {
            const date = new Date(ts);
            return date.toLocaleTimeString('en-AU', { 
                hour: '2-digit', 
                minute: '2-digit',
                hour12: false,
                timeZone: 'Australia/Sydney'
            });
        });
    }

    showLoadingChart() {
        if (!this.chartInstance) return;
        
        const option = {
            title: { text: 'Loading market data...', left: 'center' },
            grid: { left: '10%', right: '10%', top: '20%', bottom: '15%' },
            xAxis: { type: 'category', data: [] },
            yAxis: { type: 'value' },
            series: [],
            graphic: {
                elements: [{
                    type: 'text',
                    left: 'center',
                    top: 'middle',
                    style: {
                        text: '📊 Loading market data...',
                        fontSize: 16,
                        fill: '#666'
                    }
                }]
            }
        };
        
        this.chartInstance.setOption(option);
    }

    showErrorChart() {
        if (!this.chartInstance) return;
        
        const option = {
            title: { text: 'Failed to load data', left: 'center' },
            grid: { left: '10%', right: '10%', top: '20%', bottom: '15%' },
            xAxis: { type: 'category', data: [] },
            yAxis: { type: 'value' },
            series: [],
            graphic: {
                elements: [{
                    type: 'text',
                    left: 'center',
                    top: 'middle',
                    style: {
                        text: '❌ Failed to load chart data\nTry refreshing or selecting different markets',
                        fontSize: 14,
                        fill: '#ef4444',
                        textAlign: 'center'
                    }
                }]
            }
        };
        
        this.chartInstance.setOption(option);
    }

    loadDemoChart() {
        if (!this.chartInstance) return;
        
        // Create demo data to show chart capabilities
        const hours = [];
        const demoData = {
            'S&P 500': [],
            'FTSE 100': [],
            'Nikkei 225': []
        };
        
        // Generate 24 hours of demo data
        for (let i = 0; i < 24; i++) {
            const hour = String(i).padStart(2, '0') + ':00';
            hours.push(hour);
            
            // Generate realistic-looking demo data
            demoData['S&P 500'].push((4200 + Math.sin(i/4) * 50 + Math.random() * 30).toFixed(2));
            demoData['FTSE 100'].push((7300 + Math.cos(i/3) * 40 + Math.random() * 25).toFixed(2));
            demoData['Nikkei 225'].push((28000 + Math.sin(i/5) * 200 + Math.random() * 100).toFixed(0));
        }
        
        const option = {
            title: { 
                text: 'Demo Market Data (Select markets to see live data)', 
                left: 'center',
                textStyle: { fontSize: 12 }
            },
            tooltip: { trigger: 'axis' },
            legend: { data: Object.keys(demoData), bottom: 10 },
            grid: { left: '3%', right: '4%', bottom: '15%', top: '15%', containLabel: true },
            xAxis: { 
                type: 'category', 
                data: hours,
                axisLabel: { interval: 3, fontSize: 10 }
            },
            yAxis: { 
                type: 'value',
                axisLabel: { fontSize: 10 }
            },
            series: Object.entries(demoData).map(([name, data]) => ({
                name: name,
                type: 'line',
                data: data,
                smooth: true,
                symbol: 'none',
                lineStyle: { width: 2, opacity: 0.7 }
            })),
            animation: true
        };
        
        this.chartInstance.setOption(option);
        console.log('✅ Demo chart loaded');
    }

    autoSelectTestMarkets() {
        console.log('🧪 Auto-selecting Australian test markets for chart verification');
        
        // Select Australian markets as primary focus
        const testSymbols = ['^AORD', 'CBA.AX']; // All Ordinaries, Commonwealth Bank
        
        testSymbols.forEach(symbol => {
            if (this.allSymbols.has(symbol)) {
                this.selectedIndices.add(symbol);
            }
        });
        
        if (this.selectedIndices.size > 0) {
            this.updateSelectedMarketsDisplay();
            
            const analyzeBtn = document.getElementById('analyze-btn');
            if (analyzeBtn) {
                analyzeBtn.disabled = false;
            }
            
            // Automatically load chart data
            setTimeout(() => {
                console.log('🔄 Auto-loading chart data for test markets');
                this.loadChartData();
            }, 1000);
            
            this.showToast(`Auto-selected ${this.selectedIndices.size} test markets`, 'info');
        }
    }

    handleChartTypeChange(event) {
        console.log('📊 Chart type changed:', event.target.value);
        
        // Reload chart data with new chart type if we have selected indices
        if (this.selectedIndices.size > 0) {
            this.loadChartData();
        }
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