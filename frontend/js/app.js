/**
 * GSMT Ver 7.0 - Clean Architecture Frontend
 * Optimized for Netlify deployment with Railway backend integration
 */

class GSMTApp {
    constructor() {
        // Application state
        this.state = {
            apiBaseUrl: this.detectApiUrl(),
            selectedSymbols: new Set(),
            symbolsDatabase: new Map(),
            chartData: new Map(),
            chartInstance: null,
            settings: this.loadSettings(),
            isLoading: false,
            refreshTimer: null,
            sydneyTimeUtils: window.sydneyTimeUtils,
            currentSydneyTime: null,
            marketSessions: new Map(),
            liveDataEnabled: true
        };
        
        // Initialize application
        this.init();
    }
    
    /**
     * Initialize the application with Sydney timezone support
     */
    async init() {
        try {
            console.log('🚀 Initializing GSMT Ver 7.0 - Sydney Edition');
            
            // Initialize Sydney timezone utilities
            this.initializeSydneyTime();
            
            // Setup event listeners
            this.setupEventListeners();
            
            // Initialize chart
            this.initializeChart();
            
            // Check API connection and Sydney timezone info
            await this.checkApiConnection();
            
            // Load symbols database
            await this.loadSymbolsDatabase();
            
            // Load Sydney market sessions
            await this.loadMarketSessions();
            
            // Auto-select default indices with Sydney focus
            await this.autoSelectDefaultIndices();
            
            // Setup live data refresh
            this.setupLiveDataRefresh();
            
            // Apply settings
            this.applySettings();
            
            console.log('✅ GSMT Ver 7.0 Sydney Edition ready');
            this.showToast('GSMT Ver 7.0 Sydney Edition: Live data from 10am AEST/AEDT', 'success');
            
        } catch (error) {
            console.error('❌ Initialization failed:', error);
            this.updateApiStatus('error', 'Connection failed - Configure API URL in settings');
            this.showToast('Please configure your Railway API URL in Settings', 'warning');
        }
    }
    
    /**
     * Initialize Sydney timezone functionality
     */
    initializeSydneyTime() {
        if (!this.state.sydneyTimeUtils) {
            console.warn('Sydney timezone utilities not available - some features may be limited');
            return;
        }
        
        // Update Sydney time display
        this.updateSydneyTimeDisplay();
        
        // Set up Sydney time updates every minute
        setInterval(() => {
            this.updateSydneyTimeDisplay();
        }, 60000);
        
        console.log('🕒 Sydney timezone utilities initialized');
    }
    
    /**
     * Update Sydney time display in UI
     */
    updateSydneyTimeDisplay() {
        if (!this.state.sydneyTimeUtils) return;
        
        const sydneyNow = this.state.sydneyTimeUtils.getSydneyNow();
        this.state.currentSydneyTime = sydneyNow;
        
        // Update API status with Sydney time
        const statusElement = document.getElementById('api-status');
        if (statusElement) {
            const existingText = statusElement.querySelector('span').textContent;
            if (!existingText.includes('Sydney:')) {
                const sydneyTime = this.state.sydneyTimeUtils.formatSydneyTime(sydneyNow, {
                    showDate: false,
                    showTime: true,
                    showTimezone: true
                });
                statusElement.querySelector('span').textContent = `${existingText} • Sydney: ${sydneyTime}`;
            }
        }
        
        // Update any Sydney time indicators
        const indicators = document.querySelectorAll('.sydney-time-value');
        indicators.forEach(indicator => {
            const formatted = this.state.sydneyTimeUtils.formatSydneyTime(sydneyNow, {
                showDate: false,
                showTime: true,
                showTimezone: true
            });
            indicator.textContent = formatted;
        });
    }
    
    /**
     * Load market sessions from API
     */
    async loadMarketSessions() {
        if (!this.state.apiBaseUrl) {
            console.warn('API URL not configured - using fallback market sessions');
            this.loadFallbackMarketSessions();
            return;
        }
        
        try {
            const response = await fetch(`${this.state.apiBaseUrl}/market-sessions`);
            if (response.ok) {
                const data = await response.json();
                
                if (data.success) {
                    // Store market sessions
                    data.sessions.forEach(session => {
                        this.state.marketSessions.set(session.market, session);
                    });
                    
                    console.log(`📊 Loaded ${data.sessions.length} market sessions`);
                    this.updateMarketSessionsDisplay(data.sessions);
                }
            } else {
                throw new Error('Failed to load market sessions');
            }
        } catch (error) {
            console.warn('Failed to load market sessions from API:', error);
            this.loadFallbackMarketSessions();
        }
    }
    
    /**
     * Load fallback market sessions
     */
    loadFallbackMarketSessions() {
        if (!this.state.sydneyTimeUtils) return;
        
        const marketStatus = this.state.sydneyTimeUtils.getGlobalMarketStatus();
        
        Object.entries(marketStatus).forEach(([market, status]) => {
            this.state.marketSessions.set(market, {
                market,
                display_name: status.displayName,
                is_active: status.isOpen,
                local_time: status.localTime,
                color: this.getMarketColor(market)
            });
        });
        
        console.log('📊 Loaded fallback market sessions');
    }
    
    /**
     * Get color for market visualization
     */
    getMarketColor(market) {
        const colors = {
            'Australia': '#10b981',
            'Japan': '#3b82f6',
            'Hong Kong': '#8b5cf6',
            'China': '#ef4444',
            'UK': '#f59e0b',
            'Germany': '#f97316',
            'France': '#6366f1',
            'US': '#06b6d4'
        };
        return colors[market] || '#6b7280';
    }
    
    /**
     * Setup live data refresh based on market activity
     */
    setupLiveDataRefresh() {
        if (!this.state.liveDataEnabled) return;
        
        // Get recommended refresh interval
        let refreshInterval = 300000; // 5 minutes default
        
        if (this.state.sydneyTimeUtils) {
            refreshInterval = this.state.sydneyTimeUtils.getRecommendedRefreshInterval() * 1000;
        }
        
        // Clear existing timer
        if (this.state.refreshTimer) {
            clearInterval(this.state.refreshTimer);
        }
        
        // Set up new timer
        this.state.refreshTimer = setInterval(() => {
            if (this.state.selectedSymbols.size > 0) {
                console.log('🔄 Auto-refreshing live data...');
                this.handleAnalyze(true); // Silent refresh
            }
        }, refreshInterval);
        
        console.log(`🔄 Live data refresh set to ${refreshInterval / 1000} seconds`);
    }
    
    /**
     * Detect API URL based on environment
     */
    detectApiUrl() {
        // Check if custom API URL is stored
        const saved = localStorage.getItem('gsmt-api-url');
        if (saved) return saved;
        
        // Development environment detection
        if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
            return 'http://localhost:8000';
        }
        
        // Production - try to use Railway URL pattern or fallback to demo
        const hostname = window.location.hostname;
        if (hostname.includes('netlify')) {
            // This will be configured via settings modal
            return null; // Will trigger demo mode until configured
        }
        
        return window.location.origin;
    }
    
    /**
     * Setup all event listeners
     */
    setupEventListeners() {
        // Symbol search
        const symbolSearch = document.getElementById('symbol-search');
        symbolSearch.addEventListener('input', this.debounce(this.handleSymbolSearch.bind(this), 300));
        symbolSearch.addEventListener('keydown', this.handleSymbolKeydown.bind(this));
        
        // Control buttons
        document.getElementById('analyze-btn').addEventListener('click', this.handleAnalyze.bind(this));
        document.getElementById('clear-btn').addEventListener('click', this.handleClear.bind(this));
        
        // Settings
        document.getElementById('settings-btn').addEventListener('click', this.showSettings.bind(this));
        document.getElementById('close-settings').addEventListener('click', this.hideSettings.bind(this));
        document.getElementById('save-settings').addEventListener('click', this.saveSettings.bind(this));
        document.getElementById('cancel-settings').addEventListener('click', this.hideSettings.bind(this));
        
         // Chart controls
        document.getElementById('analysis-mode').addEventListener('change', this.handleAnalysisModeChange.bind(this));
        document.getElementById('chart-type').addEventListener('change', this.handleChartTypeChange.bind(this));
        document.getElementById('candlestick-interval').addEventListener('change', this.handleCandlestickIntervalChange.bind(this));
        document.getElementById('fullscreen-btn').addEventListener('click', this.toggleFullscreen.bind(this));
        
        // Global events
        document.addEventListener('click', this.handleDocumentClick.bind(this));
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
        
        // Set initial empty state
        this.updateChart();
    }
    
    /**
     * Check API connection
     */
    async checkApiConnection() {
        if (!this.state.apiBaseUrl) {
            this.updateApiStatus('disconnected', 'API URL not configured');
            return false;
        }
        
        try {
            const response = await fetch(`${this.state.apiBaseUrl}/health`, {
                method: 'GET',
                timeout: 5000
            });
            
            if (response.ok) {
                const data = await response.json();
                this.updateApiStatus('connected', `v${data.version}`);
                return true;
            } else {
                throw new Error(`API returned ${response.status}`);
            }
        } catch (error) {
            console.warn('API connection failed:', error);
            this.updateApiStatus('error', 'Connection failed');
            return false;
        }
    }
    
     /**
     * Auto-select default indices (FTSE 100, S&P 500, ASX 200, Nikkei 225)
     */
    async autoSelectDefaultIndices() {
        // Default indices as specified in requirements
        const defaultSymbols = ['^FTSE', '^GSPC', '^AXJO', '^N225'];
        
        // Add default symbols to selection
        defaultSymbols.forEach(symbol => {
            this.state.selectedSymbols.add(symbol);
        });
        
        // Add to symbols database if not already there
        if (this.state.symbolsDatabase.size === 0) {
            const defaultSymbolsInfo = [
                { symbol: '^FTSE', name: 'FTSE 100', market: 'UK', category: 'Index', priority: 1 },
                { symbol: '^GSPC', name: 'S&P 500', market: 'US', category: 'Index', priority: 1 },
                { symbol: '^AXJO', name: 'ASX 200', market: 'Australia', category: 'Index', priority: 1 },
                { symbol: '^N225', name: 'Nikkei 225', market: 'Japan', category: 'Index', priority: 1 }
            ];
            
            defaultSymbolsInfo.forEach(symbolInfo => {
                this.state.symbolsDatabase.set(symbolInfo.symbol, symbolInfo);
            });
        }
        
        this.updateSelectedSymbolsDisplay();
        
        console.log('📊 Auto-selected default indices: FTSE 100, S&P 500, ASX 200, Nikkei 225');
    }

    /**
     * Load symbols database from API
     */
    async loadSymbolsDatabase() {
        if (!this.state.apiBaseUrl) {
            this.loadFallbackSymbols();
            return;
        }
        
        try {
            const response = await fetch(`${this.state.apiBaseUrl}/symbols`);
            if (response.ok) {
                const data = await response.json();
                
                // Process symbols by category
                for (const [category, symbols] of Object.entries(data.categories)) {
                    symbols.forEach(symbol => {
                        this.state.symbolsDatabase.set(symbol.symbol, {
                            ...symbol,
                            category: category
                        });
                    });
                }
                
                console.log(`📊 Loaded ${this.state.symbolsDatabase.size} symbols`);
            } else {
                throw new Error('Failed to load symbols');
            }
        } catch (error) {
            console.warn('Failed to load symbols from API:', error);
            this.loadFallbackSymbols();
        }
    }
    
    /**
     * Load fallback symbols for demo mode
     */
    loadFallbackSymbols() {
        const fallbackSymbols = [
            { symbol: '^GSPC', name: 'S&P 500', market: 'US', category: 'Index' },
            { symbol: '^IXIC', name: 'NASDAQ', market: 'US', category: 'Index' },
            { symbol: '^AXJO', name: 'ASX 200', market: 'Australia', category: 'Index' },
            { symbol: 'AAPL', name: 'Apple Inc.', market: 'US', category: 'Technology' },
            { symbol: 'GOOGL', name: 'Alphabet Inc.', market: 'US', category: 'Technology' },
            { symbol: 'CBA.AX', name: 'Commonwealth Bank', market: 'Australia', category: 'Finance' }
        ];
        
        fallbackSymbols.forEach(symbol => {
            this.state.symbolsDatabase.set(symbol.symbol, symbol);
        });
        
        console.log('📊 Loaded fallback symbols');
    }
    
    /**
     * Handle symbol search
     */
    async handleSymbolSearch(event) {
        const query = event.target.value.trim();
        const suggestionsContainer = document.getElementById('symbol-suggestions');
        
        if (query.length < 1) {
            suggestionsContainer.classList.add('hidden');
            return;
        }
        
        let suggestions = [];
        
        // Try API search first
        if (this.state.apiBaseUrl) {
            try {
                const response = await fetch(`${this.state.apiBaseUrl}/search/${encodeURIComponent(query)}`);
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
    }
    
    /**
     * Search symbols locally
     */
    searchSymbolsLocally(query) {
        const results = [];
        const queryLower = query.toLowerCase();
        
        for (const [symbol, info] of this.state.symbolsDatabase.entries()) {
            if (symbol.toLowerCase().includes(queryLower) || 
                info.name.toLowerCase().includes(queryLower)) {
                results.push({
                    symbol: symbol,
                    name: info.name,
                    market: info.market,
                    category: info.category
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
                        <div class="text-xs text-gray-500">${suggestion.market}</div>
                    </div>
                </div>
            `).join('');
            
            // Add click listeners
            container.querySelectorAll('.suggestion-item').forEach(item => {
                item.addEventListener('click', this.handleSymbolSelection.bind(this));
            });
        }
        
        container.classList.remove('hidden');
    }
    
    /**
     * Handle symbol selection
     */
    handleSymbolSelection(event) {
        const symbol = event.currentTarget.dataset.symbol;
        this.addSymbol(symbol);
        
        // Clear search
        document.getElementById('symbol-search').value = '';
        document.getElementById('symbol-suggestions').classList.add('hidden');
    }
    
    /**
     * Handle symbol keydown
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
            const info = this.state.symbolsDatabase.get(symbol) || { name: symbol, market: 'Unknown' };
            return `
                <div class="symbol-chip flex items-center bg-primary-100 text-primary-800 px-3 py-2 rounded-full text-sm">
                    <div class="flex flex-col">
                        <span class="font-medium">${symbol}</span>
                        <span class="text-xs text-primary-600">${info.name}</span>
                    </div>
                    <button class="ml-3 text-primary-600 hover:text-primary-800 transition-colors" onclick="app.removeSymbol('${symbol}')">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
            `;
        }).join('');
    }
    
    /**
     * Handle analyze button with Sydney timezone support
     */
    async handleAnalyze(silent = false) {
        const analysisMode = document.getElementById('analysis-mode').value;
        
        // Check if Global 24H mode is selected
        if (analysisMode === 'global-24h') {
            await this.handleSydneyMarketsAnalysis(silent);
            return;
        }
        
        // Standard analysis mode
        if (this.state.selectedSymbols.size === 0) {
            if (!silent) this.showToast('Please select at least one symbol', 'warning');
            return;
        }
        
        this.setLoading(true);
        
        try {
            const symbols = Array.from(this.state.selectedSymbols);
            const period = document.getElementById('time-period').value;
            const chartType = document.getElementById('chart-type').value;
            
            let analysisData;
            
            if (this.state.apiBaseUrl) {
                // API analysis with Sydney timezone support
                const requestBody = {
                    symbols: symbols,
                    period: period,
                    chart_type: chartType,
                    sydney_start: period === '24h' // Enable Sydney 10am start for 24h periods
                };
                
                // Add current Sydney time as reference if available
                if (this.state.currentSydneyTime) {
                    requestBody.reference_time = this.state.currentSydneyTime.toISOString();
                }
                
                const response = await fetch(`${this.state.apiBaseUrl}/analyze`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(requestBody)
                });
                
                if (response.ok) {
                    analysisData = await response.json();
                } else {
                    throw new Error(`API error: ${response.status}`);
                }
            } else {
                // Generate demo data with Sydney timezone
                analysisData = this.generateDemoAnalysis(symbols, period);
            }
            
            // Store and display data
            this.state.chartData.clear();
            for (const [symbol, data] of Object.entries(analysisData.data)) {
                this.state.chartData.set(symbol, data);
            }
            
            // Store market sessions if provided
            if (analysisData.market_sessions) {
                this.state.marketSessionsData = analysisData.market_sessions;
            }
            
            this.updateChart();
            this.updatePerformanceSummary();
            
            if (!silent) {
                const message = analysisData.sydney_timestamp ? 
                    `Analysis complete (${analysisData.sydney_timestamp}): ${symbols.length} symbols` :
                    `Analysis complete: ${symbols.length} symbols processed`;
                this.showToast(message, 'success');
            }
            
        } catch (error) {
            console.error('Analysis failed:', error);
            if (!silent) this.showToast('Analysis failed. Please check your API connection.', 'error');
        } finally {
            this.setLoading(false);
        }
    }
    
    /**
     * Handle Sydney-focused market analysis (replaces Global 24H)
     */
    async handleSydneyMarketsAnalysis(silent = false) {
        this.setLoading(true);
        
        try {
            let analysisData;
            
            if (this.state.apiBaseUrl) {
                // Try Sydney markets API endpoint
                const response = await fetch(`${this.state.apiBaseUrl}/sydney-markets`);
                
                if (response.ok) {
                    analysisData = await response.json();
                } else {
                    throw new Error(`API error: ${response.status}`);
                }
            } else {
                // Generate demo Sydney markets data
                analysisData = this.generateDemoSydneyData();
            }
            
            // Store and display data
            this.state.chartData.clear();
            for (const [symbol, data] of Object.entries(analysisData.data)) {
                this.state.chartData.set(symbol, data);
            }
            
            // Store Sydney-specific data
            this.state.marketSessionsData = analysisData.market_sessions;
            this.state.sydneyContext = analysisData.sydney_context;
            this.state.refreshSchedule = analysisData.refresh_schedule;
            
            this.updateSydneyChart();
            this.updateMarketSessions();
            this.updatePerformanceSummary();
            
            if (!silent) {
                this.showToast('Sydney Markets analysis complete - Live data from 10am AEST/AEDT', 'success');
            }
            
        } catch (error) {
            console.error('Sydney Markets analysis failed:', error);
            if (!silent) this.showToast('Sydney Markets analysis failed. Please check your API connection.', 'error');
        } finally {
            this.setLoading(false);
        }
    }
    
    /**
     * Generate demo trading hours data (not 24h)
     */
    generateDemoSydneyData() {
        const defaultSymbols = ['^AXJO', '^N225', '^FTSE', '^GSPC'];
        const data = {};
        const marketSessions = [];
        
        defaultSymbols.forEach(symbol => {
            const market = this.getMarketForSymbol(symbol);
            const tradingData = this.generateTradingHoursData(symbol, market);
            data[symbol] = tradingData;
            
            // Add unique market sessions
            if (!marketSessions.find(s => s.market === market)) {
                const tradingHours = {
                    'Australia': { duration: 6, start: '10:00', end: '16:00' },
                    'Japan': { duration: 6, start: '09:00', end: '15:00' },
                    'UK': { duration: 8, start: '08:00', end: '16:00' },
                    'US': { duration: 7, start: '09:30', end: '16:00' }
                };
                
                const hours = tradingHours[market] || tradingHours['Australia'];
                
                marketSessions.push({
                    market,
                    display_name: market === 'Australia' ? '🇦🇺 Sydney (ASX)' :
                                 market === 'Japan' ? '🇯🇵 Tokyo (Nikkei)' :
                                 market === 'UK' ? '🇬🇧 London (FTSE)' :
                                 market === 'US' ? '🇺🇸 New York (S&P)' : market,
                    trading_hours: `${hours.duration}h session (${hours.start}-${hours.end} local)`,
                    duration_hours: hours.duration,
                    color: this.getMarketColor(market)
                });
            }
        });
        
        return {
            success: true,
            data,
            market_sessions: marketSessions,
            sydney_context: {
                current_sydney_time: new Date().toISOString(),
                sydney_market_status: true,
                display_note: 'Charts show trading hours only - no overnight periods'
            },
            refresh_schedule: { 
                primary_refresh: 180,
                chart_refresh: 180
            },
            display_mode: 'trading_hours_only'
        };
    }
    
    /**
     * Generate realistic trading hours data for a symbol
     */
    generateTradingHoursData(symbol, market) {
        const points = [];
        let basePrice = symbol.startsWith('^') ? 
            Math.random() * 20000 + 5000 : 
            Math.random() * 400 + 50;
        let currentPrice = basePrice;
        
        // Trading session parameters
        const tradingHours = {
            'Australia': { start: 10, duration: 6 },   // 10am-4pm
            'Japan': { start: 9, duration: 6 },        // 9am-3pm
            'UK': { start: 8, duration: 8 },           // 8am-4pm  
            'US': { start: 9, duration: 7 }            // 9:30am-4pm (simplified)
        };
        
        const schedule = tradingHours[market] || tradingHours['Australia'];
        
        // Create session start time (today in market timezone)
        const sessionStart = new Date();
        sessionStart.setHours(schedule.start, 0, 0, 0);
        
        // Generate 15-minute interval data during trading hours
        const intervalMinutes = 15;
        const totalIntervals = (schedule.duration * 60) / intervalMinutes;
        
        for (let interval = 0; interval < totalIntervals; interval++) {
            const timestamp = new Date(sessionStart);
            timestamp.setMinutes(sessionStart.getMinutes() + (interval * intervalMinutes));
            
            // Realistic price movement during trading
            const volatility = 0.008; // 0.8% per 15-min interval
            const change = (Math.random() - 0.5) * volatility;
            currentPrice *= (1 + change);
            
            // Prevent excessive drift
            const maxDrift = 0.12; // 12% max drift from base
            if (Math.abs((currentPrice - basePrice) / basePrice) > maxDrift) {
                currentPrice = basePrice * (1 + (Math.random() - 0.5) * maxDrift);
            }
            
            const percentageChange = ((currentPrice - basePrice) / basePrice) * 100;
            
            // Volume patterns - higher at open/close
            let volumeMultiplier = 1;
            if (interval < 4 || interval > totalIntervals - 4) {
                volumeMultiplier = 2.5; // First/last hour
            } else if (interval > totalIntervals / 2 - 2 && interval < totalIntervals / 2 + 2) {
                volumeMultiplier = 1.5; // Mid-session
            }
            
            points.push({
                timestamp: timestamp.toISOString(),
                timestamp_ms: timestamp.getTime(),
                open: Math.round(currentPrice * 0.998 * 100) / 100,
                high: Math.round(currentPrice * 1.004 * 100) / 100,
                low: Math.round(currentPrice * 0.996 * 100) / 100,
                close: Math.round(currentPrice * 100) / 100,
                volume: Math.floor(Math.random() * 3000000 * volumeMultiplier),
                percentage_change: Math.round(percentageChange * 100) / 100,
                market: market,
                interval: `${intervalMinutes}min`
            });
        }
        
        return points;
    }
    
    /**
     * Handle Global 24H Market Flow Analysis
     */
    async handleGlobal24HAnalysis() {
        this.setLoading(true);
        
        try {
            let analysisData;
            
            if (this.state.apiBaseUrl) {
                // Try API global 24h endpoint
                const response = await fetch(`${this.state.apiBaseUrl}/global-24h`);
                
                if (response.ok) {
                    analysisData = await response.json();
                } else {
                    throw new Error(`API error: ${response.status}`);
                }
            } else {
                // Generate demo global 24h data
                analysisData = this.generateDemo24HData();
            }
            
            // Store and display data
            this.state.chartData.clear();
            for (const [symbol, data] of Object.entries(analysisData.data)) {
                this.state.chartData.set(symbol, data);
            }
            
            // Store market hours for visualization
            this.state.marketHours = analysisData.market_hours;
            
            this.update24HChart();
            this.updateMarketSessions();
            this.updatePerformanceSummary();
            
            this.showToast('Global 24H Market Flow analysis complete', 'success');
            
        } catch (error) {
            console.error('Global 24H analysis failed:', error);
            this.showToast('Global 24H analysis failed. Please check your API connection.', 'error');
        } finally {
            this.setLoading(false);
        }
    }
    
    /**
     * Generate demo Global 24H data
     */
    generateDemo24HData() {
        const marketHours = {
            "Japan": {"open": 0, "close": 6},
            "Hong Kong": {"open": 1, "close": 8},
            "UK": {"open": 8, "close": 16},
            "Germany": {"open": 7, "close": 15},
            "France": {"open": 7, "close": 15},
            "US": {"open": 14, "close": 21}
        };
        
        const symbols = {
            "^N225": "Nikkei 225",      // Japan
            "^HSI": "Hang Seng",       // Hong Kong
            "^FTSE": "FTSE 100",       // UK
            "^GDAXI": "DAX",           // Germany
            "^FCHI": "CAC 40",        // France
            "^GSPC": "S&P 500"         // US
        };
        
        const data = {};
        
        // Generate 24 hours of data (hourly points)
        for (const [symbol, name] of Object.entries(symbols)) {
            const points = [];
            let basePrice = Math.random() * 20000 + 5000;
            
            // Get market for this symbol
            const market = this.getMarketForSymbol(symbol);
            const hours = marketHours[market];
            
            for (let hour = 0; hour < 24; hour++) {
                const timestamp = new Date();
                timestamp.setHours(hour, 0, 0, 0);
                
                // Higher volatility during market hours
                const isMarketOpen = (hour >= hours.open && hour <= hours.close);
                const volatility = isMarketOpen ? 0.02 : 0.005; // 2% vs 0.5%
                
                const change = (Math.random() - 0.5) * volatility;
                basePrice *= (1 + change);
                
                // Calculate percentage change from start of day
                const startPrice = basePrice / Math.pow(1 + change, hour + 1);
                const percentageChange = ((basePrice - startPrice) / startPrice) * 100;
                
                points.push({
                    timestamp: timestamp.toISOString(),
                    timestamp_ms: timestamp.getTime(),
                    open: basePrice * 0.999,
                    high: basePrice * 1.001,
                    low: basePrice * 0.999,
                    close: basePrice,
                    volume: isMarketOpen ? Math.floor(Math.random() * 5000000) : Math.floor(Math.random() * 500000),
                    percentage_change: percentageChange,
                    market_open: isMarketOpen
                });
            }
            
            data[symbol] = points;
        }
        
        return { data, market_hours: marketHours, success: true };
    }
    
    /**
     * Get market for symbol
     */
    getMarketForSymbol(symbol) {
        const marketMap = {
            "^N225": "Japan",
            "^HSI": "Hong Kong", 
            "^FTSE": "UK",
            "^GDAXI": "Germany",
            "^FCHI": "France",
            "^GSPC": "US"
        };
        return marketMap[symbol] || "US";
    }
    
    /**
     * Generate demo analysis data
     */
    generateDemoAnalysis(symbols, period) {
        const data = {};
        const days = this.getPeriodDays(period);
        
        symbols.forEach(symbol => {
            const points = [];
            let basePrice = symbol.startsWith('^') ? 
                Math.random() * 30000 + 5000 : 
                Math.random() * 400 + 50;
            
            for (let i = 0; i < Math.min(50, days); i++) {
                const timestamp = new Date(Date.now() - (days - i) * 24 * 60 * 60 * 1000);
                const change = (Math.random() - 0.5) * 0.04; // 4% max change
                basePrice *= (1 + change);
                
                const percentageChange = ((basePrice - (basePrice / Math.pow(1 + change, i + 1))) / (basePrice / Math.pow(1 + change, i + 1))) * 100;
                
                points.push({
                    timestamp: timestamp.toISOString(),
                    timestamp_ms: timestamp.getTime(),
                    open: basePrice * 0.99,
                    high: basePrice * 1.02,
                    low: basePrice * 0.98,
                    close: basePrice,
                    volume: Math.floor(Math.random() * 10000000),
                    percentage_change: percentageChange
                });
            }
            
            data[symbol] = points;
        });
        
        return { data, success: true };
    }
    
    /**
     * Get period in days
     */
    getPeriodDays(period) {
        const periodMap = {
            '24h': 1, '3d': 3, '1w': 7, '2w': 14, 
            '1M': 30, '3M': 90, '6M': 180, '1Y': 365, '2Y': 730
        };
        return periodMap[period] || 1;
    }
    
    /**
     * Update the main chart
     */
    updateChart() {
        if (!this.state.chartInstance) return;
        
        if (this.state.chartData.size === 0) {
            this.state.chartInstance.setOption(this.getEmptyChartOption());
            return;
        }
        
        const analysisMode = document.getElementById('analysis-mode').value;
        const option = analysisMode === 'global-24h' ? 
            this.generate24HChartOption() : 
            this.generateStandardChartOption();
        this.state.chartInstance.setOption(option, true);
    }
    
    /**
     * Update Sydney chart specifically
     */
    updateSydneyChart() {
        if (!this.state.chartInstance) return;
        
        if (this.state.chartData.size === 0) {
            this.state.chartInstance.setOption(this.getEmptyChartOption());
            return;
        }
        
        const option = this.generateSydneyChartOption();
        this.state.chartInstance.setOption(option, true);
    }
    
    /**
     * Update 24H chart specifically (legacy support)
     */
    update24HChart() {
        this.updateSydneyChart();
    }
    
    /**
     * Generate market-specific chart option showing only trading hours
     */
    generateSydneyChartOption() {
        const colors = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#f97316'];
        const series = [];
        const markAreas = [];
        let colorIndex = 0;
        
        // Group data by market and show only trading hours
        const marketGroups = new Map();
        
        for (const [symbol, data] of this.state.chartData.entries()) {
            const market = this.getMarketForSymbol(symbol);
            const displayName = this.getMarketDisplayName(market);
            
            // Filter data to show only market trading hours (6-7 hours per market)
            const tradingHoursData = this.filterToTradingHours(data, market);
            
            if (tradingHoursData.length === 0) continue;
            
            if (!marketGroups.has(market)) {
                marketGroups.set(market, []);
            }
            
            marketGroups.get(market).push({
                symbol,
                data: tradingHoursData,
                displayName
            });
            
            series.push({
                name: `${symbol} ${displayName}`,
                type: 'line',
                data: tradingHoursData.map(point => {
                    const timestamp = this.state.sydneyTimeUtils ? 
                        this.state.sydneyTimeUtils.formatTimestampForChart(point.timestamp_ms) :
                        { timestamp: point.timestamp_ms };
                    
                    return [timestamp.timestamp || point.timestamp_ms, point.percentage_change];
                }),
                smooth: true,
                symbol: 'none',
                lineStyle: { width: market === 'Australia' ? 4 : 3 },
                color: colors[colorIndex % colors.length],
                emphasis: {
                    lineStyle: { width: market === 'Australia' ? 5 : 4 }
                }
            });
            
            colorIndex++;
        }
        
        // Add market session indicators for trading hours only
        this.addTradingHoursMarketSessions(markAreas, marketGroups);
        
        // Calculate time range for all active trading periods
        const timeRange = this.calculateTradingTimeRange(marketGroups);
        
        return {
            title: {
                text: 'Live Market Data - Trading Hours Only',
                subtext: this.getTradingHoursSubtext(marketGroups),
                left: 'center',
                textStyle: { fontSize: 18, fontWeight: 'bold', color: '#374151' },
                subtextStyle: { fontSize: 12, color: '#6b7280' }
            },
            tooltip: {
                trigger: 'axis',
                axisPointer: { type: 'cross' },
                backgroundColor: 'rgba(255, 255, 255, 0.98)',
                borderColor: '#e5e7eb',
                textStyle: { color: '#374151' },
                formatter: (params) => {
                    const timestamp = params[0].axisValue;
                    
                    let timeDisplay = 'Time';
                    if (this.state.sydneyTimeUtils) {
                        const sydneyTime = this.state.sydneyTimeUtils.formatTimestampForChart(timestamp);
                        timeDisplay = `${sydneyTime.display} AEST/AEDT`;
                    } else {
                        timeDisplay = new Date(timestamp).toLocaleString();
                    }
                    
                    let html = `<div style="font-weight: bold; margin-bottom: 5px; color: #10b981;">${timeDisplay}</div>`;
                    
                    params.forEach(param => {
                        const value = param.value[1];
                        const color = param.color;
                        html += `<div style="display: flex; align-items: center; margin: 2px 0;">`;
                        html += `<span style="display: inline-block; width: 10px; height: 10px; background-color: ${color}; border-radius: 50%; margin-right: 8px;"></span>`;
                        html += `<span style="flex: 1;">${param.seriesName}: </span>`;
                        html += `<span style="font-weight: bold; color: ${value >= 0 ? '#10b981' : '#ef4444'}">${value >= 0 ? '+' : ''}${value.toFixed(2)}%</span>`;
                        html += `</div>`;
                    });
                    
                    return html;
                }
            },
            legend: {
                top: 60,
                type: 'scroll',
                textStyle: { fontSize: 11 }
            },
            grid: {
                left: '3%',
                right: '4%',
                bottom: '12%',
                top: '20%',
                containLabel: true
            },
            xAxis: {
                type: 'time',
                axisLine: { lineStyle: { color: '#d1d5db' } },
                axisLabel: { 
                    color: '#6b7280',
                    formatter: (value) => {
                        if (this.state.sydneyTimeUtils) {
                            const sydneyTime = this.state.sydneyTimeUtils.formatTimestampForChart(value);
                            const timePart = sydneyTime.display.split(' ')[1] || '';
                            return timePart.replace(':00', '');
                        }
                        return new Date(value).toLocaleTimeString('en-US', { 
                            hour: 'numeric', 
                            minute: '2-digit',
                            hour12: true 
                        });
                    },
                    interval: 0,
                    rotate: 45
                },
                splitLine: { 
                    show: true,
                    lineStyle: { color: '#f3f4f6', type: 'dashed' }
                },
                // Set min/max to show only trading hours range
                min: timeRange.start,
                max: timeRange.end
            },
            yAxis: {
                type: 'value',
                axisLine: { lineStyle: { color: '#d1d5db' } },
                axisLabel: { 
                    color: '#6b7280',
                    formatter: '{value}%'
                },
                splitLine: { lineStyle: { color: '#f3f4f6' } }
            },
            series: series.concat(markAreas.length > 0 ? {
                name: 'Trading Sessions',
                type: 'line',
                data: [],
                markArea: {
                    silent: true,
                    itemStyle: {
                        opacity: 0.15
                    },
                    data: markAreas
                }
            } : []),
            animation: true,
            animationDuration: 1500
        };
    }
    
    /**
     * Filter data to show only trading hours for a specific market
     */
    filterToTradingHours(data, market) {
        if (!data || data.length === 0) return [];
        
        // Market trading hours in local time (simplified)
        const tradingHours = {
            'Australia': { start: 10, end: 16, duration: 6 }, // 10am-4pm AEST
            'Japan': { start: 9, end: 15, duration: 6 },      // 9am-3pm JST  
            'Hong Kong': { start: 9, end: 16, duration: 7 },  // 9am-4pm HKT
            'China': { start: 9, end: 15, duration: 6 },      // 9am-3pm CST
            'UK': { start: 8, end: 16, duration: 8 },         // 8am-4pm GMT
            'Germany': { start: 9, end: 17, duration: 8 },    // 9am-5pm CET
            'France': { start: 9, end: 17, duration: 8 },     // 9am-5pm CET
            'US': { start: 9, end: 16, duration: 7 }          // 9am-4pm EST
        };
        
        const marketHours = tradingHours[market];
        if (!marketHours) return data; // Return all data if market unknown
        
        // Filter to show only the market's trading hours
        return data.filter(point => {
            const timestamp = new Date(point.timestamp_ms);
            
            // Convert to market's local time for filtering
            let localTime;
            try {
                const marketTimezone = this.getMarketTimezone(market);
                localTime = new Date(timestamp.toLocaleString("en-US", {timeZone: marketTimezone}));
            } catch (error) {
                // Fallback to Sydney time
                localTime = timestamp;
            }
            
            const hour = localTime.getHours();
            return hour >= marketHours.start && hour <= marketHours.end;
        });
    }
    
    /**
     * Get timezone string for market
     */
    getMarketTimezone(market) {
        const timezones = {
            'Australia': 'Australia/Sydney',
            'Japan': 'Asia/Tokyo',
            'Hong Kong': 'Asia/Hong_Kong',
            'China': 'Asia/Shanghai',
            'UK': 'Europe/London',
            'Germany': 'Europe/Berlin',
            'France': 'Europe/Paris',
            'US': 'America/New_York'
        };
        return timezones[market] || 'Australia/Sydney';
    }
    
    /**
     * Calculate the time range covering all active trading periods
     */
    calculateTradingTimeRange(marketGroups) {
        let minTime = Infinity;
        let maxTime = -Infinity;
        
        for (const [market, symbols] of marketGroups.entries()) {
            symbols.forEach(symbolData => {
                symbolData.data.forEach(point => {
                    minTime = Math.min(minTime, point.timestamp_ms);
                    maxTime = Math.max(maxTime, point.timestamp_ms);
                });
            });
        }
        
        // Add some padding (30 minutes before and after)
        const padding = 30 * 60 * 1000; // 30 minutes in ms
        
        return {
            start: minTime === Infinity ? Date.now() - 8 * 60 * 60 * 1000 : minTime - padding,
            end: maxTime === -Infinity ? Date.now() : maxTime + padding
        };
    }
    
    /**
     * Add market session indicators for trading hours only
     */
    addTradingHoursMarketSessions(markAreas, marketGroups) {
        const sessionColors = {
            "Australia": "rgba(16, 185, 129, 0.15)",  // emerald
            "Japan": "rgba(59, 130, 246, 0.15)",      // blue  
            "Hong Kong": "rgba(139, 92, 246, 0.15)",  // purple
            "China": "rgba(239, 68, 68, 0.15)",       // red
            "UK": "rgba(245, 158, 11, 0.15)",         // amber
            "Germany": "rgba(249, 115, 22, 0.15)",    // orange
            "France": "rgba(99, 102, 241, 0.15)",     // indigo
            "US": "rgba(6, 182, 212, 0.15)"           // cyan
        };
        
        for (const [market, symbols] of marketGroups.entries()) {
            if (symbols.length === 0) continue;
            
            // Find the time range for this market's data
            let marketStart = Infinity;
            let marketEnd = -Infinity;
            
            symbols.forEach(symbolData => {
                symbolData.data.forEach(point => {
                    marketStart = Math.min(marketStart, point.timestamp_ms);
                    marketEnd = Math.max(marketEnd, point.timestamp_ms);
                });
            });
            
            if (marketStart !== Infinity && marketEnd !== -Infinity) {
                markAreas.push([
                    { 
                        xAxis: marketStart,
                        itemStyle: { 
                            color: sessionColors[market] || 'rgba(128, 128, 128, 0.1)',
                            borderColor: sessionColors[market]?.replace('0.15', '0.3') || 'rgba(128, 128, 128, 0.3)'
                        }
                    },
                    { 
                        xAxis: marketEnd,
                        itemStyle: { 
                            color: sessionColors[market] || 'rgba(128, 128, 128, 0.1)',
                            borderColor: sessionColors[market]?.replace('0.15', '0.3') || 'rgba(128, 128, 128, 0.3)'
                        }
                    }
                ]);
            }
        }
    }
    
    /**
     * Get subtitle showing active trading periods
     */
    getTradingHoursSubtext(marketGroups) {
        const activeMarkets = Array.from(marketGroups.keys());
        
        if (activeMarkets.length === 0) {
            return 'No active trading sessions';
        } else if (activeMarkets.length === 1) {
            const market = activeMarkets[0];
            const duration = this.getTradingDuration(market);
            return `${this.getMarketDisplayName(market)} Trading Session (${duration}h)`;
        } else {
            return `${activeMarkets.length} Active Markets: ${activeMarkets.map(m => this.getMarketDisplayName(m)).join(', ')}`;
        }
    }
    
    /**
     * Get trading duration for market
     */
    getTradingDuration(market) {
        const durations = {
            'Australia': 6,   // 10am-4pm
            'Japan': 6,       // 9am-3pm
            'Hong Kong': 7,   // 9am-4pm
            'China': 6,       // 9am-3pm
            'UK': 8,          // 8am-4pm
            'Germany': 8,     // 9am-5pm
            'France': 8,      // 9am-5pm
            'US': 7           // 9:30am-4pm (simplified to 7h)
        };
        return durations[market] || 6;
    }
    
    /**
     * Legacy support - redirect to Sydney chart
     */
    generate24HChartOption() {
        return this.generateSydneyChartOption();
    }
    
    /**
     * Get Sydney chart subtitle with current context
     */
    getSydneyChartSubtext() {
        let subtext = 'Live market data starting from 10am Sydney time';
        
        if (this.state.sydneyContext) {
            const phase = this.state.sydneyContext.market_day_phase || '';
            subtext += ` • ${phase}`;
        }
        
        return subtext;
    }
    
    /**
     * Add Sydney-based market session indicators to chart
     */
    addSydneyMarketSessions(markAreas) {
        if (!this.state.marketSessionsData || !Array.isArray(this.state.marketSessionsData)) {
            return;
        }
        
        const sessionColors = {
            "Australia": "rgba(16, 185, 129, 0.15)",  // emerald
            "Japan": "rgba(59, 130, 246, 0.15)",      // blue
            "Hong Kong": "rgba(139, 92, 246, 0.15)",  // purple
            "China": "rgba(239, 68, 68, 0.15)",       // red
            "UK": "rgba(245, 158, 11, 0.15)",         // amber
            "Germany": "rgba(249, 115, 22, 0.15)",    // orange
            "France": "rgba(99, 102, 241, 0.15)",     // indigo
            "US": "rgba(6, 182, 212, 0.15)"           // cyan
        };
        
        this.state.marketSessionsData.forEach(session => {
            if (session.open_sydney && session.close_sydney) {
                const openTime = new Date(session.open_sydney).getTime();
                const closeTime = new Date(session.close_sydney).getTime();
                
                markAreas.push([
                    { xAxis: openTime },
                    { xAxis: closeTime }
                ]);
            }
        });
    }
    
    /**
     * Get market display name with emoji
     */
    getMarketDisplayName(market) {
        const displayNames = {
            'Australia': '🇦🇺',
            'Japan': '🇯🇵',
            'Hong Kong': '🇭🇰',
            'China': '🇨🇳',
            'UK': '🇬🇧',
            'Germany': '🇩🇪',
            'France': '🇫🇷',
            'US': '🇺🇸'
        };
        return displayNames[market] || '';
    }
    
    /**
     * Generate standard chart option
     */
    generateStandardChartOption() {
        const colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#f97316', '#06b6d4', '#84cc16'];
        const series = [];
        let colorIndex = 0;
        
        for (const [symbol, data] of this.state.chartData.entries()) {
            // Default to percentage view for standard analysis
            series.push({
                name: symbol,
                type: 'line',
                data: data.map(point => [point.timestamp_ms, point.percentage_change]),
                smooth: true,
                symbol: 'none',
                lineStyle: { width: 2 },
                color: colors[colorIndex % colors.length]
            });
            colorIndex++;
        }
        
        // Default to percentage view for standard analysis
        return {
            title: {
                text: 'Standard Market Analysis - Percentage Changes',
                left: 'center',
                textStyle: { fontSize: 16, fontWeight: 'bold', color: '#374151' }
            },
            tooltip: {
                trigger: 'axis',
                axisPointer: { type: 'cross' },
                backgroundColor: 'rgba(255, 255, 255, 0.95)',
                borderColor: '#e5e7eb',
                textStyle: { color: '#374151' }
            },
            legend: {
                top: 40,
                type: 'scroll'
            },
            grid: {
                left: '3%',
                right: '4%',
                bottom: '8%',
                top: '15%',
                containLabel: true
            },
            xAxis: {
                type: 'time',
                axisLine: { lineStyle: { color: '#d1d5db' } },
                axisLabel: { color: '#6b7280' }
            },
            yAxis: {
                type: 'value',
                axisLine: { lineStyle: { color: '#d1d5db' } },
                axisLabel: { 
                    color: '#6b7280',
                    formatter: '{value}%'
                },
                splitLine: { lineStyle: { color: '#f3f4f6' } }
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
                textStyle: { fontSize: 16, color: '#9ca3af' }
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
        
        const cards = Array.from(this.state.chartData.entries()).map(([symbol, data]) => {
            const latestPoint = data[data.length - 1];
            const change = latestPoint?.percentage_change || 0;
            const isPositive = change >= 0;
            const info = this.state.symbolsDatabase.get(symbol) || { name: symbol, market: 'Unknown' };
            
            return `
                <div class="performance-card bg-gray-50 rounded-lg p-4">
                    <div class="flex items-center justify-between mb-2">
                        <div class="font-medium text-gray-900">${symbol}</div>
                        <div class="text-xs text-gray-500">${info.market}</div>
                    </div>
                    <div class="text-sm text-gray-600 mb-3">${info.name}</div>
                    <div class="flex items-center justify-between">
                        <span class="text-lg font-bold ${isPositive ? 'text-success-600' : 'text-danger-600'}">
                            ${isPositive ? '+' : ''}${change.toFixed(2)}%
                        </span>
                        <i class="fas fa-arrow-${isPositive ? 'up' : 'down'} ${isPositive ? 'text-success-500' : 'text-danger-500'}"></i>
                    </div>
                </div>
            `;
        }).join('');
        
        grid.innerHTML = cards;
    }
    
    /**
     * Update market sessions display for 24H mode
     */
    updateMarketSessions() {
        const sessionsContainer = document.getElementById('market-sessions');
        
        if (!this.state.marketHours) {
            sessionsContainer.classList.add('hidden');
            return;
        }
        
        sessionsContainer.classList.remove('hidden');
        
        const currentHour = new Date().getUTCHours();
        const grid = sessionsContainer.querySelector('.grid');
        
        const sessionCards = Object.entries(this.state.marketHours).map(([market, hours]) => {
            const isActive = currentHour >= hours.open && currentHour <= hours.close;
            const statusClass = isActive ? 'border-success-500 bg-success-50' : 'border-gray-300 bg-gray-50';
            const statusIcon = isActive ? 'text-success-600 fa-circle' : 'text-gray-400 fa-circle';
            const statusText = isActive ? 'OPEN' : 'CLOSED';
            
            return `
                <div class="border-2 ${statusClass} rounded-lg p-3 text-center">
                    <div class="flex items-center justify-center mb-2">
                        <i class="fas ${statusIcon} text-xs mr-2"></i>
                        <span class="text-xs font-medium ${isActive ? 'text-success-700' : 'text-gray-600'}">${statusText}</span>
                    </div>
                    <div class="font-semibold text-sm text-gray-900 mb-1">${market}</div>
                    <div class="text-xs text-gray-600">
                        ${String(hours.open).padStart(2, '0')}:00 - ${String(hours.close).padStart(2, '0')}:00
                    </div>
                </div>
            `;
        }).join('');
        
        grid.innerHTML = sessionCards;
    }
    
     /**
     * Handle chart type change
     */
    handleChartTypeChange() {
        const chartType = document.getElementById('chart-type').value;
        const candlestickContainer = document.getElementById('candlestick-interval-container');
        
        // Show/hide candlestick interval selector
        if (chartType === 'candlestick') {
            candlestickContainer.style.display = 'block';
            this.showToast('Candlestick mode: Select interval from 5 minutes to 1 day', 'info');
        } else {
            candlestickContainer.style.display = 'none';
        }
        
        if (this.state.chartData.size > 0) {
            this.updateChart();
        }
    }

    /**
     * Handle candlestick interval change
     */
    handleCandlestickIntervalChange() {
        if (this.state.chartData.size > 0) {
            this.updateChart();
        }
    }
    
    /**
     * Handle analysis mode change
     */
    handleAnalysisModeChange() {
        const analysisMode = document.getElementById('analysis-mode').value;
        const chartTypeContainer = document.getElementById('chart-type-container');
        
        if (analysisMode === 'global-24h') {
            // Hide chart type selector for Global 24H mode
            chartTypeContainer.style.display = 'none';
            // Clear current selection and show instructions
            this.state.selectedSymbols.clear();
            this.updateSelectedSymbolsDisplay();
            document.getElementById('market-sessions').classList.remove('hidden');
            this.showToast('Global 24H Market Flow mode - Click Analyze to track global markets', 'info');
        } else {
            // Show chart type selector for Standard mode
            chartTypeContainer.style.display = 'block';
            // Clear any global data
            this.state.chartData.clear();
            this.state.marketHours = null;
            this.updateChart();
            document.getElementById('performance-summary').classList.add('hidden');
            document.getElementById('market-sessions').classList.add('hidden');
            this.showToast('Standard Analysis mode - Select individual symbols to analyze', 'info');
        }
    }
    
    /**
     * Handle clear
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
            button.innerHTML = '<i class="fas fa-chart-area mr-2"></i>Analyze';
        }
    }
    
    /**
     * Update API status
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
            case 'disconnected':
                dot.classList.add('bg-yellow-500');
                text.textContent = message;
                text.className = 'text-xs text-yellow-600';
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
        document.getElementById('api-url').value = this.state.apiBaseUrl || '';
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
    async saveSettings() {
        const apiUrl = document.getElementById('api-url').value.trim();
        const autoRefresh = document.getElementById('auto-refresh').checked;
        const refreshInterval = parseInt(document.getElementById('refresh-interval').value);
        
        // Update state
        this.state.apiBaseUrl = apiUrl;
        this.state.settings.autoRefresh = autoRefresh;
        this.state.settings.refreshInterval = refreshInterval;
        
        // Save to localStorage
        localStorage.setItem('gsmt-api-url', apiUrl);
        localStorage.setItem('gsmt-settings', JSON.stringify(this.state.settings));
        
        // Test new API connection
        if (apiUrl) {
            await this.checkApiConnection();
            await this.loadSymbolsDatabase();
        }
        
        this.applySettings();
        this.hideSettings();
        this.showToast('Settings saved successfully', 'success');
    }
    
    /**
     * Load settings
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
     * Apply settings
     */
    applySettings() {
        // Auto-refresh functionality
        if (this.state.refreshTimer) {
            clearInterval(this.state.refreshTimer);
            this.state.refreshTimer = null;
        }
        
        if (this.state.settings.autoRefresh && this.state.selectedSymbols.size > 0) {
            this.state.refreshTimer = setInterval(() => {
                this.handleAnalyze();
            }, this.state.settings.refreshInterval * 1000);
        }
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
     * Handle document clicks
     */
    handleDocumentClick(event) {
        if (!event.target.closest('#symbol-search') && !event.target.closest('#symbol-suggestions')) {
            document.getElementById('symbol-suggestions').classList.add('hidden');
        }
    }
    
    /**
     * Handle resize
     */
    handleResize() {
        if (this.state.chartInstance) {
            this.state.chartInstance.resize();
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
        toast.className = `toast flex items-center p-4 rounded-lg shadow-lg text-white ${colors[type]} transform transition-all duration-300`;
        toast.innerHTML = `
            <i class="fas ${icons[type]} mr-3"></i>
            <span class="flex-1">${message}</span>
            <button onclick="this.parentElement.remove()" class="ml-3 text-white hover:text-gray-200">
                <i class="fas fa-times"></i>
            </button>
        `;
        
        container.appendChild(toast);
        
        setTimeout(() => {
            if (toast.parentElement) {
                toast.classList.add('toast-exit');
                setTimeout(() => toast.remove(), 300);
            }
        }, 5000);
    }
    
    /**
     * Debounce utility
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

// Initialize application when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new GSMTApp();
});

// Export for external use
window.GSMTApp = GSMTApp;