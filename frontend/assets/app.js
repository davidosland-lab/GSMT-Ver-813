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
        
        // Calendar functionality
        this.selectedDate = new Date();
        this.isHistoricalMode = false;
        this.calendarVisible = false;
        this.currentMonth = new Date().getMonth();
        this.currentYear = new Date().getFullYear();
        
        this.init();
    }

    detectApiUrl() {
        // For sandbox environment, API runs on port 8000
        const currentHost = window.location.hostname;
        if (currentHost.includes('e2b.dev')) {
            // Extract sandbox ID from hostname like "3000-sandbox-id.e2b.dev"
            const sandboxId = currentHost.split('-').slice(1).join('-');
            const apiHost = `8000-${sandboxId}`;
            return `https://${apiHost}/api`;
        }
        // For local deployment fallback
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
        
        // Time interval change
        document.getElementById('time-interval').addEventListener('change', 
            this.handleIntervalChange.bind(this));
        
        // Time period change (24h vs 48h)
        document.getElementById('time-period').addEventListener('change', 
            this.handleTimePeriodChange.bind(this));
        
        // Plot mode change (combined vs individual)
        document.getElementById('plot-mode').addEventListener('change', 
            this.handlePlotModeChange.bind(this));

        // New dropdown functionality
        document.getElementById('region-dropdown').addEventListener('change', 
            this.handleRegionChange.bind(this));
        
        document.getElementById('markets-dropdown').addEventListener('change', 
            this.handleMarketDropdownChange.bind(this));
        
        document.getElementById('add-market-btn').addEventListener('click', 
            this.addSelectedMarket.bind(this));

        // Preset buttons
        document.querySelectorAll('.preset-btn').forEach(btn => {
            btn.addEventListener('click', this.handlePresetSelection.bind(this));
        });
        
        // Calendar functionality
        const datePickerBtn = document.getElementById('date-picker-btn');
        if (datePickerBtn) {
            datePickerBtn.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                this.toggleCalendar();
            });
        }
        
        const prevDayBtn = document.getElementById('prev-day-btn');
        if (prevDayBtn) {
            prevDayBtn.addEventListener('click', this.goToPreviousDay.bind(this));
        }
        
        const nextDayBtn = document.getElementById('next-day-btn');
        if (nextDayBtn) {
            nextDayBtn.addEventListener('click', this.goToNextDay.bind(this));
        }
        
        const todayBtn = document.getElementById('today-btn');
        if (todayBtn) {
            todayBtn.addEventListener('click', this.goToToday.bind(this));
        }
        
        const prevMonthBtn = document.getElementById('prev-month');
        if (prevMonthBtn) {
            prevMonthBtn.addEventListener('click', this.goToPreviousMonth.bind(this));
        }
        
        const nextMonthBtn = document.getElementById('next-month');
        if (nextMonthBtn) {
            nextMonthBtn.addEventListener('click', this.goToNextMonth.bind(this));
        }
        
        // Close calendar when clicking outside
        document.addEventListener('click', (e) => {
            if (!e.target.closest('#date-picker-btn') && !e.target.closest('#calendar-dropdown')) {
                this.hideCalendar();
            }
        });
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
        // Store the indices data for dropdown population
        this.allIndicesData = suggestedIndices;
        console.log('📊 Loaded market data:', Object.keys(suggestedIndices));
    }

    handleRegionChange(event) {
        const selectedRegion = event.target.value;
        const marketsDropdown = document.getElementById('markets-dropdown');
        const addBtn = document.getElementById('add-market-btn');
        
        if (!selectedRegion) {
            marketsDropdown.disabled = true;
            marketsDropdown.innerHTML = '<option value="">Select a region first...</option>';
            addBtn.disabled = true;
            return;
        }

        // Populate markets dropdown based on selected region
        const markets = this.allIndicesData[selectedRegion] || [];
        marketsDropdown.innerHTML = '<option value="">Choose a market...</option>';
        
        markets.forEach(market => {
            const option = document.createElement('option');
            option.value = market.symbol;
            option.textContent = `${market.name} (${market.symbol})`;
            option.dataset.marketInfo = JSON.stringify(market);
            marketsDropdown.appendChild(option);
        });

        marketsDropdown.disabled = false;
        addBtn.disabled = true; // Enable only when market is selected
    }

    handleMarketDropdownChange(event) {
        const addBtn = document.getElementById('add-market-btn');
        addBtn.disabled = !event.target.value;
    }

    addSelectedMarket() {
        const regionDropdown = document.getElementById('region-dropdown');
        const marketsDropdown = document.getElementById('markets-dropdown');
        
        if (!marketsDropdown.value) return;

        const symbol = marketsDropdown.value;
        const marketInfo = JSON.parse(marketsDropdown.selectedOptions[0].dataset.marketInfo);
        
        // Add to selected indices
        if (!this.selectedIndices.has(symbol)) {
            this.selectedIndices.add(symbol);
            this.updateSelectedMarketsDisplay();
            this.updateAnalyzeButton();
            
            // Reset dropdowns after adding
            marketsDropdown.value = '';
            document.getElementById('add-market-btn').disabled = true;
            
            this.showToast(`Added ${marketInfo.name}`, 'success');
        } else {
            this.showToast(`${marketInfo.name} already selected`, 'warning');
        }
    }

    handlePresetSelection(event) {
        const preset = event.target.dataset.preset;
        
        switch (preset) {
            case 'major-indices':
                this.loadPreset(['^GSPC', '^IXIC', '^DJI', '^FTSE', '^GDAXI', '^N225', '^AXJO']);
                break;
            case 'global-flow':
                this.loadPreset(['^N225', '^HSI', '^FTSE', '^GDAXI', '^GSPC', '^IXIC']);
                break;
            case 'tech-stocks':
                this.loadPreset(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']);
                break;
            case 'commodities':
                this.loadPreset(['GC=F', 'CL=F', 'SI=F', 'NG=F', 'ZC=F']);
                break;
            case 'crypto':
                this.loadPreset(['BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD', 'ADA-USD']);
                break;
            case 'clear-all':
                this.selectedIndices.clear();
                this.updateSelectedMarketsDisplay();
                this.updateAnalyzeButton();
                this.showToast('All markets cleared', 'info');
                break;
        }
    }

    loadPreset(symbols) {
        this.selectedIndices.clear();
        symbols.forEach(symbol => this.selectedIndices.add(symbol));
        this.updateSelectedMarketsDisplay();
        this.updateAnalyzeButton();
        this.showToast(`Loaded ${symbols.length} markets`, 'success');
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

    updateSelectedMarketsDisplay() {
        const container = document.getElementById('selected-markets');
        const gridContainer = document.getElementById('selected-markets-grid');
        const countElement = document.getElementById('selection-count');
        
        if (this.selectedIndices.size === 0) {
            container.classList.add('hidden');
            return;
        }
        
        container.classList.remove('hidden');
        countElement.textContent = `${this.selectedIndices.size} market${this.selectedIndices.size === 1 ? '' : 's'} selected`;
        
        gridContainer.innerHTML = '';
        
        this.selectedIndices.forEach(symbol => {
            const symbolInfo = this.allSymbols.get(symbol);
            if (!symbolInfo) return;
            
            const marketCard = document.createElement('div');
            marketCard.className = 'bg-gray-50 rounded-lg p-4 border border-gray-200';
            marketCard.innerHTML = `
                <div class="flex items-start justify-between mb-3">
                    <div>
                        <h4 class="text-sm font-medium text-gray-900">${symbolInfo.name}</h4>
                        <p class="text-xs text-gray-600">${symbol} • ${symbolInfo.market}</p>
                    </div>
                    <button onclick="window.tracker.removeMarket('${symbol}')" 
                            class="text-gray-400 hover:text-red-600 transition-colors">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                
                <!-- Market Intelligence Dropdown -->
                <div class="space-y-2">
                    <label class="block text-xs font-medium text-gray-700">Market Intelligence</label>
                    <select id="intelligence-${symbol}" class="w-full text-xs border border-gray-300 rounded px-2 py-1.5">
                        <option value="">Loading intelligence...</option>
                    </select>
                    
                    <!-- Intelligence Details -->
                    <div id="intelligence-details-${symbol}" class="hidden mt-2 p-2 bg-white rounded border text-xs">
                        <div class="text-gray-600">Select an item above to view details</div>
                    </div>
                </div>
            `;
            
            gridContainer.appendChild(marketCard);
            
            // Load market intelligence for this specific market
            this.loadMarketIntelligence(symbol);
        });
    }

    updateSelectedIndicesDisplay() {
        // This function is now replaced by updateSelectedMarketsDisplay
        // but keeping it for backward compatibility
        this.updateSelectedMarketsDisplay();
    }

    removeIndex(symbol) {
        this.selectedIndices.delete(symbol);
        
        // Update checkbox
        const checkbox = document.querySelector(`input[value="${symbol}"]`);
        if (checkbox) checkbox.checked = false;
        
        this.updateSelectedIndicesDisplay();
        this.updateAnalyzeButton();
    }

    removeMarket(symbol) {
        this.selectedIndices.delete(symbol);
        this.updateSelectedMarketsDisplay();
        this.updateAnalyzeButton();
        this.showToast(`Removed ${symbol}`, 'info');
    }

    updateAnalyzeButton() {
        const button = document.getElementById('analyze-btn');
        button.disabled = this.selectedIndices.size === 0;
    }

    async analyzeSelectedIndices() {
        if (this.selectedIndices.size === 0) return;
        
        // Check if we're in historical mode and delegate to historical data loading
        if (this.isHistoricalMode) {
            return await this.loadHistoricalData();
        }
        
        this.showLoading(true);
        
        try {
            const chartType = document.getElementById('chart-type').value;
            const interval = parseInt(document.getElementById('time-interval').value);
            const timePeriod = document.getElementById('time-period').value;
            
            const response = await fetch(`${this.apiBaseUrl}/analyze`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    symbols: Array.from(this.selectedIndices),
                    chart_type: chartType,
                    interval_minutes: interval,
                    time_period: timePeriod
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
        
        // Check plot mode selection
        const plotMode = document.getElementById('plot-mode').value;
        
        // Check if we have market groups and individual mode is selected
        if (plotMode === 'individual' && data.market_groups && Object.keys(data.market_groups).length > 1) {
            this.updateIndividualMarketCharts(data);
        } else {
            this.updateCombinedChart(data);
        }
    }
    
    updateCombinedChart(data) {
        const chartType = document.getElementById('chart-type').value;
        const series = [];
        const xAxisData = [];
        const markLines = [];
        
        // Get time points from first symbol - format based on interval
        const firstSymbol = Object.keys(data.data)[0];
        const selectedInterval = parseInt(document.getElementById('time-interval').value);
        
        if (firstSymbol) {
            data.data[firstSymbol].forEach(point => {
                // Parse AEST timestamp format "2025-09-08 10:00:00 AEST" 
                // Extract the date and time parts directly without timezone conversion
                const timestampStr = point.timestamp.replace(' AEST', '');
                const [datePart, timePart] = timestampStr.split(' ');
                const [year, monthNum, dayNum] = datePart.split('-').map(Number);
                const [hourNum, minuteNum] = timePart.split(':').map(Number);
                
                const month = monthNum.toString().padStart(2, '0');
                const day = dayNum.toString().padStart(2, '0');
                const hour = hourNum.toString().padStart(2, '0');
                const minute = minuteNum.toString().padStart(2, '0');
                
                // Format based on interval
                if (selectedInterval === 5) {
                    // For 5-minute intervals, show HH:MM
                    xAxisData.push(`${hour}:${minute}`);
                } else if (selectedInterval === 30) {
                    // For 30-minute intervals, show HH:MM
                    xAxisData.push(`${hour}:${minute}`);
                } else {
                    // For 1-hour intervals, show MM/DD HH:00 (original format)
                    xAxisData.push(`${month}/${day} ${hour}:00`);
                }
            });
        }
        
        // Calculate data range for intelligent y-axis scaling
        let allValues = [];
        
        // Create series for each selected index
        Object.entries(data.data).forEach(([symbol, points]) => {
            const symbolInfo = data.metadata[symbol];
            
            // Check if this is previous day data
            const isPreviousDay = symbol.endsWith('_prev_day');
            const baseName = isPreviousDay ? symbolInfo.name.replace(' (Previous Day)', '') : symbolInfo.name;
            
            if (chartType === 'candlestick') {
                // For candlestick charts, preserve all 24 time slots with null for market-closed periods
                console.log(`🕯️ Processing percentage-based candlestick data for ${symbol}, ${points.length} total points`);
                
                // Create candlestick data array - filter to only include market open periods
                const candlestickData = [];
                const candlestickXAxis = [];
                
                points.forEach((point, index) => {
                    if (point.market_open && point.open !== null && point.high !== null && 
                        point.low !== null && point.close !== null) {
                        // For percentage-based candlesticks, data is already in percentage format
                        // ECharts candlestick format: [open, close, low, high] (all as percentages)
                        const ohlc = [point.open, point.close, point.low, point.high];
                        
                        // Add all OHLC values to allValues for y-axis scaling
                        allValues.push(point.open, point.close, point.low, point.high);
                        
                        candlestickData.push(ohlc);
                        candlestickXAxis.push(xAxisData[index]); // Store corresponding x-axis label
                        
                        if (index >= 14 && index <= 21) {
                            console.log(`📊 Hour ${index}: OHLC% = [${ohlc.map(v => v.toFixed(2)).join(', ')}]%, market_open: ${point.market_open}`);
                        }
                    } else {
                        if (index >= 14 && index <= 21) {
                            console.log(`⏸️ Hour ${index}: Market closed or null data, market_open: ${point.market_open}`);
                        }
                    }
                });
                
                console.log(`🕯️ Candlestick data for ${symbol}: ${candlestickData.length} valid out of ${points.length} total points`);
                console.log(`🔍 Sample candlestick data:`, candlestickData.slice(0, 3));
                
                if (candlestickData.length === 0) {
                    console.warn(`⚠️ No valid candlestick data for ${symbol} - all market closed or null OHLC values`);
                } else {
                    // Use the filtered x-axis data that matches candlestick data points
                    if (!window.candlestickXAxisData) {
                        window.candlestickXAxisData = candlestickXAxis;
                    }
                    
                    // Enhanced candlestick series with market-specific colors
                    const marketColors = this.getMarketColors(symbolInfo.market);
                    
                    // Adjust styling for previous day data
                    const seriesName = isPreviousDay ? 
                        `${baseName} (${symbolInfo.market}) - Previous Day` : 
                        `${symbolInfo.name} (${symbolInfo.market})`;
                    
                    const candlestickSeries = {
                        name: seriesName,
                        type: 'candlestick',
                        data: candlestickData,
                        itemStyle: isPreviousDay ? {
                            // Previous day styling - more transparent/faded
                            color: this.addAlpha(marketColors.bull, 0.4),      // Faded bull candle
                            color0: this.addAlpha(marketColors.bear, 0.4),     // Faded bear candle
                            borderColor: this.addAlpha(marketColors.bull, 0.6), // Faded bull border
                            borderColor0: this.addAlpha(marketColors.bear, 0.6) // Faded bear border
                        } : {
                            color: marketColors.bull,      // Bull candle color
                            color0: marketColors.bear,     // Bear candle color  
                            borderColor: marketColors.bull, // Bull border
                            borderColor0: marketColors.bear // Bear border
                        },
                        tooltip: {
                            formatter: function(param) {
                                const [open, close, low, high] = param.value;
                                const change = (close - open).toFixed(2);
                                const changeColor = close >= open ? marketColors.bull : marketColors.bear;
                                return `
                                    <div style="margin: 0px 0 0; line-height:1;">
                                        <div style="margin: 0px 0 0; line-height:1;">
                                            ${param.marker}<strong>${param.seriesName}</strong><br/>
                                            <span style="display:inline-block;margin-right:5px;border-radius:10px;width:9px;height:9px;background-color:#333;"></span>
                                            Open: ${open.toFixed(3)}%<br/>
                                            <span style="display:inline-block;margin-right:5px;border-radius:10px;width:9px;height:9px;background-color:#333;"></span>
                                            Close: ${close.toFixed(3)}%<br/>
                                            <span style="display:inline-block;margin-right:5px;border-radius:10px;width:9px;height:9px;background-color:#333;"></span>
                                            High: ${high.toFixed(3)}%<br/>
                                            <span style="display:inline-block;margin-right:5px;border-radius:10px;width:9px;height:9px;background-color:#333;"></span>
                                            Low: ${low.toFixed(3)}%<br/>
                                            <span style="display:inline-block;margin-right:5px;border-radius:10px;width:9px;height:9px;background-color:${changeColor};"></span>
                                            Change: <span style="color:${changeColor}">${change}%</span>
                                        </div>
                                    </div>
                                `;
                            }
                        }
                    };
                    
                    console.log(`📊 Adding percentage candlestick series for ${symbol}: ${candlestickData.length} total points (${candlestickData.filter(d => d !== null).length} with data)`);
                    console.log(`🕯️ Candlestick series configuration:`, candlestickSeries);
                    series.push(candlestickSeries);
                }
            } else {
                // Handle line charts (percentage and price) - filter future data
                const now = new Date();
                const values = points.map((point, index) => {
                    // Parse timestamp to check if it's in the future
                    const timestampStr = point.timestamp.replace(' AEST', '');
                    const pointTime = new Date(timestampStr);
                    
                    // Don't show data points in the future
                    if (pointTime > now) {
                        return null;
                    }
                    
                    // Determine the value to display based on chart type
                    const value = chartType === 'percentage' ? point.percentage_change : point.close;
                    
                    // For y-axis scaling, collect all valid values regardless of market_open status
                    // Filter out extreme outliers that would skew the y-axis inappropriately
                    if (value !== null && !isNaN(value)) {
                        // For percentage charts, filter out extreme outliers (>±50%)
                        if (chartType === 'percentage') {
                            if (Math.abs(value) <= 50) {  // Reasonable percentage change limit
                                allValues.push(value);
                            } else {
                                console.warn(`Filtering extreme outlier: ${value}% for better y-axis scaling`);
                            }
                        } else {
                            // For price charts, no filtering needed
                            allValues.push(value);
                        }
                    }
                    
                    // Only show values when market is open AND we have valid data AND not in future
                    if (point.market_open && value !== null && !isNaN(value)) {
                        return value;
                    } else {
                        return null;  // This creates a gap in the line chart
                    }
                });
                
                series.push({
                    name: isPreviousDay ? `${baseName} (Previous Day)` : symbolInfo.name,
                    type: 'line',
                    data: values,
                    smooth: true,
                    symbol: 'circle',
                    symbolSize: isPreviousDay ? 3 : 4,  // Smaller symbols for previous day
                    lineWidth: isPreviousDay ? 1 : 2,   // Thinner line for previous day
                    connectNulls: false,  // Don't connect across null values (market closed periods)
                    lineStyle: isPreviousDay ? {
                        opacity: 0.5,       // Semi-transparent line
                        type: 'dashed'      // Dashed line style
                    } : undefined,
                    emphasis: {
                        focus: 'series'
                    }
                });
            }
            
            // Add market opening/closing indicators ONLY for selected markets
            // Calculate exact market open/close positions based on known market hours
            let marketOpenStart = -1;
            let marketCloseEnd = -1;
            
            // Filter points to only include past/present data (no future)
            // Get current AEST time for proper comparison
            const nowUTC = new Date();
            const aestOffsetHours = 10; // AEST is UTC+10
            const nowAEST = new Date(nowUTC.getTime() + (aestOffsetHours * 60 * 60 * 1000));
            
            // Format current AEST time to match our timestamp format
            const currentAESTStr = nowAEST.getUTCFullYear() + '-' + 
                String(nowAEST.getUTCMonth() + 1).padStart(2, '0') + '-' + 
                String(nowAEST.getUTCDate()).padStart(2, '0') + ' ' +
                String(nowAEST.getUTCHours()).padStart(2, '0') + ':' + 
                String(nowAEST.getUTCMinutes()).padStart(2, '0') + ':' +
                String(nowAEST.getUTCSeconds()).padStart(2, '0');
            

            
            const validPoints = points.filter((point) => {
                const timestampStr = point.timestamp.replace(' AEST', '');
                const isValid = timestampStr <= currentAESTStr;
                return isValid;
            });
            
            // Define exact market close times in AEST hours (24-hour format)
            const marketCloseTimes = {
                'Japan': { hour: 16, minute: 30 },      // 16:30 AEST (JST 15:30 market close)
                'Australia': { hour: 16, minute: 0 }    // 16:00 AEST (4:00 PM market close)  
            };
            

            
            // Find market open start (first market_open point)
            validPoints.forEach((point, index) => {
                if (point.market_open && marketOpenStart === -1) {
                    marketOpenStart = index; // First market open period
                }
            });
            
            // Calculate exact market close position based on timeline
            if (marketCloseTimes[symbolInfo.market]) {
                const targetClose = marketCloseTimes[symbolInfo.market];
                
                // Calculate the expected timeline position for market close
                // Timeline starts at 10:00 AEST and each point represents interval_minutes intervals
                const timelineStartHour = 10;
                const timelineStartMinute = 0;
                const intervalMinutes = 5; // Should match the interval_minutes from the request
                
                const timelineStartTotalMinutes = timelineStartHour * 60 + timelineStartMinute;
                const targetTotalMinutes = targetClose.hour * 60 + targetClose.minute;
                const minutesFromStart = targetTotalMinutes - timelineStartTotalMinutes;
                
                // Calculate the expected index on the timeline
                const expectedIndex = Math.floor(minutesFromStart / intervalMinutes);
                
                // Use the calculated index if it's within the full data range
                if (expectedIndex >= 0 && expectedIndex < points.length) {
                    marketCloseEnd = expectedIndex;
                } else {
                    marketCloseEnd = -1;
                }
            }
            
            // Only add market lines if we found valid market sessions for THIS symbol's market
            if (marketOpenStart !== -1 && marketOpenStart < xAxisData.length) {
                // Calculate label offset based on symbol index to prevent overlapping
                const symbolIndex = Object.keys(data.data).indexOf(symbol);
                const labelOffset = symbolIndex * 20; // 20px offset per symbol
                
                markLines.push({
                    name: `${symbolInfo.market} Open`,
                    xAxis: xAxisData[marketOpenStart],
                    lineStyle: { color: '#10b981', type: 'dashed', width: 2 },
                    label: { 
                        show: true, 
                        position: 'start',
                        formatter: `${symbolInfo.market}\nOpen`,
                        color: '#10b981',
                        fontSize: 10,
                        offset: [5, labelOffset], // Horizontal and vertical offset
                        backgroundColor: 'rgba(255, 255, 255, 0.9)',
                        padding: [3, 6],
                        borderRadius: 4
                    }
                });
            }
            
            if (marketCloseEnd !== -1 && marketCloseEnd < xAxisData.length) {
                // Market close line at the actual end of trading data
                const symbolIndex = Object.keys(data.data).indexOf(symbol);
                const labelOffset = symbolIndex * 20; // 20px offset per symbol
                
                markLines.push({
                    name: `${symbolInfo.market} Close`, 
                    xAxis: xAxisData[marketCloseEnd],
                    lineStyle: { color: '#ef4444', type: 'dashed', width: 2 },
                    label: { 
                        show: true, 
                        position: 'end',
                        formatter: `${symbolInfo.market}\nClose`,
                        color: '#ef4444',
                        fontSize: 10,
                        offset: [-5, labelOffset], // Horizontal and vertical offset
                        backgroundColor: 'rgba(255, 255, 255, 0.9)',
                        padding: [3, 6],
                        borderRadius: 4
                    }
                });
            }
        });
        
        // Add current time indicator for rolling window - match new x-axis format
        const now = new Date();
        // Convert current time to AEST (UTC+10)
        const aestOffset = 10 * 60; // 10 hours in minutes
        const aestNow = new Date(now.getTime() + (aestOffset * 60 * 1000));
        const currentMonth = (aestNow.getUTCMonth() + 1).toString().padStart(2, '0');
        const currentDay = aestNow.getUTCDate().toString().padStart(2, '0');
        const currentHour = aestNow.getUTCHours().toString().padStart(2, '0');
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
                    formatter: 'Now (AEST)'
                }
            });
        }
        
        // Clean up previous candlestick x-axis data
        if (chartType !== 'candlestick') {
            window.candlestickXAxisData = null;
        }
        
        console.log(`📈 Creating chart with type: ${chartType}, series count: ${series.length}`);
        if (chartType === 'candlestick') {
            console.log(`🕯️ Candlestick series data:`, series.map(s => ({name: s.name, type: s.type, dataLength: s.data?.length})));
        }
        
        // Update title to include interval and time period information
        const currentInterval = parseInt(document.getElementById('time-interval').value);
        const timePeriod = document.getElementById('time-period').value;
        const intervalText = currentInterval === 5 ? '5-Minute' : currentInterval === 30 ? '30-Minute' : 'Hourly';
        const periodText = timePeriod === '48h' ? '48-Hour' : '24-Hour';
        const chartTypeText = chartType === 'candlestick' ? 'Candlestick View' : 
                             chartType === 'percentage' ? 'Percentage Change' : 'Price Values';
        
        const option = {
            title: {
                text: `${periodText} Market Timeline (${intervalText}) - ${chartTypeText}`,
                left: 'center',
                textStyle: { fontSize: 16, fontWeight: 'bold' }
            },
            tooltip: {
                trigger: 'axis',
                axisPointer: { type: 'cross' },
                formatter: function(params) {
                    const currentChartType = document.getElementById('chart-type').value;
                    let result = `<strong>${params[0].name} UTC</strong><br/>`;
                    let hasData = false;
                    
                    params.forEach(param => {
                        if (param.value !== null && param.value !== undefined) {
                            hasData = true;
                            
                            if (currentChartType === 'candlestick' && Array.isArray(param.value)) {
                                // Candlestick tooltip: [open, close, low, high] (all in percentage)
                                const [open, close, low, high] = param.value;
                                const change = (close - open).toFixed(3);
                                const changeColor = close >= open ? '#00da3c' : '#ec0000';
                                
                                result += `${param.marker}<strong>${param.seriesName}</strong><br/>`;
                                result += `&nbsp;&nbsp;Open: ${open.toFixed(3)}%<br/>`;
                                result += `&nbsp;&nbsp;Close: ${close.toFixed(3)}%<br/>`;
                                result += `&nbsp;&nbsp;High: ${high.toFixed(3)}%<br/>`;
                                result += `&nbsp;&nbsp;Low: ${low.toFixed(3)}%<br/>`;
                                result += `&nbsp;&nbsp;<span style="color:${changeColor}">Change: ${change >= 0 ? '+' : ''}${change}%</span><br/>`;
                            } else {
                                // Line chart tooltip
                                const value = currentChartType === 'percentage' 
                                    ? `${param.value.toFixed(2)}%`
                                    : param.value.toFixed(2);
                                result += `${param.marker}${param.seriesName}: ${value}<br/>`;
                            }
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
                data: chartType === 'candlestick' && window.candlestickXAxisData ? 
                      window.candlestickXAxisData : xAxisData,
                name: 'AEST Time',
                nameLocation: 'middle',
                nameGap: 30
            },
            yAxis: this.getYAxisConfig(chartType, allValues),
            series: chartType === 'candlestick' ? 
                // For candlestick charts, add markLine/markArea to the first candlestick series
                series.map((s, index) => {
                    if (index === 0) {
                        return {
                            ...s,
                            markLine: {
                                silent: true,
                                data: markLines
                            },
                            markArea: {
                                silent: true,
                                data: this.getMarketSessionIndicators(chartType),
                                label: {
                                    show: true,
                                    position: 'top',
                                    color: '#666',
                                    fontSize: 10
                                }
                            }
                        };
                    }
                    return s;
                }) :
                // For line charts, use the original approach
                series.concat([{
                    type: 'line',
                    markLine: {
                        silent: true,
                        data: markLines
                    },
                    markArea: {
                        silent: true,
                        data: this.getMarketSessionIndicators(chartType),
                        label: {
                            show: true,
                            position: 'top',
                            color: '#666',
                            fontSize: 10
                        }
                    }
                }])
        };
        
        // Clear and re-initialize chart for candlestick mode to ensure proper rendering
        if (chartType === 'candlestick') {
            this.chartInstance.clear();
        }
        
        this.chartInstance.setOption(option, true);
    }
    
    updateIndividualMarketCharts(data) {
        const chartType = document.getElementById('chart-type').value;
        const selectedInterval = parseInt(document.getElementById('time-interval').value);
        
        // Clear existing chart and prepare for multiple subplots
        this.chartInstance.clear();
        
        const markets = Object.keys(data.market_groups);
        const marketCount = markets.length;
        
        // Calculate grid layout for subplots (e.g., 3 markets = 3 rows)
        const gridHeight = Math.floor(90 / marketCount); // Leave 10% for spacing
        const gridSpacing = Math.floor(10 / (marketCount + 1));
        
        let allSeries = [];
        let allXAxes = [];
        let allYAxes = [];
        let allTitles = [];
        
        markets.forEach((market, marketIndex) => {
            const marketSymbols = data.market_groups[market];
            const marketSeries = [];
            let marketXAxisData = [];
            let allValues = [];
            
            // Generate x-axis data from first symbol in market
            const firstSymbol = Object.keys(marketSymbols)[0];
            if (firstSymbol && marketSymbols[firstSymbol]) {
                marketSymbols[firstSymbol].forEach(point => {
                    const timestampStr = point.timestamp.replace(' AEST', '');
                    const [datePart, timePart] = timestampStr.split(' ');
                    const [year, monthNum, dayNum] = datePart.split('-').map(Number);
                    const [hourNum, minuteNum] = timePart.split(':').map(Number);
                    
                    const month = monthNum.toString().padStart(2, '0');
                    const day = dayNum.toString().padStart(2, '0');
                    const hour = hourNum.toString().padStart(2, '0');
                    const minute = minuteNum.toString().padStart(2, '0');
                    
                    if (selectedInterval === 5 || selectedInterval === 30) {
                        marketXAxisData.push(`${hour}:${minute}`);
                    } else {
                        marketXAxisData.push(`${month}/${day} ${hour}:00`);
                    }
                });
            }
            
            // Create series for each symbol in this market
            Object.entries(marketSymbols).forEach(([symbol, points]) => {
                const symbolInfo = data.metadata[symbol];
                
                if (chartType === 'candlestick') {
                    const candlestickData = [];
                    
                    points.forEach(point => {
                        if (point.market_open && point.open !== null && point.high !== null && 
                            point.low !== null && point.close !== null) {
                            const ohlc = [point.open, point.close, point.low, point.high];
                            allValues.push(point.open, point.close, point.low, point.high);
                            candlestickData.push(ohlc);
                        } else {
                            candlestickData.push(null);
                        }
                    });
                    
                    const marketColors = this.getMarketColors(market);
                    marketSeries.push({
                        name: `${symbolInfo.name}`,
                        type: 'candlestick',
                        data: candlestickData,
                        xAxisIndex: marketIndex,
                        yAxisIndex: marketIndex,
                        itemStyle: {
                            color: marketColors.bull,
                            color0: marketColors.bear,
                            borderColor: marketColors.bull,
                            borderColor0: marketColors.bear
                        }
                    });
                } else {
                    // Line charts
                    const values = points.map(point => {
                        const value = chartType === 'percentage' ? point.percentage_change : point.close;
                        
                        if (value !== null && !isNaN(value)) {
                            // Only show data during market open periods
                            if (point.market_open) {
                                if (chartType === 'percentage' && Math.abs(value) <= 50) {
                                    allValues.push(value);
                                } else if (chartType !== 'percentage') {
                                    allValues.push(value);
                                }
                                return value;
                            }
                        }
                        return null;
                    });
                    
                    marketSeries.push({
                        name: symbolInfo.name,
                        type: 'line',
                        data: values,
                        xAxisIndex: marketIndex,
                        yAxisIndex: marketIndex,
                        smooth: true,
                        symbol: 'circle',
                        symbolSize: 4,
                        lineWidth: 2,
                        connectNulls: false
                    });
                }
            });
            
            // Add market-specific title
            allTitles.push({
                text: `${market} Market`,
                left: 'center',
                top: `${marketIndex * (gridHeight + gridSpacing) + 2}%`,
                textStyle: { fontSize: 14, fontWeight: 'bold' }
            });
            
            // Add x-axis for this market
            allXAxes.push({
                type: 'category',
                data: marketXAxisData,
                gridIndex: marketIndex,
                name: 'AEST Time',
                nameLocation: 'middle',
                nameGap: 25,
                axisLabel: {
                    fontSize: 10
                }
            });
            
            // Add y-axis for this market
            allYAxes.push({
                type: 'value',
                gridIndex: marketIndex,
                axisLabel: {
                    formatter: (chartType === 'percentage' || chartType === 'candlestick') ? '{value}%' : '{value}',
                    fontSize: 10
                },
                scale: true
            });
            
            allSeries = allSeries.concat(marketSeries);
        });
        
        // Create grid configuration for subplots
        const gridConfig = markets.map((market, index) => ({
            left: '5%',
            right: '5%',
            top: `${index * (gridHeight + gridSpacing) + 8}%`,
            bottom: `${(marketCount - index - 1) * (gridHeight + gridSpacing) + 8}%`,
            containLabel: true
        }));
        
        const option = {
            title: allTitles,
            tooltip: {
                trigger: 'axis',
                axisPointer: { type: 'cross' },
                formatter: function(params) {
                    if (!params || params.length === 0) return '';
                    
                    const currentChartType = document.getElementById('chart-type').value;
                    let result = `<strong>${params[0].name} AEST</strong><br/>`;
                    
                    params.forEach(param => {
                        if (param.value !== null && param.value !== undefined) {
                            if (currentChartType === 'candlestick' && Array.isArray(param.value)) {
                                const [open, close, low, high] = param.value;
                                const change = (close - open).toFixed(3);
                                const changeColor = close >= open ? '#00da3c' : '#ec0000';
                                result += `${param.marker}<strong>${param.seriesName}</strong><br/>`;
                                result += `&nbsp;&nbsp;O: ${open.toFixed(3)}% H: ${high.toFixed(3)}% L: ${low.toFixed(3)}% C: ${close.toFixed(3)}%<br/>`;
                                result += `&nbsp;&nbsp;<span style="color:${changeColor}">Δ: ${change >= 0 ? '+' : ''}${change}%</span><br/>`;
                            } else {
                                const value = currentChartType === 'percentage' 
                                    ? `${param.value.toFixed(2)}%`
                                    : param.value.toFixed(2);
                                result += `${param.marker}${param.seriesName}: ${value}<br/>`;
                            }
                        }
                    });
                    
                    return result;
                }
            },
            legend: {
                top: 5,
                type: 'scroll'
            },
            xAxis: allXAxes,
            yAxis: allYAxes,
            grid: gridConfig,
            series: allSeries
        };
        
        // Clear and re-initialize chart for candlestick mode to ensure proper rendering
        if (chartType === 'candlestick') {
            this.chartInstance.clear();
        }
        
        this.chartInstance.setOption(option, true);
    }

    getMarketColors(market) {
        // Define color schemes for different market regions
        const marketColorSchemes = {
            'Japan': { bull: '#00da3c', bear: '#ec0000' },
            'Hong Kong': { bull: '#1890ff', bear: '#ff4d4f' },
            'China': { bull: '#52c41a', bear: '#f5222d' },
            'Australia': { bull: '#13c2c2', bear: '#eb2f96' },
            'South Korea': { bull: '#722ed1', bear: '#fa541c' },
            'India': { bull: '#faad14', bear: '#fa8c16' },
            'UK': { bull: '#2f54eb', bear: '#cf1322' },
            'Germany': { bull: '#389e0d', bear: '#d4380d' },
            'France': { bull: '#096dd9', bear: '#c41d7f' },
            'Netherlands': { bull: '#08979c', bear: '#ad2102' },
            'Spain': { bull: '#7cb305', bear: '#a8071a' },
            'US': { bull: '#00da3c', bear: '#ec0000' },
            'Canada': { bull: '#36cfc9', bear: '#ff7875' },
            'Brazil': { bull: '#95de64', bear: '#ffa39e' },
            'Global': { bull: '#73d13d', bear: '#ff9c6e' }
        };
        
        return marketColorSchemes[market] || { bull: '#00da3c', bear: '#ec0000' };
    }
    
    addAlpha(hexColor, alpha) {
        // Convert hex color to RGBA with alpha transparency
        const hex = hexColor.replace('#', '');
        const r = parseInt(hex.substring(0, 2), 16);
        const g = parseInt(hex.substring(2, 4), 16);
        const b = parseInt(hex.substring(4, 6), 16);
        return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    }

    getMarketSessionIndicators(chartType) {
        // Only add market session indicators for candlestick charts
        if (chartType !== 'candlestick') return [];

        // Market hours configuration (simplified for visualization)
        const marketSessions = [
            { name: 'Asian Markets', start: 0, end: 8, color: 'rgba(255, 193, 7, 0.1)' },    // 00:00-08:00 UTC
            { name: 'European Markets', start: 7, end: 16, color: 'rgba(40, 167, 69, 0.1)' }, // 07:00-16:00 UTC  
            { name: 'US Markets', start: 14, end: 22, color: 'rgba(0, 123, 255, 0.1)' }       // 14:00-22:00 UTC
        ];

        const markAreas = [];

        marketSessions.forEach(session => {
            markAreas.push([
                {
                    name: session.name,
                    xAxis: `${session.start.toString().padStart(2, '0')}:00`,
                    itemStyle: {
                        color: session.color
                    }
                },
                {
                    xAxis: `${session.end.toString().padStart(2, '0')}:00`
                }
            ]);
        });

        return markAreas;
    }

    getYAxisConfig(chartType, allValues) {
        // Base configuration for y-axis
        const baseConfig = {
            type: 'value',
            axisLabel: {
                formatter: (chartType === 'percentage' || chartType === 'candlestick') ? '{value}%' : '{value}'
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
            // For percentage charts, create a tight 2-3% range for better visualization
            if (range === 0) {
                // If all values are the same, create a small range around that value
                const center = minValue;
                return {
                    ...baseConfig,
                    min: center - 1.5, // 1.5% below center
                    max: center + 1.5  // 1.5% above center
                };
            } else {
                // Add minimal padding to the range (10% on each side)
                const padding = Math.max(range * 0.1, 0.2); // At least 0.2% padding
                const suggestedMin = minValue - padding;
                const suggestedMax = maxValue + padding;
                
                // Ensure the range is at least 2% but not more than 3% total
                const totalRange = suggestedMax - suggestedMin;
                if (totalRange < 2) {
                    // Expand to 2% range
                    const center = (suggestedMin + suggestedMax) / 2;
                    return {
                        ...baseConfig,
                        min: Math.round((center - 1) * 100) / 100,    // 1% below center
                        max: Math.round((center + 1) * 100) / 100     // 1% above center
                    };
                } else if (totalRange > 3) {
                    // Compress to 3% range centered on data
                    const center = (minValue + maxValue) / 2;
                    return {
                        ...baseConfig,
                        min: Math.round((center - 1.5) * 100) / 100,  // 1.5% below center
                        max: Math.round((center + 1.5) * 100) / 100   // 1.5% above center
                    };
                } else {
                    // Use calculated range (between 2-3%)
                    return {
                        ...baseConfig,
                        min: Math.round(suggestedMin * 100) / 100,
                        max: Math.round(suggestedMax * 100) / 100
                    };
                }
            }
        } else if (chartType === 'candlestick') {
            // For candlestick charts, use auto-scale with extra padding for better OHLC visibility
            return {
                ...baseConfig,
                scale: true, // Enable smart scaling
                boundaryGap: ['8%', '8%'] // Add extra padding for candlestick wicks
            };
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
        // Re-fetch data with the new chart type instead of just re-rendering
        if (this.selectedIndices.size > 0) {
            this.analyzeSelectedIndices();
        }
    }

    handleIntervalChange() {
        // Re-fetch data with the new time interval
        if (this.selectedIndices.size > 0) {
            this.analyzeSelectedIndices();
        }
    }

    handleTimePeriodChange() {
        // Re-fetch data with the new time period (24h vs 48h)
        if (this.selectedIndices.size > 0) {
            this.analyzeSelectedIndices();
        }
    }

    handlePlotModeChange() {
        // Re-render chart with the new plot mode without fetching data
        if (this.currentData && Object.keys(this.currentData.data).length > 0) {
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

    async loadMarketIntelligence(symbol) {
        const dropdown = document.getElementById(`intelligence-${symbol}`);
        const detailsContainer = document.getElementById(`intelligence-details-${symbol}`);
        
        if (!dropdown) return;
        
        try {
            // Get economic events data from the backend using existing endpoint
            const response = await fetch(`${this.apiBaseUrl}/economic-events?symbols=${encodeURIComponent(symbol)}&hours_back=48&hours_forward=24`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            
            const data = await response.json();
            
            // Clear loading state
            dropdown.innerHTML = '<option value="">Select intelligence type...</option>';
            
            // Add categories to dropdown based on API response structure
            if (data.events && data.events.length > 0) {
                // Separate events by type
                const marketSessions = data.events.filter(event => event.event_type === 'market_session');
                const economicData = data.events.filter(event => event.event_type === 'economic_data');
                const announcements = data.events.filter(event => event.event_type === 'market_announcement');
                
                // Market Sessions Group
                if (marketSessions.length > 0) {
                    const sessionGroup = document.createElement('optgroup');
                    sessionGroup.label = `Market Sessions (${marketSessions.length})`;
                    marketSessions.forEach((event, index) => {
                        const option = document.createElement('option');
                        option.value = `session-${index}`;
                        const importance = event.importance === 'high' ? '🔴' : 
                                        event.importance === 'medium' ? '🟡' : '🟢';
                        option.textContent = `${importance} ${event.title}`;
                        sessionGroup.appendChild(option);
                    });
                    dropdown.appendChild(sessionGroup);
                }
                
                // Economic Data Group  
                if (economicData.length > 0) {
                    const eventGroup = document.createElement('optgroup');
                    eventGroup.label = `Economic Events (${economicData.length})`;
                    economicData.forEach((event, index) => {
                        const option = document.createElement('option');
                        option.value = `event-${index}`;
                        const importance = event.importance === 'high' ? '🔴' : 
                                        event.importance === 'medium' ? '🟡' : '🟢';
                        option.textContent = `${importance} ${event.title}`;
                        eventGroup.appendChild(option);
                    });
                    dropdown.appendChild(eventGroup);
                }
                
                // Market Announcements Group
                if (announcements.length > 0) {
                    const announcementGroup = document.createElement('optgroup');
                    announcementGroup.label = `Market Announcements (${announcements.length})`;
                    announcements.forEach((announcement, index) => {
                        const option = document.createElement('option');
                        option.value = `announcement-${index}`;
                        option.textContent = `📢 ${announcement.title}`;
                        announcementGroup.appendChild(option);
                    });
                    dropdown.appendChild(announcementGroup);
                }
            }
            
            // Add economic calendar view option
            const calendarOption = document.createElement('option');
            calendarOption.value = 'calendar-view';
            calendarOption.textContent = '📅 Economic Calendar View';
            dropdown.appendChild(calendarOption);
            
            // Store the events data for detail display
            dropdown.dataset.intelligenceData = JSON.stringify(data);
            dropdown.dataset.marketSessions = JSON.stringify(data.events ? data.events.filter(event => event.event_type === 'market_session') : []);
            dropdown.dataset.economicData = JSON.stringify(data.events ? data.events.filter(event => event.event_type === 'economic_data') : []);  
            dropdown.dataset.announcements = JSON.stringify(data.events ? data.events.filter(event => event.event_type === 'market_announcement') : []);
            
            // Add event listener for dropdown selection
            dropdown.addEventListener('change', (e) => {
                this.displayIntelligenceDetails(symbol, e.target.value, data);
            });
            
        } catch (error) {
            console.error(`Failed to load intelligence for ${symbol}:`, error);
            dropdown.innerHTML = '<option value="">Failed to load intelligence</option>';
        }
    }

    displayIntelligenceDetails(symbol, selection, data) {
        const detailsContainer = document.getElementById(`intelligence-details-${symbol}`);
        if (!detailsContainer || !selection) {
            detailsContainer.classList.add('hidden');
            return;
        }

        let content = '';
        
        // Get event arrays from the data
        const allEvents = data.events || [];
        const marketSessions = allEvents.filter(event => event.event_type === 'market_session');
        const economicEvents = allEvents.filter(event => event.event_type === 'economic_data');
        const announcements = allEvents.filter(event => event.event_type === 'market_announcement');
        
        if (selection === 'calendar-view') {
            // Show economic calendar view
            content = `
                <div class="bg-blue-50 border border-blue-200 rounded p-2">
                    <h5 class="font-medium text-blue-800 mb-2">📅 Economic Calendar</h5>
                    <div class="space-y-2 text-blue-700">
                        <div class="flex justify-between">
                            <span>Market Sessions:</span>
                            <span class="font-medium">${marketSessions.length}</span>
                        </div>
                        <div class="flex justify-between">
                            <span>Economic Events:</span>
                            <span class="font-medium">${economicEvents.length}</span>
                        </div>
                        <div class="flex justify-between">
                            <span>Market Announcements:</span>
                            <span class="font-medium">${announcements.length}</span>
                        </div>
                        <div class="flex justify-between">
                            <span>Total Events:</span>
                            <span class="font-medium">${allEvents.length}</span>
                        </div>
                        <div class="flex justify-between">
                            <span>Next Update:</span>
                            <span class="font-medium">${new Date(Date.now() + 30*60000).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</span>
                        </div>
                    </div>
                </div>
            `;
        } else if (selection.startsWith('session-')) {
            // Show market session details
            const sessionIndex = parseInt(selection.split('-')[1]);
            const session = marketSessions[sessionIndex];
            if (session) {
                const importanceColor = session.importance === 'high' ? 'red' : 
                                      session.importance === 'medium' ? 'blue' : 'green';
                
                content = `
                    <div class="bg-${importanceColor}-50 border border-${importanceColor}-200 rounded p-2">
                        <h5 class="font-medium text-${importanceColor}-800 mb-2">📅 ${session.title}</h5>
                        <div class="space-y-1 text-${importanceColor}-700">
                            <div><strong>Type:</strong> ${session.event_type}</div>
                            <div><strong>Country:</strong> ${session.country}</div>
                            <div><strong>Currency:</strong> ${session.currency}</div>
                            <div><strong>Importance:</strong> ${session.importance.toUpperCase()}</div>
                            <div><strong>Time:</strong> ${new Date(session.timestamp).toLocaleString()}</div>
                            ${session.description ? `<div class="mt-2"><strong>Description:</strong><br/><span class="text-xs">${session.description}</span></div>` : ''}
                        </div>
                    </div>
                `;
            }
        } else if (selection.startsWith('event-')) {
            // Show economic event details
            const eventIndex = parseInt(selection.split('-')[1]);
            const event = economicEvents[eventIndex];
            if (event) {
                const importanceColor = event.importance === 'high' ? 'red' : 
                                      event.importance === 'medium' ? 'yellow' : 'green';
                
                content = `
                    <div class="bg-${importanceColor}-50 border border-${importanceColor}-200 rounded p-2">
                        <h5 class="font-medium text-${importanceColor}-800 mb-2">📊 ${event.title}</h5>
                        <div class="space-y-1 text-${importanceColor}-700">
                            <div><strong>Type:</strong> ${event.event_type}</div>
                            <div><strong>Country:</strong> ${event.country}</div>
                            <div><strong>Currency:</strong> ${event.currency}</div>
                            <div><strong>Importance:</strong> ${event.importance.toUpperCase()}</div>
                            ${event.actual_value ? `<div><strong>Actual:</strong> ${event.actual_value}</div>` : ''}
                            ${event.forecast_value ? `<div><strong>Forecast:</strong> ${event.forecast_value}</div>` : ''}
                            ${event.previous_value ? `<div><strong>Previous:</strong> ${event.previous_value}</div>` : ''}
                            <div><strong>Time:</strong> ${new Date(event.timestamp).toLocaleString()}</div>
                            ${event.description ? `<div class="mt-2"><strong>Description:</strong><br/><span class="text-xs">${event.description}</span></div>` : ''}
                        </div>
                    </div>
                `;
            }
        } else if (selection.startsWith('announcement-')) {
            // Show market announcement details
            const announcementIndex = parseInt(selection.split('-')[1]);
            const announcement = announcements[announcementIndex];
            if (announcement) {
                content = `
                    <div class="bg-purple-50 border border-purple-200 rounded p-2">
                        <h5 class="font-medium text-purple-800 mb-2">📢 ${announcement.title}</h5>
                        <div class="space-y-1 text-purple-700">
                            <div><strong>Type:</strong> ${announcement.event_type}</div>
                            <div><strong>Country:</strong> ${announcement.country}</div>
                            <div><strong>Currency:</strong> ${announcement.currency}</div>
                            <div><strong>Importance:</strong> ${announcement.importance.toUpperCase()}</div>
                            <div><strong>Time:</strong> ${new Date(announcement.timestamp).toLocaleString()}</div>
                            ${announcement.impact_markets?.length ? `<div><strong>Affects:</strong> ${announcement.impact_markets.join(', ')}</div>` : ''}
                            ${announcement.description ? `<div class="mt-2"><strong>Details:</strong><br/><span class="text-xs">${announcement.description}</span></div>` : ''}
                        </div>
                    </div>
                `;
            }
        }
        
        detailsContainer.innerHTML = content;
        detailsContainer.classList.remove('hidden');
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

    // ===== CALENDAR FUNCTIONALITY =====

    toggleCalendar() {
        if (this.calendarVisible) {
            this.hideCalendar();
        } else {
            this.showCalendar();
        }
    }

    showCalendar() {
        this.calendarVisible = true;
        const calendarElement = document.getElementById('calendar-dropdown');
        if (calendarElement) {
            calendarElement.classList.remove('hidden');
            this.renderCalendar();
        }
    }

    hideCalendar() {
        this.calendarVisible = false;
        document.getElementById('calendar-dropdown').classList.add('hidden');
    }

    renderCalendar() {
        const monthNames = [
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ];

        // Update month/year header
        const monthYearElement = document.getElementById('calendar-month-year');
        if (monthYearElement) {
            monthYearElement.textContent = `${monthNames[this.currentMonth]} ${this.currentYear}`;
        }

        // Generate calendar days
        const firstDay = new Date(this.currentYear, this.currentMonth, 1);
        const lastDay = new Date(this.currentYear, this.currentMonth + 1, 0);
        const startDate = new Date(firstDay);
        startDate.setDate(startDate.getDate() - firstDay.getDay()); // Start from Sunday

        const calendarDays = document.getElementById('calendar-days');
        calendarDays.innerHTML = '';

        // Generate 6 weeks of days
        for (let week = 0; week < 6; week++) {
            for (let day = 0; day < 7; day++) {
                const currentDate = new Date(startDate);
                currentDate.setDate(startDate.getDate() + (week * 7) + day);

                const dayElement = document.createElement('button');
                dayElement.className = 'p-2 text-sm rounded hover:bg-gray-100 transition-colors';
                dayElement.textContent = currentDate.getDate();

                // Style based on month and selection
                const isCurrentMonth = currentDate.getMonth() === this.currentMonth;
                const isSelected = this.isSameDate(currentDate, this.selectedDate);
                const isToday = this.isSameDate(currentDate, new Date());
                const isFuture = currentDate > new Date();

                if (!isCurrentMonth) {
                    dayElement.className += ' text-gray-400';
                } else if (isFuture) {
                    dayElement.className += ' text-gray-400 cursor-not-allowed';
                    dayElement.disabled = true;
                } else if (isSelected) {
                    dayElement.className += ' bg-primary-500 text-white hover:bg-primary-600';
                } else if (isToday) {
                    dayElement.className += ' bg-blue-100 text-blue-800 font-semibold';
                } else {
                    dayElement.className += ' text-gray-700';
                }

                if (!isFuture && isCurrentMonth) {
                    dayElement.addEventListener('click', () => {
                        this.selectDate(currentDate);
                    });
                }

                calendarDays.appendChild(dayElement);
            }
        }
    }

    selectDate(date) {
        this.selectedDate = new Date(date);
        this.isHistoricalMode = !this.isSameDate(date, new Date());
        this.updateSelectedDateDisplay();
        this.hideCalendar();
        
        // Update chart data based on the selected date
        // analyzeSelectedIndices() will now handle both historical and live data
        if (this.selectedIndices.size > 0) {
            this.analyzeSelectedIndices();
        }
        
        this.updatePerformanceSummary();
    }

    updateSelectedDateDisplay() {
        const dateBtn = document.getElementById('selected-date');
        const today = new Date();
        
        if (this.isSameDate(this.selectedDate, today)) {
            dateBtn.textContent = 'Today';
        } else {
            const options = { 
                month: 'short', 
                day: 'numeric',
                year: this.selectedDate.getFullYear() !== today.getFullYear() ? 'numeric' : undefined
            };
            dateBtn.textContent = this.selectedDate.toLocaleDateString('en-US', options);
        }
    }

    goToPreviousDay() {
        const newDate = new Date(this.selectedDate);
        newDate.setDate(newDate.getDate() - 1);
        
        // Don't go beyond a reasonable historical limit (e.g., 30 days)
        const minDate = new Date();
        minDate.setDate(minDate.getDate() - 30);
        
        if (newDate >= minDate) {
            this.selectDate(newDate);
        }
    }

    goToNextDay() {
        const newDate = new Date(this.selectedDate);
        newDate.setDate(newDate.getDate() + 1);
        
        // Don't go beyond today
        if (newDate <= new Date()) {
            this.selectDate(newDate);
        }
    }

    goToToday() {
        this.selectDate(new Date());
    }

    goToPreviousMonth() {
        if (this.currentMonth === 0) {
            this.currentMonth = 11;
            this.currentYear--;
        } else {
            this.currentMonth--;
        }
        this.renderCalendar();
    }

    goToNextMonth() {
        if (this.currentMonth === 11) {
            this.currentMonth = 0;
            this.currentYear++;
        } else {
            this.currentMonth++;
        }
        this.renderCalendar();
    }

    isSameDate(date1, date2) {
        return date1.getFullYear() === date2.getFullYear() &&
               date1.getMonth() === date2.getMonth() &&
               date1.getDate() === date2.getDate();
    }

    async loadHistoricalData() {
        if (this.selectedIndices.size === 0) return;
        
        this.showLoading(true);
        
        try {
            const chartType = document.getElementById('chart-type').value;
            const intervalMinutes = parseInt(document.getElementById('time-interval').value);
            const timePeriod = document.getElementById('time-period').value;
            const dateStr = this.selectedDate.toISOString().split('T')[0]; // YYYY-MM-DD
            
            const response = await fetch(`${this.apiBaseUrl}/analyze/historical?target_date=${dateStr}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    symbols: Array.from(this.selectedIndices),
                    chart_type: chartType,
                    interval_minutes: intervalMinutes,
                    time_period: timePeriod
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.success) {
                this.currentData = data;
                this.updateChart(data);
                this.updatePerformanceSummaryDisplay(data.performance_summary);
                this.showToast(`Historical data loaded for ${dateStr}`, 'success');
            } else {
                throw new Error('Failed to load historical data');
            }
            
        } catch (error) {
            console.error('❌ Error loading historical data:', error);
            this.showToast('Failed to load historical data. Please try again.', 'error');
        } finally {
            this.showLoading(false);
        }
    }

    updatePerformanceSummary() {
        const summaryContainer = document.getElementById('date-performance-summary');
        const titleElement = document.getElementById('performance-date-title');
        const weekdayElement = document.getElementById('weekday-indicator');
        
        if (this.isHistoricalMode) {
            summaryContainer.classList.remove('hidden');
            
            // Update date title
            const options = { 
                weekday: 'long',
                year: 'numeric', 
                month: 'long', 
                day: 'numeric' 
            };
            titleElement.textContent = this.selectedDate.toLocaleDateString('en-US', options);
            
            // Update weekday indicator
            const weekday = this.selectedDate.getDay();
            const isWeekend = weekday === 0 || weekday === 6;
            weekdayElement.className = `px-2 py-1 rounded text-xs font-medium ${
                isWeekend ? 'bg-orange-100 text-orange-800' : 'bg-blue-100 text-blue-800'
            }`;
            weekdayElement.textContent = isWeekend ? 'Weekend' : 'Weekday';
            
        } else {
            summaryContainer.classList.add('hidden');
        }
    }

    updatePerformanceSummaryDisplay(performanceSummary) {
        if (!performanceSummary || !this.isHistoricalMode) return;
        
        const statsContainer = document.getElementById('performance-summary-stats');
        const detailsContainer = document.getElementById('performance-details');
        
        const summary = performanceSummary.market_summary;
        
        // Update stats
        statsContainer.innerHTML = `
            <div class="text-center">
                <div class="text-lg font-semibold text-gray-800">${summary.symbols_with_data}</div>
                <div class="text-xs text-gray-500">Symbols</div>
            </div>
            <div class="text-center">
                <div class="text-lg font-semibold text-green-600">${summary.gainers}</div>
                <div class="text-xs text-gray-500">Gainers</div>
            </div>
            <div class="text-center">
                <div class="text-lg font-semibold text-red-600">${summary.losers}</div>
                <div class="text-xs text-gray-500">Losers</div>
            </div>
            <div class="text-center">
                <div class="text-lg font-semibold ${performanceSummary.average_change >= 0 ? 'text-green-600' : 'text-red-600'}">
                    ${performanceSummary.average_change >= 0 ? '+' : ''}${performanceSummary.average_change.toFixed(2)}%
                </div>
                <div class="text-xs text-gray-500">Avg Change</div>
            </div>
        `;
        
        // Update details
        let detailsText = '';
        if (performanceSummary.best_performer) {
            detailsText += `Best: ${performanceSummary.best_performer.name} (+${performanceSummary.best_performer.daily_change.toFixed(2)}%)`;
        }
        if (performanceSummary.worst_performer) {
            if (detailsText) detailsText += ' • ';
            detailsText += `Worst: ${performanceSummary.worst_performer.name} (${performanceSummary.worst_performer.daily_change.toFixed(2)}%)`;
        }
        
        detailsContainer.textContent = detailsText;
    }
}

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    window.tracker = new GlobalMarketTracker();
});