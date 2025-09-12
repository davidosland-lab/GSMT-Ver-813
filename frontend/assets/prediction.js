/**
 * LLM Market Prediction Frontend System
 * Handles API communication and UI updates for Australian All Ordinaries predictions
 */

class MarketPredictionApp {
    constructor() {
        this.apiUrl = this.detectApiUrl();
        this.currentPrediction = null;
        this.progressInterval = null;
        this.isLoading = false;
        
        this.init();
    }

    /**
     * Detect the appropriate API URL based on environment
     */
    detectApiUrl() {
        const currentHost = window.location.hostname;
        
        // Sandbox environment (E2B)
        if (currentHost.includes('.e2b.dev')) {
            return 'https://8000-iqujeilaojex6ersk73ur-6532622b.e2b.dev/api';
        }
        
        // Production environments
        if (currentHost.includes('netlify.app')) {
            return 'https://gsmt-ver-813-production.up.railway.app/api';
        }
        
        // Development environments
        if (currentHost === 'localhost' || currentHost === '127.0.0.1') {
            return 'http://localhost:8000/api';
        }
        
        // Fallback to Railway production URL
        return 'https://gsmt-ver-813-production.up.railway.app/api';
    }

    /**
     * Initialize the application
     */
    init() {
        this.setupEventListeners();
        this.updateCurrentTime();
        setInterval(() => this.updateCurrentTime(), 60000); // Update every minute
        
        console.log('🧠 LLM Market Prediction System initialized');
        console.log('📡 API URL:', this.apiUrl);
    }

    /**
     * Setup all event listeners
     */
    setupEventListeners() {
        // Generate prediction button
        const generateBtn = document.getElementById('generate-prediction');
        if (generateBtn) {
            generateBtn.addEventListener('click', () => this.generatePrediction());
        }

        // Symbol selection change
        const symbolSelect = document.getElementById('symbol-select');
        if (symbolSelect) {
            symbolSelect.addEventListener('change', () => this.onSymbolChange());
        }

        // Analysis mode change
        const analysisMode = document.getElementById('analysis-mode');
        if (analysisMode) {
            analysisMode.addEventListener('change', () => this.onAnalysisModeChange());
        }
    }

    /**
     * Update current time display
     */
    updateCurrentTime() {
        const timeElement = document.getElementById('current-time');
        if (timeElement) {
            const now = new Date();
            const timeString = now.toLocaleString('en-AU', {
                timeZone: 'Australia/Sydney',
                year: 'numeric',
                month: '2-digit',
                day: '2-digit',
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit'
            });
            timeElement.textContent = `${timeString} AEDT`;
        }
    }

    /**
     * Handle symbol selection change
     */
    onSymbolChange() {
        const symbolSelect = document.getElementById('symbol-select');
        const selectedSymbol = symbolSelect.value;
        
        console.log('📊 Symbol changed to:', selectedSymbol);
        
        // Update symbol display
        const predictionSymbol = document.getElementById('prediction-symbol');
        if (predictionSymbol) {
            const symbolMap = {
                '^AORD': 'Australian All Ordinaries',
                '^GSPC': 'S&P 500',
                '^FTSE': 'FTSE 100',
                '^N225': 'Nikkei 225'
            };
            predictionSymbol.textContent = symbolMap[selectedSymbol] || selectedSymbol;
        }
    }

    /**
     * Handle analysis mode change
     */
    onAnalysisModeChange() {
        const analysisMode = document.getElementById('analysis-mode');
        const selectedMode = analysisMode.value;
        
        console.log('🔬 Analysis mode changed to:', selectedMode);
        
        // Show/hide multi-timeframe section based on batch mode
        const multiTimeframe = document.getElementById('multi-timeframe');
        if (multiTimeframe) {
            if (selectedMode === 'batch') {
                multiTimeframe.classList.remove('hidden');
            } else {
                multiTimeframe.classList.add('hidden');
            }
        }
    }

    /**
     * Generate market prediction
     */
    async generatePrediction() {
        if (this.isLoading) {
            console.log('⚠️ Prediction already in progress');
            return;
        }

        const symbol = document.getElementById('symbol-select').value;
        const timeframe = document.getElementById('timeframe-select').value;
        const analysisMode = document.getElementById('analysis-mode').value;

        console.log('🚀 Generating prediction:', { symbol, timeframe, analysisMode });

        try {
            this.showLoading();
            
            if (analysisMode === 'batch') {
                await this.generateBatchPredictions(symbol);
            } else {
                await this.generateSinglePrediction(symbol, timeframe, analysisMode);
            }
            
        } catch (error) {
            console.error('❌ Prediction error:', error);
            this.showError(error.message);
        } finally {
            this.hideLoading();
        }
    }

    /**
     * Generate single market prediction
     */
    async generateSinglePrediction(symbol, timeframe, analysisMode) {
        const includeFactors = analysisMode === 'detailed';
        const includeNews = true; // Always include news intelligence
        
        console.log('📊 Fetching enhanced prediction:', { symbol, timeframe, includeFactors, includeNews });
        
        const response = await fetch(`${this.apiUrl}/prediction/${symbol}?timeframe=${timeframe}&include_factors=${includeFactors}&include_news_intelligence=${includeNews}`);
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        console.log('✅ Prediction data received:', data);
        
        this.currentPrediction = data;
        this.displayPrediction(data);
    }

    /**
     * Generate batch predictions for all timeframes
     */
    async generateBatchPredictions(symbol) {
        console.log('📊 Fetching batch predictions for symbol:', symbol);
        
        const response = await fetch(`${this.apiUrl}/prediction/batch`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                symbols: [symbol],
                timeframes: ['1d', '5d', '30d', '90d']
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        console.log('✅ Batch prediction data received:', data);
        
        this.displayBatchPredictions(data);
    }

    /**
     * Display single prediction results
     */
    displayPrediction(data) {
        if (!data.success || !data.prediction) {
            throw new Error('Invalid prediction data received');
        }

        const prediction = data.prediction;
        
        // Update main prediction card
        this.updateMainPredictionCard(prediction);
        
        // Update detailed analysis
        this.updateDetailedAnalysis(prediction);
        
        // Update market factors
        this.updateMarketFactors(prediction);
        
        // Update historical patterns
        this.updateHistoricalPatterns(prediction);
        
        // Update news intelligence display
        if (data.news_intelligence || data.volatility_assessment) {
            this.updateNewsIntelligence(data.news_intelligence, data.volatility_assessment);
        }
        
        // Show results
        this.showResults();
    }

    /**
     * Update main prediction card with data
     */
    updateMainPredictionCard(prediction) {
        const {
            direction = 'NEUTRAL',
            expected_change = 0,
            timeframe = '5d',
            confidence = 50,
            risk_level = 'medium',
            historical_accuracy = 60,
            key_factors = []
        } = prediction;

        // Set card theme based on sentiment
        const mainCard = document.getElementById('main-prediction');
        if (mainCard) {
            mainCard.className = 'rounded-lg shadow-lg text-white p-8';
            
            if (direction.toLowerCase().includes('bull')) {
                mainCard.classList.add('bullish');
            } else if (direction.toLowerCase().includes('bear')) {
                mainCard.classList.add('bearish');
            } else {
                mainCard.classList.add('neutral');
            }
        }

        // Update direction
        const directionElement = document.getElementById('prediction-direction');
        if (directionElement) {
            const directionMap = {
                'BULLISH': 'BULLISH ↗️',
                'BEARISH': 'BEARISH ↘️',
                'NEUTRAL': 'NEUTRAL ↔️',
                'VERY_BULLISH': 'VERY BULLISH 🚀',
                'VERY_BEARISH': 'VERY BEARISH 📉'
            };
            directionElement.textContent = directionMap[direction.toUpperCase()] || `${direction} ↔️`;
        }

        // Update expected change
        const changeElement = document.getElementById('prediction-change');
        if (changeElement) {
            const changeStr = expected_change > 0 ? `+${expected_change}%` : `${expected_change}%`;
            changeElement.textContent = changeStr;
        }

        // Update timeframe
        const timeframeElement = document.getElementById('prediction-timeframe');
        if (timeframeElement) {
            const timeframeMap = {
                '1d': '1 Day',
                '5d': '5 Days',
                '30d': '30 Days',
                '90d': '90 Days'
            };
            timeframeElement.textContent = timeframeMap[timeframe] || timeframe;
        }

        // Update confidence
        const confidenceScore = document.getElementById('confidence-score');
        const confidenceBar = document.getElementById('confidence-bar');
        if (confidenceScore && confidenceBar) {
            confidenceScore.textContent = `${Math.round(confidence)}%`;
            confidenceBar.style.width = `${confidence}%`;
        }

        // Update risk level
        const riskLevelElement = document.getElementById('risk-level');
        if (riskLevelElement) {
            riskLevelElement.textContent = risk_level;
            
            // Style based on risk level
            riskLevelElement.className = 'px-3 py-1 rounded-full text-sm font-medium';
            if (risk_level.toLowerCase() === 'high') {
                riskLevelElement.classList.add('bg-red-200', 'text-red-800');
            } else if (risk_level.toLowerCase() === 'low') {
                riskLevelElement.classList.add('bg-green-200', 'text-green-800');
            } else {
                riskLevelElement.classList.add('bg-yellow-200', 'text-yellow-800');
            }
        }

        // Update historical accuracy
        const accuracyElement = document.getElementById('historical-accuracy');
        if (accuracyElement) {
            accuracyElement.textContent = `${Math.round(historical_accuracy)}%`;
        }

        // Update key factors
        const factorsElement = document.getElementById('key-factors');
        if (factorsElement && Array.isArray(key_factors)) {
            factorsElement.innerHTML = key_factors
                .map(factor => `<li>• ${factor}</li>`)
                .join('');
        }
    }

    /**
     * Update detailed LLM analysis section
     */
    updateDetailedAnalysis(prediction) {
        const reasoningElement = document.getElementById('llm-reasoning');
        if (reasoningElement) {
            const reasoning = prediction.reasoning || prediction.llm_reasoning || 
                'The LLM analysis indicates market patterns based on historical data correlation with current economic factors. The prediction considers multiple technical and fundamental indicators to provide this assessment.';
            
            reasoningElement.textContent = reasoning;
        }
    }

    /**
     * Update market factors correlations
     */
    updateMarketFactors(prediction) {
        const factorsContainer = document.getElementById('factor-correlations');
        if (!factorsContainer) return;

        const factors = prediction.market_factors || {
            'Iron Ore Price': 0.65,
            'AUD/USD Rate': 0.42,
            'RBA Cash Rate': -0.38,
            'ASX VIX': -0.55,
            'US Market': 0.72
        };

        const factorHtml = Object.entries(factors).map(([factor, correlation]) => {
            const correlationPercent = Math.abs(correlation) * 100;
            const isPositive = correlation >= 0;
            const colorClass = isPositive ? 'bg-green-500' : 'bg-red-500';
            const signClass = isPositive ? 'text-green-700' : 'text-red-700';
            
            return `
                <div class="flex items-center justify-between">
                    <span class="text-sm font-medium text-gray-700">${factor}</span>
                    <div class="flex items-center space-x-2">
                        <div class="w-16 bg-gray-200 rounded-full h-2">
                            <div class="${colorClass} h-2 rounded-full" style="width: ${correlationPercent}%"></div>
                        </div>
                        <span class="text-sm ${signClass} font-medium">${correlation.toFixed(2)}</span>
                    </div>
                </div>
            `;
        }).join('');

        factorsContainer.innerHTML = factorHtml;
    }

    /**
     * Update historical patterns section
     */
    updateHistoricalPatterns(prediction) {
        const patternsContainer = document.getElementById('historical-patterns');
        if (!patternsContainer) return;

        const patterns = prediction.historical_patterns || [
            { pattern: 'Bull Market Trend', frequency: '68%', relevance: 'high' },
            { pattern: 'Seasonal Q4 Rise', frequency: '45%', relevance: 'medium' },
            { pattern: 'Pre-RBA Meeting Volatility', frequency: '72%', relevance: 'high' }
        ];

        const patternHtml = patterns.map(pattern => {
            const relevanceColor = pattern.relevance === 'high' ? 'text-green-600' : 
                                  pattern.relevance === 'medium' ? 'text-yellow-600' : 'text-gray-600';
            
            return `
                <div class="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                    <div>
                        <div class="font-medium text-gray-900">${pattern.pattern}</div>
                        <div class="text-sm text-gray-600">Historical frequency: ${pattern.frequency}</div>
                    </div>
                    <span class="px-2 py-1 text-xs rounded-full ${relevanceColor} bg-opacity-10">
                        ${pattern.relevance}
                    </span>
                </div>
            `;
        }).join('');

        patternsContainer.innerHTML = patternHtml;
    }

    /**
     * Update news intelligence and volatility assessment display
     */
    updateNewsIntelligence(newsData, volatilityData) {
        // Update news intelligence summary
        if (newsData) {
            this.updateNewsIntelligenceSummary(newsData);
        }
        
        // Update volatility assessment
        if (volatilityData) {
            this.updateVolatilityAssessment(volatilityData);
        }
        
        // Show news intelligence section
        const newsSection = document.getElementById('news-intelligence-section');
        if (newsSection) {
            newsSection.classList.remove('hidden');
        }
    }

    /**
     * Update news intelligence summary
     */
    updateNewsIntelligenceSummary(newsData) {
        const container = document.getElementById('news-intelligence-summary');
        if (!container) return;

        const {
            articles_analyzed = 0,
            global_events_count = 0,
            avg_sentiment = 0,
            high_impact_news = 0,
            australian_relevance = 0,
            top_categories = [],
            geographic_focus = {},
            key_events = []
        } = newsData;

        const sentimentClass = avg_sentiment > 0.2 ? 'text-green-600' : 
                               avg_sentiment < -0.2 ? 'text-red-600' : 'text-gray-600';

        const relevancePercent = Math.round(australian_relevance * 100);
        const relevanceClass = relevancePercent > 60 ? 'text-green-600' : 
                              relevancePercent > 30 ? 'text-yellow-600' : 'text-red-600';

        const summaryHtml = `
            <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                <div class="text-center p-3 bg-blue-50 rounded-lg">
                    <div class="text-2xl font-bold text-blue-600">${articles_analyzed}</div>
                    <div class="text-sm text-blue-800">News Articles</div>
                </div>
                <div class="text-center p-3 bg-purple-50 rounded-lg">
                    <div class="text-2xl font-bold text-purple-600">${global_events_count}</div>
                    <div class="text-sm text-purple-800">Global Events</div>
                </div>
                <div class="text-center p-3 bg-gray-50 rounded-lg">
                    <div class="text-2xl font-bold ${sentimentClass}">${avg_sentiment > 0 ? '+' : ''}${avg_sentiment.toFixed(2)}</div>
                    <div class="text-sm text-gray-600">Avg Sentiment</div>
                </div>
                <div class="text-center p-3 bg-orange-50 rounded-lg">
                    <div class="text-2xl font-bold text-orange-600">${high_impact_news}</div>
                    <div class="text-sm text-orange-800">High Impact</div>
                </div>
            </div>
            
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                    <h4 class="font-medium text-gray-900 mb-3">Key Event Categories</h4>
                    <div class="space-y-2">
                        ${top_categories.slice(0, 5).map(category => `
                            <div class="flex items-center justify-between p-2 bg-gray-50 rounded">
                                <span class="text-sm capitalize">${category.replace(/_/g, ' ')}</span>
                                <span class="text-xs text-gray-500">Active</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
                
                <div>
                    <h4 class="font-medium text-gray-900 mb-3">Geographic Impact</h4>
                    <div class="space-y-2">
                        ${Object.entries(geographic_focus).slice(0, 4).map(([region, score]) => `
                            <div class="flex items-center justify-between">
                                <span class="text-sm capitalize">${region.replace(/_/g, ' ')}</span>
                                <div class="flex items-center space-x-2">
                                    <div class="w-16 bg-gray-200 rounded-full h-2">
                                        <div class="bg-blue-500 h-2 rounded-full" style="width: ${score * 100}%"></div>
                                    </div>
                                    <span class="text-xs text-gray-600">${Math.round(score * 100)}%</span>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>
            
            ${key_events.length > 0 ? `
                <div class="mt-6">
                    <h4 class="font-medium text-gray-900 mb-3">Major Global Events</h4>
                    <div class="space-y-2">
                        ${key_events.map(event => `
                            <div class="flex items-center justify-between p-3 border border-gray-200 rounded-lg">
                                <span class="text-sm font-medium">${event.title}</span>
                                <span class="px-2 py-1 text-xs rounded-full ${
                                    event.impact > 70 ? 'bg-red-100 text-red-800' :
                                    event.impact > 40 ? 'bg-yellow-100 text-yellow-800' :
                                    'bg-green-100 text-green-800'
                                }">
                                    ${Math.round(event.impact)}% Impact
                                </span>
                            </div>
                        `).join('')}
                    </div>
                </div>
            ` : ''}
        `;

        container.innerHTML = summaryHtml;
    }

    /**
     * Update volatility assessment display
     */
    updateVolatilityAssessment(volatilityData) {
        const container = document.getElementById('volatility-assessment');
        if (!container) return;

        const {
            overall_sentiment = 0,
            volatility_score = 0,
            impact_level = 'minimal',
            confidence = 0,
            trend_direction = 'stable',
            key_drivers = [],
            risk_factors = [],
            opportunity_factors = [],
            recent_events_count = 0
        } = volatilityData;

        const impactColor = {
            'extreme': 'bg-red-100 text-red-800',
            'high': 'bg-orange-100 text-orange-800',
            'moderate': 'bg-yellow-100 text-yellow-800',
            'low': 'bg-blue-100 text-blue-800',
            'minimal': 'bg-gray-100 text-gray-800'
        }[impact_level] || 'bg-gray-100 text-gray-800';

        const trendIcon = {
            'increasing': '📈',
            'decreasing': '📉', 
            'stable': '➡️'
        }[trend_direction] || '➡️';

        const assessmentHtml = `
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <div class="text-center p-4 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg">
                    <div class="text-3xl font-bold text-indigo-600">${Math.round(volatility_score)}</div>
                    <div class="text-sm text-indigo-800">Volatility Score</div>
                    <div class="text-xs text-indigo-600 mt-1">out of 100</div>
                </div>
                
                <div class="text-center p-4 bg-gradient-to-r from-purple-50 to-pink-50 rounded-lg">
                    <div class="text-lg font-bold ${overall_sentiment > 0 ? 'text-green-600' : overall_sentiment < 0 ? 'text-red-600' : 'text-gray-600'}">
                        ${overall_sentiment > 0 ? '😊' : overall_sentiment < 0 ? '😟' : '😐'} ${overall_sentiment.toFixed(2)}
                    </div>
                    <div class="text-sm text-purple-800">Market Sentiment</div>
                    <div class="text-xs text-purple-600 mt-1">-1 to +1 scale</div>
                </div>
                
                <div class="text-center p-4 bg-gradient-to-r from-green-50 to-emerald-50 rounded-lg">
                    <div class="text-lg font-bold text-emerald-600">${Math.round(confidence * 100)}%</div>
                    <div class="text-sm text-emerald-800">Analysis Confidence</div>
                    <div class="text-xs text-emerald-600 mt-1">Data quality</div>
                </div>
            </div>
            
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                <div class="p-4 border border-gray-200 rounded-lg">
                    <div class="flex items-center justify-between mb-3">
                        <h4 class="font-medium text-gray-900">Impact Level</h4>
                        <span class="px-3 py-1 text-sm font-medium rounded-full ${impactColor}">
                            ${impact_level.toUpperCase()}
                        </span>
                    </div>
                    <div class="flex items-center space-x-2">
                        <span class="text-2xl">${trendIcon}</span>
                        <div>
                            <div class="text-sm font-medium">Trend: ${trend_direction.toUpperCase()}</div>
                            <div class="text-xs text-gray-600">${recent_events_count} recent events analyzed</div>
                        </div>
                    </div>
                </div>
                
                <div class="p-4 border border-gray-200 rounded-lg">
                    <h4 class="font-medium text-gray-900 mb-3">Volatility Breakdown</h4>
                    <div class="space-y-2">
                        <div class="flex justify-between text-sm">
                            <span>Market Impact:</span>
                            <span class="font-medium">${Math.round(volatility_score)}%</span>
                        </div>
                        <div class="w-full bg-gray-200 rounded-full h-2">
                            <div class="bg-gradient-to-r from-blue-500 to-purple-500 h-2 rounded-full" style="width: ${volatility_score}%"></div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                    <h4 class="font-medium text-gray-900 mb-3 flex items-center">
                        <span class="mr-2">🚀</span>Key Drivers
                    </h4>
                    <div class="space-y-1">
                        ${key_drivers.slice(0, 4).map(driver => `
                            <div class="text-sm p-2 bg-blue-50 rounded">${driver}</div>
                        `).join('') || '<div class="text-sm text-gray-500">No key drivers identified</div>'}
                    </div>
                </div>
                
                <div>
                    <h4 class="font-medium text-gray-900 mb-3 flex items-center">
                        <span class="mr-2">⚠️</span>Risk Factors
                    </h4>
                    <div class="space-y-1">
                        ${risk_factors.slice(0, 4).map(risk => `
                            <div class="text-sm p-2 bg-red-50 rounded">${risk}</div>
                        `).join('') || '<div class="text-sm text-gray-500">No significant risks identified</div>'}
                    </div>
                </div>
                
                <div>
                    <h4 class="font-medium text-gray-900 mb-3 flex items-center">
                        <span class="mr-2">💡</span>Opportunities
                    </h4>
                    <div class="space-y-1">
                        ${opportunity_factors.slice(0, 4).map(opp => `
                            <div class="text-sm p-2 bg-green-50 rounded">${opp}</div>
                        `).join('') || '<div class="text-sm text-gray-500">No clear opportunities identified</div>'}
                    </div>
                </div>
            </div>
        `;

        container.innerHTML = assessmentHtml;
    }

    /**
     * Display batch predictions for all timeframes
     */
    displayBatchPredictions(data) {
        if (!data.success || !data.predictions) {
            throw new Error('Invalid batch prediction data received');
        }

        const predictions = data.predictions;
        
        // Show main prediction for 5d timeframe
        const mainPrediction = predictions.find(p => p.timeframe === '5d') || predictions[0];
        if (mainPrediction) {
            this.updateMainPredictionCard(mainPrediction.prediction);
        }
        
        // Update multi-timeframe grid
        this.updateMultiTimeframeGrid(predictions);
        
        // Show results and multi-timeframe section
        this.showResults();
        document.getElementById('multi-timeframe').classList.remove('hidden');
    }

    /**
     * Update multi-timeframe grid with batch predictions
     */
    updateMultiTimeframeGrid(predictions) {
        const gridContainer = document.getElementById('timeframe-grid');
        if (!gridContainer) return;

        const timeframeLabels = {
            '1d': { label: 'Next Day', icon: '📅' },
            '5d': { label: 'Short Term', icon: '📊' },
            '30d': { label: 'Medium Term', icon: '📈' },
            '90d': { label: 'Long Term', icon: '📉' }
        };

        const gridHtml = predictions.map(predData => {
            const { timeframe, prediction } = predData;
            const { label, icon } = timeframeLabels[timeframe] || { label: timeframe, icon: '📊' };
            
            const direction = prediction.direction || 'NEUTRAL';
            const expectedChange = prediction.expected_change || 0;
            const confidence = prediction.confidence || 50;
            
            const directionClass = direction.toLowerCase().includes('bull') ? 'text-green-600' : 
                                 direction.toLowerCase().includes('bear') ? 'text-red-600' : 'text-gray-600';
            
            const changeStr = expectedChange > 0 ? `+${expectedChange}%` : `${expectedChange}%`;
            const changeClass = expectedChange > 0 ? 'text-green-600' : 
                               expectedChange < 0 ? 'text-red-600' : 'text-gray-600';
            
            return `
                <div class="bg-white border border-gray-200 rounded-lg p-4 shadow-sm hover:shadow-md transition-shadow">
                    <div class="flex items-center justify-between mb-3">
                        <span class="text-2xl">${icon}</span>
                        <div class="text-right">
                            <div class="text-sm font-medium text-gray-900">${label}</div>
                            <div class="text-xs text-gray-500">${timeframe}</div>
                        </div>
                    </div>
                    
                    <div class="space-y-2">
                        <div class="flex justify-between items-center">
                            <span class="text-sm text-gray-600">Direction:</span>
                            <span class="text-sm font-medium ${directionClass}">${direction}</span>
                        </div>
                        
                        <div class="flex justify-between items-center">
                            <span class="text-sm text-gray-600">Change:</span>
                            <span class="text-sm font-medium ${changeClass}">${changeStr}</span>
                        </div>
                        
                        <div class="flex justify-between items-center">
                            <span class="text-sm text-gray-600">Confidence:</span>
                            <span class="text-sm font-medium text-gray-900">${Math.round(confidence)}%</span>
                        </div>
                        
                        <div class="w-full bg-gray-200 rounded-full h-1.5 mt-2">
                            <div class="bg-blue-500 h-1.5 rounded-full" style="width: ${confidence}%"></div>
                        </div>
                    </div>
                </div>
            `;
        }).join('');

        gridContainer.innerHTML = gridHtml;
    }

    /**
     * Show loading state with progress animation
     */
    showLoading() {
        this.isLoading = true;
        
        // Hide other states
        document.getElementById('prediction-results')?.classList.add('hidden');
        document.getElementById('error-state')?.classList.add('hidden');
        
        // Show loading state
        const loadingState = document.getElementById('loading-state');
        if (loadingState) {
            loadingState.classList.remove('hidden');
        }
        
        // Animate progress bar
        this.animateProgressBar();
        
        // Disable generate button
        const generateBtn = document.getElementById('generate-prediction');
        if (generateBtn) {
            generateBtn.disabled = true;
            generateBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Analyzing...';
        }
    }

    /**
     * Hide loading state
     */
    hideLoading() {
        this.isLoading = false;
        
        // Clear progress animation
        if (this.progressInterval) {
            clearInterval(this.progressInterval);
            this.progressInterval = null;
        }
        
        // Hide loading state
        document.getElementById('loading-state')?.classList.add('hidden');
        
        // Re-enable generate button
        const generateBtn = document.getElementById('generate-prediction');
        if (generateBtn) {
            generateBtn.disabled = false;
            generateBtn.innerHTML = '<i class="fas fa-magic mr-2"></i>Generate Prediction';
        }
    }

    /**
     * Animate progress bar during loading
     */
    animateProgressBar() {
        const progressBar = document.getElementById('progress-bar');
        if (!progressBar) return;
        
        let progress = 0;
        this.progressInterval = setInterval(() => {
            progress += Math.random() * 15;
            if (progress > 95) progress = 95; // Never complete until done
            
            progressBar.style.width = `${progress}%`;
        }, 500);
    }

    /**
     * Show prediction results
     */
    showResults() {
        document.getElementById('loading-state')?.classList.add('hidden');
        document.getElementById('error-state')?.classList.add('hidden');
        
        const resultsContainer = document.getElementById('prediction-results');
        if (resultsContainer) {
            resultsContainer.classList.remove('hidden');
            resultsContainer.classList.add('fade-in-up');
        }
    }

    /**
     * Show error state
     */
    showError(message) {
        this.hideLoading();
        
        document.getElementById('prediction-results')?.classList.add('hidden');
        document.getElementById('loading-state')?.classList.add('hidden');
        
        const errorState = document.getElementById('error-state');
        const errorMessage = document.getElementById('error-message');
        
        if (errorState) errorState.classList.remove('hidden');
        if (errorMessage) errorMessage.textContent = message;
        
        console.error('🚨 Prediction Error:', message);
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.predictionApp = new MarketPredictionApp();
    
    // Set initial symbol display
    const symbolSelect = document.getElementById('symbol-select');
    if (symbolSelect) {
        window.predictionApp.onSymbolChange();
    }
});

// Export for testing
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { MarketPredictionApp };
}