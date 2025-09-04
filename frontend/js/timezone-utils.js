/**
 * GSMT Ver 7.0 - Sydney Timezone Utilities
 * JavaScript utilities for Sydney timezone handling and display
 */

class SydneyTimezoneUtils {
    constructor() {
        // Sydney timezone identifier
        this.sydneyTimezone = 'Australia/Sydney';
        
        // Market timezone mappings
        this.marketTimezones = {
            'Australia': 'Australia/Sydney',
            'Japan': 'Asia/Tokyo',
            'Hong Kong': 'Asia/Hong_Kong', 
            'China': 'Asia/Shanghai',
            'UK': 'Europe/London',
            'Germany': 'Europe/Berlin',
            'France': 'Europe/Paris',
            'US': 'America/New_York'
        };
        
        // Market display names with flags
        this.marketDisplayNames = {
            'Australia': '🇦🇺 Sydney',
            'Japan': '🇯🇵 Tokyo',
            'Hong Kong': '🇭🇰 Hong Kong',
            'China': '🇨🇳 Shanghai', 
            'UK': '🇬🇧 London',
            'Germany': '🇩🇪 Frankfurt',
            'France': '🇫🇷 Paris',
            'US': '🇺🇸 New York'
        };
        
        // Check browser support for Intl.DateTimeFormat
        this.browserSupported = typeof Intl !== 'undefined' && 
                                typeof Intl.DateTimeFormat !== 'undefined';
        
        if (!this.browserSupported) {
            console.warn('Browser timezone support limited - some features may not work correctly');
        }
    }
    
    /**
     * Get current time in Sydney timezone
     * @returns {Date} Current Sydney time
     */
    getSydneyNow() {
        return new Date(new Date().toLocaleString("en-US", {timeZone: this.sydneyTimezone}));
    }
    
    /**
     * Get Sydney time for a specific date
     * @param {Date|string|number} date - Date to convert
     * @returns {Date} Date in Sydney timezone
     */
    toSydneyTime(date) {
        const targetDate = new Date(date);
        return new Date(targetDate.toLocaleString("en-US", {timeZone: this.sydneyTimezone}));
    }
    
    /**
     * Get 10am Sydney time for today (start of 24-hour period)
     * @param {Date} referenceDate - Optional reference date
     * @returns {Date} 10am Sydney time
     */
    getSydney10amStart(referenceDate = null) {
        const sydneyDate = referenceDate ? this.toSydneyTime(referenceDate) : this.getSydneyNow();
        
        // Set to 10:00 AM Sydney time
        const start = new Date(sydneyDate);
        start.setHours(10, 0, 0, 0);
        
        return start;
    }
    
    /**
     * Get 24-hour period from 10am Sydney time
     * @param {Date} referenceDate - Optional reference date
     * @returns {Object} {start: Date, end: Date} 24-hour period
     */
    get24HourPeriodFromSydney10am(referenceDate = null) {
        const start = this.getSydney10amStart(referenceDate);
        const end = new Date(start);
        end.setHours(start.getHours() + 24);
        
        return { start, end };
    }
    
    /**
     * Format Sydney time for display
     * @param {Date} date - Date to format
     * @param {Object} options - Formatting options
     * @returns {string} Formatted Sydney time string
     */
    formatSydneyTime(date, options = {}) {
        const defaultOptions = {
            showTimezone: true,
            showDate: true,
            showTime: true,
            format: 'long' // 'short', 'medium', 'long', 'full'
        };
        
        const opts = { ...defaultOptions, ...options };
        const sydneyTime = this.toSydneyTime(date);
        
        if (!this.browserSupported) {
            return sydneyTime.toLocaleString();
        }
        
        try {
            let formatOptions = {
                timeZone: this.sydneyTimezone,
                timeZoneName: opts.showTimezone ? 'short' : undefined
            };
            
            if (opts.showDate && opts.showTime) {
                formatOptions = {
                    ...formatOptions,
                    year: 'numeric',
                    month: opts.format === 'short' ? 'numeric' : 'short',
                    day: 'numeric',
                    hour: '2-digit',
                    minute: '2-digit',
                    second: opts.format === 'full' ? '2-digit' : undefined
                };
            } else if (opts.showTime) {
                formatOptions = {
                    ...formatOptions,
                    hour: '2-digit',
                    minute: '2-digit',
                    second: opts.format === 'full' ? '2-digit' : undefined
                };
            } else if (opts.showDate) {
                formatOptions = {
                    ...formatOptions,
                    year: 'numeric',
                    month: opts.format === 'short' ? 'numeric' : 'short',
                    day: 'numeric'
                };
            }
            
            return sydneyTime.toLocaleString('en-AU', formatOptions);
            
        } catch (error) {
            console.warn('Sydney time formatting error:', error);
            return sydneyTime.toLocaleString();
        }
    }
    
    /**
     * Get timezone offset for Sydney
     * @param {Date} date - Date to check offset for
     * @returns {number} Offset in minutes
     */
    getSydneyTimezoneOffset(date = new Date()) {
        if (!this.browserSupported) {
            return -600; // Fallback: assume AEST (+10)
        }
        
        try {
            const utc = new Date(date.toISOString());
            const sydney = new Date(date.toLocaleString("en-US", {timeZone: this.sydneyTimezone}));
            
            return (sydney.getTime() - utc.getTime()) / (1000 * 60);
        } catch (error) {
            console.warn('Sydney timezone offset calculation error:', error);
            return -600; // Fallback
        }
    }
    
    /**
     * Check if Sydney is currently in daylight saving time
     * @param {Date} date - Date to check
     * @returns {boolean} True if in daylight saving time
     */
    isSydneyDST(date = new Date()) {
        // Sydney DST typically runs from early October to early April
        // This is a simplified check - actual dates vary year to year
        const month = this.toSydneyTime(date).getMonth();
        return month >= 9 || month <= 2; // Oct-Feb (approximately)
    }
    
    /**
     * Get current market status for all markets
     * @returns {Object} Market statuses
     */
    getGlobalMarketStatus() {
        const sydneyNow = this.getSydneyNow();
        const status = {};
        
        // Standard trading hours in local time (24-hour format)
        const marketHours = {
            'Australia': { open: 10, close: 16 },  // 10:00 - 16:00 AEST/AEDT
            'Japan': { open: 9, close: 15 },       // 09:00 - 15:00 JST
            'Hong Kong': { open: 9, close: 16 },   // 09:00 - 16:00 HKT
            'China': { open: 9, close: 15 },       // 09:00 - 15:00 CST
            'UK': { open: 8, close: 16 },          // 08:00 - 16:30 GMT/BST
            'Germany': { open: 9, close: 17 },     // 09:00 - 17:30 CET/CEST
            'France': { open: 9, close: 17 },      // 09:00 - 17:30 CET/CEST
            'US': { open: 9, close: 16 }           // 09:30 - 16:00 EST/EDT
        };
        
        for (const [market, timezone] of Object.entries(this.marketTimezones)) {
            try {
                const marketTime = new Date(sydneyNow.toLocaleString("en-US", {timeZone: timezone}));
                const hours = marketTime.getHours() + (marketTime.getMinutes() / 60);
                const marketSession = marketHours[market];
                
                const isOpen = hours >= marketSession.open && hours <= marketSession.close;
                
                status[market] = {
                    isOpen,
                    localTime: marketTime.toLocaleTimeString('en-US', {
                        timeZone: timezone,
                        hour: '2-digit',
                        minute: '2-digit',
                        timeZoneName: 'short'
                    }),
                    hours: marketSession,
                    displayName: this.marketDisplayNames[market] || market
                };
                
            } catch (error) {
                console.warn(`Error checking market status for ${market}:`, error);
                status[market] = {
                    isOpen: false,
                    localTime: 'Unknown',
                    hours: marketHours[market] || { open: 9, close: 17 },
                    displayName: this.marketDisplayNames[market] || market
                };
            }
        }
        
        return status;
    }
    
    /**
     * Get active markets count
     * @returns {number} Number of currently active markets
     */
    getActiveMarketsCount() {
        const status = this.getGlobalMarketStatus();
        return Object.values(status).filter(market => market.isOpen).length;
    }
    
    /**
     * Get recommended refresh interval based on market activity
     * @returns {number} Refresh interval in seconds
     */
    getRecommendedRefreshInterval() {
        const activeCount = this.getActiveMarketsCount();
        
        if (activeCount >= 3) {
            return 180; // 3 minutes - high activity
        } else if (activeCount >= 1) {
            return 300; // 5 minutes - moderate activity
        } else {
            return 600; // 10 minutes - low activity
        }
    }
    
    /**
     * Create Sydney time indicator element
     * @param {HTMLElement} container - Container to append indicator
     * @param {Object} options - Display options
     */
    createSydneyTimeIndicator(container, options = {}) {
        const defaultOptions = {
            showSeconds: false,
            showTimezone: true,
            updateInterval: 1000,
            cssClass: 'sydney-time-indicator'
        };
        
        const opts = { ...defaultOptions, ...options };
        
        // Create indicator element
        const indicator = document.createElement('div');
        indicator.className = opts.cssClass;
        indicator.innerHTML = `
            <div class="sydney-time-display">
                <span class="sydney-time-label">Sydney Time:</span>
                <span class="sydney-time-value"></span>
            </div>
        `;
        
        const timeValue = indicator.querySelector('.sydney-time-value');
        
        // Update function
        const updateTime = () => {
            const sydneyNow = this.getSydneyNow();
            const formatted = this.formatSydneyTime(sydneyNow, {
                showDate: false,
                showTime: true,
                showTimezone: opts.showTimezone,
                format: opts.showSeconds ? 'full' : 'medium'
            });
            
            timeValue.textContent = formatted;
        };
        
        // Initial update
        updateTime();
        
        // Set up interval
        if (opts.updateInterval > 0) {
            setInterval(updateTime, opts.updateInterval);
        }
        
        // Append to container
        container.appendChild(indicator);
        
        return indicator;
    }
    
    /**
     * Convert timestamp to Sydney timezone for chart display
     * @param {number} timestamp - Unix timestamp in milliseconds
     * @returns {Object} Formatted time data for charts
     */
    formatTimestampForChart(timestamp) {
        const date = new Date(timestamp);
        const sydneyTime = this.toSydneyTime(date);
        
        return {
            sydney: sydneyTime,
            display: this.formatSydneyTime(sydneyTime, { showDate: false, showTime: true }),
            iso: sydneyTime.toISOString(),
            timestamp: sydneyTime.getTime()
        };
    }
    
    /**
     * Create market session timeline for visualization
     * @returns {Array} Timeline data for charts
     */
    createMarketSessionTimeline() {
        const period = this.get24HourPeriodFromSydney10am();
        const timeline = [];
        const marketStatus = this.getGlobalMarketStatus();
        
        // Create hourly timeline
        for (let hour = 0; hour < 24; hour++) {
            const timePoint = new Date(period.start);
            timePoint.setHours(period.start.getHours() + hour);
            
            const activeMarkets = [];
            
            // Check which markets are active at this hour
            Object.entries(marketStatus).forEach(([market, status]) => {
                const marketHour = new Date(timePoint.toLocaleString("en-US", {
                    timeZone: this.marketTimezones[market]
                })).getHours();
                
                const isActive = marketHour >= status.hours.open && marketHour <= status.hours.close;
                
                if (isActive) {
                    activeMarkets.push({
                        market,
                        displayName: status.displayName
                    });
                }
            });
            
            timeline.push({
                hour,
                sydneyTime: this.formatSydneyTime(timePoint, { showDate: false }),
                timestamp: timePoint.getTime(),
                activeMarkets,
                activityLevel: activeMarkets.length
            });
        }
        
        return timeline;
    }
}

// Create global instance
window.sydneyTimeUtils = new SydneyTimezoneUtils();

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SydneyTimezoneUtils;
}