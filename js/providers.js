/* Data providers: demo data generator and YFinance fetcher for 24-hour market tracking */

(function(){
  function getCSS(varName) {
    return getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
  }

  // Index definitions with market hours (in UTC) and regions
  const INDEX_DEFS = [
    // Australian Indices (times in Australian local time reference)
    { code: '^AXJO', name: 'ASX 200 (Australia)', color: getCSS('--color-asx200'), marketOpen: '10:00', marketClose: '16:00', timezone: 'AEDT', region: 'Oceania' },
    { code: '^AXKO', name: 'ASX 300 (Australia)', color: getCSS('--color-asx300'), marketOpen: '10:00', marketClose: '16:00', timezone: 'AEDT', region: 'Oceania' },
    { code: '^AFLI', name: 'ASX 50 (Australia)', color: getCSS('--color-asx50'), marketOpen: '10:00', marketClose: '16:00', timezone: 'AEDT', region: 'Oceania' },
    { code: '^AXMD', name: 'ASX 100 (Australia)', color: getCSS('--color-asx100'), marketOpen: '10:00', marketClose: '16:00', timezone: 'AEDT', region: 'Oceania' },
    { code: '^AXSO', name: 'ASX Small Cap (Australia)', color: getCSS('--color-asx-small'), marketOpen: '10:00', marketClose: '16:00', timezone: 'AEDT', region: 'Oceania' },
    // Asian Indices (times shown in Australian reference - when they trade relative to Sydney midnight)
    { code: '^N225', name: 'Nikkei 225 (Japan)', color: getCSS('--color-nikkei'), marketOpen: '11:00', marketClose: '17:00', timezone: 'JST', region: 'Asia' },
    { code: '^HSI', name: 'Hang Seng (Hong Kong)', color: getCSS('--color-hsi'), marketOpen: '12:30', marketClose: '19:00', timezone: 'HKT', region: 'Asia' },
    { code: '000001.SS', name: 'SSE Composite (China)', color: getCSS('--color-sse'), marketOpen: '12:30', marketClose: '18:00', timezone: 'CST', region: 'Asia' },
    { code: '^NSEI', name: 'NIFTY 50 (India)', color: getCSS('--color-nifty'), marketOpen: '14:45', marketClose: '21:00', timezone: 'IST', region: 'Asia' },
    // European Indices (times shown in Australian reference)
    { code: '^FTSE', name: 'FTSE 100 (UK)', color: getCSS('--color-ftse'), marketOpen: '19:00', marketClose: '03:30', timezone: 'GMT', region: 'Europe' },
    { code: '^GDAXI', name: 'DAX (Germany)', color: getCSS('--color-dax'), marketOpen: '18:00', marketClose: '02:30', timezone: 'CET', region: 'Europe' },
    { code: '^FCHI', name: 'CAC 40 (France)', color: getCSS('--color-cac'), marketOpen: '18:00', marketClose: '02:30', timezone: 'CET', region: 'Europe' },
    // US Indices (times shown in Australian reference)  
    { code: '^GSPC', name: 'S&P 500 (US)', color: getCSS('--color-sp500'), marketOpen: '01:30', marketClose: '08:00', timezone: 'EST', region: 'Americas' },
    { code: '^DJI', name: 'Dow Jones (US)', color: getCSS('--color-djia'), marketOpen: '01:30', marketClose: '08:00', timezone: 'EST', region: 'Americas' },
    { code: '^IXIC', name: 'NASDAQ (US)', color: getCSS('--color-nasdaq'), marketOpen: '01:30', marketClose: '08:00', timezone: 'EST', region: 'Americas' },
  ];

  function intervalToMs(interval) {
    switch(interval) {
      case '5min': return 5*60*1000;
      case '15min': return 15*60*1000;
      case '30min': return 30*60*1000;
      case '1h': return 60*60*1000;
      case '4h': return 4*60*60*1000;
      case '1day': return 24*60*60*1000;
      case '1week': return 7*24*60*60*1000;
      case '1month': return 30*24*60*60*1000;
      default: return 24*60*60*1000;
    }
  }

  // Provider: Demo - Generate 24-hour data with market-specific timing
  async function fetchDemoSeries(symbol, interval, selectedDate) {
    const idx = INDEX_DEFS.find(i => i.code === symbol);
    if (!idx) return [];

    const intervalMs = intervalToMs(interval);
    const [openHour, openMin] = idx.marketOpen.split(':').map(Number);
    const [closeHour, closeMin] = idx.marketClose.split(':').map(Number);
    
    // Create date objects for market open/close times (times are now in Australian local time reference)
    const baseDate = new Date(selectedDate + 'T00:00:00.000Z');
    const marketOpenTime = new Date(baseDate);
    marketOpenTime.setUTCHours(openHour, openMin, 0, 0);
    
    const marketCloseTime = new Date(baseDate);
    marketCloseTime.setUTCHours(closeHour, closeMin, 0, 0);
    
    // Handle markets that close the next day (like European and US markets)
    if (closeHour < openHour) {
      marketCloseTime.setUTCDate(marketCloseTime.getUTCDate() + 1);
    }

    // Generate data points only during market hours
    const series = [];
    const base = Math.abs(symbol.split('').reduce((acc, ch)=>acc+ch.charCodeAt(0), 0)) % 2000 + 500;
    let price = base;
    let x = base; // seed for random walk
    
    const rand = () => { x ^= x << 13; x ^= x >> 17; x ^= x << 5; return (x >>> 0) / 4294967296; };
    
    let currentTime = new Date(marketOpenTime);
    
    while (currentTime <= marketCloseTime) {
      const r = (0.0002 - 0.5*0.01*0.01) + 0.01 * (rand()*2 - 1);
      price = price * Math.exp(r);
      const open = price * (1 + (rand()-0.5)*0.01);
      const close = price * (1 + (rand()-0.5)*0.01);
      const high = Math.max(open, close) * (1 + rand()*0.01);
      const low = Math.min(open, close) * (1 - rand()*0.01);
      
      series.push({ 
        time: new Date(currentTime), 
        open, high, low, close, 
        value: close 
      });
      
      currentTime = new Date(currentTime.getTime() + intervalMs);
    }
    
    return series;
  }

  // Provider: YFinance (via Python backend)
  async function fetchYFinanceSeries(symbol, interval, selectedDate) {
    // Map frontend intervals to YFinance intervals
    const intervalMap = { '5min': '5m', '15min': '15m', '30min': '30m', '1h': '1h', '4h': '4h', '1day': '1d' };
    const yInterval = intervalMap[interval] || '5m';
    
    // Backend API URL - Replace with your actual Railway URL!
    const API_BASE = window.location.hostname === 'localhost' 
      ? 'http://localhost:5000'  // Local development
      : 'https://global-indices-tracker-production.up.railway.app';  // ✅ Updated with live Railway URL
    
    try {
      console.log(`YFinance API call: ${symbol}, date=${selectedDate}, interval=${yInterval}`);
      
      const response = await fetch(
        `${API_BASE}/api/stock/${encodeURIComponent(symbol)}?interval=${yInterval}&date=${selectedDate}`
      );
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const json = await response.json();
      
      if (json.error) {
        throw new Error(json.error);
      }
      
      // Transform API response to expected format
      return json.data.map(item => ({
        time: new Date(item.time),
        open: item.open || 0,
        high: item.high || 0,
        low: item.low || 0,
        close: item.close || 0,
        value: item.value || item.close || 0
      }));
      
    } catch (error) {
      console.error(`YFinance API error for ${symbol}:`, error);
      throw error; // Re-throw to trigger fallback
    }
  }

  async function getSeries(symbol, interval, selectedDate, provider) {
    if (provider === 'yfinance') {
      try { 
        const data = await fetchYFinanceSeries(symbol, interval, selectedDate);
        return data;
      }
      catch(e) { 
        console.warn('YFinance fallback to demo for', symbol, e);
      }
    }
    return await fetchDemoSeries(symbol, interval, selectedDate);
  }

  // Market hours utilities
  function getMarketHours() {
    return INDEX_DEFS.map(idx => ({
      name: idx.name,
      open: idx.marketOpen,
      close: idx.marketClose,
      timezone: idx.timezone,
      color: idx.color
    }));
  }

  window.MarketsProviders = { INDEX_DEFS, getSeries, getMarketHours };
})();