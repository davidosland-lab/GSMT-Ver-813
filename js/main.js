/* Global Markets Tracker - 24-hour view with index selection */

const chartEl = document.getElementById('chart');
const chart = echarts.init(chartEl, null, { renderer: 'canvas' });

const modeSelect = document.getElementById('modeSelect');
const indexSelect = document.getElementById('indexSelect');
const intervalSelect = document.getElementById('intervalSelect');
const dateSelect = document.getElementById('dateSelect');
const providerSelect = document.getElementById('providerSelect');
const refreshBtn = document.getElementById('refreshBtn');
const legendList = document.getElementById('legendList');

const { INDEX_DEFS, getSeries } = window.MarketsProviders;

// Populate indices into candlestick picker
INDEX_DEFS.forEach((idx, i) => {
  const opt = document.createElement('option');
  opt.value = idx.code; opt.textContent = idx.name; indexSelect.appendChild(opt);
});
indexSelect.value = '^AXJO'; // Default to ASX 200 for Australian users

// Set default date to today
dateSelect.value = new Date().toISOString().split('T')[0];

// Create index selection checkboxes
function createIndexSelector() {
  const container = document.getElementById('indicesSelector');
  
  INDEX_DEFS.forEach((idx, i) => {
    const div = document.createElement('div');
    div.className = 'flex items-center gap-2';
    
    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.id = `idx_${idx.code}`;
    checkbox.value = idx.code;
    checkbox.checked = true; // Default all selected
    checkbox.className = 'index-checkbox';
    checkbox.addEventListener('change', render);
    
    const label = document.createElement('label');
    label.htmlFor = `idx_${idx.code}`;
    label.className = 'flex items-center gap-2 text-sm cursor-pointer';
    label.innerHTML = `
      <span class="inline-block w-3 h-3 rounded-full" style="background: ${idx.color}"></span>
      <span>${idx.name}</span>
    `;
    
    div.appendChild(checkbox);
    div.appendChild(label);
    container.appendChild(div);
  });
}

// Get selected indices
function getSelectedIndices() {
  const checkboxes = document.querySelectorAll('.index-checkbox:checked');
  return Array.from(checkboxes).map(cb => cb.value);
}

createIndexSelector();

// Selection helper buttons
document.getElementById('selectAllBtn').addEventListener('click', () => {
  document.querySelectorAll('.index-checkbox').forEach(cb => cb.checked = true);
  render();
});

document.getElementById('selectNoneBtn').addEventListener('click', () => {
  document.querySelectorAll('.index-checkbox').forEach(cb => cb.checked = false);
  render();
});

let regionCycle = 0;
document.getElementById('selectRegionBtn').addEventListener('click', () => {
  const regions = ['Oceania', 'Asia', 'Europe', 'Americas'];
  const currentRegion = regions[regionCycle % regions.length];
  
  document.querySelectorAll('.index-checkbox').forEach(cb => {
    const idx = INDEX_DEFS.find(i => i.code === cb.value);
    cb.checked = idx && idx.region === currentRegion;
  });
  
  regionCycle++;
  document.getElementById('selectRegionBtn').textContent = 
    `By Region (${regions[(regionCycle) % regions.length]})`;
  
  render();
});

function setControlsVisibility() {
  const mode = modeSelect.value;
  // Show index picker for candlestick view
  document.getElementById('indexPickerWrap').style.display = (mode === 'candlestick') ? 'flex' : 'none';
}
setControlsVisibility();
modeSelect.addEventListener('change', () => { setControlsVisibility(); render(); });
[intervalSelect, dateSelect, providerSelect].forEach(el => el.addEventListener('change', render));
refreshBtn.addEventListener('click', render);

// Add market hours checkbox listener
document.getElementById('showMarketHours').addEventListener('change', render);

window.addEventListener('resize', () => chart.resize());

function toPercentSeries(series) {
  if (!series.length) return [];
  const base = series[0].close ?? series[0].value;
  return series.map(p => ({ time: p.time, value: ((p.close ?? p.value) / base - 1) * 100 }));
}

function get24HourDataZoom() {
  return [
    { 
      type: 'inside',
      start: 0,
      end: 100
    }, 
    { 
      type: 'slider',
      start: 0,
      end: 100,
      height: 30
    }
  ];
}

function buildLineOption(seriesMap, selectedDate) {
  // Filter series to only show data during market hours
  const series = Object.entries(seriesMap)
    .filter(([code, data]) => data.length > 0) // Only include markets with data
    .map(([code, data]) => {
      const idx = INDEX_DEFS.find(i => i.code === code);
      return {
        name: idx?.name || code,
        type: 'line',
        showSymbol: false,
        smooth: true,
        emphasis: { focus: 'series' },
        lineStyle: { width: 2, color: idx?.color },
        data: data.map(p => [p.time, p.value])
      };
    });

  // Create 24-hour timeline
  const baseDate = new Date(selectedDate + 'T00:00:00.000Z');
  const endDate = new Date(baseDate.getTime() + 24 * 60 * 60 * 1000);

  const option = {
    tooltip: { 
      trigger: 'axis', 
      valueFormatter: v => v!=null ? v.toFixed(2)+'%' : '',
      axisPointer: { type: 'cross' }
    },
    legend: { show: false },
    grid: { left: 60, right: 30, top: 40, bottom: 80 },
    xAxis: { 
      type: 'time',
      min: baseDate,
      max: endDate,
      axisLabel: {
        formatter: function(value) {
          const date = new Date(value);
          // Display as Australian local time (00:00 = midnight Sydney time)
          const hours = date.getUTCHours().toString().padStart(2, '0');
          const minutes = date.getUTCMinutes().toString().padStart(2, '0');
          return `${hours}:${minutes}`;
        }
      },
      splitLine: {
        show: true,
        lineStyle: { type: 'dashed', color: '#e2e8f0' }
      }
    },
    yAxis: { type:'value', axisLabel: { formatter: v => v+'%' } },
    dataZoom: get24HourDataZoom(),
    series
  };

  // Add market hours visualization if enabled
  if (document.getElementById('showMarketHours')?.checked) {
    option.graphic = getMarketHoursGraphics(selectedDate);
  }

  return option;
}

function getMarketHoursGraphics(selectedDate) {
  const baseDate = new Date(selectedDate + 'T00:00:00.000Z');
  const graphics = [];

  INDEX_DEFS.forEach((idx, i) => {
    const [openHour, openMin] = idx.marketOpen.split(':').map(Number);
    const [closeHour, closeMin] = idx.marketClose.split(':').map(Number);
    
    const marketOpenTime = new Date(baseDate);
    marketOpenTime.setUTCHours(openHour, openMin, 0, 0);
    
    const marketCloseTime = new Date(baseDate);
    marketCloseTime.setUTCHours(closeHour, closeMin, 0, 0);
    
    // Handle markets that close the next day
    if (closeHour < openHour) {
      marketCloseTime.setUTCDate(marketCloseTime.getUTCDate() + 1);
    }

    // Add market name label
    graphics.push({
      type: 'text',
      left: 70 + (i * 80),
      top: 10,
      style: {
        text: idx.name.split(' ')[0], // Just the market name (e.g., "Nikkei", "FTSE")
        fontSize: 10,
        fill: idx.color,
        fontWeight: 'bold'
      },
      z: 100
    });
  });

  return graphics;
}

function buildCandleOption(candle, selectedDate) {
  const selectedIdx = INDEX_DEFS.find(idx => idx.code === indexSelect.value);
  
  // Create 24-hour timeline
  const baseDate = new Date(selectedDate + 'T00:00:00.000Z');
  const endDate = new Date(baseDate.getTime() + 24 * 60 * 60 * 1000);
  
  const option = {
    tooltip: { 
      trigger: 'axis',
      formatter: function(params) {
        if (!params || params.length === 0) return '';
        const data = params[0];
        const time = new Date(data.axisValue);
        const timeStr = time.toLocaleString();
        const ohlc = data.data;
        return `${timeStr}<br/>
               Open: ${ohlc[1]?.toFixed(2)}<br/>
               High: ${ohlc[4]?.toFixed(2)}<br/>
               Low: ${ohlc[3]?.toFixed(2)}<br/>
               Close: ${ohlc[2]?.toFixed(2)}`;
      }
    },
    legend: { show: false },
    grid: { left: 60, right: 30, top: 40, bottom: 80 },
    xAxis: { 
      type: 'time',
      min: baseDate,
      max: endDate,
      axisLabel: {
        formatter: function(value) {
          const date = new Date(value);
          // Display as Australian local time (00:00 = midnight Sydney time)
          const hours = date.getUTCHours().toString().padStart(2, '0');
          const minutes = date.getUTCMinutes().toString().padStart(2, '0');
          return `${hours}:${minutes}`;
        }
      },
      splitLine: {
        show: true,
        lineStyle: { type: 'dashed', color: '#e2e8f0' }
      }
    },
    yAxis: { scale: true },
    dataZoom: get24HourDataZoom(),
    series: candle.length > 0 ? [
      { 
        type: 'candlestick', 
        name: 'OHLC', 
        data: candle.map(p => [p.time, p.open, p.close, p.low, p.high]),
        itemStyle: {
          color: selectedIdx?.color || '#ef4444',
          color0: selectedIdx?.color || '#ef4444',
          borderColor: selectedIdx?.color || '#ef4444',
          borderColor0: selectedIdx?.color || '#ef4444',
        }
      },
      { 
        type: 'line', 
        name: 'Close', 
        showSymbol: false, 
        smooth: true, 
        data: candle.map(p => [p.time, p.close]), 
        lineStyle: { width: 1, color: '#64748b' } 
      }
    ] : []
  };

  // Add market hours indicator for the selected index
  if (document.getElementById('showMarketHours')?.checked && selectedIdx) {
    option.graphic = [{
      type: 'text',
      left: 70,
      top: 5,
      style: {
        text: `Market Hours: ${selectedIdx.marketOpen} - ${selectedIdx.marketClose} Australian Time (${selectedIdx.timezone})`,
        fontSize: 12,
        fill: selectedIdx.color
      }
    }];
  }

  return option;
}

function updateLegend(list) {
  legendList.innerHTML = '';
  list.forEach(item => {
    const li = document.createElement('li');
    li.className = 'legend-badge';
    li.innerHTML = `<span class="legend-dot" style="background:${item.color}"></span><span>${item.name}</span>`;
    legendList.appendChild(li);
  });
}

async function render() {
  const mode = modeSelect.value;
  const interval = intervalSelect.value;
  const selectedDate = dateSelect.value;
  const provider = providerSelect.value;

  if (!selectedDate) return;

  chart.showLoading('default', { maskColor: 'rgba(255,255,255,0.6)' });
  try {
    if (mode === 'line') {
      const selectedCodes = getSelectedIndices();
      if (selectedCodes.length === 0) {
        chart.setOption({ 
          title: { text: 'Please select at least one index', left: 'center', textStyle: { color: '#64748b' } },
          series: []
        });
        updateLegend([]);
        return;
      }

      const results = await Promise.all(selectedCodes.map(c => getSeries(c, interval, selectedDate, provider)));
      const percentSeries = {};
      
      // Only include markets that have data and are selected
      selectedCodes.forEach((code, i) => {
        if (results[i].length > 0) {
          percentSeries[code] = toPercentSeries(results[i]);
        }
      });
      
      const opt = buildLineOption(percentSeries, selectedDate);
      chart.setOption(opt, true);
      
      // Update legend to only show active selected markets
      const activeIndices = INDEX_DEFS.filter(idx => percentSeries[idx.code]);
      updateLegend(activeIndices);
    } else {
      const code = indexSelect.value;
      const idx = INDEX_DEFS.find(i => i.code === code) || { name: code, color: '#000' };
      const data = await getSeries(code, interval, selectedDate, provider);
      const opt = buildCandleOption(data, selectedDate);
      chart.setOption(opt, true);
      updateLegend([idx]);
    }
  } catch (e) {
    console.error(e);
    chart.setOption({ title: { text: 'Error loading data', left: 'center' } });
  } finally {
    chart.hideLoading();
  }
}

render();