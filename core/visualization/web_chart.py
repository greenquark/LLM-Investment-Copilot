from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
import json
from flask import Flask, render_template_string, jsonify
import threading
import webbrowser
import time

from core.models.bar import Bar
from core.backtest.result import BacktestResult
from core.visualization.models import TradeSignal, IndicatorData


class WebChartServer:
    """Web-based chart visualization server."""
    
    def __init__(self, port: int = 5000):
        self.port = port
        self.app = Flask(__name__)
        self.data = None
        self.server_thread = None
        
        # Setup routes
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def index():
            return render_template_string(self._get_html_template())
        
        @self.app.route('/api/data')
        def get_data():
            if self.data is None:
                return jsonify({"error": "No data available"}), 404
            return jsonify(self.data)
    
    def _get_html_template(self) -> str:
        """Get the HTML template with Chart.js."""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Trading Chart - Revised MP2.0</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3.0.0/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
    <style>
        body {
            margin: 0;
            padding: 20px;
            background: #1e1e1e;
            color: #ffffff;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .container {
            max-width: 100%;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            color: #00FF66;
            margin-bottom: 20px;
        }
        .chart-container {
            background: #2d2d2d;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }
        .price-chart-container canvas {
            height: 70vh !important;
        }
        .volume-chart-container canvas {
            height: 10vh !important;
        }
        .indicator-chart-container canvas {
            height: 20vh !important;
        }
        .equity-chart-container canvas {
            height: 20vh !important;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        .stat-card {
            background: #2d2d2d;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #00FF66;
        }
        .stat-label {
            font-size: 12px;
            color: #aaa;
            margin-bottom: 5px;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #00FF66;
        }
        .loading {
            text-align: center;
            padding: 40px;
            color: #aaa;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Revised MP2.0 - Trading Chart</h1>
        
        <div id="loading" class="loading">Loading chart data...</div>
        
        <div id="content" style="display: none;">
            <div class="stats" id="stats"></div>
            
            <div class="chart-container price-chart-container">
                <h3>Price Chart</h3>
                <canvas id="priceChart"></canvas>
            </div>
            
            <div class="chart-container volume-chart-container">
                <h3>Volume</h3>
                <canvas id="volumeChart"></canvas>
            </div>
            
            <div class="chart-container indicator-chart-container">
                <h3>Revised MP2.0 Indicator with Signals</h3>
                <canvas id="indicatorChart"></canvas>
            </div>
            
            <div class="chart-container equity-chart-container">
                <h3>Equity Curve</h3>
                <canvas id="equityChart"></canvas>
            </div>
        </div>
    </div>

    <script>
        let priceChart, volumeChart, indicatorChart, equityChart;
        
        async function loadData() {
            try {
                const response = await fetch('/api/data');
                const data = await response.json();
                
                if (data.error) {
                    document.getElementById('loading').textContent = 'Error: ' + data.error;
                    return;
                }
                
                document.getElementById('loading').style.display = 'none';
                document.getElementById('content').style.display = 'block';
                
                // Display stats
                if (data.metrics) {
                    displayStats(data.metrics);
                }
                
                // Create charts
                createPriceChart(data);
                createVolumeChart(data);
                createIndicatorChart(data);
                createEquityChart(data);
            } catch (error) {
                document.getElementById('loading').textContent = 'Error loading data: ' + error.message;
            }
        }
        
        function displayStats(metrics) {
            const statsDiv = document.getElementById('stats');
            const stats = [
                { label: 'Total Return', value: (metrics.total_return * 100).toFixed(2) + '%' },
                { label: 'CAGR', value: (metrics.cagr * 100).toFixed(2) + '%' },
                { label: 'Sharpe Ratio', value: metrics.sharpe.toFixed(2) },
                { label: 'Max Drawdown', value: (metrics.max_drawdown * 100).toFixed(2) + '%' },
            ];
            
            statsDiv.innerHTML = stats.map(stat => `
                <div class="stat-card">
                    <div class="stat-label">${stat.label}</div>
                    <div class="stat-value">${stat.value}</div>
                </div>
            `).join('');
        }
        
        function createPriceChart(data) {
            const ctx = document.getElementById('priceChart').getContext('2d');
            
            const timestamps = data.bars.map(b => new Date(b.timestamp));
            const opens = data.bars.map(b => b.open);
            const highs = data.bars.map(b => b.high);
            const lows = data.bars.map(b => b.low);
            const closes = data.bars.map(b => b.close);
            
            // Create sequential indices to remove gaps (0, 1, 2, 3...)
            const indices = timestamps.map((_, i) => i);
            
            // Store OHLC data for plugin access
            const ohlcData = { opens, highs, lows, closes };
            
            // Custom candlestick plugin
            const candlestickPlugin = {
                id: 'candlestick',
                afterDatasetsDraw: (chart) => {
                    const {ctx: chartCtx, scales} = chart;
                    const meta = chart.getDatasetMeta(0);
                    
                    chartCtx.save();
                    
                    meta.data.forEach((point, index) => {
                        if (point.skip || index >= ohlcData.opens.length) return;
                        
                        // Use the point's x position directly (it's now an index)
                        const x = point.x;
                        const open = ohlcData.opens[index];
                        const high = ohlcData.highs[index];
                        const low = ohlcData.lows[index];
                        const close = ohlcData.closes[index];
                        
                        const yOpen = scales.y.getPixelForValue(open);
                        const yHigh = scales.y.getPixelForValue(high);
                        const yLow = scales.y.getPixelForValue(low);
                        const yClose = scales.y.getPixelForValue(close);
                        
                        const isUp = close >= open;
                        const color = isUp ? '#00FF66' : '#FF1A1A';
                        const bodyTop = Math.min(yOpen, yClose);
                        const bodyBottom = Math.max(yOpen, yClose);
                        const bodyHeight = Math.abs(yClose - yOpen);
                        const bodyWidth = 8;
                        
                        // Draw wick (high-low line)
                        chartCtx.strokeStyle = color;
                        chartCtx.lineWidth = 1;
                        chartCtx.beginPath();
                        chartCtx.moveTo(x, yHigh);
                        chartCtx.lineTo(x, yLow);
                        chartCtx.stroke();
                        
                        // Draw body (open-close rectangle)
                        chartCtx.fillStyle = color;
                        chartCtx.fillRect(x - bodyWidth / 2, bodyTop, bodyWidth, Math.max(bodyHeight, 1));
                        chartCtx.strokeStyle = color;
                        chartCtx.strokeRect(x - bodyWidth / 2, bodyTop, bodyWidth, Math.max(bodyHeight, 1));
                    });
                    
                    chartCtx.restore();
                }
            };
            
            // Price chart with candlesticks (using invisible line chart as base)
            priceChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: timestamps.map(ts => {
                        const d = new Date(ts);
                        return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
                    }),
                    datasets: [{
                        label: 'Price',
                        data: closes.map((c, i) => ({x: i, y: c})),
                        borderColor: 'transparent',
                        backgroundColor: 'transparent',
                        pointRadius: 0,
                        borderWidth: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false
                        },
                        tooltip: {
                            mode: 'index',
                            intersect: false,
                            callbacks: {
                                label: function(context) {
                                    const bar = data.bars[context.dataIndex];
                                    if (bar) {
                                        return [
                                            `O: ${bar.open.toFixed(2)}`,
                                            `H: ${bar.high.toFixed(2)}`,
                                            `L: ${bar.low.toFixed(2)}`,
                                            `C: ${bar.close.toFixed(2)}`
                                        ];
                                    }
                                    return [];
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            type: 'linear',
                            position: 'bottom',
                            ticks: { 
                                color: '#fff',
                                stepSize: 1,
                                callback: function(value, index) {
                                    if (index % Math.ceil(timestamps.length / 10) === 0 || index === timestamps.length - 1) {
                                        const d = new Date(timestamps[Math.round(value)]);
                                        return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
                                    }
                                    return '';
                                }
                            },
                            grid: { color: '#444' },
                            min: 0,
                            max: timestamps.length - 1
                        },
                        y: {
                            ticks: { color: '#fff' },
                            grid: { color: '#444' },
                            position: 'left'
                        }
                    }
                },
                plugins: [candlestickPlugin]
            });
        }
        
        function createVolumeChart(data) {
            const ctx = document.getElementById('volumeChart').getContext('2d');
            
            // Ensure we have all bars with timestamps
            const timestamps = data.bars.map(b => new Date(b.timestamp));
            const volumes = data.bars.map(b => (b.volume !== null && b.volume !== undefined) ? b.volume : 0);
            const closes = data.bars.map(b => b.close);
            const opens = data.bars.map(b => b.open);
            
            // Create volume data points using sequential indices (0, 1, 2, ...)
            // This ensures proper alignment with price chart
            const volumeData = volumes.map((vol, i) => ({
                x: i,
                y: vol
            }));
            
            // Create colors array for each bar
            const barColors = volumes.map((v, i) => 
                closes[i] >= opens[i] ? 'rgba(0, 255, 102, 0.6)' : 'rgba(255, 26, 26, 0.6)'
            );
            const borderColors = volumes.map((v, i) => 
                closes[i] >= opens[i] ? '#00FF66' : '#FF1A1A'
            );
            
            // Use bar chart with proper linear scale configuration
            volumeChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    datasets: [{
                        label: 'Volume',
                        data: volumeData,
                        backgroundColor: barColors,
                        borderColor: borderColors,
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false
                        },
                        tooltip: {
                            mode: 'index',
                            intersect: false,
                            callbacks: {
                                label: function(context) {
                                    const dataIndex = context.dataIndex;
                                    const volume = volumes[dataIndex] || 0;
                                    return 'Volume: ' + volume.toLocaleString();
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            type: 'linear',
                            position: 'bottom',
                            ticks: { 
                                color: '#fff', 
                                display: false,
                                stepSize: 1
                            },
                            grid: { 
                                color: '#444', 
                                display: false 
                            },
                            min: -0.5,
                            max: timestamps.length > 0 ? timestamps.length - 0.5 : 0
                        },
                        y: {
                            ticks: { color: '#fff' },
                            grid: { color: '#444' },
                            position: 'left',
                            beginAtZero: true
                        }
                    }
                }
            });
        }
        
        function createIndicatorChart(data) {
            const ctx = document.getElementById('indicatorChart').getContext('2d');
            
            // Use bar timestamps as the base (to ensure alignment with price/volume charts)
            const barTimestamps = data.bars.map(b => new Date(b.timestamp));
            
            // Create a map of indicator data by timestamp (normalized to seconds for matching)
            const indicatorMap = new Map();
            data.indicator_data.forEach(ind => {
                const ts = new Date(ind.timestamp);
                // Normalize to seconds (remove milliseconds) for matching
                const key = Math.floor(ts.getTime() / 1000) * 1000;
                indicatorMap.set(key, ind);
            });
            
            // Match indicator data to bar timestamps
            const positive = [];
            const negative = [];
            const timestamps = [];
            
            barTimestamps.forEach((barTs, i) => {
                timestamps.push(barTs);
                // For hourly bars, normalize to hour level (remove minutes/seconds)
                // For other timeframes, normalize to seconds
                const barDate = new Date(barTs);
                // Normalize to hour level for matching (remove minutes, seconds, milliseconds)
                const barKeyHourly = new Date(barDate.getFullYear(), barDate.getMonth(), barDate.getDate(), barDate.getHours(), 0, 0, 0).getTime();
                // Also try exact match (normalized to seconds)
                const barKeyExact = Math.floor(barTs.getTime() / 1000) * 1000;
                
                let indicator = indicatorMap.get(barKeyExact) || indicatorMap.get(barKeyHourly);
                
                // If no match, try to find closest within 1 hour
                if (!indicator) {
                    let closestKey = null;
                    let minDiff = Infinity;
                    indicatorMap.forEach((ind, key) => {
                        const diff = Math.abs(key - barKeyExact);
                        if (diff < minDiff && diff < 3600000) { // Within 1 hour
                            minDiff = diff;
                            closestKey = key;
                        }
                    });
                    if (closestKey !== null) {
                        indicator = indicatorMap.get(closestKey);
                    }
                }
                
                if (indicator) {
                    positive.push(indicator.positive_count);
                    negative.push(-indicator.negative_count);
                } else {
                    // No indicator data for this bar - use previous value or zero
                    if (i > 0) {
                        positive.push(positive[i - 1] || 0);
                        negative.push(negative[i - 1] || 0);
                    } else {
                        positive.push(0);
                        negative.push(0);
                    }
                }
            });
            
            // Create a map of timestamp to index for signal positioning
            const timestampToIndex = new Map();
            timestamps.forEach((ts, i) => {
                timestampToIndex.set(ts.getTime(), i);
            });
            
            const buySignals = data.signals.filter(s => s.side === 'BUY');
            const sellSignals = data.signals.filter(s => s.side === 'SELL');
            
            // Calculate indicator range for signal offset
            const maxPositive = Math.max(...positive, 0);
            const minNegative = Math.min(...negative, 0);
            const indicatorRange = maxPositive - minNegative;
            const buyOffset = indicatorRange * 0.05;  // 5% above positive bars
            const sellOffset = Math.abs(indicatorRange * 0.05);  // 5% below negative bars
            
            // Helper function to find closest indicator value for a signal timestamp
            function findClosestIndicator(signalTime) {
                let closestIdx = 0;
                let minDiff = Math.abs(timestamps[0].getTime() - signalTime);
                for (let i = 1; i < timestamps.length; i++) {
                    const diff = Math.abs(timestamps[i].getTime() - signalTime);
                    if (diff < minDiff) {
                        minDiff = diff;
                        closestIdx = i;
                    }
                }
                return { pos: positive[closestIdx], neg: negative[closestIdx] };
            }
            
            indicatorChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: timestamps.map(ts => {
                        const d = new Date(ts);
                        return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
                    }),
                    datasets: [{
                        label: 'Positive',
                        data: positive.map((val, i) => ({x: i, y: val})),
                        backgroundColor: '#00FF66',
                        borderColor: '#00FF66'
                    }, {
                        label: 'Negative',
                        data: negative.map((val, i) => ({x: i, y: val})),
                        backgroundColor: '#FF1A1A',
                        borderColor: '#FF1A1A'
                    }, {
                        type: 'line',
                        label: 'BUY',
                        data: buySignals.map(s => {
                            const signalTime = new Date(s.timestamp).getTime();
                            const indicator = findClosestIndicator(signalTime);
                            // Position buy signal above positive bars
                            const yValue = indicator.pos + buyOffset;
                            // Find index for this signal timestamp
                            let signalIndex = 0;
                            let minDiff = Infinity;
                            timestamps.forEach((ts, i) => {
                                const diff = Math.abs(ts.getTime() - signalTime);
                                if (diff < minDiff) {
                                    minDiff = diff;
                                    signalIndex = i;
                                }
                            });
                            return {
                                x: signalIndex,
                                y: yValue
                            };
                        }),
                        xAxisID: 'x',
                        pointRadius: 8,
                        pointStyle: 'triangle',
                        pointBackgroundColor: '#00FF66',
                        pointBorderColor: '#ffffff',
                        pointBorderWidth: 2,
                        showLine: false,
                        order: 1
                    }, {
                        type: 'line',
                        label: 'SELL',
                        data: sellSignals.map(s => {
                            const signalTime = new Date(s.timestamp).getTime();
                            const indicator = findClosestIndicator(signalTime);
                            // Position sell signal below negative bars
                            const yValue = indicator.neg - sellOffset;
                            // Find index for this signal timestamp
                            let signalIndex = 0;
                            let minDiff = Infinity;
                            timestamps.forEach((ts, i) => {
                                const diff = Math.abs(ts.getTime() - signalTime);
                                if (diff < minDiff) {
                                    minDiff = diff;
                                    signalIndex = i;
                                }
                            });
                            return {
                                x: signalIndex,
                                y: yValue
                            };
                        }),
                        xAxisID: 'x',
                        pointRadius: 8,
                        pointStyle: 'triangle',
                        rotation: 180,
                        pointBackgroundColor: '#FF1A1A',
                        pointBorderColor: '#ffffff',
                        pointBorderWidth: 2,
                        showLine: false,
                        order: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: true,
                            labels: { color: '#fff' }
                        },
                        tooltip: {
                            mode: 'index',
                            intersect: false,
                            callbacks: {
                                label: function(context) {
                                    const datasetIndex = context.datasetIndex;
                                    const dataIndex = context.dataIndex;
                                    
                                    // For BUY/SELL signals, show price and trend score
                                    if (datasetIndex === 2 || datasetIndex === 3) {
                                        const signal = (datasetIndex === 2 ? buySignals : sellSignals)[dataIndex];
                                        if (signal) {
                                            return context.dataset.label + ': $' + signal.price.toFixed(2) + 
                                                   ' (Score: ' + signal.trend_score + ')';
                                        }
                                    }
                                    
                                    return context.dataset.label + ': ' + context.parsed.y.toFixed(2);
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            type: 'linear',
                            position: 'bottom',
                            ticks: { 
                                color: '#fff',
                                stepSize: 1,
                                callback: function(value, index) {
                                    const idx = Math.round(value);
                                    if (idx >= 0 && idx < timestamps.length) {
                                        if (index % Math.ceil(timestamps.length / 10) === 0 || idx === timestamps.length - 1) {
                                            const d = timestamps[idx];
                                            return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
                                        }
                                    }
                                    return '';
                                }
                            },
                            grid: { color: '#444' },
                            min: 0,
                            max: Math.max(timestamps.length - 1, 0)
                        },
                        y: {
                            ticks: { color: '#fff' },
                            grid: { color: '#444', zeroLineColor: '#fff' }
                        }
                    }
                }
            });
        }
        
        function createEquityChart(data) {
            if (!data.equity_curve || Object.keys(data.equity_curve).length === 0) {
                return;
            }
            
            const ctx = document.getElementById('equityChart').getContext('2d');
            
            // Get all timestamps from bars to map equity curve to indices
            const allTimestamps = data.bars.map(b => new Date(b.timestamp));
            const equityEntries = Object.entries(data.equity_curve)
                .map(([timestamp, value]) => ({
                    timestamp: new Date(timestamp),
                    value: value
                }))
                .sort((a, b) => a.timestamp - b.timestamp);
            
            // Map equity data to indices based on bar timestamps
            const equityData = equityEntries.map(entry => {
                // Find closest bar index
                let closestIdx = 0;
                let minDiff = Infinity;
                allTimestamps.forEach((ts, i) => {
                    const diff = Math.abs(ts.getTime() - entry.timestamp.getTime());
                    if (diff < minDiff) {
                        minDiff = diff;
                        closestIdx = i;
                    }
                });
                return {
                    x: closestIdx,
                    y: entry.value
                };
            });
            
            equityChart = new Chart(ctx, {
                type: 'line',
                data: {
                    datasets: [{
                        label: 'Equity',
                        data: equityData,
                        borderColor: '#FFD700',
                        backgroundColor: 'rgba(255, 215, 0, 0.1)',
                        fill: true,
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: true,
                            labels: { color: '#fff' }
                        },
                        tooltip: { mode: 'index', intersect: false }
                    },
                    scales: {
                        x: {
                            type: 'linear',
                            position: 'bottom',
                            ticks: { 
                                color: '#fff',
                                stepSize: 1,
                                callback: function(value, index) {
                                    if (allTimestamps.length > 0) {
                                        const idx = Math.round(value);
                                        if (idx >= 0 && idx < allTimestamps.length) {
                                            if (index % Math.ceil(allTimestamps.length / 10) === 0 || idx === allTimestamps.length - 1) {
                                                const d = allTimestamps[idx];
                                                return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
                                            }
                                        }
                                    }
                                    return '';
                                }
                            },
                            grid: { color: '#444' },
                            min: 0,
                            max: allTimestamps.length > 0 ? allTimestamps.length - 1 : 0
                        },
                        y: {
                            ticks: { color: '#fff' },
                            grid: { color: '#444' }
                        }
                    }
                }
            });
        }
        
        // Load data on page load
        loadData();
        
        // Refresh data every 5 seconds (optional)
        // setInterval(loadData, 5000);
    </script>
</body>
</html>
        """
    
    def set_data(
        self,
        bars: List[Bar],
        signals: List[TradeSignal],
        indicator_data: List[IndicatorData],
        equity_curve: Dict[datetime, float],
        metrics: Optional[Dict] = None,
        symbol: str = "UNKNOWN"
    ):
        """Set the data to display."""
        # Convert data to JSON-serializable format
        self.data = {
            "symbol": symbol,
            "bars": [
                {
                    "timestamp": b.timestamp.isoformat(),
                    "open": b.open,
                    "high": b.high,
                    "low": b.low,
                    "close": b.close,
                    "volume": b.volume
                }
                for b in bars
            ],
            "signals": [
                {
                    "timestamp": s.timestamp.isoformat(),
                    "price": s.price,
                    "side": s.side,
                    "trend_score": s.trend_score,
                    "di_plus": s.di_plus,
                    "di_minus": s.di_minus
                }
                for s in signals
            ],
            "indicator_data": [
                {
                    "timestamp": ind.timestamp.isoformat(),
                    "positive_count": ind.positive_count,
                    "negative_count": ind.negative_count,
                    "trend_score": ind.trend_score,
                    "di_plus": ind.di_plus,
                    "di_minus": ind.di_minus
                }
                for ind in indicator_data
            ],
            "equity_curve": {
                k.isoformat(): v for k, v in equity_curve.items()
            },
            "metrics": metrics or {}
        }
    
    def start_server(self, open_browser: bool = True):
        """Start the Flask server in a separate thread."""
        def run_server():
            self.app.run(port=self.port, debug=False, use_reloader=False)
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        
        # Wait for server to start
        time.sleep(1)
        
        if open_browser:
            url = f"http://localhost:{self.port}"
            print(f"\n=== Chart Server Started ===")
            print(f"Opening browser at: {url}")
            webbrowser.open(url)
            print(f"Server running. Close this window or press Ctrl+C to stop.")
    
    def stop_server(self):
        """Stop the server (not easily done with Flask, but we can note it)."""
        print("Note: Server will stop when the main process exits.")

