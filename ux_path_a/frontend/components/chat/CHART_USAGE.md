# Chart Rendering Guide

The chat interface supports interactive chart rendering similar to ChatGPT. Charts are rendered using Plotly.js and can be embedded in LLM responses.

## Chart Format

Charts are specified using markdown code blocks with special language tags:

### Option 1: Explicit chart language
````markdown
```chart
{
  "type": "line",
  "data": [
    {
      "x": ["2024-01-01", "2024-01-02", "2024-01-03"],
      "y": [100, 105, 103],
      "name": "Price"
    }
  ],
  "layout": {
    "title": "Price Chart",
    "xaxis": { "title": "Date" },
    "yaxis": { "title": "Price ($)" }
  }
}
```
````

### Option 2: JSON with chart detection
````markdown
```json
{
  "type": "line",
  "data": [
    {
      "x": ["2024-01-01", "2024-01-02"],
      "y": [100, 105],
      "name": "Price"
    }
  ]
}
```
````

## Supported Chart Types

### 1. Line Chart
```json
{
  "type": "line",
  "data": [
    {
      "x": ["2024-01-01", "2024-01-02", "2024-01-03"],
      "y": [100, 105, 103],
      "name": "Price",
      "line": { "color": "#3B82F6", "width": 2 }
    }
  ]
}
```

### 2. Candlestick Chart
```json
{
  "type": "candlestick",
  "data": [
    {
      "x": ["2024-01-01", "2024-01-02", "2024-01-03"],
      "open": [100, 103, 104],
      "high": [105, 106, 105],
      "low": [99, 102, 103],
      "close": [103, 104, 103],
      "name": "OHLC"
    }
  ]
}
```

### 3. Bar Chart
```json
{
  "type": "bar",
  "data": [
    {
      "x": ["Jan", "Feb", "Mar"],
      "y": [100, 150, 120],
      "name": "Volume"
    }
  ]
}
```

### 4. Area Chart
```json
{
  "type": "area",
  "data": [
    {
      "x": ["2024-01-01", "2024-01-02", "2024-01-03"],
      "y": [100, 105, 103],
      "name": "Price",
      "line": { "color": "#10B981" }
    }
  ]
}
```

### 5. Scatter Chart
```json
{
  "type": "scatter",
  "data": [
    {
      "x": [1, 2, 3, 4, 5],
      "y": [10, 15, 13, 17, 12],
      "mode": "markers",
      "name": "Data Points"
    }
  ]
}
```

## Layout Options

```json
{
  "layout": {
    "title": "Chart Title",
    "xaxis": {
      "title": "X Axis Label",
      "type": "date"  // or "linear", "log"
    },
    "yaxis": {
      "title": "Y Axis Label"
    },
    "height": 400,
    "showlegend": true,
    "hovermode": "x unified"
  }
}
```

## Configuration Options

```json
{
  "config": {
    "displayModeBar": true,
    "responsive": true,
    "displaylogo": false
  }
}
```

## Features

- ✅ Dark mode support
- ✅ Responsive design
- ✅ Interactive zoom and pan
- ✅ Hover tooltips
- ✅ Multiple data series
- ✅ Customizable colors and styling

## Example: Stock Price Chart

````markdown
Here's the price chart for SPY:

```chart
{
  "type": "line",
  "data": [
    {
      "x": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
      "y": [690, 692, 694, 693, 695],
      "name": "SPY Price",
      "line": { "color": "#10B981", "width": 2 }
    }
  ],
  "layout": {
    "title": "SPY Price (Last 5 Days)",
    "xaxis": { "title": "Date" },
    "yaxis": { "title": "Price ($)" },
    "height": 400
  }
}
```
````

## Backend Integration

The LLM can generate charts by including chart JSON in its markdown responses. The system prompt includes instructions for chart rendering, so the LLM will automatically use charts when displaying time series or price data.
