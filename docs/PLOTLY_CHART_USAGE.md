# Plotly Chart Usage Guide

## Overview

The new Plotly-based charting system provides interactive, professional-grade charts for trading data visualization. It replaces the previous Chart.js/Flask and matplotlib implementations with a unified, modern solution.

## Features

- **Interactive Charts**: Zoom, pan, hover tooltips
- **Multiple Themes**: Moomoo, TradingView, Dark, Light
- **Gap Removal**: Automatically removes weekends and holidays
- **Multi-Panel**: Price, Volume, Indicator, Equity Curve
- **Trading Signals**: Buy/Sell markers on price and indicator charts
- **Export**: HTML, PNG, SVG formats

## Basic Usage

### Standalone Chart

```python
from core.visualization import PlotlyChartVisualizer
from core.models.bar import Bar
from datetime import datetime

# Create visualizer
visualizer = PlotlyChartVisualizer(theme="tradingview")

# Build chart
visualizer.build_chart(
    bars=bars,  # List[Bar]
    signals=signals,  # Optional[List[TradeSignal]]
    indicator_data=indicator_data,  # Optional[List[IndicatorData]]
    equity_curve=equity_curve,  # Optional[Dict[datetime, float]]
    metrics=metrics,  # Optional[Dict]
    symbol="AAPL",
    show_equity=True,
)

# Display in browser
visualizer.show()

# Or export to HTML
visualizer.to_html("chart.html")

# Or export to image
visualizer.to_image("chart.png", format="png")
```

### FastAPI Web Server

```python
from core.visualization import get_fastapi_app, register_chart
import uvicorn

# Create FastAPI app
app = get_fastapi_app()

# Register chart data
register_chart(
    chart_id="backtest_001",
    bars=bars,
    signals=signals,
    indicator_data=indicator_data,
    equity_curve=equity_curve,
    metrics=metrics,
    symbol="AAPL",
)

# Run server
uvicorn.run(app, host="0.0.0.0", port=8000)

# Access chart at: http://localhost:8000/chart?chart_id=backtest_001&theme=tradingview
```

## Themes

Available themes:
- `"moomoo"` - Moomoo-style dark theme
- `"tradingview"` - TradingView-style dark theme (default)
- `"dark"` - Generic dark theme
- `"light"` - Light theme

```python
visualizer = PlotlyChartVisualizer(theme="moomoo")
```

## Integration with Backtest

Update your backtest script:

```python
from core.visualization import PlotlyChartVisualizer

# After backtest completes
visualizer = PlotlyChartVisualizer(theme="tradingview")
visualizer.build_chart(
    bars=bars,
    signals=strategy.get_signals(),
    indicator_data=strategy.get_indicator_history(),
    equity_curve=result.equity_curve,
    metrics={
        "total_return": result.metrics.total_return,
        "cagr": result.metrics.cagr,
        "sharpe": result.metrics.sharpe,
        "max_drawdown": result.metrics.max_drawdown,
    },
    symbol=symbol,
)
visualizer.show()  # Opens in browser
```

## Chart Components

### Price Chart (Row 1)
- Candlesticks (green/red based on theme)
- Buy/Sell signal markers
- Interactive zoom and pan

### Volume Chart (Row 2)
- Volume bars colored by price direction
- Aligned with price chart

### Indicator Chart (Row 3)
- Revised MP2.0 positive/negative counts
- Buy/Sell signals on indicator
- Zero line reference

### Equity Curve (Row 4, optional)
- Portfolio equity over time
- Filled area chart

## Advanced Usage

### Custom Figure Size

```python
visualizer = PlotlyChartVisualizer(
    theme="tradingview",
    figsize=(1920, 1080)  # Width, height in pixels
)
```

### Export Options

```python
# HTML (interactive)
html = visualizer.to_html("chart.html")

# PNG (static image)
visualizer.to_image("chart.png", format="png", width=1920, height=1080)

# SVG (vector)
visualizer.to_image("chart.svg", format="svg")
```

### Accessing the Figure

```python
fig = visualizer.build_chart(...)
# Modify figure directly if needed
fig.update_layout(title="Custom Title")
fig.show()
```

## Migration from Old Charting

### From WebChartServer (Flask)

**Old:**
```python
from core.visualization import get_web_chart_server
server = get_web_chart_server(port=5000)
server.set_data(bars, signals, indicator_data, equity_curve, metrics, symbol)
server.start_server(open_browser=True)
```

**New:**
```python
from core.visualization import PlotlyChartVisualizer
visualizer = PlotlyChartVisualizer(theme="tradingview")
visualizer.build_chart(bars, signals, indicator_data, equity_curve, metrics, symbol)
visualizer.show()
```

### From LocalChartVisualizer (Matplotlib)

**Old:**
```python
from core.visualization import LocalChartVisualizer
chart = LocalChartVisualizer()
chart.set_data(bars, signals, indicator_data, equity_curve, metrics, symbol)
chart.show()
```

**New:**
```python
from core.visualization import PlotlyChartVisualizer
visualizer = PlotlyChartVisualizer(theme="tradingview")
visualizer.build_chart(bars, signals, indicator_data, equity_curve, metrics, symbol)
visualizer.show()
```

## Benefits

1. **Better Performance**: Plotly is optimized for large datasets
2. **Interactivity**: Native zoom, pan, hover without custom code
3. **Professional Look**: Matches TradingView/Moomoo aesthetics
4. **Gap Handling**: Automatic weekend/holiday gap removal
5. **Export Options**: Multiple formats (HTML, PNG, SVG)
6. **Theme Support**: Easy customization
7. **Unified API**: One system for all charting needs

