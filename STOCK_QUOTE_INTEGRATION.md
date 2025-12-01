# Stock Quote Integration into Chart

## Overview
Add stock quote (bid/ask/last) display to the existing chart to show current market prices alongside historical candlestick data.

## Implementation Plan

### Phase 1: Create Stock Quote Model

**File:** `core/models/quote.py` (new file)

```python
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class StockQuote:
    """Real-time or delayed stock quote."""
    symbol: str
    bid: float
    ask: float
    last: float
    volume: int
    quote_timestamp: datetime
    # Optional fields
    high_52w: Optional[float] = None
    low_52w: Optional[float] = None
    market_cap: Optional[float] = None
```

### Phase 2: Add Quote Method to Data Engine

**File:** `core/data/base.py`

Add abstract method:
```python
@abstractmethod
async def get_stock_quote(
    self,
    symbol: str,
    as_of: Optional[datetime] = None,
) -> Optional[StockQuote]:
    """
    Get current or historical stock quote.
    
    Args:
        symbol: Stock symbol
        as_of: Optional timestamp for historical quote (None for current)
    
    Returns:
        StockQuote if available, None otherwise
    """
    ...
```

**File:** `core/data/marketdata_app.py`

Implement method:
```python
async def get_stock_quote(
    self,
    symbol: str,
    as_of: Optional[datetime] = None,
) -> Optional[StockQuote]:
    """
    Get stock quote from MarketData.app API.
    
    Endpoint: GET /v1/stocks/quote/{symbol}/
    """
    url = f"{self.BASE_URL}/v1/stocks/quote/{symbol}/"
    params = {"format": "json"}
    if as_of:
        params["as_of"] = as_of.isoformat()
    
    headers = {"Authorization": f"Bearer {self._token}"}
    
    # Retry logic (reuse existing pattern)
    for attempt in range(self._max_retries):
        try:
            resp = await self._client.get(url, params=params, headers=headers)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            data = resp.json()
            if data.get("s") != "ok":
                return None
            
            # Parse response
            return StockQuote(
                symbol=symbol,
                bid=data.get("bid", 0.0),
                ask=data.get("ask", 0.0),
                last=data.get("last", data.get("close", 0.0)),
                volume=data.get("volume", 0),
                quote_timestamp=datetime.fromtimestamp(data.get("timestamp", 0)),
            )
        except (httpx.ReadTimeout, httpx.TimeoutException, httpx.ConnectTimeout) as e:
            if attempt < self._max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            return None
        except Exception:
            return None
    
    return None
```

### Phase 3: Update Chart to Display Quotes

**File:** `core/visualization/local_chart.py`

**Changes:**

1. **Update `set_data()` method:**
```python
def set_data(
    self,
    bars: List[Bar],
    signals: List[TradeSignal],
    indicator_data: List[IndicatorData],
    equity_curve: Dict[datetime, float],
    metrics: Optional[Dict] = None,
    symbol: str = "UNKNOWN",
    stock_quote: Optional[StockQuote] = None,  # NEW
):
    self._data = {
        "bars": bars,
        "signals": signals,
        "indicator_data": indicator_data,
        "equity_curve": equity_curve,
        "metrics": metrics or {},
        "symbol": symbol,
        "stock_quote": stock_quote,  # NEW
    }
```

2. **Add quote display method:**
```python
def _plot_stock_quote(self, ax, quote: StockQuote, indices: List[int], timestamps: List[datetime]):
    """Plot stock quote (bid/ask/last) as horizontal lines at the most recent bar."""
    if not quote or not indices or not timestamps:
        return
    
    # Find the index of the most recent bar (last index)
    last_idx = indices[-1]
    
    # Plot bid line (blue, dashed)
    ax.axhline(
        y=quote.bid,
        xmin=0,
        xmax=1,
        color="#4A9EFF",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=f"Bid: ${quote.bid:.2f}",
    )
    
    # Plot ask line (red, dashed)
    ax.axhline(
        y=quote.ask,
        xmin=0,
        xmax=1,
        color="#FF6B6B",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=f"Ask: ${quote.ask:.2f}",
    )
    
    # Plot last price (green, solid)
    ax.axhline(
        y=quote.last,
        xmin=0,
        xmax=1,
        color="#00FF66",
        linestyle="-",
        linewidth=2,
        alpha=0.8,
        label=f"Last: ${quote.last:.2f}",
    )
    
    # Add marker at the last bar position
    ax.plot(
        last_idx,
        quote.last,
        marker="o",
        markersize=8,
        color="#00FF66",
        markeredgecolor="#ffffff",
        markeredgewidth=1.5,
        zorder=10,
    )
    
    # Add text annotation with quote info
    spread = quote.ask - quote.bid
    spread_pct = (spread / quote.last * 100) if quote.last > 0 else 0
    quote_text = (
        f"Quote: ${quote.last:.2f}\n"
        f"Bid: ${quote.bid:.2f} | Ask: ${quote.ask:.2f}\n"
        f"Spread: ${spread:.2f} ({spread_pct:.2f}%)\n"
        f"Time: {quote.quote_timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    
    # Position text box in upper right
    ax.text(
        0.98,
        0.98,
        quote_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="#1e1e1e", alpha=0.8, edgecolor="#444"),
        color="#ffffff",
    )
```

3. **Update `show()` method to call quote plotting:**
```python
# In show() method, after plotting candlesticks and signals:
if stock_quote:
    self._plot_stock_quote(ax_price, stock_quote, indices, timestamps)
```

**File:** `core/visualization/web_chart.py`

Similar updates for web chart:
- Add `stock_quote` parameter to `set_data()`
- Add quote display in JavaScript (horizontal lines + info box)

### Phase 4: Update Backtest Script to Fetch Quote

**File:** `scripts/run_backtest_mystic_pulse.py`

**Changes:**

1. **Import StockQuote:**
```python
from core.models.quote import StockQuote
```

2. **Fetch quote after backtest:**
```python
# After backtest completes, fetch current quote
try:
    stock_quote = await data_engine.get_stock_quote(symbol)
    if stock_quote:
        logger.log(f"Current quote: Bid=${stock_quote.bid:.2f}, Ask=${stock_quote.ask:.2f}, Last=${stock_quote.last:.2f}")
except Exception as e:
    logger.log(f"Could not fetch stock quote: {e}")
    stock_quote = None
```

3. **Pass quote to chart:**
```python
if use_local_chart:
    chart_visualizer = LocalChartVisualizer()
    chart_visualizer.set_data(
        bars=bars,
        signals=signals,
        indicator_data=filtered_indicator_data,
        equity_curve=result.equity_curve,
        metrics=metrics_dict,
        symbol=symbol,
        stock_quote=stock_quote,  # NEW
    )
    chart_visualizer.show()
else:
    server.set_data(
        bars=bars,
        signals=signals,
        indicator_data=filtered_indicator_data,
        equity_curve=result.equity_curve,
        metrics=metrics_dict,
        symbol=symbol,
        stock_quote=stock_quote,  # NEW
    )
```

### Phase 5: Update Web Chart (if using web visualization)

**File:** `core/visualization/web_chart.py`

Add JavaScript to display quotes:
- Horizontal lines for bid/ask/last
- Info box with quote details
- Marker at last bar position

## Visual Design

### Quote Display Options

**Option A: Horizontal Lines (Recommended)**
- Bid: Blue dashed line
- Ask: Red dashed line  
- Last: Green solid line
- All span full width of chart
- Info box in upper right corner

**Option B: Vertical Marker**
- Vertical line at last bar
- Quote values as markers on that line
- Less intrusive but less visible

**Option C: Separate Panel**
- Small panel above price chart
- Shows quote details in table format
- Takes up more space

**Recommendation:** Option A - horizontal lines are clear and don't clutter the chart.

## Implementation Order

1. ✅ Create `StockQuote` model
2. ✅ Add `get_stock_quote()` to `DataEngine` base class
3. ✅ Implement in `MarketDataAppAdapter`
4. ✅ Update `LocalChartVisualizer` to display quotes
5. ✅ Update backtest script to fetch and pass quotes
6. ✅ Update `WebChartServer` (if using web visualization)
7. ✅ Test with real data

## Testing

1. Test quote fetching with various symbols
2. Test quote display on chart
3. Test with missing quote data (graceful degradation)
4. Test with historical vs current quotes
5. Verify quote alignment with last bar

## Notes

- Quotes are time-sensitive - consider caching strategy
- If quote timestamp doesn't match last bar, show both
- Handle cases where quote is unavailable gracefully
- Consider showing quote only if it's recent (within last hour)

