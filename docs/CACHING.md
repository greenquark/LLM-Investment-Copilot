# Data Caching System

## Overview

The trading agent includes a transparent caching layer that sits in front of any data source to improve performance and reduce API calls. The cache stores historical market data in Parquet format for efficient read/write operations.

## Architecture

The caching system uses a **wrapper pattern** that implements the `DataEngine` interface:

```
┌─────────────────┐
│ CachedDataEngine│  ← Transparent caching wrapper
└────────┬────────┘
         │ wraps
         ▼
┌─────────────────┐
│ MarketDataApp   │  ← Any DataEngine implementation
│ MoomooData      │
│ (or other)      │
└─────────────────┘
```

### Key Components

1. **`CachedDataEngine`** - Transparent wrapper that adds caching to any `DataEngine`
2. **`DataCache`** - Manages file-based storage using Parquet format
3. **In-memory cache** - Fast lookup for recently accessed data

## How It Works

### Request Flow

1. **Check cache first**: When `get_bars()` is called, the cache checks if data exists
2. **Coverage analysis**: Determines if the requested range is:
   - **Fully covered** - All data in cache → return immediately (no API call)
   - **Partially covered** - Some data in cache → fetch only missing ranges
   - **Not covered** - No data in cache → fetch entire range
3. **Merge and save**: New data from API is merged with cached data and saved to disk

### Cache Storage

- **Format**: Apache Parquet (columnar, compressed)
- **Location**: `data_cache/bars/{SYMBOL}_{TIMEFRAME}.parquet`
- **Structure**: One file per symbol/timeframe combination
- **In-memory**: Full cache loaded into memory for fast lookups

### Features

- **Transparent**: Data sources don't need to know about caching
- **Efficient**: Only fetches missing data ranges
- **Deduplication**: Automatically handles overlapping date ranges
- **Multi-timeframe**: Supports all timeframes/resolutions:
  - **Minutely**: 1, 3, 5, 15, 30, 45, ... (as numbers or "15m", "30m")
  - **Hourly**: H, 1H, 2H, ... (or "1H", "2H")
  - **Daily**: D, 1D, 2D, ... (or "1D", "2D")
  - **Weekly**: W, 1W, 2W, ... (or "1W", "2W")
  - **Monthly**: M, 1M, 2M, ... (or "1M", "2M")
  - **Yearly**: Y, 1Y, 2Y, ... (or "1Y", "2Y")
- **Atomic writes**: Uses temporary files to prevent corruption

## Usage

### Basic Usage

```python
from core.data.marketdata_app import MarketDataAppAdapter
from core.data.cached_engine import CachedDataEngine

# Create base data engine (no caching)
base_engine = MarketDataAppAdapter(api_token="your_token")

# Wrap with caching layer
cached_engine = CachedDataEngine(
    base_engine,
    cache_dir="data_cache/bars",
    cache_enabled=True
)

# Use as normal - caching is transparent
bars = await cached_engine.get_bars("AAPL", start, end, "D")
```

### Works with Any Data Source

```python
# Works with MarketData.app
marketdata = CachedDataEngine(MarketDataAppAdapter(...))

# Works with Moomoo
moomoo = CachedDataEngine(MoomooDataAdapter(...))

# Works with any DataEngine implementation
custom = CachedDataEngine(YourCustomDataEngine(...))
```

## Cache Statistics

The cache tracks performance metrics:

```python
stats = cached_engine.get_cache_stats()
print(f"Cache hit rate: {stats['cache_hits'] / stats['total_requests'] * 100:.1f}%")
print(f"API calls saved: {stats['cache_hits'] + stats['cache_partial_hits']}")
```

Available statistics:
- `total_requests` - Total number of data requests
- `cache_hits` - Requests fully served from cache
- `cache_partial_hits` - Requests partially served from cache
- `api_calls` - Number of calls to underlying data source
- `total_bars_from_cache` - Total bars retrieved from cache
- `total_bars_from_api` - Total bars retrieved from API

## Data Source Logging

The cache automatically tracks data source for logging:

- `[Cache]` - Data came from cache
- `[MarketData]` - Data came from MarketData.app API
- `[Moomoo]` - Data came from Moomoo API
- (or custom source name)

This appears in backtest logs to show where data was sourced.

## Performance Benefits

1. **Faster backtests**: Cached data loads instantly vs. API latency
2. **Reduced API costs**: Fewer API calls = lower costs
3. **Offline capability**: Can run backtests with cached data
4. **Incremental updates**: Only fetches new data, not entire history

## Cache Management

### Cache Location

By default, cache files are stored in `data_cache/bars/`. This directory is:
- Git-ignored (cache files are not versioned)
- Per-project (each project has its own cache)

### Cache Invalidation

The cache automatically:
- Detects file modifications and reloads if needed
- Merges new data with existing cache
- Handles duplicate timestamps (prefers cached data)

### Manual Cache Management

To clear cache:
```bash
# Delete specific symbol/timeframe
rm data_cache/bars/AAPL_D.parquet

# Delete all cache
rm -rf data_cache/bars/*
```

## Dependencies

The caching system requires:
- `pandas>=2.0.0,<2.3.0` - Data manipulation
- `pyarrow>=14.0.0` - Parquet file format
- `numpy>=1.26.0,<2.0.0` - Numerical operations

If these are not available, caching is automatically disabled with a warning.

## Design Principles

1. **Transparency**: Caching is invisible to data sources
2. **Separation of Concerns**: Cache logic separate from data source logic
3. **Reusability**: Works with any `DataEngine` implementation
4. **Efficiency**: Only fetches what's needed
5. **Reliability**: Atomic writes prevent corruption

## Example: Cache Hit Rate

On a typical backtest rerun:
- **First run**: 0% cache hits (all data from API)
- **Second run**: 100% cache hits (all data from cache)
- **Partial updates**: 80-95% cache hits (only new data from API)

This dramatically speeds up iterative backtesting and strategy development.

