# Data Source Configuration Guide

This document describes how to configure multiple data sources for the trading agent.

## Overview

The trading agent supports multiple data sources for fetching historical and real-time market data. You can configure which data source to use for backtesting and live trading through the environment configuration files.

## Supported Data Sources

### 1. MarketData.app (`marketdata_app`)
- **Type**: Paid API service
- **Requires**: API token
- **Supports**: Historical bars, option chains
- **Timeframes**: 1m, 5m, 15m, 30m, 1H, 2H, 4H, 1D, 1W, 1M
- **Rate Limits**: Depends on subscription tier
- **Cache**: Supported

### 2. Yahoo Finance (`yfinance`)
- **Type**: Free, public API
- **Requires**: No API token
- **Supports**: Historical bars, option chains (daily-level)
- **Timeframes**: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
- **Rate Limits**: Rate limiting may apply
- **Cache**: Supported
- **Limitations**: 
  - 1m data: last 30 days only
  - 5m, 15m, 30m data: last 60 days only
  - Option data is daily-level (snapshot), not minute-level

### 3. Moomoo (`moomoo`)
- **Type**: Broker API (for live trading)
- **Requires**: Moomoo account and connection
- **Supports**: Real-time data, order execution
- **Usage**: Primarily for live trading execution

## Configuration Structure

### Environment Configuration Files

#### `config/env.backtest.yaml` (for backtesting)

```yaml
# Data Sources Configuration
# Configure available data sources and their settings
data_sources:
  marketdata_app:
    enabled: true
    api_token: "YOUR_MARKETDATA_APP_API_TOKEN"
    timeout: 30.0  # Request timeout in seconds
    max_retries: 3  # Maximum retry attempts for failed requests
  
  yfinance:
    enabled: true
    # No API token needed - yfinance is free and doesn't require authentication

# Data Engine Configuration
# Specify which data source to use for historical and real-time data
data:
  historical_source: "marketdata_app"  # Options: "marketdata_app", "yfinance"
  realtime_source: "marketdata_app"    # Options: "marketdata_app", "yfinance" (for backtest, this is not used)
  cache_enabled: true                  # Enable caching for faster reruns
  cache_dir: "data_cache/bars"         # Cache directory

# Backtest Configuration
backtest:
  symbol: "TQQQ"
  start: "2025-07-15T06:30:00"
  end: "2025-11-23T16:00:00"
  initial_cash: 100000
  timeframe: 1D
```

#### `config/env.live.yaml` (for live trading)

```yaml
# Data Sources Configuration
data_sources:
  marketdata_app:
    enabled: true
    api_token: "YOUR_MARKETDATA_APP_API_TOKEN"
    timeout: 30.0
    max_retries: 3
  
  yfinance:
    enabled: true

# Data Engine Configuration
data:
  historical_source: "marketdata_app"  # For historical lookback data
  realtime_source: "moomoo"            # For real-time data (broker API)
  cache_enabled: true
  cache_dir: "data_cache/bars"

# Moomoo Configuration (for live trading execution and real-time data)
moomoo:
  host: "127.0.0.1"
  port: 11111
  account_id: "YOUR_ACCOUNT_ID"

# Live Trading Configuration
live:
  symbol: "TQQQ"
  interval_minutes: 15
  initial_cash: 200000
```

## Switching Data Sources

### For Backtesting

To switch from MarketData.app to Yahoo Finance for backtesting:

1. Edit `config/env.backtest.yaml`
2. Change `data.historical_source` from `"marketdata_app"` to `"yfinance"`
3. Ensure `data_sources.yfinance.enabled` is `true`

Example:
```yaml
data:
  historical_source: "yfinance"  # Changed from "marketdata_app"
```

### For Live Trading

For live trading, you typically want:
- **Historical source**: For lookback data (e.g., calculating indicators)
- **Real-time source**: For current market data (usually your broker API)

Example:
```yaml
data:
  historical_source: "yfinance"  # Free source for historical data
  realtime_source: "moomoo"       # Broker API for real-time data
```

## Backward Compatibility

The system maintains backward compatibility with the old configuration structure:

**Old structure** (still works):
```yaml
marketdata:
  api_token: "YOUR_TOKEN"
```

**New structure** (recommended):
```yaml
data_sources:
  marketdata_app:
    enabled: true
    api_token: "YOUR_TOKEN"
data:
  historical_source: "marketdata_app"
```

If the old structure is detected, the system will automatically migrate it and show a warning message.

## Data Engine Factory

The `core.data.factory` module provides functions to create data engines from configuration:

### `create_data_engine_from_config(env_config, use_for="historical")`

Creates a data engine based on the configuration file.

**Parameters**:
- `env_config`: Full environment config dictionary (from YAML file)
- `use_for`: `"historical"` or `"realtime"` - determines which source to use
- `cache_enabled`: Optional override for cache setting
- `cache_dir`: Optional override for cache directory

**Returns**: `DataEngine` instance (wrapped with `CachedDataEngine` if caching enabled)

**Example**:
```python
import yaml
from core.data.factory import create_data_engine_from_config

with open("config/env.backtest.yaml") as f:
    env = yaml.safe_load(f)

# Create engine for historical data
data_engine = create_data_engine_from_config(
    env_config=env,
    use_for="historical",
)
```

## Adding New Data Sources

To add a new data source:

1. **Create the adapter**: Implement `DataEngine` interface in `core/data/your_source.py`
2. **Update factory**: Add support in `core/data/factory.py`:
   ```python
   elif source_name_lower == "your_source":
       if not _YOUR_SOURCE_AVAILABLE:
           raise ImportError("YourSourceAdapter is not available")
       base_engine = YourSourceAdapter(**config)
   ```
3. **Update config**: Add configuration section in `data_sources`:
   ```yaml
   data_sources:
     your_source:
       enabled: true
       api_key: "YOUR_API_KEY"
       # ... other settings
   ```
4. **Update documentation**: Add to this guide

## Caching

All data sources support transparent caching through `CachedDataEngine`. The cache:
- Stores data in Parquet format for efficient read/write
- Automatically handles cache hits/misses
- Supports partial cache hits (merging cached and fresh data)
- Tracks cache statistics (hit rate, API calls, etc.)

Cache settings are controlled by:
- `data.cache_enabled`: Enable/disable caching
- `data.cache_dir`: Directory for cache files

## Troubleshooting

### "Data source 'X' not found in config"
- Ensure the data source is listed in `data_sources` section
- Check that the source name matches exactly (case-insensitive)

### "Data source 'X' is disabled in config"
- Set `enabled: true` for the data source in config

### "Missing 'api_token' in config"
- For MarketData.app, ensure `data_sources.marketdata_app.api_token` is set
- For yfinance, no API token is needed

### Import errors
- Ensure required dependencies are installed:
  - `yfinance`: `pip install yfinance`
  - `pyarrow`: `pip install pyarrow` (for caching)

## Best Practices

1. **Use caching**: Always enable caching for faster reruns
2. **Choose appropriate source**: 
   - Use `yfinance` for free, basic backtesting
   - Use `marketdata_app` for more reliable, comprehensive data
   - Use broker APIs (`moomoo`) for live trading
3. **Separate historical and real-time**: Use different sources for historical lookback vs. real-time data if needed
4. **Monitor rate limits**: Be aware of API rate limits, especially for free sources
5. **Test configuration**: Verify your data source works before running long backtests

