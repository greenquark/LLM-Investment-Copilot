# Timestamp Normalization Strategy

## Problem

Different data sources return timestamps in different formats and timezones:
- **MarketData.app**: Unix timestamps (UTC-based, but `datetime.fromtimestamp()` interprets as local time)
- **yfinance**: Timezone-aware datetimes in `America/New_York` (ET/EDT)
- **Other sources**: May vary

This inconsistency can cause:
- Incorrect timestamp comparisons
- Confusion when mixing data from different sources
- Cache deduplication failures
- Chart alignment issues

## Solution

**Standard Format**: All `Bar` timestamps are stored as **timezone-naive Eastern Time (ET)** datetime objects.

Since we focus on US markets, using ET is more intuitive than UTC.

**Timestamp Convention**:
- **Complete historical bars** (daily, weekly, monthly): Use `00:00:00 ET` (midnight)
  - This represents "N/A" since the bar covers the entire period
  - It's as good as market close for historical data
- **Incomplete/ongoing bars** (real-time): Use actual timestamp
- **Intraday bars**: Use actual timestamp

### Normalization Rules

1. **All timestamps are normalized to timezone-naive Eastern Time (ET)** before being stored in `Bar` objects
2. **Unix timestamps** (MarketData.app) are interpreted as UTC and converted to ET
3. **Timezone-aware timestamps** (yfinance) are converted to ET, then made naive
4. **Naive timestamps** are assumed to be ET unless a source timezone is specified
5. **Complete historical bars** (daily, weekly, monthly) are normalized to `00:00:00 ET` (midnight)
   - This represents "N/A" since the bar covers the entire period
   - It's as good as market close for historical data
6. **Intraday bars** keep their actual timestamps
7. **Incomplete/ongoing bars** (real-time) keep their actual timestamps

### Implementation

The normalization is handled by `core.utils.timestamp`:

```python
from core.utils.timestamp import normalize_timestamp, normalize_unix_timestamp

# For Unix timestamps (MarketData.app)
ts = normalize_unix_timestamp(unix_timestamp)

# For timezone-aware timestamps (yfinance)
ts = normalize_timestamp(timezone_aware_datetime)

# For naive timestamps with known source timezone
ts = normalize_timestamp(naive_datetime, source_timezone='America/New_York')
```

## Data Source Behavior

### MarketData.app
- **Returns**: Unix timestamps (seconds since epoch, UTC-based)
- **Normalization**: `normalize_unix_timestamp()` converts UTC → ET → naive ET
- **Complete historical bars**: Normalized to `00:00:00 ET` (midnight)
- **Result**: Timestamps represent Eastern Time (e.g., `2025-06-15 00:00:00` for daily bars)

### yfinance
- **Returns**: Timezone-aware datetimes in `America/New_York` (ET/EDT)
- **Normalization**: `normalize_timestamp()` converts ET → naive ET
- **Complete historical bars**: Normalized to `00:00:00 ET` (midnight) for consistency
- **Result**: Timestamps represent Eastern Time (e.g., `2025-06-15 00:00:00` for daily bars)

### Moomoo/Futu
- **Returns**: Varies (check implementation)
- **Normalization**: Should use `normalize_timestamp()` with appropriate source timezone

## Benefits

1. **Consistency**: All `Bar` objects have timestamps in the same format
2. **Comparability**: Timestamps from different sources can be directly compared
3. **Cache Safety**: Cache deduplication works correctly across data sources
4. **Chart Alignment**: Charts can align data from multiple sources without timezone issues

## Display

For display purposes, timestamps can be converted to ET:

```python
from core.utils.timestamp import to_et_timestamp

# Convert normalized UTC timestamp to ET for display
et_ts = to_et_timestamp(normalized_ts)
```

## Example

```python
# MarketData.app returns Unix timestamp: 1718481600 (daily bar)
# This represents: 2025-06-15 20:00:00 UTC (4 PM ET in EDT)
ts1 = normalize_unix_timestamp(1718481600, timeframe="1D")
# Result: datetime(2025, 6, 15, 0, 0, 0)  # Naive ET (normalized to midnight)

# yfinance returns daily bar: datetime(2025, 6, 15, 0, 0, 0, tzinfo=ET)
# This represents: midnight ET (yfinance convention)
ts2 = normalize_timestamp(et_datetime, timeframe="1D")
# Result: datetime(2025, 6, 15, 0, 0, 0)  # Naive ET (normalized to midnight)

# Both timestamps are now directly comparable
assert ts1 == ts2  # True!

# Intraday bars keep their actual timestamps
ts3 = normalize_timestamp(intraday_datetime, timeframe="15m")
# Result: datetime(2025, 6, 15, 10, 30, 0)  # Keeps actual time
```

## Complete Historical Bar Normalization

**Important**: For complete historical bars (daily, weekly, monthly), timestamps are normalized to `00:00:00 ET` (midnight):
- **Rationale**: These bars represent the entire period, so the timestamp is "N/A" - midnight is as good as market close
- **yfinance**: Returns daily/weekly/monthly bars → normalized to `00:00:00 ET`
- **MarketData.app**: Returns daily/weekly/monthly bars → converted to ET, then normalized to `00:00:00 ET`

**Intraday bars** keep their actual timestamps (e.g., `10:30:00 ET` for a 15-minute bar).

**Incomplete/ongoing bars** (real-time) keep their actual timestamps (handled separately in real-time systems).

## Migration Notes

- Existing cached data may have timestamps in different formats
- Cache will be gradually updated as new data is fetched
- Old cached data will work but may have slight timezone differences until refreshed

