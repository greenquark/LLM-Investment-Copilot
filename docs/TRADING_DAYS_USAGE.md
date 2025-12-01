# Trading Days Usage Guide

## Overview

The codebase uses trading days (excluding weekends and holidays) instead of calendar days for daily bar operations. This ensures accurate date calculations and avoids requesting data for non-trading days.

## Core Functions

### Primary Import Location

All trading day functions are exported from `core.data`:

```python
from core.data import get_trading_days, is_trading_day, get_trading_days_set
```

### Available Functions

1. **`get_trading_days(start: date, end: date, exchange: str = "NYSE") -> List[date]`**
   - Returns a list of trading days between start and end (inclusive)
   - Primary function for getting trading days in a date range

2. **`is_trading_day(day: date, exchange: str = "NYSE") -> bool`**
   - Checks if a specific date is a trading day
   - Useful for filtering individual dates

3. **`get_trading_days_set(start: date, end: date, exchange: str = "NYSE") -> Set[date]`**
   - Returns a set of trading days (useful for membership checks)
   - More efficient than list for large-scale lookups

### Convenience Utilities

For convenience, helper functions are available in `core.utils.trading_days`:

```python
from core.utils.trading_days import (
    get_trading_days_lookback,
    get_trading_days_lookback_datetime,
)
```

- **`get_trading_days_lookback(reference_date, num_trading_days)`**: Get the date that is N trading days before a reference date
- **`get_trading_days_lookback_datetime(reference_datetime, num_trading_days)`**: Same as above but works with datetime objects

## Where Trading Days Are Used

### 1. Data Caching (`core/data/cached_engine.py`)
- **Coverage checks**: Determines which dates are missing from cache (only trading days for daily bars)
- **Date range logging**: Shows actual trading day ranges, not calendar ranges
- **API fetching**: Splits multi-day ranges into individual trading days to avoid 404s for holidays
- **"No data" tracking**: Marks trading days (not calendar days) as having no data

### 2. Cache Coverage (`core/data/cache.py`)
- **Requested units generation**: Only includes trading days when checking coverage for daily bars
- **Missing units filtering**: Filters out non-trading days from missing units

### 3. Data Adapters
- **MarketData.app**: Uses `date` parameter for single trading day requests
- **YFinance**: Filters data to trading days when appropriate

## Import Pattern

### Recommended Import Pattern

```python
# Primary import (preferred)
from core.data import get_trading_days, is_trading_day

# Or for convenience utilities
from core.utils.trading_days import get_trading_days_lookback
```

### Fallback Pattern

If `core.data` export is not available, modules can fall back to direct import:

```python
try:
    from core.data import get_trading_days
except ImportError:
    from core.data.trading_calendar import get_trading_days
```

This pattern is already implemented in:
- `core/data/cached_engine.py`
- `core/data/cache.py`

## When to Use Trading Days

### ✅ Use Trading Days For:
- **Daily bar date ranges**: When requesting or checking coverage for daily bars
- **Date range logging**: When showing which dates are being requested/fetched
- **Cache coverage checks**: When determining which dates are missing
- **API request planning**: When splitting ranges into individual days
- **Lookback calculations**: When calculating "N trading days ago" for daily bars

### ❌ Don't Use Trading Days For:
- **Intraday timeframes**: 15m, 30m, 1H, 2H, 4H use calendar days for lookback
- **Weekly/Monthly bars**: Use calendar days (these are already trading-day aligned)
- **Time-based calculations**: Hours, minutes, seconds use calendar time

## Examples

### Example 1: Get Trading Days in Range
```python
from core.data import get_trading_days
from datetime import date

# Get all trading days in June 2025
trading_days = get_trading_days(date(2025, 6, 1), date(2025, 6, 30))
print(f"Found {len(trading_days)} trading days")  # ~22 trading days, not 30 calendar days
```

### Example 2: Check if Date is Trading Day
```python
from core.data import is_trading_day
from datetime import date

# Check if Juneteenth 2025 is a trading day
juneteenth = date(2025, 6, 19)
if not is_trading_day(juneteenth):
    print("Juneteenth is a holiday - no trading")
```

### Example 3: Calculate Trading Days Lookback
```python
from core.utils.trading_days import get_trading_days_lookback_datetime
from datetime import datetime

# Get datetime that is 30 trading days before today
now = datetime.now()
lookback_start = get_trading_days_lookback_datetime(now, 30)
# This ensures we get exactly 30 trading days, not 30 calendar days
```

## Implementation Details

### Trading Calendar Library

The system uses the `exchange-calendars` library (specifically `XNYSExchangeCalendar` for NYSE) to determine trading days. This library:
- Knows about all US market holidays
- Handles early market closes
- Accounts for weekends
- Supports multiple exchanges

### Fallback Behavior

If `exchange-calendars` is not available:
- Functions fall back to calendar days
- Log messages indicate when fallback is used
- System continues to function but may request data for non-trading days

### Caching

Within a single `get_bars()` call in `CachedDataEngine`, trading day lookups are cached to avoid repeated API calls to the calendar library.

## Best Practices

1. **Always use `get_trading_days()` for daily bars**: Don't use `timedelta(days=N)` for daily bar date ranges
2. **Import from `core.data`**: Use the centralized export point
3. **Check availability**: Use `_TRADING_CALENDAR_AVAILABLE` flag if needed
4. **Log appropriately**: Show whether trading days or calendar days are being used
5. **Handle fallbacks**: Gracefully handle cases where trading calendar is unavailable

## Current Status

✅ **Implemented:**
- Trading days used in `cached_engine.py` for coverage checks and API fetching
- Trading days used in `cache.py` for coverage calculations
- Functions properly exported from `core.data`
- Convenience utilities in `core.utils.trading_days`

⚠️ **Areas for Future Enhancement:**
- Lookback periods in strategies could use trading days for daily bars (currently use calendar days)
- Backtest engine lookback could use trading days for daily bars (currently use calendar days)

Note: Lookback periods using calendar days may be intentional (wanting "30 calendar days of lookback" rather than "30 trading days"). The data engine will fetch the appropriate bars regardless.

