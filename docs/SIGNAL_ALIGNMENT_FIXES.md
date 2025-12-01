# Signal Timestamp Alignment Fixes

## Problem
Buy/sell signals were appearing one day late on charts because signal timestamps weren't being properly aligned with bar timestamps. This happened because:
1. Signal timestamps might have slight differences (microseconds, timezone issues)
2. Charts were using signal timestamps directly without matching to bar timestamps
3. Different visualization files used inconsistent alignment logic

## Solution
Implemented consistent timestamp matching logic across all visualization files:

### 1. **Normalize Timestamps**
   - Remove microseconds for better matching
   - Handle timezone-aware vs naive datetime objects

### 2. **Two-Step Matching**
   - First: Try exact match (after normalization)
   - Second: If no exact match, find closest by time difference

### 3. **Consistent Implementation**
   All visualization files now use the same pattern:
   - `local_chart.py` - Matplotlib charts
   - `plotly_chart.py` - Plotly charts
   - `chart.py` - Legacy matplotlib charts
   - `web_chart.py` - Already had matching logic (JavaScript)

## Files Fixed

### `core/visualization/local_chart.py`
- **`_plot_signals()`**: Added `find_closest_bar_index()` helper
  - Normalizes timestamps before matching
  - Tries exact match first, then closest match
- **`_plot_indicator()`**: Added same helper for indicator chart signals

### `core/visualization/plotly_chart.py`
- **`_add_price_chart()`**: Added `match_signal_to_bar()` helper
  - Uses pandas Timestamp for robust matching
  - Matches signals to bar timestamps before plotting
- **`_add_indicator_chart()`**: Added `match_signal_to_indicator()` helper
  - Matches signals to indicator timestamps
  - Ensures signals align with indicator bars

### `core/visualization/chart.py`
- **`_plot_signals()`**: Added `find_closest_bar_timestamp()` helper
  - Normalizes timestamps
  - Matches signals to bar timestamps

## Alignment Logic

```python
def normalize_timestamp(ts: datetime) -> datetime:
    """Normalize timestamp to remove microseconds for better matching."""
    return ts.replace(microsecond=0)

def find_closest_bar_index(signal_ts: datetime) -> int:
    """Find the index of the bar with the closest timestamp to the signal."""
    signal_ts_norm = normalize_timestamp(signal_ts)
    # First try exact match
    for i, bar_ts in enumerate(timestamps):
        if normalize_timestamp(bar_ts) == signal_ts_norm:
            return i
    # If no exact match, find closest by time difference
    return min(
        range(len(timestamps)),
        key=lambda i: abs((normalize_timestamp(timestamps[i]) - signal_ts_norm).total_seconds())
    )
```

## Benefits

1. **Consistent Behavior**: All charts now align signals the same way
2. **Accurate Positioning**: Signals appear on the correct bar/day
3. **Robust Matching**: Handles timestamp differences gracefully
4. **Better UX**: Users see signals exactly where they should be

## Testing

To verify the fix:
1. Run a backtest with known buy/sell signals
2. Check that signals appear on the correct day in the chart
3. Verify signals align with the bar that triggered them
4. Test with different timeframes (15m, 1H, 1D) to ensure it works for all

## Notes

- Signal timestamps come from `bar_timestamp` in the strategy (already fixed)
- Charts now match these timestamps to bar timestamps for display
- This ensures visual alignment even if there are small timestamp differences

