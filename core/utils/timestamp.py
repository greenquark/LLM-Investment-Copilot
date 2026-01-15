"""
Timestamp normalization utilities.

This module provides functions to normalize timestamps from different data sources
to a consistent format, ensuring the program doesn't get confused by timezone differences.

Standard Format:
- All Bar timestamps are stored as timezone-naive datetime objects
- Timestamps represent Eastern Time (ET/EDT) - US market timezone
- This is more intuitive for US market-focused trading systems

Timestamp Convention:
- Complete historical bars (daily, weekly, monthly): Use 00:00:00 ET (midnight)
  - This represents "N/A" since the bar covers the entire period
  - It's as good as market close for historical data
- Incomplete/ongoing bars (real-time): Use actual timestamp
- Intraday bars: Use actual timestamp

Data Source Behavior:
- MarketData.app: Returns Unix timestamps (UTC-based) → converted to ET
- yfinance: Returns timezone-aware datetimes in America/New_York (ET) → converted to naive ET
- Other sources: May vary → converted to ET

All timestamps are normalized to timezone-naive Eastern Time before being stored in Bar objects.
"""

from __future__ import annotations
from datetime import datetime, timezone, date
from typing import Optional, Union
import pandas as pd

try:
    import pytz
    _PYTZ_AVAILABLE = True
except ImportError:
    _PYTZ_AVAILABLE = False
    pytz = None  # type: ignore


def normalize_timestamp(
    ts: datetime, 
    source_timezone: Optional[str] = None,
    timeframe: Optional[str] = None,
) -> datetime:
    """
    Normalize a timestamp to timezone-naive Eastern Time (ET) format.
    
    This ensures all Bar timestamps are in a consistent format regardless of data source.
    All timestamps are normalized to ET since we focus on US markets.
    
    Timestamp Convention:
    - Complete historical bars (daily, weekly, monthly): Normalized to 00:00:00 ET (midnight)
      - This represents "N/A" since the bar covers the entire period
      - It's as good as market close for historical data
    - Intraday bars: Keep actual timestamp
    - Incomplete/ongoing bars: Keep actual timestamp (handled separately in real-time systems)
    
    Args:
        ts: Timestamp to normalize (can be timezone-aware or naive)
        source_timezone: Optional timezone name (e.g., 'UTC') if ts is naive
                        but represents a time in that timezone. If None and ts is naive,
                        assumes ts is already in ET.
        timeframe: Optional timeframe string (e.g., "1D", "1wk", "1mo") to determine if this is
                   a complete historical bar that should be normalized to 00:00:00 ET.
    
    Returns:
        Timezone-naive datetime representing Eastern Time
        
    Examples:
        >>> # MarketData.app Unix timestamp (UTC) - daily bar
        >>> ts = datetime.fromtimestamp(1718481600, tz=timezone.utc)  # UTC
        >>> normalize_timestamp(ts, timeframe="1D")  # Converts to ET naive, then to 00:00:00 ET
        
        >>> # yfinance timestamp (ET timezone-aware) - daily bar
        >>> ts = datetime(2025, 6, 15, 16, 0, 0, tzinfo=pytz.timezone('America/New_York'))
        >>> normalize_timestamp(ts, timeframe="1D")  # Converts to 00:00:00 ET (midnight)
        
        >>> # Intraday bar - keep actual timestamp
        >>> ts = datetime(2025, 6, 15, 10, 30, 0, tzinfo=pytz.timezone('America/New_York'))
        >>> normalize_timestamp(ts, timeframe="15m")  # Converts to ET naive, keeps 10:30:00
    """
    # For complete historical bars (daily, weekly, monthly), normalize to 00:00:00 ET (midnight)
    # This represents "N/A" since the bar covers the entire period
    # It's as good as market close for historical data
    if timeframe:
        timeframe_upper = timeframe.upper()
        # Check if this is a daily, weekly, or monthly bar
        is_daily = timeframe_upper in ("D", "1D", "DAILY")
        is_weekly = timeframe_upper in ("W", "1W", "WK", "1WK", "WEEKLY", "WEEK")
        is_monthly = timeframe_upper in ("M", "1M", "MO", "1MO", "MONTHLY", "MONTH")
        
        if is_daily or is_weekly or is_monthly:
            # For complete historical bars, normalize to midnight (00:00:00 ET)
            # This represents "N/A" since the bar covers the entire period
            normalized = _to_et_datetime(ts, source_timezone)
            # Set to midnight
            return normalized.replace(hour=0, minute=0, second=0, microsecond=0)
    
    # For intraday bars or incomplete bars, keep the actual timestamp but normalize to ET
    return _to_et_datetime(ts, source_timezone)


def normalize_unix_timestamp(
    unix_ts: float,
    timeframe: Optional[str] = None,
) -> datetime:
    """
    Normalize a Unix timestamp to timezone-naive Eastern Time (ET) format.
    
    Unix timestamps are typically in UTC, so this function converts from UTC to ET.
    
    Args:
        unix_ts: Unix timestamp (seconds since epoch)
        timeframe: Optional timeframe string (e.g., "1D", "1wk", "1mo") to determine if this is
                   a complete historical bar that should be normalized to 00:00:00 ET.
    
    Returns:
        Timezone-naive datetime representing Eastern Time
        
    Example:
        >>> normalize_unix_timestamp(1718481600, timeframe="1D")
        datetime.datetime(2025, 6, 15, 0, 0, 0)  # Midnight ET
    """
    # Interpret Unix timestamp as UTC
    utc_ts = datetime.fromtimestamp(unix_ts, tz=timezone.utc)
    
    # Use the main normalize_timestamp function which handles ET conversion and timeframe normalization
    return normalize_timestamp(utc_ts, source_timezone="UTC", timeframe=timeframe)


def _to_et_datetime(ts: datetime, source_timezone: Optional[str] = None) -> datetime:
    """
    Convert a datetime to timezone-naive Eastern Time (ET).
    
    This is a helper function used by normalize_timestamp.
    
    Args:
        ts: Timestamp to convert
        source_timezone: Optional timezone name if ts is naive but represents a time in that timezone.
                        If None and ts is naive, assumes ts is already in ET.
    
    Returns:
        Timezone-naive datetime representing Eastern Time
    """
    # If timestamp is timezone-aware, convert to ET
    if ts.tzinfo is not None:
        if not _PYTZ_AVAILABLE:
            # Fallback: just remove timezone info (not ideal, but works if pytz not available)
            return ts.replace(tzinfo=None)
        
        # Convert to ET
        et_tz = pytz.timezone('America/New_York')
        # Convert to ET
        et_ts = ts.astimezone(et_tz)
        # Remove timezone info to make it naive
        return et_ts.replace(tzinfo=None)
    
    # If timestamp is naive, assume it's already in ET unless source_timezone is specified
    if source_timezone:
        if not _PYTZ_AVAILABLE:
            # Fallback: just return as-is (not ideal, but works if pytz not available)
            return ts
        
        # Parse source timezone
        source_tz = pytz.timezone(source_timezone)
        # Localize naive timestamp to source timezone
        localized = source_tz.localize(ts)
        # Convert to ET
        et_tz = pytz.timezone('America/New_York')
        et_ts = localized.astimezone(et_tz)
        # Remove timezone info to make it naive
        return et_ts.replace(tzinfo=None)
    
    # Naive timestamp, assume it's already in ET
    return ts


def to_et_timestamp(ts: datetime) -> datetime:
    """
    Convert a timestamp to timezone-naive Eastern Time (ET).
    
    This is a convenience function that assumes the input timestamp is already in ET
    if it's naive, or converts from its timezone to ET if it's timezone-aware.
    
    Args:
        ts: Timestamp to convert
        
    Returns:
        Timezone-naive datetime representing Eastern Time
    """
    return _to_et_datetime(ts)


class BarIntervalSpec:
    """Specification for a bar interval."""
    def __init__(self, unit: str, value: int, is_intraday: bool):
        self.unit = unit  # 'm', 'h', 'd', 'w', 'M'
        self.value = value  # e.g., 5 for "5m"
        self.is_intraday = is_intraday  # True for intraday (m, h), False for daily+ (d, w, M)


def parse_bar_interval(interval: str) -> BarIntervalSpec:
    """
    Parse a bar interval string into a BarIntervalSpec.
    
    Examples:
        "5m" -> BarIntervalSpec(unit='m', value=5, is_intraday=True)
        "1h" -> BarIntervalSpec(unit='h', value=1, is_intraday=True)
        "1D" -> BarIntervalSpec(unit='d', value=1, is_intraday=False)
        "1W" -> BarIntervalSpec(unit='w', value=1, is_intraday=False)
        "1M" -> BarIntervalSpec(unit='M', value=1, is_intraday=False)
    """
    interval = interval.upper().strip()
    
    # Handle special cases
    if interval in ('D', 'DAILY', 'DAY'):
        return BarIntervalSpec('d', 1, False)
    if interval in ('W', 'WEEKLY', 'WEEK', 'WK'):
        return BarIntervalSpec('w', 1, False)
    if interval in ('M', 'MONTHLY', 'MONTH', 'MO'):
        return BarIntervalSpec('M', 1, False)
    
    # Parse numeric + unit format (e.g., "5m", "1h", "1D")
    import re
    match = re.match(r'^(\d+)([MHDWMOMINHOURDAYWEEKMONTH]+)$', interval)
    if match:
        value_str, unit_str = match.groups()
        value = int(value_str)
        unit_str_upper = unit_str.upper()
        
        # Map unit strings to single-letter codes
        if unit_str_upper.startswith('M') and unit_str_upper != 'MONTH' and unit_str_upper != 'MO':
            unit = 'm'  # minutes
            is_intraday = True
        elif unit_str_upper.startswith('H') or unit_str_upper.startswith('HOUR'):
            unit = 'h'  # hours
            is_intraday = True
        elif unit_str_upper.startswith('D') or unit_str_upper.startswith('DAY'):
            unit = 'd'  # days
            is_intraday = False
        elif unit_str_upper.startswith('W') or unit_str_upper.startswith('WEEK'):
            unit = 'w'  # weeks
            is_intraday = False
        elif unit_str_upper.startswith('MONTH') or unit_str_upper.startswith('MO'):
            unit = 'M'  # months
            is_intraday = False
        else:
            raise ValueError(f"Unknown unit in interval: {interval}")
        
        return BarIntervalSpec(unit, value, is_intraday)
    
    raise ValueError(f"Invalid interval format: {interval}")


def _to_datetime(x: Any) -> dt.datetime:
    """Convert various types to datetime."""
    import datetime as dt
    if isinstance(x, dt.datetime):
        return x
    if isinstance(x, str):
        return dt.datetime.fromisoformat(x)
    if isinstance(x, (int, float)):
        return dt.datetime.fromtimestamp(x)
    raise TypeError(f"Cannot convert {type(x)} to datetime")


def normalize_bar_range(start: Any, end: Any, bar_interval: str) -> Tuple[dt.datetime, dt.datetime]:
    """
    Normalize a bar range to the appropriate boundaries based on the bar interval.
    
    For daily/weekly/monthly bars, rounds to day boundaries.
    For intraday bars (hours/minutes), rounds to the appropriate time boundaries.
    
    Args:
        start: Start datetime (can be datetime, string, or timestamp)
        end: End datetime (can be datetime, string, or timestamp)
        bar_interval: Bar interval string (e.g., "5m", "1h", "1D", "1W", "1M")
        
    Returns:
        Tuple of (normalized_start, normalized_end)
    """
    import datetime as dt
    from typing import Tuple, Any
    
    sdt = _to_datetime(start)
    edt = _to_datetime(end)
    if edt < sdt:
        raise ValueError(f'end must be >= start (start={sdt!r}, end={edt!r})')
    spec = parse_bar_interval(bar_interval)
    if not spec.is_intraday:
        s = dt.datetime.combine(sdt.date(), dt.time.min)
        e = dt.datetime.combine(edt.date(), dt.time.max)
        return s, e
    # intraday rounding
    if spec.unit == 'h':
        s = sdt.replace(minute=0, second=0, microsecond=0)
        e = edt.replace(minute=59, second=59, microsecond=999999)
        return s, e
    # minutes
    s = sdt.replace(second=0, microsecond=0)
    e = edt.replace(minute=59, second=59, microsecond=999999)
    return s, e


def normalize_to_date(ts: Union[datetime, pd.Timestamp, date, str]) -> date:
    """
    Normalize a timestamp to a Python date object for comparison.
    
    This is useful for matching timestamps that may have different times but represent
    the same trading day. Handles pandas Timestamps, datetime objects, date objects,
    and string representations.
    
    Args:
        ts: Timestamp to normalize (can be datetime, pd.Timestamp, date, or string)
        
    Returns:
        Python date object
        
    Examples:
        >>> from datetime import datetime
        >>> normalize_to_date(datetime(2025, 6, 15, 10, 30, 0))
        datetime.date(2025, 6, 15)
        >>> normalize_to_date(pd.Timestamp('2025-06-15 16:00:00'))
        datetime.date(2025, 6, 15)
    """
    if isinstance(ts, pd.Timestamp):
        return ts.normalize().date()
    elif isinstance(ts, datetime):
        return ts.date()
    elif isinstance(ts, date):
        return ts
    else:
        # Try to parse string or other format
        return pd.to_datetime(ts).normalize().date()


def normalize_timestamp_for_comparison(ts: datetime) -> datetime:
    """
    Normalize a timestamp by removing microseconds for comparison purposes.
    
    This is useful when comparing timestamps that may have slight microsecond differences
    but represent the same point in time. Commonly used in cache deduplication and
    timestamp matching.
    
    Args:
        ts: Timestamp to normalize
        
    Returns:
        Timestamp with microseconds set to 0
        
    Examples:
        >>> from datetime import datetime
        >>> ts1 = datetime(2025, 6, 15, 10, 30, 45, 123456)
        >>> ts2 = datetime(2025, 6, 15, 10, 30, 45, 789012)
        >>> normalize_timestamp_for_comparison(ts1) == normalize_timestamp_for_comparison(ts2)
        True
    """
    return ts.replace(microsecond=0)
