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
from datetime import datetime, timezone
from typing import Optional

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
    # This represents "N/A" since the bar covers the entire period - it's as good as market close
    # Intraday bars keep their actual timestamps
    normalize_to_midnight = False
    if timeframe and _PYTZ_AVAILABLE:
        timeframe_upper = timeframe.upper()
        # Check if this is a complete historical bar (daily, weekly, monthly)
        # Daily: ends with "D" but not "WD" (weekly daily)
        is_daily = (timeframe_upper.endswith("D") or timeframe_upper == "D") and not timeframe_upper.startswith("W")
        # Weekly: ends with "W" or is "W"
        is_weekly = timeframe_upper.endswith("W") or timeframe_upper == "W"
        # Monthly: ends with "MO" (yfinance uses "1mo", "3mo") or is "MO"
        # Avoid matching minutely timeframes like "1m", "15m", "30m" which end with "M" but not "MO"
        is_monthly = timeframe_upper.endswith("MO") or timeframe_upper == "MO"
        
        if is_daily or is_weekly or is_monthly:
            # Complete historical bar - normalize to midnight (00:00:00 ET)
            normalize_to_midnight = True
    
    # Convert to ET naive
    et_tz = None
    if _PYTZ_AVAILABLE:
        et_tz = pytz.timezone('America/New_York')
    
    if ts.tzinfo is not None:
        # Timezone-aware: convert to ET, then make naive
        if et_tz:
            et_ts = ts.astimezone(et_tz)
            result = et_ts.replace(tzinfo=None)
        else:
            # Fallback: convert to UTC if pytz not available
            utc_ts = ts.astimezone(timezone.utc)
            result = utc_ts.replace(tzinfo=None)
    elif source_timezone and _PYTZ_AVAILABLE and et_tz:
        # Naive timestamp with known source timezone: convert to ET
        try:
            source_tz = pytz.timezone(source_timezone)
            # Localize naive datetime to source timezone
            localized = source_tz.localize(ts)
            # Convert to ET
            et_ts = localized.astimezone(et_tz)
            result = et_ts.replace(tzinfo=None)
        except Exception:
            # If timezone conversion fails, assume it's already ET
            result = ts
    else:
        # Naive timestamp, no source timezone: assume it's already ET
        result = ts
    
    # If this is a complete historical bar (daily, weekly, monthly), normalize to midnight (00:00:00 ET)
    # This represents "N/A" since the bar covers the entire period - it's as good as market close
    if normalize_to_midnight:
        # Complete historical bar - use midnight (00:00:00 ET)
        # Keep the same date, set time to midnight
        result = result.replace(hour=0, minute=0, second=0, microsecond=0)
    
    return result


def normalize_unix_timestamp(
    unix_ts: float | int,
    timeframe: Optional[str] = None,
) -> datetime:
    """
    Convert Unix timestamp to normalized timezone-naive Eastern Time (ET) datetime.
    
    Unix timestamps are UTC-based, so we convert UTC → ET → naive ET.
    
    For complete historical bars (daily, weekly, monthly), timestamps are normalized
    to 00:00:00 ET (midnight) to represent "N/A" since the bar covers the entire period.
    
    Args:
        unix_ts: Unix timestamp (seconds since epoch)
        timeframe: Optional timeframe string (e.g., "1D", "1wk", "1mo") to determine
                   if this is a complete historical bar that should be normalized to midnight.
        
    Returns:
        Timezone-naive datetime representing Eastern Time
    """
    # Convert Unix timestamp (UTC) to ET
    if _PYTZ_AVAILABLE:
        et_tz = pytz.timezone('America/New_York')
        utc_ts = datetime.fromtimestamp(unix_ts, tz=timezone.utc)
        et_ts = utc_ts.astimezone(et_tz)
        result = et_ts.replace(tzinfo=None)
        
        # For complete historical bars, normalize to midnight (00:00:00 ET)
        if timeframe:
            timeframe_upper = timeframe.upper()
            is_daily = (timeframe_upper.endswith("D") or timeframe_upper == "D") and not timeframe_upper.startswith("W")
            is_weekly = timeframe_upper.endswith("W") or timeframe_upper == "W"
            is_monthly = timeframe_upper.endswith("MO") or timeframe_upper == "MO"
            
            if is_daily or is_weekly or is_monthly:
                # Complete historical bar - use midnight (00:00:00 ET)
                result = result.replace(hour=0, minute=0, second=0, microsecond=0)
        
        return result
    else:
        # Fallback: return UTC if pytz not available
        return datetime.fromtimestamp(unix_ts, tz=timezone.utc).replace(tzinfo=None)


def to_et_timestamp(ts: datetime) -> datetime:
    """
    Convert a timestamp to ET for display purposes.
    
    Since timestamps are already normalized to ET, this function is mainly
    for backward compatibility or when converting from other timezones.
    
    Args:
        ts: Timestamp (timezone-naive, assumed to be ET already)
        
    Returns:
        Timezone-naive datetime representing ET time
        
    Note:
        Timestamps are already normalized to ET, so this is mainly a no-op
        unless the timestamp is timezone-aware.
    """
    if ts.tzinfo is not None:
        # Timezone-aware: convert to ET
        if _PYTZ_AVAILABLE:
            et_tz = pytz.timezone('America/New_York')
            et_ts = ts.astimezone(et_tz)
            return et_ts.replace(tzinfo=None)
        else:
            # Fallback: convert to UTC
            utc_ts = ts.astimezone(timezone.utc)
            return utc_ts.replace(tzinfo=None)
    else:
        # Already timezone-naive, assume it's ET
        return ts

