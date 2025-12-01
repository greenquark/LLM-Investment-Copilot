"""
Trading days utility functions.

This module provides helper functions for working with trading days.
It re-exports functions from core.data.trading_calendar for convenience.

Usage:
    from core.utils.trading_days import get_trading_days, is_trading_day
    
    # Get trading days in a range
    trading_days = get_trading_days(date(2025, 6, 1), date(2025, 6, 30))
    
    # Check if a date is a trading day
    if is_trading_day(date(2025, 6, 15)):
        print("It's a trading day!")
"""

from __future__ import annotations
from datetime import date, datetime, timedelta
from typing import List, Optional

# Import from core.data (centralized export)
try:
    from core.data import get_trading_days, is_trading_day, get_trading_days_set
    _TRADING_CALENDAR_AVAILABLE = True
except ImportError:
    # Fallback: try direct import
    try:
        from core.data.trading_calendar import get_trading_days, is_trading_day, get_trading_days_set
        _TRADING_CALENDAR_AVAILABLE = True
    except ImportError:
        _TRADING_CALENDAR_AVAILABLE = False
        get_trading_days = None  # type: ignore
        is_trading_day = None  # type: ignore
        get_trading_days_set = None  # type: ignore


def get_trading_days_lookback(reference_date: date, num_trading_days: int, exchange: str = "NYSE") -> date:
    """
    Get the date that is N trading days before the reference date.
    
    This is useful for calculating lookback periods for daily bars.
    For example, to get 30 trading days of data, use:
        lookback_start = get_trading_days_lookback(today, 30)
    
    Args:
        reference_date: The reference date (usually today or current date)
        num_trading_days: Number of trading days to look back
        exchange: Exchange name (default: "NYSE")
        
    Returns:
        The date that is N trading days before the reference date
        
    Note:
        If trading calendar is not available, falls back to calendar days
        (approximately num_trading_days * 1.4 to account for weekends/holidays)
    """
    if not _TRADING_CALENDAR_AVAILABLE or not get_trading_days:
        # Fallback: approximate trading days (account for weekends/holidays)
        # Roughly 5 trading days per week, so multiply by 1.4 to get calendar days
        calendar_days = int(num_trading_days * 1.4)
        return reference_date - timedelta(days=calendar_days)
    
    # Get trading days going backwards
    # Start from a date far enough back to ensure we get enough trading days
    # Use approximately 1.4x calendar days to account for weekends/holidays
    start_date = reference_date - timedelta(days=int(num_trading_days * 1.4) + 10)
    
    try:
        trading_days = get_trading_days(start_date, reference_date, exchange=exchange)
        # Normalize to Python date objects (trading_days may contain pandas Timestamps)
        normalized_days = []
        for d in trading_days:
            if hasattr(d, 'date') and callable(getattr(d, 'date', None)):
                normalized_days.append(d.date())
            elif isinstance(d, date):
                normalized_days.append(d)
            else:
                try:
                    normalized_days.append(date.fromisoformat(str(d)))
                except (ValueError, AttributeError):
                    continue
        
        if len(normalized_days) >= num_trading_days:
            # Return the date that is N trading days before
            return normalized_days[-num_trading_days]
        else:
            # Not enough trading days in range, go further back
            # Recursively try with a larger range
            start_date = reference_date - timedelta(days=int(num_trading_days * 2))
            trading_days = get_trading_days(start_date, reference_date, exchange=exchange)
            # Normalize again
            normalized_days = []
            for d in trading_days:
                if hasattr(d, 'date') and callable(getattr(d, 'date', None)):
                    normalized_days.append(d.date())
                elif isinstance(d, date):
                    normalized_days.append(d)
                else:
                    try:
                        normalized_days.append(date.fromisoformat(str(d)))
                    except (ValueError, AttributeError):
                        continue
            
            if len(normalized_days) >= num_trading_days:
                return normalized_days[-num_trading_days]
            else:
                # Fallback to calendar days approximation
                calendar_days = int(num_trading_days * 1.4)
                return reference_date - timedelta(days=calendar_days)
    except Exception:
        # Fallback to calendar days approximation
        calendar_days = int(num_trading_days * 1.4)
        return reference_date - timedelta(days=calendar_days)


def get_trading_days_lookback_datetime(reference_datetime: datetime, num_trading_days: int, exchange: str = "NYSE") -> datetime:
    """
    Get the datetime that is N trading days before the reference datetime.
    
    Useful for calculating lookback periods when working with datetime objects.
    
    Args:
        reference_datetime: The reference datetime
        num_trading_days: Number of trading days to look back
        exchange: Exchange name (default: "NYSE")
        
    Returns:
        The datetime that is N trading days before the reference datetime
        (time component is preserved from reference_datetime)
    """
    reference_date = reference_datetime.date()
    lookback_date = get_trading_days_lookback(reference_date, num_trading_days, exchange=exchange)
    # Preserve the time component from the reference datetime
    return datetime.combine(lookback_date, reference_datetime.time())


__all__ = [
    "get_trading_days",
    "is_trading_day",
    "get_trading_days_set",
    "get_trading_days_lookback",
    "get_trading_days_lookback_datetime",
]

