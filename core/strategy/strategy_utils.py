"""
Common utility functions for strategy implementations.

This module provides shared helper functions used across multiple strategies
to reduce code duplication and ensure consistency.
"""

from __future__ import annotations
from datetime import datetime, date
from typing import Optional


def check_trading_day(current_date: date) -> bool:
    """
    Check if a date is a trading day, handling errors gracefully.
    
    This is a common pattern across strategies to skip weekends and holidays.
    If the trading calendar check fails, returns True (fallback behavior)
    to allow the strategy to continue.
    
    Args:
        current_date: Date to check
        
    Returns:
        True if it's a trading day, False if not (weekend/holiday).
        Returns True if the check fails (fallback behavior).
        
    Examples:
        >>> from datetime import date
        >>> check_trading_day(date(2025, 6, 15))  # Sunday
        False
        >>> check_trading_day(date(2025, 6, 16))  # Monday
        True
    """
    try:
        from core.data.trading_calendar import is_trading_day
        return is_trading_day(current_date)
    except Exception:
        # If trading calendar check fails, continue (fallback behavior)
        return True


def get_end_time_for_timeframe(now: datetime, current_date: date, timeframe: str) -> datetime:
    """
    Get the appropriate end time for fetching bars based on timeframe.
    
    For daily bars, requests up to end of current day to ensure we get today's bar.
    For intraday bars, uses the current time.
    
    Args:
        now: Current datetime
        current_date: Current date
        timeframe: Timeframe string (e.g., "1D", "15m")
        
    Returns:
        Appropriate end time for bar fetching
        
    Examples:
        >>> from datetime import datetime, date
        >>> now = datetime(2025, 6, 15, 14, 30, 0)
        >>> get_end_time_for_timeframe(now, now.date(), "1D")
        datetime.datetime(2025, 6, 15, 23, 59, 59, 999999)
        >>> get_end_time_for_timeframe(now, now.date(), "15m")
        datetime.datetime(2025, 6, 15, 14, 30, 0)
    """
    if timeframe.upper() in ("D", "1D"):
        # For daily bars, request up to end of current day to ensure we get today's bar
        return datetime.combine(current_date, datetime.max.time())
    else:
        # For intraday bars, use current time
        return now
