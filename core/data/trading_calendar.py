"""
Trading calendar utilities for determining trading days.

Uses exchange-calendars library to get valid trading days for US exchanges.
"""
from __future__ import annotations
from datetime import date, datetime
from typing import List, Optional, Set
import logging

logger = logging.getLogger(__name__)

# Lazy import for exchange-calendars (optional dependency)
try:
    import exchange_calendars as ecals
    _EXCHANGE_CALENDARS_AVAILABLE = True
except ImportError:
    _EXCHANGE_CALENDARS_AVAILABLE = False
    ecals = None  # type: ignore


class TradingCalendar:
    """Wrapper for exchange calendar functionality."""
    
    def __init__(self, exchange: str = "NYSE"):
        """
        Initialize trading calendar.
        
        Args:
            exchange: Exchange name (default: "NYSE")
                     Options: "NYSE", "NASDAQ", "CME", etc.
        """
        if not _EXCHANGE_CALENDARS_AVAILABLE:
            raise ImportError(
                "exchange-calendars is not installed. Install it with: pip install exchange-calendars"
            )
        
        self.exchange = exchange
        try:
            self.calendar = ecals.get_calendar(exchange)
            logger.debug(f"TradingCalendar initialized for {exchange}")
        except Exception as e:
            logger.error(f"Failed to get calendar for {exchange}: {e}")
            raise RuntimeError(f"Failed to initialize trading calendar for {exchange}: {e}") from e
    
    def get_trading_days(self, start: date, end: date) -> List[date]:
        """
        Get list of trading days between start and end (inclusive).
        
        Args:
            start: Start date
            end: End date
            
        Returns:
            List of trading days (date objects)
        """
        try:
            # Use sessions_in_range method (correct API for exchange-calendars)
            # Returns a DatetimeIndex, convert to list of date objects
            trading_days = self.calendar.sessions_in_range(start, end)
            # Convert to list of date objects
            # Handle pandas Timestamp, datetime, and date objects
            # Note: pandas Timestamp is a subclass of date, so isinstance(d, date) returns True
            # We need to check if it has a date() method (Timestamp/datetime) vs plain date
            dates = []
            for d in trading_days:
                # Check if it has a date() method (pandas Timestamp or datetime objects)
                # Plain Python date objects don't have a date() method
                if hasattr(d, 'date') and callable(getattr(d, 'date', None)):
                    # pandas Timestamp or datetime object - convert to Python date
                    dates.append(d.date())
                elif isinstance(d, date):
                    # Plain Python date object (no date() method)
                    dates.append(d)
                else:
                    # Fallback: try to convert string or other format
                    try:
                        if hasattr(d, 'to_pydatetime'):
                            dates.append(d.to_pydatetime().date())
                        else:
                            # Last resort: try parsing as string
                            dates.append(date.fromisoformat(str(d)[:10]))
                    except Exception as e:
                        logger.warning(f"Could not convert {d} (type: {type(d)}) to date: {e}")
            return sorted(dates)
        except Exception as e:
            logger.error(f"Failed to get trading days for {start} to {end}: {e}")
            raise RuntimeError(f"Failed to get trading days: {e}") from e
    
    def is_trading_day(self, day: date) -> bool:
        """
        Check if a given date is a trading day.
        
        Args:
            day: Date to check
            
        Returns:
            True if it's a trading day, False otherwise
        """
        try:
            return self.calendar.is_session(day)
        except Exception as e:
            logger.warning(f"Failed to check if {day} is trading day: {e}")
            # Fallback: assume it's a trading day if we can't check
            return True
    
    def get_trading_days_set(self, start: date, end: date) -> Set[date]:
        """
        Get set of trading days between start and end (inclusive).
        
        Args:
            start: Start date
            end: End date
            
        Returns:
            Set of trading days (date objects)
        """
        return set(self.get_trading_days(start, end))
    
    def next_trading_day(self, day: date) -> Optional[date]:
        """
        Get the next trading day after the given date.
        
        Args:
            day: Reference date
            
        Returns:
            Next trading day, or None if not found
        """
        try:
            next_day = self.calendar.next_session(day)
            return next_day.date() if isinstance(next_day, datetime) else next_day
        except Exception as e:
            logger.warning(f"Failed to get next trading day after {day}: {e}")
            return None
    
    def previous_trading_day(self, day: date) -> Optional[date]:
        """
        Get the previous trading day before the given date.
        
        Args:
            day: Reference date (must be a pure date object, not datetime)
            
        Returns:
            Previous trading day, or None if not found
        """
        try:
            # Ensure we have a pure date object (not datetime or string)
            if isinstance(day, datetime):
                day = day.date()
            elif isinstance(day, str):
                day = date.fromisoformat(day[:10])
            elif not isinstance(day, date):
                # Try to convert other types
                if hasattr(day, 'date'):
                    day = day.date()
                else:
                    day = date.fromisoformat(str(day)[:10])
            
            # Use get_trading_days to find previous trading day
            # This is more reliable than previous_session which may fail for non-session dates
            from datetime import timedelta
            start_date = day - timedelta(days=60)  # Look back 60 days to ensure we find one
            trading_days = self.get_trading_days(start_date, day)
            # Filter out the current day and get the previous one
            prev_days = [d for d in trading_days if d < day]
            if prev_days:
                return prev_days[-1]
            else:
                # Fallback: try calendar method if get_trading_days didn't work
                try:
                    prev_day = self.calendar.previous_session(day)
                    return prev_day.date() if isinstance(prev_day, datetime) else prev_day
                except Exception:
                    # If that also fails, return None
                    return None
        except Exception as e:
            logger.warning(f"Failed to get previous trading day before {day}: {e}")
            return None


# Global calendar instance (lazy initialization)
_global_calendar: Optional[TradingCalendar] = None


def get_trading_calendar(exchange: str = "NYSE") -> TradingCalendar:
    """
    Get or create a global trading calendar instance.
    
    Args:
        exchange: Exchange name (default: "NYSE")
        
    Returns:
        TradingCalendar instance
    """
    global _global_calendar
    if _global_calendar is None or _global_calendar.exchange != exchange:
        _global_calendar = TradingCalendar(exchange)
    return _global_calendar


def get_trading_days(start: date, end: date, exchange: str = "NYSE") -> List[date]:
    """
    Convenience function to get trading days.
    
    Args:
        start: Start date
        end: End date
        exchange: Exchange name (default: "NYSE")
        
    Returns:
        List of trading days
    """
    calendar = get_trading_calendar(exchange)
    return calendar.get_trading_days(start, end)


def is_trading_day(day: date, exchange: str = "NYSE") -> bool:
    """
    Convenience function to check if a date is a trading day.
    
    Args:
        day: Date to check
        exchange: Exchange name (default: "NYSE")
        
    Returns:
        True if it's a trading day, False otherwise
    """
    calendar = get_trading_calendar(exchange)
    return calendar.is_trading_day(day)


def get_trading_days_set(start: date, end: date, exchange: str = "NYSE") -> Set[date]:
    """
    Convenience function to get trading days as a set.
    
    Args:
        start: Start date
        end: End date
        exchange: Exchange name (default: "NYSE")
        
    Returns:
        Set of trading days (date objects)
    """
    return set(get_trading_days(start, end, exchange))

