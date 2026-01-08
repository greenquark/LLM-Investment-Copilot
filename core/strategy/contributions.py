"""
Contribution Manager - Handles period calculation for DCA strategies.

This module provides a unified way to calculate contribution periods
for different frequencies (weekly, monthly, quarterly, yearly).
"""

from __future__ import annotations
from datetime import date, timedelta
from typing import Optional


class ContributionManager:
    """
    Manages contribution period calculations for DCA strategies.
    
    This class eliminates code duplication by providing a single implementation
    of period key calculation, period detection, and related utilities.
    """
    
    def __init__(self, frequency: str):
        """
        Initialize the contribution manager.
        
        Args:
            frequency: Contribution frequency - "weekly", "monthly", "quarterly", or "yearly"
        """
        frequency_lower = frequency.lower()
        if frequency_lower not in ("weekly", "monthly", "quarterly", "yearly"):
            raise ValueError(
                f"Invalid contribution frequency: {frequency}. "
                f"Must be one of: weekly, monthly, quarterly, yearly"
            )
        self.frequency = frequency_lower
    
    def period_key(self, d: date) -> tuple[int, int]:
        """
        Return (year, period_id) for the configured contribution frequency.
        
        Args:
            d: Date to calculate period key for
        
        Returns:
            Tuple of (year, period_id) where period_id depends on frequency:
            - weekly: (year, ISO week number)
            - monthly: (year, month number 1-12)
            - quarterly: (year, quarter number 1-4)
            - yearly: (year, 1)
        """
        if self.frequency == "weekly":
            iso_year, iso_week, _ = d.isocalendar()
            return (iso_year, iso_week)
        if self.frequency == "monthly":
            return (d.year, d.month)
        if self.frequency == "quarterly":
            q = (d.month - 1) // 3 + 1
            return (d.year, q)
        # yearly
        return (d.year, 1)
    
    def is_new_period(self, date: date, last_period: Optional[tuple[int, int]]) -> bool:
        """
        Check if the given date represents a new contribution period.
        
        Args:
            date: Date to check
            last_period: Last period key (year, period_id), or None if no previous period
        
        Returns:
            True if this is a new period, False otherwise
        """
        if last_period is None:
            return True
        current_period = self.period_key(date)
        return current_period != last_period
    
    def next_contribution_date(self, date: date) -> date:
        """
        Calculate the next contribution date after the given date.
        
        Args:
            date: Reference date
        
        Returns:
            Next contribution date based on frequency
        """
        if self.frequency == "weekly":
            # Next week (7 days later)
            return date + timedelta(days=7)
        if self.frequency == "monthly":
            # Next month (approximate - add 30 days, then adjust to first of month)
            next_month = date.month + 1
            next_year = date.year
            if next_month > 12:
                next_month = 1
                next_year += 1
            # Return first day of next month
            return date.replace(year=next_year, month=next_month, day=1)
        if self.frequency == "quarterly":
            # Next quarter (3 months later)
            next_month = date.month + 3
            next_year = date.year
            if next_month > 12:
                next_month -= 12
                next_year += 1
            # Return first day of next quarter
            quarter_start_month = ((next_month - 1) // 3) * 3 + 1
            return date.replace(year=next_year, month=quarter_start_month, day=1)
        # yearly
        # Next year, same month and day
        return date.replace(year=date.year + 1)

