"""
Decision Tracker - Tracks trading decisions and portfolio state during backtests.

This module provides a clean separation between logging and decision tracking,
allowing strategies to record decisions without mixing concerns with logging.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, date
from typing import Dict, Optional, Tuple
from core.strategy.contributions import ContributionManager


@dataclass
class Decision:
    """Represents a single trading decision."""
    date: date
    timestamp: datetime
    fgi: Optional[float] = None
    action: str = "HOLD"
    action_type: Optional[str] = None  # Raw action type (BUY_FEAR, SELL_PROPORTIONAL, etc.)
    price: float = 0.0
    cash: Optional[float] = None
    shares: Optional[float] = None


class DecisionTracker:
    """
    Tracks trading decisions and portfolio state during backtests.
    
    This class is separate from logging to maintain clear separation of concerns.
    It tracks decisions by period (week/month/quarter/year) and can record
    portfolio state at decision points.
    """
    
    def __init__(self, contribution_frequency: str = "weekly"):
        """
        Initialize the decision tracker.
        
        Args:
            contribution_frequency: Frequency of contributions ("weekly", "monthly", etc.)
        """
        self._contribution_manager = ContributionManager(contribution_frequency)
        self._decisions_by_period: Dict[Tuple[int, int], Decision] = {}
        self._recorded_periods: set[Tuple[int, int]] = set()
    
    def period_key(self, d: date) -> Tuple[int, int]:
        """
        Get period key for a date.
        
        Args:
            d: Date to get period key for
        
        Returns:
            Tuple of (year, period_id)
        """
        return self._contribution_manager.period_key(d)
    
    def record_decision(
        self,
        timestamp: datetime,
        fgi_value: Optional[float],
        action: str,
        price: float,
        formatted_action: Optional[str] = None,
    ) -> None:
        """
        Record a trading decision.
        
        Args:
            timestamp: Timestamp of the decision
            fgi_value: Fear & Greed Index value (if applicable)
            action: Action type (e.g., "BUY_FEAR", "SELL_PROPORTIONAL", "HOLD")
            price: Price at which decision was made
            formatted_action: Optional formatted action string for display
        """
        current_date = timestamp.date()
        period_key = self.period_key(current_date)
        
        decision = Decision(
            date=current_date,
            timestamp=timestamp,
            fgi=fgi_value,
            action=formatted_action or action,
            action_type=action,
            price=price,
        )
        
        # If period already exists, update it (but preserve earlier date if it was a trading day)
        if period_key in self._decisions_by_period:
            existing = self._decisions_by_period[period_key]
            # If existing had price=0.0 (non-trading day) and new has price>0.0 (trading day),
            # update to use the trading day's date
            if existing.price == 0.0 and price > 0.0:
                decision.date = current_date  # Use trading day date
            elif existing.price > 0.0 and price > 0.0:
                # Both are trading days - keep the earlier date
                if current_date < existing.date:
                    decision.date = current_date
                else:
                    decision.date = existing.date
            else:
                decision.date = existing.date  # Keep existing date
        else:
            self._decisions_by_period[period_key] = decision
        
        self._decisions_by_period[period_key] = decision
        self._recorded_periods.add(period_key)
    
    def record_state(
        self,
        period_key: Tuple[int, int],
        cash: float,
        shares: float,
    ) -> None:
        """
        Record portfolio state for a period.
        
        Args:
            period_key: Period key (year, period_id)
            cash: Cash balance
            shares: Number of shares held
        """
        if period_key in self._decisions_by_period:
            self._decisions_by_period[period_key].cash = cash
            self._decisions_by_period[period_key].shares = shares
    
    def get_decision(self, period_key: Tuple[int, int]) -> Optional[Decision]:
        """
        Get decision for a period.
        
        Args:
            period_key: Period key (year, period_id)
        
        Returns:
            Decision object or None if not found
        """
        return self._decisions_by_period.get(period_key)
    
    def get_all_decisions(self) -> Dict[Tuple[int, int], Decision]:
        """
        Get all recorded decisions.
        
        Returns:
            Dictionary mapping period_key -> Decision
        """
        return self._decisions_by_period.copy()
    
    def is_period_recorded(self, period_key: Tuple[int, int]) -> bool:
        """
        Check if a period has been recorded.
        
        Args:
            period_key: Period key (year, period_id)
        
        Returns:
            True if period has been recorded, False otherwise
        """
        return period_key in self._recorded_periods

