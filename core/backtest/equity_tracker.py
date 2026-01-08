"""
Equity Tracker - Tracks equity curve during backtests.

This module provides a dedicated class for tracking equity curves over time,
handling price updates and equity calculations.
"""

from __future__ import annotations
from datetime import datetime
from typing import Dict, Optional

from core.portfolio.portfolio import Portfolio


class EquityTracker:
    """
    Tracks equity curve over time during backtests.
    
    This class handles price updates and equity calculations, maintaining
    a complete equity curve for performance evaluation.
    """
    
    def __init__(self):
        """Initialize the equity tracker."""
        self._equity_curve: Dict[datetime, float] = {}
        self._last_price: Dict[str, float] = {}
    
    def update_price(self, symbol: str, price: float) -> None:
        """
        Update the last known price for a symbol.
        
        Args:
            symbol: Stock symbol
            price: Current price
        """
        self._last_price[symbol] = price
    
    def record_equity(
        self,
        timestamp: datetime,
        portfolio: Portfolio,
        symbol: str,
        fallback_price: Optional[float] = None,
    ) -> None:
        """
        Record equity value at a timestamp.
        
        Args:
            timestamp: Timestamp to record equity at
            portfolio: Portfolio to calculate equity for
            symbol: Stock symbol for price lookup
            fallback_price: Optional fallback price if last_price not available
        """
        # Get price from last_price or fallback
        price = self._last_price.get(symbol)
        if price is None or price == 0.0:
            price = fallback_price or 0.0
        
        if price > 0:
            prices = {symbol: price}
            equity = portfolio.equity(prices)
        else:
            # If no price available, use previous equity value or portfolio cash
            if self._equity_curve:
                # Use last known equity
                last_equity = list(self._equity_curve.values())[-1]
                equity = last_equity
            else:
                # Use portfolio cash as initial equity
                equity = portfolio.state.cash
        
        self._equity_curve[timestamp] = equity
    
    def get_equity_curve(self) -> Dict[datetime, float]:
        """
        Get the complete equity curve.
        
        Returns:
            Dictionary mapping timestamp -> equity value
        """
        return self._equity_curve.copy()
    
    def get_last_price(self, symbol: str) -> Optional[float]:
        """
        Get the last known price for a symbol.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Last known price or None if not available
        """
        return self._last_price.get(symbol)

