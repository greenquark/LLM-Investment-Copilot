"""
Backtest Data Provider - Handles data fetching for backtests.

This module provides a dedicated data provider for backtests that handles
lookback period calculation and bar caching for strategies.
"""

from __future__ import annotations
from datetime import datetime, timedelta
from typing import List, Optional

from core.data.base import DataEngine
from core.models.bar import Bar


class BacktestDataProvider:
    """
    Handles data fetching for backtests with appropriate lookback periods.
    
    This class encapsulates the logic for determining lookback periods based on
    timeframe and fetching bars for strategies.
    """
    
    def __init__(self, data_engine: DataEngine):
        """
        Initialize the backtest data provider.
        
        Args:
            data_engine: DataEngine instance for fetching bars
        """
        self._data_engine = data_engine
        self._cached_bars: Optional[List[Bar]] = None
        self._cached_timestamp: Optional[datetime] = None
    
    def _calculate_lookback_start(self, now: datetime, timeframe: str) -> datetime:
        """
        Calculate the lookback start time based on timeframe.
        
        Args:
            now: Current decision timestamp
            timeframe: Bar timeframe (e.g., "1D", "15m", "1W")
        
        Returns:
            Start datetime for lookback period
        """
        timeframe_upper = timeframe.upper()
        
        if timeframe_upper in ("D", "1D") or (timeframe_upper.endswith("D") and not timeframe_upper.startswith("W")):
            # For daily: strategy needs 30 days
            return now - timedelta(days=30)
        elif timeframe_upper in ("W", "1W") or timeframe_upper.endswith("W"):
            # For weekly: get bars up to current decision point
            return now - timedelta(weeks=14)
        else:
            # For intraday (hourly, minutely): strategy needs longer lookback
            if timeframe_upper.endswith("H") or timeframe_upper == "H":
                # Hourly: strategy needs 30 days
                return now - timedelta(days=30)
            else:
                # Minutely: strategy needs 7 days
                return now - timedelta(days=7)
    
    async def get_bars(
        self,
        symbol: str,
        now: datetime,
        timeframe: str,
    ) -> List[Bar]:
        """
        Get bars for a symbol up to the current decision point.
        
        This method fetches bars with appropriate lookback period and caches them
        for reuse by strategies.
        
        Args:
            symbol: Stock symbol
            now: Current decision timestamp
            timeframe: Bar timeframe
        
        Returns:
            List of Bar objects
        """
        # Calculate lookback start
        lookback_start = self._calculate_lookback_start(now, timeframe)
        
        # Fetch bars
        bars = await self._data_engine.get_bars(symbol, lookback_start, now, timeframe=timeframe)
        
        # Cache bars for potential reuse
        self._cached_bars = bars
        self._cached_timestamp = now
        
        return bars
    
    def get_cached_bars(self) -> Optional[List[Bar]]:
        """
        Get cached bars from the last fetch.
        
        Returns:
            Cached bars or None if not available
        """
        return self._cached_bars

