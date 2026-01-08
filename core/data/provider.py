"""
Data Provider Abstractions - Interfaces for accessing various data sources.

This module provides abstract interfaces for data providers, allowing strategies
to access data without directly importing data source modules.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import datetime, date
from typing import List, Optional, Tuple

from core.models.bar import Bar


class DataProvider(ABC):
    """
    Abstract interface for providing market data (bars).
    
    This abstraction allows strategies to access bars without directly
    depending on DataEngine implementations.
    """
    
    @abstractmethod
    async def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str,
    ) -> List[Bar]:
        """
        Get bars for a symbol over a time range.
        
        Args:
            symbol: Stock symbol
            start: Start datetime
            end: End datetime
            timeframe: Bar timeframe (e.g., "1D", "15m")
        
        Returns:
            List of Bar objects
        """
        ...


class FGIProvider(ABC):
    """
    Abstract interface for providing Fear & Greed Index (FGI) data.
    
    This abstraction allows strategies to access FGI values without directly
    importing the fear_greed_index module.
    """
    
    @abstractmethod
    async def get_fgi(self, target_date: Optional[date] = None) -> Tuple[Optional[float], Optional[str]]:
        """
        Get Fear & Greed Index value for a date.
        
        Args:
            target_date: Date to get FGI for, or None for current value
        
        Returns:
            Tuple of (fgi_value, source) where:
            - fgi_value: FGI value (0-100) or None if unavailable
            - source: Source of the data ("csv", "api", "current") or None
        """
        ...


class DataEngineProvider(DataProvider):
    """
    Concrete implementation of DataProvider using a DataEngine.
    """
    
    def __init__(self, data_engine):
        """
        Initialize with a DataEngine instance.
        
        Args:
            data_engine: DataEngine instance (from core.data.base)
        """
        self._data_engine = data_engine
    
    async def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str,
    ) -> List[Bar]:
        """Get bars using the underlying DataEngine."""
        return await self._data_engine.get_bars(symbol, start, end, timeframe=timeframe)


class FearGreedIndexProvider(FGIProvider):
    """
    Concrete implementation of FGIProvider using the fear_greed_index module.
    """
    
    async def get_fgi(self, target_date: Optional[date] = None) -> Tuple[Optional[float], Optional[str]]:
        """
        Get FGI value using the core.data.fear_greed_index module.
        
        Note: This is a synchronous function wrapped in async for interface compatibility.
        """
        from core.data.fear_greed_index import get_fgi_value
        # get_fgi_value is synchronous, but we wrap it in async for interface consistency
        return get_fgi_value(target_date=target_date)

