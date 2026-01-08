"""
Backtest Context - Context implementation for backtests.

This module provides the SimpleContext class used by the backtest engine
to provide context to strategies.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any

from core.portfolio.portfolio import Portfolio
from core.execution.simulated import SimulatedExecutionEngine
from core.utils.logging import Logger
from core.models.bar import Bar
from core.data.provider import DataEngineProvider, FearGreedIndexProvider


@dataclass
class SimpleContext:
    """Context implementation for backtest strategies."""
    portfolio: Portfolio
    execution: SimulatedExecutionEngine
    logger: Logger
    symbol: str
    _data_source: str = ""  # Track data source name (e.g., "Cache", "MarketData", "Moomoo")
    _bars: Optional[List[Bar]] = None  # Bars fetched by engine, can be reused by strategy
    _now: Optional[datetime] = None  # Current decision timestamp
    _config: Optional[Dict[str, Any]] = None  # Strategy configuration
    _data_provider: Optional[DataEngineProvider] = None  # Data provider for bars
    _fgi_provider: Optional[FearGreedIndexProvider] = None  # FGI provider

    @property
    def now(self) -> datetime:
        """Current decision timestamp."""
        if self._now is None:
            return datetime.utcnow()
        return self._now
    
    @property
    def config(self) -> Dict[str, Any]:
        """Strategy configuration dictionary."""
        return self._config or {}
    
    @property
    def data_provider(self) -> DataEngineProvider:
        """Data provider for accessing bars."""
        if self._data_provider is None:
            raise RuntimeError("Data provider not initialized in context")
        return self._data_provider
    
    @property
    def fgi_provider(self) -> FearGreedIndexProvider:
        """FGI provider for accessing Fear & Greed Index values."""
        if self._fgi_provider is None:
            raise RuntimeError("FGI provider not initialized in context")
        return self._fgi_provider

    def log(self, msg: str) -> None:
        # Add data source prefix if available
        prefix = f"[{self._data_source}] " if self._data_source else ""
        self.logger.log(f"{prefix}{msg}")
    
    def get_bars(self, lookback_days: Optional[int] = None) -> Optional[List[Bar]]:
        """
        Get bars that were fetched by the engine.
        
        Args:
            lookback_days: Optional lookback period (currently ignored, returns all bars)
        
        Returns:
            List of bars fetched by the engine, or None if not available
        """
        return self._bars

