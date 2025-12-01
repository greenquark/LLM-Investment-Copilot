"""
Data module - provides data engines and caching.

This module exports:
- DataEngine: Base class for all data sources
- MarketDataAppAdapter: MarketData.app API adapter
- MoomooDataAdapter: Moomoo/Futu API adapter
- YFinanceDataAdapter: Yahoo Finance data adapter (yfinance library)
- CachedDataEngine: Transparent caching wrapper for any DataEngine
- create_data_engine: Factory function to create data engines from config
- create_data_engine_from_config: Factory function to create data engines from full env config
- get_trading_days: Get list of trading days between two dates
- is_trading_day: Check if a date is a trading day
- get_trading_days_set: Get set of trading days between two dates
"""

from core.data.base import DataEngine
from core.data.marketdata_app import MarketDataAppAdapter
from core.data.moomoo_data import MoomooDataAdapter

# Lazy import for trading calendar (optional dependency)
try:
    from core.data.trading_calendar import get_trading_days, is_trading_day, get_trading_days_set
    _TRADING_CALENDAR_AVAILABLE = True
except ImportError:
    _TRADING_CALENDAR_AVAILABLE = False
    get_trading_days = None  # type: ignore
    is_trading_day = None  # type: ignore
    get_trading_days_set = None  # type: ignore

# Lazy import for yfinance adapter (optional dependency)
try:
    from core.data.yfinance_data import YFinanceDataAdapter
    _YFINANCE_AVAILABLE = True
except ImportError:
    YFinanceDataAdapter = None  # type: ignore
    _YFINANCE_AVAILABLE = False

# Lazy import for cached engine (has optional dependencies)
try:
    from core.data.cached_engine import CachedDataEngine
    from core.data.factory import create_data_engine, create_data_engine_from_config, get_data_source_config
    __all__ = [
        "DataEngine",
        "MarketDataAppAdapter",
        "MoomooDataAdapter",
        "CachedDataEngine",
        "create_data_engine",
        "create_data_engine_from_config",
        "get_data_source_config",
    ]
    if _TRADING_CALENDAR_AVAILABLE:
        __all__.extend(["get_trading_days", "is_trading_day", "get_trading_days_set"])
    if _YFINANCE_AVAILABLE:
        __all__.append("YFinanceDataAdapter")
except ImportError:
    __all__ = [
        "DataEngine",
        "MarketDataAppAdapter",
        "MoomooDataAdapter",
    ]
    if _TRADING_CALENDAR_AVAILABLE:
        __all__.extend(["get_trading_days", "is_trading_day", "get_trading_days_set"])
    if _YFINANCE_AVAILABLE:
        __all__.append("YFinanceDataAdapter")

