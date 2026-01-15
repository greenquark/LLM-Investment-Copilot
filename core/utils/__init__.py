"""
Utility modules for the trading agent.

This module provides various utility functions and helpers.
"""

# Export trading days utilities for convenience
try:
    from core.utils.trading_days import (
        get_trading_days,
        is_trading_day,
        get_trading_days_set,
        get_trading_days_lookback,
        get_trading_days_lookback_datetime,
    )
    _TRADING_DAYS_AVAILABLE = True
except ImportError:
    _TRADING_DAYS_AVAILABLE = False

# Export timestamp normalization utilities
try:
    from core.utils.timestamp import (
        normalize_timestamp,
        normalize_unix_timestamp,
        to_et_timestamp,
        normalize_to_date,
        normalize_timestamp_for_comparison,
    )
    _TIMESTAMP_UTILS_AVAILABLE = True
except ImportError:
    _TIMESTAMP_UTILS_AVAILABLE = False

__all__ = []
if _TRADING_DAYS_AVAILABLE:
    __all__.extend([
        "get_trading_days",
        "is_trading_day",
        "get_trading_days_set",
        "get_trading_days_lookback",
        "get_trading_days_lookback_datetime",
    ])
if _TIMESTAMP_UTILS_AVAILABLE:
    __all__.extend([
        "normalize_timestamp",
        "normalize_unix_timestamp",
        "to_et_timestamp",
        "normalize_to_date",
        "normalize_timestamp_for_comparison",
    ])

