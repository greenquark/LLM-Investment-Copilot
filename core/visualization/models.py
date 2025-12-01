from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class TradeSignal:
    """Represents a buy or sell signal."""
    timestamp: datetime
    price: float
    side: str  # "BUY" or "SELL"
    trend_score: int
    di_plus: Optional[float]
    di_minus: Optional[float]


@dataclass
class IndicatorData:
    """Data for the Revised MP2.0 indicator."""
    timestamp: datetime
    positive_count: int
    negative_count: int
    trend_score: int
    di_plus: Optional[float]
    di_minus: Optional[float]


@dataclass
class LeveragedETFIndicatorData:
    """Indicator data for Leveraged ETF Volatility Swing strategy."""
    timestamp: datetime
    price: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    atr: float
    rsi_fast: float
    rsi_slow: float
    volume: float
    volume_ma: float
    regime: str  # "bull", "bear", "neutral"
    entry_setup_detected: bool


@dataclass
class LLMTrendIndicatorData:
    """Indicator data for LLM Trend Detection strategy."""
    timestamp: datetime
    price: float
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    rsi: Optional[float] = None
    ma_short: Optional[float] = None
    ma_medium: Optional[float] = None
    ma_long: Optional[float] = None
    regime: Optional[str] = None  # "TREND_UP", "TREND_DOWN", "RANGE"
    trend_strength: float = 0.0
    range_strength: float = 0.0

