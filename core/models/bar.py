from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Bar:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: str  # e.g. "1m", "15m", "D"
