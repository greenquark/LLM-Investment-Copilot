from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Dict

@dataclass
class PerformanceMetrics:
    total_return: float
    cagr: float
    volatility: float
    sharpe: float
    max_drawdown: float
    max_drawdown_start: datetime | None
    max_drawdown_end: datetime | None

@dataclass
class BacktestResult:
    equity_curve: Dict[datetime, float]
    metrics: PerformanceMetrics
