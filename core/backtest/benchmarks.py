from __future__ import annotations
from datetime import datetime
from typing import Dict

from core.data.base import DataEngine
from core.backtest.performance import evaluate_performance
from core.backtest.result import BacktestResult

async def run_buy_and_hold(
    data_engine: DataEngine,
    symbol: str,
    start: datetime,
    end: datetime,
    initial_cash: float = 100_000.0,
    timeframe: str = "D",
) -> BacktestResult:
    bars = await data_engine.get_bars(symbol, start, end, timeframe=timeframe)
    if not bars:
        raise RuntimeError("No data for buy-and-hold benchmark")

    first_bar = bars[0]
    entry_price = first_bar.close
    shares = int(initial_cash // entry_price)
    cash_left = initial_cash - shares * entry_price

    equity_curve: Dict[datetime, float] = {}
    for bar in bars:
        equity = cash_left + shares * bar.close
        equity_curve[bar.timestamp] = equity

    metrics = evaluate_performance(equity_curve)
    return BacktestResult(equity_curve=equity_curve, metrics=metrics)
