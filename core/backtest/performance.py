from __future__ import annotations
from datetime import datetime
from typing import Dict, List, Tuple
import math

from core.backtest.result import PerformanceMetrics

def _sorted_equity(equity_curve: Dict[datetime, float]) -> List[Tuple[datetime, float]]:
    return sorted(equity_curve.items(), key=lambda x: x[0])

def _daily_returns(equity_curve: Dict[datetime, float]) -> List[float]:
    """
    Calculate daily returns from equity curve.
    Uses the last value of each day to compute returns.
    """
    by_day: Dict[datetime.date, float] = {}
    for ts, eq in _sorted_equity(equity_curve):
        day = ts.date()
        # Use the last value of the day if multiple entries exist
        by_day[day] = eq

    days_sorted = sorted(by_day.items(), key=lambda x: x[0])
    returns: List[float] = []
    for i in range(1, len(days_sorted)):
        prev_eq = days_sorted[i - 1][1]
        curr_eq = days_sorted[i][1]
        if prev_eq > 0:
            r = (curr_eq / prev_eq) - 1.0
            returns.append(r)
    return returns

def _max_drawdown(equity_curve: Dict[datetime, float]) -> Tuple[float, datetime | None, datetime | None]:
    """
    Calculate maximum drawdown as a positive percentage.
    Returns: (max_drawdown as positive %, start_date, end_date)
    """
    items = _sorted_equity(equity_curve)
    if not items:
        return 0.0, None, None

    peak_eq = items[0][1]
    peak_ts = items[0][0]
    max_dd = 0.0
    dd_start = peak_ts
    dd_end = peak_ts

    for ts, eq in items[1:]:
        if eq > peak_eq:
            peak_eq = eq
            peak_ts = ts
        # Drawdown is negative when equity is below peak
        dd = (eq / peak_eq - 1.0)  # This will be negative
        if dd < max_dd:
            max_dd = dd
            dd_start = peak_ts
            dd_end = ts

    # Return as positive percentage (max_dd is already negative)
    return abs(max_dd), dd_start, dd_end

def evaluate_performance(equity_curve: Dict[datetime, float]) -> PerformanceMetrics:
    """
    Evaluate performance metrics from equity curve.
    
    Returns:
        - total_return: Total return as decimal (e.g., 0.15 = 15%)
        - cagr: Compound Annual Growth Rate as decimal
        - volatility: Annualized volatility (standard deviation of returns) as decimal
        - sharpe: Sharpe ratio (CAGR / volatility, assuming 0% risk-free rate)
        - max_drawdown: Maximum drawdown as positive decimal (e.g., 0.20 = 20%)
    """
    items = _sorted_equity(equity_curve)
    if len(items) < 2:
        return PerformanceMetrics(
            total_return=0.0,
            cagr=0.0,
            volatility=0.0,
            sharpe=0.0,
            max_drawdown=0.0,
            max_drawdown_start=None,
            max_drawdown_end=None,
        )

    start_ts, start_eq = items[0]
    end_ts, end_eq = items[-1]

    # Ensure we have valid equity values
    if start_eq <= 0:
        raise ValueError(f"Invalid starting equity: {start_eq}")
    if end_eq < 0:
        raise ValueError(f"Invalid ending equity: {end_eq}")

    # Total Return: (Final / Initial) - 1
    total_return = (end_eq / start_eq) - 1.0

    # CAGR: Compound Annual Growth Rate
    days = (end_ts - start_ts).days
    if days <= 0:
        cagr = total_return
    else:
        years = days / 365.25
        if years > 0 and end_eq > 0:
            # CAGR = (End/Start)^(1/years) - 1
            cagr = (end_eq / start_eq) ** (1.0 / years) - 1.0
        else:
            cagr = total_return

    # Volatility: Annualized standard deviation of daily returns
    daily_rets = _daily_returns(equity_curve)
    if len(daily_rets) > 1:
        # Calculate sample standard deviation
        mean = sum(daily_rets) / len(daily_rets)
        variance = sum((r - mean) ** 2 for r in daily_rets) / (len(daily_rets) - 1)
        vol_daily = math.sqrt(variance)
        # Annualize: multiply by sqrt(252 trading days per year)
        vol_ann = vol_daily * math.sqrt(252.0)
    else:
        vol_ann = 0.0

    # Sharpe Ratio: (CAGR - RiskFreeRate) / Volatility
    # Using 0% risk-free rate for simplicity
    risk_free_rate = 0.0
    excess_return = cagr - risk_free_rate
    if vol_ann > 0:
        sharpe = excess_return / vol_ann
    else:
        # If no volatility, Sharpe is undefined, return 0
        sharpe = 0.0

    # Max Drawdown: Already returned as positive percentage
    max_dd, dd_start, dd_end = _max_drawdown(equity_curve)

    return PerformanceMetrics(
        total_return=total_return,
        cagr=cagr,
        volatility=vol_ann,
        sharpe=sharpe,
        max_drawdown=max_dd,
        max_drawdown_start=dd_start,
        max_drawdown_end=dd_end,
    )
