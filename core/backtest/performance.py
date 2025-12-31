from __future__ import annotations
from datetime import datetime, date
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

def evaluate_performance_dca(
    equity_curve: Dict[datetime, float],
    total_contributions: float,
    initial_cash: float = 0.0,
) -> PerformanceMetrics:
    """
    Evaluate performance metrics for DCA strategies with ongoing contributions.
    
    Args:
        equity_curve: Dictionary of timestamp -> equity value
        total_contributions: Total amount contributed over the period
        initial_cash: Initial cash at start (default 0.0 for DCA starting from 0)
    
    Returns:
        PerformanceMetrics with DCA-adjusted total return and CAGR
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

    # Total invested = initial cash + total contributions
    total_invested = initial_cash + total_contributions
    
    if total_invested <= 0:
        # No investments made
        return PerformanceMetrics(
            total_return=0.0,
            cagr=0.0,
            volatility=0.0,
            sharpe=0.0,
            max_drawdown=0.0,
            max_drawdown_start=None,
            max_drawdown_end=None,
        )

    # Total Return for DCA: (Final Equity - Total Invested) / Total Invested
    total_return = (end_eq - total_invested) / total_invested

    # CAGR for DCA: Use the full investment period
    # For DCA, CAGR represents the annualized return over the full period
    # Since money was invested over time, this is an approximation
    # More accurate would be Modified Dietz or XIRR, but for simplicity we use:
    # CAGR = (1 + Total Return)^(1/years) - 1
    days = (end_ts - start_ts).days
    if days <= 0:
        cagr = total_return
    else:
        years = days / 365.25
        if years > 0 and total_return > -1.0:  # Avoid negative base for fractional exponent
            # CAGR based on total return over full period
            cagr = (1.0 + total_return) ** (1.0 / years) - 1.0
        else:
            cagr = total_return

    # Volatility: Annualized standard deviation of daily returns
    daily_rets = _daily_returns(equity_curve)
    if len(daily_rets) > 1:
        mean = sum(daily_rets) / len(daily_rets)
        variance = sum((r - mean) ** 2 for r in daily_rets) / (len(daily_rets) - 1)
        vol_daily = math.sqrt(variance)
        vol_ann = vol_daily * math.sqrt(252.0)
    else:
        vol_ann = 0.0

    # Sharpe Ratio
    risk_free_rate = 0.0
    excess_return = cagr - risk_free_rate
    if vol_ann > 0:
        sharpe = excess_return / vol_ann
    else:
        sharpe = 0.0

    # Max Drawdown
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


def calculate_dca_total_return(
    final_cash: float,
    final_shares: float,
    final_price: float,
    total_cash_contributed: float,
    symbol: str,
    end_date: date,
    contribution_frequency: str,
    period_count: int,
    contribution_amount: float,
    initial_cash: float,
    start_date: date,
    strategy_name: str = "DCA",
    print_steps: bool = True,
) -> float:
    """
    Calculate DCA total return using the formula:
    Total Return = (Final Account Value) / (Total Cash Contributed) - 1
    
    Where:
    - Final Account Value = final_cash + (final_shares * final_price)
    - Total Cash Contributed = initial_cash + total_contributions
    
    Args:
        final_cash: Final cash balance in portfolio
        final_shares: Final number of shares held
        final_price: Final price per share at end date
        total_cash_contributed: Total cash contributed (initial + contributions)
        symbol: Stock symbol (for display)
        end_date: End date (for display)
        contribution_frequency: Frequency of contributions (e.g., "weekly", "monthly")
        period_count: Number of contribution periods
        contribution_amount: Amount contributed per period
        initial_cash: Initial cash at start
        start_date: Start date (for display)
        strategy_name: Name of strategy (for display)
        print_steps: Whether to print detailed calculation steps
    
    Returns:
        Total return as a decimal (e.g., 0.15 for 15%)
    """
    # Calculate final account value
    final_account_value = final_cash + (final_shares * final_price)
    
    # Calculate total return
    total_return = (final_account_value / total_cash_contributed) - 1.0 if total_cash_contributed > 0 else 0.0
    
    if print_steps:
        freq_name = contribution_frequency
        period_name = freq_name.rstrip('ly') if freq_name.endswith('ly') else freq_name
        
        print(f"\n=== {strategy_name} Total Return Calculation ===")
        print(f"Step 1: Calculate total {freq_name}")
        print(f"  Start date: {start_date}")
        print(f"  End date: {end_date}")
        print(f"  Calendar days: {(end_date - start_date).days}")
        print(f"  Total {freq_name} periods: {period_count}")
        print(f"")
        print(f"Step 2: Calculate total cash contributed")
        print(f"  Initial cash: ${initial_cash:,.2f}")
        print(f"  Contribution amount per period: ${contribution_amount:,.2f}")
        total_contributions = period_count * contribution_amount
        print(f"  Total contributions: ${total_contributions:,.2f} = {period_count} {freq_name} periods × ${contribution_amount:,.2f}")
        print(f"  Total cash contributed: ${total_cash_contributed:,.2f} = ${initial_cash:,.2f} + ${total_contributions:,.2f}")
        print(f"")
        print(f"Step 3: Get final account value")
        print(f"  Final cash: ${final_cash:,.2f}")
        print(f"  Final shares: {final_shares:,.2f}")
        print(f"  Final price ({symbol} at {end_date}): ${final_price:,.2f}")
        print(f"  Final account value: ${final_account_value:,.2f} = ${final_cash:,.2f} + ({final_shares:,.2f} shares × ${final_price:,.2f})")
        print(f"")
        print(f"Step 4: Calculate Total Return")
        print(f"  Formula: Total Return = (Final Account Value / Total Cash Contributed) - 1")
        print(f"  Total Return = (${final_account_value:,.2f} / ${total_cash_contributed:,.2f}) - 1")
        print(f"  Total Return = {total_return:.6f} = {total_return:.2%}")
    
    return total_return
