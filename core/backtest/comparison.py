"""
Strategy Comparison Module - Compare multiple strategies and their results.

This module provides utilities for running and comparing multiple strategies
in a backtest scenario.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, date
from typing import Dict, Optional, List, Any

from core.backtest.result import BacktestResult, PerformanceMetrics
from core.backtest.performance import evaluate_performance_dca, calculate_dca_total_return
from core.utils.price_utils import get_final_price


@dataclass
class StrategyComparisonResult:
    """
    Result of comparing two strategies.
    
    Contains results for both strategies and comparison metrics.
    """
    strategy1_name: str
    strategy1_result: BacktestResult
    strategy1_metrics: PerformanceMetrics
    strategy1_total_return: float
    strategy1_final_cash: float
    strategy1_final_shares: float
    strategy1_final_price: float
    strategy1_total_contributions: float
    
    strategy2_name: str
    strategy2_result: BacktestResult
    strategy2_metrics: PerformanceMetrics
    strategy2_total_return: float
    strategy2_final_cash: float
    strategy2_final_shares: float
    strategy2_final_price: float
    strategy2_total_contributions: float
    
    # Comparison metrics
    return_difference: float  # strategy1_total_return - strategy2_total_return
    cagr_difference: float  # strategy1_cagr - strategy2_cagr
    sharpe_difference: float  # strategy1_sharpe - strategy2_sharpe


async def compare_strategies(
    strategy1_name: str,
    strategy1_result: BacktestResult,
    strategy1_final_cash: float,
    strategy1_final_shares: float,
    strategy1_final_price: float,
    strategy1_total_contributions: float,
    strategy1_total_return: float,
    strategy1_metrics: PerformanceMetrics,
    strategy2_name: str,
    strategy2_result: BacktestResult,
    strategy2_final_cash: float,
    strategy2_final_shares: float,
    strategy2_final_price: float,
    strategy2_total_contributions: float,
    strategy2_total_return: float,
    strategy2_metrics: PerformanceMetrics,
) -> StrategyComparisonResult:
    """
    Compare two strategy results and return comparison metrics.
    
    Args:
        strategy1_name: Name of first strategy
        strategy1_result: Backtest result for first strategy
        strategy1_final_cash: Final cash for first strategy
        strategy1_final_shares: Final shares for first strategy
        strategy1_final_price: Final price used for first strategy
        strategy1_total_contributions: Total contributions for first strategy
        strategy1_total_return: Total return for first strategy
        strategy1_metrics: Performance metrics for first strategy
        strategy2_name: Name of second strategy
        strategy2_result: Backtest result for second strategy
        strategy2_final_cash: Final cash for second strategy
        strategy2_final_shares: Final shares for second strategy
        strategy2_final_price: Final price used for second strategy
        strategy2_total_contributions: Total contributions for second strategy
        strategy2_total_return: Total return for second strategy
        strategy2_metrics: Performance metrics for second strategy
    
    Returns:
        StrategyComparisonResult with comparison metrics
    """
    return_difference = strategy1_total_return - strategy2_total_return
    cagr_difference = strategy1_metrics.cagr - strategy2_metrics.cagr
    sharpe_difference = strategy1_metrics.sharpe - strategy2_metrics.sharpe
    
    return StrategyComparisonResult(
        strategy1_name=strategy1_name,
        strategy1_result=strategy1_result,
        strategy1_metrics=strategy1_metrics,
        strategy1_total_return=strategy1_total_return,
        strategy1_final_cash=strategy1_final_cash,
        strategy1_final_shares=strategy1_final_shares,
        strategy1_final_price=strategy1_final_price,
        strategy1_total_contributions=strategy1_total_contributions,
        strategy2_name=strategy2_name,
        strategy2_result=strategy2_result,
        strategy2_metrics=strategy2_metrics,
        strategy2_total_return=strategy2_total_return,
        strategy2_final_cash=strategy2_final_cash,
        strategy2_final_shares=strategy2_final_shares,
        strategy2_final_price=strategy2_final_price,
        strategy2_total_contributions=strategy2_total_contributions,
        return_difference=return_difference,
        cagr_difference=cagr_difference,
        sharpe_difference=sharpe_difference,
    )


def print_comparison_table(comparison: StrategyComparisonResult) -> None:
    """
    Print a formatted comparison table for two strategies.
    
    Args:
        comparison: StrategyComparisonResult to print
    """
    print(f"\n{'='*80}")
    print(f"{'Strategy Comparison':^80}")
    print(f"{'='*80}")
    print(f"{'Metric':<40} {comparison.strategy1_name:>18} {comparison.strategy2_name:>18}")
    print(f"{'-'*80}")
    print(f"{'Total Return':<40} {comparison.strategy1_total_return:>18.2%} {comparison.strategy2_total_return:>18.2%}")
    print(f"{'CAGR':<40} {comparison.strategy1_metrics.cagr:>18.2%} {comparison.strategy2_metrics.cagr:>18.2%}")
    print(f"{'Volatility':<40} {comparison.strategy1_metrics.volatility:>18.2%} {comparison.strategy2_metrics.volatility:>18.2%}")
    print(f"{'Sharpe Ratio':<40} {comparison.strategy1_metrics.sharpe:>18.2f} {comparison.strategy2_metrics.sharpe:>18.2f}")
    print(f"{'Max Drawdown':<40} {comparison.strategy1_metrics.max_drawdown:>18.2%} {comparison.strategy2_metrics.max_drawdown:>18.2%}")
    print(f"{'-'*80}")
    print(f"{'Final Cash':<40} ${comparison.strategy1_final_cash:>17,.2f} ${comparison.strategy2_final_cash:>17,.2f}")
    print(f"{'Final Shares':<40} {comparison.strategy1_final_shares:>18,.4f} {comparison.strategy2_final_shares:>18,.4f}")
    print(f"{'Final Price':<40} ${comparison.strategy1_final_price:>17,.2f} ${comparison.strategy2_final_price:>17,.2f}")
    print(f"{'Total Contributions':<40} ${comparison.strategy1_total_contributions:>17,.2f} ${comparison.strategy2_total_contributions:>17,.2f}")
    print(f"{'='*80}")
    print(f"\nDifference ({comparison.strategy1_name} - {comparison.strategy2_name}):")
    print(f"  Total Return: {comparison.return_difference:+.2%}")
    print(f"  CAGR: {comparison.cagr_difference:+.2%}")
    print(f"  Sharpe Ratio: {comparison.sharpe_difference:+.2f}")

