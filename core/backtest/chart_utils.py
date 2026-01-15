"""
Common utilities for chart generation in backtest scripts.

This module provides shared functions for creating and displaying charts
across different backtest scripts to reduce code duplication.
"""

from __future__ import annotations
from typing import Optional, Dict, List
from datetime import datetime

from core.models.bar import Bar
from core.visualization.models import TradeSignal
from core.visualization import PlotlyChartVisualizer, LocalChartVisualizer
from core.visualization.chart_config import get_chart_config, ChartConfig


def create_metrics_dict(result) -> Dict:
    """
    Create a standardized metrics dictionary from backtest result.
    
    Args:
        result: BacktestResult object with metrics attribute
        
    Returns:
        Dictionary with standardized metric keys
    """
    return {
        "total_return": result.metrics.total_return,
        "cagr": result.metrics.cagr,
        "volatility": result.metrics.volatility,
        "sharpe": result.metrics.sharpe,
        "max_drawdown": result.metrics.max_drawdown,
    }


def create_plotly_chart(
    bars: List[Bar],
    signals: Optional[List[TradeSignal]] = None,
    indicator_data: Optional[List] = None,
    equity_curve: Optional[Dict[datetime, float]] = None,
    metrics: Optional[Dict] = None,
    symbol: str = "UNKNOWN",
    strategy_name: str = "generic",
    regime_history: Optional[List] = None,
    trade_signals: Optional[List[TradeSignal]] = None,
    chart_config: Optional[ChartConfig] = None,
    theme: str = "tradingview",
    figsize: tuple = (1400, 900),
) -> PlotlyChartVisualizer:
    """
    Create and configure a Plotly chart visualizer with common settings.
    
    This function standardizes chart creation across backtest scripts.
    
    Args:
        bars: List of price bars
        signals: Optional list of trading signals
        indicator_data: Optional list of indicator data points
        equity_curve: Optional equity curve data
        metrics: Optional performance metrics dictionary
        symbol: Stock symbol
        strategy_name: Strategy name for auto-detecting chart config
        regime_history: Optional regime history for LLM trend indicator
        trade_signals: Optional trade entry/exit signals for overlay
        chart_config: Optional chart configuration (auto-detected if None)
        theme: Chart theme name
        figsize: Figure size (width, height) in pixels
        
    Returns:
        Configured PlotlyChartVisualizer instance
        
    Examples:
        >>> visualizer = create_plotly_chart(
        ...     bars=bars,
        ...     signals=signals,
        ...     equity_curve=result.equity_curve,
        ...     metrics=metrics_dict,
        ...     symbol="AAPL",
        ...     strategy_name="llm_trend",
        ... )
        >>> visualizer.show(renderer="browser")
    """
    visualizer = PlotlyChartVisualizer(theme=theme, figsize=figsize)
    
    # Auto-detect chart config if not provided
    if chart_config is None:
        chart_config = get_chart_config(
            strategy_name,
            use_regime_history=regime_history is not None
        )
    
    visualizer.build_chart(
        bars=bars,
        signals=signals,
        indicator_data=indicator_data,
        equity_curve=equity_curve,
        metrics=metrics,
        symbol=symbol,
        show_equity=True,
        regime_history=regime_history,
        chart_config=chart_config,
        strategy_name=strategy_name,
        trade_signals=trade_signals,
    )
    
    return visualizer


def create_local_chart(
    bars: List[Bar],
    signals: Optional[List[TradeSignal]] = None,
    indicator_data: Optional[List] = None,
    equity_curve: Optional[Dict[datetime, float]] = None,
    metrics: Optional[Dict] = None,
    symbol: str = "UNKNOWN",
    style: str = "dark_background",
    figsize: tuple = (16, 12),
) -> LocalChartVisualizer:
    """
    Create and configure a local Matplotlib chart visualizer.
    
    Args:
        bars: List of price bars
        signals: Optional list of trading signals
        indicator_data: Optional list of indicator data points
        equity_curve: Optional equity curve data
        metrics: Optional performance metrics dictionary
        symbol: Stock symbol
        style: Matplotlib style name
        figsize: Figure size (width, height) in inches
        
    Returns:
        Configured LocalChartVisualizer instance
    """
    chart = LocalChartVisualizer(style=style, figsize=figsize)
    chart.set_data(
        bars=bars,
        signals=signals,
        indicator_data=indicator_data,
        equity_curve=equity_curve,
        metrics=metrics,
        symbol=symbol,
    )
    return chart
