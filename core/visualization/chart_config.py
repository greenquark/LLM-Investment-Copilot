"""
Chart configuration for strategy-specific visualizations.

This module provides configuration classes that define how charts should be
rendered for different strategies. Each strategy can have its own chart config
that specifies which indicators, panels, and signals to display.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Type, Literal
from enum import Enum


class IndicatorType(Enum):
    """Types of indicator data that can be displayed."""
    MYSTIC_PULSE = "mystic_pulse"  # IndicatorData (positive_count, negative_count)
    LEVERAGED_ETF = "leveraged_etf"  # LeveragedETFIndicatorData (RSI, BB, ATR)
    LLM_TREND = "llm_trend"  # LLMTrendIndicatorData (RSI, BB, MA, regime)
    TREND_REGIME = "trend_regime"  # Regime history (for LLM trend detection)
    BASIC_RSI = "basic_rsi"  # Basic RSI calculated from bars
    NONE = "none"  # No indicator panel


@dataclass
class ChartConfig:
    """
    Base chart configuration.
    
    This class defines how a chart should be rendered, including which panels
    to show, what indicators to display, and how to style them.
    """
    
    # Strategy identifier
    strategy_name: str
    
    # Indicator configuration
    indicator_type: IndicatorType = IndicatorType.NONE
    
    # Panel configuration
    show_price_panel: bool = True
    show_volume_panel: bool = True
    show_indicator_panel: bool = True
    show_equity_panel: bool = True
    
    # Price panel options
    show_bollinger_bands: bool = False  # Show BB on price chart
    show_moving_averages: bool = False  # Show MAs on price chart
    
    # Volume panel options
    show_volume_ma: bool = False  # Show volume moving average
    
    # Signal configuration
    show_signals_on_price: bool = False  # Show buy/sell signals on price chart
    show_signals_on_indicator: bool = False  # Show buy/sell signals on indicator chart
    
    # Indicator panel configuration
    indicator_panel_title: str = "Indicator"
    
    # Row heights (as fractions of total height)
    row_heights: Optional[List[float]] = None  # Auto-calculated if None
    
    def get_row_heights(self) -> List[float]:
        """Calculate row heights based on which panels are shown."""
        if self.row_heights is not None:
            return self.row_heights
        
        panels = []
        if self.show_price_panel:
            panels.append(0.55)
        if self.show_volume_panel:
            panels.append(0.10)
        if self.show_indicator_panel:
            panels.append(0.20)
        if self.show_equity_panel:
            panels.append(0.15)
        
        # Normalize to sum to 1.0
        total = sum(panels)
        return [h / total for h in panels] if total > 0 else [1.0]
    
    def get_subplot_titles(self, symbol: str) -> tuple:
        """Get subplot titles based on which panels are shown."""
        titles = []
        if self.show_price_panel:
            titles.append(f"{symbol} Price")
        if self.show_volume_panel:
            titles.append("Volume")
        if self.show_indicator_panel:
            titles.append(self.indicator_panel_title)
        if self.show_equity_panel:
            titles.append("Equity Curve")
        return tuple(titles)


# Strategy-specific chart configurations

@dataclass
class MysticPulseChartConfig(ChartConfig):
    """Chart configuration for Mystic Pulse (Revised MP2.0) strategy."""
    
    def __init__(self):
        super().__init__(
            strategy_name="mystic_pulse",
            indicator_type=IndicatorType.MYSTIC_PULSE,
            indicator_panel_title="Revised MP2.0 Indicator",
            show_bollinger_bands=False,
            show_moving_averages=False,
            show_volume_ma=False,
            show_signals_on_price=False,  # Signals not shown on price chart
            show_signals_on_indicator=True,  # Signals shown on indicator chart for Mystic Pulse
        )


@dataclass
class LeveragedETFChartConfig(ChartConfig):
    """Chart configuration for Leveraged ETF Volatility Swing strategy."""
    
    def __init__(self):
        super().__init__(
            strategy_name="leveraged_etf",
            indicator_type=IndicatorType.LEVERAGED_ETF,
            indicator_panel_title="RSI & Indicators",
            show_bollinger_bands=True,  # Show BB on price chart
            show_moving_averages=False,
            show_volume_ma=True,  # Show volume MA
            show_signals_on_price=False,  # Signals not shown on price chart
            show_signals_on_indicator=True,  # Signals shown on indicator chart
        )


@dataclass
class LLMTrendChartConfig(ChartConfig):
    """Chart configuration for LLM Trend Detection strategy."""
    
    use_regime_history: bool = False  # If True, use regime_history instead of indicator_data
    
    def __init__(self, use_regime_history: bool = False):
        self.use_regime_history = use_regime_history
        super().__init__(
            strategy_name="llm_trend",
            indicator_type=IndicatorType.TREND_REGIME if use_regime_history else IndicatorType.LLM_TREND,
            indicator_panel_title="Trend Indicator" if use_regime_history else "RSI & Indicators",
            show_bollinger_bands=True,  # Show BB on price chart
            show_moving_averages=False,
            show_volume_ma=False,
            show_signals_on_price=False,  # Signals not shown on price chart
            show_signals_on_indicator=True,  # Signals shown on indicator chart
        )


@dataclass
class AdaptiveDCAChartConfig(ChartConfig):
    """Chart configuration for Adaptive DCA strategy."""
    
    def __init__(self):
        super().__init__(
            strategy_name="adaptive_dca",
            indicator_type=IndicatorType.BASIC_RSI,  # Show basic RSI (no custom indicator)
            indicator_panel_title="RSI",
            show_bollinger_bands=False,
            show_moving_averages=False,
            show_volume_ma=False,
            show_signals_on_price=True,  # Show signals on price chart for AdaptiveDCA
            show_signals_on_indicator=False,  # Signals not shown on indicator chart
        )


@dataclass
class GenericChartConfig(ChartConfig):
    """Generic chart configuration for strategies without specific indicators."""
    
    def __init__(self, strategy_name: str = "generic"):
        super().__init__(
            strategy_name=strategy_name,
            indicator_type=IndicatorType.BASIC_RSI,  # Show basic RSI if no indicator data
            indicator_panel_title="RSI",
            show_bollinger_bands=False,
            show_moving_averages=False,
            show_volume_ma=False,
            show_signals_on_price=False,  # Signals not shown on price chart by default
            show_signals_on_indicator=False,  # Signals not shown on indicator chart by default
        )


# Factory function to get chart config for a strategy
def get_chart_config(strategy_name: str, **kwargs) -> ChartConfig:
    """
    Get chart configuration for a strategy.
    
    Args:
        strategy_name: Name of the strategy (e.g., "mystic_pulse", "leveraged_etf")
        **kwargs: Additional arguments for specific configs (e.g., use_regime_history for LLM trend)
    
    Returns:
        ChartConfig instance for the strategy
    """
    strategy_name_lower = strategy_name.lower()
    
    if strategy_name_lower in ("mystic_pulse", "mysticpulse", "revised_mp2.0", "revised_mp2"):
        return MysticPulseChartConfig()
    elif strategy_name_lower in ("leveraged_etf", "leveragedetf", "volatility_swing"):
        return LeveragedETFChartConfig()
    elif strategy_name_lower in ("llm_trend", "llmtrend", "llm_trend_detection"):
        return LLMTrendChartConfig(use_regime_history=kwargs.get("use_regime_history", False))
    elif strategy_name_lower in ("adaptive_dca", "adaptivedca", "adaptive_dollar_cost_averaging"):
        return AdaptiveDCAChartConfig()
    else:
        return GenericChartConfig(strategy_name=strategy_name)

