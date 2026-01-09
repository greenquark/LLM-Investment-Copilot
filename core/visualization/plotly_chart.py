"""
Plotly-based chart visualization for trading data.

This module provides a modern, interactive charting solution using Plotly,
inspired by professional trading platforms like TradingView and Moomoo.
"""

from __future__ import annotations
from datetime import datetime
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.models.bar import Bar
from core.visualization.models import TradeSignal, IndicatorData, LeveragedETFIndicatorData, LLMTrendIndicatorData
from core.visualization.chart_config import (
    ChartConfig, 
    IndicatorType, 
    get_chart_config,
    GenericChartConfig,
)
from core.visualization.chart_config import (
    ChartConfig, 
    IndicatorType, 
    get_chart_config,
    GenericChartConfig,
)


# ---------- THEME DEFINITIONS ----------

THEMES = {
    "moomoo": {
        "bg_color": "#050608",
        "plot_bg": "#050608",
        "grid_color": "#303030",
        "axis_color": "#aaaaaa",
        "text_color": "#ffffff",
        "candle_up": "#00c08b",
        "candle_down": "#ff4d4f",
        "volume_up": "#00c08b",
        "volume_down": "#ff4d4f",
        "buy_signal": "#00FF66",
        "sell_signal": "#FF1A1A",
        "indicator_pos": "#00c08b",
        "indicator_neg": "#ff4d4f",
        "equity_line": "#FFD700",
        "ma_colors": {
            5:  "#f6c453",
            20: "#00bcd4",
            60: "#b388ff",
            120: "#ff80ab",
            250: "#ffffff",
        },
        "boll_mid": "#ffa726",
        "boll_up": "#66bb6a",
        "boll_low": "#9575cd",
        "rsi_colors": {
            6:  "#f6c453",
            14: "#ff80ff",
            24: "#80d8ff",
        },
    },
    "tradingview": {
        "bg_color": "#131722",
        "plot_bg": "#131722",
        "grid_color": "#363c4e",
        "axis_color": "#a0b0c0",
        "text_color": "#e0e3eb",
        "candle_up": "#26a69a",
        "candle_down": "#ef5350",
        "volume_up": "#26a69a",
        "volume_down": "#ef5350",
        "buy_signal": "#26a69a",
        "sell_signal": "#ef5350",
        "indicator_pos": "#26a69a",
        "indicator_neg": "#ef5350",
        "equity_line": "#ffa726",
        "ma_colors": {
            5:  "#ffeb3b",
            20: "#42a5f5",
            60: "#ab47bc",
            120: "#ef6c00",
            250: "#ffffff",
        },
        "boll_mid": "#ffb74d",
        "boll_up": "#81c784",
        "boll_low": "#ba68c8",
        "rsi_colors": {
            6:  "#ffeb3b",
            14: "#ff80ab",
            24: "#80d8ff",
        },
    },
    "dark": {
        "bg_color": "#111111",
        "plot_bg": "#111111",
        "grid_color": "#333333",
        "axis_color": "#cccccc",
        "text_color": "#eeeeee",
        "candle_up": "#26a69a",
        "candle_down": "#ef5350",
        "volume_up": "#26a69a",
        "volume_down": "#ef5350",
        "buy_signal": "#00FF66",
        "sell_signal": "#FF1A1A",
        "indicator_pos": "#26a69a",
        "indicator_neg": "#ef5350",
        "equity_line": "#FFD700",
        "ma_colors": {
            5:  "#ffeb3b",
            20: "#42a5f5",
            60: "#ab47bc",
            120: "#ef6c00",
            250: "#ffffff",
        },
        "boll_mid": "#ffa726",
        "boll_up": "#66bb6a",
        "boll_low": "#9575cd",
        "rsi_colors": {
            6:  "#ffeb3b",
            14: "#ff80ff",
            24: "#80d8ff",
        },
    },
    "light": {
        "bg_color": "#ffffff",
        "plot_bg": "#ffffff",
        "grid_color": "#e0e0e0",
        "axis_color": "#555555",
        "text_color": "#222222",
        "candle_up": "#00897b",
        "candle_down": "#e53935",
        "volume_up": "#00897b",
        "volume_down": "#e53935",
        "buy_signal": "#00FF66",
        "sell_signal": "#FF1A1A",
        "indicator_pos": "#26a69a",
        "indicator_neg": "#ef5350",
        "equity_line": "#FFD700",
        "ma_colors": {
            5:  "#f9a825",
            20: "#0277bd",
            60: "#7b1fa2",
            120: "#d84315",
            250: "#000000",
        },
        "boll_mid": "#fb8c00",
        "boll_up": "#2e7d32",
        "boll_low": "#5e35b1",
        "rsi_colors": {
            6:  "#f9a825",
            14: "#d81b60",
            24: "#0277bd",
        },
    },
}


def get_theme(theme_name: str) -> Dict[str, str]:
    """Get theme configuration by name."""
    return THEMES.get(theme_name, THEMES["tradingview"])


# ---------- Data Conversion ----------

def bars_to_dataframe(bars: List[Bar]) -> pd.DataFrame:
    """Convert list of Bar objects to pandas DataFrame."""
    data = {
        "Open": [b.open for b in bars],
        "High": [b.high for b in bars],
        "Low": [b.low for b in bars],
        "Close": [b.close for b in bars],
        "Volume": [b.volume for b in bars],
    }
    df = pd.DataFrame(data, index=[b.timestamp for b in bars])
    df = df.sort_index()
    return df


def add_indicators(
    df: pd.DataFrame,
    ma_list: List[int],
    boll_len: Optional[int],
    boll_std: Optional[float],
    rsi_list: List[int],
) -> pd.DataFrame:
    """
    Add technical indicators to DataFrame.
    
    Args:
        df: DataFrame with OHLCV data
        ma_list: List of moving average periods (e.g., [5, 20, 60])
        boll_len: Bollinger Band length (e.g., 20)
        boll_std: Bollinger Band standard deviation multiplier (e.g., 2.0)
        rsi_list: List of RSI periods (e.g., [6, 14, 24])
    
    Returns:
        DataFrame with indicators added
    """
    # Moving Averages
    for length in ma_list:
        if length > 0:
            df[f"MA{length}"] = df["Close"].rolling(length).mean()
    
    # Bollinger Bands
    if boll_len and boll_len > 0 and boll_std and boll_std > 0:
        mid = df["Close"].rolling(boll_len).mean()
        std = df["Close"].rolling(boll_len).std()  # Uses ddof=1 (sample std) by default
        df["BOLL_MID"] = mid
        df["BOLL_UPPER"] = mid + std * boll_std
        df["BOLL_LOWER"] = mid - std * boll_std
    
    # RSI
    for length in rsi_list:
        if length <= 0:
            continue
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(length).mean()
        avg_loss = loss.rolling(length).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df[f"RSI{length}"] = 100 - (100 / (1 + rs))
    
    return df


def signals_to_dataframe(signals: List[TradeSignal]) -> pd.DataFrame:
    """Convert list of TradeSignal objects to pandas DataFrame."""
    if not signals:
        return pd.DataFrame()
    
    data = {
        "timestamp": [s.timestamp for s in signals],
        "price": [s.price for s in signals],
        "side": [s.side for s in signals],
        "trend_score": [s.trend_score for s in signals],
    }
    
    return pd.DataFrame(data)


def indicator_to_dataframe(indicator_data: List) -> pd.DataFrame:
    """Convert list of IndicatorData, LeveragedETFIndicatorData, or LLMTrendIndicatorData objects to pandas DataFrame."""
    if not indicator_data:
        return pd.DataFrame()
    
    # Detect type by checking first element
    first_item = indicator_data[0]
    
    if isinstance(first_item, LeveragedETFIndicatorData):
        # Leveraged ETF indicator data
        data = {
            "timestamp": [ind.timestamp for ind in indicator_data],
            "price": [ind.price for ind in indicator_data],
            "bb_upper": [ind.bb_upper for ind in indicator_data],
            "bb_middle": [ind.bb_middle for ind in indicator_data],
            "bb_lower": [ind.bb_lower for ind in indicator_data],
            "atr": [ind.atr for ind in indicator_data],
            "rsi_fast": [ind.rsi_fast for ind in indicator_data],
            "rsi_slow": [ind.rsi_slow for ind in indicator_data],
            "volume": [ind.volume for ind in indicator_data],
            "volume_ma": [ind.volume_ma for ind in indicator_data],
            "regime": [ind.regime for ind in indicator_data],
            "entry_setup_detected": [ind.entry_setup_detected for ind in indicator_data],
        }
    elif isinstance(first_item, LLMTrendIndicatorData):
        # LLM Trend Detection indicator data
        data = {
            "timestamp": [ind.timestamp for ind in indicator_data],
            "price": [ind.price for ind in indicator_data],
            "bb_upper": [ind.bb_upper for ind in indicator_data],
            "bb_middle": [ind.bb_middle for ind in indicator_data],
            "bb_lower": [ind.bb_lower for ind in indicator_data],
            "rsi": [ind.rsi for ind in indicator_data],
            "ma_short": [ind.ma_short for ind in indicator_data],
            "ma_medium": [ind.ma_medium for ind in indicator_data],
            "ma_long": [ind.ma_long for ind in indicator_data],
            "regime": [ind.regime for ind in indicator_data],
            "trend_strength": [ind.trend_strength for ind in indicator_data],
            "range_strength": [ind.range_strength for ind in indicator_data],
        }
    else:
        # Original MP2.0 indicator data
        data = {
            "timestamp": [ind.timestamp for ind in indicator_data],
            "positive_count": [ind.positive_count for ind in indicator_data],
            "negative_count": [-ind.negative_count for ind in indicator_data],  # Negative for display
            "trend_score": [ind.trend_score for ind in indicator_data],
        }
    
    return pd.DataFrame(data)


# ---------- Chart Builder ----------

class PlotlyChartVisualizer:
    """
    Plotly-based interactive chart visualizer for trading data.
    
    Features:
    - Multi-panel charts (Price, Volume, Indicators, Equity)
    - Bollinger Bands and RSI support
    - Trading signals (Buy/Sell markers)
    - Multiple themes
    """
    
    def __init__(self, theme: str = "tradingview", figsize: tuple = (1400, 900)):
        """
        Initialize the Plotly chart visualizer.
        
        Args:
            theme: Theme name ("moomoo", "tradingview", "dark", "light")
            figsize: Figure size (width, height) in pixels
        """
        self.theme_name = theme
        self.theme = get_theme(theme)
        self.figsize = figsize
        self.fig: Optional[go.Figure] = None
    
    def build_chart(
        self,
        bars: List[Bar],
        signals: Optional[List[TradeSignal]] = None,
        indicator_data: Optional[List] = None,  # Can be List[IndicatorData] or List[LeveragedETFIndicatorData] or List[LLMTrendIndicatorData]
        equity_curve: Optional[Dict[datetime, float]] = None,
        metrics: Optional[Dict] = None,
        symbol: str = "UNKNOWN",
        show_equity: bool = True,
        regime_history: Optional[List[Dict[str, Any]]] = None,  # For trend indicator
        chart_config: Optional[ChartConfig] = None,  # Strategy-specific chart configuration
        strategy_name: Optional[str] = None,  # Auto-detect config if chart_config not provided
    ) -> go.Figure:
        """
        Build a complete multi-panel chart.
        
        Args:
            bars: List of price bars
            signals: Optional list of trading signals
            indicator_data: Optional list of indicator data points
            equity_curve: Optional equity curve data
            metrics: Optional performance metrics dictionary
            symbol: Stock symbol
            show_equity: Whether to show equity curve panel
            regime_history: Optional regime history for LLM trend indicator
            chart_config: Strategy-specific chart configuration (if None, will auto-detect)
            strategy_name: Strategy name for auto-detection (used if chart_config is None)
        
        Returns:
            Plotly Figure object
        """
        if not bars:
            raise ValueError("No bars provided for charting")
        
        # Get or auto-detect chart configuration
        if chart_config is None:
            # Auto-detect based on indicator_data type or strategy_name
            if strategy_name:
                chart_config = get_chart_config(strategy_name, use_regime_history=regime_history is not None)
            elif indicator_data and len(indicator_data) > 0:
                # Auto-detect from indicator data type
                if isinstance(indicator_data[0], LeveragedETFIndicatorData):
                    chart_config = get_chart_config("leveraged_etf")
                elif isinstance(indicator_data[0], LLMTrendIndicatorData):
                    chart_config = get_chart_config("llm_trend", use_regime_history=regime_history is not None)
                elif isinstance(indicator_data[0], IndicatorData):
                    chart_config = get_chart_config("mystic_pulse")
                else:
                    chart_config = GenericChartConfig()
            elif regime_history:
                chart_config = get_chart_config("llm_trend", use_regime_history=True)
            else:
                chart_config = GenericChartConfig()
        
        # Override show_equity if provided
        if not show_equity:
            chart_config.show_equity_panel = False
        
        # Convert to DataFrames
        df = bars_to_dataframe(bars)
        df = df.sort_index()
        
        signals_df = signals_to_dataframe(signals or [])
        indicator_df = indicator_to_dataframe(indicator_data or [])
        
        # If no indicator data provided, calculate basic indicators from bars (like fastapi_stockchart)
        # This allows the chart to show MAs, Bollinger Bands, and RSI even without strategy indicator data
        if indicator_df.empty and chart_config.indicator_type == IndicatorType.BASIC_RSI:
            # Default indicator parameters matching fastapi_stockchart defaults
            ma_list = [5, 20, 60, 120, 250]
            boll_len = 20
            boll_std = 2.0
            rsi_list = [14]
            df = add_indicators(df, ma_list, boll_len, boll_std, rsi_list)
        
        # Determine number of rows based on config
        num_rows = sum([
            chart_config.show_price_panel,
            chart_config.show_volume_panel,
            chart_config.show_indicator_panel,
            chart_config.show_equity_panel and equity_curve is not None,
        ])
        
        if num_rows == 0:
            raise ValueError("At least one panel must be enabled in chart_config")
        
        # Get row heights and titles from config
        row_heights = chart_config.get_row_heights()
        subplot_titles = chart_config.get_subplot_titles(symbol)
        
        # Create subplots
        self.fig = make_subplots(
            rows=num_rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=row_heights,
            subplot_titles=subplot_titles,
        )
        
        # Apply theme
        self.fig.update_layout(
            template="plotly_dark" if self.theme_name != "light" else "plotly_white",
            plot_bgcolor=self.theme["plot_bg"],
            paper_bgcolor=self.theme["bg_color"],
            font=dict(color=self.theme["text_color"], size=12),
            width=self.figsize[0],
            height=self.figsize[1],
            hovermode="x unified",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )
        
        # Add panels based on config
        current_row = 1
        
        # Row 1: Price Chart
        if chart_config.show_price_panel:
            indicator_df_for_price = indicator_df if chart_config.show_bollinger_bands else None
            self._add_price_chart(
                df, 
                signals_df, 
                row=current_row, 
                indicator_df=indicator_df_for_price,
                show_signals=chart_config.show_signals_on_price,
            )
            current_row += 1
        
        # Row 2: Volume
        if chart_config.show_volume_panel:
            indicator_df_for_volume = indicator_df if chart_config.show_volume_ma else None
            self._add_volume_chart(df, row=current_row, indicator_df=indicator_df_for_volume)
            current_row += 1
        
        # Row 3: Indicator
        if chart_config.show_indicator_panel:
            show_signals = chart_config.show_signals_on_indicator
            if chart_config.indicator_type == IndicatorType.TREND_REGIME and regime_history:
                self._add_trend_indicator_chart(regime_history, row=current_row, signals_df=signals_df if show_signals else pd.DataFrame(), show_signals=show_signals)
            elif chart_config.indicator_type == IndicatorType.LEVERAGED_ETF:
                self._add_leveraged_etf_indicator_chart(indicator_df, signals_df, row=current_row, show_signals=show_signals)
            elif chart_config.indicator_type == IndicatorType.LLM_TREND:
                self._add_llm_trend_indicator_chart(indicator_df, signals_df, row=current_row, show_signals=show_signals)
            elif chart_config.indicator_type == IndicatorType.BASIC_RSI and any(f"RSI{length}" in df.columns for length in [6, 14, 24]):
                self._add_basic_rsi_chart(df, row=current_row, signals_df=signals_df if show_signals else pd.DataFrame())
            elif chart_config.indicator_type == IndicatorType.MYSTIC_PULSE:
                self._add_indicator_chart(indicator_df, signals_df, row=current_row, show_signals=show_signals)
            else:
                # Fallback: try to determine from data
                if indicator_df.empty and any(f"RSI{length}" in df.columns for length in [6, 14, 24]):
                    self._add_basic_rsi_chart(df, row=current_row, signals_df=signals_df if show_signals else pd.DataFrame())
                elif not indicator_df.empty:
                    self._add_indicator_chart(indicator_df, signals_df, row=current_row, show_signals=show_signals)
            current_row += 1
        
        # Row 4: Equity Curve (if provided)
        if chart_config.show_equity_panel and equity_curve:
            self._add_equity_chart(equity_curve, row=current_row)
        
        # Update layout to match fastapi_stockchart exactly
        self.fig.update_layout(
            margin=dict(l=40, r=40, t=60, b=40),
            xaxis_rangeslider_visible=False,
            dragmode="pan",  # Enable panning mode (horizontal only when y-axes are fixed)
        )
        
        # Build rangebreaks: weekends + missing business days (holidays) - match fastapi_stockchart
        rangebreaks = [dict(bounds=["sat", "mon"])]
        if not df.empty:
            idx_dates = pd.to_datetime(df.index).normalize()
            all_bus_days = pd.date_range(
                start=idx_dates.min(),
                end=idx_dates.max(),
                freq="B",
            )
            present_days = pd.Index(idx_dates.unique())
            missing_days = all_bus_days.difference(present_days)
            if len(missing_days) > 0:
                rangebreaks.append(dict(values=missing_days))
        
        # Update x-axes with gap removal
        self.fig.update_xaxes(
            showgrid=True,
            gridcolor=self.theme["grid_color"],
            linecolor=self.theme["axis_color"],
            tickfont=dict(color=self.theme["axis_color"]),
            rangebreaks=rangebreaks,
        )
        
        # Update y-axes
        # Set fixedrange=True to prevent vertical panning (only horizontal panning allowed)
        # This still allows zooming via scroll wheel, but prevents dragging vertically
        self.fig.update_yaxes(
            showgrid=True,
            gridcolor=self.theme["grid_color"],
            linecolor=self.theme["axis_color"],
            tickfont=dict(color=self.theme["axis_color"]),
            fixedrange=True,  # Prevent vertical panning - only allow horizontal panning
        )
        
        return self.fig
    
    def _add_price_chart(self, df: pd.DataFrame, signals_df: pd.DataFrame, row: int, indicator_df: Optional[pd.DataFrame] = None, show_signals: bool = False):
        """Add price chart with candlesticks, signals, and optionally Bollinger Bands."""
        # Candlesticks - match fastapi_stockchart styling exactly
        self.fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="Candles",
                increasing_line_color=self.theme["candle_up"],
                decreasing_line_color=self.theme["candle_down"],
                increasing_fillcolor=self.theme["candle_up"],
                decreasing_fillcolor=self.theme["candle_down"],
                increasing_line_width=1,
                decreasing_line_width=1,
            ),
            row=row,
            col=1,
        )
        
        # Add Bollinger Bands if indicator_df is provided (LeveragedETFIndicatorData or LLMTrendIndicatorData)
        if indicator_df is not None and not indicator_df.empty and "bb_upper" in list(indicator_df.columns):
            # Normalize timestamps to date-only for better matching (handles timezone/time differences)
            def normalize_to_date(ts):
                """Normalize timestamp to date for comparison."""
                if isinstance(ts, pd.Timestamp):
                    return ts.normalize().date()
                elif isinstance(ts, datetime):
                    return ts.date()
                else:
                    return pd.to_datetime(ts).normalize().date()
            
            # Create a mapping from normalized date to indicator data for faster lookup
            indicator_map = {}
            for _, row_data in indicator_df.iterrows():
                ts = pd.to_datetime(row_data["timestamp"])
                date_key = normalize_to_date(ts)
                indicator_map[date_key] = {
                    "bb_upper": float(row_data["bb_upper"]) if pd.notna(row_data["bb_upper"]) else None,
                    "bb_middle": float(row_data["bb_middle"]) if pd.notna(row_data["bb_middle"]) else None,
                    "bb_lower": float(row_data["bb_lower"]) if pd.notna(row_data["bb_lower"]) else None,
                }
            
            # Align indicator data with price bars by timestamp
            bb_upper = []
            bb_middle = []
            bb_lower = []
            df_index_list = list(df.index)
            
            for ts in df_index_list:
                date_key = normalize_to_date(ts)
                
                # Try exact date match first
                if date_key in indicator_map:
                    bb_upper.append(indicator_map[date_key]["bb_upper"])
                    bb_middle.append(indicator_map[date_key]["bb_middle"])
                    bb_lower.append(indicator_map[date_key]["bb_lower"])
                else:
                    # Fallback to finding closest match (within 1 day) if exact match not found
                    indicator_timestamps = pd.to_datetime(indicator_df["timestamp"])
                    ts_pd = pd.Timestamp(ts)
                    time_diffs = pd.Series(indicator_timestamps - ts_pd).abs()
                    closest_idx = time_diffs.idxmin()
                    
                    if time_diffs[closest_idx] < pd.Timedelta(days=1):
                        bb_upper.append(float(indicator_df.iloc[closest_idx]["bb_upper"]) if pd.notna(indicator_df.iloc[closest_idx]["bb_upper"]) else None)
                        bb_middle.append(float(indicator_df.iloc[closest_idx]["bb_middle"]) if pd.notna(indicator_df.iloc[closest_idx]["bb_middle"]) else None)
                        bb_lower.append(float(indicator_df.iloc[closest_idx]["bb_lower"]) if pd.notna(indicator_df.iloc[closest_idx]["bb_lower"]) else None)
                    else:
                        bb_upper.append(None)
                        bb_middle.append(None)
                        bb_lower.append(None)
            
            # Forward-fill None values to prevent gaps (use last known value)
            last_upper = None
            last_middle = None
            last_lower = None
            for i in range(len(bb_upper)):
                if bb_upper[i] is not None:
                    last_upper = bb_upper[i]
                    last_middle = bb_middle[i]
                    last_lower = bb_lower[i]
                elif last_upper is not None:
                    # Forward-fill with last known value
                    bb_upper[i] = last_upper
                    bb_middle[i] = last_middle
                    bb_lower[i] = last_lower
            
            # Add filled area between upper and lower bands
            df_index_list_reversed = df_index_list[::-1]
            bb_upper_reversed = bb_upper[::-1]
            bb_lower_reversed = bb_lower[::-1]
            
            self.fig.add_trace(
                go.Scatter(
                    x=df_index_list + df_index_list_reversed,
                    y=bb_upper + bb_lower_reversed,
                    fill="toself",
                    fillcolor="rgba(128, 128, 128, 0.1)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="BB Band",
                    showlegend=True,
                    hoverinfo="skip",
                ),
                row=row,
                col=1,
            )
            
            # Add Bollinger Bands lines
            # BB Middle (SMA) - solid line
            self.fig.add_trace(
                go.Scatter(
                    x=df_index_list,
                    y=bb_middle,
                    mode="lines",
                    name="BOLL MID",
                    line=dict(
                        color=self.theme.get("boll_mid", "#ffa726"),
                        width=1.1
                    ),
                    opacity=0.8,
                    hovertemplate="BOLL MID: %{y:.2f}<extra></extra>",
                    connectgaps=True,  # Connect gaps to prevent visual breaks
                ),
                row=row,
                col=1,
            )
            
            # BB Upper - dotted line (match fastapi_stockchart: dash="dot")
            self.fig.add_trace(
                go.Scatter(
                    x=df_index_list,
                    y=bb_upper,
                    mode="lines",
                    name="BOLL UP",
                    line=dict(
                        color=self.theme.get("boll_up", "#66bb6a"),
                        width=1.0,
                        dash="dot"
                    ),
                    opacity=0.7,
                    hovertemplate="BOLL UP: %{y:.2f}<extra></extra>",
                    connectgaps=True,  # Connect gaps to prevent visual breaks
                ),
                row=row,
                col=1,
            )
            
            # BB Lower - dotted line (match fastapi_stockchart: dash="dot")
            self.fig.add_trace(
                go.Scatter(
                    x=df_index_list,
                    y=bb_lower,
                    mode="lines",
                    name="BOLL LOW",
                    line=dict(
                        color=self.theme.get("boll_low", "#9575cd"),
                        width=1.0,
                        dash="dot"
                    ),
                    opacity=0.7,
                    hovertemplate="BOLL LOW: %{y:.2f}<extra></extra>",
                    connectgaps=True,  # Connect gaps to prevent visual breaks
                ),
                row=row,
                col=1,
            )
        
        # Add Moving Averages if calculated in dataframe (match fastapi_stockchart)
        ma_colors = self.theme.get("ma_colors", {})
        for length in sorted(ma_colors.keys()):
            colname = f"MA{length}"
            if colname in df.columns:
                color = ma_colors.get(length)
                line_kwargs = {"width": 1.1}
                if color:
                    line_kwargs["color"] = color
                self.fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[colname],
                        mode="lines",
                        name=f"MA{length}",
                        line=line_kwargs,
                        opacity=0.9,
                    ),
                    row=row,
                    col=1,
                )
        
        # Add trading signals if enabled
        if show_signals and not signals_df.empty:
            self._add_signals_to_chart(df, signals_df, row=row, panel="price")
    
    def _add_volume_chart(self, df: pd.DataFrame, row: int, indicator_df: Optional[pd.DataFrame] = None):
        """Add volume chart with optional volume MA."""
        # Color volume bars by price direction - match fastapi_stockchart exactly
        vol_colors = np.where(
            df["Close"] >= df["Open"],
            self.theme["volume_up"],
            self.theme["volume_down"]
        )
        
        # Use go.Bar - match fastapi_stockchart styling
        self.fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["Volume"],
                name="Volume",
                marker_color=vol_colors,
                showlegend=False,
            ),
            row=row,
            col=1,
        )
        
        # Add volume MA if available (LeveragedETFIndicatorData only)
        if indicator_df is not None and not indicator_df.empty and "volume_ma" in list(indicator_df.columns):
            volume_ma_values = []
            for ts in df.index:
                if isinstance(ts, pd.Timestamp):
                    ts_dt = ts.to_pydatetime()
                else:
                    ts_dt = ts
                
                indicator_timestamps = pd.to_datetime(indicator_df["timestamp"])
                time_diffs = pd.Series(indicator_timestamps - ts_dt).abs()
                closest_idx = time_diffs.idxmin()
                
                if time_diffs[closest_idx] < pd.Timedelta(days=1):
                    vol_ma_val = indicator_df.iloc[closest_idx]["volume_ma"]
                    volume_ma_values.append(float(vol_ma_val) if pd.notna(vol_ma_val) else None)
                else:
                    volume_ma_values.append(None)
            
            self.fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=volume_ma_values,
                    mode="lines",
                    name="Volume MA",
                    line=dict(color=self.theme.get("indicator_line", "#888888"), width=2),
                    opacity=0.8,
                    showlegend=True,
                ),
                row=row,
                col=1,
            )
        
        # Set y-axis to start at 0 for volume
        self.fig.update_yaxes(title_text="Volume", row=row, col=1, rangemode="tozero")
    
    def _add_indicator_chart(self, indicator_df: pd.DataFrame, signals_df: pd.DataFrame, row: int, show_signals: bool = False):
        """Add Revised MP2.0 indicator chart."""
        if indicator_df.empty:
            return
        
        # Positive count bars
        self.fig.add_trace(
            go.Bar(
                x=indicator_df["timestamp"],
                y=indicator_df["positive_count"],
                name="Positive",
                marker_color=self.theme["indicator_pos"],
                opacity=0.7,
            ),
            row=row,
            col=1,
        )
        
        # Negative count bars (already negative in dataframe)
        self.fig.add_trace(
            go.Bar(
                x=indicator_df["timestamp"],
                y=indicator_df["negative_count"],
                name="Negative",
                marker_color=self.theme["indicator_neg"],
                opacity=0.7,
            ),
            row=row,
            col=1,
        )
        
        # Zero line
        if not indicator_df.empty:
            x_min = indicator_df["timestamp"].min()
            x_max = indicator_df["timestamp"].max()
            self.fig.add_trace(
                go.Scatter(
                    x=[x_min, x_max],
                    y=[0, 0],
                    mode="lines",
                    line=dict(dash="dot", width=1, color=self.theme["grid_color"]),
                    showlegend=False,
                ),
                    row=row,
                    col=1,
                )
        
        # Add trading signals if enabled
        if show_signals and not signals_df.empty:
            self._add_signals_to_chart(indicator_df, signals_df, row=row, panel="indicator")
    
    def _add_basic_rsi_chart(self, df: pd.DataFrame, row: int, signals_df: pd.DataFrame = pd.DataFrame()):
        """Add basic RSI chart from calculated indicators (like fastapi_stockchart)."""
        rsi_colors = self.theme.get("rsi_colors", {})
        
        # Add RSI lines for each calculated RSI period
        for length in [6, 14, 24]:
            colname = f"RSI{length}"
            if colname in df.columns:
                color = rsi_colors.get(length, "#ffffff")
                self.fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[colname],
                        mode="lines",
                        name=f"RSI{length}",
                        line=dict(color=color, width=1.5),
                    ),
                    row=row,
                    col=1,
                )
        
        # Add RSI threshold lines (70/30) - match fastapi_stockchart
        if not df.empty:
            x_min = df.index.min()
            x_max = df.index.max()
            for level in (70, 30):
                self.fig.add_trace(
                    go.Scatter(
                        x=[x_min, x_max],
                        y=[level, level],
                        mode="lines",
                        line=dict(dash="dot", width=1, color=self.theme["grid_color"]),
                        showlegend=False,
                    ),
                    row=row,
                    col=1,
                )
        
        # Add trading signals if provided
        if not signals_df.empty:
            self._add_signals_to_chart(df, signals_df, row=row, panel="indicator")
    
    def _add_leveraged_etf_indicator_chart(self, indicator_df: pd.DataFrame, signals_df: pd.DataFrame, row: int, show_signals: bool = False):
        """Add Leveraged ETF indicator chart (RSI)."""
        if indicator_df.empty:
            return
        
        # RSI Fast - use theme colors if available
        rsi_colors = self.theme.get("rsi_colors", {})
        rsi_fast_color = rsi_colors.get(6, "#ff6b6b")  # Default to red if not in theme
        self.fig.add_trace(
            go.Scatter(
                x=indicator_df["timestamp"],
                y=indicator_df["rsi_fast"],
                mode="lines",
                name="RSI Fast",
                line=dict(color=rsi_fast_color, width=1.5),
                opacity=0.8,
            ),
            row=row,
            col=1,
        )
        
        # RSI Slow - use theme colors if available
        rsi_slow_color = rsi_colors.get(14, "#4ecdc4")  # Default to teal if not in theme
        self.fig.add_trace(
            go.Scatter(
                x=indicator_df["timestamp"],
                y=indicator_df["rsi_slow"],
                mode="lines",
                name="RSI Slow",
                line=dict(color=rsi_slow_color, width=1.5),
                opacity=0.8,
            ),
            row=row,
            col=1,
        )
        
        # RSI thresholds (horizontal lines)
        if not indicator_df.empty:
            x_min = indicator_df["timestamp"].min()
            x_max = indicator_df["timestamp"].max()
            
            # RSI fast threshold (typically 10)
            self.fig.add_trace(
                go.Scatter(
                    x=[x_min, x_max],
                    y=[10, 10],
                    mode="lines",
                    name="RSI Fast Threshold",
                    line=dict(dash="dot", width=1, color="#ff6b6b"),
                    opacity=0.5,
                    showlegend=False,
                ),
                row=row,
                col=1,
            )
            
            # RSI slow threshold (typically 45)
            self.fig.add_trace(
                go.Scatter(
                    x=[x_min, x_max],
                    y=[45, 45],
                    mode="lines",
                    name="RSI Slow Threshold",
                    line=dict(dash="dot", width=1, color="#4ecdc4"),
                    opacity=0.5,
                    showlegend=False,
                ),
                row=row,
                col=1,
            )
            
            # Overbought/oversold levels - match fastapi_stockchart (dash="dot")
            self.fig.add_trace(
                go.Scatter(
                    x=[x_min, x_max],
                    y=[70, 70],
                    mode="lines",
                    name="Overbought",
                    line=dict(dash="dot", width=1, color=self.theme["grid_color"]),
                    showlegend=False,
                ),
                row=row,
                col=1,
            )
            self.fig.add_trace(
                go.Scatter(
                    x=[x_min, x_max],
                    y=[30, 30],
                    mode="lines",
                    name="Oversold",
                    line=dict(dash="dot", width=1, color=self.theme["grid_color"]),
                    showlegend=False,
                ),
                row=row,
                col=1,
            )
        
        # Mark entry setups
        if "entry_setup_detected" in indicator_df.columns:
            entry_setups = indicator_df[indicator_df["entry_setup_detected"] == True]
            if not entry_setups.empty:
                self.fig.add_trace(
                    go.Scatter(
                        x=entry_setups["timestamp"],
                        y=entry_setups["rsi_fast"],
                        mode="markers",
                        name="Entry Setup",
                        marker=dict(
                            symbol="star",
                            size=10,
                            color="#ffd700",
                            line=dict(width=1, color="black"),
                        ),
                        hovertemplate="Entry Setup<extra></extra>",
                    ),
                    row=row,
                    col=1,
                )
    
    def _add_llm_trend_indicator_chart(self, indicator_df: pd.DataFrame, signals_df: pd.DataFrame, row: int, show_signals: bool = False):
        """Add LLM Trend Detection indicator chart (RSI)."""
        if indicator_df.empty:
            return
        
        # RSI - use theme colors
        rsi_colors = self.theme.get("rsi_colors", {})
        rsi_color = rsi_colors.get(14, "#4ecdc4")  # Default to teal if not in theme
        if "rsi" in indicator_df.columns:
            self.fig.add_trace(
                go.Scatter(
                    x=indicator_df["timestamp"],
                    y=indicator_df["rsi"],
                    mode="lines",
                    name="RSI",
                    line=dict(color=rsi_color, width=1.5),
                    opacity=0.8,
                ),
                row=row,
                col=1,
            )
            
            # RSI thresholds (horizontal lines) - match fastapi_stockchart
            if not indicator_df.empty:
                x_min = indicator_df["timestamp"].min()
                x_max = indicator_df["timestamp"].max()
                
                # Overbought/oversold levels - match fastapi_stockchart (dash="dot")
                self.fig.add_trace(
                    go.Scatter(
                        x=[x_min, x_max],
                        y=[70, 70],
                        mode="lines",
                        name="Overbought (70)",
                        line=dict(dash="dot", width=1, color=self.theme["grid_color"]),
                        showlegend=False,
                    ),
                    row=row,
                    col=1,
                )
                self.fig.add_trace(
                    go.Scatter(
                        x=[x_min, x_max],
                        y=[30, 30],
                        mode="lines",
                        name="Oversold (30)",
                        line=dict(dash="dot", width=1, color=self.theme["grid_color"]),
                        showlegend=False,
                    ),
                    row=row,
                    col=1,
                )
    
    def _add_trend_indicator_chart(self, regime_history: List[Dict[str, Any]], row: int, signals_df: Optional[pd.DataFrame] = None, show_signals: bool = False):
        """Add LLM trend indicator chart showing regime decisions."""
        if not regime_history:
            return
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(regime_history)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Trend strength bars (positive for TREND_UP, negative for TREND_DOWN)
        # Use trend_strength for UP, range_strength for RANGE, negative trend_strength for DOWN
        trend_up = df[df["regime"] == "TREND_UP"]
        trend_down = df[df["regime"] == "TREND_DOWN"]
        trend_range = df[df["regime"] == "RANGE"]
        
        # Green bars for TREND_UP
        if not trend_up.empty:
            self.fig.add_trace(
                go.Bar(
                    x=trend_up["timestamp"],
                    y=trend_up["trend_strength"] * 100,  # Scale for visibility
                    name="Uptrend Strength",
                    marker_color=self.theme["buy_signal"],  # Green
                    opacity=0.7,
                    hovertemplate="Uptrend: %{y:.0f}<extra></extra>",
                ),
                row=row,
                col=1,
            )
        
        # Red bars for TREND_DOWN
        if not trend_down.empty:
            self.fig.add_trace(
                go.Bar(
                    x=trend_down["timestamp"],
                    y=-(trend_down["trend_strength"] * 100),  # Negative for display
                    name="Downtrend Strength",
                    marker_color=self.theme["sell_signal"],  # Red
                    opacity=0.7,
                    hovertemplate="Downtrend: %{y:.0f}<extra></extra>",
                ),
                row=row,
                col=1,
            )
        
        # Yellow bars for RANGE (Hold)
        if not trend_range.empty:
            self.fig.add_trace(
                go.Bar(
                    x=trend_range["timestamp"],
                    y=[5] * len(trend_range),  # Small bar for visibility
                    name="Range/Hold",
                    marker_color=self.theme["equity_line"],  # Yellow/Gold
                    opacity=0.7,
                    hovertemplate="Range/Hold<extra></extra>",
                ),
                row=row,
                col=1,
            )
        
        # Zero line
        if not df.empty:
            x_min = df["timestamp"].min()
            x_max = df["timestamp"].max()
            self.fig.add_trace(
                go.Scatter(
                    x=[x_min, x_max],
                    y=[0, 0],
                    mode="lines",
                    line=dict(dash="dot", width=1, color=self.theme["grid_color"]),
                    showlegend=False,
                ),
                row=row,
                col=1,
            )
        
        # Add trading signals if enabled
        if show_signals and signals_df is not None and not signals_df.empty:
            self._add_signals_to_chart(df, signals_df, row=row, panel="indicator")
    
    def _add_signals_to_chart(self, df: pd.DataFrame, signals_df: pd.DataFrame, row: int, panel: str = "price"):
        """
        Add trading signal markers (buy/sell) to the chart.
        
        Args:
            df: DataFrame with timestamp index/column for alignment
            signals_df: DataFrame with columns: timestamp, price, side
            row: Subplot row number
            panel: Panel type ("price" or "indicator") - determines y-position
        """
        if signals_df.empty:
            return
        
        # Ensure signals_df has timestamp column
        if "timestamp" not in signals_df.columns:
            return
        
        # Convert timestamp to datetime if needed
        signals_df = signals_df.copy()
        signals_df["timestamp"] = pd.to_datetime(signals_df["timestamp"])
        
        # Get timestamp column from df
        if "timestamp" in df.columns:
            df_timestamps = pd.to_datetime(df["timestamp"])
        elif isinstance(df.index, pd.DatetimeIndex):
            df_timestamps = df.index
        else:
            return  # Can't align signals without timestamps
        
        # Filter buy and sell signals
        buy_signals = signals_df[signals_df["side"].str.upper() == "BUY"]
        sell_signals = signals_df[signals_df["side"].str.upper() == "SELL"]
        
        # Determine y-position based on panel type
        if panel == "price":
            # For price panel, use the signal price
            if not buy_signals.empty:
                buy_y = buy_signals["price"].values
            else:
                buy_y = []
            
            if not sell_signals.empty:
                sell_y = sell_signals["price"].values
            else:
                sell_y = []
        else:
            # For indicator panel, position at top/bottom of indicator range
            if not df.empty:
                if "timestamp" in df.columns:
                    y_min = df.select_dtypes(include=[np.number]).min().min()
                    y_max = df.select_dtypes(include=[np.number]).max().max()
                else:
                    y_min = df.select_dtypes(include=[np.number]).min().min()
                    y_max = df.select_dtypes(include=[np.number]).max().max()
                
                # Position buy signals near top, sell signals near bottom
                buy_y = [y_max * 0.95] * len(buy_signals) if not buy_signals.empty else []
                sell_y = [y_min * 1.05] * len(sell_signals) if not sell_signals.empty else []
            else:
                buy_y = []
                sell_y = []
        
        # Add buy signal markers
        if not buy_signals.empty and len(buy_y) > 0:
            self.fig.add_trace(
                go.Scatter(
                    x=buy_signals["timestamp"],
                    y=buy_y,
                    mode="markers",
                    name="Buy",
                    marker=dict(
                        symbol="triangle-up",
                        size=12,
                        color=self.theme["buy_signal"],
                        line=dict(width=1, color="white"),
                    ),
                    hovertemplate="Buy @ %{y:.2f}<br>%{x}<extra></extra>",
                ),
                row=row,
                col=1,
            )
        
        # Add sell signal markers
        if not sell_signals.empty and len(sell_y) > 0:
            self.fig.add_trace(
                go.Scatter(
                    x=sell_signals["timestamp"],
                    y=sell_y,
                    mode="markers",
                    name="Sell",
                    marker=dict(
                        symbol="triangle-down",
                        size=12,
                        color=self.theme["sell_signal"],
                        line=dict(width=1, color="white"),
                    ),
                    hovertemplate="Sell @ %{y:.2f}<br>%{x}<extra></extra>",
                ),
                row=row,
                col=1,
            )
    
    def _add_equity_chart(self, equity_curve: Dict[datetime, float], row: int):
        """Add equity curve chart."""
        if not equity_curve:
            return
        
        # Sort by timestamp
        sorted_items = sorted(equity_curve.items(), key=lambda x: x[0])
        timestamps = [item[0] for item in sorted_items]
        equity_values = [item[1] for item in sorted_items]
        
        # Convert hex color to rgba for transparency
        def hex_to_rgba(hex_color: str, alpha: float = 0.25) -> str:
            """Convert hex color to rgba string."""
            hex_color = hex_color.lstrip('#')
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            return f"rgba({r},{g},{b},{alpha})"
        
        equity_color = self.theme["equity_line"]
        equity_fill_color = hex_to_rgba(equity_color, alpha=0.25)
        
        self.fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=equity_values,
                mode="lines",
                name="Equity",
                line=dict(color=equity_color, width=2),
                fill="tozeroy",
                fillcolor=equity_fill_color,
            ),
            row=row,
            col=1,
        )
    
    def show(self, renderer: str = "browser"):
        """Display the chart in browser or specified renderer."""
        if self.fig is None:
            raise ValueError("Chart not built. Call build_chart() first.")
        self.fig.show(renderer=renderer)
    
    def to_html(self, filename: Optional[str] = None) -> str:
        """Export chart to HTML file."""
        if self.fig is None:
            raise ValueError("Chart not built. Call build_chart() first.")
        html = self.fig.to_html()
        if filename:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(html)
        return html
    
    def to_image(self, filename: str, format: str = "png", width: Optional[int] = None, height: Optional[int] = None):
        """Export chart to image file."""
        if self.fig is None:
            raise ValueError("Chart not built. Call build_chart() first.")
        width = width or self.figsize[0]
        height = height or self.figsize[1]
        self.fig.write_image(filename, format=format, width=width, height=height)
