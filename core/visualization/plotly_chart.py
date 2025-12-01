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


# ---------- THEME DEFINITIONS ----------

THEMES = {
    "moomoo": {
        "bg_color": "#050608",
        "plot_bg": "#050608",
        "grid_color": "#303030",
        "axis_color": "#aaaaaa",
        "text_color": "#ffffff",
        "candle_up": "#00c08b",      # teal
        "candle_down": "#ff4d4f",    # red
        "volume_up": "#00c08b",
        "volume_down": "#ff4d4f",
        "buy_signal": "#00FF66",     # green
        "sell_signal": "#FF1A1A",    # red
        "indicator_pos": "#00c08b",  # teal for positive
        "indicator_neg": "#ff4d4f",  # red for negative
        "equity_line": "#FFD700",    # gold
        "boll_mid": "#ffa726",       # Bollinger Middle Band
        "boll_up": "#66bb6a",        # Bollinger Upper Band
        "boll_low": "#9575cd",       # Bollinger Lower Band
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
        "boll_mid": "#ffb74d",       # Bollinger Middle Band
        "boll_up": "#81c784",        # Bollinger Upper Band
        "boll_low": "#ba68c8",       # Bollinger Lower Band
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
        "boll_mid": "#ffa726",       # Bollinger Middle Band
        "boll_up": "#66bb6a",        # Bollinger Upper Band
        "boll_low": "#9575cd",       # Bollinger Lower Band
    },
    "light": {
        "bg_color": "#ffffff",
        "plot_bg": "#ffffff",
        "grid_color": "#e0e0e0",
        "axis_color": "#666666",
        "text_color": "#000000",
        "candle_up": "#26a69a",
        "candle_down": "#ef5350",
        "volume_up": "#26a69a",
        "volume_down": "#ef5350",
        "buy_signal": "#00FF66",
        "sell_signal": "#FF1A1A",
        "indicator_pos": "#26a69a",
        "indicator_neg": "#ef5350",
        "equity_line": "#FFD700",
        "boll_mid": "#ffa726",       # Bollinger Middle Band
        "boll_up": "#66bb6a",        # Bollinger Upper Band
        "boll_low": "#9575cd",       # Bollinger Lower Band
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
        
        Returns:
            Plotly Figure object
        """
        if not bars:
            raise ValueError("No bars provided for charting")
        
        # Convert to DataFrames
        df = bars_to_dataframe(bars)
        df = df.sort_index()
        
        signals_df = signals_to_dataframe(signals or [])
        indicator_df = indicator_to_dataframe(indicator_data or [])
        
        # Detect indicator type
        is_leveraged_etf_indicator = (
            indicator_data 
            and len(indicator_data) > 0 
            and isinstance(indicator_data[0], LeveragedETFIndicatorData)
        )
        is_llm_trend_indicator = (
            indicator_data 
            and len(indicator_data) > 0 
            and isinstance(indicator_data[0], LLMTrendIndicatorData)
        )
        
        # Determine number of rows
        num_rows = 3  # Price, Volume, Indicator
        if show_equity and equity_curve:
            num_rows = 4
        
        # Row heights: Price (60%), Volume (10%), Indicator (20%), Equity (10% if shown)
        if num_rows == 4:
            row_heights = [0.55, 0.10, 0.20, 0.15]
        else:
            row_heights = [0.60, 0.10, 0.30]
        
        # Determine indicator panel title
        if regime_history:
            indicator_title = "Trend Indicator"
        elif is_leveraged_etf_indicator:
            indicator_title = "RSI & Indicators"
        elif is_llm_trend_indicator:
            indicator_title = "RSI & Indicators"
        else:
            indicator_title = "Revised MP2.0 Indicator"
        
        # Create subplots
        self.fig = make_subplots(
            rows=num_rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=row_heights,
            subplot_titles=(
                f"{symbol} Price",
                "Volume",
                indicator_title,
                "Equity Curve" if show_equity and equity_curve else None,
            ),
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
        
        # Row 1: Price Chart
        # For LeveragedETFIndicatorData or LLMTrendIndicatorData, also add Bollinger Bands
        has_bb_data = is_leveraged_etf_indicator or is_llm_trend_indicator
        self._add_price_chart(df, signals_df, row=1, indicator_df=indicator_df if has_bb_data else None)
        
        # Row 2: Volume
        # For LeveragedETFIndicatorData, also add volume MA
        self._add_volume_chart(df, row=2, indicator_df=indicator_df if is_leveraged_etf_indicator else None)
        
        # Row 3: Indicator
        if regime_history:
            self._add_trend_indicator_chart(regime_history, row=3)
        elif is_leveraged_etf_indicator:
            self._add_leveraged_etf_indicator_chart(indicator_df, signals_df, row=3)
        elif is_llm_trend_indicator:
            self._add_llm_trend_indicator_chart(indicator_df, signals_df, row=3)
        else:
            self._add_indicator_chart(indicator_df, signals_df, row=3)
        
        # Row 4: Equity Curve (if provided)
        if show_equity and equity_curve and num_rows == 4:
            self._add_equity_chart(equity_curve, row=4)
        
        # Update layout
        self.fig.update_xaxes(
            showgrid=True,
            gridcolor=self.theme["grid_color"],
            gridwidth=1,
            showspikes=True,
            spikecolor=self.theme["grid_color"],
            spikethickness=1,
        )
        self.fig.update_yaxes(
            showgrid=True,
            gridcolor=self.theme["grid_color"],
            gridwidth=1,
        )
        
        return self.fig
    
    def _add_price_chart(self, df: pd.DataFrame, signals_df: pd.DataFrame, row: int, indicator_df: Optional[pd.DataFrame] = None):
        """Add price chart with candlesticks, signals, and optionally Bollinger Bands."""
        # Candlesticks
        self.fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="Price",
                increasing_line_color=self.theme["candle_up"],
                decreasing_line_color=self.theme["candle_down"],
                increasing_fillcolor=self.theme["candle_up"],
                decreasing_fillcolor=self.theme["candle_down"],
            ),
            row=row,
            col=1,
        )
        
        # Add Bollinger Bands if indicator_df is provided (LeveragedETFIndicatorData or LLMTrendIndicatorData)
        if indicator_df is not None and not indicator_df.empty and "bb_upper" in list(indicator_df.columns):
            # Create a mapping from timestamp to indicator data for faster lookup
            indicator_map = {}
            for _, row_data in indicator_df.iterrows():
                ts = pd.to_datetime(row_data["timestamp"])
                indicator_map[ts] = {
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
                # Convert to pandas Timestamp for comparison
                ts_key = pd.Timestamp(ts)
                
                # Try exact match first
                if ts_key in indicator_map:
                    bb_upper.append(indicator_map[ts_key]["bb_upper"])
                    bb_middle.append(indicator_map[ts_key]["bb_middle"])
                    bb_lower.append(indicator_map[ts_key]["bb_lower"])
                else:
                    # Fallback to finding closest match (within 1 day) if exact match not found
                    indicator_timestamps = pd.to_datetime(indicator_df["timestamp"])
                    time_diffs = pd.Series(indicator_timestamps - ts_key).abs()
                    closest_idx = time_diffs.idxmin()
                    
                    if time_diffs[closest_idx] < pd.Timedelta(days=1):
                        bb_upper.append(float(indicator_df.iloc[closest_idx]["bb_upper"]) if pd.notna(indicator_df.iloc[closest_idx]["bb_upper"]) else None)
                        bb_middle.append(float(indicator_df.iloc[closest_idx]["bb_middle"]) if pd.notna(indicator_df.iloc[closest_idx]["bb_middle"]) else None)
                        bb_lower.append(float(indicator_df.iloc[closest_idx]["bb_lower"]) if pd.notna(indicator_df.iloc[closest_idx]["bb_lower"]) else None)
                    else:
                        bb_upper.append(None)
                        bb_middle.append(None)
                        bb_lower.append(None)
            
            # Use the most recent BB value if the last bar doesn't have one
            if len(bb_upper) > 0 and bb_upper[-1] is None:
                for i in range(len(bb_upper) - 2, -1, -1):
                    if bb_upper[i] is not None:
                        bb_upper[-1] = bb_upper[i]
                        bb_middle[-1] = bb_middle[i]
                        bb_lower[-1] = bb_lower[i]
                        break
            
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
                    connectgaps=False,
                ),
                row=row,
                col=1,
            )
            
            # BB Upper - dotted line
            self.fig.add_trace(
                go.Scatter(
                    x=df_index_list,
                    y=bb_upper,
                    mode="lines",
                    name="BOLL UP",
                    line=dict(
                        color=self.theme.get("boll_up", "#66bb6a"),
                        width=1.0,
                        dash="dash"
                    ),
                    opacity=0.7,
                    hovertemplate="BOLL UP: %{y:.2f}<extra></extra>",
                    connectgaps=False,
                ),
                row=row,
                col=1,
            )
            
            # BB Lower - dotted line
            self.fig.add_trace(
                go.Scatter(
                    x=df_index_list,
                    y=bb_lower,
                    mode="lines",
                    name="BOLL LOW",
                    line=dict(
                        color=self.theme.get("boll_low", "#9575cd"),
                        width=1.0,
                        dash="dash"
                    ),
                    opacity=0.7,
                    hovertemplate="BOLL LOW: %{y:.2f}<extra></extra>",
                    connectgaps=False,
                ),
                row=row,
                col=1,
            )
        
        # Trading signals removed from price chart as requested
    
    def _add_volume_chart(self, df: pd.DataFrame, row: int, indicator_df: Optional[pd.DataFrame] = None):
        """Add volume chart with optional volume MA."""
        # Color volume bars by price direction
        # Create a list of colors, one for each bar
        colors = []
        for i in range(len(df)):
            if df["Close"].iloc[i] >= df["Open"].iloc[i]:
                colors.append(self.theme["volume_up"])
            else:
                colors.append(self.theme["volume_down"])
        
        # Use go.Bar with proper marker configuration
        self.fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["Volume"],
                name="Volume",
                marker=dict(
                    color=colors,
                    line=dict(width=0),
                ),
                opacity=0.7,
                showlegend=True,
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
    
    def _add_indicator_chart(self, indicator_df: pd.DataFrame, signals_df: pd.DataFrame, row: int):
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
    
    def _add_leveraged_etf_indicator_chart(self, indicator_df: pd.DataFrame, signals_df: pd.DataFrame, row: int):
        """Add Leveraged ETF indicator chart (RSI)."""
        if indicator_df.empty:
            return
        
        # RSI Fast
        self.fig.add_trace(
            go.Scatter(
                x=indicator_df["timestamp"],
                y=indicator_df["rsi_fast"],
                mode="lines",
                name="RSI Fast",
                line=dict(color="#ff6b6b", width=1.5),
                opacity=0.8,
            ),
            row=row,
            col=1,
        )
        
        # RSI Slow
        self.fig.add_trace(
            go.Scatter(
                x=indicator_df["timestamp"],
                y=indicator_df["rsi_slow"],
                mode="lines",
                name="RSI Slow",
                line=dict(color="#4ecdc4", width=1.5),
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
            
            # Overbought/oversold levels
            self.fig.add_trace(
                go.Scatter(
                    x=[x_min, x_max],
                    y=[70, 70],
                    mode="lines",
                    name="Overbought",
                    line=dict(dash="dash", width=1, color="#888888"),
                    opacity=0.3,
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
                    line=dict(dash="dash", width=1, color="#888888"),
                    opacity=0.3,
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
    
    def _add_llm_trend_indicator_chart(self, indicator_df: pd.DataFrame, signals_df: pd.DataFrame, row: int):
        """Add LLM Trend Detection indicator chart (RSI)."""
        if indicator_df.empty:
            return
        
        # RSI
        if "rsi" in indicator_df.columns:
            self.fig.add_trace(
                go.Scatter(
                    x=indicator_df["timestamp"],
                    y=indicator_df["rsi"],
                    mode="lines",
                    name="RSI",
                    line=dict(color="#4ecdc4", width=1.5),
                    opacity=0.8,
                ),
                row=row,
                col=1,
            )
            
            # RSI thresholds (horizontal lines)
            if not indicator_df.empty:
                x_min = indicator_df["timestamp"].min()
                x_max = indicator_df["timestamp"].max()
                
                # Overbought/oversold levels
                self.fig.add_trace(
                    go.Scatter(
                        x=[x_min, x_max],
                        y=[70, 70],
                        mode="lines",
                        name="Overbought (70)",
                        line=dict(dash="dash", width=1, color="#888888"),
                        opacity=0.3,
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
                        line=dict(dash="dash", width=1, color="#888888"),
                        opacity=0.3,
                        showlegend=False,
                    ),
                    row=row,
                    col=1,
                )
    
    def _add_trend_indicator_chart(self, regime_history: List[Dict[str, Any]], row: int):
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
