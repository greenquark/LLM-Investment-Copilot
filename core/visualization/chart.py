"""
Plotly-based chart visualization for trading data.

This module provides a standalone chart builder similar to fastapi_stockchart/main.py,
but adapted to work with the trading-agent's data structures (Bar objects).

Features:
- Multi-panel charts (Price, Volume, RSI)
- Moving Averages (multiple periods)
- Bollinger Bands
- RSI (multiple periods)
- Volume with auto green/red coloring
- Multiple themes (tradingview, moomoo, dark, light)
- Weekend/holiday gap removal
"""

from __future__ import annotations
from datetime import datetime
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.models.bar import Bar


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


def get_theme(name: str) -> Dict[str, Any]:
    """Get theme configuration by name."""
    return THEMES.get((name or "").lower(), THEMES["tradingview"])


# ---------- Data Conversion ----------

def bars_to_dataframe(bars: List[Bar]) -> pd.DataFrame:
    """Convert list of Bar objects to pandas DataFrame."""
    if not bars:
        return pd.DataFrame()
    
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


# ---------- Indicator Calculations ----------

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


# ---------- Chart Builder ----------

class ChartVisualizer:
    """
    Plotly-based chart visualizer for trading data.
    
    Similar to fastapi_stockchart/main.py but works with Bar objects.
    """
    
    def __init__(self, theme: str = "tradingview", figsize: tuple = (1400, 900)):
        """
        Initialize the chart visualizer.
        
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
        symbol: str = "UNKNOWN",
        ma_list: List[int] = None,
        boll_len: Optional[int] = 20,
        boll_std: Optional[float] = 2.0,
        rsi_list: List[int] = None,
        title: Optional[str] = None,
    ) -> go.Figure:
        """
        Build a multi-panel chart with price, volume, and RSI.
        
        Args:
            bars: List of Bar objects
            symbol: Stock symbol
            ma_list: List of moving average periods (e.g., [5, 20, 60])
            boll_len: Bollinger Band length (default 20)
            boll_std: Bollinger Band standard deviation multiplier (default 2.0)
            rsi_list: List of RSI periods (e.g., [6, 14, 24])
            title: Chart title (default: "{symbol} Multi-Panel Chart")
        
        Returns:
            Plotly Figure object
        """
        if not bars:
            raise ValueError("No bars provided for charting")
        
        # Convert to DataFrame
        df = bars_to_dataframe(bars)
        if df.empty:
            raise ValueError("Empty DataFrame after conversion")
        
        # Default indicators if not provided
        if ma_list is None:
            ma_list = [5, 20, 60, 120, 250]
        if rsi_list is None:
            rsi_list = [14]
        
        # Add indicators
        df = add_indicators(df, ma_list, boll_len, boll_std, rsi_list)
        
        # Build chart
        self.fig = self._build_full_chart(df, symbol, ma_list, rsi_list, title)
        
        return self.fig
    
    def _build_full_chart(
        self,
        df: pd.DataFrame,
        ticker: str,
        ma_list: List[int],
        rsi_list: List[int],
        title: Optional[str],
    ) -> go.Figure:
        """Build the full multi-panel chart."""
        theme = self.theme
        
        if title is None:
            title = f"{ticker} Multi-Panel Chart"
        
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.6, 0.2, 0.2],
            vertical_spacing=0.03,
            subplot_titles=(
                f"{ticker} Price",
                "Volume",
                "RSI",
            ),
        )
        
        # ----- Row 1: Candles + overlays -----
        
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="Candles",
                increasing_line_color=theme["candle_up"],
                decreasing_line_color=theme["candle_down"],
                increasing_fillcolor=theme["candle_up"],
                decreasing_fillcolor=theme["candle_down"],
                increasing_line_width=1,
                decreasing_line_width=1,
            ),
            row=1,
            col=1,
        )
        
        # Bollinger Bands
        if "BOLL_MID" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["BOLL_MID"],
                    mode="lines",
                    name="BOLL MID",
                    line=dict(width=1.1, color=theme["boll_mid"]),
                    opacity=0.8,
                ),
                row=1,
                col=1,
            )
        if "BOLL_UPPER" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["BOLL_UPPER"],
                    mode="lines",
                    name="BOLL UP",
                    line=dict(width=1.0, dash="dot", color=theme["boll_up"]),
                    opacity=0.7,
                ),
                row=1,
                col=1,
            )
        if "BOLL_LOWER" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["BOLL_LOWER"],
                    mode="lines",
                    name="BOLL LOW",
                    line=dict(width=1.0, dash="dot", color=theme["boll_low"]),
                    opacity=0.7,
                ),
                row=1,
                col=1,
            )
        
        # Moving Averages
        for length in ma_list:
            colname = f"MA{length}"
            if colname in df.columns:
                color = theme["ma_colors"].get(length)
                line_kwargs = {"width": 1.1}
                if color:
                    line_kwargs["color"] = color
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[colname],
                        mode="lines",
                        name=f"MA{length}",
                        line=line_kwargs,
                        opacity=0.9,
                    ),
                    row=1,
                    col=1,
                )
        
        # ----- Row 2: Volume -----
        
        vol_colors = np.where(
            df["Close"] >= df["Open"],
            theme["volume_up"],
            theme["volume_down"]
        )
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["Volume"],
                name="Volume",
                marker_color=vol_colors,
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        
        # ----- Row 3: RSI -----
        
        for length in rsi_list:
            colname = f"RSI{length}"
            if colname in df.columns:
                color = theme["rsi_colors"].get(length, "#ffffff")
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[colname],
                        mode="lines",
                        name=f"RSI{length}",
                        line=dict(color=color, width=1.5),
                    ),
                    row=3,
                    col=1,
                )
        
        # RSI overbought/oversold levels
        if not df.empty and any(f"RSI{length}" in df.columns for length in rsi_list):
            x_min = df.index.min()
            x_max = df.index.max()
            for level in (70, 30):
                fig.add_trace(
                    go.Scatter(
                        x=[x_min, x_max],
                        y=[level, level],
                        mode="lines",
                        line=dict(dash="dot", width=1, color=theme["grid_color"]),
                        showlegend=False,
                    ),
                    row=3,
                    col=1,
                )
        
        # ----- Layout + gap tightening (weekends + holidays) -----
        
        fig.update_layout(
            title=title,
            paper_bgcolor=theme["bg_color"],
            plot_bgcolor=theme["plot_bg"],
            font=dict(color=theme["text_color"]),
            xaxis_rangeslider_visible=False,
            margin=dict(l=40, r=40, t=60, b=40),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
            width=self.figsize[0],
            height=self.figsize[1],
        )
        
        # Build rangebreaks: weekends + missing business days (holidays)
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
        
        fig.update_xaxes(
            showgrid=True,
            gridcolor=theme["grid_color"],
            linecolor=theme["axis_color"],
            tickfont=dict(color=theme["axis_color"]),
            rangebreaks=rangebreaks,
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor=theme["grid_color"],
            linecolor=theme["axis_color"],
            tickfont=dict(color=theme["axis_color"]),
        )
        
        # Set volume y-axis to start at 0
        fig.update_yaxes(title_text="Volume", row=2, col=1, rangemode="tozero")
        fig.update_yaxes(title_text="RSI", row=3, col=1)
        
        return fig
    
    def show(self, renderer: str = "browser"):
        """Display the chart in browser or specified renderer."""
        if self.fig is None:
            raise ValueError("Chart not built. Call build_chart() first.")
        self.fig.show(renderer=renderer)
    
    def to_html(self, filename: Optional[str] = None) -> str:
        """Export chart to HTML file."""
        if self.fig is None:
            raise ValueError("Chart not built. Call build_chart() first.")
        html = self.fig.to_html(full_html=True, include_plotlyjs="cdn")
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
    
    def close(self):
        """Close the figure (no-op for Plotly, kept for compatibility)."""
        pass
