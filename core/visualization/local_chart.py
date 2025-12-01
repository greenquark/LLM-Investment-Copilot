from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import numpy as np

from core.models.bar import Bar
from core.visualization.models import TradeSignal, IndicatorData


class LocalChartVisualizer:
    """
    Python-only charting using matplotlib.
    Provides the same functionality as WebChartServer but runs locally without Flask.
    """
    
    def __init__(self, style: str = "dark_background", figsize: tuple = (16, 12)):
        """
        Initialize the local chart visualizer.
        
        Args:
            style: Matplotlib style (default: "dark_background" to match web chart)
            figsize: Figure size (width, height) in inches
        """
        # Try to use an interactive backend if available
        try:
            current_backend = matplotlib.get_backend()
            if current_backend.lower() == 'agg':  # Non-interactive backend
                # Try to switch to a GUI backend
                backends_to_try = ['TkAgg', 'Qt5Agg', 'QtAgg']
                for backend_name in backends_to_try:
                    try:
                        matplotlib.use(backend_name, force=True)
                        break
                    except Exception:
                        continue
        except Exception:
            pass  # Use default backend
        
        plt.style.use(style)
        self.fig = None
        self.axes = None
        self._figsize = figsize
        self._data = None
    
    def set_data(
        self,
        bars: List[Bar],
        signals: List[TradeSignal],
        indicator_data: List[IndicatorData],
        equity_curve: Dict[datetime, float],
        metrics: Optional[Dict] = None,
        symbol: str = "UNKNOWN"
    ):
        """
        Set the data to display (same interface as WebChartServer).
        
        Args:
            bars: List of price bars
            signals: List of buy/sell signals
            indicator_data: List of indicator data points
            equity_curve: Equity curve data
            metrics: Performance metrics dictionary
            symbol: Stock symbol
        """
        self._data = {
            "bars": bars,
            "signals": signals,
            "indicator_data": indicator_data,
            "equity_curve": equity_curve,
            "metrics": metrics or {},
            "symbol": symbol
        }
    
    def show(self, save_path: Optional[Path] = None, block: bool = True):
        """
        Display the chart.
        
        Args:
            save_path: Optional path to save the chart as an image
            block: Whether to block execution until the chart window is closed
        """
        if not self._data:
            raise ValueError("No data set. Call set_data() first.")
        
        bars = self._data["bars"]
        signals = self._data["signals"]
        indicator_data = self._data["indicator_data"]
        equity_curve = self._data["equity_curve"]
        metrics = self._data["metrics"]
        symbol = self._data["symbol"]
        
        if not bars:
            print("No bars available for charting")
            return
        
        # Sort bars by timestamp
        bars = sorted(bars, key=lambda b: b.timestamp)
        
        # Create figure with subplots matching web chart layout:
        # - Price chart: 70% of height
        # - Volume: 10% of height
        # - Indicator: 20% of height
        # - Equity: 20% of height (if provided)
        num_plots = 3 if equity_curve else 2
        height_ratios = [7, 1, 2] if equity_curve else [7, 1, 2]
        
        self.fig, axes = plt.subplots(
            num_plots, 1,
            figsize=self._figsize,
            sharex=True,
            gridspec_kw={'height_ratios': height_ratios}
        )
        
        if num_plots == 1:
            axes = [axes]
        self.axes = axes if isinstance(axes, np.ndarray) else [axes]
        
        # Extract data
        timestamps = [b.timestamp for b in bars]
        opens = [b.open for b in bars]
        highs = [b.high for b in bars]
        lows = [b.low for b in bars]
        closes = [b.close for b in bars]
        volumes = [b.volume or 0 for b in bars]
        
        # Create sequential indices to remove gaps (0, 1, 2, 3, ...)
        # This ensures even spacing regardless of weekends/holidays
        indices = list(range(len(timestamps)))
        
        # Plot 1: Price Chart (Candlesticks) - 70% of screen
        ax_price = self.axes[0]
        self._plot_candlesticks(ax_price, indices, timestamps, opens, highs, lows, closes)
        self._plot_signals(ax_price, signals, indices, timestamps, closes)
        
        # Add title with metrics if available
        title = f"Revised MP2.0 - {symbol}"
        if metrics:
            title += f"\nReturn: {metrics.get('total_return', 0):.2%} | "
            title += f"CAGR: {metrics.get('cagr', 0):.2%} | "
            title += f"Sharpe: {metrics.get('sharpe', 0):.2f} | "
            title += f"Max DD: {metrics.get('max_drawdown', 0):.2%}"
        ax_price.set_title(title, fontsize=12, fontweight="bold", color="#00FF66")
        ax_price.set_ylabel("Price ($)", fontsize=10)
        ax_price.grid(True, alpha=0.3, color="#444")
        ax_price.legend(loc="upper left", fontsize=8)
        
        # Plot 2: Volume - 10% of screen
        ax_vol = self.axes[1]
        self._plot_volume(ax_vol, indices, timestamps, volumes, closes, opens)
        ax_vol.set_ylabel("Volume", fontsize=10)
        ax_vol.grid(True, alpha=0.3, color="#444")
        
        # Plot 3: Revised MP2.0 Indicator - 20% of screen
        ax_ind = self.axes[2]
        self._plot_indicator(ax_ind, indicator_data, indices, timestamps, signals)
        ax_ind.set_ylabel("Revised MP2.0", fontsize=10)
        ax_ind.grid(True, alpha=0.3, color="#444")
        ax_ind.axhline(y=0, color="#fff", linewidth=0.5, alpha=0.5)
        ax_ind.legend(loc="upper left", fontsize=8)
        
        # Plot 4: Equity Curve (if provided)
        if equity_curve and len(self.axes) > 3:
            ax_eq = self.axes[3]
            self._plot_equity(ax_eq, equity_curve, indices, timestamps)
            ax_eq.set_ylabel("Equity ($)", fontsize=10)
            ax_eq.grid(True, alpha=0.3, color="#444")
            ax_eq.legend(loc="upper left", fontsize=8)
        
        # Format x-axis (bottom subplot) - use sequential indices with date labels
        ax_bottom = self.axes[-1]
        ax_bottom.set_xlabel("Time", fontsize=10)
        ax_bottom.set_xlim(-0.5, len(indices) - 0.5)
        
        # Set custom tick positions and labels based on data range
        if len(timestamps) > 0:
            # Show approximately 10-15 ticks
            num_ticks = min(15, len(timestamps))
            tick_indices = [int(i * (len(indices) - 1) / (num_ticks - 1)) for i in range(num_ticks)]
            tick_indices = [i for i in tick_indices if 0 <= i < len(timestamps)]
            
            # Always include the last index
            if len(timestamps) - 1 not in tick_indices:
                tick_indices.append(len(timestamps) - 1)
            
            tick_indices = sorted(set(tick_indices))
            tick_positions = tick_indices
            tick_labels = []
            
            # Format labels based on data range
            date_range = (timestamps[-1] - timestamps[0]).days if len(timestamps) > 1 else 0
            for idx in tick_indices:
                ts = timestamps[idx]
                if date_range > 90:
                    # More than 3 months: show month/day
                    tick_labels.append(ts.strftime("%m/%d"))
                elif date_range > 7:
                    # More than a week: show month/day
                    tick_labels.append(ts.strftime("%m/%d"))
                else:
                    # Less than a week: show date and time
                    tick_labels.append(ts.strftime("%m/%d %H:%M"))
            
            ax_bottom.set_xticks(tick_positions)
            ax_bottom.set_xticklabels(tick_labels, rotation=45, ha="right")
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            try:
                self.fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor='#1e1e1e')
                print(f"Chart saved to {save_path}")
            except Exception as e:
                print(f"Error saving chart: {e}")
        
        # Display chart
        try:
            plt.show(block=block)
        except Exception as e:
            print(f"Error displaying chart: {e}")
            if save_path:
                print(f"Chart has been saved to: {save_path}")
    
    def _plot_candlesticks(self, ax, indices, timestamps, opens, highs, lows, closes):
        """Plot candlestick chart using sequential indices to remove gaps."""
        for i, (idx, o, h, l, c) in enumerate(zip(indices, opens, highs, lows, closes)):
            color = "#00FF66" if c >= o else "#FF1A1A"  # Green for up, red for down
            
            # Draw the wick (high-low line) using index position
            ax.plot([idx, idx], [l, h], color=color, linewidth=1, alpha=0.8)
            
            # Draw the body (open-close rectangle) using index position
            body_height = abs(c - o)
            body_bottom = min(o, c)
            if body_height == 0:
                body_height = 0.01 * c  # Minimum visible height
            
            rect = Rectangle(
                (idx - 0.4, body_bottom),
                0.8,
                body_height,
                facecolor=color,
                edgecolor=color,
                alpha=0.8,
                linewidth=0.5
            )
            ax.add_patch(rect)
    
    def _plot_signals(self, ax, signals: List[TradeSignal], indices, timestamps, closes):
        """Plot buy/sell signals on the price chart using sequential indices."""
        if not signals:
            return
        
        buy_signals = [s for s in signals if s.side == "BUY"]
        sell_signals = [s for s in signals if s.side == "SELL"]
        
        # Calculate offset for signal markers (slightly below buy, slightly above sell)
        price_range = max(closes) - min(closes) if closes else 1
        buy_offset = -price_range * 0.02  # 2% below price
        sell_offset = price_range * 0.02   # 2% above price
        
        # Helper function to normalize timestamp for matching
        def normalize_timestamp(ts: datetime) -> datetime:
            """Normalize timestamp to remove microseconds for better matching."""
            return ts.replace(microsecond=0)
        
        # Helper function to find closest bar index for a signal timestamp
        def find_closest_bar_index(signal_ts: datetime) -> int:
            """Find the index of the bar with the closest timestamp to the signal."""
            signal_ts_norm = normalize_timestamp(signal_ts)
            # First try exact match
            for i, bar_ts in enumerate(timestamps):
                if normalize_timestamp(bar_ts) == signal_ts_norm:
                    return i
            # If no exact match, find closest by time difference
            return min(
                range(len(timestamps)),
                key=lambda i: abs((normalize_timestamp(timestamps[i]) - signal_ts_norm).total_seconds())
            )
        
        if buy_signals:
            buy_indices = []
            buy_prices = []
            for s in buy_signals:
                closest_idx = find_closest_bar_index(s.timestamp)
                buy_indices.append(indices[closest_idx])
                buy_prices.append(s.price + buy_offset)
            ax.scatter(
                buy_indices, buy_prices,
                color="#00FF66",
                marker="^",
                s=150,
                label="BUY",
                zorder=5,
                edgecolors="#ffffff",
                linewidths=1.5
            )
        
        if sell_signals:
            sell_indices = []
            sell_prices = []
            for s in sell_signals:
                closest_idx = find_closest_bar_index(s.timestamp)
                sell_indices.append(indices[closest_idx])
                sell_prices.append(s.price + sell_offset)
            ax.scatter(
                sell_indices, sell_prices,
                color="#FF1A1A",
                marker="v",
                s=150,
                label="SELL",
                zorder=5,
                edgecolors="#ffffff",
                linewidths=1.5
            )
    
    def _plot_volume(self, ax, indices, timestamps, volumes, closes, opens):
        """Plot volume bars using sequential indices to remove gaps."""
        colors = ["#00FF66" if c >= o else "#FF1A1A" 
                 for c, o in zip(closes, opens)]
        
        # Use fixed width for even spacing (no gaps)
        width = 0.8
        
        ax.bar(indices, volumes, color=colors, alpha=0.6, width=width)
        ax.set_ylim(bottom=0)
    
    def _plot_indicator(self, ax, indicator_data: List[IndicatorData], indices, timestamps, signals: List[TradeSignal]):
        """Plot Revised MP2.0 indicator with buy/sell signals using sequential indices."""
        if not indicator_data:
            return
        
        # Create a map of indicator data by timestamp (normalized for matching)
        indicator_map = {}
        for ind in indicator_data:
            # Normalize timestamp to hour level for hourly bars, seconds for others
            if len(timestamps) > 0:
                time_diff = (timestamps[1] - timestamps[0]).total_seconds() if len(timestamps) > 1 else 3600
                if time_diff >= 3600:  # Hourly or longer
                    key = ind.timestamp.replace(minute=0, second=0, microsecond=0)
                else:
                    key = ind.timestamp.replace(microsecond=0)
                indicator_map[key] = ind
        
        # Match indicator data to bar timestamps
        pos_counts = []
        neg_counts = []
        
        for ts in timestamps:
            # Normalize timestamp for matching
            if len(timestamps) > 1:
                time_diff = (timestamps[1] - timestamps[0]).total_seconds()
                if time_diff >= 3600:
                    key = ts.replace(minute=0, second=0, microsecond=0)
                else:
                    key = ts.replace(microsecond=0)
            else:
                key = ts.replace(microsecond=0)
            
            ind = indicator_map.get(key)
            if ind:
                pos_counts.append(ind.positive_count)
                neg_counts.append(-ind.negative_count)
            else:
                # Use previous value or zero
                if pos_counts:
                    pos_counts.append(pos_counts[-1])
                    neg_counts.append(neg_counts[-1])
                else:
                    pos_counts.append(0)
                    neg_counts.append(0)
        
        # Plot positive and negative bars using sequential indices
        width = 0.8
        ax.bar(indices, pos_counts, width=width, color="#00FF66", alpha=0.7, label="Positive")
        ax.bar(indices, neg_counts, width=width, color="#FF1A1A", alpha=0.7, label="Negative")
        
        # Plot buy/sell signals on indicator chart
        buy_signals = [s for s in signals if s.side == "BUY"]
        sell_signals = [s for s in signals if s.side == "SELL"]
        
        if buy_signals or sell_signals:
            # Calculate offset for signals
            max_pos = max(pos_counts) if pos_counts else 0
            min_neg = min(neg_counts) if neg_counts else 0
            indicator_range = max_pos - min_neg if (pos_counts and neg_counts) else 1
            buy_offset = indicator_range * 0.05
            sell_offset = -indicator_range * 0.05
            
            # Helper function to normalize timestamp for matching
            def normalize_timestamp(ts: datetime) -> datetime:
                """Normalize timestamp to remove microseconds for better matching."""
                return ts.replace(microsecond=0)
            
            # Helper function to find closest bar index for a signal timestamp
            def find_closest_bar_index(signal_ts: datetime) -> int:
                """Find the index of the bar with the closest timestamp to the signal."""
                signal_ts_norm = normalize_timestamp(signal_ts)
                # First try exact match
                for i, bar_ts in enumerate(timestamps):
                    if normalize_timestamp(bar_ts) == signal_ts_norm:
                        return i
                # If no exact match, find closest by time difference
                return min(
                    range(len(timestamps)),
                    key=lambda i: abs((normalize_timestamp(timestamps[i]) - signal_ts_norm).total_seconds())
                )
            
            if buy_signals:
                buy_indices = []
                buy_values = []
                for s in buy_signals:
                    closest_idx = find_closest_bar_index(s.timestamp)
                    buy_indices.append(indices[closest_idx])
                    buy_values.append(pos_counts[closest_idx] + buy_offset)
                ax.scatter(
                    buy_indices, buy_values,
                    color="#00FF66",
                    marker="^",
                    s=150,
                    label="BUY",
                    zorder=5,
                    edgecolors="#ffffff",
                    linewidths=1.5
                )
            
            if sell_signals:
                sell_indices = []
                sell_values = []
                for s in sell_signals:
                    closest_idx = find_closest_bar_index(s.timestamp)
                    sell_indices.append(indices[closest_idx])
                    sell_values.append(neg_counts[closest_idx] + sell_offset)
                ax.scatter(
                    sell_indices, sell_values,
                    color="#FF1A1A",
                    marker="v",
                    s=150,
                    label="SELL",
                    zorder=5,
                    edgecolors="#ffffff",
                    linewidths=1.5
                )
    
    def _plot_equity(self, ax, equity_curve: Dict[datetime, float], indices, timestamps):
        """Plot equity curve using sequential indices to remove gaps."""
        if not equity_curve:
            return
        
        # Map equity data to bar indices
        eq_times = sorted(equity_curve.keys())
        eq_values = []
        eq_indices = []
        
        for eq_time in eq_times:
            # Find closest bar index for this equity timestamp
            closest_idx = min(range(len(timestamps)),
                             key=lambda i: abs((timestamps[i] - eq_time).total_seconds()))
            eq_indices.append(indices[closest_idx])
            eq_values.append(equity_curve[eq_time])
        
        ax.plot(eq_indices, eq_values, color="#FFD700", linewidth=2, label="Equity")
        ax.fill_between(eq_indices, eq_values, alpha=0.3, color="#FFD700")
    
    def close(self):
        """Close the figure."""
        if self.fig:
            plt.close(self.fig)
            self.fig = None
            self.axes = None

