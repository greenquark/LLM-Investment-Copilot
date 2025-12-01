from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional
import math

from core.strategy.base import Strategy, Context
from core.data.base import DataEngine
from core.models.bar import Bar
from core.models.order import Order, Side, OrderType, InstrumentType
from core.visualization.models import TradeSignal, IndicatorData


@dataclass
class MysticPulseConfig:
    """Configuration for Revised MP2.0 strategy.
    
    Revised MP2.0 is an improved version built on top of Mystic Pulse 2.0.
    """
    adx_length: int = 9
    smoothing_factor: int = 1
    collect_length: int = 100
    contrast_gamma_bars: float = 0.7
    contrast_gamma_plots: float = 0.8
    
    # Trading parameters
    min_trend_score: int = 5  # Minimum trend_score to generate signal
    timeframe: str = "1D"  # Bar timeframe to use (allowed: 15m, 30m, 1H, 2H, 4H, 1D)
    capital_deployment_pct: float = 1.0  # Percentage of available capital to deploy per trade (0.0 to 1.0)
    
    # Allowed timeframes for this strategy
    ALLOWED_TIMEFRAMES = {"15m", "30m", "1H", "2H", "4H", "1D"}
    
    def __post_init__(self):
        """Validate timeframe and capital deployment after initialization."""
        # Normalize timeframe for comparison (uppercase, handle variations)
        timeframe_normalized = self.timeframe.upper()
        # Handle variations: "D" -> "1D", "H" -> "1H", "15M" -> "15m"
        if timeframe_normalized == "D":
            timeframe_normalized = "1D"
        elif timeframe_normalized == "H":
            timeframe_normalized = "1H"
        elif timeframe_normalized.endswith("M") and not timeframe_normalized.endswith("H"):
            # Convert "15M" to "15m" (lowercase m for minutely)
            timeframe_normalized = timeframe_normalized[:-1] + "m"
        
        if timeframe_normalized not in self.ALLOWED_TIMEFRAMES:
            raise ValueError(
                f"Invalid timeframe '{self.timeframe}'. "
                f"Revised MP2.0 strategy only supports: {', '.join(sorted(self.ALLOWED_TIMEFRAMES))}"
            )
        
        if not 0.0 <= self.capital_deployment_pct <= 1.0:
            raise ValueError(
                f"Invalid capital_deployment_pct '{self.capital_deployment_pct}'. "
                f"Must be between 0.0 and 1.0 (0% to 100%)."
            )


class MysticPulseStrategy(Strategy):
    """Revised MP2.0 strategy based on ADX with trend counting.
    
    Revised MP2.0 is an improved version built on top of Mystic Pulse 2.0.
    """
    
    def __init__(self, symbol: str, config: MysticPulseConfig, data_engine: DataEngine):
        self._symbol = symbol
        self._cfg = config
        self._data = data_engine
        
        # Validate timeframe (config validation should catch this, but double-check)
        timeframe_normalized = config.timeframe.upper()
        if timeframe_normalized == "D":
            timeframe_normalized = "1D"
        elif timeframe_normalized == "H":
            timeframe_normalized = "1H"
        elif timeframe_normalized.endswith("M") and not timeframe_normalized.endswith("H"):
            timeframe_normalized = timeframe_normalized[:-1] + "m"
        
        if timeframe_normalized not in MysticPulseConfig.ALLOWED_TIMEFRAMES:
            raise ValueError(
                f"Invalid timeframe '{config.timeframe}'. "
                f"Revised MP2.0 strategy only supports: {', '.join(sorted(MysticPulseConfig.ALLOWED_TIMEFRAMES))}"
            )
        
        # Persistent state variables (equivalent to Pine Script var)
        self._smoothed_true_range: Optional[float] = None
        self._smoothed_dm_plus: Optional[float] = None
        self._smoothed_dm_minus: Optional[float] = None
        self._positive_count: int = 0
        self._negative_count: int = 0
        self._total_count: int = 0
        
        # Store recent bars for calculations
        self._bars: List[Bar] = []
        self._prev_di_plus: Optional[float] = None
        self._prev_di_minus: Optional[float] = None
        
        # Track previous counts for flip detection
        self._prev_positive_count: int = 0
        self._prev_negative_count: int = 0
        
        # Track signals and indicator data for visualization
        self._signals: List = []  # Will store TradeSignal objects
        self._indicator_history: List = []  # Will store IndicatorData objects
        
        # Track current signal state to avoid clutter
        self._current_signal_state: Optional[str] = None  # "BUY", "SELL", or None
    
    @staticmethod
    def clamp01(x: float) -> float:
        """Clamp value between 0 and 1."""
        return max(0.0, min(1.0, x))
    
    @staticmethod
    def gamma_adj(x: float, g: float) -> float:
        """Apply gamma adjustment."""
        return math.pow(MysticPulseStrategy.clamp01(x), g)
    
    @staticmethod
    def norm_in_window(values: List[float], winlen: int) -> float:
        """Normalize value within a window."""
        if not values or winlen <= 0:
            return 0.0
        window = values[-winlen:] if len(values) >= winlen else values
        min_v = min(window)
        max_v = max(window)
        span_v = max_v - min_v
        span_safe = span_v if span_v != 0 else 1.0
        val = values[-1]
        return (val - min_v) / span_safe
    
    def _sma(self, values: List[float], length: int) -> float:
        """Simple Moving Average."""
        if not values or length <= 0:
            return 0.0
        window = values[-length:] if len(values) >= length else values
        return sum(window) / len(window)
    
    def _wilder_smooth(self, prev_smoothed: Optional[float], current: float, length: int) -> float:
        """Wilder's smoothing: smoothed = prev - (prev / length) + current."""
        if prev_smoothed is None:
            return current
        return prev_smoothed - (prev_smoothed / length) + current
    
    async def on_start(self, ctx: Context) -> None:
        """Initialize strategy."""
        ctx.log(f"MysticPulseStrategy started for {self._symbol}")
        ctx.log(f"Config: ADX={self._cfg.adx_length}, Smoothing={self._cfg.smoothing_factor}, Timeframe={self._cfg.timeframe}")
    
    async def on_decision(self, ctx: Context, now: datetime) -> None:
        """Main decision logic."""
        # Reuse bars from context if available (engine already fetched with max lookback)
        # This avoids duplicate API calls
        if hasattr(ctx, '_bars') and ctx._bars is not None:
            bars = ctx._bars
        else:
            # Fallback: fetch bars if not provided by engine
            # Get historical bars (need enough for calculations)
            # Adjust lookback period based on timeframe (only allowed: 15m, 30m, 1H, 2H, 4H, 1D)
            lookback = max(self._cfg.collect_length, self._cfg.adx_length * 2, 50)
            timeframe_upper = self._cfg.timeframe.upper()
            
            # Normalize timeframe for comparison
            if timeframe_upper == "D":
                timeframe_upper = "1D"
            elif timeframe_upper == "H":
                timeframe_upper = "1H"
            elif timeframe_upper.endswith("M") and not timeframe_upper.endswith("H"):
                timeframe_upper = timeframe_upper[:-1] + "m"
            
            # Calculate appropriate lookback period based on timeframe
            if timeframe_upper == "1D":
                # Daily: look back 30 days
                start = now - timedelta(days=30)
            elif timeframe_upper.endswith("H"):
                # Hourly (1H, 2H, 4H): look back 30 days
                start = now - timedelta(days=30)
            elif timeframe_upper.endswith("m"):
                # Minutely (15m, 30m): look back 7 days
                start = now - timedelta(days=7)
            else:
                # Default fallback: 7 days
                start = now - timedelta(days=7)
            
            bars = await self._data.get_bars(self._symbol, start, now, self._cfg.timeframe)
        
        # Update context with data source from data engine
        if hasattr(ctx, '_data_source'):
            if hasattr(self._data, 'last_data_source'):
                ctx._data_source = self._data.last_data_source
            elif hasattr(self._data, '_last_data_source'):
                ctx._data_source = self._data._last_data_source
        
        if len(bars) < 2:
            ctx.log(f"Insufficient bars: {len(bars)}")
            return
        
        # Sort by timestamp and update our bar list
        bars.sort(key=lambda b: b.timestamp)
        self._bars = bars
        
        # Calculate smoothed OHLC for each bar
        opens = [b.open for b in self._bars]
        highs = [b.high for b in self._bars]
        lows = [b.low for b in self._bars]
        closes = [b.close for b in self._bars]
        
        # Calculate smoothed values for current and previous bar
        if len(self._bars) < 2:
            return
        
        # For current bar (last in series)
        open_s = self._sma(opens, self._cfg.smoothing_factor)
        high_s = self._sma(highs, self._cfg.smoothing_factor)
        low_s = self._sma(lows, self._cfg.smoothing_factor)
        close_s = self._sma(closes, self._cfg.smoothing_factor)
        
        # For previous bar (second to last)
        prev_opens = opens[:-1] if len(opens) > 1 else opens
        prev_highs = highs[:-1] if len(highs) > 1 else highs
        prev_lows = lows[:-1] if len(lows) > 1 else lows
        prev_closes = closes[:-1] if len(closes) > 1 else closes
        
        prev_open_s = self._sma(prev_opens, self._cfg.smoothing_factor)
        prev_high_s = self._sma(prev_highs, self._cfg.smoothing_factor)
        prev_low_s = self._sma(prev_lows, self._cfg.smoothing_factor)
        prev_close_s = self._sma(prev_closes, self._cfg.smoothing_factor)
        
        # Calculate True Range for current bar
        tr1 = high_s - low_s
        tr2 = abs(high_s - prev_close_s)
        tr3 = abs(low_s - prev_close_s)
        true_range = max(tr1, tr2, tr3)
        
        # Calculate Directional Movement
        high_diff = high_s - prev_high_s
        low_diff = prev_low_s - low_s
        
        dm_plus = max(high_diff, 0) if high_diff > low_diff else 0.0
        dm_minus = max(low_diff, 0) if low_diff > high_diff else 0.0
        
        # Apply Wilder smoothing
        self._smoothed_true_range = self._wilder_smooth(
            self._smoothed_true_range, true_range, self._cfg.adx_length
        )
        self._smoothed_dm_plus = self._wilder_smooth(
            self._smoothed_dm_plus, dm_plus, self._cfg.adx_length
        )
        self._smoothed_dm_minus = self._wilder_smooth(
            self._smoothed_dm_minus, dm_minus, self._cfg.adx_length
        )
        
        # Calculate Directional Indicators
        denom = self._smoothed_true_range
        if denom is None or denom == 0:
            ctx.log("Denominator is zero, skipping")
            return
        
        di_plus = (self._smoothed_dm_plus / denom * 100) if self._smoothed_dm_plus is not None else None
        di_minus = (self._smoothed_dm_minus / denom * 100) if self._smoothed_dm_minus is not None else None
        
        # Store previous counts before updating (for flip detection)
        self._prev_positive_count = self._positive_count
        self._prev_negative_count = self._negative_count
        
        # Update counts based on DI trends
        if (di_plus is not None and self._prev_di_plus is not None and
            di_plus > self._prev_di_plus and di_plus > (di_minus or 0)):
            self._positive_count += 1
            self._negative_count = 0
        elif (di_minus is not None and self._prev_di_minus is not None and
              di_minus > self._prev_di_minus and di_minus > (di_plus or 0)):
            self._negative_count += 1
            self._positive_count = 0
        
        self._total_count = self._positive_count if self._positive_count > 0 else self._negative_count
        self._prev_di_plus = di_plus
        self._prev_di_minus = di_minus
        
        # Calculate trend score
        trend_score = self._positive_count - self._negative_count
        trend_direction = "bullish" if trend_score >= 0 else "bearish"
        
        # Store indicator data for visualization
        # Use the timestamp of the most recent complete bar (not 'now') to align with bar timestamps
        # This ensures indicator data points match the bar timestamps in the chart
        # Use the close price of the most recent complete bar (today's close)
        # This is the execution price for orders generated at this decision point
        # Decision point is at market close (4:00 PM ET), execution can happen in extended hours
        # TODO: Future improvement - Generate signals 15 minutes before market close (3:45 PM ET)
        #       This would allow execution at the close price on the same day the signal is generated
        current_price = closes[-1]
        # Use the timestamp of the most recent bar, not 'now', to align with chart bars
        bar_timestamp = self._bars[-1].timestamp if self._bars else now
        indicator_data = IndicatorData(
            timestamp=bar_timestamp,
            positive_count=self._positive_count,
            negative_count=self._negative_count,
            trend_score=trend_score,
            di_plus=di_plus,
            di_minus=di_minus,
        )
        self._indicator_history.append(indicator_data)
        
        # Log current state
        di_plus_str = f"{di_plus:.2f}" if di_plus is not None else "0.00"
        di_minus_str = f"{di_minus:.2f}" if di_minus is not None else "0.00"
        ctx.log(
            f"[{now}] price={current_price:.2f} "
            f"DI+={di_plus_str} "
            f"DI-={di_minus_str} "
            f"pos={self._positive_count} neg={self._negative_count} "
            f"score={trend_score} {trend_direction}"
        )
        
        # Generate trading signals based on flip logic:
        # Buy when trend flips from negative to positive
        # Sell when trend flips from positive to negative
        #
        # Signal timing: Signals are generated at market close (4:00 PM ET = 16:00)
        #                based on the complete bar that just closed (same day).
        # Execution price: Orders execute at today's close price.
        #                  In reality, execution can happen in extended hours after market close.
        # TODO: Future improvement - Generate signals 15 minutes before market close (3:45 PM ET)
        #       This would allow execution at the close price on the same day the signal is generated
        positions = ctx.portfolio.get_positions()
        has_position = any(p.symbol == self._symbol and p.quantity > 0 for p in positions)
        
        # Debug logging for position tracking
        if positions:
            ctx.log(f"  Portfolio positions: {[(p.symbol, p.quantity, p.avg_price) for p in positions]}")
        else:
            ctx.log(f"  Portfolio positions: None (cash: ${ctx.portfolio.state.cash:,.2f})")
        
        # Buy signal: Flip to bullish - we have >0 positive_count, zero negative_count now,
        # but previously had some negative_count
        buy_signal = (self._positive_count > 0 and 
                     self._negative_count == 0 and 
                     self._prev_negative_count > 0)
        
        # Sell signal: Flip to bearish - we have >0 negative_count, zero positive_count now,
        # but previously had some positive_count
        sell_signal = (self._negative_count > 0 and 
                      self._positive_count == 0 and 
                      self._prev_positive_count > 0)
        
        # Generate buy signal on flip to bullish
        if buy_signal and self._current_signal_state != "BUY":
            if not has_position:
                # Calculate quantity based on capital deployment percentage
                available_cash = ctx.portfolio.state.cash
                capital_to_deploy = available_cash * self._cfg.capital_deployment_pct
                
                # Calculate quantity based on current price and capital to deploy
                quantity = int(capital_to_deploy / current_price) if current_price > 0 else 0
                
                if quantity > 0:
                    order = Order(
                        id=f"mp_buy_{now.timestamp()}",
                        symbol=self._symbol,
                        side=Side.BUY,
                        quantity=quantity,
                        order_type=OrderType.MARKET,
                        limit_price=None,
                        instrument_type=InstrumentType.STOCK,
                    )
                    await ctx.execution.submit_order(order)  # type: ignore[attr-defined]
                    ctx.log(f"BUY signal: Trend flipped to bullish (pos={self._positive_count}, neg={self._negative_count}, prev_neg={self._prev_negative_count})")
                    ctx.log(f"  Deploying {self._cfg.capital_deployment_pct*100:.0f}% capital: ${capital_to_deploy:,.2f} for {quantity} shares @ ${current_price:.2f}")
                else:
                    ctx.log(f"BUY signal: Insufficient capital to deploy (available: ${available_cash:,.2f}, price: ${current_price:.2f})")
            
            # Store signal for visualization
            # Use the bar's timestamp (not 'now') to align with the bar that triggered the signal
            signal = TradeSignal(
                timestamp=bar_timestamp,
                price=current_price,
                side="BUY",
                trend_score=trend_score,
                di_plus=di_plus,
                di_minus=di_minus,
            )
            self._signals.append(signal)
            self._current_signal_state = "BUY"
        
        # Generate sell signal on flip to bearish
        elif sell_signal and self._current_signal_state != "SELL":
            ctx.log(f"SELL signal: Trend flipped to bearish (neg={self._negative_count}, pos={self._positive_count}, prev_pos={self._prev_positive_count})")
            
            # Close position if we have one
            if has_position:
                for pos in positions:
                    if pos.symbol == self._symbol and pos.quantity > 0:
                        # Calculate realized P&L before closing
                        cost_basis = pos.avg_price * pos.quantity
                        sale_proceeds = current_price * pos.quantity
                        realized_pnl = sale_proceeds - cost_basis
                        realized_pnl_pct = ((current_price / pos.avg_price) - 1.0) * 100 if pos.avg_price > 0 else 0.0
                        
                        order = Order(
                            id=f"mp_sell_{now.timestamp()}",
                            symbol=self._symbol,
                            side=Side.SELL,
                            quantity=pos.quantity,
                            order_type=OrderType.MARKET,
                            limit_price=None,
                            instrument_type=InstrumentType.STOCK,
                        )
                        await ctx.execution.submit_order(order)  # type: ignore[attr-defined]
                        ctx.log(f"  Closing position: {pos.quantity} shares @ ${current_price:.2f} (cost basis: ${pos.avg_price:.2f})")
                        ctx.log(f"  Realized P&L: ${realized_pnl:,.2f} ({realized_pnl_pct:+.2f}%)")
                        break
            else:
                ctx.log(f"  No position to close (already flat)")
            
            # Store signal for visualization (even if no position to close)
            # Use the bar's timestamp (not 'now') to align with the bar that triggered the signal
            signal = TradeSignal(
                timestamp=bar_timestamp,
                price=current_price,
                side="SELL",
                trend_score=trend_score,
                di_plus=di_plus,
                di_minus=di_minus,
            )
            self._signals.append(signal)
            self._current_signal_state = "SELL"
    
    async def on_end(self, ctx: Context) -> None:
        """Cleanup on strategy end."""
        ctx.log(f"MysticPulseStrategy finished. Final counts: pos={self._positive_count}, neg={self._negative_count}")
    
    def get_signals(self) -> List[TradeSignal]:
        """Get all trading signals for visualization."""
        return self._signals
    
    def get_indicator_history(self) -> List[IndicatorData]:
        """Get indicator history for visualization."""
        return self._indicator_history
    
    def get_bars(self) -> List[Bar]:
        """Get all bars used in calculations."""
        return self._bars

