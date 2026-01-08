"""
Regular DCA Strategy - Simple Dollar-Cost Averaging

A simple dollar-cost averaging strategy that buys a fixed amount every week
regardless of market conditions. Never sells - just accumulates shares over time.

Strategy Logic:
- Weekly contributions: fixed amount per week (configurable)
- Each week: Buy 100% of weekly contribution, cash position is always $0 after buy
- Example: With $1000 weekly contribution, buys $1000 worth of shares, leaves $0 cash
- This is the baseline comparison for AdaptiveDCA strategy
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import Optional, List
import logging

from core.strategy.base import Strategy, Context
from core.data.base import DataEngine
from core.models.bar import Bar
from core.models.order import Order, Side, OrderType, InstrumentType

logger = logging.getLogger(__name__)


# Valid timeframes for RegularDCA strategy
VALID_TIMEFRAMES = {"1D", "1W", "1Q", "1Y"}


@dataclass
class RegularDCAConfig:
    """Configuration for Regular DCA Strategy."""
    contribution_amount: float = 1000.0
    fee_per_trade: float = 0.0
    
    # Timeframe: Only "1D", "1W", "1Q", or "1Y" are allowed
    timeframe: str = "1D"
    
    # Lookback period in days for fetching historical bars
    # Defaults based on timeframe: 5D (trading week), 20D (trading month), 60D (trading quarter), 252D (trading year)
    lookback_days: int = 5
    
    # Contribution frequency: "weekly", "monthly", "quarterly", "yearly"
    contribution_frequency: str = "weekly"


class RegularDCAStrategy(Strategy):
    """Regular DCA Strategy - Simple Dollar-Cost Averaging."""
    
    def __init__(self, symbol: str, config: RegularDCAConfig, data_engine: DataEngine):
        # Validate timeframe
        if config.timeframe not in VALID_TIMEFRAMES:
            raise ValueError(
                f"Invalid timeframe '{config.timeframe}'. RegularDCA strategy only supports: {', '.join(sorted(VALID_TIMEFRAMES))}"
            )
        
        self.symbol = symbol
        self.config = config
        self.data_engine = data_engine

        self._last_week_key: Optional[tuple[int, int]] = None
        self._bars: List[Bar] = []
        self._initialized = False

    async def on_start(self, ctx: Context) -> None:
        """Initialize the strategy."""
        ctx.log(f"Starting Regular DCA strategy for {self.symbol}")
        ctx.log(
            f"Config: contribution_amount=${self.config.contribution_amount:,.2f}, "
            f"timeframe={self.config.timeframe}, lookback_days={self.config.lookback_days}, "
            f"contribution_frequency={self.config.contribution_frequency}"
        )
        self._initialized = True
        ctx.log("Strategy initialized")

    async def on_decision(self, ctx: Context, now: datetime) -> None:
        """Make trading decision - buy 100% of weekly contribution."""
        # Check if this is a trading day - skip weekends/holidays silently
        current_date = now.date()
        try:
            from core.data.trading_calendar import is_trading_day
            if not is_trading_day(current_date):
                # Not a trading day (weekend/holiday) - skip silently
                return
        except Exception:
            # If trading calendar check fails, continue (fallback behavior)
            pass

        # For daily bars, request up to end of current day to ensure we get today's bar
        if self.config.timeframe.upper() in ("D", "1D"):
            end_time = datetime.combine(current_date, datetime.max.time())
        else:
            end_time = now

        lookback_start = now - timedelta(days=self.config.lookback_days)
        bars = await self.data_engine.get_bars(
            self.symbol, lookback_start, end_time, timeframe=self.config.timeframe
        )

        # If no bars returned, try fetching just the current day as a fallback
        # This handles cases where the cached engine might have marked the date as "no data"
        # but the data is actually available (e.g., from a previous failed request with wrong time range)
        if not bars:
            ctx.log(f"No bars from range request ({lookback_start} to {end_time}), trying fallback: fetch current day only (bypassing cache)")
            # Try fetching just the current day, bypassing cache if using CachedDataEngine
            # This forces a fresh API call even if the date was previously marked as "no data"
            day_start = datetime.combine(current_date, datetime.min.time())
            day_end = datetime.combine(current_date, datetime.max.time())
            
            # Check if we're using a CachedDataEngine and can access the base engine directly
            # This bypasses the cache's "no data" markers
            from core.data.cached_engine import CachedDataEngine
            if isinstance(self.data_engine, CachedDataEngine):
                # Clear the "no data" marker for this date to force a fresh fetch
                date_key = (self.symbol.upper(), self.config.timeframe, current_date)
                if date_key in self.data_engine._no_data_ranges:
                    ctx.log(f"Clearing 'no data' marker for {current_date} to force fresh fetch")
                    del self.data_engine._no_data_ranges[date_key]
                
                # Access base engine directly to bypass cache
                base_engine = self.data_engine._base_engine
                ctx.log(f"Bypassing cache, using base engine directly for fallback")
                try:
                    bars = await base_engine.get_bars(
                        self.symbol, day_start, day_end, timeframe=self.config.timeframe
                    )
                except Exception as e:
                    ctx.log(f"Base engine fallback raised exception: {e}")
                    bars = None
            else:
                # Not using cached engine, try normal request
                try:
                    bars = await self.data_engine.get_bars(
                        self.symbol, day_start, day_end, timeframe=self.config.timeframe
                    )
                except Exception as e:
                    ctx.log(f"Fallback request raised exception: {e}")
                    bars = None
            
            if bars:
                ctx.log(f"Fallback succeeded: got {len(bars)} bar(s) for {current_date}, fetching lookback period")
                
                # Save fallback-fetched bars to cache silently (if using CachedDataEngine)
                from core.data.cached_engine import CachedDataEngine
                if isinstance(self.data_engine, CachedDataEngine) and self.data_engine._cache_enabled and self.data_engine._cache:
                    try:
                        await self.data_engine._cache.save_bars(self.symbol, self.config.timeframe, bars)
                        # Silent save (debug level only) for trading day data
                        import logging
                        logging.getLogger(__name__).debug(f"[Cache] Silently saved {len(bars)} fallback bar(s) to cache")
                    except Exception as e:
                        ctx.log(f"Failed to save fallback bars to cache: {e}")
                
                # Got data for current day - now fetch the lookback period
                lookback_bars = await self.data_engine.get_bars(
                    self.symbol, lookback_start, day_start - timedelta(seconds=1), timeframe=self.config.timeframe
                )
                # Combine lookback bars with current day bar
                if lookback_bars:
                    bars = sorted(lookback_bars + bars, key=lambda b: b.timestamp)
                    ctx.log(f"Combined {len(lookback_bars)} lookback bars with {len(bars) - len(lookback_bars)} current day bar(s)")
                else:
                    bars = sorted(bars, key=lambda b: b.timestamp)
                    ctx.log(f"Using only current day bar(s), no lookback bars available")
            else:
                ctx.log(f"Fallback also failed: no bars for {current_date} even with direct base engine request")
        
        if not bars:
            # Only error if this is a trading day - weekends/holidays are expected to have no bars
            raise RuntimeError(
                f"No bars available for {self.symbol} on {current_date} (trading day). "
                f"This indicates a data gap or missing data. "
                f"Requested bars from {lookback_start} to {end_time}. "
                f"Fallback single-day request also returned no bars."
            )
        
        bars = sorted(bars, key=lambda b: b.timestamp)
        self._bars = bars

        # Get the latest bar (current price)
        latest_bar = bars[-1]
        price = latest_bar.close

        if price <= 0:
            ctx.log(f"Invalid price: {price}, skipping decision")
            return

        # Check if this is a new week
        current_week_key = (current_date.isocalendar()[0], current_date.isocalendar()[1])
        is_new_week = (self._last_week_key is None or current_week_key != self._last_week_key)

        if not is_new_week:
            # Not a new week, skip
            return

        # Update week tracking
        self._last_week_key = current_week_key

        # Add weekly contribution
        # The portfolio will have cash from previous weeks (if any) plus this week's contribution
        contribution = self.config.contribution_amount
        ctx.portfolio.state.cash += contribution
        available_cash = ctx.portfolio.state.cash

        # Buy 100% of available cash (all cash after adding contribution)
        if available_cash > 0 and price > 0:
            # Calculate shares to buy (fractional shares supported)
            qty = available_cash / price
            
            if qty > 0:
                # Submit buy order (using same signature as AdaptiveDCA)
                await ctx.execution.submit_order(
                    Order("", self.symbol, Side.BUY, qty, OrderType.MARKET, None, InstrumentType.STOCK)
                )
                
                ctx.log(
                    f"Regular DCA: BUY {qty:.4f} shares @ ${price:.2f} "
                    f"(${available_cash:,.2f} = 100% of contribution ${contribution:,.2f})"
                )

    async def on_end(self, ctx: Context) -> None:
        """Clean up strategy."""
        ctx.log("Regular DCA strategy ended")

