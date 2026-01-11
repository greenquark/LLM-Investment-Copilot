"""
Regular DCA Strategy - Simple Dollar-Cost Averaging

A simple dollar-cost averaging strategy that buys a fixed amount every period
regardless of market conditions. Never sells - just accumulates shares over time.

Strategy Logic:
- Regular contributions: weekly, monthly, quarterly, or yearly
- Each period: Buy 100% of contribution amount, cash position is always $0 after buy
- Example: With $1000 weekly contribution, buys $1000 worth of shares, leaves $0 cash
- This is the baseline comparison for AdaptiveDCA strategy
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict
import logging

from core.strategy.base import Strategy, Context
from core.strategy.contributions import ContributionManager
from core.data.base import DataEngine
from core.models.bar import Bar
from core.models.order import Order, Side, OrderType, InstrumentType
from core.data.cached_engine import CachedDataEngine

logger = logging.getLogger(__name__)

VALID_TIMEFRAMES = {"1D", "1W", "1Q", "1Y"}


@dataclass
class RegularDCAConfig:
    """Configuration for Regular DCA Strategy."""
    contribution_amount: float = 1000.0
    fee_per_trade: float = 0.0
    timeframe: str = "1D"
    lookback_days: int = 5
    contribution_frequency: str = "weekly"
    
    @classmethod
    def from_dict(cls, config: dict) -> "RegularDCAConfig":
        """Create config from dictionary (e.g., from YAML)."""
        timeframe = config.get("timeframe", "1D")
        if timeframe not in VALID_TIMEFRAMES:
            raise ValueError(
                f"Invalid timeframe '{timeframe}'. RegularDCA strategy only supports: {', '.join(sorted(VALID_TIMEFRAMES))}"
            )
        
        # Default lookback days based on timeframe
        default_lookback = {
            "1D": 5,
            "1W": 20,
            "1Q": 60,
            "1Y": 252
        }
        lookback_days = config.get("lookback_days", default_lookback.get(timeframe, 5))
        
        return cls(
            contribution_amount=float(config.get("contribution_amount", config.get("weekly_contribution", 1000.0))),
            fee_per_trade=float(config.get("fee_per_trade", 0.0)),
            timeframe=timeframe,
            lookback_days=lookback_days,
            contribution_frequency=config.get("contribution_frequency", "weekly")
        )


class RegularDCA(Strategy):
    """Regular DCA Strategy - Simple Dollar-Cost Averaging."""
    
    def __init__(self, symbol: str, config: RegularDCAConfig, data_engine: DataEngine):
        if config.timeframe not in VALID_TIMEFRAMES:
            raise ValueError(
                f"Invalid timeframe '{config.timeframe}'. RegularDCA strategy only supports: {', '.join(sorted(VALID_TIMEFRAMES))}"
            )
        
        self.symbol = symbol
        self.config = config
        self.data_engine = data_engine
        
        self._last_period_key: Optional[tuple[int, int]] = None
        self._bars: List[Bar] = []
        self._initialized = False
        self._total_contributions: float = 0.0
        self._contribution_count: int = 0
        self._end_date: Optional[date] = None
        
        # Use ContributionManager for period calculations
        self._contribution_manager = ContributionManager(self.config.contribution_frequency)
    
    @property
    def total_contributions(self) -> float:
        """Total contributions made so far."""
        return self._total_contributions
    
    @property
    def contribution_count(self) -> int:
        """Number of contributions made so far."""
        return self._contribution_count
    
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
        """Make trading decision - buy 100% of contribution."""
        from core.strategy.strategy_utils import check_trading_day, get_end_time_for_timeframe
        
        # Check if this is a trading day - skip weekends/holidays silently
        current_date = now.date()
        if not check_trading_day(current_date):
            # Not a trading day (weekend/holiday) - skip silently
            return
        
        # Get appropriate end time for bar fetching
        end_time = get_end_time_for_timeframe(now, current_date, self.config.timeframe)
        
        lookback_start = now - timedelta(days=self.config.lookback_days)
        bars = await self.data_engine.get_bars(
            self.symbol, lookback_start, end_time, timeframe=self.config.timeframe
        )
        
        # If no bars returned, try fetching just the current day as a fallback
        if not bars:
            ctx.log(f"No bars from range request ({lookback_start} to {end_time}), trying fallback: fetch current day only (bypassing cache)")
            day_start = datetime.combine(current_date, datetime.min.time())
            day_end = datetime.combine(current_date, datetime.max.time())
            
            if isinstance(self.data_engine, CachedDataEngine):
                date_key = (self.symbol.upper(), self.config.timeframe, current_date)
                if date_key in self.data_engine._no_data_ranges:
                    ctx.log(f"Clearing 'no data' marker for {current_date} to force fresh fetch")
                    del self.data_engine._no_data_ranges[date_key]
                
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
                if isinstance(self.data_engine, CachedDataEngine) and self.data_engine._cache_enabled and self.data_engine._cache:
                    try:
                        await self.data_engine._cache.save_bars(self.symbol, self.config.timeframe, bars)
                        import logging
                        logging.getLogger(__name__).debug(f"[Cache] Silently saved {len(bars)} fallback bar(s) to cache")
                    except Exception as e:
                        ctx.log(f"Failed to save fallback bars to cache: {e}")
                
                lookback_bars = await self.data_engine.get_bars(
                    self.symbol, lookback_start, day_start - timedelta(seconds=1), timeframe=self.config.timeframe
                )
                if lookback_bars:
                    bars = sorted(lookback_bars + bars, key=lambda b: b.timestamp)
                    ctx.log(f"Combined {len(lookback_bars)} lookback bars with {len(bars) - len(lookback_bars)} current day bar(s)")
                else:
                    bars = sorted(bars, key=lambda b: b.timestamp)
                    ctx.log(f"Using only current day bar(s), no lookback bars available")
            else:
                ctx.log(f"Fallback also failed: no bars for {current_date} even with direct base engine request")
        
        if not bars:
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
        
        # Check if it's a new contribution period
        period_key = self._contribution_manager.period_key(current_date)
        is_new_period = self._contribution_manager.is_new_period(current_date, self._last_period_key)
        
        if is_new_period:
            self._last_period_key = period_key
            # Add contribution
            ctx.portfolio.state.cash += self.config.contribution_amount
            self._total_contributions += self.config.contribution_amount
            self._contribution_count += 1
            
            available_cash = ctx.portfolio.state.cash
            
            # Buy 100% of available cash (all cash after adding contribution)
            if available_cash > 0 and price > 0:
                # Calculate shares to buy (fractional shares supported)
                qty = available_cash / price
                
                if qty > 0:
                    # Submit buy order
                    order = Order(
                        id="",  # Empty ID, will be assigned by execution engine
                        symbol=self.symbol,
                        side=Side.BUY,
                        quantity=qty,
                        order_type=OrderType.MARKET,
                        limit_price=None,  # Market order, no limit price
                        instrument_type=InstrumentType.STOCK,
                    )
                    await ctx.execution.submit_order(order)
                    
                    ctx.log(
                        f"Regular DCA: BUY {qty:.4f} shares @ ${price:.2f} "
                        f"(${available_cash:,.2f} = 100% of contribution ${self.config.contribution_amount:,.2f})"
                    )
        else:
            # If not a new period, no action for Regular DCA
            pass
    
    async def on_end(self, ctx: Context) -> None:
        """Clean up when strategy ends."""
        ctx.log(f"Regular DCA strategy ended. Total contributions: ${self._total_contributions:,.2f}")
        
        # Get final price using utility function
        # Use the end date if set, otherwise use the last bar's date
        final_date = None
        if self._end_date:
            final_date = self._end_date
        elif self._bars:
            final_date = self._bars[-1].timestamp.date()
        else:
            from datetime import date as date_type
            final_date = date_type.today()
        
        # Fetch final price for the date (using the same function as the script)
        from core.utils.price_utils import get_final_price
        current_price = await get_final_price(
            data_engine=self.data_engine,
            symbol=self.symbol,
            target_date=final_date,
            timeframe=self.config.timeframe,
            equity_curve=None,  # Strategy doesn't have access to equity curve
            final_shares=0.0,
            fallback_to_equity_curve=False,  # Don't use equity curve fallback in strategy
        )
        
        # Fallback to last bar if price fetch failed
        if current_price == 0.0 and self._bars:
            current_price = self._bars[-1].close
            final_date = self._bars[-1].timestamp.date()  # Update final_date for logging
            ctx.log(f"[Cache] Using last cached bar price: ${current_price:.2f} (date: {final_date})")
        elif current_price > 0:
            ctx.log(f"[Cache] Fetched final bar for {final_date}: ${current_price:.2f}")
        
        # Log final position
        position = ctx.portfolio.state.positions.get(self.symbol)
        if position and position.quantity > 0:
            position_value = position.quantity * current_price
            ctx.log(f"Final position: {position.quantity:.4f} shares @ ${current_price:.2f} = ${position_value:,.2f}")
        
        # Log portfolio state
        if current_price > 0:
            prices = {self.symbol: current_price}
            portfolio_value = ctx.portfolio.equity(prices)
        else:
            portfolio_value = ctx.portfolio.state.cash
        
        ctx.log(f"Final portfolio value: ${portfolio_value:,.2f}")

