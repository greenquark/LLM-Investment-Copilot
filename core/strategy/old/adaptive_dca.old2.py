"""
Adaptive DCA Strategy - Fear & Greed Index Based

A dollar-cost averaging strategy that adapts contributions based solely on:
- CNN Fear & Greed Index (0-100) to gauge market sentiment

Strategy Logic:
- Adaptive DCA Strategy (FGI-driven) with configurable base_buy_fraction
- Regular contributions: weekly, monthly, quarterly, or yearly, e.g. $1000 per week regardless of Fear & Greed Index.
- Buy/Sell actions happen during the contribution day or the day after, same as regular DCA
- Buy/Sell decisions:
  * Fear (FGI 0-45): Buy base_buy_fraction of contribution amount + proportional of remaining cash
    - Buy amount = (base_buy_fraction * contribution_amount) + (remaining_cash * buy_multiplier)
    - Formula: buy_multiplier = (45 - fgi_value) / 45
    - At FGI=0: buy_multiplier = 1.0 (buy 100% of remaining cash)
    - At FGI=45: buy_multiplier = 0.0 (buy 0% of remaining cash)
    - Linear scaling between FGI 0 and 45, e.g. if FGI=30, buy_multiplier = (45 - 30) / 45 = 0.33
  * Neutral/Greed (FGI 46-75): Buy base_buy_fraction of contribution amount, leave the remaining cash in cash
    - Buy amount = (base_buy_fraction * contribution_amount)
    - Remaining cash: (1 - base_buy_fraction) * contribution_amount
  * Extreme Greed (FGI 76-100): Do not contribute. Sell a portion of the position, proportional to FGI value
    - Sell amount = (sell_fraction * position_quantity)
    - Formula: sell_fraction = max_sell_fraction * (fgi_value - fgi_extreme_greed_min) / (100 - fgi_extreme_greed_min)
    - At FGI=fgi_extreme_greed_min: sell_fraction = 0.0 (no selling)
    - At FGI=100: sell_fraction = max_sell_fraction (sell max_sell_fraction% of position quantity)

FGI Categories:
- 0-45:   Fear (buy base_buy_fraction of contribution amount + proportional of remaining cash)
- 46-75:  Neutral/Greed (buy base_buy_fraction of contribution amount, leave the remaining cash in cash)
- 76-100: Extreme Greed (do not contribute. Sell a portion of the position, proportional to FGI value, up to max_sell_fraction)

Notes:
- CNN Fear & Greed data loaded from core.data.fear_greed_index module
- Uses CSV file first, then CNN API, then current value as fallback
"""


# Adaptive DCA Strategy (FGI-driven) with configurable base_buy_fraction
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import Optional, Dict, List
import logging

from core.strategy.base import Strategy, Context
from core.data.base import DataEngine
from core.data.fear_greed_index import get_fgi_value
from core.models.bar import Bar
from core.models.order import Order, Side, OrderType, InstrumentType
from core.models.position import Position

logger = logging.getLogger(__name__)


# Valid timeframes for AdaptiveDCA strategy
VALID_TIMEFRAMES = {"1D", "1W", "1Q", "1Y"}


@dataclass
class AdaptiveDCAConfig:
    """Configuration for Adaptive DCA Strategy."""
    contribution_amount: float = 1000.0

    # NEW: configurable base fraction (replaces hard-coded 90%)
    base_buy_fraction: float = 0.95

    fgi_fear_max: int = 45
    fgi_greed_max: int = 75
    fgi_extreme_greed_min: int = 76
    
    # Max sell fraction for extreme greed (0.0 to 1.0, e.g. 0.2 = 20% max)
    max_sell_fraction: float = 0.2
    
    # Sell only once per week during extreme greed
    sell_once_per_week: bool = True

    fee_per_trade: float = 0.0
    
    # Timeframe: Only "1D", "1W", "1Q", or "1Y" are allowed
    timeframe: str = "1D"
    
    # Lookback period in days for fetching historical bars
    # Defaults based on timeframe: 5D (trading week), 20D (trading month), 60D (trading quarter), 252D (trading year)
    lookback_days: int = 5
    
    # Contribution frequency: "weekly", "monthly", "quarterly", "yearly"
    contribution_frequency: str = "weekly"

    @classmethod
    def from_dict(cls, config: dict) -> "AdaptiveDCAConfig":
        timeframe = config.get("timeframe", "1D")
        if timeframe not in VALID_TIMEFRAMES:
            raise ValueError(
                f"Invalid timeframe '{timeframe}'. AdaptiveDCA strategy only supports: {', '.join(sorted(VALID_TIMEFRAMES))}"
            )
        
        # Set default lookback_days based on timeframe if not specified
        default_lookback = {
            "1D": 5,   # Trading week (5 trading days)
            "1W": 20,  # Trading month (~4 weeks)
            "1Q": 60,  # Trading quarter (~3 months)
            "1Y": 252  # Trading year (~252 trading days)
        }
        lookback_days = config.get("lookback_days", default_lookback.get(timeframe, 5))
        
        # Ensure numeric values are properly converted to floats
        # Strip whitespace and special characters that might come from YAML parsing
        base_buy_frac = config.get("base_buy_fraction", 0.90)
        if isinstance(base_buy_frac, str):
            base_buy_frac = float(base_buy_frac.strip().rstrip('`').rstrip("'").rstrip('"'))
        elif isinstance(base_buy_frac, int):
            base_buy_frac = float(base_buy_frac)
        
        max_sell_frac = config.get("max_sell_fraction", 0.2)
        if isinstance(max_sell_frac, str):
            max_sell_frac = float(max_sell_frac.strip().rstrip('`').rstrip("'").rstrip('"'))
        elif isinstance(max_sell_frac, int):
            max_sell_frac = float(max_sell_frac)
        
        return cls(
            contribution_amount=config.get("weekly_contribution", config.get("contribution_amount", 1000.0)),
            base_buy_fraction=base_buy_frac,
            fgi_fear_max=config.get("fgi_fear_max", 45),
            fgi_greed_max=config.get("fgi_greed_max", 75),
            fgi_extreme_greed_min=config.get("fgi_extreme_greed_min", 76),
            max_sell_fraction=max_sell_frac,
            sell_once_per_week=config.get("sell_once_per_week", True),
            fee_per_trade=config.get("fee_per_trade", 0.0),
            timeframe=timeframe,
            lookback_days=lookback_days,
            contribution_frequency=config.get("contribution_frequency", "weekly")
        )

class AdaptiveDCAStrategy(Strategy):
    def __init__(self, symbol: str, config: AdaptiveDCAConfig, data_engine: DataEngine):
        # Validate timeframe
        if config.timeframe not in VALID_TIMEFRAMES:
            raise ValueError(
                f"Invalid timeframe '{config.timeframe}'. AdaptiveDCA strategy only supports: {', '.join(sorted(VALID_TIMEFRAMES))}"
            )
        
        self.symbol = symbol
        self.config = config
        self.data_engine = data_engine

        self._last_week_key: Optional[tuple[int, int]] = None
        self._bars: List[Bar] = []
        self._initialized = False
        self._sold_this_week: bool = False  # Track if we've sold during current week
        self._current_week_key: Optional[tuple[int, int]] = None  # Track current week

    async def on_start(self, ctx: Context) -> None:
        """Initialize the strategy."""
        ctx.log(f"Starting Adaptive DCA strategy for {self.symbol}")
        ctx.log(
            f"Config: contribution_amount=${self.config.contribution_amount:,.2f}, "
            f"base_buy_fraction={self.config.base_buy_fraction:.1%}, "
            f"max_sell_fraction={self.config.max_sell_fraction:.1%}, "
            f"timeframe={self.config.timeframe}, lookback_days={self.config.lookback_days}, "
            f"FGI thresholds: Fear 0-{self.config.fgi_fear_max}, "
            f"Neutral/Greed {self.config.fgi_fear_max + 1}-{self.config.fgi_greed_max}, "
            f"Extreme Greed >= {self.config.fgi_extreme_greed_min}"
        )
        ctx.log("FGI data: Using core.data.fear_greed_index module")
        ctx.log("  - Priority: CSV file -> CNN API -> Current value")
        self._initialized = True
        ctx.log("Strategy initialized")

    async def on_decision(self, ctx: Context, now: datetime) -> None:
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
        # Daily bars are typically timestamped at 00:00:00, so requesting up to current time (e.g., 16:00:00)
        # might miss the bar for the current day
        if self.config.timeframe.upper() in ("D", "1D"):
            # Request up to end of current day (23:59:59) instead of current time
            end_time = datetime.combine(current_date, datetime.max.time())
        else:
            # For other timeframes, use current time
            end_time = now
        
        # Use the configured timeframe and lookback period (validated in __init__)
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

        bar = bars[-1]
        price = bar.close
        current_date = bar.timestamp.date()

        fgi_value, _ = get_fgi_value(target_date=current_date)
        if fgi_value is None:
            raise RuntimeError(
                f"FGI (Fear & Greed Index) data unavailable for {current_date} (trading day). "
                f"Cannot make trading decision without FGI value. "
                f"Check FGI data source (CSV file, CNN API, or current value fallback)."
            )

        iso_year, iso_week, _ = current_date.isocalendar()
        week_key = (iso_year, iso_week)
        # First week: _last_week_key is None, so is_new_week = True
        # Subsequent weeks: is_new_week = True only when week_key changes
        is_new_week = (self._last_week_key is None) or (self._last_week_key != week_key)
        if is_new_week:
            self._last_week_key = week_key

        # Reset sell flag if we're in a new week (for sell_once_per_week logic)
        if self._current_week_key is None or week_key != self._current_week_key:
            self._sold_this_week = False
            self._current_week_key = week_key

        position = ctx.portfolio.state.positions.get(self.symbol)
        shares = position.quantity if position else 0.0

        # Determine FGI category and action for logging
        category = None
        action = None

        # ----- SELL in extreme greed -----
        # Extreme Greed (FGI 76-100): Contribute as usual. Sell a portion of the position, proportional to FGI value
        # Formula: sell_fraction = max_sell_fraction * (fgi_value - fgi_extreme_greed_min) / (100 - fgi_extreme_greed_min)
        # At FGI=fgi_extreme_greed_min: sell_fraction = 0.0 (no selling)
        # At FGI=100: sell_fraction = max_sell_fraction (sell max_sell_fraction% of position quantity)
        if fgi_value >= self.config.fgi_extreme_greed_min and shares > 0:
            # Check if we should sell only once per week
            if self.config.sell_once_per_week and self._sold_this_week:
                # Already sold this week, skip selling
                return
            category = "Extreme Greed"
            action = "SELL_PROPORTIONAL"
            # Calculate sell fraction: max_sell_fraction * (fgi_value - fgi_extreme_greed_min) / (100 - fgi_extreme_greed_min)
            # Range: 0.0 at FGI=fgi_extreme_greed_min to max_sell_fraction at FGI=100
            sell_frac = self.config.max_sell_fraction * (fgi_value - self.config.fgi_extreme_greed_min) / (100 - self.config.fgi_extreme_greed_min)
            sell_frac = max(0.0, min(self.config.max_sell_fraction, sell_frac))  # Clamp to [0, max_sell_fraction]
            qty = shares * sell_frac  # Support fractional shares
            if qty > 0:
                # Mark that we've sold this week (if sell_once_per_week is enabled)
                if self.config.sell_once_per_week:
                    self._sold_this_week = True
                
                # Record decision using data structure (if logger supports it)
                if hasattr(ctx.logger, 'record_decision'):
                    ctx.logger.record_decision(now, fgi_value, action, price)
                
                # Log for debugging (optional)
                ctx.log(f"FGI Category: {category} ({fgi_value:.2f}), Action: {action}")
                ctx.log(f"Selling {qty} shares @ ${price:.2f}")
                
                # Log portfolio state before sell
                sell_proceeds = qty * price
                current_cash = ctx.portfolio.state.cash
                current_shares = shares
                ctx.log(f"Before sell: Cash=${current_cash:,.2f}, Shares={current_shares:.2f}")
                
                # Submit order - portfolio state will be updated by backtest engine via apply_fill
                await ctx.execution.submit_order(
                    Order("", self.symbol, Side.SELL, qty, OrderType.MARKET, None, InstrumentType.STOCK)
                )
                
                # Log expected portfolio state after sell
                expected_cash = current_cash + sell_proceeds
                expected_shares = current_shares - qty
                ctx.log(f"After sell: Cash=${expected_cash:,.2f}, Shares={expected_shares:.2f}")
            return

        if not is_new_week:
            return

        # ----- WEEKLY CONTRIBUTION -----
        ctx.portfolio.state.cash += self.config.contribution_amount
        # Get actual cash balance from portfolio (after adding weekly contribution)
        available_cash = ctx.portfolio.state.cash

        base_buy = self.config.contribution_amount * self.config.base_buy_fraction
        min_remaining = self.config.contribution_amount * (1 - self.config.base_buy_fraction)

        buy_amount = 0.0
        if fgi_value <= self.config.fgi_fear_max:
            category = "Fear" if fgi_value > 25 else "Extreme Fear"
            action = "BUY_FEAR"
            multiplier = (self.config.fgi_fear_max - fgi_value) / self.config.fgi_fear_max
            multiplier = max(0.0, min(1.0, multiplier))
            # For fear: buy base_buy_fraction of weekly + proportional of remaining cash (cash from previous weeks)
            # remaining_cash = cash balance BEFORE this week's contribution
            remaining_cash = available_cash - self.config.contribution_amount
            buy_from_weekly = base_buy  # base_buy_fraction of weekly contribution
            buy_from_remaining = remaining_cash * multiplier  # Proportional of accumulated cash
            buy_amount = buy_from_weekly + buy_from_remaining
        elif fgi_value <= self.config.fgi_greed_max:
            category = "Neutral" if fgi_value <= 55 else "Greed"
            action = "BUY_NEUTRAL_GREED"
            buy_amount = base_buy
        else:
            return

        buy_amount = min(buy_amount, available_cash - min_remaining)
        
        # Calculate maximum affordable shares based on available cash (ensuring we keep min_remaining)
        # Support fractional shares
        max_affordable_qty = (available_cash - min_remaining) / price
        if max_affordable_qty <= 0:
            return
        
        # Calculate desired shares from buy_amount (fractional shares supported)
        desired_qty = buy_amount / price
        
        # Use the minimum of desired and affordable to prevent overspending
        qty = min(desired_qty, max_affordable_qty)
        if qty <= 0:
            return
        
        # Calculate actual cost to ensure we don't exceed available cash
        actual_cost = qty * price
        
        # Final safety check: ensure we have enough cash
        if actual_cost > available_cash - min_remaining:
            # Adjust qty down to exactly what we can afford
            qty = (available_cash - min_remaining) / price
            actual_cost = qty * price

        # Record decision using data structure (if logger supports it)
        if hasattr(ctx.logger, 'record_decision'):
            ctx.logger.record_decision(now, fgi_value, action, price)
        
        # Log for debugging (optional)
        ctx.log(f"FGI Category: {category} ({fgi_value:.2f}), Action: {action}")
        
        # Log portfolio state before buy
        current_cash = ctx.portfolio.state.cash
        current_shares = shares
        buy_cost = qty * price
        ctx.log(f"Before buy: Cash=${current_cash:,.2f}, Shares={current_shares:.2f}")
        ctx.log(f"Buying {qty} shares @ ${price:.2f} (Cost=${buy_cost:,.2f})")

        # Submit order - portfolio state will be updated by backtest engine via apply_fill
        # The engine processes fills after on_decision and updates positions/cash automatically
        await ctx.execution.submit_order(
            Order("", self.symbol, Side.BUY, qty, OrderType.MARKET, None, InstrumentType.STOCK)
        )
        
        # Log expected portfolio state after buy
        expected_cash = current_cash - buy_cost
        expected_shares = current_shares + qty
        ctx.log(f"After buy: Cash=${expected_cash:,.2f}, Shares={expected_shares:.2f}")

    async def on_end(self, ctx: Context) -> None:
        """Clean up when strategy ends."""
        ctx.log(f"Ending Adaptive DCA strategy for {self.symbol}")
        
        # Log final position
        position = ctx.portfolio.state.positions.get(self.symbol)
        if position and position.quantity > 0:
            current_price = self._bars[-1].close if self._bars else 0.0
            position_value = position.quantity * current_price
            ctx.log(f"Final position: {position.quantity:.4f} shares @ ${current_price:.2f} = ${position_value:,.2f}")
        
        # Log portfolio state
        if self._bars:
            current_price = self._bars[-1].close
            prices = {self.symbol: current_price}
            portfolio_value = ctx.portfolio.equity(prices)
        else:
            portfolio_value = ctx.portfolio.state.cash
        
        ctx.log(f"Final portfolio value: ${portfolio_value:,.2f}")
