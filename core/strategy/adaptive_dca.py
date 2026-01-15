"""
Adaptive DCA Strategy - Fear & Greed Index Based

A dollar-cost averaging strategy that adapts contributions based solely on:
- CNN Fear & Greed Index (0-100) to gauge market sentiment

FGI Categories: 

Definition of CNN's Fear & Greed Index
CNN's Fear & Greed Index is a tool that measures the emotions driving investor behavior in the stock market. It operates on a scale from 0 to 100, where:

0-25.00 inclusive: Extreme Fear
25-45.00 inclusive: Fear
45-55 inclusive: Neutral
55-75 inclusive: Greed
75-100 inclusive: Extreme Greed

Sources for the Fear & Greed Index:
- CNN Fear & Greed Index: https://money.cnn.com/data/fear-and-greed/
- CNN Fear & Greed Index API from CNN:
    - https://production.dataviz.cnn.io/index/fearandgreed/graphdata/YYYY-MM-DD
    - Example: https://production.dataviz.cnn.io/index/fearandgreed/graphdata/2025-12-29
- Historical chart https://www.finhacker.cz/en/fear-and-greed-index-historical-data-and-chart/
- CNN Fear & Greed Index CSV file: https://github.com/whit3rabbit/fear-greed-data. It is not super accurate. 
    - Example: https://github.com/whit3rabbit/fear-greed-data/blob/main/fear-greed-2011-2023.csv
- By the way, this is for Gold only: https://www.jmbullion.com/fear-greed-index/


Strategy Logic:
- Adaptive DCA Strategy (FGI-driven) with configurable base_buy_fraction
- Regular contributions: weekly, monthly, quarterly, or yearly, e.g. $1000 per week regardless of Fear & Greed Index.
- Buy/Sell actions happen during the contribution day or the day after, same as regular DCA
- Buy/Sell decisions:
  * Fear & Extreme Fear Buy base_buy_fraction of contribution amount + proportional of remaining cash
    - Buy amount = (base_buy_fraction * contribution_amount) + (remaining_cash * buy_multiplier)
    - Formula: buy_multiplier = (45 - fgi_value) / 45
    - At FGI=0: buy_multiplier = 1.0 (buy 100% of remaining cash)
    - At FGI=45: buy_multiplier = 0.0 (buy 0% of remaining cash)
    - Linear scaling between FGI 0 and 45, e.g. if FGI=30, buy_multiplier = (45 - 30) / 45 = 0.33
  * Neutral & Greed : Buy base_buy_fraction of contribution amount, leave the remaining cash in cash
    - Buy amount = (base_buy_fraction * contribution_amount)
    - Remaining cash: (1 - base_buy_fraction) * contribution_amount
  * Extreme Greed : Do not contribute. Sell a portion of the position, proportional to FGI value
    - Sell amount = (sell_fraction * position_quantity)
    - Formula: sell_fraction = max_sell_fraction * (fgi_value - fgi_extreme_greed_min) / (100 - fgi_extreme_greed_min)
    - At FGI=fgi_extreme_greed_min: sell_fraction = 0.0 (no selling)
    - At FGI=100: sell_fraction = max_sell_fraction (sell max_sell_fraction% of position quantity)



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
import math

from core.strategy.base import Strategy, Context
from core.strategy.contributions import ContributionManager
from core.data.base import DataEngine
from core.models.bar import Bar
from core.models.order import Order, Side, OrderType, InstrumentType
from core.models.position import Position
from core.visualization.models import TradeSignal

logger = logging.getLogger(__name__)


# Valid timeframes for AdaptiveDCA strategy
VALID_TIMEFRAMES = {"1D", "1W", "1Q", "1Y"}


@dataclass
class AdaptiveDCAConfig:
    """Configuration for Adaptive DCA Strategy."""
    contribution_amount: float = 2000.0

    # NEW: configurable base fraction (replaces hard-coded 90%)
    base_buy_fraction: float = 0.95

    # FGI thresholds (all inclusive ranges)
    fgi_extreme_fear_max: float = 25.00  # Extreme Fear range: 0 <= fgi_value <= 25.00
    fgi_fear_max: float = 45.00  # Fear range: 25.00 < fgi_value <= 45.00
    fgi_neutral_max: float = 55.00  # Neutral range: 45.00 < fgi_value <= 55.00
    fgi_greed_max: float = 75.00  # Greed range: 55.00 < fgi_value < 75.00 (exclusive upper bound)
    fgi_extreme_greed_min: float = 75.00  # Extreme Greed range: 75.00 <= fgi_value <= 100.00
    fgi_max: float = 100.00  # Maximum FGI value (used in calculations)
    
    # Max sell fraction for extreme greed (0.0 to 1.0, e.g. 0.2 = 20% max)
    max_sell_fraction: float = 0.2
    
    # Sell only once per week during extreme greed
    sell_once_per_week: bool = True
    
    # Action names for logging and decision tracking (configurable)
    action_extreme_fear: str = "BUY_FEAR"
    action_fear: str = "BUY_FEAR"
    action_neutral: str = "BUY_NEUTRAL_GREED"
    action_greed: str = "BUY_NEUTRAL_GREED"
    action_extreme_greed_sell: str = "SELL_PROPORTIONAL"
    action_extreme_greed_hold: str = "HOLD"
    action_hold: str = "HOLD"

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
            fgi_extreme_fear_max=float(config.get("fgi_extreme_fear_max", 25.00)),
            fgi_fear_max=float(config.get("fgi_fear_max", 45.00)),
            fgi_neutral_max=float(config.get("fgi_neutral_max", 55.00)),
            fgi_greed_max=float(config.get("fgi_greed_max", 75.00)),
            fgi_extreme_greed_min=float(config.get("fgi_extreme_greed_min", 75.00)),
            fgi_max=float(config.get("fgi_max", 100.00)),
            max_sell_fraction=max_sell_frac,
            sell_once_per_week=config.get("sell_once_per_week", True),
            # Action names
            action_extreme_fear=config.get("action_extreme_fear", "BUY_FEAR"),
            action_fear=config.get("action_fear", "BUY_FEAR"),
            action_neutral=config.get("action_neutral", "BUY_NEUTRAL_GREED"),
            action_greed=config.get("action_greed", "BUY_NEUTRAL_GREED"),
            action_extreme_greed_sell=config.get("action_extreme_greed_sell", "SELL_PROPORTIONAL"),
            action_extreme_greed_hold=config.get("action_extreme_greed_hold", "HOLD"),
            action_hold=config.get("action_hold", "HOLD"),
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

        # === Contribution schedule & trade-window state (spec-aligned) ===
        # Track contribution periods for weekly/monthly/quarterly/yearly contributions.
        self._last_period_key: Optional[tuple[int, int]] = None  # (year, period_id)
        self._last_contribution_date: Optional[date] = None

        # Trade window: allow ONE trade per contribution period, on contribution day or the next trading day.
        self._last_trade_period_key: Optional[tuple[int, int]] = None

        # Track which periods have had decisions recorded (to ensure all periods are logged)
        self._recorded_periods: set[tuple[int, int]] = set()

        # Cache contribution_date -> next trading day (bar date) lookup to avoid repeated API calls.
        self._next_trading_day_cache: Dict[date, Optional[date]] = {}
        
        # Store end date for final price lookup (set by backtest script)
        self._end_date: Optional[date] = None
        
        # Track contributions accurately (count and total amount)
        self._contribution_count: int = 0
        self._total_contributions: float = 0.0
        
        # Use ContributionManager for period calculations
        self._contribution_manager = ContributionManager(self.config.contribution_frequency)
        
        # Track signals for visualization
        self._signals: List[TradeSignal] = []
    
    @property
    def total_contributions(self) -> float:
        """Return the total amount of contributions made during the backtest."""
        return self._total_contributions
    
    @property
    def contribution_count(self) -> int:
        """Return the number of contributions made during the backtest."""
        return self._contribution_count

    async def _get_next_trading_day(self, contrib_date: date) -> Optional[date]:
        """Find the next trading day after contrib_date by looking for the next available bar date."""
        if contrib_date in self._next_trading_day_cache:
            return self._next_trading_day_cache[contrib_date]

        # Look ahead up to 10 calendar days for the next bar date.
        start_dt = datetime.combine(contrib_date + timedelta(days=1), datetime.min.time())
        end_dt = start_dt + timedelta(days=10)

        try:
            bars = await self.data_engine.get_bars(self.symbol, start_dt, end_dt, timeframe=self.config.timeframe)
        except Exception:
            bars = None

        next_day = None
        if bars:
            bars = sorted(bars, key=lambda b: b.timestamp)
            next_day = bars[0].timestamp.date()

        self._next_trading_day_cache[contrib_date] = next_day
        return next_day

    async def on_start(self, ctx: Context) -> None:
        """Initialize the strategy."""
        ctx.log(f"Starting Adaptive DCA strategy for {self.symbol}")
        ctx.log(
            f"Config: contribution_amount=${self.config.contribution_amount:,.2f}, "
            f"base_buy_fraction={self.config.base_buy_fraction:.1%}, "
            f"max_sell_fraction={self.config.max_sell_fraction:.1%}, "
            f"timeframe={self.config.timeframe}, lookback_days={self.config.lookback_days}, "
            f"FGI thresholds: Extreme Fear 0-{self.config.fgi_extreme_fear_max}, "
            f"Fear {self.config.fgi_extreme_fear_max}-{self.config.fgi_fear_max}, "
            f"Neutral {self.config.fgi_fear_max}-{self.config.fgi_neutral_max}, "
            f"Greed {self.config.fgi_neutral_max}-{self.config.fgi_greed_max}, "
            f"Extreme Greed >= {self.config.fgi_extreme_greed_min}"
        )
        ctx.log("FGI data: Using core.data.fear_greed_index module")
        ctx.log("  - Priority: CSV file -> CNN API -> Current value")
        self._initialized = True
        ctx.log("Strategy initialized")

    async def on_decision(self, ctx: Context, now: datetime) -> None:
        from core.strategy.strategy_utils import check_trading_day, get_end_time_for_timeframe
        
        # Check if this is a trading day - skip weekends/holidays silently
        current_date = now.date()
        is_td = check_trading_day(current_date)
        if not is_td:
            # Not a trading day (weekend/holiday) - skip silently
            # However, check if this is a new contribution period and record it
            # This ensures all contribution periods appear in the weekly actions table
            period_key = self._contribution_manager.period_key(current_date)
            is_new_period = self._contribution_manager.is_new_period(current_date, self._last_period_key)
            if is_new_period and hasattr(ctx.logger, 'record_decision'):
                # This is a new contribution period on a non-trading day
                # Record a HOLD decision so the period appears in the table
                # The actual contribution will happen on the next trading day
                # Don't fetch FGI for non-trading days - use default neutral value
                if period_key not in self._recorded_periods:
                    fgi_value = 50.0  # Default to neutral for non-trading days
                    ctx.logger.record_decision(now, fgi_value, self.config.action_hold, 0.0)  # Price will be updated on next trading day
                    self._recorded_periods.add(period_key)
            return
        
        # Get appropriate end time for bar fetching
        end_time = get_end_time_for_timeframe(now, current_date, self.config.timeframe)
        
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

        # Fetch FGI value - this MUST be available for trading days
        try:
            fgi_value, fgi_classification = await ctx.fgi_provider.get_fgi(target_date=current_date)
        except Exception as e:
            # If get_fgi_value itself raises an exception, that's an error
            raise RuntimeError(
                f"FGI (Fear & Greed Index) data fetch failed for {current_date} (trading day). "
                f"Exception: {e}. "
                f"Cannot make trading decision without FGI value. "
                f"Check FGI data source (CSV file, CNN API, or current value fallback)."
            ) from e
        
        # Validate FGI value - must be a valid number between 0 and fgi_max
        if fgi_value is None:
            raise RuntimeError(
                f"FGI (Fear & Greed Index) data unavailable for {current_date} (trading day). "
                f"get_fgi_value returned None. "
                f"Cannot make trading decision without FGI value. "
                f"Check FGI data source (CSV file, CNN API, or current value fallback)."
            )
        
        # Check if FGI is a valid number (not NaN, not inf)
        if not isinstance(fgi_value, (int, float)) or math.isnan(fgi_value) or math.isinf(fgi_value):
            raise RuntimeError(
                f"FGI (Fear & Greed Index) value is invalid for {current_date} (trading day). "
                f"Got value: {fgi_value} (type: {type(fgi_value)}). "
                f"Expected a number between 0 and {self.config.fgi_max}. "
                f"Cannot make trading decision with invalid FGI value."
            )
        
        # Check if FGI is within valid range (0 to fgi_max)
        if fgi_value < 0 or fgi_value > self.config.fgi_max:
            raise RuntimeError(
                f"FGI (Fear & Greed Index) value out of range for {current_date} (trading day). "
                f"Got value: {fgi_value}. "
                f"Expected a number between 0 and {self.config.fgi_max}. "
                f"Cannot make trading decision with out-of-range FGI value."
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

        # Identify whether we should contribute this period
        period_key = self._contribution_manager.period_key(current_date)
        is_new_period = self._contribution_manager.is_new_period(current_date, self._last_period_key)

        # Always contribute on schedule regardless of FGI (per spec)
        if is_new_period:
            self._last_period_key = period_key
            self._last_contribution_date = current_date
            cash_before = ctx.portfolio.state.cash
            ctx.portfolio.state.cash += self.config.contribution_amount
            # Track contribution accurately
            self._contribution_count += 1
            self._total_contributions += self.config.contribution_amount
            cash_after = ctx.portfolio.state.cash
            ctx.log(f"[Contribution] Added ${self.config.contribution_amount:,.2f} on {current_date} (period_key={period_key}). Cash: ${cash_before:,.2f} -> ${cash_after:,.2f}")

        # Determine whether we are within the trade window: contrib day OR next trading day
        trade_allowed = False
        if self._last_contribution_date is not None:
            if current_date == self._last_contribution_date:
                trade_allowed = True
            else:
                next_td = await self._get_next_trading_day(self._last_contribution_date)
                if next_td is not None and current_date == next_td:
                    trade_allowed = True

        # Check if we've already traded in this period
        if self._last_trade_period_key == period_key:
            trade_allowed = False

        position = ctx.portfolio.state.positions.get(self.symbol)
        shares = position.quantity if position else 0.0

        # Helper function to record decision if not already recorded for this period
        def record_decision_if_needed(fgi_val: float, act: str, px: float):
            """Record decision for this contribution period. Always call record_decision to allow date updates."""
            if hasattr(ctx.logger, 'record_decision'):
                # Always call record_decision to allow it to update the date from non-trading day to trading day
                # Use current_date (from bar timestamp) instead of now (scheduler date) to ensure we use the actual trading day
                # Create a datetime from current_date with the time from now (or default to end of day)
                from datetime import time
                decision_timestamp = datetime.combine(current_date, now.time() if now else time(16, 0))
                ctx.logger.record_decision(decision_timestamp, fgi_val, act, px)
                # Track that we've recorded a decision for this period (but allow updates)
                if period_key not in self._recorded_periods:
                    self._recorded_periods.add(period_key)
        
        # Helper function to format FGI log message
        def format_fgi_log_message(cat: str, fgi_val: float, act: str, executed: bool, px: float) -> str:
            """Format FGI category log message. Show 'Trade executed already' if trade was already executed in this period."""
            if self._last_trade_period_key == period_key and not executed:
                # Trade already executed in this period, and we're not executing a new trade
                return f"FGI Category: {cat} ({fgi_val:.2f}), Price: ${px:.2f}, Trade executed already"
            else:
                return f"FGI Category: {cat} ({fgi_val:.2f}), Action: {act}"

        # Determine FGI category and action for logging
        category = None
        action = None
        trade_executed = False  # Track if we actually executed a trade

        # ----- SELL in extreme greed -----
        # Extreme Greed (FGI fgi_extreme_greed_min-fgi_max inclusive): Contribute as usual. Sell a portion of the position, proportional to FGI value
        # Formula: sell_fraction = max_sell_fraction * (fgi_value - fgi_extreme_greed_min) / (fgi_max - fgi_extreme_greed_min)
        # At FGI=fgi_extreme_greed_min: sell_fraction = 0.0 (no selling)
        # At FGI=fgi_max: sell_fraction = max_sell_fraction (sell max_sell_fraction% of position quantity)
        # Note: This logic runs even when shares == 0, resulting in qty = 0 which is handled in the else branch
        if fgi_value >= self.config.fgi_extreme_greed_min:
            category = "Extreme Greed"
            action = self.config.action_extreme_greed_sell
            # Trade window guard: only sell on contribution day or next trading day
            if not trade_allowed:
                # Trade window not allowed, but still record decision
                record_decision_if_needed(fgi_value, action, price)
                # Log FGI category/action (or "Trade executed already" if trade was already executed in this period)
                ctx.log(format_fgi_log_message(category, fgi_value, action, trade_executed, price))
                return
            # Check if we should sell only once per week
            if self.config.sell_once_per_week and self._sold_this_week:
                # Already sold this week, but still record decision for this period
                record_decision_if_needed(fgi_value, action, price)
                # Log FGI category/action (or "Trade executed already" if trade was already executed in this period)
                ctx.log(format_fgi_log_message(category, fgi_value, action, trade_executed, price))
                return
            # Calculate sell fraction: max_sell_fraction * (fgi_value - fgi_extreme_greed_min) / (fgi_max - fgi_extreme_greed_min)
            # Range: 0.0 at FGI=fgi_extreme_greed_min to max_sell_fraction at FGI=fgi_max
            sell_frac = self.config.max_sell_fraction * (fgi_value - self.config.fgi_extreme_greed_min) / (self.config.fgi_max - self.config.fgi_extreme_greed_min)
            sell_frac = max(0.0, min(self.config.max_sell_fraction, sell_frac))  # Clamp to [0, max_sell_fraction]
            qty = shares * sell_frac  # Support fractional shares
            if qty > 0:
                # Mark that we've sold this week (if sell_once_per_week is enabled)
                if self.config.sell_once_per_week:
                    self._sold_this_week = True
                
                # Record decision using data structure (if logger supports it)
                record_decision_if_needed(fgi_value, action, price)
                
                # Mark trade window consumed for this period (before logging, so log shows correct state)
                self._last_trade_period_key = period_key
                trade_executed = True
                
                # Log for debugging (optional)
                ctx.log(format_fgi_log_message(category, fgi_value, action, trade_executed, price))
                ctx.log(f"Selling {qty} shares @ ${price:.2f}")
                
                # Log portfolio state before sell
                sell_proceeds = qty * price
                current_cash = ctx.portfolio.state.cash
                current_shares = shares
                ctx.log(f"Before sell: Cash=${current_cash:,.2f}, Shares={current_shares:.2f}")

                # Create sell signal for visualization
                signal = TradeSignal(
                    timestamp=now,
                    price=price,
                    side="SELL",
                    trend_score=int(fgi_value),
                    di_plus=None,
                    di_minus=None,
                )
                self._signals.append(signal)
                
                # Submit order - portfolio state will be updated by backtest engine via apply_fill
                await ctx.execution.submit_order(
                    Order("", self.symbol, Side.SELL, qty, OrderType.MARKET, None, InstrumentType.STOCK)
                )
                
                # Log expected portfolio state after sell
                expected_cash = current_cash + sell_proceeds
                expected_shares = current_shares - qty
                ctx.log(f"After sell: Cash=${expected_cash:,.2f}, Shares={expected_shares:.2f}")
            else:
                # Log when sell quantity is 0 (e.g., FGI exactly at threshold)
                ctx.log(format_fgi_log_message(category, fgi_value, action, trade_executed, price))
                ctx.log(f"Sell quantity is 0 (FGI={fgi_value:.2f} at threshold={self.config.fgi_extreme_greed_min}, sell_fraction={sell_frac:.6f}). No sell order submitted.")
                record_decision_if_needed(fgi_value, action, price)
            return

        # Only trade within the allowed window (contribution day or the next trading day)
        if not trade_allowed:
            # Trade window not allowed, but still record decision for this period
            # Category determination based on new FGI definitions:
            if fgi_value <= self.config.fgi_extreme_fear_max:
                category = "Extreme Fear"
                action = self.config.action_extreme_fear
            elif fgi_value <= self.config.fgi_fear_max:  # fgi_extreme_fear_max < fgi_value <= fgi_fear_max
                category = "Fear"
                action = self.config.action_fear
            elif fgi_value <= self.config.fgi_neutral_max:  # fgi_fear_max < fgi_value <= fgi_neutral_max
                category = "Neutral"
                action = self.config.action_neutral
            elif fgi_value < self.config.fgi_extreme_greed_min:  # fgi_neutral_max < fgi_value < fgi_extreme_greed_min
                category = "Greed"
                action = self.config.action_greed
            else:  # fgi_value >= fgi_extreme_greed_min (Extreme Greed)
                category = "Extreme Greed"
                action = self.config.action_extreme_greed_hold
            record_decision_if_needed(fgi_value, action, price)
            # Log FGI category/action (or "Trade executed already" if trade was already executed in this period)
            ctx.log(format_fgi_log_message(category, fgi_value, action, trade_executed, price))
            return

        # Mark that we've consumed the trade window for this period
        self._last_trade_period_key = period_key

        # Cash already includes any scheduled contribution from above
        available_cash = ctx.portfolio.state.cash

        base_buy = self.config.contribution_amount * self.config.base_buy_fraction
        min_remaining = self.config.contribution_amount * (1 - self.config.base_buy_fraction)

        buy_amount = 0.0
        # Category determination based on new FGI definitions:
        # 0-fgi_extreme_fear_max inclusive: Extreme Fear
        # fgi_extreme_fear_max-fgi_fear_max inclusive: Fear
        # fgi_fear_max-fgi_neutral_max inclusive: Neutral
        # fgi_neutral_max-fgi_greed_max inclusive: Greed
        # fgi_extreme_greed_min-fgi_max inclusive: Extreme Greed
        if fgi_value <= self.config.fgi_extreme_fear_max:
            # Extreme Fear: buy base_buy_fraction of weekly + proportional of remaining cash
            category = "Extreme Fear"
            action = self.config.action_extreme_fear
            multiplier = (self.config.fgi_fear_max - fgi_value) / self.config.fgi_fear_max
            multiplier = max(0.0, min(1.0, multiplier))
            # For fear: buy base_buy_fraction of weekly + proportional of remaining cash (cash from previous weeks)
            # remaining_cash = cash balance BEFORE this week's contribution
            remaining_cash = available_cash - self.config.contribution_amount
            buy_from_weekly = base_buy  # base_buy_fraction of weekly contribution
            buy_from_remaining = remaining_cash * multiplier  # Proportional of accumulated cash
            buy_amount = buy_from_weekly + buy_from_remaining
        elif fgi_value <= self.config.fgi_fear_max:  # fgi_extreme_fear_max < fgi_value <= fgi_fear_max
            # Fear: buy base_buy_fraction of weekly + proportional of remaining cash
            category = "Fear"
            action = self.config.action_fear
            multiplier = (self.config.fgi_fear_max - fgi_value) / self.config.fgi_fear_max
            multiplier = max(0.0, min(1.0, multiplier))
            # For fear: buy base_buy_fraction of weekly + proportional of remaining cash (cash from previous weeks)
            # remaining_cash = cash balance BEFORE this week's contribution
            remaining_cash = available_cash - self.config.contribution_amount
            buy_from_weekly = base_buy  # base_buy_fraction of weekly contribution
            buy_from_remaining = remaining_cash * multiplier  # Proportional of accumulated cash
            buy_amount = buy_from_weekly + buy_from_remaining
        elif fgi_value <= self.config.fgi_neutral_max:  # fgi_fear_max < fgi_value <= fgi_neutral_max
            # Neutral: buy base_buy_fraction of contribution amount
            category = "Neutral"
            action = self.config.action_neutral
            buy_amount = base_buy
        elif fgi_value < self.config.fgi_extreme_greed_min:  # fgi_neutral_max < fgi_value < fgi_extreme_greed_min
            # Greed: buy base_buy_fraction of contribution amount
            category = "Greed"
            action = self.config.action_greed
            buy_amount = base_buy
        else:  # fgi_value >= fgi_extreme_greed_min (Extreme Greed - shouldn't reach here in buy path)
            # FGI beyond greed_max (shouldn't happen, but handle gracefully)
            category = "Extreme Greed"
            action = self.config.action_extreme_greed_hold
            record_decision_if_needed(fgi_value, action, price)
            # Log FGI category/action (or "Trade executed already" if trade was already executed in this period)
            ctx.log(format_fgi_log_message(category, fgi_value, action, trade_executed, price))
            return

        buy_amount = min(buy_amount, available_cash - min_remaining)
        
        # Calculate maximum affordable shares based on available cash (ensuring we keep min_remaining)
        # Support fractional shares
        max_affordable_qty = (available_cash - min_remaining) / price
        if max_affordable_qty <= 0:
            # Can't afford any shares, but still record decision
            record_decision_if_needed(fgi_value, action, price)
            # Log FGI category/action (or "Trade executed already" if trade was already executed in this period)
            ctx.log(format_fgi_log_message(category, fgi_value, action, trade_executed, price))
            return
        
        # Calculate desired shares from buy_amount (fractional shares supported)
        desired_qty = buy_amount / price
        
        # Use the minimum of desired and affordable to prevent overspending
        qty = min(desired_qty, max_affordable_qty)
        if qty <= 0:
            # Calculated qty is 0 or negative, but still record decision
            record_decision_if_needed(fgi_value, action, price)
            # Log FGI category/action (or "Trade executed already" if trade was already executed in this period)
            ctx.log(format_fgi_log_message(category, fgi_value, action, trade_executed, price))
            return
        
        # Calculate actual cost to ensure we don't exceed available cash
        actual_cost = qty * price
        
        # Final safety check: ensure we have enough cash
        if actual_cost > available_cash - min_remaining:
            # Adjust qty down to exactly what we can afford
            qty = (available_cash - min_remaining) / price
            actual_cost = qty * price

        # Record decision using data structure (if logger supports it)
        record_decision_if_needed(fgi_value, action, price)
        
        # Mark that we're executing a trade (before logging, so log shows correct state)
        trade_executed = True
        
        # Create buy signal for visualization
        signal = TradeSignal(
            timestamp=now,
            price=price,
            side="BUY",
            trend_score=int(fgi_value),
            di_plus=None,
            di_minus=None,
        )
        self._signals.append(signal)
        
        # Log for debugging (optional)
        ctx.log(format_fgi_log_message(category, fgi_value, action, trade_executed, price))
        
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
        
        # Get final price using utility function
        # Use the end date (last trading day) if set, otherwise use the last bar's date
        # This ensures consistency with the script's calculation
        final_date = None
        if self._end_date:
            # Use the end date (last trading day) set by the backtest script
            final_date = self._end_date
        elif self._bars:
            # Fallback to last bar's date if end date not set
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
        
        # Log total contributions count and amount
        ctx.log(f"Total contributions: {self._contribution_count} × ${self.config.contribution_amount:,.2f} = ${self._total_contributions:,.2f}")
    
    def get_signals(self) -> List[TradeSignal]:
        """Get list of trading signals for visualization."""
        return self._signals
