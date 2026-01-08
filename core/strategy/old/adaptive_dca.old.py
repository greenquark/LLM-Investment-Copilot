"""
Adaptive DCA Strategy - Fear & Greed Index Based

A dollar-cost averaging strategy that adapts contributions based solely on:
- CNN Fear & Greed Index (0-100) to gauge market sentiment

Strategy Logic:
- Weekly contributions: $1000 per week
- Buy decisions:
  * Fear (FGI 0-45): Buy $900 (90% of weekly) + proportional of remaining cash
    - Buy amount = 900 + (remaining_cash * buy_multiplier)
    - Formula: buy_multiplier = (45 - fgi_value) / 45
    - At FGI=0: buy_multiplier = 1.0 (buy 100% of remaining cash)
    - At FGI=45: buy_multiplier = 0.0 (buy 0% of remaining cash)
    - Linear scaling between FGI 0 and 45
    - Example: With $100 remaining cash at FGI=30, buys $900 + ($100 * 0.33) = $933
  * Neutral/Greed (FGI 46-75): Buy $900 (90% of weekly contribution), leave $100 in cash
    - Buy amount = 0.9 * weekly_contribution = $900
    - Remaining cash: $100 (10% of weekly contribution)
  * Extreme Greed (FGI 76-100): May not contribute, or sell a portion
    - Sell up to 50% of position, proportional to FGI value
    - Formula: sell_fraction = 0.5 * (fgi_value - 76) / 24
    - At FGI=76: sell_fraction = 0.0 (no selling)
    - At FGI=100: sell_fraction = 0.5 (sell 50% of position)

FGI Categories:
- 0-45:   Fear (buy $900 + proportional of remaining cash)
- 46-75:  Neutral/Greed (buy $900, leave $100 in cash)
- 76-100: Extreme Greed (sell up to 50%, proportional to FGI)

Notes:
- CNN Fear & Greed data loaded from core.data.fear_greed_index module
- Uses CSV file first, then CNN API, then current value as fallback
"""

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

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveDCAConfig:
    """Configuration for Adaptive DCA strategy."""
    
    # Weekly contribution
    weekly_contribution: float = 1000.0

    # Fear & Greed Index thresholds
    fgi_fear_max: int = 45  # FGI 0-45: Fear (buy 90% weekly + proportional of remaining cash)
    fgi_greed_max: int = 75  # FGI 46-75: Neutral/Greed (buy 9% of weekly contribution)
    fgi_extreme_greed_min: int = 76  # FGI >= 76: Extreme Greed (sell)

    # Trading fees
    fee_per_trade: float = 0.0  # set if you want e.g. 1.0 dollars per trade

    @classmethod
    def from_dict(cls, config: dict) -> "AdaptiveDCAConfig":
        """Create config from dictionary (typically loaded from YAML)."""
        return cls(
            weekly_contribution=config.get("weekly_contribution", 1000.0),
            fgi_fear_max=config.get("fgi_fear_max", 45),
            fgi_greed_max=config.get("fgi_greed_max", 75),
            fgi_extreme_greed_min=config.get("fgi_extreme_greed_min", 76),
            fee_per_trade=config.get("fee_per_trade", 0.0),
        )


class AdaptiveDCAStrategy(Strategy):
    """
    Adaptive DCA Strategy - Fear & Greed Index Based.
    
    Implements dollar-cost averaging with adaptive contributions based solely on:
    - CNN Fear & Greed Index (market sentiment)
    
    Weekly contributions with buy/sell decisions:
    - Fear (FGI 0-45): Buy 90% of weekly contribution + proportional of remaining cash
    - Neutral/Greed (FGI 46-75): Buy 90% of weekly contribution
    - Extreme Greed (FGI 76-100): Sell up to 50% proportional to FGI
    """
    
    def __init__(
        self,
        symbol: str,
        config: AdaptiveDCAConfig,
        data_engine: DataEngine,
    ):
        """
        Initialize the strategy.
        
        Args:
            symbol: Trading symbol (e.g., "SOXL", "TQQQ")
            config: Strategy configuration
            data_engine: Data engine for fetching market data
        """
        self.symbol = symbol
        self.config = config
        self.data_engine = data_engine
        
        # Strategy state
        self._initialized = False
        self._bars: List[Bar] = []
        self._last_week: Optional[int] = None  # Track last week we contributed (week number)
        self._last_contribution_date: Optional[date] = None
        self._total_contributions: float = 0.0  # Track total contributions made
        self._last_fgi_value: Optional[float] = None  # Track last FGI value to avoid duplicate logging
        self._last_fgi_date: Optional[date] = None  # Track last date FGI was logged
        
    async def on_start(self, ctx: Context) -> None:
        """Initialize the strategy."""
        ctx.log(f"Starting Adaptive DCA strategy for {self.symbol}")
        ctx.log(
            f"Config: weekly_contribution=${self.config.weekly_contribution:,.2f}, "
            f"FGI thresholds: Fear 0-{self.config.fgi_fear_max} (90% weekly + proportional of remaining cash), "
            f"Neutral/Greed {self.config.fgi_fear_max + 1}-{self.config.fgi_greed_max} (90% weekly), "
            f"Extreme Greed >= {self.config.fgi_extreme_greed_min} (sell)"
        )
        
        # Note about FGI data availability
        ctx.log("FGI data: Using core.data.fear_greed_index module")
        ctx.log("  - Priority: CSV file -> CNN API -> Current value")
        
        self._initialized = True
        ctx.log("Strategy initialized")
    
    async def on_decision(self, ctx: Context, now: datetime) -> None:
        """Make trading decisions based on current market conditions."""
        if not self._initialized:
            ctx.log("Warning: Strategy not initialized, skipping decision")
            return
        
        # Fetch current bar for price
        try:
            bars = await self.data_engine.get_bars(
                self.symbol, now - timedelta(days=5), now, timeframe="1D"
            )
            if not bars:
                ctx.log(f"No bars available for {self.symbol}")
                return
            
            self._bars = bars
        except Exception as e:
            ctx.log(f"Error fetching bars: {e}")
            return
        
        # Get current bar
        current_bar = self._bars[-1]
        current_price = current_bar.close
        current_date = current_bar.timestamp.date()
        
        # Get CNN Fear & Greed Index using shared module
        # Priority: CSV file -> CNN API -> Current value
        fgi_value, fgi_classification = get_fgi_value(target_date=current_date)
        
        # Only log FGI if value changed or it's a new day
        # This avoids spamming the same value for historical backtest dates
        should_log_fgi = (
            fgi_value is not None and (
                self._last_fgi_value != fgi_value or 
                self._last_fgi_date != current_date
            )
        )
        
        if should_log_fgi:
            # Check if this is a historical date (more than 7 days ago)
            today = date.today()
            days_diff = (today - current_date).days
            
            if days_diff > 7:
                # Historical date - log with note about data source
                ctx.log(f"FGI (Fear & Greed Index): {fgi_value:.2f} / 100 (historical from CNN API)")
            else:
                # Recent date - log normally
                ctx.log(f"FGI (Fear & Greed Index): {fgi_value:.2f} / 100")
            
            if fgi_classification:
                ctx.log(f"FGI Classification: {fgi_classification}")
            
            self._last_fgi_value = fgi_value
            self._last_fgi_date = current_date
        elif fgi_value is None and self._last_fgi_date != current_date:
            # Only log once per day if FGI unavailable
            ctx.log("FGI: Not available (fear-and-greed package or CNN API unavailable)")
            self._last_fgi_date = current_date
        
        # Check if this is a new week (weekly contribution)
        # Use ISO week number to determine if it's a new week
        current_week = current_date.isocalendar()[1]  # ISO week number
        is_new_week = False
        if self._last_week is None or current_week != self._last_week:
            is_new_week = True
            self._last_week = current_week
            self._last_contribution_date = current_date
        
        # Get current position
        position = ctx.portfolio.state.positions.get(self.symbol)
        current_shares = position.quantity if position else 0.0
        
        # Strategy logic based solely on Fear & Greed Index
        if fgi_value is None:
            ctx.log("FGI unavailable, skipping trading decision")
            return
        
        # Determine FGI category and action
        if fgi_value <= self.config.fgi_fear_max:
            # Fear (0-45): Buy 90% of weekly contribution + proportional of remaining cash
            if fgi_value <= 25:
                category = "Extreme Fear"
            else:
                category = "Fear"
            action = "BUY_FEAR"
            
        elif fgi_value <= self.config.fgi_greed_max:
            # Neutral/Greed (46-75): Buy only 9% of weekly contribution
            if fgi_value <= 55:
                category = "Neutral"
            else:
                category = "Greed"
            action = "BUY_NEUTRAL_GREED"
            
        else:
            # Extreme Greed (76-100): May not contribute, or sell a portion proportional to FGI
            category = "Extreme Greed"
            action = "SELL_PROPORTIONAL"
        
        # Log FGI category and action
        if should_log_fgi:
            ctx.log(f"FGI Category: {category} ({fgi_value:.2f}), Action: {action}")
        
        # Execute sell decisions (can happen any day, not just new week)
        if action == "SELL_PROPORTIONAL":
            # Extreme Greed: Sell up to 50% proportional to FGI
            # Formula: sell_fraction = 0.5 * (fgi_value - 76) / 24
            # At FGI=76: sell_fraction = 0.0 (no selling)
            # At FGI=100: sell_fraction = 0.5 (sell 50%)
            fgi_greed_range = 100 - self.config.fgi_extreme_greed_min
            sell_fraction = 0.5 * (fgi_value - self.config.fgi_extreme_greed_min) / fgi_greed_range
            sell_fraction = max(0.0, min(0.5, sell_fraction))  # Clamp to [0, 0.5]
            
            if current_shares > 0 and sell_fraction > 0:
                shares_to_sell = current_shares * sell_fraction
                if shares_to_sell > 0:
                    ctx.log(
                        f"Extreme Greed (FGI {fgi_value:.2f}): Selling {shares_to_sell:.4f} shares "
                        f"({sell_fraction:.1%} of position, {current_shares:.4f} total) @ ${current_price:.2f}"
                    )
                    sell_order = Order(
                        id="",
                        symbol=self.symbol,
                        side=Side.SELL,
                        quantity=max(1, round(shares_to_sell)),
                        order_type=OrderType.MARKET,
                        limit_price=None,
                        instrument_type=InstrumentType.STOCK,
                    )
                    await ctx.execution.submit_order(sell_order)
                    if self.config.fee_per_trade > 0:
                        ctx.log(f"Trade fee: ${self.config.fee_per_trade:.2f}")
            else:
                ctx.log(
                    f"Extreme Greed (FGI {fgi_value:.2f}): No sell (fraction {sell_fraction:.1%}, "
                    f"shares {current_shares:.4f})"
                )
        
        # Execute buy decisions (only on new week)
        if is_new_week:
            # Weekly contribution logic
            # Add weekly contribution to portfolio cash
            weekly_cash = self.config.weekly_contribution
            ctx.portfolio.state.cash += weekly_cash
            self._total_contributions += weekly_cash
            
            # Get current cash position (after adding weekly contribution)
            available_cash = ctx.portfolio.state.cash
            
            if action == "BUY_FEAR":
                # Fear (0-45): Buy 90% of weekly contribution + proportional of remaining cash
                # Formula: buy_multiplier = (45 - fgi_value) / 45
                # At FGI=0: buy_multiplier = 1.0 (buy 100% of remaining cash)
                # At FGI=45: buy_multiplier = 0.0 (buy 0% of remaining cash)
                # Linear scaling between FGI 0 and 45
                buy_multiplier = (self.config.fgi_fear_max - fgi_value) / self.config.fgi_fear_max
                buy_multiplier = max(0.0, min(1.0, buy_multiplier))  # Clamp to [0, 1]
                
                # Calculate buy amounts:
                # 1. Always buy $900 (90% of weekly contribution)
                # 2. Additionally buy proportional of remaining cash (accumulated from previous weeks)
                buy_from_weekly = weekly_cash * 0.9  # $900
                # Calculate remaining cash before this week's contribution
                remaining_cash_before_weekly = available_cash - weekly_cash
                buy_from_remaining = remaining_cash_before_weekly * buy_multiplier
                total_buy_amount = buy_from_weekly + buy_from_remaining
                
                if total_buy_amount > 0 and current_price > 0:
                    shares_to_buy = total_buy_amount / current_price
                    if shares_to_buy > 0:
                        ctx.log(
                            f"Weekly contribution: ${weekly_cash:,.2f} | "
                            f"Remaining cash (before weekly): ${remaining_cash_before_weekly:,.2f} | "
                            f"{category} (FGI {fgi_value:.2f}): Buying ${total_buy_amount:,.2f} "
                            f"($900 + {buy_multiplier:.1%} of ${remaining_cash_before_weekly:,.2f} = ${buy_from_remaining:,.2f}) = {shares_to_buy:.4f} shares @ ${current_price:.2f}"
                        )
                        buy_order = Order(
                            id="",
                            symbol=self.symbol,
                            side=Side.BUY,
                            quantity=max(1, round(shares_to_buy)),
                            order_type=OrderType.MARKET,
                            limit_price=None,
                            instrument_type=InstrumentType.STOCK,
                        )
                        await ctx.execution.submit_order(buy_order)
                        # Note: total_contributions already includes weekly_cash added above
                        if self.config.fee_per_trade > 0:
                            ctx.log(f"Trade fee: ${self.config.fee_per_trade:.2f}")
                else:
                    ctx.log(
                        f"Weekly contribution: ${weekly_cash:,.2f} | "
                        f"Remaining cash: ${available_cash:,.2f} | "
                        f"{category} (FGI {fgi_value:.2f}): No buy (total ${total_buy_amount:,.2f})"
                    )
            
            elif action == "BUY_NEUTRAL_GREED":
                # Neutral/Greed (46-75): Buy 90% of weekly contribution
                # Target: leave exactly 10% of weekly contribution as cash (same as DCA logic)
                # For consistency with DCA: each week we add $1000, buy $900, leave $100
                target_buy_amount = weekly_cash * 0.9  # Always buy 90% of weekly contribution
                
                if target_buy_amount > 0 and current_price > 0:
                    shares_to_buy = target_buy_amount / current_price
                    if shares_to_buy > 0:
                        # Round shares to whole number
                        rounded_shares = max(1, round(shares_to_buy))
                        actual_cost = rounded_shares * current_price
                        
                        # Ensure we don't overspend (must leave at least 10% of weekly)
                        # If rounding causes us to spend more than target, reduce shares
                        max_allowed_spend = available_cash - (weekly_cash * 0.1)
                        if actual_cost > max_allowed_spend:
                            rounded_shares = max(1, rounded_shares - 1)
                            actual_cost = rounded_shares * current_price
                        
                        ctx.log(
                            f"Weekly contribution: ${weekly_cash:,.2f} | "
                            f"{category} (FGI {fgi_value:.2f}): Buying ${actual_cost:,.2f} "
                            f"({rounded_shares} shares) @ ${current_price:.2f}"
                        )
                        buy_order = Order(
                            id="",
                            symbol=self.symbol,
                            side=Side.BUY,
                            quantity=rounded_shares,
                            order_type=OrderType.MARKET,
                            limit_price=None,
                            instrument_type=InstrumentType.STOCK,
                        )
                        await ctx.execution.submit_order(buy_order)
                        # Note: total_contributions already includes weekly_cash added above
                        if self.config.fee_per_trade > 0:
                            ctx.log(f"Trade fee: ${self.config.fee_per_trade:.2f}")
                else:
                    ctx.log(
                        f"Weekly contribution: ${weekly_cash:,.2f} | "
                        f"{category} (FGI {fgi_value:.2f}): No buy (target ${target_buy_amount:,.2f})"
                    )
    
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
