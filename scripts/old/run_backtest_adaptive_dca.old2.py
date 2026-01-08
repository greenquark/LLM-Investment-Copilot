"""
Backtest script for Adaptive DCA Strategy - Fear & Greed Index Based.

Evaluates AdaptiveDCA against regular DCA with same capital contributions.

"""

import asyncio
import sys
import logging
import argparse
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.data.factory import create_data_engine_from_config
from core.data.trading_calendar import get_trading_calendar
from core.backtest.scheduler import DecisionScheduler
from core.backtest.engine import BacktestEngine
from core.strategy.regular_dca import RegularDCAStrategy, RegularDCAConfig
from core.backtest.performance import evaluate_performance_dca
from core.strategy.adaptive_dca import (
    AdaptiveDCAStrategy,
    AdaptiveDCAConfig,
)
from core.utils.logging import Logger
from core.utils.config_loader import load_config_with_secrets
from core.visualization import PlotlyChartVisualizer


async def compare_adaptive_dca_vs_dca(
    data_engine,
    symbol: str,
    strategy_config: AdaptiveDCAConfig,
    start: datetime,
    end: datetime,
    initial_cash: float,
    timeframe: str,
):
    """
    Run AdaptiveDCA backtest and compare with Regular DCA benchmark.
    
    Uses data structures to track decisions and state instead of parsing logs.
    """
    
    class ActionCollectorLogger(Logger):
        """Logger that tracks decisions and portfolio state using data structures."""
        
        def __init__(self, prefix: str = "", symbol: str = ""):
            super().__init__(prefix=prefix)
            self.symbol = symbol
            self.weekly_actions = []
            # Track decisions by week: {week_key: {date, fgi, action, price, ...}}
            self._decisions_by_week: Dict[tuple, Dict] = {}
            self._engine = None
            self._initial_cash = None
            self._strategy_config = None
            self._dca_shares_tracker = {}  # Track DCA cumulative shares by week
            self._suppress_logs = False
        
        def set_engine(self, engine):
            """Set engine reference to access portfolio."""
            self._engine = engine
        
        def set_initial_cash(self, initial_cash: float):
            """Set initial cash to calculate relative cash (starting from 0)."""
            self._initial_cash = initial_cash
        
        def set_strategy_config(self, config):
            """Set strategy config for DCA calculations."""
            self._strategy_config = config
        
        def record_decision(self, timestamp: datetime, fgi_value: float, action: str, price: float):
            """Record a strategy decision (called from strategy)."""
            current_date = timestamp.date()
            week_key = (current_date.isocalendar()[0], current_date.isocalendar()[1])
            
            # Format action description
            if self._strategy_config:
                if action == "BUY_FEAR":
                    # Placeholder - will be updated with actual remaining_cash in finalize_weekly_actions
                    base_buy = self._strategy_config.contribution_amount * self._strategy_config.base_buy_fraction
                    formatted_action = f"Buy ${base_buy:,.0f} + X% of remaining cash"  # Placeholder, updated later
                elif action == "BUY_NEUTRAL_GREED":
                    base_buy = self._strategy_config.contribution_amount * self._strategy_config.base_buy_fraction
                    formatted_action = f"Buy ${base_buy:,.0f}"
                elif action == "SELL_PROPORTIONAL":
                    # Use max_sell_fraction from config
                    max_sell = self._strategy_config.max_sell_fraction if self._strategy_config else 0.2
                    fgi_extreme_min = self._strategy_config.fgi_extreme_greed_min if self._strategy_config else 76
                    sell_fraction = max_sell * (fgi_value - fgi_extreme_min) / (100 - fgi_extreme_min) if fgi_value >= fgi_extreme_min else 0
                    formatted_action = f"SELL {sell_fraction:.1%} of position"
                else:
                    formatted_action = "HOLD"
            else:
                formatted_action = action
            
            # Store decision with timestamp for later state recording
            if week_key not in self._decisions_by_week:
                self._decisions_by_week[week_key] = {
                    "date": current_date,
                    "timestamp": timestamp,  # Store timestamp for state recording
                    "fgi": fgi_value,
                    "action": formatted_action,  # Formatted for display
                    "action_type": action,  # Raw action type (BUY_FEAR, BUY_NEUTRAL_GREED, SELL_PROPORTIONAL)
                    "price": price,
                }
            else:
                # Update with latest decision for the week
                self._decisions_by_week[week_key].update({
                    "fgi": fgi_value,
                    "action": formatted_action,
                    "action_type": action,  # Raw action type
                    "price": price,
                    "timestamp": timestamp,  # Update timestamp
                })
        
        def record_week_state(self, timestamp: datetime):
            """Record portfolio state after fills are processed (called after fills)."""
            if self._engine is None or self._engine._portfolio is None:
                return
            
            current_date = timestamp.date()
            week_key = (current_date.isocalendar()[0], current_date.isocalendar()[1])
            
            # Get portfolio state
            portfolio = self._engine._portfolio
            portfolio_cash = portfolio.state.cash
            
            # Use absolute cash (consistent with finalize_weekly_actions)
            # Check that absolute cash is not negative
            if portfolio_cash < 0:
                raise ValueError(f"Cash position is negative: ${portfolio_cash:,.2f}. This should not happen.")
            adaptive_cash = portfolio_cash  # Absolute cash, includes initial cash
            
            # Get AdaptiveDCA shares
            position = portfolio.state.positions.get(self.symbol)
            adaptive_shares = position.quantity if position else 0.0
            
            # Update decision with state (ensure week_key exists)
            if week_key not in self._decisions_by_week:
                self._decisions_by_week[week_key] = {
                    "date": current_date,
                    "timestamp": timestamp,
                    "fgi": None,
                    "action": "HOLD",
                    "price": None,
                }
            
            self._decisions_by_week[week_key].update({
                "adaptive_cash": adaptive_cash,
                "adaptive_shares": adaptive_shares,
            })
            
            # Calculate DCA shares (fractional)
            if self._strategy_config and week_key in self._decisions_by_week:
                price = self._decisions_by_week[week_key].get("price")
                if price and price > 0:
                    # Get previous week's DCA shares
                    prev_week_key = None
                    for wk in sorted(self._dca_shares_tracker.keys()):
                        if wk < week_key:
                            prev_week_key = wk
                    prev_dca_shares = self._dca_shares_tracker.get(prev_week_key, 0.0) if prev_week_key else 0.0
                    
                    # Calculate shares bought this week: contribution_amount / price (fractional)
                    shares_bought_this_week = self._strategy_config.contribution_amount / price
                    
                    # Cumulative DCA shares = previous + new shares bought this week
                    current_dca_shares = prev_dca_shares + shares_bought_this_week
                    self._dca_shares_tracker[week_key] = current_dca_shares
                    
                    self._decisions_by_week[week_key]["dca_cash"] = 0.0  # DCA always buys all cash
                    self._decisions_by_week[week_key]["dca_shares"] = current_dca_shares
        
        def finalize_weekly_actions(self):
            """Convert decisions_by_week to weekly_actions list, calculating state incrementally from decisions."""
            self.weekly_actions = []
            
            if not self._strategy_config:
                # Fallback to stored state if no config
                for week_key in sorted(self._decisions_by_week.keys()):
                    decision = self._decisions_by_week[week_key]
                    self.weekly_actions.append({
                        "date": decision.get("date"),
                        "price": decision.get("price"),
                        "fgi": decision.get("fgi"),
                        "adaptive_action": decision.get("action", "HOLD"),
                        "cash": decision.get("adaptive_cash"),
                        "adaptive_shares": decision.get("adaptive_shares"),
                        "dca_cash": decision.get("dca_cash", 0.0),
                        "dca_shares": decision.get("dca_shares", 0.0),
                    })
                return
            
            # Track state incrementally week by week (starting from initial_cash)
            current_cash = self._initial_cash if self._initial_cash is not None else 0.0
            current_shares = 0.0
            current_dca_shares = 0.0
            contribution = self._strategy_config.contribution_amount
            base_buy_fraction = self._strategy_config.base_buy_fraction  # 0.9
            
            for week_key in sorted(self._decisions_by_week.keys()):
                decision = self._decisions_by_week[week_key]
                date = decision.get("date")
                price = decision.get("price", 0.0)
                fgi_value = decision.get("fgi")
                action_str = decision.get("action", "HOLD")  # Formatted for display
                action_type = decision.get("action_type", "")  # Raw action type
                
                # Determine buy/sell amounts based on raw action type (before adding contribution)
                buy_amount = 0.0
                sell_shares = 0.0
                
                if action_type == "BUY_NEUTRAL_GREED":
                    # Neutral/Greed: Buy base_buy_fraction of weekly contribution, leave (1 - base_buy_fraction)
                    # First add contribution, then calculate buy
                    current_cash += contribution
                    base_buy = contribution * base_buy_fraction
                    min_remaining = contribution * (1 - base_buy_fraction)
                    buy_amount = min(base_buy, current_cash - min_remaining)
                elif action_type == "BUY_FEAR":
                    # Fear: Buy base_buy_fraction of weekly + proportional of remaining cash
                    # remaining_cash = cash BEFORE this week's contribution (cash from previous weeks)
                    remaining_cash = current_cash  # Cash before adding this week's contribution
                    # Now add this week's contribution
                    current_cash += contribution
                    base_buy = contribution * base_buy_fraction  # base_buy_fraction of weekly
                    if fgi_value is not None and fgi_value <= 45:
                        multiplier = (45 - fgi_value) / 45
                        buy_from_remaining = remaining_cash * multiplier  # Proportional of previous weeks' cash
                        buy_amount = base_buy + buy_from_remaining
                        # Update action string with actual remaining cash amount
                        action_str = f"Buy ${base_buy:,.0f} + {multiplier:.1%} of ${remaining_cash:,.2f}"
                    else:
                        buy_amount = base_buy
                        action_str = f"Buy ${base_buy:,.0f}"
                    # Cap at available cash (after adding contribution)
                    buy_amount = min(buy_amount, current_cash)
                elif action_type == "SELL_PROPORTIONAL":
                    # For sell, still add contribution first
                    current_cash += contribution
                    # Extract sell fraction from FGI using config
                    if fgi_value is not None and fgi_value >= self._strategy_config.fgi_extreme_greed_min:
                        max_sell = self._strategy_config.max_sell_fraction
                        sell_fraction = max_sell * (fgi_value - self._strategy_config.fgi_extreme_greed_min) / (100 - self._strategy_config.fgi_extreme_greed_min)
                        sell_fraction = max(0.0, min(max_sell, sell_fraction))
                        sell_shares = current_shares * sell_fraction
                else:
                    # HOLD or other: just add contribution
                    current_cash += contribution
                
                # Execute sell first (if any)
                if sell_shares > 0 and price > 0:
                    sell_proceeds = sell_shares * price
                    current_cash += sell_proceeds
                    current_shares -= sell_shares
                
                # Execute buy (if any)
                if buy_amount > 0 and price > 0:
                    # Calculate shares to buy (fractional)
                    shares_to_buy = buy_amount / price
                    current_shares += shares_to_buy
                    current_cash -= buy_amount
                
                # Calculate DCA shares (always buys 100% of contribution at this price)
                if price > 0:
                    dca_shares_bought = contribution / price
                    current_dca_shares += dca_shares_bought
                
                # Store calculated state (absolute cash, including initial cash)
                # For AdaptiveDCA: show absolute cash (includes initial cash)
                adaptive_cash = current_cash
                
                # For DCA: calculate absolute cash (initial + contributions - buys)
                # DCA always buys 100% of contribution, so cash = initial_cash (if any)
                dca_cash = self._initial_cash if self._initial_cash is not None else 0.0
                
                self.weekly_actions.append({
                    "date": date,
                    "price": price,
                    "fgi": fgi_value,
                    "adaptive_action": action_str,
                    "cash": adaptive_cash,
                    "adaptive_shares": current_shares,
                    "dca_cash": dca_cash,  # DCA cash = initial cash (always buys 100% of contributions)
                    "dca_shares": current_dca_shares,
                })
        
        def log(self, msg: str) -> None:
            """Override log to suppress verbose output during backtest."""
            if not self._suppress_logs:
                super().log(msg)
        
        def record_all_weeks_state(self, equity_curve):
            """Record portfolio state for all weeks that have decisions."""
            # Record state for each week that has a decision
            # Use the decision timestamps to record state
            # Note: This reads current portfolio state, which is cumulative
            # We need to record state during backtest, but for now this ensures
            # we at least have state recorded for weeks with decisions
            for week_key, decision in self._decisions_by_week.items():
                timestamp = decision.get("timestamp")
                if timestamp:
                    # Record state for this week's decision timestamp
                    self.record_week_state(timestamp)
        
        def record_state_after_decision(self, timestamp: datetime):
            """Record portfolio state right after a decision and fills are processed."""
            # This should be called right after fills are processed in the backtest engine
            # For now, we'll call it from record_decision, but ideally it should be
            # called after fills are processed
            self.record_week_state(timestamp)
    
    logger = ActionCollectorLogger(prefix="[AdaptiveDCA]", symbol=symbol)
    
    # Suppress verbose logging during backtest (only show warnings and errors)
    logging.getLogger("core.strategy.adaptive_dca").setLevel(logging.WARNING)
    logging.getLogger("core.data").setLevel(logging.WARNING)
    
    # Create scheduler (daily decisions for weekly DCA)
    scheduler = DecisionScheduler(interval_minutes=24 * 60)
    
    # Create backtest engine
    engine = BacktestEngine(data_engine, scheduler, logger)
    
    # Set engine reference in logger so it can access portfolio
    logger.set_engine(engine)
    logger.set_initial_cash(initial_cash)
    logger.set_strategy_config(strategy_config)
    
    # Create strategy
    strategy = AdaptiveDCAStrategy(symbol, strategy_config, data_engine)
    
    # Adjust end date to the last trading day on or before the end date
    try:
        calendar = get_trading_calendar("NYSE")
        end_date_original = end.date() if isinstance(end, datetime) else end
        if not isinstance(end_date_original, date):
            raise ValueError(f"Invalid end date type: {type(end_date_original)}")
        
        # Get the last trading day on or before the end date
        if calendar.is_trading_day(end_date_original):
            last_trading_day = end_date_original
        else:
            # Get the previous trading day
            last_trading_day = calendar.previous_trading_day(end_date_original)
            if last_trading_day is None:
                # Fallback: use the original date if we can't find a previous trading day
                last_trading_day = end_date_original
        
        end = datetime.combine(last_trading_day, datetime.min.time())
    except Exception as e:
        logging.warning(f"Could not adjust end date using trading calendar: {e}. Using original end date.")
        # end is already set from parameter
    
    # Run backtest
    print(f"\n=== Running Adaptive DCA Backtest ===")
    print(f"Symbol: {symbol}")
    print(f"Start: {start.date()}")
    print(f"End: {end.date()}")
    print(f"Initial Cash: ${initial_cash:,.2f}")
    print(f"Weekly Contribution: ${strategy_config.contribution_amount:,.2f}")
    print(f"Timeframe: {timeframe}")
    
    result = await engine.run(
        symbol, 
        strategy, 
        start, 
        end, 
        initial_cash, 
        timeframe=timeframe,
        is_dca=True,
        contribution_amount=strategy_config.contribution_amount,
        contribution_frequency=strategy_config.contribution_frequency,
    )
    
    # Record portfolio state for all weeks that have decisions
    # This reads the current portfolio state (final state), but ensures we have values
    logger.record_all_weeks_state(result.equity_curve)
    
    # Also ensure we record state for any weeks that might have been missed
    for week_key in list(logger._decisions_by_week.keys()):
        decision = logger._decisions_by_week[week_key]
        if decision.get("adaptive_cash") is None or decision.get("adaptive_shares") is None:
            timestamp = decision.get("timestamp")
            if timestamp:
                logger.record_week_state(timestamp)
    
    # Finalize weekly actions after backtest completes
    logger.finalize_weekly_actions()
    
    # Get equity values at start and end dates (from config, not equity curve dates)
    def get_equity_at_date(equity_curve: Dict[datetime, float], target_date: datetime, prefer_before: bool = True) -> float:
        """
        Get equity value at or closest to target date.
        
        Args:
            equity_curve: Dictionary of datetime -> equity value
            target_date: Target date to find equity for
            prefer_before: If True, prefer dates on or before target_date (for start dates).
                          If False, prefer dates on or before target_date (for end dates - use last available).
        
        Returns:
            Equity value at or closest to target date
        """
        if not equity_curve:
            return initial_cash
        
        equity_items = sorted(equity_curve.items(), key=lambda x: x[0])
        
        # For both start and end dates, we want the latest date <= target_date
        # This ensures we use the actual portfolio value at the target date (or the last available before it)
        # For end dates, we don't want to use future dates - we want the last known equity value
        best_date = None
        best_equity = initial_cash
        
        for eq_date, eq_value in equity_items:
            if eq_date <= target_date:
                # This date is on or before target - prefer the latest one
                if best_date is None or eq_date > best_date:
                    best_date = eq_date
                    best_equity = eq_value
            elif best_date is None:
                # No date found before target yet, but this is the first date after
                # Use it as fallback (closest available) - this handles edge cases where
                # the target date is before any equity curve data
                best_date = eq_date
                best_equity = eq_value
                break
        
        return best_equity
    
    # Get equity at start date (for reference) and end date (for final calculation)
    # For both, we want the latest date <= target_date (the actual portfolio value at that point in time)
    start_equity = get_equity_at_date(result.equity_curve, start, prefer_before=True) if result.equity_curve else initial_cash
    final_equity = get_equity_at_date(result.equity_curve, end, prefer_before=True) if result.equity_curve else initial_cash
    
    # Calculate total contributions for DCA performance calculation
    total_weeks = ((end.date() - start.date()).days // 7) + 1
    total_contributions = total_weeks * strategy_config.contribution_amount
    total_invested = initial_cash + total_contributions
    
    # P&L for DCA: (Final - Total Invested) / Total Invested
    pnl = final_equity - total_invested
    pnl_pct = (pnl / total_invested * 100) if total_invested > 0 else 0.0
    
    print(f"\n=== Adaptive DCA Results ===")
    print(f"Initial Cash: ${initial_cash:,.2f}")
    print(f"Total Contributions: ${total_contributions:,.2f} ({total_weeks} weeks × ${strategy_config.contribution_amount:,.2f})")
    print(f"Total Invested: ${total_invested:,.2f}")
    print(f"Final Equity: ${final_equity:,.2f}")
    print(f"P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%)")
    
    # Recalculate metrics using DCA-specific calculation
    strategy_dca_metrics = evaluate_performance_dca(
        result.equity_curve,
        total_contributions=total_contributions,
        initial_cash=initial_cash
    )
    
    # Print detailed calculation steps for AdaptiveDCA Total Return
    print(f"\n=== AdaptiveDCA Total Return Calculation ===")
    print(f"Step 1: Calculate total weeks")
    print(f"  Start date: {start.date()}")
    print(f"  End date: {end.date()}")
    print(f"  Calendar days: {(end.date() - start.date()).days}")
    print(f"  Total weeks: {total_weeks} = (({end.date()} - {start.date()}).days // 7) + 1")
    print(f"")
    print(f"Step 2: Calculate total contributions")
    print(f"  Weekly contribution: ${strategy_config.contribution_amount:,.2f}")
    print(f"  Total contributions: ${total_contributions:,.2f} = {total_weeks} weeks × ${strategy_config.contribution_amount:,.2f}")
    print(f"")
    print(f"Step 3: Calculate total invested")
    print(f"  Initial cash: ${initial_cash:,.2f}")
    print(f"  Total contributions: ${total_contributions:,.2f}")
    print(f"  Total invested: ${total_invested:,.2f} = ${initial_cash:,.2f} + ${total_contributions:,.2f}")
    print(f"")
    print(f"Step 4: Get equity values at start and end dates (from config)")
    if result.equity_curve:
        # Find closest equity values to start and end dates
        equity_items = sorted(result.equity_curve.items(), key=lambda x: x[0])
        
        # Find closest to start date
        start_closest_date = None
        start_min_diff = float('inf')
        for eq_date, eq_value in equity_items:
            diff = abs((eq_date - start).total_seconds())
            if diff < start_min_diff:
                start_min_diff = diff
                start_closest_date = (eq_date, eq_value)
        
        # Find closest to end date
        end_closest_date = None
        end_min_diff = float('inf')
        for eq_date, eq_value in equity_items:
            diff = abs((eq_date - end).total_seconds())
            if diff < end_min_diff:
                end_min_diff = diff
                end_closest_date = (eq_date, eq_value)
        
        if start_closest_date:
            print(f"  Start date (config): {start.date()}")
            print(f"  Equity at start: ${start_equity:,.2f} (closest equity point: {start_closest_date[0].date()} = ${start_closest_date[1]:,.2f})")
        else:
            print(f"  Start date (config): {start.date()}")
            print(f"  Equity at start: ${start_equity:,.2f} (no equity curve data)")
        
        if end_closest_date:
            print(f"  End date (config): {end.date()}")
            print(f"  Final equity: ${final_equity:,.2f} (closest equity point: {end_closest_date[0].date()} = ${end_closest_date[1]:,.2f})")
        else:
            print(f"  End date (config): {end.date()}")
            print(f"  Final equity: ${final_equity:,.2f} (no equity curve data)")
    else:
        print(f"  Start date (config): {start.date()}")
        print(f"  Equity at start: ${start_equity:,.2f} (no equity curve data, using initial cash)")
        print(f"  End date (config): {end.date()}")
        print(f"  Final equity: ${final_equity:,.2f} (no equity curve data, using initial cash)")
    print(f"")
    print(f"Step 5: Calculate Total Return")
    print(f"  Formula: Total Return = (Final Equity - Total Invested) / Total Invested")
    print(f"  Total Return = (${final_equity:,.2f} - ${total_invested:,.2f}) / ${total_invested:,.2f}")
    total_return_calc = (final_equity - total_invested) / total_invested
    print(f"  Total Return = ${final_equity - total_invested:,.2f} / ${total_invested:,.2f}")
    print(f"  Total Return = {total_return_calc:.6f} = {strategy_dca_metrics.total_return:.2%}")
    
    print(f"\n=== Performance Metrics (DCA-adjusted) ===")
    print(f"Total Return: {strategy_dca_metrics.total_return:.2%}")
    print(f"CAGR: {strategy_dca_metrics.cagr:.2%}")
    print(f"Volatility: {strategy_dca_metrics.volatility:.2%}")
    print(f"Sharpe Ratio: {strategy_dca_metrics.sharpe:.2f}")
    print(f"Max Drawdown: {strategy_dca_metrics.max_drawdown:.2%}")
    
    # Print weekly actions table (always visible)
    print("\n=== Week by Week Actions ===")
    print(f"{'Date':<12} {'Quote':<10} {'FGI':<8} {'AdaptiveDCA Action':<35} {'AdaptiveDCA Cash':<18} {'AdaptiveDCA Shares':<18} {'DCA Action':<20} {'DCA Cash':<12} {'DCA Shares':<15}")
    print("-" * 160)
    
    for action in logger.weekly_actions:
        date_str = action['date'].strftime("%Y-%m-%d") if action['date'] else "N/A"
        price_str = f"${action['price']:.2f}" if action['price'] is not None else "N/A"
        fgi_str = f"{action['fgi']:.2f}" if action['fgi'] is not None else "N/A"
        adaptive_cash_str = f"${action['cash']:,.2f}" if action['cash'] is not None else "N/A"
        adaptive_shares_str = f"{action['adaptive_shares']:,.2f}" if action['adaptive_shares'] is not None else "N/A"
        dca_action = f"BUY ${strategy_config.contribution_amount:,.0f}"
        dca_cash_str = f"${action['dca_cash']:,.2f}" if action['dca_cash'] is not None else "N/A"
        dca_shares_str = f"{action['dca_shares']:,.2f}" if action['dca_shares'] is not None else "N/A"
        print(f"{date_str:<12} {price_str:<10} {fgi_str:<8} {action['adaptive_action']:<35} {adaptive_cash_str:<18} {adaptive_shares_str:<18} {dca_action:<20} {dca_cash_str:<12} {dca_shares_str:<15}")
    
    print("")
    
    # Run Regular DCA benchmark using BacktestEngine (same method as AdaptiveDCA)
    weekly_contribution = strategy_config.contribution_amount
    print(f"\n=== Calculating Regular DCA Benchmark ({symbol}) ===")
    
    # Create Regular DCA strategy config
    dca_config = RegularDCAConfig(
        contribution_amount=weekly_contribution,
        timeframe=timeframe,
        lookback_days=strategy_config.lookback_days,
        contribution_frequency=strategy_config.contribution_frequency,
    )
    
    # Create Regular DCA strategy
    dca_strategy = RegularDCAStrategy(symbol, dca_config, data_engine)
    
    # Create logger for Regular DCA (suppress verbose output)
    dca_logger = Logger(prefix="[RegularDCA]")
    dca_logger._suppress_logs = True  # Suppress verbose output
    
    # Create scheduler and engine for Regular DCA (daily decisions for weekly DCA)
    dca_scheduler = DecisionScheduler(interval_minutes=24 * 60)
    dca_engine = BacktestEngine(data_engine, dca_scheduler, dca_logger)
    
    # Run Regular DCA backtest
    dca_result = await dca_engine.run(
        symbol=symbol,
        strategy=dca_strategy,
        start=start,
        end=end,
        initial_cash=initial_cash,
        timeframe=timeframe,
    )
    
    # Recalculate metrics using DCA-specific calculation for Regular DCA
    # Use calendar days (consistent with Strategy calculation)
    dca_total_weeks = ((end.date() - start.date()).days // 7) + 1
    dca_total_contributions = dca_total_weeks * weekly_contribution
    
    # Get equity values at start and end dates for Regular DCA (using same helper function)
    # For both, we want the latest date <= target_date (the actual portfolio value at that point in time)
    dca_start_equity = get_equity_at_date(dca_result.equity_curve, start, prefer_before=True) if dca_result.equity_curve else initial_cash
    dca_final_equity = get_equity_at_date(dca_result.equity_curve, end, prefer_before=True) if dca_result.equity_curve else initial_cash
    dca_total_invested = initial_cash + dca_total_contributions
    
    # Print detailed calculation steps for Regular DCA Total Return
    print(f"\n=== Regular DCA Total Return Calculation ===")
    print(f"Step 1: Calculate total weeks")
    print(f"  Start date: {start.date()}")
    print(f"  End date: {end.date()}")
    print(f"  Calendar days: {(end.date() - start.date()).days}")
    print(f"  Total weeks: {dca_total_weeks} = (({end.date()} - {start.date()}).days // 7) + 1")
    print(f"")
    print(f"Step 2: Calculate total contributions")
    print(f"  Weekly contribution: ${weekly_contribution:,.2f}")
    print(f"  Total contributions: ${dca_total_contributions:,.2f} = {dca_total_weeks} weeks × ${weekly_contribution:,.2f}")
    print(f"")
    print(f"Step 3: Calculate total invested")
    print(f"  Initial cash: ${initial_cash:,.2f}")
    print(f"  Total contributions: ${dca_total_contributions:,.2f}")
    print(f"  Total invested: ${dca_total_invested:,.2f} = ${initial_cash:,.2f} + ${dca_total_contributions:,.2f}")
    print(f"")
    print(f"Step 4: Get equity values at start and end dates (from config)")
    if dca_result.equity_curve:
        dca_equity_items = sorted(dca_result.equity_curve.items(), key=lambda x: x[0])
        
        # Debug: Show first and last equity points
        if dca_equity_items:
            first_eq_date, first_eq_value = dca_equity_items[0]
            last_eq_date, last_eq_value = dca_equity_items[-1]
            print(f"  Equity curve range: {first_eq_date.date()} to {last_eq_date.date()} ({len(dca_equity_items)} points)")
            print(f"  First equity point: {first_eq_date.date()} = ${first_eq_value:,.2f}")
            print(f"  Last equity point: {last_eq_date.date()} = ${last_eq_value:,.2f}")
        
        # Find the actual dates used by get_equity_at_date for display
        # get_equity_at_date finds the latest date <= target_date
        dca_start_closest_date = None
        dca_start_best_date = None
        for eq_date, eq_value in dca_equity_items:
            if eq_date <= start:
                # This date is on or before start - prefer the latest one
                if dca_start_best_date is None or eq_date > dca_start_best_date:
                    dca_start_best_date = eq_date
                    dca_start_closest_date = (eq_date, eq_value)
        
        # Find closest to end date (latest date <= end)
        dca_end_closest_date = None
        dca_end_best_date = None
        for eq_date, eq_value in reversed(dca_equity_items):  # Search backwards to find latest <= end
            if eq_date <= end:
                # This date is on or before end - prefer the latest one
                if dca_end_best_date is None or eq_date > dca_end_best_date:
                    dca_end_best_date = eq_date
                    dca_end_closest_date = (eq_date, eq_value)
                    break  # Since we're going backwards, first match is the latest
        
        if dca_start_closest_date:
            print(f"  Start date (config): {start.date()}")
            print(f"  Equity at start: ${dca_start_equity:,.2f} (closest equity point: {dca_start_closest_date[0].date()} = ${dca_start_closest_date[1]:,.2f})")
        else:
            print(f"  Start date (config): {start.date()}")
            print(f"  Equity at start: ${dca_start_equity:,.2f} (no equity curve data)")
        
        if dca_end_closest_date:
            # Get the final bar price to calculate shares
            final_bar_date = dca_end_closest_date[0]
            final_equity_value = dca_end_closest_date[1]
            
            # Get shares from weekly actions table (source of truth from logger)
            dca_shares_from_table = None
            dca_shares_date_from_table = None
            dca_price_from_table = None
            if logger.weekly_actions:
                # Get the last entry from weekly actions (most recent)
                last_action = logger.weekly_actions[-1]
                dca_shares_from_table = last_action.get('dca_shares')
                dca_shares_date_from_table = last_action.get('date')
                dca_price_from_table = last_action.get('price')
            
            # Fetch the bar for the final date to get the price
            try:
                final_bar_date_only = final_bar_date.date() if hasattr(final_bar_date, 'date') else final_bar_date
                final_bars = await data_engine.get_bars(
                    symbol, 
                    datetime.combine(final_bar_date_only, datetime.min.time()),
                    datetime.combine(final_bar_date_only, datetime.max.time()),
                    timeframe=timeframe
                )
                if final_bars:
                    final_bar = final_bars[-1]
                    final_price = final_bar.close
                    # For Regular DCA, cash is always $0, so shares = equity / price
                    final_shares_from_equity = final_equity_value / final_price if final_price > 0 else 0.0
                    
                    print(f"  End date (config): {end.date()}")
                    print(f"  Final equity: ${dca_final_equity:,.2f} (closest equity point: {final_bar_date_only} = ${final_equity_value:,.2f})")
                    print(f"  {symbol} price at {final_bar_date_only}: ${final_price:,.2f}")
                    print(f"  Shares from equity curve: {final_shares_from_equity:,.2f} (equity ${final_equity_value:,.2f} / price ${final_price:,.2f})")
                    
                    # Show shares from weekly actions table
                    if dca_shares_from_table is not None:
                        print(f"  Shares from weekly actions table: {dca_shares_from_table:,.2f} (as of {dca_shares_date_from_table})")
                        if dca_price_from_table and dca_price_from_table > 0:
                            expected_equity_from_table = dca_shares_from_table * dca_price_from_table
                            print(f"  Expected equity from table: {dca_shares_from_table:,.2f} shares × ${dca_price_from_table:,.2f} = ${expected_equity_from_table:,.2f}")
                        
                        # If we have shares from table, calculate what equity should be at final price
                        if final_price > 0:
                            expected_equity_at_final_price = dca_shares_from_table * final_price
                            print(f"  Expected equity at final price: {dca_shares_from_table:,.2f} shares × ${final_price:,.2f} = ${expected_equity_at_final_price:,.2f}")
                            
                            # Compare
                            equity_diff = expected_equity_at_final_price - final_equity_value
                            if abs(equity_diff) > 0.01:
                                print(f"  ⚠️  DISCREPANCY: Equity curve shows ${final_equity_value:,.2f}, but table suggests ${expected_equity_at_final_price:,.2f} (diff: ${equity_diff:,.2f})")
                    else:
                        print(f"  Note: Could not get shares from weekly actions table")
                else:
                    print(f"  End date (config): {end.date()}")
                    print(f"  Final equity: ${dca_final_equity:,.2f} (closest equity point: {final_bar_date_only} = ${final_equity_value:,.2f})")
                    print(f"  Note: Could not fetch {symbol} price for verification")
            except Exception as e:
                final_bar_date_only = final_bar_date.date() if hasattr(final_bar_date, 'date') else final_bar_date
                print(f"  End date (config): {end.date()}")
                print(f"  Final equity: ${dca_final_equity:,.2f} (closest equity point: {final_bar_date_only} = ${final_equity_value:,.2f})")
                print(f"  Note: Error fetching {symbol} price: {e}")
        else:
            print(f"  End date (config): {end.date()}")
            print(f"  Final equity: ${dca_final_equity:,.2f} (no equity curve data)")
    else:
        print(f"  Start date (config): {start.date()}")
        print(f"  Equity at start: ${dca_start_equity:,.2f} (no equity curve data, using initial cash)")
        print(f"  End date (config): {end.date()}")
        print(f"  Final equity: ${dca_final_equity:,.2f} (no equity curve data, using initial cash)")
    print(f"")
    print(f"Step 5: Calculate Total Return")
    print(f"  Formula: Total Return = (Final Equity - Total Invested) / Total Invested")
    print(f"  Total Return = (${dca_final_equity:,.2f} - ${dca_total_invested:,.2f}) / ${dca_total_invested:,.2f}")
    dca_total_return_calc = (dca_final_equity - dca_total_invested) / dca_total_invested
    print(f"  Total Return = ${dca_final_equity - dca_total_invested:,.2f} / ${dca_total_invested:,.2f}")
    
    dca_benchmark_metrics = evaluate_performance_dca(
        dca_result.equity_curve,
        total_contributions=dca_total_contributions,
        initial_cash=initial_cash
    )
    print(f"  Total Return = {dca_total_return_calc:.6f} = {dca_benchmark_metrics.total_return:.2%}")
    
    print(f"\n=== Strategy vs Regular DCA Comparison ===")
    print(f"{'Metric':<20} {'Strategy':>15} {'Regular DCA':>15} {'Difference':>15}")
    print("-" * 65)
    
    ret_diff = strategy_dca_metrics.total_return - dca_benchmark_metrics.total_return
    print(f"{'Total Return':<20} {strategy_dca_metrics.total_return:>14.2%} {dca_benchmark_metrics.total_return:>14.2%} {ret_diff:>+14.2%}")
    
    cagr_diff = strategy_dca_metrics.cagr - dca_benchmark_metrics.cagr
    print(f"{'CAGR':<20} {strategy_dca_metrics.cagr:>14.2%} {dca_benchmark_metrics.cagr:>14.2%} {cagr_diff:>+14.2%}")
    
    sharpe_diff = strategy_dca_metrics.sharpe - dca_benchmark_metrics.sharpe
    print(f"{'Sharpe Ratio':<20} {strategy_dca_metrics.sharpe:>14.2f} {dca_benchmark_metrics.sharpe:>14.2f} {sharpe_diff:>+14.2f}")
    
    vol_diff = strategy_dca_metrics.volatility - dca_benchmark_metrics.volatility
    print(f"{'Volatility':<20} {strategy_dca_metrics.volatility:>14.2%} {dca_benchmark_metrics.volatility:>14.2%} {vol_diff:>+14.2%}")
    
    dd_diff = strategy_dca_metrics.max_drawdown - dca_benchmark_metrics.max_drawdown
    print(f"{'Max Drawdown':<20} {strategy_dca_metrics.max_drawdown:>14.2%} {dca_benchmark_metrics.max_drawdown:>14.2%} {dd_diff:>+14.2%}")
    
    return result, dca_result


async def main(use_local_chart: bool = False):
    """Main entry point for Adaptive DCA backtest."""
    
    parser = argparse.ArgumentParser(description="Run Adaptive DCA backtest")
    parser.add_argument("--config", type=str, help="Path to strategy config file")
    parser.add_argument("--env", type=str, default="backtest", help="Environment config (backtest or live)")
    args = parser.parse_args()
    
    # Load configuration
    # Resolve paths relative to project root (parent of scripts directory)
    project_root = Path(__file__).parent.parent
    if args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = project_root / config_path
    else:
        config_path = project_root / "config" / "strategy.adaptive_dca.yaml"
    
    env_config_path = project_root / "config" / f"env.{args.env}.yaml"
    
    strat_cfg_raw = load_config_with_secrets(config_path)
    env_cfg = load_config_with_secrets(env_config_path)
    bt_cfg = env_cfg.get("backtest", {})
    
    # Get symbol from strategy config
    symbol = strat_cfg_raw.get("symbol", "SOXL")
    
    # Create strategy config
    strategy_config = AdaptiveDCAConfig.from_dict(strat_cfg_raw)
    
    # Create data engine
    # Pass the entire env config (like controlled_panic_bear script does)
    data_engine = create_data_engine_from_config(
        env_config=env_cfg,
        use_for="historical",
    )
    
    # Get timeframe from strategy config or env config
    timeframe = strategy_config.timeframe or env_cfg.get("backtest", {}).get("timeframe", "1D")
    
    # Run comparison
    await compare_adaptive_dca_vs_dca(
        data_engine,
        symbol,
        strategy_config,
        datetime.fromisoformat(bt_cfg["start"]),
        datetime.fromisoformat(bt_cfg["end"]),
        bt_cfg["initial_cash"],
        timeframe=timeframe,
    )


if __name__ == "__main__":
    asyncio.run(main())
