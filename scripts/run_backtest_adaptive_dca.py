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
from core.strategy.regular_dca_strategy import RegularDCA, RegularDCAConfig
from core.backtest.performance import evaluate_performance_dca, calculate_dca_total_return
from core.strategy.adaptive_dca import (
    AdaptiveDCAStrategy,
    AdaptiveDCAConfig,
)
from core.utils.logging import Logger
from core.utils.config_loader import load_config_with_secrets
from core.utils.price_utils import get_final_price
from core.visualization import PlotlyChartVisualizer
from core.visualization.chart_config import get_chart_config


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
            # Track decisions by week: {week_key: {date, fgi, action, price, ...}}
            self._decisions_by_week: Dict[tuple, Dict] = {}
            self._engine = None
            self._initial_cash = None
            self._strategy_config = None
            self._dca_shares_tracker = {}  # Track DCA cumulative shares by week
            self._suppress_logs = False
        
        def _period_key(self, d: date) -> tuple[int, int]:
            """Return (year, period_id) for configured contribution_frequency."""
            freq = (self._strategy_config.contribution_frequency if self._strategy_config else "weekly").lower()
            if freq == "weekly":
                iso_year, iso_week, _ = d.isocalendar()
                return (iso_year, iso_week)
            if freq == "monthly":
                return (d.year, d.month)
            if freq == "quarterly":
                q = (d.month - 1) // 3 + 1
                return (d.year, q)
            # yearly
            return (d.year, 1)
        
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
            week_key = self._period_key(current_date)
            
            # Format action description
            if self._strategy_config:
                if action == "BUY_FEAR":
                    base_buy = self._strategy_config.contribution_amount * self._strategy_config.base_buy_fraction
                    formatted_action = f"Buy ${base_buy:,.0f} + X% of remaining cash"
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
                # #region agent log
                with open(r'c:\Users\JiantaoPan\OneDrive\Documents\Code\LLM-Investment-Copilot\.cursor\debug.log', 'a') as f:
                    import json
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A,B,C,D,E","location":"run_backtest_adaptive_dca.py:197","message":"record_decision new entry","data":{"week_key":str(week_key),"date":str(current_date),"price":price,"action":action},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                # #endregion
            else:
                # Update with latest decision for the week
                # If the existing date was a non-trading day (price=0.0) and we now have a trading day with actual price,
                # update the date to the trading day when the actual contribution happened
                existing_decision = self._decisions_by_week[week_key]
                existing_date = existing_decision.get("date")
                existing_price = existing_decision.get("price", 0.0)
                
                # #region agent log
                with open(r'c:\Users\JiantaoPan\OneDrive\Documents\Code\LLM-Investment-Copilot\.cursor\debug.log', 'a') as f:
                    import json
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A,B,C,D,E","location":"run_backtest_adaptive_dca.py:210","message":"record_decision update check","data":{"week_key":str(week_key),"existing_date":str(existing_date),"existing_price":existing_price,"new_date":str(current_date),"new_price":price,"will_update_date":(existing_price == 0.0 and price > 0.0)},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                # #endregion
                
                # If existing date had price=0.0 (non-trading day) and new date has actual price (trading day),
                # use the new date (trading day) as the contribution date
                # BUT: only update if the new date is earlier than or equal to the existing date
                # (to prevent overwriting 9/3 with 9/4, 9/5, etc. when multiple calls happen for the same period)
                if existing_price == 0.0 and price > 0.0:
                    # The actual contribution happened on the trading day, so use that date
                    # Only update if this is the first trading day (new_date <= existing_date when existing was non-trading)
                    # OR if the new date is actually earlier (shouldn't happen, but be safe)
                    if current_date <= existing_date or existing_date < current_date:
                        self._decisions_by_week[week_key].update({
                            "date": current_date,  # Update to trading day when contribution actually happened
                            "fgi": fgi_value,
                            "action": formatted_action,
                            "action_type": action,  # Raw action type
                            "price": price,
                            "timestamp": timestamp,  # Update timestamp
                        })
                        # #region agent log
                        with open(r'c:\Users\JiantaoPan\OneDrive\Documents\Code\LLM-Investment-Copilot\.cursor\debug.log', 'a') as f:
                            import json
                            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A,B,C,D,E","location":"run_backtest_adaptive_dca.py:225","message":"record_decision updated date","data":{"week_key":str(week_key),"old_date":str(existing_date),"new_date":str(current_date)},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                        # #endregion
                    else:
                        # New date is later than existing date, but existing was non-trading day
                        # This shouldn't happen, but if it does, keep the earlier trading day
                        self._decisions_by_week[week_key].update({
                            "fgi": fgi_value,
                            "action": formatted_action,
                            "action_type": action,  # Raw action type
                            "price": price,
                            "timestamp": timestamp,  # Update timestamp
                        })
                elif existing_price > 0.0 and price > 0.0:
                    # Both are trading days - keep the earlier date (first trading day when trade happened)
                    # Only update FGI/action/price, but keep the original date
                    if current_date < existing_date:
                        # New date is earlier - this is the actual trading day when trade happened
                        self._decisions_by_week[week_key].update({
                            "date": current_date,  # Use earlier trading day
                            "fgi": fgi_value,
                            "action": formatted_action,
                            "action_type": action,  # Raw action type
                            "price": price,
                            "timestamp": timestamp,  # Update timestamp
                        })
                    else:
                        # Keep existing date (earlier trading day)
                        self._decisions_by_week[week_key].update({
                            "fgi": fgi_value,
                            "action": formatted_action,
                            "action_type": action,  # Raw action type
                            "price": price,
                            "timestamp": timestamp,  # Update timestamp
                        })
                else:
                    # Update with latest decision for the week (keep existing date)
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
            week_key = self._period_key(current_date)
            
            # Get portfolio state
            portfolio = self._engine._portfolio
            portfolio_cash = portfolio.state.cash
            
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
                    # TODO: this is a bit clunky and could be improved, by remembering last week's key
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
        
        # Store the adjusted end date in the strategy so it can use it in on_end
        strategy._end_date = last_trading_day
    except Exception as e:
        logging.warning(f"Could not adjust end date using trading calendar: {e}. Using original end date.")
        # end is already set from parameter
        # Still try to set the end date in strategy
        strategy._end_date = end.date() if isinstance(end, datetime) else end
    
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
    
    # Calculate period_count and total_contributions from strategy's actual contribution accounting
    # Strategy tracks contributions accurately as they are made, not using calendar math
    if hasattr(strategy, "total_contributions"):
        total_contributions = float(strategy.total_contributions)
        # Derive period count for display from actual contribution count
        if hasattr(strategy, "contribution_count"):
            period_count = strategy.contribution_count
        else:
            # Fallback: calculate from total contributions
            period_count = int(round(total_contributions / strategy_config.contribution_amount)) if strategy_config.contribution_amount > 0 else 0
    else:
        raise ValueError("Strategy must track total_contributions. This should not happen with AdaptiveDCAStrategy.")
    
    # Run Regular DCA benchmark using BacktestEngine (same method as AdaptiveDCA)
    # Load Regular DCA config from file
    project_root = Path(__file__).parent.parent
    regular_dca_config_path = project_root / "config" / "strategy.regular_dca.yaml"
    
    if not regular_dca_config_path.exists():
        raise FileNotFoundError(
            f"Regular DCA config file not found: {regular_dca_config_path}. "
            f"Please create the config file before running the backtest."
        )
    
    from core.utils.config_loader import load_config_with_secrets
    dca_cfg_raw = load_config_with_secrets(regular_dca_config_path)
    dca_strategy_config = RegularDCAConfig.from_dict(dca_cfg_raw)
    
    # Use the same symbol as AdaptiveDCA (from env.backtest.yaml)
    # Symbol should not be in strategy config files - it comes from backtest config
    dca_symbol = symbol
    
    print(f"\n=== Calculating Regular DCA Benchmark ({dca_symbol}) ===")
    
    # Create Regular DCA strategy
    dca_strategy = RegularDCA(dca_symbol, dca_strategy_config, data_engine)
    
    # Create logger for Regular DCA (suppress verbose output)
    dca_logger = Logger(prefix="[RegularDCA]")
    dca_logger._suppress_logs = True  # Suppress verbose output
    
    # Create scheduler and engine for Regular DCA (daily decisions for weekly DCA)
    dca_scheduler = DecisionScheduler(interval_minutes=24 * 60)
    dca_engine = BacktestEngine(data_engine, dca_scheduler, dca_logger)
    
    # Set end date in strategy (for final price lookup)
    dca_strategy._end_date = end.date()
    
    # Run Regular DCA backtest
    dca_result = await dca_engine.run(
        symbol=dca_symbol,
        strategy=dca_strategy,
        start=start,
        end=end,
        initial_cash=initial_cash,
        timeframe=dca_strategy_config.timeframe,
    )
    
    # Recalculate metrics using DCA-specific calculation for Regular DCA
    # Use strategy's own contribution accounting as source of truth (no calendar math)
    if hasattr(dca_strategy, "total_contributions"):
        dca_total_contributions = float(dca_strategy.total_contributions)
        # Use actual contribution count from strategy
        if hasattr(dca_strategy, "contribution_count"):
            dca_total_weeks = dca_strategy.contribution_count
        else:
            # Fallback: calculate from total contributions
            dca_total_weeks = int(round(dca_total_contributions / dca_strategy_config.contribution_amount)) if dca_strategy_config.contribution_amount > 0 else 0
    else:
        raise ValueError("Regular DCA strategy must track total_contributions. This should not happen with RegularDCA.")
    
    # Calculate total cash contributed (for Regular DCA return calculation)
    dca_total_cash_contributed = initial_cash + dca_total_contributions
    
    # Access the final portfolio state from the engine
    dca_final_cash = dca_engine._portfolio.state.cash if dca_engine._portfolio else 0.0
    dca_final_position = dca_engine._portfolio.state.positions.get(dca_symbol) if dca_engine._portfolio else None
    dca_final_shares = dca_final_position.quantity if dca_final_position else 0.0
    
    # Determine the actual final date to use (same as AdaptiveDCA)
    dca_final_date = end.date()  # This is already adjusted to the last trading day
    
    # Verify this is a trading day
    try:
        from core.data.trading_calendar import is_trading_day
        calendar = get_trading_calendar("NYSE")
        if not is_trading_day(dca_final_date):
            # If somehow not a trading day, get the previous one
            prev_trading_day = calendar.previous_trading_day(dca_final_date)
            if prev_trading_day:
                dca_final_date = prev_trading_day
    except Exception:
        pass  # If trading calendar unavailable, use the date as-is
    
    # Get final price at the determined final date (consistent with strategy)
    dca_final_price = await get_final_price(
        data_engine=data_engine,
        symbol=dca_symbol,
        target_date=dca_final_date,
        timeframe=dca_strategy_config.timeframe,
        equity_curve=dca_result.equity_curve,
        final_shares=dca_final_shares,
        fallback_to_equity_curve=True,
    )
    
    # Calculate total return using common function
    dca_total_return_calc = calculate_dca_total_return(
        final_cash=dca_final_cash,
        final_shares=dca_final_shares,
        final_price=dca_final_price,
        total_cash_contributed=dca_total_cash_contributed,
        symbol=dca_symbol,
        end_date=dca_final_date,
        contribution_frequency=dca_strategy_config.contribution_frequency,
        period_count=dca_total_weeks,
        contribution_amount=dca_strategy_config.contribution_amount,
        initial_cash=initial_cash,
        start_date=start.date(),
        strategy_name="Regular DCA",
        print_steps=True,
    )
    print(f"")
    
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
    

    # Calculate total cash contributed (period_count and total_contributions already calculated above)
    total_cash_contributed = initial_cash + total_contributions
    
    # Get final portfolio state for AdaptiveDCA (cash + shares * final_price)
    adaptive_final_cash = engine._portfolio.state.cash if engine._portfolio else 0.0
    adaptive_final_position = engine._portfolio.state.positions.get(symbol) if engine._portfolio else None
    adaptive_final_shares = adaptive_final_position.quantity if adaptive_final_position else 0.0
    
    # Determine the actual final date to use
    # Use the adjusted end date (last trading day on or before the original end date)
    # This ensures we use the correct last trading day, not just the last equity curve date
    # (which might be earlier if the backtest stopped early)
    adaptive_final_date = end.date()  # This is already adjusted to the last trading day (see lines 434-454)
    
    # Verify this is a trading day
    try:
        from core.data.trading_calendar import is_trading_day
        calendar = get_trading_calendar("NYSE")
        if not is_trading_day(adaptive_final_date):
            # If somehow not a trading day, get the previous one
            prev_trading_day = calendar.previous_trading_day(adaptive_final_date)
            if prev_trading_day:
                adaptive_final_date = prev_trading_day
    except Exception:
        pass  # If trading calendar unavailable, use the date as-is
    
    # Get final price at the determined final date (consistent with strategy)
    # Both use the same get_final_price function with the same date
    adaptive_final_price = await get_final_price(
        data_engine=data_engine,
        symbol=symbol,
        target_date=adaptive_final_date,
        timeframe=timeframe,
        equity_curve=result.equity_curve,
        final_shares=adaptive_final_shares,
        fallback_to_equity_curve=True,
    )
    
    # Recalculate metrics using DCA-specific calculation
    strategy_dca_metrics = evaluate_performance_dca(
        result.equity_curve,
        total_contributions=total_contributions,
        initial_cash=initial_cash
    )
    
    # Calculate total return using common function
    adaptive_total_return = calculate_dca_total_return(
        final_cash=adaptive_final_cash,
        final_shares=adaptive_final_shares,
        final_price=adaptive_final_price,
        total_cash_contributed=total_cash_contributed,
        symbol=symbol,
        end_date=adaptive_final_date,
        contribution_frequency=strategy_config.contribution_frequency,
        period_count=period_count,
        contribution_amount=strategy_config.contribution_amount,
        initial_cash=initial_cash,
        start_date=start.date(),
        strategy_name="AdaptiveDCA",
        print_steps=True,
    )
    
    # Override total return in metrics with our calculation
    strategy_dca_metrics.total_return = adaptive_total_return
    
    print(f"\n=== Performance Metrics (DCA-adjusted) ===")
    print(f"Total Return: {strategy_dca_metrics.total_return:.2%}")
    print(f"CAGR: {strategy_dca_metrics.cagr:.2%}")
    print(f"Volatility: {strategy_dca_metrics.volatility:.2%}")
    print(f"Sharpe Ratio: {strategy_dca_metrics.sharpe:.2f}")
    print(f"Max Drawdown: {strategy_dca_metrics.max_drawdown:.2%}")
    # Calculate Regular DCA performance metrics
    dca_benchmark_metrics = evaluate_performance_dca(
        dca_result.equity_curve,
        total_contributions=dca_total_contributions,
        initial_cash=initial_cash
    )
    # Override total return with our calculation (same method as AdaptiveDCA)
    dca_benchmark_metrics.total_return = dca_total_return_calc
    
    # Print Regular DCA performance metrics
    print(f"\n=== Regular DCA Performance Metrics ===")
    print(f"Total Return: {dca_benchmark_metrics.total_return:.2%}")
    print(f"CAGR: {dca_benchmark_metrics.cagr:.2%}")
    print(f"Volatility: {dca_benchmark_metrics.volatility:.2%}")
    print(f"Sharpe Ratio: {dca_benchmark_metrics.sharpe:.2f}")
    print(f"Max Drawdown: {dca_benchmark_metrics.max_drawdown:.2%}")
    
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
    
    return result, dca_result, strategy


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
    
    # Get symbol from backtest config (env.backtest.yaml)
    symbol = bt_cfg.get("symbol")
    if not symbol:
        raise ValueError("Symbol must be specified in env.backtest.yaml under backtest.symbol")
    
    # Create strategy config
    strategy_config = AdaptiveDCAConfig.from_dict(strat_cfg_raw)
    
    # Print FGI bands and associated actions
    print("\n" + "=" * 80)
    print("FGI (Fear & Greed Index) Bands and Associated Actions")
    print("=" * 80)
    print(f"{'Band':<20} {'FGI Range':<25} {'Action':<25}")
    print("-" * 80)
    print(f"{'Extreme Fear':<20} {f'0.00 - {strategy_config.fgi_extreme_fear_max:.2f}':<25} {strategy_config.action_extreme_fear:<25}")
    print(f"{'Fear':<20} {f'{strategy_config.fgi_extreme_fear_max:.2f} - {strategy_config.fgi_fear_max:.2f}':<25} {strategy_config.action_fear:<25}")
    print(f"{'Neutral':<20} {f'{strategy_config.fgi_fear_max:.2f} - {strategy_config.fgi_neutral_max:.2f}':<25} {strategy_config.action_neutral:<25}")
    print(f"{'Greed':<20} {f'{strategy_config.fgi_neutral_max:.2f} - {strategy_config.fgi_greed_max:.2f}':<25} {strategy_config.action_greed:<25}")
    print(f"{'Extreme Greed':<20} {f'{strategy_config.fgi_extreme_greed_min:.2f} - {strategy_config.fgi_max:.2f}':<25} {strategy_config.action_extreme_greed_sell:<25}")
    print("=" * 80)
    print()
    
    # Create data engine
    # Pass the entire env config (like controlled_panic_bear script does)
    data_engine = create_data_engine_from_config(
        env_config=env_cfg,
        use_for="historical",
    )
    
    # Get timeframe from strategy config or env config
    timeframe = strategy_config.timeframe or env_cfg.get("backtest", {}).get("timeframe", "1D")
    
    # Parse backtest date range
    start = datetime.fromisoformat(bt_cfg["start"])
    end = datetime.fromisoformat(bt_cfg["end"])
    initial_cash = bt_cfg["initial_cash"]
    
    # Print backtest date range
    print("\n" + "=" * 80)
    print("BACKTEST DATE RANGE")
    print("=" * 80)
    print(f"Start Date: {start.date()}")
    print(f"End Date:   {end.date()}")
    print(f"Initial Cash: ${initial_cash:,.2f}")
    print(f"Timeframe: {timeframe}")
    print("=" * 80)
    print()
    
    # Run comparison
    result, dca_result, strategy = await compare_adaptive_dca_vs_dca(
        data_engine,
        symbol,
        strategy_config,
        start,
        end,
        initial_cash,
        timeframe=timeframe,
    )
    
    # Generate chart visualization
    print(f"\n=== Starting Chart Visualization ===")
    try:
        bars = await data_engine.get_bars(symbol, start, end, timeframe)
        
        if bars:
            # Get signals from strategy
            signals = strategy.get_signals() if hasattr(strategy, 'get_signals') else []
            
            metrics_dict = {
                "total_return": result.metrics.total_return,
                "cagr": result.metrics.cagr,
                "volatility": result.metrics.volatility,
                "sharpe": result.metrics.sharpe,
                "max_drawdown": result.metrics.max_drawdown,
            }
            
            visualizer = PlotlyChartVisualizer(theme="tradingview", figsize=(1400, 900))
            chart_config = get_chart_config("adaptive_dca")
            visualizer.build_chart(
                bars=bars,
                signals=signals,
                indicator_data=None,  # AdaptiveDCA doesn't have custom indicators
                equity_curve=result.equity_curve,
                metrics=metrics_dict,
                symbol=symbol,
                show_equity=True,
                chart_config=chart_config,
                strategy_name="adaptive_dca",
            )
            
            print("\n📊 Displaying interactive chart...")
            visualizer.show(renderer="browser")
            
            # Optionally save to HTML
            save_html = input("\nSave chart to HTML file? (y/n): ").strip().lower()
            if save_html == 'y':
                output_file = project_root / "charts" / f"adaptive_dca_{start.date()}_to_{end.date()}.html"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                visualizer.to_html(str(output_file))
                print(f"✅ Chart saved to: {output_file}")
    
    except Exception as e:
        print(f"\n⚠️  Could not generate chart: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
