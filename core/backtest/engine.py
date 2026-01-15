from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any

from core.backtest.scheduler import DecisionScheduler
from core.backtest.result import BacktestResult, PerformanceMetrics
from core.backtest.performance import evaluate_performance
from core.backtest.data_provider import BacktestDataProvider
from core.backtest.equity_tracker import EquityTracker
from core.backtest.orchestrator import BacktestOrchestrator
from core.backtest.context import SimpleContext
from core.data.base import DataEngine
from core.data.provider import DataEngineProvider, FearGreedIndexProvider
from core.execution.simulated import SimulatedExecutionEngine
from core.portfolio.portfolio import Portfolio, PortfolioState
from core.strategy.base import Strategy
from core.utils.logging import Logger
from core.models.bar import Bar

class BacktestEngine:
    """
    Backtest Engine - Thin wrapper around BacktestOrchestrator.
    
    This class maintains the existing public API while delegating to
    the orchestrator for actual execution.
    """
    def __init__(
        self,
        data_engine: DataEngine,
        scheduler: DecisionScheduler,
        logger: Logger,
    ):
        self._data = data_engine
        self._scheduler = scheduler
        self._logger = logger
        
        # Create components
        self._data_provider = BacktestDataProvider(data_engine)
        self._equity_tracker = EquityTracker()
        self._orchestrator = BacktestOrchestrator(
            data_provider=self._data_provider,
            equity_tracker=self._equity_tracker,
            scheduler=scheduler,
            logger=logger,
        )
        
        # Store state for partial result retrieval and backward compatibility
        self._equity_curve: Dict[datetime, float] = {}
        self._portfolio: Portfolio | None = None
        self._exec_engine: SimulatedExecutionEngine | None = None
        self._strategy: Strategy | None = None
        self._ctx: SimpleContext | None = None

    async def run(
        self,
        symbol: str,
        strategy: Strategy,
        start: datetime,
        end: datetime,
        initial_cash: float = 100_000.0,
        timeframe: str = "15m",
        # DCA-specific parameters
        is_dca: bool = False,
        contribution_amount: float = 0.0,
        contribution_frequency: str = "weekly",  # "weekly", "monthly", "quarterly", "yearly"
    ) -> BacktestResult:
        portfolio = Portfolio(PortfolioState(cash=initial_cash))
        exec_engine = SimulatedExecutionEngine()
        now = self._scheduler.align(start)

        # Store state for partial result retrieval
        self._portfolio = portfolio
        self._exec_engine = exec_engine
        self._strategy = strategy
        self._equity_curve = {}

        # Create providers
        data_provider = DataEngineProvider(self._data)
        fgi_provider = FearGreedIndexProvider()
        
        ctx = SimpleContext(
            portfolio=portfolio,
            execution=exec_engine,
            logger=self._logger,
            symbol=symbol,
            _data_provider=data_provider,
            _fgi_provider=fgi_provider,
        )
        self._ctx = ctx
        
        # Set initial logger timestamp
        self._logger.set_timestamp(now)
        
        await strategy.on_start(ctx)

        try:
            while now <= end:
                # Signal generation timing:
                # - Decision point is at market close (4:00 PM ET = 16:00) for daily bars
                # - We evaluate the complete bar that just closed (same day)
                # - Execution happens at that day's close price
                # - In reality, execution can happen in extended hours after market close
                # TODO: Future improvement - Generate signals 15 minutes before market close (3:45 PM ET)
                #       This would allow execution at the close price on the same day the signal is generated
                
                # Fetch bars up to the current decision point with maximum lookback needed by strategy
                # This avoids duplicate fetches - engine fetches once with max lookback, strategy reuses
                from datetime import timedelta
                if timeframe.upper() in ("D", "1D") or (timeframe.upper().endswith("D") and not timeframe.upper().startswith("W")):
                    # For daily bars, use strategy's required bars to calculate trading days lookback
                    # Check if strategy has a method to get minimum required bars
                    min_bars_required = 250  # Default fallback (covers most strategies)
                    if hasattr(strategy, '_cfg') and hasattr(strategy._cfg, 'get_min_bars_required'):
                        min_bars_required = strategy._cfg.get_min_bars_required()
                    
                    # Use trading days lookback instead of calendar days
                    # This ensures we fetch enough bars (e.g., 200 trading days, not 30 calendar days)
                    # Calculate the exact trading day we need to start from to get min_bars_required bars including today
                    from core.utils.trading_days import get_trading_days
                    # Get trading days from a date far enough back to today
                    # We need exactly min_bars_required bars including today (or the last trading day if today isn't one)
                    estimated_calendar_days = int(min_bars_required * 1.5)  # Account for weekends/holidays
                    start_date = now.date() - timedelta(days=estimated_calendar_days)
                    trading_days_list = get_trading_days(start_date, now.date(), exchange="NYSE")
                    # Normalize to Python date objects and sort
                    normalized_trading_days = []
                    for d in trading_days_list:
                        if hasattr(d, 'date') and callable(getattr(d, 'date', None)):
                            normalized_trading_days.append(d.date())
                        elif isinstance(d, date):
                            normalized_trading_days.append(d)
                        else:
                            try:
                                normalized_trading_days.append(date.fromisoformat(str(d)))
                            except (ValueError, AttributeError):
                                continue
                    normalized_trading_days = sorted(set(normalized_trading_days))
                    # Take the last min_bars_required trading days - the first one is our start date
                    if len(normalized_trading_days) >= min_bars_required:
                        lookback_date = normalized_trading_days[-min_bars_required]
                    else:
                        # Not enough trading days in range, try with a larger range
                        start_date = now.date() - timedelta(days=int(min_bars_required * 2))
                        trading_days_list = get_trading_days(start_date, now.date(), exchange="NYSE")
                        normalized_trading_days = []
                        for d in trading_days_list:
                            if hasattr(d, 'date') and callable(getattr(d, 'date', None)):
                                normalized_trading_days.append(d.date())
                            elif isinstance(d, date):
                                normalized_trading_days.append(d)
                            else:
                                try:
                                    normalized_trading_days.append(date.fromisoformat(str(d)))
                                except (ValueError, AttributeError):
                                    continue
                        normalized_trading_days = sorted(set(normalized_trading_days))
                        if len(normalized_trading_days) >= min_bars_required:
                            lookback_date = normalized_trading_days[-min_bars_required]
                        else:
                            # Fallback: use the first trading day in the list
                            lookback_date = normalized_trading_days[0] if normalized_trading_days else start_date
                    # Convert to datetime - use start of day (00:00:00) for daily bars to ensure we include the bar for that day
                    # Using the time component from now (16:00:00) can cause issues with timestamp normalization
                    lookback_start = datetime.combine(lookback_date, datetime.min.time())
                    # For daily bars, also set end to start of day to ensure we only get bars up to and including today
                    # This prevents including bars from the next trading day
                    if timeframe.upper() in ("D", "1D") or (timeframe.upper().endswith("D") and not timeframe.upper().startswith("W")):
                        end_datetime = datetime.combine(now.date(), datetime.min.time())
                    else:
                        end_datetime = now
                    bars = await self._data.get_bars(symbol, lookback_start, end_datetime, timeframe=timeframe)
                elif timeframe.upper() in ("W", "1W") or timeframe.upper().endswith("W"):
                    # For weekly: get bars up to current decision point
                    lookback_start = now - timedelta(weeks=14)
                    bars = await self._data.get_bars(symbol, lookback_start, now, timeframe=timeframe)
                else:
                    # For intraday (hourly, minutely): strategy needs longer lookback
                    if timeframe.upper().endswith("H") or timeframe.upper() == "H":
                        # Hourly: strategy needs 30 days, engine fetches that (instead of 7)
                        lookback_start = now - timedelta(days=30)
                    else:
                        # Minutely: strategy needs 7 days, engine fetches that (instead of 1)
                        lookback_start = now - timedelta(days=7)
                    bars = await self._data.get_bars(symbol, lookback_start, now, timeframe=timeframe)
                
                # Update context with data source from data engine
                if hasattr(self._data, 'last_data_source'):
                    ctx._data_source = self._data.last_data_source
                elif hasattr(self._data, '_last_data_source'):
                    ctx._data_source = self._data._last_data_source
                
                # Set execution price to the closing price of the bar that just closed
                # This is the price at which orders will be filled (can execute in extended hours)
                if bars:
                    last_bar = bars[-1]
                    # Use the close price of the most recent complete bar (today's close)
                    exec_engine.update_market_price(symbol, last_bar.close)
                
                # Pass bars to context so strategy can reuse them (avoids duplicate fetch)
                ctx._bars = bars
                ctx._now = now  # Set current decision timestamp

                # Set logger timestamp to backtest decision time (so logs show backtest date, not current time)
                self._logger.set_timestamp(now)
                # #region agent log
                with open(r'c:\Users\JiantaoPan\OneDrive\Documents\Code\LLM-Investment-Copilot\.cursor\debug.log', 'a') as f:
                    import json
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"E","location":"engine.py:132","message":"calling on_decision","data":{"now":str(now),"now_date":str(now.date()),"bars_count":len(bars) if bars else 0,"last_bar_timestamp":str(bars[-1].timestamp) if bars else None},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                # #endregion
                await strategy.on_decision(ctx, now)
                
                # Clear timestamp after decision (optional, but clean)
                # self._logger.clear_timestamp()  # Keep timestamp for any post-decision logging
                
                # Update market price from strategy's current bar if available
                # This ensures we have the latest price for order processing
                if hasattr(strategy, 'get_bars') and hasattr(strategy, '_bars'):
                    strategy_bars = strategy.get_bars()  # type: ignore[attr-defined]
                    if strategy_bars:
                        latest_bar = strategy_bars[-1]
                        exec_engine.update_market_price(symbol, latest_bar.close)

                # Process pending orders using the decision timestamp (now)
                # This ensures fills have the correct timestamp matching the decision point
                fills = await exec_engine.process_pending(timestamp=now)
                for fill in fills:
                    portfolio.apply_fill(fill)

                # Update equity curve with current market price
                # Use the last known price from execution engine
                current_price = exec_engine._last_price.get(symbol, 0.0)  # type: ignore[attr-defined]
                if current_price == 0.0:
                    # Fallback: if no price available, use bars we already fetched
                    if bars:
                        current_price = bars[-1].close
                        exec_engine.update_market_price(symbol, current_price)
                    else:
                        # Only fetch if we somehow don't have bars (shouldn't happen)
                        from datetime import timedelta
                        lookback_start = now - timedelta(days=7)
                        fallback_bars = await self._data.get_bars(symbol, lookback_start, now, timeframe=timeframe)
                        if fallback_bars:
                            current_price = fallback_bars[-1].close
                            exec_engine.update_market_price(symbol, current_price)
                
                if current_price > 0:
                    prices = {symbol: current_price}
                    self._equity_curve[now] = portfolio.equity(prices)
                else:
                    # If still no price, use previous equity value or initial cash
                    if self._equity_curve:
                        # Use last known equity
                        last_equity = list(self._equity_curve.values())[-1]
                        self._equity_curve[now] = last_equity
                    else:
                        self._equity_curve[now] = initial_cash

                now = self._scheduler.next(now)
                # Update logger timestamp to new decision time
                self._logger.set_timestamp(now)

            await strategy.on_end(ctx)
        except KeyboardInterrupt:
            self._logger.log(f"\nBacktest interrupted at {now}. Computing partial results...")
            # Try to clean up strategy state if possible
            try:
                await strategy.on_end(ctx)
            except Exception:
                pass
            raise

        # Return raw results - performance evaluation will be done by the calling script
        # Create a minimal metrics object (will be recalculated by script)
        from core.backtest.result import PerformanceMetrics
        dummy_metrics = PerformanceMetrics(
            total_return=0.0,
            cagr=0.0,
            volatility=0.0,
            sharpe=0.0,
            max_drawdown=0.0,
            max_drawdown_start=None,
            max_drawdown_end=None,
        )
        return BacktestResult(equity_curve=self._equity_curve, metrics=dummy_metrics)

    def get_partial_result(self) -> BacktestResult | None:
        """Get partial backtest results if available (e.g., after interruption)."""
        if not self._equity_curve:
            return None
        
        try:
            metrics = evaluate_performance(self._equity_curve)
            return BacktestResult(equity_curve=self._equity_curve, metrics=metrics)
        except Exception:
            return None
