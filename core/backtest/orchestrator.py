"""
Backtest Orchestrator - High-level orchestration of backtest execution.

This module provides the main orchestration logic for running backtests,
coordinating data provider, equity tracker, and strategy execution.
"""

from __future__ import annotations
from datetime import datetime

from core.backtest.data_provider import BacktestDataProvider
from core.backtest.equity_tracker import EquityTracker
from core.backtest.context import SimpleContext
from core.backtest.scheduler import DecisionScheduler
from core.execution.simulated import SimulatedExecutionEngine
from core.strategy.base import Strategy
from core.utils.logging import Logger


class BacktestOrchestrator:
    """
    High-level orchestrator for backtest execution.
    
    This class coordinates data fetching, equity tracking, and strategy execution,
    separating concerns from the execution loop.
    """
    
    def __init__(
        self,
        data_provider: BacktestDataProvider,
        equity_tracker: EquityTracker,
        scheduler: DecisionScheduler,
        logger: Logger,
    ):
        """
        Initialize the backtest orchestrator.
        
        Args:
            data_provider: BacktestDataProvider for fetching bars
            equity_tracker: EquityTracker for tracking equity curve
            scheduler: DecisionScheduler for decision timing
            logger: Logger for logging
        """
        self._data_provider = data_provider
        self._equity_tracker = equity_tracker
        self._scheduler = scheduler
        self._logger = logger
    
    async def run(
        self,
        symbol: str,
        strategy: Strategy,
        start: datetime,
        end: datetime,
        initial_cash: float,
        timeframe: str,
        ctx: SimpleContext,
        exec_engine: SimulatedExecutionEngine,
    ) -> None:
        """
        Run the backtest orchestration loop.
        
        This method coordinates the backtest execution, calling the strategy
        at each decision point and tracking equity.
        
        Args:
            symbol: Stock symbol
            strategy: Strategy to run
            start: Start datetime
            end: End datetime
            initial_cash: Initial cash
            timeframe: Bar timeframe
            ctx: Context for strategy
            exec_engine: Execution engine for order processing
        """
        now = self._scheduler.align(start)
        
        # Set initial logger timestamp
        self._logger.set_timestamp(now)
        
        await strategy.on_start(ctx)
        
        try:
            while now <= end:
                # Fetch bars with appropriate lookback
                bars = await self._data_provider.get_bars(symbol, now, timeframe)
                
                # Update execution price from bars
                if bars:
                    last_bar = bars[-1]
                    exec_engine.update_market_price(symbol, last_bar.close)
                    self._equity_tracker.update_price(symbol, last_bar.close)
                
                # Pass bars to context
                ctx._bars = bars
                ctx._now = now
                
                # Set logger timestamp
                self._logger.set_timestamp(now)
                
                # Call strategy decision
                await strategy.on_decision(ctx, now)
                
                # Process pending orders
                fills = await exec_engine.process_pending(timestamp=now)
                for fill in fills:
                    ctx.portfolio.apply_fill(fill)
                
                # Record equity
                fallback_price = bars[-1].close if bars else None
                self._equity_tracker.record_equity(
                    timestamp=now,
                    portfolio=ctx.portfolio,
                    symbol=symbol,
                    fallback_price=fallback_price,
                )
                
                # Move to next decision point
                now = self._scheduler.next(now)
                self._logger.set_timestamp(now)
            
            await strategy.on_end(ctx)
        except KeyboardInterrupt:
            self._logger.log(f"\nBacktest interrupted at {now}.")
            try:
                await strategy.on_end(ctx)
            except Exception:
                pass
            raise
