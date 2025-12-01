from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from core.backtest.scheduler import DecisionScheduler
from core.backtest.result import BacktestResult
from core.backtest.performance import evaluate_performance
from core.data.base import DataEngine
from core.execution.simulated import SimulatedExecutionEngine
from core.portfolio.portfolio import Portfolio, PortfolioState
from core.strategy.base import Strategy
from core.utils.logging import Logger
from core.models.bar import Bar

@dataclass
class SimpleContext:
    portfolio: Portfolio
    execution: SimulatedExecutionEngine
    logger: Logger
    symbol: str
    _data_source: str = ""  # Track data source name (e.g., "Cache", "MarketData", "Moomoo")
    _bars: Optional[List[Bar]] = None  # Bars fetched by engine, can be reused by strategy

    def log(self, msg: str) -> None:
        # Add data source prefix if available
        prefix = f"[{self._data_source}] " if self._data_source else ""
        self.logger.log(f"{prefix}{msg}")

class BacktestEngine:
    def __init__(
        self,
        data_engine: DataEngine,
        scheduler: DecisionScheduler,
        logger: Logger,
    ):
        self._data = data_engine
        self._scheduler = scheduler
        self._logger = logger
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
    ) -> BacktestResult:
        portfolio = Portfolio(PortfolioState(cash=initial_cash))
        exec_engine = SimulatedExecutionEngine()
        now = self._scheduler.align(start)

        # Store state for partial result retrieval
        self._portfolio = portfolio
        self._exec_engine = exec_engine
        self._strategy = strategy
        self._equity_curve = {}

        ctx = SimpleContext(
            portfolio=portfolio,
            execution=exec_engine,
            logger=self._logger,
            symbol=symbol,
        )
        self._ctx = ctx
        
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
                    # For daily: strategy needs 30 days, engine fetches that
                    lookback_start = now - timedelta(days=30)
                    bars = await self._data.get_bars(symbol, lookback_start, now, timeframe=timeframe)
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

                await strategy.on_decision(ctx, now)
                
                # Update market price from strategy's current bar if available
                # This ensures we have the latest price for order processing
                if hasattr(strategy, 'get_bars') and hasattr(strategy, '_bars'):
                    strategy_bars = strategy.get_bars()  # type: ignore[attr-defined]
                    if strategy_bars:
                        latest_bar = strategy_bars[-1]
                        exec_engine.update_market_price(symbol, latest_bar.close)

                fills = await exec_engine.process_pending()
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

            await strategy.on_end(ctx)
        except KeyboardInterrupt:
            self._logger.log(f"\nBacktest interrupted at {now}. Computing partial results...")
            # Try to clean up strategy state if possible
            try:
                await strategy.on_end(ctx)
            except Exception:
                pass
            raise

        metrics = evaluate_performance(self._equity_curve)
        self._logger.log(
            f"Backtest finished: TotalReturn={metrics.total_return:.2%}, "
            f"CAGR={metrics.cagr:.2%}, Sharpe={metrics.sharpe:.2f}, "
            f"MaxDD={metrics.max_drawdown:.2%}"
        )
        return BacktestResult(equity_curve=self._equity_curve, metrics=metrics)

    def get_partial_result(self) -> BacktestResult | None:
        """Get partial backtest results if available (e.g., after interruption)."""
        if not self._equity_curve:
            return None
        
        try:
            metrics = evaluate_performance(self._equity_curve)
            return BacktestResult(equity_curve=self._equity_curve, metrics=metrics)
        except Exception:
            return None
