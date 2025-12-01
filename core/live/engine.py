from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
import asyncio

from core.data.moomoo_data import MoomooDataAdapter
from core.execution.moomoo_exec import MoomooExecutionEngine
from core.portfolio.portfolio import Portfolio
from core.strategy.base import Strategy
from core.utils.logging import Logger

@dataclass
class LiveEngineConfig:
    interval_minutes: int = 15

class LiveEngine:
    def __init__(
        self,
        data_engine: MoomooDataAdapter,
        exec_engine: MoomooExecutionEngine,
        portfolio: Portfolio,
        strategy: Strategy,
        logger: Logger,
        cfg: LiveEngineConfig,
    ):
        self._data = data_engine
        self._exec = exec_engine
        self._portfolio = portfolio
        self._strategy = strategy
        self._logger = logger
        self._cfg = cfg

    async def run(self, symbol: str, stop_event: asyncio.Event):
        class LiveContext:
            def __init__(self, portfolio, execution, logger, symbol):
                self.portfolio = portfolio
                self.execution = execution
                self.logger = logger
                self.symbol = symbol

            def log(self, msg: str) -> None:
                self.logger.log(msg)

        ctx = LiveContext(self._portfolio, self._exec, self._logger, symbol)
        await self._strategy.on_start(ctx)

        interval = self._cfg.interval_minutes * 60

        while not stop_event.is_set():
            now = datetime.now(timezone.utc)
            await self._strategy.on_decision(ctx, now)
            await asyncio.sleep(interval)

        await self._strategy.on_end(ctx)
