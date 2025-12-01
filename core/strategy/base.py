from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Protocol

from core.portfolio.portfolio import Portfolio
from core.execution.base import ExecutionEngine

class Context(Protocol):
    @property
    def portfolio(self) -> Portfolio: ...
    
    @property
    def execution(self) -> ExecutionEngine: ...
    
    @property
    def symbol(self) -> str: ...
    
    def log(self, msg: str) -> None: ...

class Strategy(ABC):
    @abstractmethod
    async def on_start(self, ctx: Context) -> None:
        ...

    @abstractmethod
    async def on_decision(self, ctx: Context, now: datetime) -> None:
        ...

    @abstractmethod
    async def on_end(self, ctx: Context) -> None:
        ...
