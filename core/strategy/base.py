from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Protocol, List, Optional, Dict, Any

from core.portfolio.portfolio import Portfolio
from core.execution.base import ExecutionEngine
from core.models.bar import Bar

# Forward declarations to avoid circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from core.data.provider import DataProvider, FGIProvider

class Context(Protocol):
    @property
    def portfolio(self) -> Portfolio: ...
    
    @property
    def execution(self) -> ExecutionEngine: ...
    
    @property
    def symbol(self) -> str: ...
    
    @property
    def now(self) -> datetime: ...
    
    @property
    def config(self) -> Dict[str, Any]: ...
    
    @property
    def data_provider(self) -> "DataProvider": ...
    
    @property
    def fgi_provider(self) -> "FGIProvider": ...
    
    def log(self, msg: str) -> None: ...
    
    def get_bars(self, lookback_days: Optional[int] = None) -> Optional[List[Bar]]: ...

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
