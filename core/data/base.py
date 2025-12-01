from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import datetime, date
from typing import AsyncIterator, List

from core.models.bar import Bar
from core.models.option import OptionContract

class DataEngine(ABC):
    @abstractmethod
    async def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str,
    ) -> List[Bar]:
        ...

    @abstractmethod
    async def stream_bars(
        self,
        symbol: str,
        timeframe: str,
    ) -> AsyncIterator[Bar]:
        ...

    @abstractmethod
    async def get_option_chain(
        self,
        underlying: str,
        as_of: date,
    ) -> list[OptionContract]:
        ...
