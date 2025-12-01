from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List

from core.models.order import Order
from core.models.fill import Fill

class ExecutionEngine(ABC):
    @abstractmethod
    async def submit_order(self, order: Order) -> Order:
        ...

    @abstractmethod
    async def get_open_orders(self) -> List[Order]:
        ...

    @abstractmethod
    async def cancel_order(self, order_id: str) -> None:
        ...

    @abstractmethod
    async def process_pending(self) -> List[Fill]:
        """Process in-flight orders and return fills."""
        ...
