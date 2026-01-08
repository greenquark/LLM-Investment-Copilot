from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional
from datetime import datetime

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
    async def process_pending(self, timestamp: Optional[datetime] = None) -> List[Fill]:
        """
        Process in-flight orders and return fills.
        
        Args:
            timestamp: Optional timestamp to use for fills. If None, uses current time.
                        In backtests, this should be the decision timestamp.
        
        Returns:
            List of Fill objects for executed orders
        """
        ...
