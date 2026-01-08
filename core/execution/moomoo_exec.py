from __future__ import annotations
from typing import Dict, List, Optional
from datetime import datetime
import asyncio

from core.execution.base import ExecutionEngine
from core.models.order import Order, Side, OrderType
from core.models.fill import Fill

# Placeholder imports for moomoo / futu
try:
    from moomoo import OpenTradeContext  # type: ignore
    from futu import TrdSide, OrderType as FutuOrderType  # type: ignore
except Exception:  # pragma: no cover
    OpenTradeContext = object  # type: ignore
    class TrdSide:  # type: ignore
        BUY = None
        SELL = None
    class FutuOrderType:  # type: ignore
        MARKET = None
        NORMAL = None

class MoomooExecutionEngine(ExecutionEngine):
    def __init__(self, host: str = "127.0.0.1", port: int = 11111, account_id: str | None = None):
        self._ctx = OpenTradeContext(host=host, port=port)
        self._account_id = account_id
        self._open_orders: Dict[str, Order] = {}

    async def submit_order(self, order: Order) -> Order:
        def _place():
            trd_side = TrdSide.BUY if order.side == Side.BUY else TrdSide.SELL
            if order.order_type == OrderType.MARKET:
                futu_type = FutuOrderType.MARKET
                price = 0
            else:
                futu_type = FutuOrderType.NORMAL
                price = order.limit_price or 0
            try:
                ret, data = self._ctx.place_order(
                    price=price,
                    qty=order.quantity,
                    code=order.symbol,
                    trd_side=trd_side,
                    order_type=futu_type,
                    acc_id=self._account_id,
                )
            except Exception as e:  # pragma: no cover
                raise RuntimeError(f"moomoo place_order exception: {e}")
            if ret != 0:
                raise RuntimeError(f"moomoo place_order error: {data}")
            order_id = str(data["order_id"][0])
            return order_id

        order_id = await asyncio.to_thread(_place)
        order.id = order_id
        self._open_orders[order_id] = order
        return order

    async def get_open_orders(self) -> List[Order]:
        return list(self._open_orders.values())

    async def cancel_order(self, order_id: str) -> None:
        def _cancel():
            try:
                ret, data = self._ctx.cancel_order(order_id=order_id)
            except Exception as e:  # pragma: no cover
                raise RuntimeError(f"moomoo cancel_order exception: {e}")
            if ret != 0:
                raise RuntimeError(f"moomoo cancel_order error: {data}")
        await asyncio.to_thread(_cancel)
        self._open_orders.pop(order_id, None)

    async def process_pending(self, timestamp: Optional[datetime] = None) -> List[Fill]:
        """
        Process in-flight orders and return fills.
        
        Args:
            timestamp: Optional timestamp (ignored in live trading, uses actual fill time)
        
        Returns:
            List of Fill objects for executed orders
        """
        # TODO: poll moomoo for fills and translate into Fill objects.
        return []
