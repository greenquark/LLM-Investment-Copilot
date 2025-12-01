from __future__ import annotations
from typing import Dict, List
from datetime import datetime
from uuid import uuid4

from core.execution.base import ExecutionEngine
from core.models.order import Order, OrderType, Side
from core.models.fill import Fill

class SimulatedExecutionEngine(ExecutionEngine):
    def __init__(self):
        self._open_orders: Dict[str, Order] = {}
        self._fills: List[Fill] = []
        self._last_price: Dict[str, float] = {}

    def update_market_price(self, symbol: str, price: float):
        self._last_price[symbol] = price

    async def submit_order(self, order: Order) -> Order:
        if not order.id:
            order.id = str(uuid4())
        self._open_orders[order.id] = order
        return order

    async def get_open_orders(self) -> List[Order]:
        return list(self._open_orders.values())

    async def cancel_order(self, order_id: str) -> None:
        self._open_orders.pop(order_id, None)

    async def process_pending(self) -> List[Fill]:
        fills: List[Fill] = []
        to_delete: List[str] = []
        now = datetime.utcnow()

        for oid, order in list(self._open_orders.items()):
            px = self._last_price.get(order.symbol)
            if px is None:
                continue

            if order.order_type == OrderType.MARKET:
                fill_price = px
            else:
                if order.side == Side.BUY and px <= (order.limit_price or px):
                    fill_price = px
                elif order.side == Side.SELL and px >= (order.limit_price or px):
                    fill_price = px
                else:
                    continue

            qty_signed = order.quantity if order.side == Side.BUY else -order.quantity
            fill = Fill(
                order_id=oid,
                symbol=order.symbol,
                quantity=qty_signed,
                price=fill_price,
                commission=0.0,
                timestamp=now,
            )
            fills.append(fill)
            to_delete.append(oid)

        for oid in to_delete:
            del self._open_orders[oid]

        self._fills.extend(fills)
        return fills
