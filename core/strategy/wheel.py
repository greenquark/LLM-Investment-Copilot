from __future__ import annotations
from datetime import datetime

from core.strategy.base import Strategy, Context
from core.models.order import Order, Side, OrderType, InstrumentType
from core.data.wheel_view import WheelDataView

class WheelStrategyConfig:
    def __init__(self, raw: dict):
        self.max_csp_contracts = raw.get("max_csp_contracts", 4)
        self.max_cc_contracts = raw.get("max_cc_contracts", 4)
        self.roll_dte_threshold = raw.get("roll_dte_threshold", 5)
        self.bias_rules = raw.get("bias_rules", {})

class WheelStrategy(Strategy):
    def __init__(self, symbol: str, config: WheelStrategyConfig, wheel_data_view: WheelDataView):
        self._symbol = symbol
        self._cfg = config
        self._view = wheel_data_view

    async def on_start(self, ctx: Context) -> None:
        ctx.log(f"WheelStrategy started for {self._symbol}")

    async def on_decision(self, ctx: Context, now: datetime) -> None:
        data = await self._view.get_15m_context(now)

        # Placeholder bias logic – replace with your DBS / canvas implementation.
        if data.price_now > data.ma20 and data.ma20 > data.ma60:
            bias = "bullish"
        elif data.price_now < data.ma20 and data.ma20 < data.ma60:
            bias = "bearish"
        else:
            bias = "neutral"

        ctx.log(
            f"[{now}] price={data.price_now:.2f} ma5={data.ma5:.2f} "
            f"ma20={data.ma20:.2f} ma60={data.ma60:.2f} bias={bias}"
        )

        positions = ctx.portfolio.get_positions()
        # TODO: inspect positions to distinguish CSP vs CC options.

        # TODO: manage existing options (roll, close, etc.) using your rules.

        # Simple placeholder: if bullish, sell one CSP using first put in chain.
        if bias == "bullish":
            puts = [c for c in data.option_chain if c.right == "P"]
            if not puts:
                return
            target = puts[0]

            order = Order(
                id="",
                symbol=target.symbol,
                side=Side.SELL,
                quantity=1,
                order_type=OrderType.MARKET,
                limit_price=None,
                instrument_type=InstrumentType.OPTION,
            )
            await ctx.execution.submit_order(order)  # type: ignore[attr-defined]
            ctx.log(f"Submitted CSP on {target.symbol}")

    async def on_end(self, ctx: Context) -> None:
        ctx.log("WheelStrategy finished")
