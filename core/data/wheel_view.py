from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List

import numpy as np

from core.data.base import DataEngine
from core.models.bar import Bar
from core.models.option import OptionContract

@dataclass
class WheelContextData:
    now: datetime
    price_now: float
    bars_15m: List[Bar]
    ma5: float
    ma20: float
    ma60: float
    bb_upper: float
    bb_lower: float
    option_chain: list[OptionContract]

class WheelDataView:
    def __init__(self, data_engine: DataEngine, symbol: str):
        self._data = data_engine
        self._symbol = symbol

    async def get_15m_context(
        self,
        now: datetime,
        lookback_days: int = 10,
    ) -> WheelContextData:
        start = now - timedelta(days=lookback_days)
        bars = await self._data.get_bars(self._symbol, start, now, timeframe="15m")
        if not bars:
            raise RuntimeError("No 15m bars returned for WheelContextData")

        closes = np.array([b.close for b in bars], dtype=float)
        ma5 = float(closes[-5:].mean()) if len(closes) >= 5 else float(closes.mean())
        ma20 = float(closes[-20:].mean()) if len(closes) >= 20 else ma5
        ma60 = float(closes[-60:].mean()) if len(closes) >= 60 else ma20

        if len(closes) >= 20:
            window = closes[-20:]
            mid = window.mean()
            std = window.std()
            bb_upper = float(mid + 2 * std)
            bb_lower = float(mid - 2 * std)
        else:
            bb_upper = float(closes.max())
            bb_lower = float(closes.min())

        option_chain = await self._data.get_option_chain(self._symbol, now.date())

        return WheelContextData(
            now=now,
            price_now=float(closes[-1]),
            bars_15m=bars,
            ma5=ma5,
            ma20=ma20,
            ma60=ma60,
            bb_upper=bb_upper,
            bb_lower=bb_lower,
            option_chain=option_chain,
        )
