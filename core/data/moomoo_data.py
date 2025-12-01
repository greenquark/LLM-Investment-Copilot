from __future__ import annotations
from datetime import datetime, date
from typing import AsyncIterator, List
import asyncio

from core.data.base import DataEngine
from core.models.bar import Bar
from core.models.option import OptionContract

# NOTE: This module assumes you have moomoo / futu API installed and configured.
# The imports below are placeholders; adapt to your installed SDK.
try:
    from moomoo import OpenQuoteContext  # type: ignore
    from futu import KLType  # type: ignore
except Exception:  # pragma: no cover - SDK not present in this skeleton
    OpenQuoteContext = object  # type: ignore
    class KLType:  # type: ignore
        K_15M = None
        K_1M = None

class MoomooDataAdapter(DataEngine):
    def __init__(self, host: str = "127.0.0.1", port: int = 11111):
        self._ctx = OpenQuoteContext(host=host, port=port)
        self._last_data_source = "Moomoo"  # Track data source name for last request

    async def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str,
    ) -> List[Bar]:
        if timeframe == "15m":
            kl_type = KLType.K_15M
        elif timeframe == "1m":
            kl_type = KLType.K_1M
        else:
            raise ValueError(f"Unsupported timeframe {timeframe}")

        def _fetch():
            # This is placeholder logic; replace with the correct futu API calls.
            try:
                ret, df, err = self._ctx.request_history_kline(
                    code=symbol,
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    ktype=kl_type,
                )
            except Exception as e:  # pragma: no cover - SDK not present
                raise RuntimeError(f"moomoo kline exception: {e}")
            if ret != 0:
                raise RuntimeError(f"moomoo kline error: {err}")
            return df

        df = await asyncio.to_thread(_fetch)

        bars: List[Bar] = []
        # Normalize timestamps to timezone-naive UTC for consistency
        from core.utils.timestamp import normalize_timestamp
        
        for _, row in df.iterrows():
            ts = row["time_key"]
            if hasattr(ts, "to_pydatetime"):
                ts = ts.to_pydatetime()
            
            # Normalize timestamp to timezone-naive UTC
            ts_normalized = normalize_timestamp(ts)
            
            if ts_normalized < start or ts_normalized > end:
                continue
            bars.append(
                Bar(
                    symbol=symbol,
                    timestamp=ts_normalized,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row.get("volume", 0)),
                    timeframe=timeframe,
                )
            )
        return bars

    async def stream_bars(
        self,
        symbol: str,
        timeframe: str,
    ) -> AsyncIterator[Bar]:
        # TODO: implement real-time subscription to moomoo / futu API
        raise NotImplementedError("MoomooDataAdapter.stream_bars is not implemented")

    async def get_option_chain(
        self,
        underlying: str,
        as_of: date,
    ) -> list[OptionContract]:
        # TODO: wrap moomoo option chain API here.
        raise NotImplementedError("MoomooDataAdapter.get_option_chain is not implemented")
