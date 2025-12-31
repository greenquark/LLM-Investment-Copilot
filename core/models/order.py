from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from datetime import datetime

class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"

class InstrumentType(str, Enum):
    STOCK = "STOCK"
    OPTION = "OPTION"

@dataclass
class OptionDetails:
    underlying: str
    expiry: datetime
    strike: float
    right: str   # "C" or "P"

@dataclass
class Order:
    id: str
    symbol: str
    side: Side
    quantity: float  # Changed to float to support fractional shares
    order_type: OrderType
    limit_price: Optional[float]
    instrument_type: InstrumentType
    option: Optional[OptionDetails] = None
    time_in_force: str = "DAY"
    created_at: Optional[datetime] = None
