from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

@dataclass
class Position:
    symbol: str
    quantity: float  # Changed to float to support fractional shares
    avg_price: float
    instrument_type: str      # "STOCK" or "OPTION"
    last_updated: Optional[datetime] = None
