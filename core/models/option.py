from __future__ import annotations
from dataclasses import dataclass
from datetime import date

@dataclass
class OptionContract:
    symbol: str              # OCC / broker symbol
    underlying: str
    expiry: date
    strike: float
    right: str               # "C" or "P"
