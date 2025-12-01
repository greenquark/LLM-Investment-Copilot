from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict
from .position import Position

@dataclass
class PortfolioState:
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
