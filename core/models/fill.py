from dataclasses import dataclass
from datetime import datetime

@dataclass
class Fill:
    order_id: str
    symbol: str
    quantity: int
    price: float
    commission: float
    timestamp: datetime
