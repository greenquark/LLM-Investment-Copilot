from __future__ import annotations
from datetime import datetime
from typing import Optional

class Logger:
    def __init__(self, prefix: str = ""):
        self._prefix = prefix
        self._current_timestamp: Optional[datetime] = None

    def set_timestamp(self, timestamp: datetime) -> None:
        """Set the current timestamp for logging (used in backtests)."""
        self._current_timestamp = timestamp

    def clear_timestamp(self) -> None:
        """Clear the timestamp override (revert to using current time)."""
        self._current_timestamp = None

    def log(self, msg: str) -> None:
        # Use provided timestamp if available (for backtests), otherwise use current time
        if self._current_timestamp is not None:
            ts = self._current_timestamp.isoformat()
        else:
            ts = datetime.utcnow().isoformat()
        print(f"{ts} {self._prefix} {msg}")
