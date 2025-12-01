from __future__ import annotations
from datetime import datetime

class Logger:
    def __init__(self, prefix: str = ""):
        self._prefix = prefix

    def log(self, msg: str) -> None:
        ts = datetime.utcnow().isoformat()
        print(f"{ts} {self._prefix} {msg}")
