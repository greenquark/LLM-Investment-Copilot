from __future__ import annotations
from datetime import datetime, timedelta

class DecisionScheduler:
    def __init__(self, interval_minutes: int = 15):
        self._interval = timedelta(minutes=interval_minutes)

    def align(self, ts: datetime) -> datetime:
        # For intervals >= 1 day, align to market close (4:00 PM ET = 16:00)
        # This allows signals to be generated after market close using that day's complete bar
        # Execution happens at the closing price (can execute in extended hours in reality)
        if self._interval >= timedelta(days=1):
            # Align to market close: 4:00 PM ET = 16:00
            # TODO: Future improvement - Generate signals 15 minutes before market close (3:45 PM ET)
            #       This would allow execution at the close price on the same day the signal is generated
            return ts.replace(hour=16, minute=0, second=0, microsecond=0)
        # For smaller intervals, align to 15-minute boundaries
        minute = (ts.minute // 15) * 15
        return ts.replace(minute=minute, second=0, microsecond=0)

    def is_decision_time(self, ts: datetime) -> bool:
        return True

    def next(self, ts: datetime) -> datetime:
        return ts + self._interval
