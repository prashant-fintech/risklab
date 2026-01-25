from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from .base import TradingCalendar

@dataclass(frozen=True)
class WeekdayCalendar(TradingCalendar):
    name: str = "weekday"

    def is_trading_day(self, day: date) -> bool:
        return day.weekday() < 5