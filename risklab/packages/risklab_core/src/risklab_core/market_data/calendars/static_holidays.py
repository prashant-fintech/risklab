from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, Optional, Set

from .base import TradingCalendar, HolidayCalendarMixin


@dataclass(frozen=True)
class StaticHolidayCalendar(TradingCalendar, HolidayCalendarMixin):
    """
        Trading days = Mon-Fri excluding explicit holiday set.
        Use for country/market calendars where you store holidays in YAML/JSON
    """
    name: str = "static"
    holidays: Optional[Set[date]] = None

    def is_trading_day(self, d) -> bool:
        if d.weekday() >= 5:
            return False
        if self.is_holiday(d):
            return False
        return True
    
    @classmethod
    def from_holidays(cls, name: str, holidays: Iterable[date]) -> StaticHolidayCalendar:
        return cls(name=name, holidays=set(holidays))