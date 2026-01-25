from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import Iterable, Optional, Set

import pandas as pd


class TradingCalendar(ABC):
    """
    Strategy interface for trading calendars/business calendars.
    Implementation can be exchange specific (e.g., NYSE, NSE, LSE) or country specific.

    Core Contract:
    - is_trading_day(date: date) -> bool
    - trading_days(start: date, end: date) -> pd.DatetimeIndex (inclusive)
    """
    @abstractmethod
    def is_trading_day(self, date: date) -> bool:
        raise NotImplementedError
    
    def trading_days(self, start: date, end: date) -> pd.DatetimeIndex:
        idx = pd.date_range(start=start, end=end, freq='D')
        keep = [ts for ts in idx if self.is_trading_day(ts.date())]
        return pd.DatetimeIndex(keep)
    

@dataclass(frozen=True)
class HolidayCalendarMixin:
    """
    Optional helper mixin for holiday-based calendars
    """
    holidays: Optional[Set[date]] = None

    def is_holiday(self, d: date) -> bool:
        return bool(self.holidays and d in self.holidays)