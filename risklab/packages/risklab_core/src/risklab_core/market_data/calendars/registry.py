from __future__ import annotations

from typing import Dict
from .base import TradingCalendar


class CalendarRegistry:
    def __init__(self):
        self._calendars: Dict[str, TradingCalendar] = {}

    def register(self, name: str, calendar: TradingCalendar) -> None:
        name = name.lower().strip()
        self._calendars[name] = calendar

    def get(self, name: str) -> TradingCalendar:
        name = name.lower().strip()
        if name not in self._calendars:
            raise KeyError(f"Calendar '{name}' not found in registry. Registered: {sorted(self._calendars.keys())}")
        return self._calendars[name]

    def list_names(self) -> list[str]:
        return sorted(self._calendars.keys())