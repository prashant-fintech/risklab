from __future__ import annotations

from pydantic import Field
from typing import List, Literal, Optional
from datetime import date
from .base import RiskLabModel, VendorMeta


class PriceRequest(RiskLabModel):
    symbols: List[str]
    start_date: date
    end_date: date
    field: Literal["open", "high", "low", "close", "volume", "adj_close"] = "adj_close"
    tz: str = "UTC"
    vendor: Optional[str] = None #optional override



class PriceResponse(RiskLabModel):
    symbols: List[str]
    field: str
    vendor: VendorMeta


class ReturnsSpec(RiskLabModel):
    method: Literal["simple", "log"] = "simple"
    dropna: bool = True


class ReSampleSpec(RiskLabModel):
    rule: Literal["D", "W", "M", "Q", "Y"] = "D"
    how: Literal["mean", "sum", "first", "last"] = "last"
    label: Literal["left", "right"] = "right"
    closed: Literal["left", "right"] = "right"


class AlignSpec(RiskLabModel):
    join: Literal["inner", "outer", "left", "right"] = "inner"
    fill_method: Literal["ffill", "bfill", "pad", "backfill", None] = None


class OutlierSpec(RiskLabModel):
    method: Literal[None, "winsorize", "clip"] = None
    lower_q: float = Field(0.01, ge=0.0, le=0.5)
    upper_q: float = Field(0.99, ge=0.5, le=1.0)
    clip_low: Optional[float] = None
    clip_high: Optional[float] = None