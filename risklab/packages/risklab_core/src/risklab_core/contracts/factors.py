from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import Field
from .base import RiskLabModel

class FactorSpec(RiskLabModel):
    """
    Specification for a single factor computation
    Examples:
        - kind="rolling_vol", windows=21, name="vol_1m"
        - kind="drawdown", name="dd_current"
    """
    name: str = Field(..., description="Unique name for this feature column")
    kind: Literal["rolling_mean", "rolling_vol", "rolling_corr", "drawdown"]
    window: Optional[int] = Field(None, description="Lookback window (required for rolling stats)")
    benchmark: Optional[str] = Field(None, description="Benchmark symbol reuqired for correlation calculations")
    min_periods: Optional[int] = Field(None, description="Minimum number of observations in window required to have a value (for rolling stats)")

    def get_output_name(self, symbol: str) -> str:
        """Return the stable feature name e.g. 'AAPL_vol_1m'."""
        return f"{symbol}_{self.name}"
    

class FactorConfig(RiskLabModel):
    """
    Configuration for a factor computation run
    """
    factors: List[FactorSpec]
    defaults: Optional[dict] = Field(default_factory=dict, description="Global defaults")
    