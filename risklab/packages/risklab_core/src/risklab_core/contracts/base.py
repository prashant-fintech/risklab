from __future__ import annotations

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Dict, Optional


class RiskLabModel(BaseModel):
    """Base model for all RiskLab models."""
    model_config = {"extra": "forbid"}

class RunMeta(RiskLabModel):
    run_id: str
    created_at: datetime = Field(default_factory=datetime.now)
    tags: Dict[str, str] = Field(default_factory=dict)

class VendorMeta(RiskLabModel):
    vendor: str
    dataset: str
    request_id: Optional[str] = None