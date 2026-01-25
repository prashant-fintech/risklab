from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Literal

from risklab_core.contracts import ReturnsSpec, ReSampleSpec, AlignSpec


def to_returns(prices: pd.DataFrame, spec: ReturnsSpec=ReturnsSpec()) -> pd.DataFrame:
    """ 
    prices: index datetime-like; columns assets; value prices
    """
    if prices.empty:
        return prices.copy()
    
    p = prices.astype(float)
    if spec.method == "simple":
        returns = p.pct_change()
    elif spec.method == "log":
        returns = np.log(p).diff()
    else:
        raise ValueError(f"Unknown returns method: {spec.method}")
    
    if spec.dropna:
        returns = returns.dropna(how="all")
    return returns
 
def resample_prices(prices: pd.DataFrame, spec: ReturnsSpec=ReSampleSpec()) -> pd.DataFrame:
    if prices.empty:
        return prices.copy()
    
    p = prices.sort_index()
    if not isinstance(p.index, pd.DatetimeIndex):
        p.index = pd.to_datetime(p.index)

    if spec.rule == "D":
        return p
    
    resampler = p.resample(spec.rule, label=spec.label, closed=spec.closed)
    if spec.how == "last":
        return resampler.last()
    if spec.how == "mean":
        return resampler.mean()
    raise ValueError(f"Unknown resample method: {spec.how}")

def align_assets(df: pd.DataFrame, spec: AlignSpec=AlignSpec()) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    
    p = df.sort_index()
    if not isinstance(p.index, pd.DatetimeIndex):
        p.index = pd.to_datetime(p.index)
    
    if spec.join == "inner":
        p = p.dropna(how="any")
    elif spec.join == "outer":
        pass
    else:
        raise ValueError(f"Unknown align method: {spec.join}")
    
    if spec.fill_method == "ffill":
        p = p.ffill()
    elif spec.fill_method == "bfill":
        p = p.bfill()
    elif spec.fill_method == "none" or spec.fill_method is None:
        pass
    else:
        raise ValueError(f"Unknown fill method: {spec.fill_method}")
    return p