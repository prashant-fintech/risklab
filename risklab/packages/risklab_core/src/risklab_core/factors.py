from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict

from risklab_core.contracts.factors import FactorConfig, FactorSpec

def compute_drawdown(returns: pd.Series) -> pd.Series:
    """Compute drawdown series from returns"""
    #convert to wealth index
    wealth_index = (1 + returns).cumprod()
    # Compute running peak
    previous_peaks = wealth_index.cummax()
    # calculate drawdown
    drawdown = (wealth_index - previous_peaks) / previous_peaks
    return drawdown.fillna(0.0)

def compute_factors(returns: pd.DataFrame, config: FactorConfig) -> pd.DataFrame:
    """
    Compute a set of risk factors for the provided assets based on configuration

    Args:
        returns: DataFrame of asset return (index=Date, col=Symbols)
        config: FactorConfig object defining the features to generate

    Returns:
        DataFrame containing all computed features.
        Structure: Index=Date, Columns=[Symbol_FeatureName, .....]
    """
    if returns.empty:
        return pd.DataFrame(index=returns.index)
    
    output_frames = {}

    for spec in config.factors:
        min_periods = spec.min_periods if spec.min_periods else spec.window
        if spec.kind == "rolling_vol":
            if not spec.window:
                raise ValueError(f"Window required for {spec.kind}")
            # Calculate rolling std dev * sqrt(252) to annualized vol if needed,
            # but keeping raw per-period volatility here for generality
            res = returns.rolling(window=spec.window, min_periods=min_periods).std()

        elif spec.kind == "rolling_mean":
            if not spec.window:
                raise ValueError(f"Window required for {spec.kind}")
            res = returns.rolling(window=spec.window, min_periods=min_periods).mean()
        elif spec.kind == "drawdown":
            res = returns.apply(compute_drawdown, axis=0)
        elif spec.kind == "rolling_corr":
            if not spec.window or not spec.benchmark:
                raise ValueError(f"Window and benchmark required for {spec.kind}")
            res = returns.rolling(window=spec.window, min_periods=min_periods).corr(returns[spec.benchmark])
        else:
            continue

        res.columns = [spec.get_output_name(col) for col in res.columns]
        output_frames[spec.name] = res

    # Concatenate all features horizontally
    # Resulting Shape: (Dates, n_assets * n_factors)
    # We sort columns to make it deterministic
    full_df = pd.concat(output_frames.values(), axis=1)
    full_df = full_df.reindex(sorted(full_df.columns), axis=1)
    return full_df