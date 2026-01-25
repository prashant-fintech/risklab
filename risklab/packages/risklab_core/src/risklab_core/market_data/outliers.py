from __future__ import annotations

import pandas as pd
from risklab_core.contracts import OutlierSpec


def winsorize(df: pd.DataFrame, lower_q: float=0.01, upper_q: float=0.99) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    lo = df.quantile(lower_q)
    hi = df.quantile(upper_q)
    return df.clip(lower=lo, upper=hi, axis=1)

def clip(df: pd.DataFrame, lower: float|None, upper: float|None) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return df.clip(lower=lower, upper=upper, axis=1)

def handle_outliers(df: pd.DataFrame, spec: OutlierSpec=OutlierSpec()) -> pd.DataFrame:
    if spec.method == "none" or spec.method is None:
        return df.copy()
    if spec.method == "winsorize":
        return winsorize(df, lower_q=spec.lower_q, upper_q=spec.upper_q)
    if spec.method == "clip":
        return clip(df, lower=spec.clip_low, upper=spec.clip_high)
    raise ValueError(f"Unknown outlier method: {spec.method}")