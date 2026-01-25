from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_prices(
        symbols: list[str],
        n: int = 260,
        start: str = "2024-01-01",
        mu: float = 0.0002,
        sigma: float = 0.01,
        seed: int = 7
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=n, freq="B")
    prices = {}
    for i, symbol in enumerate(symbols):
        returns = rng.normal(loc=mu, scale=sigma, size=n)
        prices[symbol] = 100 * (1 + returns).cumprod()
    return pd.DataFrame(prices, index=dates)