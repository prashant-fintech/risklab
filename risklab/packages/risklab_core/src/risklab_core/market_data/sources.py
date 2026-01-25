from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Protocol
import pandas as pd


from risklab_core.contracts import PriceRequest, VendorMeta

class MarketDataSource(ABC):
    """Strategu interface
        Implementations: YahooFinanceSource, AlphaVantageSource, CSVSource, DBSource, etc
    """
    name: str

    @abstractmethod
    def get_prices(self, request: PriceRequest) -> pd.DataFrame:
        """
        Return a DataFrame indexed by Timestamp/date with columns = symbols.
        Values are prices for req.field

        Required:
        - index must be datetime-like
        - columns match req.symbols order or superset (caller with subset)
        """
        raise NotImplementedError
    

@dataclass(frozen=True)
class MarketDataSourceResult:
    name: str
    vendor: VendorMeta


class SourceRegistry:
    _sources: Dict[str, MarketDataSource] = {}


    
    def register(self, source: MarketDataSource) -> None:
        key = source.name.lower().strip()
        self._sources[key] = source

    
    def get(self, name: str) -> MarketDataSource:
        key = name.lower().strip()
        if key not in self._sources:
            raise KeyError(f"Unknown data source: {name}. Registered: {sorted(self._sources.keys())}")
        return self._sources[key]
    

class InMemorySource(MarketDataSource):

    """Useful for testing/demos Provide a dict symbol -> Series/DataFrame"""
    name = "memory"

    def __init__(self, data: Dict[str, pd.Series | pd.DataFrame]) -> None:
        self._data = data

    def get_prices(self, request: PriceRequest) -> pd.DataFrame:
        frames = []
        for symbol in request.symbols:
            obj = self._data.get(symbol)
            if obj is None:
                raise KeyError(f"Symbol {symbol} not found in InMemorySource: {symbol}")
            if isinstance(obj, pd.DataFrame):
                if request.field in obj.columns:
                    s = obj[request.field]
                else:
                    s = obj.iloc[:, 0]
            else:
                s = obj
            s = s.rename(symbol)
            frames.append(s)
        df = pd.concat(frames, axis=1).sort_index()
        df = df.loc[pd.to_datetime(request.start_date):pd.to_datetime(request.end_date)]
        return df
    

class CSVSource(MarketDataSource):
    """
        CSV Format Expectation:
        - index column: date/time
        - columns: symbols
    """

    name = "csv"

    def __init__(self, path: str, index_col: str = "date") -> None:
        self._path = path
        self._index_col = index_col

    def get_prices(self, request: PriceRequest) -> pd.DataFrame:
        df = pd.read_csv(self._path)
        if self._index_col not in df.columns:
            raise ValueError(f"Index column {self._index_col} not found in CSV file: {self._path}")
        df[self._index_col] = pd.to_datetime(df[self._index_col])
        df = df.set_index(self._index_col).sort_index()
        df = df[request.symbols]
        df = df.loc[pd.to_datetime(request.start_date):pd.to_datetime(request.end_date)]
        return df


class YahooFinanceSource(MarketDataSource):
    """
    Yahoo Finance strategy via yfinance.

    Notes:
    - yfinance download behavior has changed across versions (multi-index columns, auto_adjust defaults).
    - We set auto_adjust=False so that 'Adj Close' exists and is consistent.
    - Output: DataFrame indexed by DatetimeIndex, columns = req.symbols (subset/ordered).
    """
    name = "yahoo"

    _FIELD_MAP = {
        "adj_close": "Adj Close",
        "close": "Close",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "volume": "Volume",
    }

    def __init__(
        self,
        *,
        auto_adjust: bool = False,
        actions: bool = False,
        threads: bool = True,
        progress: bool = False,
    ) -> None:
        self._auto_adjust = auto_adjust
        self._actions = actions
        self._threads = threads
        self._progress = progress

    def get_prices(self, req: PriceRequest) -> pd.DataFrame:
        try:
            import yfinance as yf
        except Exception as e:
            raise RuntimeError(
                "yfinance is not installed. Install with: uv add yfinance"
            ) from e

        field_label = self._FIELD_MAP.get(req.field)
        if not field_label:
            raise ValueError(f"Unsupported field for YahooFinanceSource: {req.field}")

        tickers = req.symbols if len(req.symbols) > 1 else req.symbols[0]

        df = yf.download(
            tickers=tickers,
            start=str(req.start_date),
            end=str(req.end_date),
            interval="1d",
            group_by="column",
            auto_adjust=self._auto_adjust,
            actions=self._actions,
            threads=self._threads,
            progress=self._progress,
        )

        if df is None or df.empty:
            return pd.DataFrame(index=pd.DatetimeIndex([]), columns=req.symbols)

        df = df.sort_index()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        out = self._extract_field(df, field_label, req.symbols)

        # Ensure requested ordering + subset
        out = out.reindex(columns=req.symbols)

        # Drop rows where all missing (optional; caller can decide)
        out = out.dropna(how="all")

        return out

    @classmethod
    def _extract_field(cls, df: pd.DataFrame, field_label: str, symbols: list[str]) -> pd.DataFrame:
        """
        Normalize yfinance outputs (single-level or MultiIndex columns) to:
        index = dates, columns = tickers, values = requested field.
        """
        # Case A: MultiIndex columns (common for multiple tickers; sometimes even single ticker)
        if isinstance(df.columns, pd.MultiIndex):
            lvl0 = df.columns.get_level_values(0)
            lvl1 = df.columns.get_level_values(1)

            # Common format: (Field, Ticker)
            if field_label in set(lvl0):
                x = df[field_label]
                if isinstance(x, pd.Series):
                    # single ticker edge case
                    x = x.to_frame(symbols[0])
                return cls._ensure_df(x)

            # Alternative format: (Ticker, Field)
            if field_label in set(lvl1):
                x = df.xs(field_label, axis=1, level=1, drop_level=True)
                return cls._ensure_df(x)

            # Fallback: try either level contains field-like names
            # (rare but protects against ordering changes)
            candidates0 = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
            if any(v in candidates0 for v in set(lvl0)):
                # assume level0 is field
                if field_label in set(lvl0):
                    x = df[field_label]
                    return cls._ensure_df(x)

            if any(v in candidates0 for v in set(lvl1)):
                # assume level1 is field
                x = df.xs(field_label, axis=1, level=1, drop_level=True)
                return cls._ensure_df(x)

            raise ValueError(
                f"Could not locate field '{field_label}' in MultiIndex columns. "
                f"Levels sample: level0={list(pd.unique(lvl0))[:8]}, level1={list(pd.unique(lvl1))[:8]}"
            )

        # Case B: Single-level columns (common for single ticker)
        if field_label not in df.columns:
            # If auto_adjust=True, Adj Close may not exist (Close becomes adjusted)
            if field_label == "Adj Close" and "Close" in df.columns:
                x = df["Close"]
            else:
                raise ValueError(f"Field '{field_label}' not found in columns: {list(df.columns)}")
        else:
            x = df[field_label]

        if isinstance(x, pd.Series):
            # single ticker
            return x.to_frame(symbols[0])

        # If somehow multiple tickers came back single-level (unlikely), keep as is
        return cls._ensure_df(x)

    @staticmethod
    def _ensure_df(x: pd.DataFrame | pd.Series) -> pd.DataFrame:
        if isinstance(x, pd.Series):
            return x.to_frame()
        return x