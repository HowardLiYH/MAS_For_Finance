"""Multi-asset data loader for Bybit CSV files.

Loads and aligns price data for multiple crypto assets (BTC, ETH, SOL, DOGE, XRP)
with rich derivative market features from Bybit perpetual futures.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone

import pandas as pd
import numpy as np

# Default symbols supported
DEFAULT_SYMBOLS = ["BTC", "ETH", "SOL", "DOGE", "XRP"]

# Column mapping from Bybit CSV format
BYBIT_COLUMNS = [
    "timestamp_utc",  # Datetime index
    "open", "high", "low", "close", "volume", "turnover",
    "oi_btc", "long_ratio", "short_ratio", "long_short_ratio",
    "funding_rate",
    "mark_open", "mark_high", "mark_low", "mark_close",
    "index_open", "index_high", "index_low", "index_close",
    "basis", "premium_pct", "oi_usd",
]

# Essential columns that must be present
REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]


def load_bybit_csv(
    symbol: str,
    csv_path: Path,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Load a single Bybit CSV file and return a properly indexed DataFrame.

    Args:
        symbol: Asset symbol (e.g., "BTC")
        csv_path: Path to the CSV file
        start_date: Optional start date filter
        end_date: Optional end date filter

    Returns:
        DataFrame indexed by timestamp with all available columns
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Handle timestamp column
    if "timestamp_utc" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    else:
        raise ValueError(f"No timestamp column found in {csv_path}")

    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)

    # Drop duplicate timestamps if any
    df = df[~df.index.duplicated(keep="last")]

    # Filter by date range if specified
    if start_date:
        start_dt = pd.Timestamp(start_date, tz="UTC")
        df = df[df.index >= start_dt]
    if end_date:
        end_dt = pd.Timestamp(end_date, tz="UTC")
        df = df[df.index <= end_dt]

    # Add symbol column for reference
    df["symbol"] = symbol

    # Ensure required columns exist
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing in {csv_path}")

    return df


def load_all_assets(
    csv_dir: Path,
    symbols: Optional[List[str]] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    file_pattern: str = "Bybit_{symbol}.csv",
) -> Dict[str, pd.DataFrame]:
    """
    Load all asset CSVs from a directory.

    Args:
        csv_dir: Directory containing Bybit CSV files
        symbols: List of symbols to load (default: DEFAULT_SYMBOLS)
        start_date: Optional start date filter
        end_date: Optional end date filter
        file_pattern: Pattern for CSV filenames (use {symbol} placeholder)

    Returns:
        Dict mapping symbol -> DataFrame
    """
    csv_dir = Path(csv_dir)
    if not csv_dir.exists():
        raise FileNotFoundError(f"CSV directory not found: {csv_dir}")

    symbols = symbols or DEFAULT_SYMBOLS
    assets: Dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        filename = file_pattern.format(symbol=symbol)
        csv_path = csv_dir / filename

        try:
            df = load_bybit_csv(symbol, csv_path, start_date, end_date)
            assets[symbol] = df
            print(f"  ✓ Loaded {symbol}: {len(df)} bars")
        except FileNotFoundError:
            print(f"  ⚠ {symbol} CSV not found: {csv_path}")
        except Exception as e:
            print(f"  ✗ Error loading {symbol}: {e}")

    if not assets:
        raise RuntimeError("No asset data loaded")

    return assets


def align_timestamps(
    assets: Dict[str, pd.DataFrame],
    method: str = "inner",
) -> Tuple[Dict[str, pd.DataFrame], pd.DatetimeIndex]:
    """
    Align all asset DataFrames to a common timestamp index.

    Args:
        assets: Dict of symbol -> DataFrame
        method: Alignment method ("inner" = intersection, "outer" = union)

    Returns:
        Tuple of (aligned_assets dict, common_index)
    """
    if not assets:
        raise ValueError("No assets to align")

    # Get all indices
    indices = [df.index for df in assets.values()]

    if method == "inner":
        # Intersection of all timestamps
        common_index = indices[0]
        for idx in indices[1:]:
            common_index = common_index.intersection(idx)
    else:
        # Union of all timestamps
        common_index = indices[0]
        for idx in indices[1:]:
            common_index = common_index.union(idx)

    common_index = common_index.sort_values()

    # Reindex all DataFrames
    aligned: Dict[str, pd.DataFrame] = {}
    for symbol, df in assets.items():
        aligned_df = df.reindex(common_index)

        # Forward-fill then back-fill for small gaps
        aligned_df = aligned_df.ffill(limit=2).bfill(limit=1)

        aligned[symbol] = aligned_df

    print(f"  Aligned to {len(common_index)} common timestamps")

    return aligned, common_index


def get_price_matrix(
    assets: Dict[str, pd.DataFrame],
    column: str = "close",
) -> pd.DataFrame:
    """
    Create a price matrix with symbols as columns.

    Args:
        assets: Dict of symbol -> DataFrame
        column: Price column to extract (default: "close")

    Returns:
        DataFrame with symbols as columns, timestamp as index
    """
    prices = {}
    for symbol, df in assets.items():
        if column in df.columns:
            prices[symbol] = df[column]

    return pd.DataFrame(prices)


def get_returns_matrix(
    assets: Dict[str, pd.DataFrame],
    column: str = "close",
    log_returns: bool = True,
) -> pd.DataFrame:
    """
    Create a returns matrix with symbols as columns.

    Args:
        assets: Dict of symbol -> DataFrame
        column: Price column to use
        log_returns: Use log returns (True) or simple returns (False)

    Returns:
        DataFrame with returns, symbols as columns
    """
    prices = get_price_matrix(assets, column)

    if log_returns:
        returns = np.log(prices).diff()
    else:
        returns = prices.pct_change()

    return returns


def get_oi_matrix(assets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Get Open Interest matrix (USD values)."""
    oi = {}
    for symbol, df in assets.items():
        if "oi_usd" in df.columns:
            oi[symbol] = df["oi_usd"]
        elif "oi_btc" in df.columns and "close" in df.columns:
            # Approximate USD OI
            oi[symbol] = df["oi_btc"] * df["close"]

    return pd.DataFrame(oi)


def get_funding_matrix(assets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Get funding rate matrix."""
    funding = {}
    for symbol, df in assets.items():
        if "funding_rate" in df.columns:
            funding[symbol] = df["funding_rate"]

    return pd.DataFrame(funding)


def get_long_short_matrix(assets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Get long/short ratio matrix."""
    ls_ratio = {}
    for symbol, df in assets.items():
        if "long_short_ratio" in df.columns:
            ls_ratio[symbol] = df["long_short_ratio"]

    return pd.DataFrame(ls_ratio)


class MultiAssetLoader:
    """
    Convenience class for loading and managing multi-asset data.

    Usage:
        loader = MultiAssetLoader(csv_dir="/path/to/csvs")
        loader.load()
        btc_df = loader.get("BTC")
        returns = loader.returns_matrix()
    """

    def __init__(
        self,
        csv_dir: Path | str,
        symbols: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        align: bool = True,
        align_method: str = "inner",
    ):
        self.csv_dir = Path(csv_dir)
        self.symbols = symbols or DEFAULT_SYMBOLS
        self.start_date = start_date
        self.end_date = end_date
        self.align = align
        self.align_method = align_method

        self._assets: Dict[str, pd.DataFrame] = {}
        self._common_index: Optional[pd.DatetimeIndex] = None
        self._loaded = False

    def load(self) -> "MultiAssetLoader":
        """Load all assets from CSV files."""
        print(f"📊 Loading {len(self.symbols)} assets from {self.csv_dir}")

        self._assets = load_all_assets(
            self.csv_dir,
            self.symbols,
            self.start_date,
            self.end_date,
        )

        if self.align and len(self._assets) > 1:
            self._assets, self._common_index = align_timestamps(
                self._assets, self.align_method
            )
        elif self._assets:
            # Single asset or no alignment
            first_df = next(iter(self._assets.values()))
            self._common_index = first_df.index

        self._loaded = True
        return self

    def get(self, symbol: str) -> pd.DataFrame:
        """Get DataFrame for a specific symbol."""
        if not self._loaded:
            self.load()
        if symbol not in self._assets:
            raise KeyError(f"Symbol '{symbol}' not loaded")
        return self._assets[symbol]

    def all_assets(self) -> Dict[str, pd.DataFrame]:
        """Get all loaded asset DataFrames."""
        if not self._loaded:
            self.load()
        return self._assets

    @property
    def index(self) -> pd.DatetimeIndex:
        """Get common timestamp index."""
        if not self._loaded:
            self.load()
        return self._common_index

    def price_matrix(self, column: str = "close") -> pd.DataFrame:
        """Get price matrix."""
        return get_price_matrix(self._assets, column)

    def returns_matrix(self, log_returns: bool = True) -> pd.DataFrame:
        """Get returns matrix."""
        return get_returns_matrix(self._assets, log_returns=log_returns)

    def oi_matrix(self) -> pd.DataFrame:
        """Get open interest matrix."""
        return get_oi_matrix(self._assets)

    def funding_matrix(self) -> pd.DataFrame:
        """Get funding rate matrix."""
        return get_funding_matrix(self._assets)

    def long_short_matrix(self) -> pd.DataFrame:
        """Get long/short ratio matrix."""
        return get_long_short_matrix(self._assets)

    def summary(self) -> Dict[str, Dict]:
        """Get summary statistics for all assets."""
        if not self._loaded:
            self.load()

        summary = {}
        for symbol, df in self._assets.items():
            summary[symbol] = {
                "rows": len(df),
                "start": df.index.min().isoformat() if len(df) > 0 else None,
                "end": df.index.max().isoformat() if len(df) > 0 else None,
                "columns": list(df.columns),
                "missing_pct": (df.isna().sum().sum() / df.size * 100) if df.size > 0 else 0,
            }
        return summary
