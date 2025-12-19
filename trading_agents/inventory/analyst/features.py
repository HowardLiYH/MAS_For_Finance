"""Feature construction methods for Analyst Agent (Step A-B)."""
from __future__ import annotations
import pandas as pd
import numpy as np
from ..interfaces import FeatureMethod
from ..registry import register


@register("analyst.features", "talib_stack")
class TALibStack(FeatureMethod):
    """
    Technical analysis feature stack using pandas.

    Computes: EMA(12), EMA(26), RSI(14), ATR(14)
    """
    name = "talib_stack"

    def run(self, price_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = price_df.copy()
        close = df["close"]

        # EMA (12, 26)
        df["ema12"] = close.ewm(span=12, adjust=False).mean()
        df["ema26"] = close.ewm(span=26, adjust=False).mean()

        # RSI (14)
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / (loss + 1e-9)
        df["rsi14"] = 100 - (100 / (1 + rs))

        # ATR (14)
        high, low = df["high"], df["low"]
        prev_close = df["close"].shift(1)
        tr = pd.concat([
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        df["atr14"] = tr.ewm(alpha=1/14, adjust=False).mean()

        return (
            df[["ema12", "ema26", "rsi14", "atr14"]]
            .bfill()
            .ffill()
            .fillna(0)
        )


@register("analyst.features", "stl")
class STLDecomposition(FeatureMethod):
    """
    Simplified STL decomposition using moving averages.

    Outputs: trend, seasonal, residual components
    """
    name = "stl"

    def run(self, price_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = price_df.copy()
        close = df["close"]

        # Trend: 24-bar moving average (~4 days at 4h bars)
        trend = close.rolling(window=24, min_periods=1).mean()

        # Seasonal: deviation from trend
        seasonal = close - trend

        # Residual: deviation from seasonal pattern
        residual = seasonal - seasonal.rolling(window=6, min_periods=1).mean()

        out = pd.DataFrame({
            "stl_trend": trend,
            "stl_seasonal": seasonal,
            "stl_residual": residual
        })
        return out.bfill().ffill().fillna(0)
