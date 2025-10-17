
from __future__ import annotations
import pandas as pd
import numpy as np
from .interfaces import FeatureConstructionMethod
from .registry import register

@register("analyst.feature", "talib_stack")
class TALibStack(FeatureConstructionMethod):
    """Lightweight TA-Lib-like stack implemented with pandas (EMA, RSI, ATR)."""
    name = "talib_stack"
    def run(self, price_df: pd.DataFrame, **kw) -> pd.DataFrame:
        df = price_df.copy()
        close = df["close"]
        # EMA (12, 26)
        df["ema12"] = close.ewm(span=12, adjust=False).mean()
        df["ema26"] = close.ewm(span=26, adjust=False).mean()
        # RSI (14)
        delta = close.diff()
        gain = (delta.clip(lower=0)).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / (loss + 1e-9)
        df["rsi14"] = 100 - (100 / (1 + rs))
        # ATR (14)
        high, low, prevclose = df["high"], df["low"], df["close"].shift(1)
        tr = pd.concat([(high - low).abs(), (high - prevclose).abs(), (low - prevclose).abs()], axis=1).max(axis=1)
        df["atr14"] = tr.ewm(alpha=1/14, adjust=False).mean()
        return (
            df[["ema12","ema26","rsi14","atr14"]]
            .bfill()     # replace fillna(method="bfill")
            .ffill()     # optional: also fill the other edge
            .fillna(0)   # final safety for any all-NaN columns
        )

@register("analyst.feature", "stl")
class STLDecomposition(FeatureConstructionMethod):
    """Simplified STL using moving averages (fallback if statsmodels not available)."""
    name = "stl"
    def run(self, price_df: pd.DataFrame, **kw) -> pd.DataFrame:
        df = price_df.copy()
        close = df["close"]
        trend = close.rolling(window=24, min_periods=1).mean()  # ~4 days trend at 4h bars
        seasonal = close - trend
        resid = seasonal - seasonal.rolling(window=6, min_periods=1).mean()
        out = pd.DataFrame({"stl_trend": trend, "stl_seasonal": seasonal, "stl_resid": resid})
        return out.bfill().ffill().fillna(0)
