"""Forecasting methods for Researcher Agent (Step R-A)."""
from __future__ import annotations
from typing import Dict, List
import pandas as pd
import numpy as np
from ..interfaces import ForecastMethod
from ..registry import register


@register("researcher.forecasting", "arima_x")
class ARIMAX(ForecastMethod):
    """
    ARIMA-X surrogate using EMA spread and trend slope.

    Returns predicted % change for each horizon.
    """
    name = "arima_x"

    def run(self, features: pd.DataFrame, trend: pd.DataFrame, **kwargs) -> Dict[str, float]:
        horizons: List[str] = kwargs.get("horizons", ["8h", "24h"])

        # Use EMA spread and slope as regressors
        ema_spread = (features["ema12"] - features["ema26"]).iloc[-1]
        slope = trend["slope"].iloc[-1]

        # Normalize and combine signals
        ema_std = features["ema26"].std() + 1e-9
        signal = np.tanh((ema_spread / ema_std) + np.sign(slope) * 0.5)
        base_forecast = float(signal * 0.01)  # Scale to ~1% max

        return {h: base_forecast for h in horizons}


@register("researcher.forecasting", "tft")
class TFT(ForecastMethod):
    """
    Temporal Fusion Transformer surrogate.

    Uses attention-like weighting of momentum and trend probability.
    """
    name = "tft"

    def run(self, features: pd.DataFrame, trend: pd.DataFrame, **kwargs) -> Dict[str, float]:
        horizons: List[str] = kwargs.get("horizons", ["8h", "24h"])

        # Momentum signal
        momentum = (features["ema12"].iloc[-1] / (features["ema26"].iloc[-1] + 1e-9)) - 1

        # Trend probability signal
        prob_up = trend["prob_up"].iloc[-1]

        # Weighted combination (attention-like)
        base_forecast = float((0.5 * momentum + 0.5 * (prob_up - 0.5)) * 0.02)

        return {h: base_forecast for h in horizons}
