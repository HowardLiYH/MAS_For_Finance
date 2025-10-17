
from __future__ import annotations
import pandas as pd
import numpy as np
from .interfaces import ForecastingMethod
from .registry import register

@register("researcher.forecast", "arima_x")
class ARIMAX(ForecastingMethod):
    """Light ARIMA-X surrogate: uses EMA + linear drift; no heavy dependencies."""
    name = "arima_x"
    def run(self, features: pd.DataFrame, trend: pd.DataFrame, **kw):
        # Forecast next-step % change at horizons
        horizons = kw.get("horizons", ["8h","24h"])
        # Use last ema diff and trend slope as regressors
        ema_spread = (features["ema12"] - features["ema26"]).iloc[-1]
        slope = trend["slope"].iloc[-1]
        base = float(np.tanh((ema_spread / (features["ema26"].std()+1e-9)) + np.sign(slope)*0.5)) * 0.01
        return {h: base for h in horizons}

@register("researcher.forecast", "tft")
class TFT(ForecastingMethod):
    """TFT surrogate: attention-like weighting of recent changes; outputs % change per horizon."""
    name = "tft"
    def run(self, features: pd.DataFrame, trend: pd.DataFrame, **kw):
        horizons = kw.get("horizons", ["8h","24h"])
        momentum = (features["ema12"].iloc[-1] / (features["ema26"].iloc[-1] + 1e-9)) - 1
        prob_up = trend["prob_up"].iloc[-1]
        base = (0.5*momentum + 0.5*(prob_up - 0.5)) * 0.02
        return {h: base for h in horizons}
