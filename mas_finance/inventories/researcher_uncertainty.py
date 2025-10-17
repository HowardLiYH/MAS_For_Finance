
from __future__ import annotations
import pandas as pd
import numpy as np
from .interfaces import UncertaintyMethod
from .registry import register

@register("researcher.uncertainty", "bootstrap_ensemble")
class Bootstrap(UncertaintyMethod):
    """Bootstrap predictive interval from recent returns."""
    name = "bootstrap_ensemble"
    def run(self, features: pd.DataFrame, trend: pd.DataFrame, forecast: dict, **kw):
        window = kw.get("window", 100)
        # approximate returns from ema12
        ret = features["ema12"].pct_change().dropna()[-window:]
        q10, q90 = np.percentile(ret, 10), np.percentile(ret, 90)
        return {"q10": float(q10), "q90": float(q90)}

@register("researcher.uncertainty", "quantile_regression")
class QuantileReg(UncertaintyMethod):
    """Quantile regression surrogate: use rolling quantiles as plug-in estimates."""
    name = "quantile_regression"
    def run(self, features: pd.DataFrame, trend: pd.DataFrame, forecast: dict, **kw):
        ret = features["ema26"].pct_change().dropna()
        q05, q95 = ret.rolling(50, min_periods=10).quantile(0.05).iloc[-1], ret.rolling(50, min_periods=10).quantile(0.95).iloc[-1]
        return {"q05": float(q05), "q95": float(q95)}
