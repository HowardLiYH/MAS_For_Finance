"""Uncertainty quantification methods for Researcher Agent (Step R-B)."""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np
from ..interfaces import UncertaintyMethod
from ..registry import register


@register("researcher.uncertainty", "bootstrap_ensemble")
class BootstrapEnsemble(UncertaintyMethod):
    """
    Bootstrap predictive intervals from recent returns.

    Returns q10 and q90 quantiles.
    """
    name = "bootstrap_ensemble"

    def run(self, features: pd.DataFrame, trend: pd.DataFrame,
            forecast: Dict[str, float], **kwargs) -> Dict[str, float]:
        window = kwargs.get("window", 100)

        # Get recent returns from EMA
        returns = features["ema12"].pct_change().dropna()[-window:]

        if len(returns) > 0:
            q10 = float(np.percentile(returns, 10))
            q90 = float(np.percentile(returns, 90))
        else:
            q10, q90 = -0.01, 0.01

        return {"q10": q10, "q90": q90}


@register("researcher.uncertainty", "quantile_regression")
class QuantileRegression(UncertaintyMethod):
    """
    Quantile regression surrogate using rolling quantiles.

    Returns q05 and q95 quantiles.
    """
    name = "quantile_regression"

    def run(self, features: pd.DataFrame, trend: pd.DataFrame,
            forecast: Dict[str, float], **kwargs) -> Dict[str, float]:

        returns = features["ema26"].pct_change().dropna()

        if len(returns) >= 10:
            q05_series = returns.rolling(50, min_periods=10).quantile(0.05)
            q95_series = returns.rolling(50, min_periods=10).quantile(0.95)
            q05 = float(q05_series.iloc[-1])
            q95 = float(q95_series.iloc[-1])
        else:
            q05, q95 = -0.02, 0.02

        return {"q05": q05, "q95": q95}
