
from __future__ import annotations
import pandas as pd
import numpy as np
from .interfaces import TrendDetectionMethod
from .registry import register

@register("analyst.trend", "gaussian_hmm")
class GaussianHMM(TrendDetectionMethod):
    """Placeholder HMM regime detection using slope sign & volatility clustering (no hmmlearn dependency)."""
    name = "gaussian_hmm"
    def run(self, price_df: pd.DataFrame, **kw) -> pd.DataFrame:
        df = price_df.copy()
        ret = df["close"].pct_change().fillna(0.0)
        vol = ret.rolling(12, min_periods=1).std().fillna(0)
        # 2-state proxy: 0=low-vol bull, 1=high-vol bear
        regime = ((ret.rolling(6, min_periods=1).mean() < 0) | (vol > vol.median())).astype(int)
        prob_up = (1 - regime) * 0.65 + regime * 0.35
        prob_down = 1 - prob_up
        slope = df["close"].diff().fillna(0)
        return pd.DataFrame({"prob_up": prob_up, "prob_down": prob_down, "regime": regime, "slope": slope})

@register("analyst.trend", "kalman_filter")
class KalmanFilter(TrendDetectionMethod):
    """Simple 1D Kalman smoother for price; converts slope to directional probabilities."""
    name = "kalman_filter"
    def run(self, price_df: pd.DataFrame, **kw) -> pd.DataFrame:
        df = price_df.copy()
        z = df["close"].to_numpy()
        # scalar Kalman (constant velocity model omitted for simplicity)
        x, P = z[0], 1.0
        Q, R = 0.001, 0.1
        xs = []
        for obs in z:
            # predict
            x_pred, P_pred = x, P + Q
            # update
            K = P_pred / (P_pred + R)
            x = x_pred + K * (obs - x_pred)
            P = (1 - K) * P_pred
            xs.append(x)
        smooth = pd.Series(xs, index=df.index)
        slope = smooth.diff().fillna(0)
        prob_up = (1 / (1 + np.exp(-slope / (smooth.std() + 1e-9)))).clip(0,1)
        return pd.DataFrame({"prob_up": prob_up, "prob_down": 1 - prob_up, "regime": (prob_up<0.5).astype(int), "slope": slope})
