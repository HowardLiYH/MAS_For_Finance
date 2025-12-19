"""Trend detection methods for Analyst Agent (Step A-C)."""
from __future__ import annotations
import pandas as pd
import numpy as np
from ..interfaces import TrendMethod
from ..registry import register


@register("analyst.trends", "gaussian_hmm")
class GaussianHMM(TrendMethod):
    """
    Gaussian HMM regime detection.

    Uses slope sign and volatility clustering to proxy 2-state regime.
    Outputs: prob_up, prob_down, regime, slope
    """
    name = "gaussian_hmm"

    def run(self, price_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = price_df.copy()
        returns = df["close"].pct_change().fillna(0.0)
        volatility = returns.rolling(12, min_periods=1).std().fillna(0)

        # 2-state proxy: 0=low-vol bull, 1=high-vol bear
        mean_return = returns.rolling(6, min_periods=1).mean()
        regime = ((mean_return < 0) | (volatility > volatility.median())).astype(int)

        prob_up = (1 - regime) * 0.65 + regime * 0.35
        prob_down = 1 - prob_up
        slope = df["close"].diff().fillna(0)

        return pd.DataFrame({
            "prob_up": prob_up,
            "prob_down": prob_down,
            "regime": regime,
            "slope": slope
        })


@register("analyst.trends", "kalman_filter")
class KalmanFilter(TrendMethod):
    """
    Simple 1D Kalman filter for trend extraction.

    Smooths price and converts slope to directional probabilities.
    """
    name = "kalman_filter"

    def run(self, price_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = price_df.copy()
        observations = df["close"].to_numpy()

        # Kalman filter parameters
        x = observations[0]  # Initial state
        P = 1.0              # Initial covariance
        Q = 0.001            # Process noise
        R = 0.1              # Measurement noise

        smoothed = []
        for obs in observations:
            # Predict
            x_pred = x
            P_pred = P + Q
            # Update
            K = P_pred / (P_pred + R)
            x = x_pred + K * (obs - x_pred)
            P = (1 - K) * P_pred
            smoothed.append(x)

        smooth_series = pd.Series(smoothed, index=df.index)
        slope = smooth_series.diff().fillna(0)

        # Convert slope to probability
        prob_up = (1 / (1 + np.exp(-slope / (smooth_series.std() + 1e-9)))).clip(0, 1)

        return pd.DataFrame({
            "prob_up": prob_up,
            "prob_down": 1 - prob_up,
            "regime": (prob_up < 0.5).astype(int),
            "slope": slope
        })
