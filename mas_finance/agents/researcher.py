
from __future__ import annotations
from typing import Dict
import pandas as pd
from .base import BaseAgent
from ..dto.types import ResearchSummary
from ..inventories.registry import get
from ..inventories import researcher_forecast as _rf
from ..inventories import researcher_uncertainty as _ru
from ..inventories import researcher_calibration as _rc

class ResearcherAgent(BaseAgent):
    """Implements MAS R-A/B/C/D steps. Produces ResearchSummary JSON-like object."""
    def run(self, features: pd.DataFrame, trend: pd.DataFrame) -> ResearchSummary:
        self.log("🧪 R-A Forecasting → running [arima_x, tft]")
        f1 = get("researcher.forecast", "arima_x")().run(features, trend, horizons=["8h","24h"])
        f2 = get("researcher.forecast", "tft")().run(features, trend, horizons=["8h","24h"])
        # average forecasts
        forecast = {h: float(0.5*(f1[h] + f2[h])) for h in f1.keys()}
        self.log(f"🧪 R-A forecast: {forecast}")
        self.log("🧪 R-B Uncertainty → [bootstrap_ensemble, quantile_regression]")
        u1 = get("researcher.uncertainty", "bootstrap_ensemble")().run(features, trend, forecast)
        u2 = get("researcher.uncertainty", "quantile_regression")().run(features, trend, forecast)
        self.log(f"Uncertainty u1={u1}, u2={u2}")
        self.log("🧪 R-C Calibration → [temperature_scaling, conformal_icp]")
        c1 = get("researcher.calibration", "temperature_scaling")().run(forecast, T=1.5)
        c2 = get("researcher.calibration", "conformal_icp")().run(forecast, alpha=0.1)
        calib = {k: float(0.5*(c1[k] + c2[k])) for k in forecast.keys()}
        # Compose ResearchSummary
        prob_up = float(trend["prob_up"].iloc[-1])
        market_state = "bull" if prob_up>0.55 else ("bear" if prob_up<0.45 else "sideways")
        rec = "BUY" if calib["8h"]>0 else ("SELL" if calib["8h"]<0 else "HOLD")
        confidence = min(0.95, 0.55 + abs(calib["8h"])*50.0)
        rs = ResearchSummary(
            market_state=market_state,
            forecast=calib,
            signals=["ema_spread","regime_prob"],
            risk={"q05": float(u2.get("q05", -0.02)), "q95": float(u2.get("q95", 0.02))},
            recommendation=rec,
            scenarios=["breakout","mean_revert","trend_continuation"],
            explainability=[f"prob_up={prob_up:.2f}", f"calib8h={calib['8h']:.4f}"],
            constraints=["max_leverage<=5","position<=1.0"],
            confidence=float(confidence),
            post_trade_evaluation_keys=["Sharpe","PnL","HitRate"]
        )
        self.log(f"ResearchSummary: rec={rs.recommendation}, conf={rs.confidence:.2f}")
        return rs
