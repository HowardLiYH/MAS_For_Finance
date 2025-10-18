from __future__ import annotations
from typing import Dict, List, Any, Union
import pandas as pd

from .base import BaseAgent
from ..dto.types import ResearchSummary
from ..inventories.registry import get as registry_get

MethodEntry = Union[Any, tuple]

class ResearcherAgent(BaseAgent):
    """Implements MAS R-A/B/C/D steps. Produces ResearchSummary."""

    def __init__(self, id: str = "R1", inventory: Dict[str, List[MethodEntry]] | None = None):
        super().__init__(id=id)
        self.inventory = inventory or self._default_inventory()

    def _default_inventory(self) -> Dict[str, List[MethodEntry]]:
        return {
            "researcher.forecast": [
                registry_get("researcher.forecast","arima_x")(),
                registry_get("researcher.forecast","tft")(),
            ],
            "researcher.uncertainty": [
                registry_get("researcher.uncertainty","bootstrap_ensemble")(),
                registry_get("researcher.uncertainty","quantile_regression")(),
            ],
            "researcher.calibration": [
                registry_get("researcher.calibration","temperature_scaling")(),
                registry_get("researcher.calibration","conformal_icp")(),
            ],
        }

    def _run_entry(self, entry: MethodEntry, *args, **kwargs):
        if isinstance(entry, tuple) and len(entry) == 2:
            inst, run_kwargs = entry
            return inst.run(*args, **{**run_kwargs, **kwargs})
        return entry.run(*args, **kwargs)

    def run(self, features: pd.DataFrame, trend: pd.DataFrame) -> ResearchSummary:
        self.log("🧪 R-A Forecasting")
        forecasts = [self._run_entry(m, features, trend, horizons=["8h","24h"])
                     for m in self.inventory.get("researcher.forecast", [])]
        # simple combine (first wins if only one present)
        forecast = forecasts[0] if forecasts else {}

        self.log("🧪 R-B Uncertainty")
        for m in self.inventory.get("researcher.uncertainty", []):
            _ = self._run_entry(m, features, trend, forecast)

        self.log("🧪 R-C Calibration")
        for m in self.inventory.get("researcher.calibration", []):
            forecast = self._run_entry(m, forecast)

        # Build a summary (same shape you used before)
        prob_up = float(trend["prob_up"].iloc[-1]) if "prob_up" in trend.columns else 0.5
        market_state = "bull" if prob_up > 0.55 else ("bear" if prob_up < 0.45 else "sideways")
        rec = "BUY" if forecast.get("8h", 0.0) > 0 else ("SELL" if forecast.get("8h", 0.0) < 0 else "HOLD")
        confidence = min(0.95, 0.55 + abs(forecast.get("8h",0.0))*50.0)

        return ResearchSummary(
            market_state=market_state,
            forecast=forecast,
            signals=[],
            risk={},  # keep your earlier risk dictionary if you had one
            recommendation=rec,
            scenarios=[],
            explainability=[],
            constraints=[],
            confidence=confidence,
        )
