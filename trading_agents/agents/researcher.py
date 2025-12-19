"""Researcher Agent - generates forecasts and signals."""
from __future__ import annotations
from typing import Dict, List, Any, Union, Tuple
import pandas as pd

from .base import BaseAgent
from ..models import ResearchSummary
from ..inventory.registry import get as registry_get

MethodEntry = Union[Any, Tuple[Any, Dict[str, Any]]]


class ResearcherAgent(BaseAgent):
    """
    Researcher Agent: Implements steps R-A, R-B, R-C, R-D.

    Outputs:
    - ResearchSummary with forecasts, signals, and recommendations
    """

    def __init__(
        self,
        id: str = "R1",
        inventory: Dict[str, List[MethodEntry]] | None = None,
    ):
        super().__init__(id=id)
        self.inventory = inventory or self._default_inventory()

    def _default_inventory(self) -> Dict[str, List[MethodEntry]]:
        return {
            "researcher.forecasting": [
                registry_get("researcher.forecasting", "arima_x")(),
                registry_get("researcher.forecasting", "tft")(),
            ],
            "researcher.uncertainty": [
                registry_get("researcher.uncertainty", "bootstrap_ensemble")(),
                registry_get("researcher.uncertainty", "quantile_regression")(),
            ],
            "researcher.calibration": [
                registry_get("researcher.calibration", "temperature_scaling")(),
                registry_get("researcher.calibration", "conformal_icp")(),
            ],
        }

    def _run_method(self, entry: MethodEntry, *args, **kwargs):
        if isinstance(entry, tuple) and len(entry) == 2:
            instance, run_kwargs = entry
            return instance.run(*args, **{**run_kwargs, **kwargs})
        return entry.run(*args, **kwargs)

    def run(self, features: pd.DataFrame, trend: pd.DataFrame) -> ResearchSummary:
        """
        Run the research pipeline.

        Args:
            features: Feature DataFrame from Analyst
            trend: Trend DataFrame from Analyst

        Returns:
            ResearchSummary with forecasts and recommendations
        """
        # R-A: Forecasting
        self.log("R-A: Forecasting")
        forecasts = [
            self._run_method(m, features, trend, horizons=["8h", "24h"])
            for m in self.inventory.get("researcher.forecasting", [])
        ]

        # Combine forecasts (average if multiple)
        if forecasts:
            if len(forecasts) > 1:
                all_horizons = set()
                for f in forecasts:
                    all_horizons.update(f.keys())
                forecast = {
                    h: sum(f.get(h, 0.0) for f in forecasts) / len(forecasts)
                    for h in all_horizons
                }
            else:
                forecast = forecasts[0]
        else:
            forecast = {}

        # R-B: Uncertainty Quantification
        self.log("R-B: Uncertainty Quantification")
        uncertainty = {}
        for m in self.inventory.get("researcher.uncertainty", []):
            result = self._run_method(m, features, trend, forecast)
            if isinstance(result, dict):
                uncertainty.update(result)

        # R-C: Probability Calibration
        self.log("R-C: Probability Calibration")
        for m in self.inventory.get("researcher.calibration", []):
            calibrated = self._run_method(m, forecast)
            if isinstance(calibrated, dict):
                forecast = calibrated

        # R-D: Signal Packaging
        self.log("R-D: Signal Packaging")
        return self._package_signals(features, trend, forecast, uncertainty)

    def _package_signals(
        self,
        features: pd.DataFrame,
        trend: pd.DataFrame,
        forecast: Dict[str, float],
        uncertainty: Dict[str, float],
    ) -> ResearchSummary:
        """Package all outputs into ResearchSummary."""
        # Extract market state
        prob_up = float(trend["prob_up"].iloc[-1]) if "prob_up" in trend.columns else 0.5
        market_state = "bull" if prob_up > 0.55 else ("bear" if prob_up < 0.45 else "sideways")

        # Calculate confidence
        forecast_8h = forecast.get("8h", 0.0)
        forecast_24h = forecast.get("24h", 0.0)
        confidence = min(0.95, 0.55 + abs(forecast_8h) * 50.0)

        # Adjust confidence for uncertainty
        if uncertainty:
            q_range = abs(uncertainty.get("q95", 0.0) - uncertainty.get("q05", 0.0))
            if q_range > 0.05:
                confidence *= 0.8

        # Generate recommendation
        if forecast_8h > 0.005:
            recommendation = "BUY"
        elif forecast_8h < -0.005:
            recommendation = "SELL"
        else:
            recommendation = "HOLD"

        # Generate signals
        signals = []
        if abs(forecast_8h) > 0.01:
            signals.append(f"Strong {recommendation}: {forecast_8h*100:.2f}% expected in 8h")
        if abs(forecast_24h) > 0.02:
            signals.append(f"Medium-term signal: {forecast_24h*100:.2f}% in 24h")
        if prob_up > 0.65 or prob_up < 0.35:
            signals.append(f"Trend: {market_state} (prob={prob_up:.2f})")

        # Generate scenarios
        scenarios = []
        if forecast_8h > 0:
            scenarios.append(f"Bullish: +{forecast_8h*100:.2f}% (8h), +{forecast_24h*100:.2f}% (24h)")
        elif forecast_8h < 0:
            scenarios.append(f"Bearish: {forecast_8h*100:.2f}% (8h), {forecast_24h*100:.2f}% (24h)")
        else:
            scenarios.append("Neutral: Limited movement expected")

        if uncertainty:
            scenarios.append(f"Range: {uncertainty.get('q05', 0)*100:.2f}% to {uncertainty.get('q95', 0)*100:.2f}%")

        # Generate explainability
        explainability = [
            f"Market: {market_state} (prob_up={prob_up:.2f})",
            f"Confidence: {confidence:.2f}",
            f"Forecast: 8h={forecast_8h*100:.2f}%, 24h={forecast_24h*100:.2f}%",
        ]

        # Generate constraints
        constraints = []
        if uncertainty:
            q_range = abs(uncertainty.get("q95", 0.0) - uncertainty.get("q05", 0.0))
            if q_range > 0.1:
                constraints.append("High uncertainty: Reduce position size")
        if confidence < 0.6:
            constraints.append("Low confidence: Use conservative leverage")

        # Build meta
        meta = {
            "researcher_id": self.id,
            "methods": {
                pool: [m.name if hasattr(m, 'name') else str(type(m).__name__) for m in methods]
                for pool, methods in self.inventory.items()
            },
            "timestamp": pd.Timestamp.now().isoformat(),
        }

        return ResearchSummary(
            meta=meta,
            market_state=market_state,
            forecast=forecast,
            signals=signals,
            risk={"confidence": confidence, **uncertainty},
            recommendation=recommendation,
            scenarios=scenarios,
            explainability=explainability,
            constraints=constraints,
            confidence=confidence,
        )
