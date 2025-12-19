"""Calibration methods for Researcher Agent (Step R-C)."""
from __future__ import annotations
from typing import Dict
from ..interfaces import CalibrationMethod
from ..registry import register


@register("researcher.calibration", "temperature_scaling")
class TemperatureScaling(CalibrationMethod):
    """
    Temperature scaling for forecast calibration.

    Scales forecast magnitudes by temperature parameter T.
    """
    name = "temperature_scaling"

    def run(self, forecast: Dict[str, float], **kwargs) -> Dict[str, float]:
        temperature = float(kwargs.get("T", 1.5))
        return {k: float(v) / temperature for k, v in forecast.items()}


@register("researcher.calibration", "conformal_icp")
class ConformalICP(CalibrationMethod):
    """
    Inductive Conformal Prediction (ICP) surrogate.

    Shrinks forecast magnitudes by (1 - alpha) factor.
    """
    name = "conformal_icp"

    def run(self, forecast: Dict[str, float], **kwargs) -> Dict[str, float]:
        alpha = float(kwargs.get("alpha", 0.1))
        return {k: float(v) * (1.0 - alpha) for k, v in forecast.items()}
