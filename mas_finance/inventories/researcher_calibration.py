
from __future__ import annotations
import numpy as np
from .interfaces import CalibrationMethod
from .registry import register

@register("researcher.calibration", "temperature_scaling")
class TemperatureScaling(CalibrationMethod):
    """Temperature scaling surrogate for probabilities in forecast dict (if present)."""
    name = "temperature_scaling"
    def run(self, forecast: dict, **kw):
        T = float(kw.get("T", 1.5))
        return {k: float(v)/T for k,v in forecast.items()}

@register("researcher.calibration", "conformal_icp")
class ConformalICP(CalibrationMethod):
    """Conformal ICP surrogate: widen forecast magnitudes by an alpha quantile factor."""
    name = "conformal_icp"
    def run(self, forecast: dict, **kw):
        alpha = float(kw.get("alpha", 0.1))
        return {k: float(v)*(1.0 - alpha) for k,v in forecast.items()}
