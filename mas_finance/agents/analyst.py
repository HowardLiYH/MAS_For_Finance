from __future__ import annotations
from typing import Dict, List, Any, Tuple, Union
import pandas as pd

from .base import BaseAgent
from ..inventories.registry import get as registry_get

MethodEntry = Union[Any, tuple]

class AnalystAgent(BaseAgent):
    """Implements MAS A-A/A-B/A-C steps. Outputs FeatureDF and TrendDF."""

    def __init__(self, id: str = "A1", inventory: Dict[str, List[MethodEntry]] | None = None):
        super().__init__(id=id)
        self.inventory = inventory or self._default_inventory()

    def _default_inventory(self) -> Dict[str, List[MethodEntry]]:
        return {
            "analyst.feature": [
                registry_get("analyst.feature","talib_stack")(),
                registry_get("analyst.feature","stl")(),
            ],
            "analyst.trend": [
                registry_get("analyst.trend","gaussian_hmm")(),
                registry_get("analyst.trend","kalman_filter")(),
            ],
        }

    def _run_entry(self, entry: MethodEntry, *args, **kwargs):
        if isinstance(entry, tuple) and len(entry) == 2:
            inst, run_kwargs = entry
            return inst.run(*args, **{**run_kwargs, **kwargs})
        return entry.run(*args, **kwargs)

    def run(self, price_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self.log("📊 A-A Data Alignment → sorting index & ensuring UTC")
        price_df = price_df.copy().sort_index()

        self.log("📊 A-B Feature Construction → running configured feature methods")
        feat_parts = [self._run_entry(m, price_df=price_df) for m in self.inventory.get("analyst.feature", [])]
        feature_df = pd.concat(feat_parts, axis=1) if feat_parts else pd.DataFrame(index=price_df.index)

        self.log("📊 A-C Trend Detection → running configured trend methods")
        trend_parts = [self._run_entry(m, price_df=price_df) for m in self.inventory.get("analyst.trend", [])]
        trend_df = trend_parts[0] if trend_parts else pd.DataFrame(index=price_df.index)
        if len(trend_parts) > 1 and "prob_up" in trend_parts[0].columns:
            # simple average; feel free to replace by a named combine policy in config
            trend_df = trend_parts[0].copy()
            trend_df["prob_up"] = sum(tp["prob_up"] for tp in trend_parts) / len(trend_parts)
            trend_df["prob_down"] = 1 - trend_df["prob_up"]

        return feature_df, trend_df
