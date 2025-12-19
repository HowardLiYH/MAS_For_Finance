"""Analyst Agent - processes time-series price data."""
from __future__ import annotations
from typing import Dict, List, Any, Tuple, Union
import pandas as pd

from .base import BaseAgent
from ..inventory.registry import get as registry_get

MethodEntry = Union[Any, Tuple[Any, Dict[str, Any]]]


class AnalystAgent(BaseAgent):
    """
    Analyst Agent: Implements steps A-A, A-B, A-C.

    Outputs:
    - Feature DataFrame (from A-B Feature Construction)
    - Trend DataFrame (from A-C Trend Detection)
    """

    def __init__(
        self,
        id: str = "A1",
        inventory: Dict[str, List[MethodEntry]] | None = None,
    ):
        super().__init__(id=id)
        self.inventory = inventory or self._default_inventory()

    def _default_inventory(self) -> Dict[str, List[MethodEntry]]:
        return {
            "analyst.features": [
                registry_get("analyst.features", "talib_stack")(),
                registry_get("analyst.features", "stl")(),
            ],
            "analyst.trends": [
                registry_get("analyst.trends", "gaussian_hmm")(),
                registry_get("analyst.trends", "kalman_filter")(),
            ],
        }

    def _run_method(self, entry: MethodEntry, *args, **kwargs):
        if isinstance(entry, tuple) and len(entry) == 2:
            instance, run_kwargs = entry
            return instance.run(*args, **{**run_kwargs, **kwargs})
        return entry.run(*args, **kwargs)

    def run(self, price_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run the analyst pipeline.

        Args:
            price_df: DataFrame with columns [open, high, low, close, volume]

        Returns:
            Tuple of (feature_df, trend_df)
        """
        self.log("A-A: Data Alignment")
        price_df = price_df.copy().sort_index()

        self.log("A-B: Feature Construction")
        feature_parts = [
            self._run_method(m, price_df=price_df)
            for m in self.inventory.get("analyst.features", [])
        ]
        feature_df = pd.concat(feature_parts, axis=1) if feature_parts else pd.DataFrame(index=price_df.index)

        self.log("A-C: Trend Detection")
        trend_parts = [
            self._run_method(m, price_df=price_df)
            for m in self.inventory.get("analyst.trends", [])
        ]

        if trend_parts:
            trend_df = trend_parts[0].copy()
            if len(trend_parts) > 1 and "prob_up" in trend_parts[0].columns:
                # Average multiple trend detectors
                trend_df["prob_up"] = sum(tp["prob_up"] for tp in trend_parts) / len(trend_parts)
                trend_df["prob_down"] = 1 - trend_df["prob_up"]
        else:
            trend_df = pd.DataFrame(index=price_df.index)

        return feature_df, trend_df
