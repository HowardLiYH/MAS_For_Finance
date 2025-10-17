
from __future__ import annotations
from typing import Tuple
import pandas as pd
from .base import BaseAgent
from ..inventories.registry import get
from ..inventories import analyst_feature as _af  # register classes
from ..inventories import analyst_trend as _at    # register classes

class AnalystAgent(BaseAgent):
    """Implements MAS A-A/A-B/A-C steps. Outputs FeatureDF and TrendDF."""
    def run(self, price_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Step A-A: Data Alignment
        self.log("📊 A-A Data Alignment → sorting index & ensuring UTC")
        price_df = price_df.copy().sort_index()
        # Step A-B: feature construction using two methods
        self.log("📊 A-B Feature Construction → running [talib_stack, stl]")
        feat1 = get("analyst.feature","talib_stack")().run(price_df)
        feat2 = get("analyst.feature","stl")().run(price_df)
        feature_df = pd.concat([feat1, feat2], axis=1)
        self.log(f"FeatureDF columns: {list(feature_df.columns)} (shape={feature_df.shape})")
        # Step A-C: trend detection
        self.log("📊 A-C Trend Detection → running [gaussian_hmm, kalman_filter] and merging")
        tr1 = get("analyst.trend","gaussian_hmm")().run(price_df)
        tr2 = get("analyst.trend","kalman_filter")().run(price_df)
        trend_df = tr1.copy()
        trend_df["prob_up"] = 0.5*(tr1["prob_up"] + tr2["prob_up"])
        trend_df["prob_down"] = 1 - trend_df["prob_up"]
        trend_df["regime"] = ((tr1["regime"] + tr2["regime"])>0).astype(int)
        trend_df["slope"] = 0.5*(tr1["slope"] + tr2["slope"])
        self.log(f"TrendDF columns: {list(trend_df.columns)} (shape={trend_df.shape})")
        return feature_df, trend_df
