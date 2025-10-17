
from __future__ import annotations
from typing import Tuple, Dict, Any, List
import pandas as pd
from .base import BaseAgent
from ..dto.types import ExecutionSummary, RiskReview
from ..inventories.registry import get
from ..inventories import risk_checks as _rc

class RiskManagerAgent(BaseAgent):
    """Implements MAS M-A/B. Aggregates checks and returns pass/soft_fail/hard_fail."""
    def run(self, exec_sum: ExecutionSummary, price_df: pd.DataFrame, *, regen_attempted: bool=False) -> RiskReview:
        context = {"price_df": price_df, "var_limit": 0.02, "max_leverage": 5.0, "max_position": 1.0}
        # Run VaR band then leverage/size checks
        self.log("🛡️ M-A Risk Analysis → [var_safe_band, leverage_position_limits]")
        checks = [
            get("risk.checks","var_safe_band")(),
            get("risk.checks","leverage_position_limits")()
        ]
        envelopes = {}; reasons: List[str] = []
        verdict = "pass"
        for chk in checks:
            out = chk.evaluate(exec_sum.__dict__, context)
            reasons += out.get("reasons", [])
            if out["verdict"] == "hard_fail":
                verdict = "hard_fail"; break
            if out["verdict"] == "soft_fail":
                verdict = "soft_fail"; envelopes.update(out.get("envelope", {}))
        review = RiskReview(verdict=verdict, reasons=reasons, envelope=envelopes, approved=(verdict=="pass"))
        self.log(f"Risk verdict: {review.verdict} | envelope={review.envelope}")
        return review
