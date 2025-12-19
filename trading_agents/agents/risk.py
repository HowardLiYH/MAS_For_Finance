"""Risk Manager Agent - validates execution orders."""
from __future__ import annotations
from typing import Dict, List, Any, Union, Tuple
import pandas as pd

from .base import BaseAgent
from ..models import ExecutionSummary, RiskReview
from ..inventory.registry import get as registry_get

MethodEntry = Union[Any, Tuple[Any, Dict[str, Any]]]


class RiskAgent(BaseAgent):
    """
    Risk Manager Agent: Implements step M-A.

    Outputs:
    - RiskReview with verdict (pass/soft_fail/hard_fail)
    """

    def __init__(
        self,
        id: str = "M1",
        inventory: Dict[str, List[MethodEntry]] | None = None,
    ):
        super().__init__(id=id)
        self.inventory = inventory or self._default_inventory()

    def _default_inventory(self) -> Dict[str, List[MethodEntry]]:
        return {
            "risk.checks": [
                registry_get("risk.checks", "var_safe_band")(),
                registry_get("risk.checks", "leverage_position_limits")(),
                registry_get("risk.checks", "liquidation_safety")(),
                registry_get("risk.checks", "margin_call_risk")(),
            ],
        }

    def run(
        self,
        execution: ExecutionSummary,
        price_df: pd.DataFrame,
        regen_attempted: bool = False,
        context_overrides: Dict[str, Any] | None = None,
    ) -> RiskReview:
        """
        Run risk validation.

        Args:
            execution: ExecutionSummary from Trader
            price_df: Price DataFrame
            regen_attempted: Whether this is a regenerated order
            context_overrides: Optional context overrides

        Returns:
            RiskReview with verdict
        """
        self.log("M-A: Risk Validation")

        # Build execution dict
        exec_dict = {
            "order_id": execution.order_id,
            "direction": execution.direction,
            "position_size": execution.position_size,
            "leverage": execution.leverage,
            "entry_price": execution.entry_price,
            "take_profit": execution.take_profit,
            "stop_loss": execution.stop_loss,
            "liquidation_price": execution.liquidation_price,
        }

        # Build context
        context = {
            "price_df": price_df,
            "account_value": 10000.0,
            "max_leverage": 5.0,
            "max_position": 0.5,
            "var_limit": 0.02,
            "min_liquidation_buffer": 0.01,
            **(context_overrides or {}),
        }

        # Run all risk checks
        all_reasons: List[str] = []
        combined_envelope: Dict[str, float] = {}
        worst_verdict = "pass"

        for check in self.inventory.get("risk.checks", []):
            if isinstance(check, tuple):
                instance = check[0]
            else:
                instance = check

            result = instance.evaluate(exec_dict, context)

            # Collect reasons
            all_reasons.extend(result.get("reasons", []))

            # Merge envelope
            combined_envelope.update(result.get("envelope", {}))

            # Track worst verdict
            verdict = result.get("verdict", "pass")
            if verdict == "hard_fail":
                worst_verdict = "hard_fail"
            elif verdict == "soft_fail" and worst_verdict != "hard_fail":
                worst_verdict = "soft_fail"

        # Determine final result
        if worst_verdict == "hard_fail":
            self.log(f"HARD FAIL: {all_reasons}")
            return RiskReview(
                verdict="hard_fail",
                reasons=all_reasons,
                envelope=combined_envelope,
                approved=False,
            )

        if worst_verdict == "soft_fail":
            if regen_attempted:
                self.log(f"Soft fail after regeneration: {all_reasons}")
                return RiskReview(
                    verdict="soft_fail",
                    reasons=all_reasons,
                    envelope=combined_envelope,
                    approved=False,
                )
            else:
                self.log(f"Soft fail (can regenerate): {all_reasons}")
                return RiskReview(
                    verdict="soft_fail",
                    reasons=all_reasons,
                    envelope=combined_envelope,
                    approved=False,
                )

        self.log("PASS: All risk checks passed")
        return RiskReview(
            verdict="pass",
            reasons=["All risk checks passed"],
            envelope={},
            approved=True,
        )
