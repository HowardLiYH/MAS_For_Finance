from __future__ import annotations
from typing import Dict, List, Any, Union
import pandas as pd

from .base import BaseAgent
from ..dto.types import RiskReview, ExecutionSummary
from ..inventories.registry import get as registry_get

MethodEntry = Union[Any, tuple]

class RiskManagerAgent(BaseAgent):
    def __init__(self, id: str = "M1", inventory: Dict[str, List[MethodEntry]] | None = None):
        super().__init__(id=id)
        self.inventory = inventory or self._default_inventory()

    def _default_inventory(self) -> Dict[str, List[MethodEntry]]:
        return {
            "risk.checks": [
                registry_get("risk.checks","var_safe_band")(),
                registry_get("risk.checks","leverage_position_limits")(),
            ]
        }

    def _run_check(self, entry: MethodEntry, exec_sum: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(entry, tuple) and len(entry) == 2:
            inst, run_kwargs = entry
            return inst.evaluate(exec_sum, {**context, **run_kwargs})
        return entry.evaluate(exec_sum, context)

    def run(self, execution: ExecutionSummary, price_df: pd.DataFrame, regen_attempted: bool = False, **kwargs) -> RiskReview:
        self.log("🛡️ Risk: applying checks")
        exec_sum = {
            "direction": execution.direction,
            "position_size": execution.position_size,
            "leverage": execution.leverage,
            "entry_price": execution.entry_price,
            "take_profit": execution.take_profit,
            "stop_loss": execution.stop_loss,
        }
        context = {"price_df": price_df}

        decision = {"verdict": "pass", "reasons": [], "envelope": {}}
        for m in self.inventory.get("risk.checks", []):
            result = self._run_check(m, exec_sum, context)
            if result.get("verdict") in ("soft_fail", "hard_fail"):
                decision = result
                break

        return RiskReview(
            verdict=decision["verdict"],
            reasons=decision.get("reasons", []),
            envelope=decision.get("envelope", {}),
            approved=decision.get("verdict") == "pass",
        )
