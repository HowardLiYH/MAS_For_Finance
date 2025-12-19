"""Execution style methods for Trader Agent (Step T-A)."""
from __future__ import annotations
from typing import Dict, Any, List
from ..interfaces import ExecutionStyleMethod
from ..registry import register


@register("trader.styles", "aggressive_market")
class AggressiveMarket(ExecutionStyleMethod):
    """
    Aggressive market order execution style.

    Best for: High confidence, clear market direction, strong signals.
    """
    name = "aggressive_market"

    def choose(self, research_summary: Dict[str, Any], news: List[Any], **kwargs) -> str:
        confidence = research_summary.get("confidence", 0.5)
        market_state = research_summary.get("market_state", "unknown")
        signals = research_summary.get("signals", [])

        # High confidence and clear direction → aggressive
        if confidence > 0.7 and market_state in ("bull", "bear"):
            return "Aggressive_Market"

        # Strong signals → aggressive
        if len(signals) >= 2:
            return "Aggressive_Market"

        return "Aggressive_Market"


@register("trader.styles", "passive_laddered_limit")
class PassiveLadderedLimit(ExecutionStyleMethod):
    """
    Passive laddered limit order execution style.

    Best for: Low confidence, sideways market, high uncertainty.
    """
    name = "passive_laddered_limit"

    def choose(self, research_summary: Dict[str, Any], news: List[Any], **kwargs) -> str:
        confidence = research_summary.get("confidence", 0.5)
        market_state = research_summary.get("market_state", "unknown")
        risk = research_summary.get("risk", {})

        # Low confidence → passive
        if confidence < 0.6:
            return "Passive_Laddered_Limit"

        # Sideways market → passive
        if market_state == "sideways":
            return "Passive_Laddered_Limit"

        # High uncertainty → passive
        if risk:
            q_range = abs(risk.get("q95", 0) - risk.get("q05", 0))
            if q_range > 0.1:
                return "Passive_Laddered_Limit"

        return "Passive_Laddered_Limit"
