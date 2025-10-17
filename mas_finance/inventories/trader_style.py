
from __future__ import annotations
from .interfaces import ExecutionStyleMethod
from .registry import register

@register("trader.exec_style", "aggressive_market")
class AggressiveMarket(ExecutionStyleMethod):
    name = "aggressive_market"
    def choose(self, summary: dict, news: list, **kw) -> str:
        return "Aggressive_Market"

@register("trader.exec_style", "passive_laddered_limit")
class PassiveLadderedLimit(ExecutionStyleMethod):
    name = "passive_laddered_limit"
    def choose(self, summary: dict, news: list, **kw) -> str:
        return "Passive_Laddered_Limit"
