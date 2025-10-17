from __future__ import annotations
from typing import Dict, List, Any, Union
from datetime import datetime, timezone
import pandas as pd

from .base import BaseAgent
from ..dto.types import ExecutionSummary, ResearchSummary
from ..inventories.registry import get as registry_get

MethodEntry = Union[Any, tuple]

class TraderAgent(BaseAgent):
    """Implements MAS T-A (choose style) and T-B (generate order)."""

    def __init__(self, id: str = "T1", inventory: Dict[str, List[MethodEntry]] | None = None):
        super().__init__(id=id)
        self.inventory = inventory or self._default_inventory()

    def _default_inventory(self) -> Dict[str, List[MethodEntry]]:
        return {
            "trader.exec_style": [
                registry_get("trader.exec_style","aggressive_market")(),
                registry_get("trader.exec_style","passive_laddered_limit")(),
            ]
        }

    def _choose_style(self, summary: Dict[str, Any], news: list) -> str:
        styles = self.inventory.get("trader.exec_style", [])
        if not styles:
            return "Aggressive_Market"
        # prefer first; you can add scoring here later
        entry = styles[0]
        if isinstance(entry, tuple) and len(entry) == 2:
            inst, run_kwargs = entry
            return inst.choose(summary, news, **run_kwargs)
        return entry.choose(summary, news)

    def run(self, summary: ResearchSummary, news_items: List[Dict[str,Any]], price_df: pd.DataFrame) -> ExecutionSummary:
        self.log("🎯 T-A Choose execution style")
        style = self._choose_style(summary.__dict__, news_items)

        self.log("🧾 T-B Build order proposal")
        last_close = float(price_df["close"].iloc[-1])
        direction = "LONG" if summary.recommendation=="BUY" else ("SHORT" if summary.recommendation=="SELL" else "LONG")
        position_size = 0.2 if summary.confidence < 0.6 else 0.5
        leverage = 3.0 if summary.confidence >= 0.6 else 2.0

        f24 = float(summary.forecast.get("24h", 0.005))
        q05 = float(summary.risk.get("q05", -0.01)) if hasattr(summary, "risk") else -0.01
        q95 = float(summary.risk.get("q95",  0.01)) if hasattr(summary, "risk") else  0.01

        entry_price = last_close
        take_profit = last_close * (1 + (abs(f24)*2 if direction=="LONG" else -abs(f24)*2))
        stop_loss   = last_close * (1 - (abs(q05)   if direction=="LONG" else -abs(q95)))

        return ExecutionSummary(
            order_id=f"sim-{int(datetime.now(tz=timezone.utc).timestamp())}",
            ts_create=datetime.now(tz=timezone.utc),
            style=style,
            order_type="MARKET",
            direction=direction,
            position_size=position_size,
            leverage=leverage,
            entry_price=entry_price,
            take_profit=take_profit,
            stop_loss=stop_loss,
        )
