
from __future__ import annotations
from typing import Tuple, List, Dict, Any
from datetime import datetime, timezone
import pandas as pd
from .base import BaseAgent
from ..dto.types import ExecutionSummary, ResearchSummary
from ..inventories.registry import get
from ..inventories import trader_style as _ts
from ..tools.news_filter import filter_news_3_stage

class TraderAgent(BaseAgent):
    """Implements MAS T-A (choose style) and T-B (generate order)."""
    def run(self, summary: ResearchSummary, news_items: List[Dict[str,Any]], price_df: pd.DataFrame) -> ExecutionSummary:
        # T-A: select style from inventory
        self.log("🎯 T-A Obtain Execution Style → choose between [aggressive_market, passive_laddered_limit]")
        style = get("trader.exec_style","aggressive_market")().choose(summary.__dict__, news_items)
        # News hygiene
        from_dt, to_dt = price_df.index[0].to_pydatetime(), price_df.index[-1].to_pydatetime()
        news_items = filter_news_3_stage(news_items, from_dt=from_dt, to_dt=to_dt)
        self.log(f"News after 3-stage filter: {len(news_items)} items (time-bounded)")
        # T-B: create order proposal
        last_close = float(price_df["close"].iloc[-1])
        direction = "LONG" if summary.recommendation=="BUY" else ("SHORT" if summary.recommendation=="SELL" else "LONG")
        position_size = 0.2 if summary.confidence<0.6 else 0.5
        leverage = 3.0 if summary.confidence>=0.6 else 2.0
        tp = last_close * (1 + (abs(summary.forecast.get("24h", 0.005))*2 if direction=="LONG" else -abs(summary.forecast.get("24h",0.005))*2))
        sl = last_close * (1 - (abs(summary.risk.get("q05",-0.01))*2 if direction=="LONG" else -abs(summary.risk.get("q95",0.01))*2))
        order = ExecutionSummary(
            order_id=f"sim-{int(datetime.now(tz=timezone.utc).timestamp())}",
            ts_create=datetime.now(tz=timezone.utc),
            style=style,
            order_type="MARKET",
            direction=direction,
            position_size=float(position_size),
            leverage=float(leverage),
            entry_price=last_close,
            take_profit=float(tp),
            stop_loss=float(sl),
            liquidation_price=None,
            closed_price=None
        )
        self.log(f"Order Proposal: {order.direction} size={order.position_size} lev={order.leverage} entry={order.entry_price:.2f}")
        return order
