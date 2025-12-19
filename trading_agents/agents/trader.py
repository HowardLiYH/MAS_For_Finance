"""Trader Agent - generates execution orders."""
from __future__ import annotations
from typing import Dict, List, Any, Union, Tuple, Optional
from datetime import datetime, timedelta, timezone
import uuid
import pandas as pd

from .base import BaseAgent
from ..models import ResearchSummary, ExecutionSummary
from ..inventory.registry import get as registry_get
from ..services.llm import generate_trading_proposal
from ..utils.thought_logger import log_thought_process

MethodEntry = Union[Any, Tuple[Any, Dict[str, Any]]]


class TraderAgent(BaseAgent):
    """
    Trader Agent: Implements steps T-A, T-B.

    Outputs:
    - ExecutionSummary with order details
    """

    def __init__(
        self,
        id: str = "T1",
        inventory: Dict[str, List[MethodEntry]] | None = None,
        use_llm: bool = True,
        llm_model: str = "gpt-4o-mini",
        log_thoughts: bool = True,
    ):
        super().__init__(id=id)
        self.inventory = inventory or self._default_inventory()
        self.use_llm = use_llm
        self.llm_model = llm_model
        self.log_thoughts = log_thoughts

    def _default_inventory(self) -> Dict[str, List[MethodEntry]]:
        return {
            "trader.styles": [
                registry_get("trader.styles", "aggressive_market")(),
                registry_get("trader.styles", "passive_laddered_limit")(),
            ],
        }

    def _run_method(self, entry: MethodEntry, *args, **kwargs):
        if isinstance(entry, tuple) and len(entry) == 2:
            instance, run_kwargs = entry
            return instance.run(*args, **{**run_kwargs, **kwargs})
        return entry.run(*args, **kwargs)

    def determine_news_lookback_days(
        self,
        research: ResearchSummary,
        price_df: pd.DataFrame,
        max_lookback_days: int = 30,
    ) -> int:
        """Determine appropriate news lookback based on market state."""
        volatility = float(price_df["close"].pct_change().rolling(24).std().iloc[-1] or 0.02)

        if volatility > 0.03:
            return min(7, max_lookback_days)
        elif research.confidence < 0.6:
            return min(14, max_lookback_days)
        else:
            return min(7, max_lookback_days)

    def _choose_style(self, research: ResearchSummary, news: List[Any]) -> str:
        """Choose execution style based on market context."""
        # Convert research to dict for style methods
        research_dict = research.to_dict() if hasattr(research, 'to_dict') else research.__dict__

        # Evaluate all styles and pick best match
        style_methods = self.inventory.get("trader.styles", [])

        for method in style_methods:
            if isinstance(method, tuple):
                instance = method[0]
            else:
                instance = method

            # High confidence + trending market → aggressive
            if research.confidence > 0.7 and research.market_state in ("bull", "bear"):
                if hasattr(instance, 'name') and "aggressive" in instance.name.lower():
                    return instance.name

            # Low confidence or sideways → passive
            if research.confidence < 0.6 or research.market_state == "sideways":
                if hasattr(instance, 'name') and "passive" in instance.name.lower():
                    return instance.name

        # Default
        return "Aggressive_Market"

    def _calculate_liquidation_price(
        self,
        entry_price: float,
        direction: str,
        leverage: float,
    ) -> float:
        """Calculate liquidation price."""
        maintenance_margin = 0.5
        buffer = 0.5

        if leverage <= 1.0:
            if direction == "LONG":
                return entry_price * 0.01
            else:
                return entry_price * 100.0

        liquidation_distance = (1.0 - maintenance_margin * buffer) / leverage

        if direction == "LONG":
            return entry_price * (1 - liquidation_distance)
        else:
            return entry_price * (1 + liquidation_distance)

    def run(
        self,
        research: ResearchSummary,
        news: List[Any],
        price_df: pd.DataFrame,
    ) -> ExecutionSummary:
        """
        Run the trader pipeline.

        Args:
            research: ResearchSummary from Researcher
            news: List of news items
            price_df: Price DataFrame

        Returns:
            ExecutionSummary with order details
        """
        # T-A: Choose execution style
        self.log("T-A: Choosing Execution Style")
        style = self._choose_style(research, news)
        self.log(f"Selected style: {style}")

        # T-B: Generate order
        self.log("T-B: Generating Order Proposal")

        order_id = f"ORD-{uuid.uuid4().hex[:8].upper()}"
        current_price = float(price_df["close"].iloc[-1])

        # Prepare inputs
        research_dict = research.to_dict() if hasattr(research, 'to_dict') else research.__dict__
        news_dicts = [
            item if isinstance(item, dict) else getattr(item, '__dict__', {})
            for item in news[:20]
        ]
        price_summary = {
            "high": float(price_df["high"].tail(24).max()),
            "low": float(price_df["low"].tail(24).min()),
            "volume": float(price_df["volume"].tail(24).sum()),
        }

        if self.use_llm:
            try:
                proposal, thought_process = generate_trading_proposal(
                    execution_style=style,
                    research_summary=research_dict,
                    news_items=news_dicts,
                    price_data_summary=price_summary,
                    current_price=current_price,
                    model=self.llm_model,
                )

                if self.log_thoughts:
                    log_thought_process(
                        trader_id=self.id,
                        order_id=order_id,
                        thought_process=thought_process,
                        inputs={
                            "research_summary": research_dict,
                            "news_count": len(news_dicts),
                            "current_price": current_price,
                        },
                        outputs=proposal,
                    )
            except Exception as e:
                self.log(f"LLM failed: {e}, using fallback")
                proposal = self._fallback_proposal(research_dict, current_price)
        else:
            proposal = self._fallback_proposal(research_dict, current_price)

        # Calculate liquidation price
        liquidation_price = self._calculate_liquidation_price(
            entry_price=proposal.get("entry_price", current_price),
            direction=proposal.get("direction", "LONG"),
            leverage=proposal.get("leverage", 1.0),
        )

        # Parse execution expired time
        expired_str = proposal.get("execution_expired_time")
        expired_time = None
        if expired_str:
            try:
                expired_time = datetime.fromisoformat(expired_str.replace("Z", "+00:00"))
            except Exception:
                expired_time = datetime.now(tz=timezone.utc) + timedelta(hours=4)

        return ExecutionSummary(
            order_id=order_id,
            timestamp=datetime.now(tz=timezone.utc),
            style=style,
            order_type=proposal.get("order_type", "MARKET"),
            direction=proposal.get("direction", "LONG"),
            position_size=float(proposal.get("position_size", 0.1)),
            leverage=float(proposal.get("leverage", 1.0)),
            entry_price=float(proposal.get("entry_price", current_price)),
            take_profit=float(proposal.get("take_profit", current_price * 1.02)),
            stop_loss=float(proposal.get("stop_loss", current_price * 0.98)),
            liquidation_price=liquidation_price,
            execution_expired_time=expired_time,
        )

    def _fallback_proposal(self, research_dict: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """Fallback rule-based proposal."""
        recommendation = research_dict.get("recommendation", "HOLD")
        confidence = research_dict.get("confidence", 0.5)

        direction = "LONG" if recommendation == "BUY" else "SHORT"
        position_size = 0.2 if confidence < 0.6 else 0.4
        leverage = 2.0 if confidence < 0.6 else 3.0

        forecast = research_dict.get("forecast", {})
        forecast_24h = forecast.get("24h", 0.005)

        risk = research_dict.get("risk", {})
        q05 = risk.get("q05", -0.01)
        q95 = risk.get("q95", 0.01)

        if direction == "LONG":
            take_profit = current_price * (1 + abs(forecast_24h) * 2)
            stop_loss = current_price * (1 + q05)
        else:
            take_profit = current_price * (1 - abs(forecast_24h) * 2)
            stop_loss = current_price * (1 + q95)

        return {
            "direction": direction,
            "position_size": position_size,
            "leverage": leverage,
            "order_type": "MARKET",
            "entry_price": current_price,
            "take_profit": take_profit,
            "stop_loss": stop_loss,
            "execution_expired_time": None,
            "reasoning": "Fallback rule-based logic",
        }
