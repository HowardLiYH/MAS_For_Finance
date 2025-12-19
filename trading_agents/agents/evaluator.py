"""Evaluator Agent - scores agent performance."""
from __future__ import annotations
from typing import Dict, Any, Optional

from .base import BaseAgent
from ..models import AgentScores, ExecutionSummary, ResearchSummary
from ..services.metrics import PerformanceTracker


class EvaluatorAgent(BaseAgent):
    """
    Evaluator Agent: Implements step E-A.

    Scores agents based on performance metrics:
    - Sharpe ratio
    - PnL
    - Hit rate
    - Max drawdown
    - Calibration ECE
    """

    def __init__(
        self,
        id: str = "E1",
        tracker: Optional[PerformanceTracker] = None,
    ):
        super().__init__(id=id)
        self.tracker = tracker or PerformanceTracker()

    def record_trade(
        self,
        order_id: str,
        agent_id: str,
        agent_type: str,
        execution_summary: ExecutionSummary,
        research_summary: Dict[str, Any],
        inventory_methods_used: Dict[str, list[str]],
    ):
        """Record a trade for performance tracking."""
        exec_dict = execution_summary.__dict__ if hasattr(execution_summary, '__dict__') else execution_summary

        self.tracker.record_trade(
            order_id=order_id,
            agent_id=agent_id,
            agent_type=agent_type,
            execution_summary=exec_dict,
            research_summary=research_summary,
            inventory_methods_used=inventory_methods_used,
        )

    def update_trade_result(
        self,
        order_id: str,
        pnl: float,
        pnl_pct: float,
        close_reason: Optional[str] = None,
    ):
        """Update a trade with final PnL."""
        self.tracker.update_trade_result(
            order_id=order_id,
            pnl=pnl,
            pnl_pct=pnl_pct,
            close_reason=close_reason,
        )

    def score(
        self,
        context: Dict[str, Any],
        agent_id: str,
        agent_type: str,
    ) -> AgentScores:
        """
        Score an agent based on performance.

        Args:
            context: Context with execution and risk review
            agent_id: ID of the agent to score
            agent_type: Type of the agent

        Returns:
            AgentScores with metrics
        """
        self.log(f"E-A: Scoring {agent_type}:{agent_id}")

        metrics = self.tracker.calculate_agent_metrics(agent_id, agent_type)

        # Calculate overall score
        score = (
            metrics.get("Sharpe", 0.0) * 0.3 +
            metrics.get("HitRate", 0.0) * 0.3 +
            (metrics.get("PnL", 0.0) / 1000.0) * 0.2 +
            (1.0 - metrics.get("MaxDD", 0.0)) * 0.1 +
            (1.0 - metrics.get("CalibECE", 0.0)) * 0.1
        )

        self.log(f"Score: {score:.4f}, Metrics: {metrics}")

        return AgentScores(
            agent_type=agent_type,
            score=score,
            metrics=metrics,
            period="latest",
        )

    def get_top_agents(self, agent_type: str, top_n: int = 5) -> list[tuple[str, Dict[str, float]]]:
        """Get top N agents by score."""
        return self.tracker.get_top_agents(agent_type, top_n)
