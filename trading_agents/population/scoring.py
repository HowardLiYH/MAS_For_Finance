"""Scoring mechanisms for population-based learning.

This module implements various strategies for scoring and ranking agents,
including individual performance, pipeline contribution, and credit assignment.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime
from itertools import combinations

from .base import PopulationAgent, AgentScore, AgentPopulation


@dataclass
class TradeResult:
    """Result of a single trade for evaluation."""
    entry_price: float
    exit_price: float
    direction: str  # "LONG" or "SHORT"
    position_size: float
    leverage: float
    pnl: float
    pnl_pct: float
    duration_hours: float
    agent_ids: Dict[str, str]  # role -> agent_id mapping
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "direction": self.direction,
            "position_size": self.position_size,
            "leverage": self.leverage,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "duration_hours": self.duration_hours,
            "agent_ids": self.agent_ids,
            "timestamp": self.timestamp.isoformat(),
        }


class ScoringStrategy(ABC):
    """Base class for scoring strategies."""

    @abstractmethod
    def score(
        self,
        agents: List[PopulationAgent],
        trade_results: List[TradeResult]
    ) -> Dict[str, AgentScore]:
        """
        Score agents based on trade results.

        Returns:
            Dictionary mapping agent_id to AgentScore
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the scoring strategy."""
        pass


class IndividualScorer(ScoringStrategy):
    """
    Score agents based on their individual contribution to trades.

    Each agent is scored based on the average performance of trades
    they participated in.
    """

    def __init__(
        self,
        sharpe_weight: float = 0.4,
        pnl_weight: float = 0.3,
        hit_rate_weight: float = 0.3
    ):
        self.sharpe_weight = sharpe_weight
        self.pnl_weight = pnl_weight
        self.hit_rate_weight = hit_rate_weight

    @property
    def name(self) -> str:
        return "individual"

    def score(
        self,
        agents: List[PopulationAgent],
        trade_results: List[TradeResult]
    ) -> Dict[str, AgentScore]:
        scores = {}

        # Group trades by agent
        agent_trades: Dict[str, List[TradeResult]] = {}
        for agent in agents:
            agent_trades[agent.id] = []

        for trade in trade_results:
            for role, agent_id in trade.agent_ids.items():
                if agent_id in agent_trades:
                    agent_trades[agent_id].append(trade)

        # Calculate scores for each agent
        all_scores = []
        for agent in agents:
            trades = agent_trades.get(agent.id, [])

            if not trades:
                individual_score = 0.0
            else:
                # Calculate metrics
                pnls = [t.pnl_pct for t in trades]
                sharpe = self._calculate_sharpe(pnls)
                avg_pnl = np.mean(pnls)
                hit_rate = sum(1 for p in pnls if p > 0) / len(pnls)

                # Weighted score
                individual_score = (
                    self.sharpe_weight * self._normalize_sharpe(sharpe) +
                    self.pnl_weight * self._normalize_pnl(avg_pnl) +
                    self.hit_rate_weight * hit_rate
                )

            all_scores.append((agent.id, individual_score))

        # Calculate ranks
        sorted_scores = sorted(all_scores, key=lambda x: x[1], reverse=True)
        ranks = {aid: (len(sorted_scores) - i) / len(sorted_scores)
                 for i, (aid, _) in enumerate(sorted_scores)}

        # Create AgentScore objects
        iteration = 0  # Will be set externally
        for agent_id, ind_score in all_scores:
            scores[agent_id] = AgentScore(
                agent_id=agent_id,
                individual_score=ind_score,
                relative_rank=ranks[agent_id],
                pipeline_contribution=0.0,  # Not calculated in individual scoring
                diversity_bonus=0.0,  # Will be added by diversity preserver
                total_score=ind_score,  # For individual scoring, total = individual
                iteration=iteration,
            )

        return scores

    def _calculate_sharpe(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio."""
        if not returns or len(returns) < 2:
            return 0.0

        mean_ret = np.mean(returns)
        std_ret = np.std(returns)

        if std_ret == 0:
            return 0.0

        # Annualized for 4h intervals (6 * 365 = 2190 periods per year)
        return mean_ret / std_ret * np.sqrt(2190)

    def _normalize_sharpe(self, sharpe: float) -> float:
        """Normalize Sharpe ratio to [0, 1]."""
        # Assume Sharpe of 2 is excellent
        return np.clip(sharpe / 2.0, 0.0, 1.0)

    def _normalize_pnl(self, pnl: float) -> float:
        """Normalize PnL to [0, 1]."""
        # Assume 5% per trade is excellent
        return np.clip((pnl + 0.05) / 0.10, 0.0, 1.0)


class PipelineScorer(ScoringStrategy):
    """
    Score agents based on full pipeline performance.

    Considers how well agents work together in the complete pipeline,
    not just individual performance.
    """

    def __init__(
        self,
        individual_weight: float = 0.4,
        pipeline_weight: float = 0.4,
        consistency_weight: float = 0.2
    ):
        self.individual_weight = individual_weight
        self.pipeline_weight = pipeline_weight
        self.consistency_weight = consistency_weight
        self.individual_scorer = IndividualScorer()

    @property
    def name(self) -> str:
        return "pipeline"

    def score(
        self,
        agents: List[PopulationAgent],
        trade_results: List[TradeResult]
    ) -> Dict[str, AgentScore]:
        # Get individual scores first
        individual_scores = self.individual_scorer.score(agents, trade_results)

        # Calculate pipeline contributions
        pipeline_contributions = self._calculate_pipeline_contributions(
            agents, trade_results
        )

        # Calculate consistency scores
        consistency_scores = self._calculate_consistency(agents, trade_results)

        # Combine scores
        scores = {}
        for agent in agents:
            ind_score = individual_scores.get(agent.id)
            if ind_score:
                pipeline_contrib = pipeline_contributions.get(agent.id, 0.0)
                consistency = consistency_scores.get(agent.id, 0.0)

                total = (
                    self.individual_weight * ind_score.individual_score +
                    self.pipeline_weight * pipeline_contrib +
                    self.consistency_weight * consistency
                )

                scores[agent.id] = AgentScore(
                    agent_id=agent.id,
                    individual_score=ind_score.individual_score,
                    relative_rank=ind_score.relative_rank,
                    pipeline_contribution=pipeline_contrib,
                    diversity_bonus=0.0,
                    total_score=total,
                    iteration=ind_score.iteration,
                )

        return scores

    def _calculate_pipeline_contributions(
        self,
        agents: List[PopulationAgent],
        trade_results: List[TradeResult]
    ) -> Dict[str, float]:
        """Calculate each agent's contribution to pipeline success."""
        contributions = {a.id: [] for a in agents}

        # Group trades by pipeline (agent combination)
        pipeline_results: Dict[tuple, List[TradeResult]] = {}
        for trade in trade_results:
            pipeline_key = tuple(sorted(trade.agent_ids.items()))
            if pipeline_key not in pipeline_results:
                pipeline_results[pipeline_key] = []
            pipeline_results[pipeline_key].append(trade)

        # Calculate average performance for each pipeline
        pipeline_perf: Dict[tuple, float] = {}
        for pipeline_key, trades in pipeline_results.items():
            pnls = [t.pnl_pct for t in trades]
            pipeline_perf[pipeline_key] = np.mean(pnls) if pnls else 0.0

        # Attribute performance to agents
        for trade in trade_results:
            pipeline_key = tuple(sorted(trade.agent_ids.items()))
            perf = pipeline_perf.get(pipeline_key, 0.0)

            for role, agent_id in trade.agent_ids.items():
                if agent_id in contributions:
                    contributions[agent_id].append(perf)

        # Average contributions
        return {
            agent_id: np.mean(perfs) if perfs else 0.0
            for agent_id, perfs in contributions.items()
        }

    def _calculate_consistency(
        self,
        agents: List[PopulationAgent],
        trade_results: List[TradeResult]
    ) -> Dict[str, float]:
        """Calculate how consistently each agent performs."""
        consistency = {}

        for agent in agents:
            agent_trades = [
                t for t in trade_results
                if agent.id in t.agent_ids.values()
            ]

            if len(agent_trades) < 2:
                consistency[agent.id] = 0.5
            else:
                pnls = [t.pnl_pct for t in agent_trades]
                std = np.std(pnls)
                mean = abs(np.mean(pnls))

                # Lower variance relative to mean = more consistent
                if mean > 0:
                    cv = std / mean  # Coefficient of variation
                    consistency[agent.id] = np.exp(-cv)  # High CV = low consistency
                else:
                    consistency[agent.id] = 0.5

        return consistency


class ShapleyScorer(ScoringStrategy):
    """
    Score agents using Shapley values for credit assignment.

    Shapley values provide a fair way to distribute credit among agents
    in a coalition (pipeline), accounting for marginal contributions.
    """

    def __init__(self, num_samples: int = 100):
        self.num_samples = num_samples  # Monte Carlo sampling for approximation

    @property
    def name(self) -> str:
        return "shapley"

    def score(
        self,
        agents: List[PopulationAgent],
        trade_results: List[TradeResult]
    ) -> Dict[str, AgentScore]:
        if not trade_results:
            return {
                a.id: AgentScore(
                    agent_id=a.id,
                    individual_score=0.0,
                    relative_rank=0.5,
                    pipeline_contribution=0.0,
                    diversity_bonus=0.0,
                    total_score=0.0,
                    iteration=0,
                )
                for a in agents
            }

        # Calculate Shapley values
        shapley_values = self._calculate_shapley_values(agents, trade_results)

        # Normalize to [0, 1]
        max_val = max(shapley_values.values()) if shapley_values else 1.0
        min_val = min(shapley_values.values()) if shapley_values else 0.0
        range_val = max_val - min_val if max_val != min_val else 1.0

        normalized = {
            aid: (val - min_val) / range_val
            for aid, val in shapley_values.items()
        }

        # Calculate ranks
        sorted_agents = sorted(normalized.items(), key=lambda x: x[1], reverse=True)
        ranks = {aid: (len(sorted_agents) - i) / len(sorted_agents)
                 for i, (aid, _) in enumerate(sorted_agents)}

        # Create scores
        scores = {}
        for agent in agents:
            shapley = normalized.get(agent.id, 0.0)
            scores[agent.id] = AgentScore(
                agent_id=agent.id,
                individual_score=shapley,  # Use Shapley as individual score
                relative_rank=ranks.get(agent.id, 0.5),
                pipeline_contribution=shapley_values.get(agent.id, 0.0),
                diversity_bonus=0.0,
                total_score=shapley,
                iteration=0,
            )

        return scores

    def _calculate_shapley_values(
        self,
        agents: List[PopulationAgent],
        trade_results: List[TradeResult]
    ) -> Dict[str, float]:
        """
        Calculate Shapley values using Monte Carlo sampling.

        For each permutation of agents, calculate the marginal contribution
        of each agent when they join the coalition.
        """
        shapley = {a.id: 0.0 for a in agents}
        agent_ids = [a.id for a in agents]

        # Group trades by agent participation
        agent_trade_pnl: Dict[str, List[float]] = {aid: [] for aid in agent_ids}
        for trade in trade_results:
            for role, aid in trade.agent_ids.items():
                if aid in agent_trade_pnl:
                    agent_trade_pnl[aid].append(trade.pnl_pct)

        # Monte Carlo sampling of permutations
        for _ in range(self.num_samples):
            perm = np.random.permutation(agent_ids).tolist()

            coalition_value = 0.0
            for i, agent_id in enumerate(perm):
                # Value of coalition without this agent
                prev_value = coalition_value

                # Value of coalition with this agent
                # Approximate by average performance of trades this agent was in
                agent_pnls = agent_trade_pnl.get(agent_id, [])
                if agent_pnls:
                    new_value = prev_value + np.mean(agent_pnls)
                else:
                    new_value = prev_value

                # Marginal contribution
                marginal = new_value - prev_value
                shapley[agent_id] += marginal

                coalition_value = new_value

        # Average over samples
        for agent_id in shapley:
            shapley[agent_id] /= self.num_samples

        return shapley


class PopulationScorer:
    """
    Main scorer that combines multiple scoring strategies and
    manages score history.
    """

    def __init__(
        self,
        strategy: Optional[ScoringStrategy] = None,
        diversity_weight: float = 0.1
    ):
        self.strategy = strategy or PipelineScorer()
        self.diversity_weight = diversity_weight
        self.history: List[Dict[str, AgentScore]] = []
        self.iteration = 0

    def score_population(
        self,
        population: AgentPopulation,
        trade_results: List[TradeResult],
        diversity_bonuses: Optional[Dict[str, float]] = None
    ) -> Dict[str, AgentScore]:
        """
        Score all agents in a population.

        Args:
            population: The agent population to score
            trade_results: Recent trade results
            diversity_bonuses: Optional pre-calculated diversity bonuses

        Returns:
            Dictionary mapping agent_id to AgentScore
        """
        # Get base scores from strategy
        scores = self.strategy.score(population.agents, trade_results)

        # Add diversity bonuses
        if diversity_bonuses:
            for agent_id, score in scores.items():
                bonus = diversity_bonuses.get(agent_id, 0.0)
                score.diversity_bonus = bonus
                score.total_score = (
                    score.total_score * (1 - self.diversity_weight) +
                    bonus * self.diversity_weight
                )

        # Update iteration
        for score in scores.values():
            score.iteration = self.iteration

        self.iteration += 1
        self.history.append(scores.copy())

        return scores

    def get_rankings(
        self,
        scores: Dict[str, AgentScore]
    ) -> List[Tuple[str, float]]:
        """Get ranked list of (agent_id, total_score) tuples."""
        return sorted(
            [(aid, s.total_score) for aid, s in scores.items()],
            key=lambda x: x[1],
            reverse=True
        )

    def get_best_agent_id(self, scores: Dict[str, AgentScore]) -> str:
        """Get the ID of the best-performing agent."""
        if not scores:
            return ""
        return max(scores, key=lambda x: scores[x].total_score)

    def get_score_history(self, agent_id: str) -> List[float]:
        """Get the score history for a specific agent."""
        return [
            h[agent_id].total_score
            for h in self.history
            if agent_id in h
        ]

    def get_population_stats(self) -> Dict[str, Any]:
        """Get statistics about population scoring over time."""
        if not self.history:
            return {"message": "No scoring history available"}

        # Track average scores over time
        avg_scores = []
        best_scores = []

        for h in self.history:
            scores = [s.total_score for s in h.values()]
            if scores:
                avg_scores.append(np.mean(scores))
                best_scores.append(np.max(scores))

        return {
            "num_iterations": len(self.history),
            "avg_score_trend": avg_scores,
            "best_score_trend": best_scores,
            "final_avg": avg_scores[-1] if avg_scores else 0.0,
            "final_best": best_scores[-1] if best_scores else 0.0,
            "improvement": (
                (avg_scores[-1] - avg_scores[0]) / (abs(avg_scores[0]) + 1e-6)
                if len(avg_scores) > 1 else 0.0
            ),
        }


def create_scoring_strategy(name: str, **kwargs) -> ScoringStrategy:
    """Factory function to create scoring strategies."""
    strategies = {
        "individual": IndividualScorer,
        "pipeline": PipelineScorer,
        "shapley": ShapleyScorer,
    }

    if name not in strategies:
        raise ValueError(f"Unknown scoring strategy: {name}. Available: {list(strategies.keys())}")

    return strategies[name](**kwargs)
