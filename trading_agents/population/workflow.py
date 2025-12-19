"""Population-based workflow engine.

This module implements the main workflow engine that orchestrates
population-based multi-agent learning for trading.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
from itertools import product
import random

from .base import AgentPopulation, PopulationConfig, AgentRole, PopulationState
from .variants import (
    AnalystVariant, ResearcherVariant, TraderVariant, RiskVariant,
    create_analyst_population, create_researcher_population,
    create_trader_population, create_risk_population,
)
from .transfer import (
    KnowledgeTransferStrategy, SoftUpdateTransfer, TransferSchedule,
    create_transfer_strategy,
)
from .diversity import DiversityPreserver, DiversityPreservationConfig, ParameterDiversity
from .scoring import (
    PopulationScorer, TradeResult, PipelineScorer,
    create_scoring_strategy,
)


@dataclass
class PopulationWorkflowConfig:
    """Configuration for population-based workflow."""
    population_size: int = 5
    transfer_frequency: int = 10
    transfer_tau: float = 0.1
    diversity_weight: float = 0.1
    min_diversity: float = 0.2
    scoring_strategy: str = "pipeline"
    transfer_strategy: str = "soft_update"
    max_pipeline_samples: int = 25  # Max combinations to evaluate per iteration
    elite_fraction: float = 0.2


@dataclass
class IterationResult:
    """Result of a single workflow iteration."""
    iteration: int
    timestamp: datetime
    best_pipeline: Dict[str, str]  # role -> agent_id
    best_pipeline_pnl: float
    avg_pipeline_pnl: float
    population_states: Dict[str, PopulationState]
    transfer_performed: bool
    diversity_action: str
    trade_results: List[TradeResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "timestamp": self.timestamp.isoformat(),
            "best_pipeline": self.best_pipeline,
            "best_pipeline_pnl": self.best_pipeline_pnl,
            "avg_pipeline_pnl": self.avg_pipeline_pnl,
            "population_states": {k: v.to_dict() for k, v in self.population_states.items()},
            "transfer_performed": self.transfer_performed,
            "diversity_action": self.diversity_action,
            "num_trades": len(self.trade_results),
        }


class PopulationWorkflow:
    """
    Main workflow engine for population-based multi-agent trading.

    This engine:
    1. Maintains populations of agents for each role
    2. Samples and evaluates pipeline combinations
    3. Transfers knowledge from best performers
    4. Maintains population diversity
    5. Tracks performance over time
    """

    def __init__(self, config: Optional[PopulationWorkflowConfig] = None):
        self.config = config or PopulationWorkflowConfig()

        # Initialize populations
        self.analyst_pop = create_analyst_population(self.config.population_size)
        self.researcher_pop = create_researcher_population(self.config.population_size)
        self.trader_pop = create_trader_population(self.config.population_size)
        self.risk_pop = create_risk_population(self.config.population_size)

        self.populations = {
            AgentRole.ANALYST: self.analyst_pop,
            AgentRole.RESEARCHER: self.researcher_pop,
            AgentRole.TRADER: self.trader_pop,
            AgentRole.RISK: self.risk_pop,
        }

        # Initialize components
        self.transfer_strategy = create_transfer_strategy(self.config.transfer_strategy)
        self.transfer_schedule = TransferSchedule(
            frequency=self.config.transfer_frequency,
            initial_tau=self.config.transfer_tau * 2,
            final_tau=self.config.transfer_tau * 0.5,
            decay_iterations=100,
            strategy=self.transfer_strategy,
        )

        self.diversity_preserver = DiversityPreserver(
            config=DiversityPreservationConfig(
                min_diversity=self.config.min_diversity,
                target_diversity=self.config.min_diversity * 2,
            ),
            metric=ParameterDiversity(),
        )

        self.scorer = PopulationScorer(
            strategy=create_scoring_strategy(self.config.scoring_strategy),
            diversity_weight=self.config.diversity_weight,
        )

        # History
        self.iteration = 0
        self.history: List[IterationResult] = []
        self.all_trade_results: List[TradeResult] = []

    def run_iteration(
        self,
        price_data: pd.DataFrame,
        news_digest: Optional[Dict[str, Any]] = None,
        market_context: Optional[Dict[str, Any]] = None,
    ) -> IterationResult:
        """
        Run one iteration of the population-based workflow.

        Args:
            price_data: Price data DataFrame
            news_digest: Optional news digest
            market_context: Optional cross-asset market context

        Returns:
            IterationResult with all statistics
        """
        self.iteration += 1
        timestamp = datetime.now()

        # 1. Sample pipeline combinations
        pipelines = self._sample_pipelines()

        # 2. Evaluate each pipeline
        trade_results = []
        pipeline_results = []

        for pipeline in pipelines:
            result = self._evaluate_pipeline(
                pipeline, price_data, news_digest, market_context
            )
            if result:
                trade_results.append(result)
                pipeline_results.append((pipeline, result.pnl_pct))

        # 3. Score agents
        for role, population in self.populations.items():
            # Calculate diversity bonuses
            diversity_bonuses = {
                agent.id: self.diversity_preserver.calculate_diversity_bonus(
                    agent, population
                )
                for agent in population.agents
            }

            # Score population
            scores = self.scorer.score_population(
                population, trade_results, diversity_bonuses
            )
            population.update_scores(scores)

        # 4. Knowledge transfer
        transfer_performed = False
        if self.transfer_schedule.should_transfer(self.iteration):
            tau = self.transfer_schedule.get_tau(self.iteration)
            for population in self.populations.values():
                population.transfer_knowledge(self.transfer_strategy)
            transfer_performed = True

        # 5. Diversity preservation
        diversity_action = "none"
        for population in self.populations.values():
            div_result = self.diversity_preserver.check_and_preserve(
                population, self.iteration
            )
            if div_result["action"] != "none":
                diversity_action = div_result["action"]

        # 6. Find best pipeline
        best_pipeline = {}
        best_pnl = float("-inf")

        for pipeline, pnl in pipeline_results:
            if pnl > best_pnl:
                best_pnl = pnl
                best_pipeline = pipeline

        avg_pnl = np.mean([pnl for _, pnl in pipeline_results]) if pipeline_results else 0.0

        # 7. Record results
        result = IterationResult(
            iteration=self.iteration,
            timestamp=timestamp,
            best_pipeline=best_pipeline,
            best_pipeline_pnl=best_pnl if best_pnl != float("-inf") else 0.0,
            avg_pipeline_pnl=avg_pnl,
            population_states={
                role.value: pop.get_state()
                for role, pop in self.populations.items()
            },
            transfer_performed=transfer_performed,
            diversity_action=diversity_action,
            trade_results=trade_results,
        )

        self.history.append(result)
        self.all_trade_results.extend(trade_results)

        return result

    def _sample_pipelines(self) -> List[Dict[str, str]]:
        """
        Sample pipeline combinations to evaluate.

        Returns list of dicts mapping role -> agent_id
        """
        # Get all possible combinations
        analyst_ids = [a.id for a in self.analyst_pop.agents]
        researcher_ids = [a.id for a in self.researcher_pop.agents]
        trader_ids = [a.id for a in self.trader_pop.agents]
        risk_ids = [a.id for a in self.risk_pop.agents]

        all_combos = list(product(analyst_ids, researcher_ids, trader_ids, risk_ids))

        # If too many, sample
        if len(all_combos) > self.config.max_pipeline_samples:
            # Always include elite combinations
            elite_combos = []
            for a in self.analyst_pop.get_elite():
                for r in self.researcher_pop.get_elite():
                    for t in self.trader_pop.get_elite():
                        for k in self.risk_pop.get_elite():
                            elite_combos.append((a.id, r.id, t.id, k.id))

            # Sample remaining
            remaining = [c for c in all_combos if c not in elite_combos]
            sample_size = self.config.max_pipeline_samples - len(elite_combos)
            if sample_size > 0 and remaining:
                sampled = random.sample(remaining, min(sample_size, len(remaining)))
                all_combos = elite_combos + sampled
            else:
                all_combos = elite_combos[:self.config.max_pipeline_samples]

        return [
            {
                "analyst": combo[0],
                "researcher": combo[1],
                "trader": combo[2],
                "risk": combo[3],
            }
            for combo in all_combos
        ]

    def _evaluate_pipeline(
        self,
        pipeline: Dict[str, str],
        price_data: pd.DataFrame,
        news_digest: Optional[Dict[str, Any]],
        market_context: Optional[Dict[str, Any]],
    ) -> Optional[TradeResult]:
        """
        Evaluate a single pipeline combination.
        """
        try:
            # Get agents
            analyst = self.analyst_pop.get_agent(pipeline["analyst"])
            researcher = self.researcher_pop.get_agent(pipeline["researcher"])
            trader = self.trader_pop.get_agent(pipeline["trader"])
            risk_mgr = self.risk_pop.get_agent(pipeline["risk"])

            if not all([analyst, researcher, trader, risk_mgr]):
                return None

            # Run pipeline
            # 1. Analyst
            analyst_output = analyst.run({"price_data": price_data})

            # 2. Researcher
            researcher_input = {
                "features": analyst_output.get("features", {}),
                "trend": analyst_output.get("trend", "neutral"),
                "price_data": price_data,
            }
            research_output = researcher.run(researcher_input)

            # 3. Trader
            current_price = price_data["close"].iloc[-1] if "close" in price_data.columns else 0
            trader_input = {
                "research": research_output,
                "news_digest": news_digest,
                "current_price": current_price,
                "market_context": market_context,
            }
            trader_output = trader.run(trader_input)

            # 4. Risk Manager
            risk_input = {
                "proposal": trader_output,
                "portfolio_state": {"drawdown": 0.0},  # Simplified
            }
            risk_output = risk_mgr.run(risk_input)

            # 5. Simulate trade result
            if risk_output.get("verdict") == "hard_fail":
                return None

            # Apply adjustments if soft fail
            if risk_output.get("verdict") == "soft_fail":
                adjustments = risk_output.get("adjustments", {})
                for key, value in adjustments.items():
                    if key in trader_output:
                        trader_output[key] = value

            # Simulate simple trade outcome
            direction = trader_output.get("direction", "LONG")
            entry_price = current_price

            # Look ahead for exit (simplified simulation)
            if len(price_data) > 1:
                # Use next period's close as exit (for backtesting)
                exit_price = price_data["close"].iloc[-1] * (1 + np.random.normal(0, 0.01))
            else:
                exit_price = entry_price * (1 + np.random.normal(0, 0.01))

            # Calculate PnL
            if direction == "LONG":
                pnl_pct = (exit_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - exit_price) / entry_price

            leverage = trader_output.get("leverage", 1.0)
            position_size = trader_output.get("position_size", 0.1)
            pnl = pnl_pct * leverage * position_size

            return TradeResult(
                entry_price=entry_price,
                exit_price=exit_price,
                direction=direction,
                position_size=position_size,
                leverage=leverage,
                pnl=pnl,
                pnl_pct=pnl_pct,
                duration_hours=4.0,
                agent_ids=pipeline,
            )

        except Exception as e:
            print(f"Pipeline evaluation error: {e}")
            return None

    def get_best_pipeline(self) -> Dict[str, Any]:
        """Get the current best-performing pipeline combination."""
        best = {
            "analyst": self.analyst_pop.get_best(),
            "researcher": self.researcher_pop.get_best(),
            "trader": self.trader_pop.get_best(),
            "risk": self.risk_pop.get_best(),
        }

        return {
            role: agent.variant_name if agent else "unknown"
            for role, agent in best.items()
        }

    def get_population_summary(self) -> Dict[str, Any]:
        """Get summary of all populations."""
        return {
            role.value: {
                "size": pop.size,
                "iteration": pop.iteration,
                "diversity": pop.calculate_diversity(),
                "best_variant": pop.get_best().variant_name if pop.get_best() else "none",
                "avg_score": np.mean([s.total_score for s in pop.scores.values()]) if pop.scores else 0.0,
            }
            for role, pop in self.populations.items()
        }

    def get_learning_curves(self) -> Dict[str, List[float]]:
        """Get learning curves for each population."""
        curves = {}

        for role, pop in self.populations.items():
            if pop.history:
                curves[role.value] = [
                    state.avg_score for state in pop.history
                ]

        return curves

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        if not self.history:
            return {"message": "No history available"}

        # Aggregate metrics
        all_pnls = [r.best_pipeline_pnl for r in self.history]
        avg_pnls = [r.avg_pipeline_pnl for r in self.history]

        return {
            "total_iterations": self.iteration,
            "best_pnl_history": all_pnls,
            "avg_pnl_history": avg_pnls,
            "final_best_pnl": all_pnls[-1] if all_pnls else 0.0,
            "final_avg_pnl": avg_pnls[-1] if avg_pnls else 0.0,
            "sharpe_ratio": self._calculate_sharpe(all_pnls),
            "best_pipeline": self.get_best_pipeline(),
            "population_summary": self.get_population_summary(),
            "total_trades": len(self.all_trade_results),
            "knowledge_transfers": sum(1 for r in self.history if r.transfer_performed),
            "diversity_actions": sum(1 for r in self.history if r.diversity_action != "none"),
        }

    def _calculate_sharpe(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio."""
        if not returns or len(returns) < 2:
            return 0.0

        mean_ret = np.mean(returns)
        std_ret = np.std(returns)

        if std_ret == 0:
            return 0.0

        return mean_ret / std_ret * np.sqrt(2190)  # Annualized for 4h

    def save_state(self, path: str) -> None:
        """Save workflow state to file."""
        import json

        state = {
            "iteration": self.iteration,
            "config": {
                "population_size": self.config.population_size,
                "transfer_frequency": self.config.transfer_frequency,
                "transfer_tau": self.config.transfer_tau,
                "diversity_weight": self.config.diversity_weight,
                "min_diversity": self.config.min_diversity,
                "scoring_strategy": self.config.scoring_strategy,
                "transfer_strategy": self.config.transfer_strategy,
            },
            "populations": {
                role.value: {
                    "agents": [
                        {
                            "id": a.id,
                            "variant": a.variant_name,
                            "parameters": {
                                k: v.tolist() if isinstance(v, np.ndarray) else v
                                for k, v in a.get_parameters().items()
                            },
                        }
                        for a in pop.agents
                    ],
                }
                for role, pop in self.populations.items()
            },
            "history": [r.to_dict() for r in self.history[-100:]],  # Last 100
        }

        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def load_state(self, path: str) -> None:
        """Load workflow state from file."""
        import json

        with open(path, "r") as f:
            state = json.load(f)

        self.iteration = state["iteration"]

        # Restore agent parameters
        for role_name, pop_state in state.get("populations", {}).items():
            role = AgentRole(role_name)
            if role in self.populations:
                pop = self.populations[role]
                for agent_state in pop_state.get("agents", []):
                    agent = pop.get_agent(agent_state["id"])
                    if agent:
                        params = agent_state.get("parameters", {})
                        # Convert lists back to numpy arrays where appropriate
                        for k, v in params.items():
                            if isinstance(v, list) and v and isinstance(v[0], (int, float)):
                                params[k] = np.array(v)
                        agent.set_parameters(params)


def create_population_workflow(
    population_size: int = 5,
    transfer_frequency: int = 10,
    **kwargs
) -> PopulationWorkflow:
    """Factory function to create a population workflow."""
    config = PopulationWorkflowConfig(
        population_size=population_size,
        transfer_frequency=transfer_frequency,
        **kwargs
    )
    return PopulationWorkflow(config)
