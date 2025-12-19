"""Selector-based Population Workflow.

This workflow engine uses MethodSelector agents that dynamically choose
which methods to use from their role's inventory.

Key differences from fixed-variant approach:
1. Agents SELECT methods, not locked into fixed strategies
2. Continual learning is about WHICH methods to pick
3. Knowledge transfer shares selection preferences
4. Diversity is about method selection diversity
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
from itertools import product
import random

from .selector import (
    MethodSelector,
    SelectorPopulation,
    SelectorPopulationConfig,
    AgentRole,
    SelectionOutcome,
)
from .inventories import (
    ANALYST_INVENTORY,
    RESEARCHER_INVENTORY,
    TRADER_INVENTORY,
    RISK_INVENTORY,
    get_method_info,
)


@dataclass
class SelectorWorkflowConfig:
    """Configuration for selector-based workflow."""
    population_size: int = 5
    max_methods_per_agent: int = 3
    transfer_frequency: int = 10
    transfer_tau: float = 0.1
    exploration_rate: float = 0.15
    learning_rate: float = 0.1
    max_pipeline_samples: int = 25


@dataclass
class PipelineResult:
    """Result of running a single pipeline."""
    analyst_id: str
    researcher_id: str
    trader_id: str
    risk_id: str
    analyst_methods: List[str]
    researcher_methods: List[str]
    trader_methods: List[str]
    risk_methods: List[str]
    pnl: float
    sharpe: float
    success: bool
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agents": {
                "analyst": self.analyst_id,
                "researcher": self.researcher_id,
                "trader": self.trader_id,
                "risk": self.risk_id,
            },
            "methods": {
                "analyst": self.analyst_methods,
                "researcher": self.researcher_methods,
                "trader": self.trader_methods,
                "risk": self.risk_methods,
            },
            "pnl": self.pnl,
            "sharpe": self.sharpe,
            "success": self.success,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass 
class IterationSummary:
    """Summary of one iteration."""
    iteration: int
    best_pnl: float
    avg_pnl: float
    best_pipeline: Dict[str, List[str]]  # role -> methods
    selection_diversity: Dict[str, float]  # role -> diversity
    transfer_performed: bool
    timestamp: datetime = field(default_factory=datetime.now)


class SelectorWorkflow:
    """
    Main workflow engine for PopAgent with method selection.
    
    Each iteration:
    1. Agents select methods from inventory
    2. Sample pipeline combinations
    3. Evaluate each pipeline
    4. Update agent preferences based on outcomes
    5. Transfer knowledge from best performers
    6. Ensure selection diversity
    """
    
    def __init__(self, config: Optional[SelectorWorkflowConfig] = None):
        self.config = config or SelectorWorkflowConfig()
        
        # Create populations for each role
        self.analyst_pop = SelectorPopulation(SelectorPopulationConfig(
            role=AgentRole.ANALYST,
            inventory=ANALYST_INVENTORY,
            population_size=self.config.population_size,
            max_methods_per_agent=self.config.max_methods_per_agent,
            exploration_rate=self.config.exploration_rate,
            learning_rate=self.config.learning_rate,
            transfer_frequency=self.config.transfer_frequency,
            transfer_tau=self.config.transfer_tau,
        ))
        
        self.researcher_pop = SelectorPopulation(SelectorPopulationConfig(
            role=AgentRole.RESEARCHER,
            inventory=RESEARCHER_INVENTORY,
            population_size=self.config.population_size,
            max_methods_per_agent=self.config.max_methods_per_agent,
            exploration_rate=self.config.exploration_rate,
            learning_rate=self.config.learning_rate,
            transfer_frequency=self.config.transfer_frequency,
            transfer_tau=self.config.transfer_tau,
        ))
        
        self.trader_pop = SelectorPopulation(SelectorPopulationConfig(
            role=AgentRole.TRADER,
            inventory=TRADER_INVENTORY,
            population_size=self.config.population_size,
            max_methods_per_agent=self.config.max_methods_per_agent,
            exploration_rate=self.config.exploration_rate,
            learning_rate=self.config.learning_rate,
            transfer_frequency=self.config.transfer_frequency,
            transfer_tau=self.config.transfer_tau,
        ))
        
        self.risk_pop = SelectorPopulation(SelectorPopulationConfig(
            role=AgentRole.RISK,
            inventory=RISK_INVENTORY,
            population_size=self.config.population_size,
            max_methods_per_agent=self.config.max_methods_per_agent,
            exploration_rate=self.config.exploration_rate,
            learning_rate=self.config.learning_rate,
            transfer_frequency=self.config.transfer_frequency,
            transfer_tau=self.config.transfer_tau,
        ))
        
        self.populations = {
            AgentRole.ANALYST: self.analyst_pop,
            AgentRole.RESEARCHER: self.researcher_pop,
            AgentRole.TRADER: self.trader_pop,
            AgentRole.RISK: self.risk_pop,
        }
        
        # History
        self.iteration = 0
        self.history: List[IterationSummary] = []
        self.all_results: List[PipelineResult] = []
    
    def run_iteration(
        self,
        price_data: pd.DataFrame,
        market_context: Optional[Dict[str, Any]] = None,
        news_digest: Optional[Dict[str, Any]] = None,
    ) -> IterationSummary:
        """
        Run one iteration of the population-based method selection workflow.
        """
        self.iteration += 1
        context = market_context or {}
        
        # 1. Each agent selects methods
        for pop in self.populations.values():
            for agent in pop.agents:
                agent.select_methods(context)
        
        # 2. Sample pipeline combinations
        pipelines = self._sample_pipelines()
        
        # 3. Evaluate each pipeline
        results: List[PipelineResult] = []
        for pipeline in pipelines:
            result = self._evaluate_pipeline(pipeline, price_data, context, news_digest)
            if result:
                results.append(result)
        
        # 4. Update agent preferences based on outcomes
        self._update_preferences(results, context)
        
        # 5. Score agents and transfer knowledge
        self._score_and_transfer()
        
        # 6. Ensure diversity
        for pop in self.populations.values():
            pop.ensure_diversity()
        
        # Create summary
        best_result = max(results, key=lambda r: r.pnl) if results else None
        
        summary = IterationSummary(
            iteration=self.iteration,
            best_pnl=best_result.pnl if best_result else 0.0,
            avg_pnl=np.mean([r.pnl for r in results]) if results else 0.0,
            best_pipeline={
                "analyst": best_result.analyst_methods if best_result else [],
                "researcher": best_result.researcher_methods if best_result else [],
                "trader": best_result.trader_methods if best_result else [],
                "risk": best_result.risk_methods if best_result else [],
            },
            selection_diversity={
                role.value: pop.calculate_selection_diversity()
                for role, pop in self.populations.items()
            },
            transfer_performed=self.analyst_pop.should_transfer(),
        )
        
        self.history.append(summary)
        self.all_results.extend(results)
        
        return summary
    
    def _sample_pipelines(self) -> List[Dict[str, MethodSelector]]:
        """Sample pipeline combinations of agents."""
        analysts = self.analyst_pop.agents
        researchers = self.researcher_pop.agents
        traders = self.trader_pop.agents
        risks = self.risk_pop.agents
        
        # All possible combinations
        all_combos = list(product(analysts, researchers, traders, risks))
        
        # Sample if too many
        if len(all_combos) > self.config.max_pipeline_samples:
            # Include best agents
            elite_combo = (
                self.analyst_pop.get_best() or analysts[0],
                self.researcher_pop.get_best() or researchers[0],
                self.trader_pop.get_best() or traders[0],
                self.risk_pop.get_best() or risks[0],
            )
            
            sampled = random.sample(
                [c for c in all_combos if c != elite_combo],
                self.config.max_pipeline_samples - 1
            )
            all_combos = [elite_combo] + sampled
        
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
        pipeline: Dict[str, MethodSelector],
        price_data: pd.DataFrame,
        context: Dict[str, Any],
        news_digest: Optional[Dict[str, Any]],
    ) -> Optional[PipelineResult]:
        """Evaluate a single pipeline combination."""
        try:
            analyst = pipeline["analyst"]
            researcher = pipeline["researcher"]
            trader = pipeline["trader"]
            risk = pipeline["risk"]
            
            # Get selected methods
            analyst_methods = analyst.current_selection
            researcher_methods = researcher.current_selection
            trader_methods = trader.current_selection
            risk_methods = risk.current_selection
            
            # Simulate pipeline execution
            # In real implementation, this would call actual method implementations
            
            # For now, simulate based on method characteristics
            pnl = self._simulate_pipeline_pnl(
                analyst_methods,
                researcher_methods,
                trader_methods,
                risk_methods,
                price_data,
                context,
            )
            
            # Calculate Sharpe approximation
            sharpe = pnl / 0.02 if pnl != 0 else 0.0  # Rough approximation
            
            return PipelineResult(
                analyst_id=analyst.id,
                researcher_id=researcher.id,
                trader_id=trader.id,
                risk_id=risk.id,
                analyst_methods=analyst_methods,
                researcher_methods=researcher_methods,
                trader_methods=trader_methods,
                risk_methods=risk_methods,
                pnl=pnl,
                sharpe=sharpe,
                success=pnl > 0,
            )
            
        except Exception as e:
            print(f"Pipeline evaluation error: {e}")
            return None
    
    def _simulate_pipeline_pnl(
        self,
        analyst_methods: List[str],
        researcher_methods: List[str],
        trader_methods: List[str],
        risk_methods: List[str],
        price_data: pd.DataFrame,
        context: Dict[str, Any],
    ) -> float:
        """
        Simulate PnL based on method selection.
        
        In production, this would actually execute the methods.
        For research, we can use historical performance or learned simulators.
        """
        # Base PnL from price movement
        if len(price_data) > 1 and "close" in price_data.columns:
            price_return = (
                price_data["close"].iloc[-1] / price_data["close"].iloc[-2] - 1
            )
        else:
            price_return = np.random.normal(0, 0.01)
        
        # Method synergy bonuses (some combinations work better)
        synergy = 0.0
        
        # Analyst synergies
        if "HMM_Regime" in analyst_methods and "VolatilityClustering" in analyst_methods:
            synergy += 0.002  # Regime + volatility work well together
        if "RSI" in analyst_methods and "MACD" in analyst_methods:
            synergy += 0.001  # Classic combo
        
        # Researcher synergies
        if "BootstrapEnsemble" in researcher_methods or "QuantileRegression" in researcher_methods:
            synergy += 0.001  # Uncertainty helps
        if "TemporalFusion" in researcher_methods:
            synergy += 0.002  # Advanced model bonus
        
        # Trader synergies
        if "VolatilityScaled" in trader_methods and "KellyCriterion" in trader_methods:
            synergy += 0.001  # Good sizing combo
        
        # Risk protection
        risk_penalty = 0.0
        if "MaxDrawdown" not in risk_methods and "DailyStopLoss" not in risk_methods:
            risk_penalty = 0.005  # No stop loss is risky
        
        # Context-aware bonuses
        trend = context.get("trend", "neutral")
        if trend == "bullish" and "MomentumEntry" in trader_methods:
            synergy += 0.002
        if trend == "bearish" and "ContrarianEntry" in trader_methods:
            synergy += 0.001
        
        # Final PnL
        pnl = price_return + synergy - risk_penalty + np.random.normal(0, 0.005)
        
        return pnl
    
    def _update_preferences(
        self,
        results: List[PipelineResult],
        context: Dict[str, Any],
    ) -> None:
        """Update agent preferences based on pipeline results."""
        # Group results by agent
        agent_outcomes: Dict[str, List[Tuple[List[str], float]]] = {}
        
        for result in results:
            # Analyst
            key = f"analyst:{result.analyst_id}"
            if key not in agent_outcomes:
                agent_outcomes[key] = []
            agent_outcomes[key].append((result.analyst_methods, result.pnl))
            
            # Researcher
            key = f"researcher:{result.researcher_id}"
            if key not in agent_outcomes:
                agent_outcomes[key] = []
            agent_outcomes[key].append((result.researcher_methods, result.pnl))
            
            # Trader
            key = f"trader:{result.trader_id}"
            if key not in agent_outcomes:
                agent_outcomes[key] = []
            agent_outcomes[key].append((result.trader_methods, result.pnl))
            
            # Risk
            key = f"risk:{result.risk_id}"
            if key not in agent_outcomes:
                agent_outcomes[key] = []
            agent_outcomes[key].append((result.risk_methods, result.pnl))
        
        # Update each agent
        for key, outcomes in agent_outcomes.items():
            role, agent_id = key.split(":")
            pop = self.populations[AgentRole(role)]
            agent = pop.get_agent(agent_id)
            
            if agent:
                for methods, pnl in outcomes:
                    agent.update_from_outcome(SelectionOutcome(
                        methods_used=methods,
                        reward=pnl,
                        market_context=context,
                    ))
    
    def _score_and_transfer(self) -> None:
        """Score agents and perform knowledge transfer if needed."""
        for role, pop in self.populations.items():
            # Calculate scores
            scores = {}
            for agent in pop.agents:
                if agent.outcome_history:
                    recent = agent.outcome_history[-20:]
                    avg_reward = np.mean([o.reward for o in recent])
                    scores[agent.id] = avg_reward
                else:
                    scores[agent.id] = 0.0
            
            pop.update_scores(scores)
            
            # Transfer if it's time
            if pop.should_transfer():
                pop.transfer_knowledge()
    
    def get_best_methods(self) -> Dict[str, List[str]]:
        """Get the currently best method selection for each role."""
        result = {}
        for role, pop in self.populations.items():
            best = pop.get_best()
            if best:
                result[role.value] = best.current_selection
            else:
                result[role.value] = []
        return result
    
    def get_method_popularity(self) -> Dict[str, Dict[str, float]]:
        """Get how popular each method is across the population."""
        result = {}
        for role, pop in self.populations.items():
            usage = pop._get_method_usage()
            total = sum(usage.values()) or 1
            result[role.value] = {m: c / total for m, c in usage.items()}
        return result
    
    def get_learning_progress(self) -> Dict[str, Any]:
        """Get learning progress metrics."""
        if not self.history:
            return {"message": "No history yet"}
        
        pnls = [h.best_pnl for h in self.history]
        
        return {
            "iterations": len(self.history),
            "avg_pnl_first_10": np.mean(pnls[:10]) if len(pnls) >= 10 else np.mean(pnls),
            "avg_pnl_last_10": np.mean(pnls[-10:]),
            "improvement": (np.mean(pnls[-10:]) - np.mean(pnls[:10])) if len(pnls) >= 20 else 0.0,
            "best_methods": self.get_best_methods(),
            "method_popularity": self.get_method_popularity(),
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary."""
        return {
            "config": {
                "population_size": self.config.population_size,
                "max_methods_per_agent": self.config.max_methods_per_agent,
                "transfer_frequency": self.config.transfer_frequency,
            },
            "inventory_sizes": {
                "analyst": len(ANALYST_INVENTORY),
                "researcher": len(RESEARCHER_INVENTORY),
                "trader": len(TRADER_INVENTORY),
                "risk": len(RISK_INVENTORY),
            },
            "iteration": self.iteration,
            "total_pipelines_evaluated": len(self.all_results),
            "learning_progress": self.get_learning_progress(),
            "population_stats": {
                role.value: pop.get_population_stats()
                for role, pop in self.populations.items()
            },
        }


def create_selector_workflow(
    population_size: int = 5,
    max_methods: int = 3,
    **kwargs
) -> SelectorWorkflow:
    """Factory function to create a selector workflow."""
    config = SelectorWorkflowConfig(
        population_size=population_size,
        max_methods_per_agent=max_methods,
        **kwargs
    )
    return SelectorWorkflow(config)

