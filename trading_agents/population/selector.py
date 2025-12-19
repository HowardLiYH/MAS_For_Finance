"""Method Selector - Agents that dynamically select methods from inventory.

This module implements the core innovation of PopAgent: agents that learn
to SELECT which methods to use from a shared inventory, rather than being
locked into fixed strategies.

Key concepts:
- Inventory: Pool of available methods (10-15 per role)
- Selection: Each agent picks a subset of methods (e.g., 3 out of 15)
- Preference Learning: Agents learn which methods work via reinforcement
- Knowledge Transfer: Best agent's preferences inform others' selections
"""
from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable
from enum import Enum
import numpy as np
from datetime import datetime
import random


class AgentRole(Enum):
    """Roles that agents can take in the trading pipeline."""
    ANALYST = "analyst"
    RESEARCHER = "researcher"
    TRADER = "trader"
    RISK = "risk"


@dataclass
class MethodInfo:
    """Information about a method in the inventory."""
    name: str
    category: str  # e.g., "technical", "statistical", "ml"
    description: str
    compute_cost: float = 1.0  # relative computational cost
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "compute_cost": self.compute_cost,
            "parameters": self.parameters,
        }


@dataclass
class SelectionState:
    """Current selection state of an agent."""
    selected_methods: List[str]
    selection_time: datetime
    market_context: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_methods": self.selected_methods,
            "selection_time": self.selection_time.isoformat(),
            "market_context": self.market_context,
        }


@dataclass
class SelectionOutcome:
    """Outcome of a selection for learning."""
    methods_used: List[str]
    reward: float  # PnL or other performance metric
    market_context: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


class MethodSelector:
    """
    An agent that dynamically selects methods from an inventory.
    
    This is the core innovation of PopAgent:
    - Agents are not fixed to specific strategies
    - They LEARN which methods to select
    - Selection is context-aware (market conditions matter)
    - Knowledge transfer shares selection preferences
    """
    
    def __init__(
        self,
        role: AgentRole,
        inventory: List[str],
        max_methods: int = 3,
        exploration_rate: float = 0.15,
        learning_rate: float = 0.1,
    ):
        self.id = str(uuid.uuid4())[:8]
        self.role = role
        self.inventory = inventory  # All available methods
        self.max_methods = max_methods  # How many to select
        self.exploration_rate = exploration_rate
        self.learning_rate = learning_rate
        
        # Learned preferences (method -> preference score)
        self.preferences: Dict[str, float] = {m: 0.0 for m in inventory}
        
        # UCB statistics
        self.method_counts: Dict[str, int] = {m: 0 for m in inventory}
        self.method_rewards: Dict[str, float] = {m: 0.0 for m in inventory}
        
        # Context-aware preferences (context_key -> method -> bonus)
        self.context_preferences: Dict[str, Dict[str, float]] = {}
        
        # History
        self.selection_history: List[SelectionState] = []
        self.outcome_history: List[SelectionOutcome] = []
        
        # Current selection
        self.current_selection: List[str] = []
        
        # Metadata
        self.created_at = datetime.now()
        self.total_selections = 0
    
    def select_methods(
        self,
        market_context: Optional[Dict[str, Any]] = None,
        force_exploration: bool = False,
    ) -> List[str]:
        """
        Select methods from inventory using UCB + context-aware preferences.
        
        Args:
            market_context: Current market conditions for context-aware selection
            force_exploration: If True, force exploration of less-used methods
            
        Returns:
            List of selected method names
        """
        context = market_context or {}
        context_key = self._context_to_key(context)
        
        # Calculate scores for each method
        scores = {}
        total_selections = max(1, self.total_selections)
        
        for method in self.inventory:
            # Base preference (learned)
            base_pref = self.preferences.get(method, 0.0)
            
            # UCB exploration bonus
            count = max(1, self.method_counts.get(method, 0))
            ucb_bonus = np.sqrt(2 * np.log(total_selections) / count)
            
            # Context-specific bonus
            context_bonus = 0.0
            if context_key in self.context_preferences:
                context_bonus = self.context_preferences[context_key].get(method, 0.0)
            
            # Random exploration
            if force_exploration or random.random() < self.exploration_rate:
                exploration_noise = random.gauss(0, 0.5)
            else:
                exploration_noise = 0.0
            
            scores[method] = base_pref + 0.5 * ucb_bonus + context_bonus + exploration_noise
        
        # Select top-k methods
        sorted_methods = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected = [m for m, _ in sorted_methods[:self.max_methods]]
        
        # Record selection
        self.current_selection = selected
        self.total_selections += 1
        
        for method in selected:
            self.method_counts[method] = self.method_counts.get(method, 0) + 1
        
        # Save to history
        self.selection_history.append(SelectionState(
            selected_methods=selected.copy(),
            selection_time=datetime.now(),
            market_context=context.copy(),
        ))
        
        return selected
    
    def update_from_outcome(self, outcome: SelectionOutcome) -> None:
        """
        Update preferences based on selection outcome.
        
        Uses a simple reinforcement learning update:
        preference[method] += learning_rate * (reward - baseline)
        """
        # Baseline: average reward
        if self.outcome_history:
            baseline = np.mean([o.reward for o in self.outcome_history[-20:]])
        else:
            baseline = 0.0
        
        advantage = outcome.reward - baseline
        
        # Update preferences for methods that were used
        for method in outcome.methods_used:
            old_pref = self.preferences.get(method, 0.0)
            new_pref = old_pref + self.learning_rate * advantage
            self.preferences[method] = np.clip(new_pref, -2.0, 2.0)
            
            # Update method reward statistics
            old_reward = self.method_rewards.get(method, 0.0)
            count = self.method_counts.get(method, 1)
            # Incremental mean update
            self.method_rewards[method] = old_reward + (outcome.reward - old_reward) / count
        
        # Update context-specific preferences
        context_key = self._context_to_key(outcome.market_context)
        if context_key not in self.context_preferences:
            self.context_preferences[context_key] = {}
        
        for method in outcome.methods_used:
            old_ctx_pref = self.context_preferences[context_key].get(method, 0.0)
            new_ctx_pref = old_ctx_pref + self.learning_rate * 0.5 * advantage
            self.context_preferences[context_key][method] = np.clip(new_ctx_pref, -1.0, 1.0)
        
        self.outcome_history.append(outcome)
    
    def _context_to_key(self, context: Dict[str, Any]) -> str:
        """Convert market context to a discrete key for context-aware learning."""
        if not context:
            return "default"
        
        # Discretize key context features
        parts = []
        
        # Trend
        trend = context.get("trend", "neutral")
        parts.append(f"trend:{trend}")
        
        # Volatility regime
        vol = context.get("volatility", 0.5)
        if vol < 0.3:
            parts.append("vol:low")
        elif vol > 0.7:
            parts.append("vol:high")
        else:
            parts.append("vol:mid")
        
        # Market regime
        regime = context.get("regime", "normal")
        parts.append(f"regime:{regime}")
        
        return "|".join(parts)
    
    def get_preferences(self) -> Dict[str, float]:
        """Get current method preferences."""
        return self.preferences.copy()
    
    def set_preferences(self, preferences: Dict[str, float]) -> None:
        """Set method preferences (for knowledge transfer)."""
        for method, pref in preferences.items():
            if method in self.preferences:
                self.preferences[method] = pref
    
    def soft_update_from(self, other: "MethodSelector", tau: float = 0.1) -> None:
        """
        Soft update preferences from another agent (knowledge transfer).
        
        new_pref = (1 - tau) * self.pref + tau * other.pref
        """
        for method in self.preferences:
            if method in other.preferences:
                old_pref = self.preferences[method]
                other_pref = other.preferences[method]
                self.preferences[method] = (1 - tau) * old_pref + tau * other_pref
        
        # Also transfer context preferences
        for context_key, other_ctx_prefs in other.context_preferences.items():
            if context_key not in self.context_preferences:
                self.context_preferences[context_key] = {}
            
            for method, other_pref in other_ctx_prefs.items():
                if method in self.preferences:
                    old_pref = self.context_preferences[context_key].get(method, 0.0)
                    self.context_preferences[context_key][method] = (
                        (1 - tau) * old_pref + tau * other_pref
                    )
    
    def get_selection_stats(self) -> Dict[str, Any]:
        """Get statistics about method selection."""
        if not self.outcome_history:
            return {"message": "No outcomes yet"}
        
        # Method performance
        method_perf = {}
        for method in self.inventory:
            outcomes = [o for o in self.outcome_history if method in o.methods_used]
            if outcomes:
                method_perf[method] = {
                    "count": len(outcomes),
                    "avg_reward": np.mean([o.reward for o in outcomes]),
                    "preference": self.preferences.get(method, 0.0),
                }
        
        return {
            "total_selections": self.total_selections,
            "total_outcomes": len(self.outcome_history),
            "avg_reward": np.mean([o.reward for o in self.outcome_history]),
            "method_performance": method_perf,
            "current_selection": self.current_selection,
            "top_preferences": sorted(
                self.preferences.items(), key=lambda x: x[1], reverse=True
            )[:5],
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize agent state."""
        return {
            "id": self.id,
            "role": self.role.value,
            "inventory": self.inventory,
            "max_methods": self.max_methods,
            "preferences": self.preferences,
            "method_counts": self.method_counts,
            "method_rewards": self.method_rewards,
            "current_selection": self.current_selection,
            "total_selections": self.total_selections,
            "created_at": self.created_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MethodSelector":
        """Deserialize agent state."""
        agent = cls(
            role=AgentRole(data["role"]),
            inventory=data["inventory"],
            max_methods=data["max_methods"],
        )
        agent.id = data["id"]
        agent.preferences = data["preferences"]
        agent.method_counts = data["method_counts"]
        agent.method_rewards = data["method_rewards"]
        agent.current_selection = data["current_selection"]
        agent.total_selections = data["total_selections"]
        return agent
    
    def __repr__(self) -> str:
        return f"MethodSelector(id={self.id}, role={self.role.value}, selection={self.current_selection})"


@dataclass
class SelectorPopulationConfig:
    """Configuration for a population of method selectors."""
    role: AgentRole
    inventory: List[str]
    population_size: int = 5
    max_methods_per_agent: int = 3
    exploration_rate: float = 0.15
    learning_rate: float = 0.1
    transfer_frequency: int = 10
    transfer_tau: float = 0.1
    diversity_weight: float = 0.1


class SelectorPopulation:
    """
    A population of MethodSelector agents for a single role.
    
    Each agent learns to select methods independently, but knowledge
    is transferred from best performers to others periodically.
    """
    
    def __init__(self, config: SelectorPopulationConfig):
        self.config = config
        self.agents: List[MethodSelector] = []
        self.scores: Dict[str, float] = {}
        self.iteration = 0
        
        # Initialize population
        for _ in range(config.population_size):
            agent = MethodSelector(
                role=config.role,
                inventory=config.inventory,
                max_methods=config.max_methods_per_agent,
                exploration_rate=config.exploration_rate,
                learning_rate=config.learning_rate,
            )
            self.agents.append(agent)
    
    @property
    def size(self) -> int:
        return len(self.agents)
    
    def get_agent(self, agent_id: str) -> Optional[MethodSelector]:
        """Get agent by ID."""
        for agent in self.agents:
            if agent.id == agent_id:
                return agent
        return None
    
    def get_best(self) -> Optional[MethodSelector]:
        """Get the best-performing agent."""
        if not self.scores:
            return self.agents[0] if self.agents else None
        
        best_id = max(self.scores, key=lambda x: self.scores[x])
        return self.get_agent(best_id)
    
    def update_scores(self, new_scores: Dict[str, float]) -> None:
        """Update agent scores."""
        self.scores.update(new_scores)
        self.iteration += 1
    
    def transfer_knowledge(self, tau: Optional[float] = None) -> None:
        """Transfer selection preferences from best to others."""
        best = self.get_best()
        if not best:
            return
        
        transfer_tau = tau or self.config.transfer_tau
        
        for agent in self.agents:
            if agent.id != best.id:
                agent.soft_update_from(best, transfer_tau)
    
    def should_transfer(self) -> bool:
        """Check if it's time for knowledge transfer."""
        return self.iteration > 0 and self.iteration % self.config.transfer_frequency == 0
    
    def calculate_selection_diversity(self) -> float:
        """
        Calculate how diverse the method selections are across agents.
        
        High diversity = agents selecting different methods
        Low diversity = all agents selecting same methods
        """
        if len(self.agents) < 2:
            return 1.0
        
        # Get all current selections
        selections = [set(a.current_selection) for a in self.agents if a.current_selection]
        
        if not selections:
            return 1.0
        
        # Calculate pairwise Jaccard distance
        distances = []
        for i in range(len(selections)):
            for j in range(i + 1, len(selections)):
                intersection = len(selections[i] & selections[j])
                union = len(selections[i] | selections[j])
                if union > 0:
                    jaccard = intersection / union
                    distances.append(1 - jaccard)  # Distance = 1 - similarity
        
        return np.mean(distances) if distances else 1.0
    
    def ensure_diversity(self) -> None:
        """Ensure population maintains diverse selections."""
        diversity = self.calculate_selection_diversity()
        
        if diversity < 0.3:
            # Force exploration for non-best agents
            best_id = self.get_best().id if self.get_best() else None
            for agent in self.agents:
                if agent.id != best_id:
                    agent.exploration_rate = min(0.4, agent.exploration_rate * 1.5)
        else:
            # Reset exploration rates
            for agent in self.agents:
                agent.exploration_rate = self.config.exploration_rate
    
    def get_population_stats(self) -> Dict[str, Any]:
        """Get population statistics."""
        return {
            "role": self.config.role.value,
            "size": self.size,
            "iteration": self.iteration,
            "selection_diversity": self.calculate_selection_diversity(),
            "avg_score": np.mean(list(self.scores.values())) if self.scores else 0.0,
            "best_agent_id": self.get_best().id if self.get_best() else None,
            "method_usage": self._get_method_usage(),
        }
    
    def _get_method_usage(self) -> Dict[str, int]:
        """Count how often each method is currently selected."""
        usage = {}
        for agent in self.agents:
            for method in agent.current_selection:
                usage[method] = usage.get(method, 0) + 1
        return usage
    
    def __repr__(self) -> str:
        return f"SelectorPopulation(role={self.config.role.value}, size={self.size})"

