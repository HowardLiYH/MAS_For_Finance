"""Method Selector - Agents that dynamically select methods from inventory.

This module implements the core innovation of PopAgent: agents that learn
to SELECT which methods to use from a shared inventory, rather than being
locked into fixed strategies.

Key concepts:
- Inventory: Pool of available methods (10-15 per role)
- Selection: Each agent picks a subset of methods (e.g., 3 out of 15)
- Preference Learning: Agents learn which methods work via reinforcement
- Knowledge Transfer: Best agent's preferences inform others' selections

RL Enhancements (v0.7.0):
- Thompson Sampling: Bayesian exploration via Beta distributions
- Contextual Baselines: Per-regime reward normalization
- Multi-Step Returns: Temporal credit assignment with discounting
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
from collections import deque


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


@dataclass
class PendingReturn:
    """Pending return for multi-step calculation."""
    methods_used: List[str]
    market_context: Dict[str, Any]
    immediate_reward: float
    timestamp: datetime
    steps_remaining: int


class MethodSelector:
    """
    An agent that dynamically selects methods from an inventory.

    This is the core innovation of PopAgent:
    - Agents are not fixed to specific strategies
    - They LEARN which methods to select
    - Selection is context-aware (market conditions matter)
    - Knowledge transfer shares selection preferences

    RL Enhancements:
    - Thompson Sampling for Bayesian exploration
    - Contextual baselines for regime-aware learning
    - Multi-step returns for temporal credit assignment
    """

    def __init__(
        self,
        role: AgentRole,
        inventory: List[str],
        max_methods: int = 3,
        exploration_rate: float = 0.15,
        learning_rate: float = 0.1,
        # RL Enhancement parameters
        use_thompson_sampling: bool = True,
        gamma: float = 0.9,  # Discount factor for multi-step returns
        n_step: int = 3,  # Number of steps for multi-step returns
    ):
        self.id = str(uuid.uuid4())[:8]
        self.role = role
        self.inventory = inventory  # All available methods
        self.max_methods = max_methods  # How many to select
        self.exploration_rate = exploration_rate
        self.learning_rate = learning_rate

        # RL Enhancement parameters
        self.use_thompson_sampling = use_thompson_sampling
        self.gamma = gamma
        self.n_step = n_step

        # Learned preferences (method -> preference score)
        self.preferences: Dict[str, float] = {m: 0.0 for m in inventory}

        # UCB statistics (fallback if not using Thompson Sampling)
        self.method_counts: Dict[str, int] = {m: 0 for m in inventory}
        self.method_rewards: Dict[str, float] = {m: 0.0 for m in inventory}

        # ===========================================
        # THOMPSON SAMPLING: Beta distribution params
        # ===========================================
        # alpha = successes + 1, beta = failures + 1 (prior: Beta(1,1) = uniform)
        self.alpha: Dict[str, float] = {m: 1.0 for m in inventory}
        self.beta_param: Dict[str, float] = {m: 1.0 for m in inventory}

        # ===========================================
        # CONTEXTUAL BASELINES: Per-regime tracking
        # ===========================================
        # context_key -> running average reward
        self.context_baselines: Dict[str, float] = {}
        # context_key -> count (for incremental mean)
        self.context_counts: Dict[str, int] = {}

        # Context-aware preferences (context_key -> method -> bonus)
        self.context_preferences: Dict[str, Dict[str, float]] = {}

        # ===========================================
        # MULTI-STEP RETURNS: Pending rewards queue
        # ===========================================
        self.pending_returns: deque = deque(maxlen=100)

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
        Select methods from inventory using Thompson Sampling or UCB.

        Thompson Sampling: Sample from Beta distributions to balance
        exploration (uncertain methods have high variance) and
        exploitation (successful methods have high mean).

        Args:
            market_context: Current market conditions for context-aware selection
            force_exploration: If True, force exploration of less-used methods

        Returns:
            List of selected method names
        """
        context = market_context or {}
        context_key = self._context_to_key(context)

        scores = {}

        if self.use_thompson_sampling:
            # =========================================
            # THOMPSON SAMPLING: Sample from Beta dist
            # =========================================
            for method in self.inventory:
                # Sample from Beta distribution
                alpha = self.alpha.get(method, 1.0)
                beta = self.beta_param.get(method, 1.0)
                thompson_sample = np.random.beta(alpha, beta)

                # Add context bonus
                context_bonus = 0.0
                if context_key in self.context_preferences:
                    context_bonus = self.context_preferences[context_key].get(method, 0.0)

                # Combine: Thompson sample + context + small preference influence
                scores[method] = (
                    thompson_sample +
                    0.3 * context_bonus +
                    0.1 * self.preferences.get(method, 0.0)
                )
        else:
            # Fallback: UCB-based selection
            total_selections = max(1, self.total_selections)

            for method in self.inventory:
                base_pref = self.preferences.get(method, 0.0)
                count = max(1, self.method_counts.get(method, 0))
                ucb_bonus = np.sqrt(2 * np.log(total_selections) / count)

                context_bonus = 0.0
                if context_key in self.context_preferences:
                    context_bonus = self.context_preferences[context_key].get(method, 0.0)

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

        # Add to pending returns for multi-step calculation
        self.pending_returns.append(PendingReturn(
            methods_used=selected.copy(),
            market_context=context.copy(),
            immediate_reward=0.0,  # Will be filled when outcome arrives
            timestamp=datetime.now(),
            steps_remaining=self.n_step,
        ))

        return selected

    def update_from_outcome(self, outcome: SelectionOutcome) -> None:
        """
        Update preferences based on selection outcome.

        Enhancements:
        1. Thompson Sampling: Update Beta distribution parameters
        2. Contextual Baselines: Use per-regime baseline for advantage
        3. Multi-Step Returns: Process pending returns with discounting
        """
        context_key = self._context_to_key(outcome.market_context)

        # =========================================
        # CONTEXTUAL BASELINE: Per-regime baseline
        # =========================================
        if context_key not in self.context_baselines:
            self.context_baselines[context_key] = 0.0
            self.context_counts[context_key] = 0

        # Get context-specific baseline
        context_baseline = self.context_baselines[context_key]

        # Calculate advantage relative to context
        advantage = outcome.reward - context_baseline

        # Update context baseline (incremental mean)
        self.context_counts[context_key] += 1
        n = self.context_counts[context_key]
        self.context_baselines[context_key] = (
            context_baseline + (outcome.reward - context_baseline) / n
        )

        # =========================================
        # THOMPSON SAMPLING: Update Beta params
        # =========================================
        # Convert reward to success/failure (threshold at 0)
        is_success = outcome.reward > 0

        for method in outcome.methods_used:
            if is_success:
                # Increase alpha (successes)
                self.alpha[method] = self.alpha.get(method, 1.0) + 1.0
            else:
                # Increase beta (failures)
                self.beta_param[method] = self.beta_param.get(method, 1.0) + 1.0

        # Also update based on magnitude (soft update)
        # Higher reward = more evidence of success
        reward_magnitude = min(abs(outcome.reward) * 10, 2.0)  # Cap at 2
        for method in outcome.methods_used:
            if outcome.reward > 0:
                self.alpha[method] += reward_magnitude * 0.5
            else:
                self.beta_param[method] += reward_magnitude * 0.5

        # =========================================
        # PREFERENCE UPDATE: With contextual advantage
        # =========================================
        for method in outcome.methods_used:
            old_pref = self.preferences.get(method, 0.0)
            new_pref = old_pref + self.learning_rate * advantage
            self.preferences[method] = np.clip(new_pref, -2.0, 2.0)

            # Update method reward statistics
            old_reward = self.method_rewards.get(method, 0.0)
            count = self.method_counts.get(method, 1)
            self.method_rewards[method] = old_reward + (outcome.reward - old_reward) / count

        # Update context-specific preferences
        if context_key not in self.context_preferences:
            self.context_preferences[context_key] = {}

        for method in outcome.methods_used:
            old_ctx_pref = self.context_preferences[context_key].get(method, 0.0)
            new_ctx_pref = old_ctx_pref + self.learning_rate * 0.5 * advantage
            self.context_preferences[context_key][method] = np.clip(new_ctx_pref, -1.0, 1.0)

        # =========================================
        # MULTI-STEP RETURNS: Process pending queue
        # =========================================
        self._process_pending_returns(outcome.reward)

        self.outcome_history.append(outcome)

    def _process_pending_returns(self, current_reward: float) -> None:
        """
        Process pending returns for multi-step credit assignment.

        Each pending entry gets discounted future reward added:
        G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ...
        """
        # Update all pending returns with discounted current reward
        for pending in self.pending_returns:
            if pending.steps_remaining > 0:
                # Calculate discount factor for this step
                steps_passed = self.n_step - pending.steps_remaining
                discount = self.gamma ** steps_passed

                # Add discounted reward
                pending.immediate_reward += discount * current_reward
                pending.steps_remaining -= 1

                # If completed all steps, apply the full return
                if pending.steps_remaining == 0:
                    self._apply_multistep_update(pending)

    def _apply_multistep_update(self, pending: PendingReturn) -> None:
        """Apply the accumulated multi-step return to preferences."""
        context_key = self._context_to_key(pending.market_context)

        # Get baseline for this context
        baseline = self.context_baselines.get(context_key, 0.0)

        # Multi-step advantage
        multistep_advantage = pending.immediate_reward - baseline * self.n_step

        # Smaller learning rate for multi-step updates (already have immediate updates)
        multistep_lr = self.learning_rate * 0.3

        for method in pending.methods_used:
            old_pref = self.preferences.get(method, 0.0)
            new_pref = old_pref + multistep_lr * multistep_advantage
            self.preferences[method] = np.clip(new_pref, -2.0, 2.0)

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

    def get_thompson_stats(self) -> Dict[str, Dict[str, float]]:
        """Get Thompson Sampling statistics for each method."""
        stats = {}
        for method in self.inventory:
            alpha = self.alpha.get(method, 1.0)
            beta = self.beta_param.get(method, 1.0)
            # Expected value of Beta distribution
            expected = alpha / (alpha + beta)
            # Variance (uncertainty)
            variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
            stats[method] = {
                "alpha": alpha,
                "beta": beta,
                "expected_success_rate": expected,
                "uncertainty": np.sqrt(variance),
            }
        return stats

    def get_contextual_stats(self) -> Dict[str, Any]:
        """Get contextual baseline statistics."""
        return {
            "contexts_seen": list(self.context_baselines.keys()),
            "baselines": self.context_baselines.copy(),
            "counts": self.context_counts.copy(),
        }

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

        Also transfers Thompson Sampling parameters and contextual baselines.
        """
        # Transfer preferences
        for method in self.preferences:
            if method in other.preferences:
                old_pref = self.preferences[method]
                other_pref = other.preferences[method]
                self.preferences[method] = (1 - tau) * old_pref + tau * other_pref

        # Transfer Thompson Sampling parameters
        for method in self.alpha:
            if method in other.alpha:
                self.alpha[method] = (1 - tau) * self.alpha[method] + tau * other.alpha[method]
                self.beta_param[method] = (1 - tau) * self.beta_param[method] + tau * other.beta_param[method]

        # Transfer context preferences
        for context_key, other_ctx_prefs in other.context_preferences.items():
            if context_key not in self.context_preferences:
                self.context_preferences[context_key] = {}

            for method, other_pref in other_ctx_prefs.items():
                if method in self.preferences:
                    old_pref = self.context_preferences[context_key].get(method, 0.0)
                    self.context_preferences[context_key][method] = (
                        (1 - tau) * old_pref + tau * other_pref
                    )

        # Transfer contextual baselines
        for context_key, other_baseline in other.context_baselines.items():
            if context_key in self.context_baselines:
                self.context_baselines[context_key] = (
                    (1 - tau) * self.context_baselines[context_key] + tau * other_baseline
                )
            else:
                self.context_baselines[context_key] = other_baseline * tau

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
                    "thompson_expected": self.alpha[method] / (self.alpha[method] + self.beta_param[method]),
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
            "contexts_seen": len(self.context_baselines),
            "pending_returns": len(self.pending_returns),
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
            "alpha": self.alpha,
            "beta_param": self.beta_param,
            "context_baselines": self.context_baselines,
            "context_counts": self.context_counts,
            "gamma": self.gamma,
            "n_step": self.n_step,
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
            gamma=data.get("gamma", 0.9),
            n_step=data.get("n_step", 3),
        )
        agent.id = data["id"]
        agent.preferences = data["preferences"]
        agent.method_counts = data["method_counts"]
        agent.method_rewards = data["method_rewards"]
        agent.alpha = data.get("alpha", {m: 1.0 for m in data["inventory"]})
        agent.beta_param = data.get("beta_param", {m: 1.0 for m in data["inventory"]})
        agent.context_baselines = data.get("context_baselines", {})
        agent.context_counts = data.get("context_counts", {})
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
    # RL enhancement configs
    use_thompson_sampling: bool = True
    gamma: float = 0.9
    n_step: int = 3


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
                use_thompson_sampling=config.use_thompson_sampling,
                gamma=config.gamma,
                n_step=config.n_step,
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
            "rl_config": {
                "thompson_sampling": self.config.use_thompson_sampling,
                "gamma": self.config.gamma,
                "n_step": self.config.n_step,
            },
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
