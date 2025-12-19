"""Base classes for population-based multi-agent learning."""
from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Generic, TypeVar
from enum import Enum
import numpy as np
from datetime import datetime


class AgentRole(Enum):
    """Roles that agents can take in the trading pipeline."""
    ANALYST = "analyst"
    RESEARCHER = "researcher"
    TRADER = "trader"
    RISK = "risk"


@dataclass
class PopulationConfig:
    """Configuration for a population of agents."""
    role: AgentRole
    size: int = 5
    transfer_frequency: int = 10  # Transfer knowledge every N iterations
    transfer_tau: float = 0.1  # Soft update coefficient
    diversity_weight: float = 0.1  # Weight for diversity bonus in scoring
    min_diversity: float = 0.2  # Minimum diversity to maintain
    elite_fraction: float = 0.2  # Top fraction considered "elite"
    exploration_rate: float = 0.1  # Probability of random action for exploration


@dataclass
class AgentScore:
    """Score for an individual agent."""
    agent_id: str
    individual_score: float  # Agent's own performance
    relative_rank: float  # Rank within population (0-1, higher is better)
    pipeline_contribution: float  # Contribution to full pipeline
    diversity_bonus: float  # Bonus for being different
    total_score: float  # Weighted combination
    iteration: int
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "individual_score": self.individual_score,
            "relative_rank": self.relative_rank,
            "pipeline_contribution": self.pipeline_contribution,
            "diversity_bonus": self.diversity_bonus,
            "total_score": self.total_score,
            "iteration": self.iteration,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PopulationState:
    """Current state of a population."""
    role: AgentRole
    size: int
    iteration: int
    best_agent_id: str
    avg_score: float
    score_std: float
    diversity: float
    agent_scores: Dict[str, AgentScore]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role.value,
            "size": self.size,
            "iteration": self.iteration,
            "best_agent_id": self.best_agent_id,
            "avg_score": self.avg_score,
            "score_std": self.score_std,
            "diversity": self.diversity,
            "agent_scores": {k: v.to_dict() for k, v in self.agent_scores.items()},
        }


T = TypeVar("T", bound="PopulationAgent")


class PopulationAgent(ABC):
    """Base class for agents that can participate in population-based learning."""

    def __init__(self, variant_name: str, variant_config: Dict[str, Any]):
        self.id = str(uuid.uuid4())[:8]
        self.variant_name = variant_name
        self.variant_config = variant_config
        self.parameters: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []
        self.created_at = datetime.now()
        self.last_updated = datetime.now()

    @property
    @abstractmethod
    def role(self) -> AgentRole:
        """Return the role of this agent."""
        pass

    @abstractmethod
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent's main logic."""
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get the agent's learnable parameters."""
        pass

    @abstractmethod
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """Set the agent's learnable parameters."""
        pass

    def soft_update_from(self, other: "PopulationAgent", tau: float = 0.1) -> None:
        """Soft update this agent's parameters toward another agent's parameters.

        new_params = (1 - tau) * self.params + tau * other.params
        """
        other_params = other.get_parameters()
        self_params = self.get_parameters()

        new_params = {}
        for key in self_params:
            if key in other_params:
                self_val = self_params[key]
                other_val = other_params[key]

                if isinstance(self_val, (int, float)) and isinstance(other_val, (int, float)):
                    new_params[key] = (1 - tau) * self_val + tau * other_val
                elif isinstance(self_val, np.ndarray) and isinstance(other_val, np.ndarray):
                    new_params[key] = (1 - tau) * self_val + tau * other_val
                elif isinstance(self_val, list) and isinstance(other_val, list):
                    # For lists, probabilistically adopt elements
                    new_params[key] = [
                        other_val[i] if np.random.random() < tau else self_val[i]
                        for i in range(min(len(self_val), len(other_val)))
                    ]
                else:
                    # For non-numeric, probabilistically adopt
                    new_params[key] = other_val if np.random.random() < tau else self_val
            else:
                new_params[key] = self_params[key]

        self.set_parameters(new_params)
        self.last_updated = datetime.now()
        self.history.append({
            "action": "soft_update",
            "from_agent": other.id,
            "tau": tau,
            "timestamp": datetime.now().isoformat(),
        })

    def clone(self: T) -> T:
        """Create a copy of this agent with a new ID."""
        new_agent = self.__class__(self.variant_name, self.variant_config.copy())
        new_agent.set_parameters(self.get_parameters().copy())
        new_agent.history.append({
            "action": "cloned_from",
            "parent_id": self.id,
            "timestamp": datetime.now().isoformat(),
        })
        return new_agent

    def mutate(self, mutation_rate: float = 0.1, mutation_strength: float = 0.1) -> None:
        """Apply random mutations to parameters for exploration."""
        params = self.get_parameters()
        mutated_params = {}

        for key, value in params.items():
            if np.random.random() < mutation_rate:
                if isinstance(value, (int, float)):
                    noise = np.random.normal(0, mutation_strength * abs(value) + 0.01)
                    mutated_params[key] = value + noise
                elif isinstance(value, np.ndarray):
                    noise = np.random.normal(0, mutation_strength, size=value.shape)
                    mutated_params[key] = value + noise * np.abs(value).mean()
                else:
                    mutated_params[key] = value
            else:
                mutated_params[key] = value

        self.set_parameters(mutated_params)
        self.history.append({
            "action": "mutation",
            "mutation_rate": mutation_rate,
            "mutation_strength": mutation_strength,
            "timestamp": datetime.now().isoformat(),
        })

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id}, variant={self.variant_name})"


class AgentPopulation(Generic[T]):
    """A population of agents of the same role that evolve together."""

    def __init__(self, config: PopulationConfig, agents: Optional[List[T]] = None):
        self.config = config
        self.agents: List[T] = agents or []
        self.scores: Dict[str, AgentScore] = {}
        self.iteration = 0
        self.history: List[PopulationState] = []
        self._best_agent_id: Optional[str] = None

    @property
    def size(self) -> int:
        return len(self.agents)

    @property
    def role(self) -> AgentRole:
        return self.config.role

    def add_agent(self, agent: T) -> None:
        """Add an agent to the population."""
        if agent.role != self.config.role:
            raise ValueError(f"Agent role {agent.role} doesn't match population role {self.config.role}")
        self.agents.append(agent)

    def remove_agent(self, agent_id: str) -> Optional[T]:
        """Remove an agent from the population by ID."""
        for i, agent in enumerate(self.agents):
            if agent.id == agent_id:
                return self.agents.pop(i)
        return None

    def get_agent(self, agent_id: str) -> Optional[T]:
        """Get an agent by ID."""
        for agent in self.agents:
            if agent.id == agent_id:
                return agent
        return None

    def get_best(self) -> Optional[T]:
        """Return the current best performing agent."""
        if not self.scores:
            return self.agents[0] if self.agents else None

        best_id = max(self.scores, key=lambda x: self.scores[x].total_score)
        self._best_agent_id = best_id
        return self.get_agent(best_id)

    def get_elite(self) -> List[T]:
        """Return the top-performing agents (elite fraction)."""
        if not self.scores:
            return self.agents[:max(1, int(len(self.agents) * self.config.elite_fraction))]

        sorted_ids = sorted(self.scores, key=lambda x: self.scores[x].total_score, reverse=True)
        elite_count = max(1, int(len(sorted_ids) * self.config.elite_fraction))
        return [self.get_agent(aid) for aid in sorted_ids[:elite_count] if self.get_agent(aid)]

    def update_scores(self, new_scores: Dict[str, AgentScore]) -> None:
        """Update agent scores."""
        self.scores.update(new_scores)
        self.iteration += 1

        # Record state
        self._record_state()

    def _record_state(self) -> None:
        """Record current population state for history."""
        if not self.scores:
            return

        scores_list = [s.total_score for s in self.scores.values()]
        best = self.get_best()

        state = PopulationState(
            role=self.config.role,
            size=self.size,
            iteration=self.iteration,
            best_agent_id=best.id if best else "",
            avg_score=np.mean(scores_list) if scores_list else 0.0,
            score_std=np.std(scores_list) if scores_list else 0.0,
            diversity=self.calculate_diversity(),
            agent_scores=self.scores.copy(),
        )
        self.history.append(state)

    def calculate_diversity(self) -> float:
        """Calculate the diversity of the population based on parameters."""
        if len(self.agents) < 2:
            return 1.0

        # Collect all parameter vectors
        param_vectors = []
        for agent in self.agents:
            params = agent.get_parameters()
            # Flatten to vector
            vec = []
            for v in params.values():
                if isinstance(v, (int, float)):
                    vec.append(float(v))
                elif isinstance(v, np.ndarray):
                    vec.extend(v.flatten().tolist())
            if vec:
                param_vectors.append(np.array(vec))

        if not param_vectors or len(param_vectors) < 2:
            return 1.0

        # Pad to same length
        max_len = max(len(v) for v in param_vectors)
        param_vectors = [np.pad(v, (0, max_len - len(v))) for v in param_vectors]

        # Calculate pairwise distances
        distances = []
        for i in range(len(param_vectors)):
            for j in range(i + 1, len(param_vectors)):
                dist = np.linalg.norm(param_vectors[i] - param_vectors[j])
                distances.append(dist)

        # Normalize
        if distances:
            max_dist = max(distances) if max(distances) > 0 else 1.0
            return np.mean(distances) / max_dist
        return 1.0

    def transfer_knowledge(self, strategy: "KnowledgeTransferStrategy" = None) -> None:
        """Transfer knowledge from best performer to others."""
        best = self.get_best()
        if not best:
            return

        if strategy:
            strategy.transfer(best, self.agents, self.config.transfer_tau)
        else:
            # Default: soft update
            for agent in self.agents:
                if agent.id != best.id:
                    agent.soft_update_from(best, self.config.transfer_tau)

    def should_transfer(self) -> bool:
        """Check if it's time to transfer knowledge."""
        return self.iteration % self.config.transfer_frequency == 0

    def ensure_diversity(self, min_diversity: Optional[float] = None) -> None:
        """Ensure population maintains minimum diversity through mutation."""
        target_diversity = min_diversity or self.config.min_diversity
        current_diversity = self.calculate_diversity()

        if current_diversity < target_diversity:
            # Mutate non-elite agents to increase diversity
            elite_ids = {a.id for a in self.get_elite()}
            for agent in self.agents:
                if agent.id not in elite_ids:
                    mutation_strength = (target_diversity - current_diversity) * 0.5
                    agent.mutate(mutation_rate=0.3, mutation_strength=mutation_strength)

    def get_state(self) -> PopulationState:
        """Get current population state."""
        if self.history:
            return self.history[-1]

        return PopulationState(
            role=self.config.role,
            size=self.size,
            iteration=self.iteration,
            best_agent_id=self.get_best().id if self.get_best() else "",
            avg_score=0.0,
            score_std=0.0,
            diversity=self.calculate_diversity(),
            agent_scores=self.scores,
        )

    def __repr__(self) -> str:
        return f"AgentPopulation(role={self.config.role.value}, size={self.size}, iteration={self.iteration})"


# Forward declaration for type hints
class KnowledgeTransferStrategy(ABC):
    """Base class for knowledge transfer strategies."""

    @abstractmethod
    def transfer(self, source: PopulationAgent, targets: List[PopulationAgent], tau: float) -> None:
        """Transfer knowledge from source to targets."""
        pass
