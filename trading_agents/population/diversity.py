"""Diversity preservation mechanisms for population-based learning.

This module implements strategies to maintain population diversity,
preventing premature convergence to a single solution.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime

from .base import PopulationAgent, AgentPopulation


class DiversityMetric(ABC):
    """Base class for measuring population diversity."""

    @abstractmethod
    def calculate(self, agents: List[PopulationAgent]) -> float:
        """
        Calculate diversity score for a population.

        Returns:
            Diversity score between 0 (identical) and 1 (maximally diverse)
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the diversity metric."""
        pass


class ParameterDiversity(DiversityMetric):
    """
    Measure diversity based on parameter space distribution.

    Calculates the average pairwise distance between agent parameters,
    normalized to [0, 1].
    """

    @property
    def name(self) -> str:
        return "parameter_diversity"

    def calculate(self, agents: List[PopulationAgent]) -> float:
        if len(agents) < 2:
            return 1.0

        # Extract parameter vectors
        param_vectors = []
        for agent in agents:
            params = agent.get_parameters()
            vec = self._flatten_params(params)
            if vec:
                param_vectors.append(np.array(vec))

        if len(param_vectors) < 2:
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

    def _flatten_params(self, params: Dict[str, Any]) -> List[float]:
        """Flatten parameters to a list of floats."""
        vec = []
        for v in params.values():
            if isinstance(v, (int, float)):
                vec.append(float(v))
            elif isinstance(v, np.ndarray):
                vec.extend(v.flatten().tolist())
            elif isinstance(v, dict):
                vec.extend(self._flatten_params(v))
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, (int, float)):
                        vec.append(float(item))
        return vec


class BehavioralDiversity(DiversityMetric):
    """
    Measure diversity based on agent behaviors on reference inputs.

    Runs each agent on a set of reference inputs and measures
    the variance in their outputs.
    """

    def __init__(self, reference_inputs: Optional[List[Dict[str, Any]]] = None):
        self.reference_inputs = reference_inputs or []

    @property
    def name(self) -> str:
        return "behavioral_diversity"

    def set_reference_inputs(self, inputs: List[Dict[str, Any]]) -> None:
        """Set the reference inputs for behavioral evaluation."""
        self.reference_inputs = inputs

    def calculate(self, agents: List[PopulationAgent]) -> float:
        if len(agents) < 2 or not self.reference_inputs:
            return 1.0

        # Collect outputs for each agent
        all_outputs = []
        for agent in agents:
            agent_outputs = []
            for ref_input in self.reference_inputs:
                try:
                    output = agent.run(ref_input)
                    agent_outputs.append(self._output_to_vector(output))
                except Exception:
                    agent_outputs.append([])
            all_outputs.append(agent_outputs)

        # Calculate output variance
        diversities = []
        for input_idx in range(len(self.reference_inputs)):
            outputs_for_input = [
                ao[input_idx] for ao in all_outputs
                if input_idx < len(ao) and ao[input_idx]
            ]

            if len(outputs_for_input) >= 2:
                # Pad to same length
                max_len = max(len(o) for o in outputs_for_input)
                outputs_for_input = [
                    o + [0] * (max_len - len(o)) for o in outputs_for_input
                ]

                # Calculate variance
                output_array = np.array(outputs_for_input)
                variance = np.mean(np.var(output_array, axis=0))
                diversities.append(variance)

        if diversities:
            # Normalize to [0, 1] using sigmoid-like transformation
            avg_variance = np.mean(diversities)
            return 1 - np.exp(-avg_variance * 10)

        return 1.0

    def _output_to_vector(self, output: Dict[str, Any]) -> List[float]:
        """Convert output dict to a vector of floats."""
        vec = []
        for v in output.values():
            if isinstance(v, (int, float)):
                vec.append(float(v))
            elif isinstance(v, str):
                # Hash strings to numbers for comparison
                vec.append(hash(v) % 1000 / 1000.0)
        return vec


class OutputDiversity(DiversityMetric):
    """
    Measure diversity based on categorical outputs.

    For agents that produce discrete decisions (BUY/SELL/HOLD, LONG/SHORT),
    measures the entropy of the decision distribution.
    """

    def __init__(self, output_key: str = "recommendation"):
        self.output_key = output_key
        self.reference_inputs: List[Dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "output_diversity"

    def set_reference_inputs(self, inputs: List[Dict[str, Any]]) -> None:
        """Set the reference inputs."""
        self.reference_inputs = inputs

    def calculate(self, agents: List[PopulationAgent]) -> float:
        if len(agents) < 2 or not self.reference_inputs:
            return 1.0

        # Count decisions per input
        diversities = []

        for ref_input in self.reference_inputs:
            decisions = []
            for agent in agents:
                try:
                    output = agent.run(ref_input)
                    if self.output_key in output:
                        decisions.append(output[self.output_key])
                except Exception:
                    pass

            if decisions:
                # Calculate entropy of decisions
                unique, counts = np.unique(decisions, return_counts=True)
                probs = counts / len(decisions)
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                max_entropy = np.log(len(agents))  # Maximum possible entropy
                normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
                diversities.append(normalized_entropy)

        return np.mean(diversities) if diversities else 1.0


class CompositeDiversity(DiversityMetric):
    """
    Combine multiple diversity metrics.
    """

    def __init__(self, metrics: Optional[List[Tuple[DiversityMetric, float]]] = None):
        # List of (metric, weight) tuples
        self.metrics = metrics or [
            (ParameterDiversity(), 0.5),
            (BehavioralDiversity(), 0.5),
        ]

    @property
    def name(self) -> str:
        return "composite_diversity"

    def calculate(self, agents: List[PopulationAgent]) -> float:
        if not self.metrics:
            return 1.0

        total_weight = sum(w for _, w in self.metrics)
        weighted_sum = 0.0

        for metric, weight in self.metrics:
            diversity = metric.calculate(agents)
            weighted_sum += diversity * weight

        return weighted_sum / total_weight if total_weight > 0 else 1.0


@dataclass
class DiversityPreservationConfig:
    """Configuration for diversity preservation."""
    min_diversity: float = 0.2  # Minimum acceptable diversity
    target_diversity: float = 0.5  # Target diversity level
    mutation_rate: float = 0.3  # Base mutation rate
    mutation_strength: float = 0.1  # Base mutation strength
    check_frequency: int = 5  # Check diversity every N iterations
    preserve_elite: bool = True  # Don't mutate elite agents


class DiversityPreserver:
    """
    Maintains population diversity through various mechanisms.

    Strategies:
    1. Mutation: Add random noise to non-elite agents
    2. Respawning: Replace low-diversity agents with new random variants
    3. Crowding: Penalize agents that are too similar to others
    4. Speciation: Group similar agents and only compete within groups
    """

    def __init__(
        self,
        config: Optional[DiversityPreservationConfig] = None,
        metric: Optional[DiversityMetric] = None
    ):
        self.config = config or DiversityPreservationConfig()
        self.metric = metric or ParameterDiversity()
        self.history: List[Dict[str, Any]] = []

    def check_and_preserve(
        self,
        population: AgentPopulation,
        iteration: int
    ) -> Dict[str, Any]:
        """
        Check population diversity and take action if needed.

        Returns:
            Dictionary with preservation statistics
        """
        if iteration % self.config.check_frequency != 0:
            return {"action": "skipped", "iteration": iteration}

        current_diversity = self.metric.calculate(population.agents)

        stats = {
            "iteration": iteration,
            "current_diversity": current_diversity,
            "min_threshold": self.config.min_diversity,
            "target": self.config.target_diversity,
            "action": "none",
            "mutations": [],
        }

        if current_diversity < self.config.min_diversity:
            # Diversity too low, need to take action
            stats["action"] = "mutate"

            # Get elite agents (to preserve)
            elite_ids = set()
            if self.config.preserve_elite:
                elite_ids = {a.id for a in population.get_elite()}

            # Calculate mutation strength based on diversity deficit
            deficit = self.config.target_diversity - current_diversity
            mutation_strength = self.config.mutation_strength * (1 + deficit * 2)

            for agent in population.agents:
                if agent.id not in elite_ids:
                    # Apply mutation
                    agent.mutate(
                        mutation_rate=self.config.mutation_rate,
                        mutation_strength=mutation_strength
                    )
                    stats["mutations"].append({
                        "agent_id": agent.id,
                        "mutation_strength": mutation_strength,
                    })

            # Recalculate diversity after mutations
            stats["new_diversity"] = self.metric.calculate(population.agents)

        self.history.append(stats)
        return stats

    def calculate_diversity_bonus(
        self,
        agent: PopulationAgent,
        population: AgentPopulation
    ) -> float:
        """
        Calculate a diversity bonus for an agent's score.

        Agents that are more different from the population mean
        receive a higher bonus.
        """
        if len(population.agents) < 2:
            return 0.0

        agent_params = agent.get_parameters()
        agent_vec = self._params_to_vector(agent_params)

        if not agent_vec:
            return 0.0

        # Calculate mean parameter vector
        all_vecs = []
        for a in population.agents:
            vec = self._params_to_vector(a.get_parameters())
            if vec:
                all_vecs.append(vec)

        if not all_vecs:
            return 0.0

        # Pad to same length
        max_len = max(len(v) for v in all_vecs + [agent_vec])
        all_vecs = [np.pad(v, (0, max_len - len(v))) for v in all_vecs]
        agent_vec = np.pad(agent_vec, (0, max_len - len(agent_vec)))

        mean_vec = np.mean(all_vecs, axis=0)

        # Distance from mean
        distance = np.linalg.norm(agent_vec - mean_vec)
        max_distance = np.max([np.linalg.norm(v - mean_vec) for v in all_vecs])

        if max_distance > 0:
            return distance / max_distance * 0.1  # Max 10% bonus

        return 0.0

    def _params_to_vector(self, params: Dict[str, Any]) -> np.ndarray:
        """Convert parameters to numpy vector."""
        vec = []
        for v in params.values():
            if isinstance(v, (int, float)):
                vec.append(float(v))
            elif isinstance(v, np.ndarray):
                vec.extend(v.flatten().tolist())
        return np.array(vec) if vec else np.array([])

    def get_diversity_report(self) -> Dict[str, Any]:
        """Generate a report on diversity history."""
        if not self.history:
            return {"message": "No diversity history available"}

        diversities = [h["current_diversity"] for h in self.history]
        actions = [h["action"] for h in self.history]

        return {
            "num_checks": len(self.history),
            "avg_diversity": np.mean(diversities),
            "min_diversity": np.min(diversities),
            "max_diversity": np.max(diversities),
            "mutation_count": sum(1 for a in actions if a == "mutate"),
            "history": self.history[-10:],  # Last 10 entries
        }


def create_diversity_metric(name: str, **kwargs) -> DiversityMetric:
    """Factory function to create diversity metrics."""
    metrics = {
        "parameter": ParameterDiversity,
        "behavioral": BehavioralDiversity,
        "output": OutputDiversity,
        "composite": CompositeDiversity,
    }

    if name not in metrics:
        raise ValueError(f"Unknown diversity metric: {name}. Available: {list(metrics.keys())}")

    return metrics[name](**kwargs)
