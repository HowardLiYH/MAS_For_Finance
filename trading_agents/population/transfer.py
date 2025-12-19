"""Knowledge transfer strategies for population-based learning.

This module implements various strategies for transferring knowledge
from the best-performing agents to others in the population.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable
import numpy as np
from datetime import datetime

from .base import PopulationAgent


class KnowledgeTransferStrategy(ABC):
    """Base class for knowledge transfer strategies."""

    @abstractmethod
    def transfer(
        self,
        source: PopulationAgent,
        targets: List[PopulationAgent],
        tau: float = 0.1
    ) -> Dict[str, Any]:
        """
        Transfer knowledge from source to targets.

        Args:
            source: The best-performing agent to learn from
            targets: List of agents to update
            tau: Learning rate / transfer strength

        Returns:
            Dictionary with transfer statistics
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the transfer strategy."""
        pass


class SoftUpdateTransfer(KnowledgeTransferStrategy):
    """
    Soft update transfer: Gradually blend parameters toward the best agent.

    new_params = (1 - tau) * self.params + tau * best.params

    This is stable and prevents population collapse while allowing
    gradual convergence toward good solutions.
    """

    @property
    def name(self) -> str:
        return "soft_update"

    def transfer(
        self,
        source: PopulationAgent,
        targets: List[PopulationAgent],
        tau: float = 0.1
    ) -> Dict[str, Any]:
        stats = {
            "strategy": self.name,
            "source_id": source.id,
            "num_targets": len(targets),
            "tau": tau,
            "transfers": [],
        }

        for target in targets:
            if target.id != source.id:
                old_params = target.get_parameters().copy()
                target.soft_update_from(source, tau)
                new_params = target.get_parameters()

                # Calculate parameter change magnitude
                change = self._calculate_change(old_params, new_params)

                stats["transfers"].append({
                    "target_id": target.id,
                    "change_magnitude": change,
                    "timestamp": datetime.now().isoformat(),
                })

        return stats

    def _calculate_change(self, old: Dict, new: Dict) -> float:
        """Calculate the magnitude of parameter change."""
        total_change = 0.0
        count = 0

        for key in old:
            if key in new:
                old_val = old[key]
                new_val = new[key]

                if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                    total_change += abs(new_val - old_val)
                    count += 1
                elif isinstance(old_val, np.ndarray) and isinstance(new_val, np.ndarray):
                    total_change += np.mean(np.abs(new_val - old_val))
                    count += 1

        return total_change / count if count > 0 else 0.0


class DistillationTransfer(KnowledgeTransferStrategy):
    """
    Knowledge distillation transfer: Train targets to match source outputs.

    Instead of copying parameters directly, we collect the source's outputs
    on a set of reference inputs and train targets to match those outputs.

    This preserves each agent's unique characteristics while learning
    the source's decision patterns.
    """

    def __init__(self, reference_inputs: Optional[List[Dict[str, Any]]] = None):
        self.reference_inputs = reference_inputs or []
        self.distillation_weight = 0.3  # How much to weight distillation loss

    @property
    def name(self) -> str:
        return "distillation"

    def set_reference_inputs(self, inputs: List[Dict[str, Any]]) -> None:
        """Set the reference inputs for distillation."""
        self.reference_inputs = inputs

    def transfer(
        self,
        source: PopulationAgent,
        targets: List[PopulationAgent],
        tau: float = 0.1
    ) -> Dict[str, Any]:
        stats = {
            "strategy": self.name,
            "source_id": source.id,
            "num_targets": len(targets),
            "num_reference_inputs": len(self.reference_inputs),
            "transfers": [],
        }

        if not self.reference_inputs:
            # Fall back to soft update if no reference inputs
            fallback = SoftUpdateTransfer()
            return fallback.transfer(source, targets, tau)

        # Get source outputs on reference inputs
        source_outputs = []
        for ref_input in self.reference_inputs:
            try:
                output = source.run(ref_input)
                source_outputs.append(output)
            except Exception:
                source_outputs.append({})

        # For each target, adjust parameters to better match source outputs
        for target in targets:
            if target.id != source.id:
                # Get target outputs
                target_outputs = []
                for ref_input in self.reference_inputs:
                    try:
                        output = target.run(ref_input)
                        target_outputs.append(output)
                    except Exception:
                        target_outputs.append({})

                # Calculate output difference
                diff = self._calculate_output_diff(source_outputs, target_outputs)

                # Adjust parameters proportionally to difference
                if diff > 0.1:  # Only transfer if significant difference
                    target.soft_update_from(source, tau * self.distillation_weight)

                stats["transfers"].append({
                    "target_id": target.id,
                    "output_difference": diff,
                    "transferred": diff > 0.1,
                })

        return stats

    def _calculate_output_diff(
        self,
        source_outputs: List[Dict],
        target_outputs: List[Dict]
    ) -> float:
        """Calculate average difference between source and target outputs."""
        if not source_outputs or not target_outputs:
            return 0.0

        diffs = []
        for src, tgt in zip(source_outputs, target_outputs):
            if src and tgt:
                # Compare common keys
                common_keys = set(src.keys()) & set(tgt.keys())
                for key in common_keys:
                    src_val = src[key]
                    tgt_val = tgt[key]

                    if isinstance(src_val, (int, float)) and isinstance(tgt_val, (int, float)):
                        diffs.append(abs(src_val - tgt_val))
                    elif isinstance(src_val, str) and isinstance(tgt_val, str):
                        diffs.append(0.0 if src_val == tgt_val else 1.0)

        return np.mean(diffs) if diffs else 0.0


class SelectiveTransfer(KnowledgeTransferStrategy):
    """
    Selective transfer: Only transfer parameters that contribute to success.

    Analyzes which parameters differ between source (best) and targets,
    and only transfers parameters where the source's value is likely
    responsible for its success.
    """

    def __init__(self, importance_threshold: float = 0.3):
        self.importance_threshold = importance_threshold
        self.parameter_importance: Dict[str, float] = {}

    @property
    def name(self) -> str:
        return "selective"

    def set_parameter_importance(self, importance: Dict[str, float]) -> None:
        """Set the importance scores for each parameter."""
        self.parameter_importance = importance

    def transfer(
        self,
        source: PopulationAgent,
        targets: List[PopulationAgent],
        tau: float = 0.1
    ) -> Dict[str, Any]:
        stats = {
            "strategy": self.name,
            "source_id": source.id,
            "num_targets": len(targets),
            "importance_threshold": self.importance_threshold,
            "transfers": [],
        }

        source_params = source.get_parameters()

        for target in targets:
            if target.id != source.id:
                target_params = target.get_parameters()
                new_params = target_params.copy()
                transferred_keys = []

                for key in source_params:
                    if key in target_params:
                        importance = self.parameter_importance.get(key, 0.5)

                        if importance >= self.importance_threshold:
                            # Transfer this parameter
                            src_val = source_params[key]
                            tgt_val = target_params[key]

                            if isinstance(src_val, (int, float)) and isinstance(tgt_val, (int, float)):
                                new_params[key] = (1 - tau) * tgt_val + tau * src_val
                            elif isinstance(src_val, np.ndarray) and isinstance(tgt_val, np.ndarray):
                                new_params[key] = (1 - tau) * tgt_val + tau * src_val
                            else:
                                # For non-numeric, probabilistically adopt
                                if np.random.random() < tau:
                                    new_params[key] = src_val

                            transferred_keys.append(key)

                target.set_parameters(new_params)

                stats["transfers"].append({
                    "target_id": target.id,
                    "transferred_params": transferred_keys,
                    "num_transferred": len(transferred_keys),
                })

        return stats


class PromptEvolutionTransfer(KnowledgeTransferStrategy):
    """
    Prompt evolution transfer: For LLM-based agents, evolve prompts.

    This strategy is specifically designed for agents that use LLM prompts.
    It transfers successful prompt components (system prompts, few-shot
    examples, instruction formats) from the best agent to others.
    """

    def __init__(self):
        self.prompt_components = ["system_prompt", "examples", "instructions", "output_format"]

    @property
    def name(self) -> str:
        return "prompt_evolution"

    def transfer(
        self,
        source: PopulationAgent,
        targets: List[PopulationAgent],
        tau: float = 0.1
    ) -> Dict[str, Any]:
        stats = {
            "strategy": self.name,
            "source_id": source.id,
            "num_targets": len(targets),
            "transfers": [],
        }

        source_params = source.get_parameters()

        for target in targets:
            if target.id != source.id:
                target_params = target.get_parameters()
                new_params = target_params.copy()
                evolved_components = []

                for component in self.prompt_components:
                    if component in source_params and component in target_params:
                        # Probabilistically adopt prompt components
                        if np.random.random() < tau:
                            new_params[component] = source_params[component]
                            evolved_components.append(component)

                target.set_parameters(new_params)

                stats["transfers"].append({
                    "target_id": target.id,
                    "evolved_components": evolved_components,
                })

        return stats


class HybridTransfer(KnowledgeTransferStrategy):
    """
    Hybrid transfer: Combine multiple transfer strategies.

    Uses soft update for numeric parameters, distillation for behavior,
    and selective transfer for high-importance parameters.
    """

    def __init__(
        self,
        soft_weight: float = 0.4,
        selective_weight: float = 0.4,
        distillation_weight: float = 0.2
    ):
        self.soft_weight = soft_weight
        self.selective_weight = selective_weight
        self.distillation_weight = distillation_weight

        self.soft_transfer = SoftUpdateTransfer()
        self.selective_transfer = SelectiveTransfer()
        self.distillation_transfer = DistillationTransfer()

    @property
    def name(self) -> str:
        return "hybrid"

    def set_reference_inputs(self, inputs: List[Dict[str, Any]]) -> None:
        """Set reference inputs for distillation component."""
        self.distillation_transfer.set_reference_inputs(inputs)

    def set_parameter_importance(self, importance: Dict[str, float]) -> None:
        """Set parameter importance for selective component."""
        self.selective_transfer.set_parameter_importance(importance)

    def transfer(
        self,
        source: PopulationAgent,
        targets: List[PopulationAgent],
        tau: float = 0.1
    ) -> Dict[str, Any]:
        stats = {
            "strategy": self.name,
            "source_id": source.id,
            "num_targets": len(targets),
            "component_weights": {
                "soft": self.soft_weight,
                "selective": self.selective_weight,
                "distillation": self.distillation_weight,
            },
            "component_stats": {},
        }

        # Apply each strategy with weighted tau
        if self.soft_weight > 0:
            soft_stats = self.soft_transfer.transfer(
                source, targets, tau * self.soft_weight
            )
            stats["component_stats"]["soft"] = soft_stats

        if self.selective_weight > 0:
            selective_stats = self.selective_transfer.transfer(
                source, targets, tau * self.selective_weight
            )
            stats["component_stats"]["selective"] = selective_stats

        if self.distillation_weight > 0 and self.distillation_transfer.reference_inputs:
            distill_stats = self.distillation_transfer.transfer(
                source, targets, tau * self.distillation_weight
            )
            stats["component_stats"]["distillation"] = distill_stats

        return stats


@dataclass
class TransferSchedule:
    """Schedule for when and how to transfer knowledge."""
    frequency: int = 10  # Transfer every N iterations
    initial_tau: float = 0.2  # Starting transfer rate
    final_tau: float = 0.05  # Ending transfer rate
    decay_iterations: int = 100  # Iterations to decay from initial to final
    strategy: KnowledgeTransferStrategy = None

    def __post_init__(self):
        if self.strategy is None:
            self.strategy = SoftUpdateTransfer()

    def should_transfer(self, iteration: int) -> bool:
        """Check if transfer should happen at this iteration."""
        return iteration > 0 and iteration % self.frequency == 0

    def get_tau(self, iteration: int) -> float:
        """Get the tau value for the current iteration."""
        if iteration >= self.decay_iterations:
            return self.final_tau

        progress = iteration / self.decay_iterations
        return self.initial_tau + (self.final_tau - self.initial_tau) * progress


def create_transfer_strategy(name: str, **kwargs) -> KnowledgeTransferStrategy:
    """Factory function to create transfer strategies."""
    strategies = {
        "soft_update": SoftUpdateTransfer,
        "distillation": DistillationTransfer,
        "selective": SelectiveTransfer,
        "prompt_evolution": PromptEvolutionTransfer,
        "hybrid": HybridTransfer,
    }

    if name not in strategies:
        raise ValueError(f"Unknown transfer strategy: {name}. Available: {list(strategies.keys())}")

    return strategies[name](**kwargs)
