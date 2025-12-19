"""Inventory pruning mechanism."""
from __future__ import annotations
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timezone
from collections import defaultdict

from ..services.metrics import PerformanceTracker


class InventoryPruner:
    """
    Tracks inventory method usage and prunes rarely used methods.

    Methods are ranked by usage frequency and success rate.
    Rarely used methods with low success rates are pruned.
    """

    def __init__(
        self,
        tracker: PerformanceTracker,
        pruning_frequency: int = 50,
        min_usage_threshold: int = 5,
        min_success_rate: float = 0.3,
        protected_methods: Optional[Set[str]] = None,
    ):
        """
        Initialize inventory pruner.

        Args:
            tracker: Performance tracker instance
            pruning_frequency: Prune every M rounds
            min_usage_threshold: Minimum uses before pruning
            min_success_rate: Minimum success rate
            protected_methods: Methods that should never be pruned
        """
        self.tracker = tracker
        self.pruning_frequency = pruning_frequency
        self.min_usage_threshold = min_usage_threshold
        self.min_success_rate = min_success_rate
        self.protected_methods = protected_methods or set()
        self.last_pruning = 0

    def should_prune(self, iteration: int) -> bool:
        """Check if pruning should occur."""
        return (iteration - self.last_pruning) >= self.pruning_frequency

    def get_methods_to_prune(
        self,
        agent_type: str,
        current_inventories: Dict[str, List[str]],
    ) -> Dict[str, List[str]]:
        """
        Identify methods to prune.

        Args:
            agent_type: Type of agent
            current_inventories: Dict of pool -> [method_names]

        Returns:
            Dict of pool -> [method_names] to remove
        """
        usage_stats = self.tracker.get_method_usage_stats()
        to_prune: Dict[str, List[str]] = defaultdict(list)

        for pool, method_names in current_inventories.items():
            for method_name in method_names:
                method_key = f"{pool}:{method_name}"

                # Protected methods are never pruned
                if method_key in self.protected_methods:
                    continue

                stats = usage_stats.get(method_key)

                # Never used
                if not stats:
                    to_prune[pool].append(method_name)
                    continue

                usage_count = stats.get("usage_count", 0)
                success_rate = stats.get("success_rate", 0.0)
                last_used = stats.get("last_used")

                # Prune if low usage AND low success rate
                if usage_count < self.min_usage_threshold:
                    if success_rate < self.min_success_rate:
                        to_prune[pool].append(method_name)
                        print(f"[PRUNE] {method_key}: low usage ({usage_count}), low success ({success_rate:.2f})")
                    elif last_used:
                        last_used_dt = datetime.fromisoformat(last_used.replace('Z', '+00:00'))
                        days_since = (datetime.now(timezone.utc) - last_used_dt).days
                        if days_since > (self.pruning_frequency * 2):
                            to_prune[pool].append(method_name)
                            print(f"[PRUNE] {method_key}: not used in {days_since} days")

        return dict(to_prune)

    def prune_inventories(
        self,
        agents: Dict[str, Any],
        agent_type: str,
        iteration: int,
    ) -> Dict[str, Any]:
        """
        Prune rarely used methods from agent inventories.

        Args:
            agents: Dict of agent_id -> agent_instance
            agent_type: Type of agents to prune
            iteration: Current iteration number

        Returns:
            Updated agents dict
        """
        if not self.should_prune(iteration):
            return agents

        self.last_pruning = iteration

        # Get agents of this type
        type_agents = {
            agent_id: agent
            for agent_id, agent in agents.items()
            if hasattr(agent, '__class__') and agent_type in agent.__class__.__name__.lower()
        }

        total_pruned = 0

        for agent_id, agent in type_agents.items():
            if not hasattr(agent, 'inventory'):
                continue

            # Extract current method names
            current_inventories = {}
            for pool, methods in agent.inventory.items():
                method_names = []
                for entry in methods:
                    if hasattr(entry, 'name'):
                        method_names.append(entry.name)
                    elif isinstance(entry, tuple) and hasattr(entry[0], 'name'):
                        method_names.append(entry[0].name)
                if method_names:
                    current_inventories[pool] = method_names

            # Get methods to prune
            to_prune = self.get_methods_to_prune(agent_type, current_inventories)

            if not to_prune:
                continue

            # Remove pruned methods
            pruned_count = 0
            for pool, names_to_remove in to_prune.items():
                if pool not in agent.inventory:
                    continue

                original = agent.inventory[pool]
                filtered = []

                for entry in original:
                    method_name = None
                    if hasattr(entry, 'name'):
                        method_name = entry.name
                    elif isinstance(entry, tuple) and hasattr(entry[0], 'name'):
                        method_name = entry[0].name

                    if method_name and method_name not in names_to_remove:
                        filtered.append(entry)
                    elif not method_name:
                        filtered.append(entry)

                if len(filtered) < len(original):
                    agent.inventory[pool] = filtered
                    pruned_count += len(original) - len(filtered)

            if pruned_count > 0:
                total_pruned += pruned_count
                print(f"[PRUNE] {agent_type}:{agent_id} - removed {pruned_count} methods")

        if total_pruned > 0:
            print(f"[PRUNE] Pruned {total_pruned} methods from {agent_type} agents")

        return agents

    def prune_all_types(
        self,
        agents: Dict[str, Any],
        iteration: int,
    ) -> Dict[str, Any]:
        """Prune inventories for all agent types."""
        for agent_type in ["analyst", "researcher", "trader", "risk"]:
            if self.should_prune(iteration):
                agents = self.prune_inventories(agents, agent_type, iteration)
        return agents
