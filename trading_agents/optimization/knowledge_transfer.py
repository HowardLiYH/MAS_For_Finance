"""Knowledge transfer mechanism for continual learning."""
from __future__ import annotations
from typing import Dict, Any
import copy

from ..services.metrics import PerformanceTracker


class KnowledgeTransfer:
    """
    Implements knowledge transfer from top-performing agents to bottom performers.

    After K rounds, top agents in each category transfer their knowledge
    (inventory configurations, parameters) to bottom performers.
    """

    def __init__(
        self,
        tracker: PerformanceTracker,
        transfer_frequency: int = 10,
        trader_transfer_frequency: int = 20,
        transfer_ratio: float = 0.5,
    ):
        """
        Initialize knowledge transfer.

        Args:
            tracker: Performance tracker instance
            transfer_frequency: Transfer every K rounds
            trader_transfer_frequency: Transfer for traders every 2K rounds
            transfer_ratio: Transfer to bottom N% of agents
        """
        self.tracker = tracker
        self.transfer_frequency = transfer_frequency
        self.trader_transfer_frequency = trader_transfer_frequency
        self.transfer_ratio = transfer_ratio
        self.last_transfer: Dict[str, int] = {}

    def should_transfer(self, iteration: int, agent_type: str) -> bool:
        """Check if knowledge transfer should occur."""
        frequency = (
            self.trader_transfer_frequency
            if agent_type == "trader"
            else self.transfer_frequency
        )
        last = self.last_transfer.get(agent_type, 0)
        return (iteration - last) >= frequency

    def transfer_knowledge(
        self,
        agents: Dict[str, Any],
        agent_type: str,
        iteration: int,
    ) -> Dict[str, Any]:
        """
        Transfer knowledge from top performers to bottom performers.

        Args:
            agents: Dict of agent_id -> agent_instance
            agent_type: Type of agents ("analyst", "researcher", "trader")
            iteration: Current iteration number

        Returns:
            Updated agents dict
        """
        if not self.should_transfer(iteration, agent_type):
            return agents

        self.last_transfer[agent_type] = iteration

        # Get top agents
        top_agents = self.tracker.get_top_agents(agent_type, top_n=1)
        if not top_agents:
            return agents

        top_agent_id, top_metrics = top_agents[0]

        # Get agents of this type
        type_agents = {
            agent_id: agent
            for agent_id, agent in agents.items()
            if hasattr(agent, '__class__') and agent_type in agent.__class__.__name__.lower()
        }

        if len(type_agents) < 2:
            return agents

        # Get top agent's inventory
        top_agent = type_agents.get(top_agent_id)
        if not top_agent or not hasattr(top_agent, 'inventory'):
            return agents

        top_inventory = copy.deepcopy(top_agent.inventory)

        # Rank other agents
        agent_rankings = []
        for agent_id, agent in type_agents.items():
            if agent_id == top_agent_id:
                continue
            metrics = self.tracker.calculate_agent_metrics(agent_id, agent_type)
            score = (
                metrics.get("Sharpe", 0.0) * 0.3 +
                metrics.get("HitRate", 0.0) * 0.3 +
                (metrics.get("PnL", 0.0) / 1000.0) * 0.2 +
                (1.0 - metrics.get("MaxDD", 0.0)) * 0.1 +
                (1.0 - metrics.get("CalibECE", 0.0)) * 0.1
            )
            agent_rankings.append((agent_id, agent, score))

        # Sort by score (ascending - worst first)
        agent_rankings.sort(key=lambda x: x[2])

        # Transfer to bottom N%
        num_to_transfer = max(1, int(len(agent_rankings) * self.transfer_ratio))

        transferred = 0
        for agent_id, agent, _ in agent_rankings[:num_to_transfer]:
            if hasattr(agent, 'inventory'):
                agent.inventory = copy.deepcopy(top_inventory)
                transferred += 1
                print(f"[KNOWLEDGE] {agent_type}:{agent_id} ← {agent_type}:{top_agent_id}")

        if transferred > 0:
            print(f"[KNOWLEDGE] Transferred to {transferred} {agent_type} agents")

        return agents

    def transfer_all_types(
        self,
        agents: Dict[str, Any],
        iteration: int,
    ) -> Dict[str, Any]:
        """Transfer knowledge for all agent types that are due."""
        for agent_type in ["analyst", "researcher", "trader"]:
            if self.should_transfer(iteration, agent_type):
                agents = self.transfer_knowledge(agents, agent_type, iteration)
        return agents
