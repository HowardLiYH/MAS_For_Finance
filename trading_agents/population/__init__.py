"""Population-based multi-agent learning system with method selection.

This module implements PopAgent, a novel framework where agents LEARN to SELECT
which methods to use from a shared inventory, rather than being locked into
fixed strategies.

Key Innovation:
- Inventory > Agents: Each role has 10-15 methods, agents pick 3
- Selection Learning: Agents learn WHICH methods work via reinforcement
- Preference Transfer: Best agent's selection preferences inform others
- Context-Aware: Selection adapts to market conditions

Architecture:
    Inventory (15 methods)          Agent Population (5 agents)
    ┌─────────────────────┐         ┌───────────────────────┐
    │ RSI, MACD, HMM, ... │ ←select─│ Agent picks [RSI, HMM]│
    └─────────────────────┘         └───────────────────────┘
"""

# Method Selector (core innovation)
from .selector import (
    MethodSelector,
    SelectorPopulation,
    SelectorPopulationConfig,
    AgentRole,
    SelectionOutcome,
    SelectionState,
    MethodInfo,
)

# Extended Inventories (10-15 methods per role)
from .inventories import (
    ANALYST_INVENTORY,
    RESEARCHER_INVENTORY,
    TRADER_INVENTORY,
    RISK_INVENTORY,
    ANALYST_METHOD_INFO,
    RESEARCHER_METHOD_INFO,
    TRADER_METHOD_INFO,
    RISK_METHOD_INFO,
    get_inventory,
    get_method_info,
    get_all_inventories,
    get_inventory_sizes,
)

# Selector Workflow
from .selector_workflow import (
    SelectorWorkflow,
    SelectorWorkflowConfig,
    PipelineResult,
    IterationSummary,
    create_selector_workflow,
)

# Legacy support (fixed variants approach)
from .base import AgentPopulation, PopulationConfig, PopulationAgent
from .variants import (
    AnalystVariant,
    ResearcherVariant,
    TraderVariant,
    RiskVariant,
    create_analyst_population,
    create_researcher_population,
    create_trader_population,
    create_risk_population,
)
from .transfer import (
    KnowledgeTransferStrategy,
    SoftUpdateTransfer,
    DistillationTransfer,
    SelectiveTransfer,
)
from .diversity import (
    DiversityMetric,
    ParameterDiversity,
    BehavioralDiversity,
    DiversityPreserver,
)
from .scoring import (
    PopulationScorer,
    IndividualScorer,
    PipelineScorer,
    ShapleyScorer,
    TradeResult,
)
from .workflow import PopulationWorkflow

__all__ = [
    # Core (Method Selection)
    "MethodSelector",
    "SelectorPopulation",
    "SelectorPopulationConfig",
    "AgentRole",
    "SelectionOutcome",
    "SelectionState",
    # Inventories
    "ANALYST_INVENTORY",
    "RESEARCHER_INVENTORY",
    "TRADER_INVENTORY",
    "RISK_INVENTORY",
    "get_inventory",
    "get_method_info",
    "get_all_inventories",
    "get_inventory_sizes",
    # Workflow
    "SelectorWorkflow",
    "SelectorWorkflowConfig",
    "create_selector_workflow",
    # Legacy
    "AgentPopulation",
    "PopulationConfig",
    "PopulationWorkflow",
]
