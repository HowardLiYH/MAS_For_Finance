"""Population-based multi-agent learning system.

This module implements a novel framework for training multi-agent LLM systems
through population-based continual learning. Each agent role (Analyst, Researcher,
Trader, Risk Manager) maintains a population of diverse agents that evolve by
transferring knowledge from top performers to the rest of the population.
"""

from .base import AgentPopulation, PopulationConfig
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
    PromptEvolutionTransfer,
)
from .diversity import (
    DiversityMetric,
    BehavioralDiversity,
    ParameterDiversity,
    OutputDiversity,
    DiversityPreserver,
)
from .scoring import (
    PopulationScorer,
    IndividualScorer,
    PipelineScorer,
    ShapleyScorer,
)
from .workflow import PopulationWorkflow

__all__ = [
    # Base
    "AgentPopulation",
    "PopulationConfig",
    # Variants
    "AnalystVariant",
    "ResearcherVariant",
    "TraderVariant",
    "RiskVariant",
    "create_analyst_population",
    "create_researcher_population",
    "create_trader_population",
    "create_risk_population",
    # Transfer
    "KnowledgeTransferStrategy",
    "SoftUpdateTransfer",
    "DistillationTransfer",
    "SelectiveTransfer",
    "PromptEvolutionTransfer",
    # Diversity
    "DiversityMetric",
    "BehavioralDiversity",
    "ParameterDiversity",
    "OutputDiversity",
    "DiversityPreserver",
    # Scoring
    "PopulationScorer",
    "IndividualScorer",
    "PipelineScorer",
    "ShapleyScorer",
    # Workflow
    "PopulationWorkflow",
]

