"""
Multi-Agent Trading System

A modular, LLM-powered trading system with:
- Analyst, Researcher, Trader, Risk, Evaluator agents
- Plugin-based inventory system for extensibility
- Continual learning and inventory optimization
"""
__version__ = "0.2.0"

# Lazy imports to avoid loading pandas on package import
def __getattr__(name):
    """Lazy import for heavy modules."""
    if name == "WorkflowEngine":
        from .workflow import WorkflowEngine
        return WorkflowEngine
    elif name == "run_single_iteration":
        from .workflow import run_single_iteration
        return run_single_iteration
    elif name == "run_multi_asset":
        from .workflow import run_multi_asset
        return run_multi_asset
    elif name == "ResearchSummary":
        from .models import ResearchSummary
        return ResearchSummary
    elif name == "ExecutionSummary":
        from .models import ExecutionSummary
        return ExecutionSummary
    elif name == "RiskReview":
        from .models import RiskReview
        return RiskReview
    elif name == "AgentScores":
        from .models import AgentScores
        return AgentScores
    elif name == "AppConfig":
        from .config import AppConfig
        return AppConfig
    elif name == "OrchestratorInput":
        from .config import OrchestratorInput
        return OrchestratorInput
    elif name == "load_config":
        from .config import load_config
        return load_config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Core
    "WorkflowEngine",
    "run_single_iteration",
    "run_multi_asset",
    # Models
    "ResearchSummary",
    "ExecutionSummary",
    "RiskReview",
    "AgentScores",
    # Config
    "AppConfig",
    "OrchestratorInput",
    "load_config",
]
