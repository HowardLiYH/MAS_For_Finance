"""External services module."""
from .llm import generate_trading_proposal
from .metrics import PerformanceTracker

__all__ = ["generate_trading_proposal", "PerformanceTracker"]
