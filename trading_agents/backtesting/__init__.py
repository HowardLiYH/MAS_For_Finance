# Backtesting module for MAS trading system
"""Backtesting engine for PopAgent trading system.

Supports both single-agent and population-based backtesting.
"""

from .engine import BacktestEngine, setup_validation_test_periods
from .executor import OrderExecutor, OrderExecution, OrderState

__all__ = [
    "BacktestEngine",
    "setup_validation_test_periods",
    "OrderExecutor",
    "OrderExecution",
    "OrderState",
]
