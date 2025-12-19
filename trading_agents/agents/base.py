"""Base agent class."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class BaseAgent:
    """Base class for all agents."""
    id: str

    def log(self, msg: str):
        """Log a message with agent identifier."""
        print(f"[{self.__class__.__name__}:{self.id}] {msg}")
