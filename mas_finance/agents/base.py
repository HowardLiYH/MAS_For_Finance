
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass
class BaseAgent:
    id: str
    def log(self, msg: str):
        print(f"[{self.__class__.__name__}:{self.id}] {msg}")
