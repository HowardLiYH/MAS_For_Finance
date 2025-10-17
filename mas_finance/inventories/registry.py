
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Type

REGISTRY: Dict[str, Dict[str, type]] = defaultdict(dict)  # REGISTRY["analyst.feature"]["talib_stack"] = Class

def register(pool: str, name: str):
    def deco(cls: Type):
        REGISTRY[pool][name] = cls
        cls.pool, cls.name = pool, name
        return cls
    return deco

def get(pool: str, name: str):
    return REGISTRY[pool][name]

def list_methods(pool: str):
    return list(REGISTRY[pool].keys())
