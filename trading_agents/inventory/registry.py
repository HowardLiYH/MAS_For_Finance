"""Central registry for inventory methods."""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Type, List

REGISTRY: Dict[str, Dict[str, type]] = defaultdict(dict)

def register(pool: str, name: str):
    """
    Decorator to register an inventory method.

    Usage:
        @register("analyst.features", "talib_stack")
        class TALibStack(FeatureMethod):
            ...
    """
    def decorator(cls: Type):
        REGISTRY[pool][name] = cls
        cls.pool = pool
        cls.name = name
        return cls
    return decorator

def get(pool: str, name: str) -> Type:
    """Get a registered method class by pool and name."""
    if pool not in REGISTRY:
        raise KeyError(f"Unknown pool: {pool}")
    if name not in REGISTRY[pool]:
        raise KeyError(f"Unknown method '{name}' in pool '{pool}'")
    return REGISTRY[pool][name]

def list_methods(pool: str) -> List[str]:
    """List all registered methods in a pool."""
    return list(REGISTRY[pool].keys())

def list_pools() -> List[str]:
    """List all registered pools."""
    return list(REGISTRY.keys())
