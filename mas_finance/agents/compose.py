from __future__ import annotations
from typing import Dict, List, Any, Tuple, Union
from pathlib import Path
import yaml

from .analyst import AnalystAgent
from .researcher import ResearcherAgent
from .trader import TraderAgent
from .risk import RiskManagerAgent
from .evaluator import EvaluatorAgent
from ..inventories.registry import get as registry_get

# Optional plugin auto-discovery
try:
    from ..inventories import load_plugins as _load_inventory_plugins
except Exception:
    _load_inventory_plugins = None

MethodInstance = Any
RunKw = Dict[str, Any]
MethodEntry = Union[MethodInstance, Tuple[MethodInstance, RunKw]]

def load_inventory_plugins() -> None:
    if _load_inventory_plugins:
        _load_inventory_plugins()

def _instance_from_token(pool: str, token: Union[str, Dict[str, Any]]) -> MethodEntry:
    if isinstance(token, str):
        return registry_get(pool, token)()
    if not isinstance(token, dict) or "name" not in token:
        raise ValueError(f"Invalid inventory token for pool '{pool}': {token!r}")
    name = token["name"]
    init_kwargs = token.get("init", {}) or {}
    run_kwargs = token.get("run", {}) or {}
    inst = registry_get(pool, name)(**init_kwargs)
    return (inst, run_kwargs) if run_kwargs else inst

def _build_inventory(spec: Any) -> Dict[str, List[MethodEntry]]:
    inv: Dict[str, List[MethodEntry]] = {}
    # Option A: dict per pool
    if isinstance(spec, dict):
        for pool, tokens in spec.items():
            if not isinstance(tokens, list):
                raise ValueError(f"Inventory for pool '{pool}' must be a list.")
            inv[pool] = [_instance_from_token(pool, t) for t in tokens]
        return inv
    # Option B: flat list of "pool:name" strings
    if isinstance(spec, list):
        for item in spec:
            if not isinstance(item, str) or ":" not in item:
                raise ValueError(f"Inventory items must be 'pool:name' strings, got {item!r}")
            pool, name = item.split(":", 1)
            inv.setdefault(pool, []).append(_instance_from_token(pool, name))
        return inv
    raise ValueError(f"Unsupported inventory spec type: {type(spec)}")

def build_agents_from_dict(agents_cfg: Dict[str, Any]) -> Dict[str, Any]:
    agents: Dict[str, Any] = {}
    def _inv(spec_keyed: Dict[str, Any]) -> Dict[str, List[MethodEntry]]:
        spec = spec_keyed.get("inventory", spec_keyed.get("inventories", []))
        return _build_inventory(spec) if spec else {}

    if "analyst" in agents_cfg:
        a = agents_cfg["analyst"] or {}
        agents["analyst"] = AnalystAgent(id=a.get("id","A1"), inventory=_inv(a))
    if "researcher" in agents_cfg:
        r = agents_cfg["researcher"] or {}
        agents["researcher"] = ResearcherAgent(id=r.get("id","R1"), inventory=_inv(r))
    if "trader" in agents_cfg:
        t = agents_cfg["trader"] or {}
        agents["trader"] = TraderAgent(id=t.get("id","T1"), inventory=_inv(t))
    if "risk" in agents_cfg:
        m = agents_cfg["risk"] or {}
        agents["risk"] = RiskManagerAgent(id=m.get("id","M1"), inventory=_inv(m))
    if "evaluator" in agents_cfg:
        e = agents_cfg["evaluator"] or {}
        agents["evaluator"] = EvaluatorAgent(id=e.get("id","E1"))
    return agents

def build_agents_from_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    y = yaml.safe_load(path.read_text()) or {}
    agents_cfg = y.get("agents", {})
    if not isinstance(agents_cfg, dict):
        return {}
    return build_agents_from_dict(agents_cfg)
