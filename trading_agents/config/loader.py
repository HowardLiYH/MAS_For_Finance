"""Configuration loading and agent building."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Any, Union, Tuple, Optional
import os
import yaml

from .schemas import DataConfig, NewsConfig, LearningConfig, AppConfig
from ..inventory.registry import get as registry_get


# Type aliases
MethodInstance = Any
MethodEntry = Union[MethodInstance, Tuple[MethodInstance, Dict[str, Any]]]


def _deep_update(base: dict, updates: dict) -> dict:
    """Recursively update base dict with updates."""
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: Optional[Union[str, Path]] = None) -> AppConfig:
    """
    Load configuration from YAML file with environment overrides.

    Args:
        path: Path to YAML config file

    Returns:
        AppConfig instance
    """
    config = AppConfig()

    if path:
        filepath = Path(path)
        if filepath.exists():
            data = yaml.safe_load(filepath.read_text()) or {}
            base = asdict(config)
            merged = _deep_update(base, data)

            config = AppConfig(
                symbol=merged.get("symbol", config.symbol),
                timeframe=merged.get("timeframe", config.timeframe),
                data=DataConfig(**merged.get("data", {})),
                news=NewsConfig(**merged.get("news", {})),
                learning=LearningConfig(**merged.get("learning", {})),
            )

    # Environment overrides
    config.data.exchange_id = os.getenv("EXCHANGE_ID", config.data.exchange_id)
    config.news.search_provider = os.getenv("NEWS_PROVIDER", config.news.search_provider)

    return config


def _parse_inventory_token(pool: str, token: Union[str, Dict[str, Any]]) -> MethodEntry:
    """Parse an inventory token into a method instance."""
    if isinstance(token, str):
        return registry_get(pool, token)()

    if not isinstance(token, dict) or "name" not in token:
        raise ValueError(f"Invalid inventory token for pool '{pool}': {token!r}")

    name = token["name"]
    init_kwargs = token.get("init", {}) or {}
    run_kwargs = token.get("run", {}) or {}

    instance = registry_get(pool, name)(**init_kwargs)
    return (instance, run_kwargs) if run_kwargs else instance


def _build_inventory(spec: Any) -> Dict[str, List[MethodEntry]]:
    """Build inventory dict from config spec."""
    inventory: Dict[str, List[MethodEntry]] = {}

    # Option A: dict per pool
    if isinstance(spec, dict):
        for pool, tokens in spec.items():
            if not isinstance(tokens, list):
                raise ValueError(f"Inventory for pool '{pool}' must be a list")
            inventory[pool] = [_parse_inventory_token(pool, t) for t in tokens]
        return inventory

    # Option B: flat list of "pool:name" strings
    if isinstance(spec, list):
        for item in spec:
            if not isinstance(item, str) or ":" not in item:
                raise ValueError(f"Inventory items must be 'pool:name' strings, got {item!r}")
            pool, name = item.split(":", 1)
            inventory.setdefault(pool, []).append(_parse_inventory_token(pool, name))
        return inventory

    raise ValueError(f"Unsupported inventory spec type: {type(spec)}")


def build_agents(
    agents_config: Dict[str, Any],
    trader_llm_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build agent instances from configuration.

    Args:
        agents_config: Dict with agent configurations
        trader_llm_config: Optional LLM config for trader

    Returns:
        Dict mapping agent type to agent instance
    """
    # Lazy imports to avoid circular dependencies
    from ..agents.analyst import AnalystAgent
    from ..agents.researcher import ResearcherAgent
    from ..agents.trader import TraderAgent
    from ..agents.risk import RiskAgent
    from ..agents.evaluator import EvaluatorAgent

    agents: Dict[str, Any] = {}

    def get_inventory(spec: Dict[str, Any]) -> Dict[str, List[MethodEntry]]:
        inv_spec = spec.get("inventory", spec.get("inventories", []))
        return _build_inventory(inv_spec) if inv_spec else {}

    if "analyst" in agents_config:
        cfg = agents_config["analyst"] or {}
        agents["analyst"] = AnalystAgent(
            id=cfg.get("id", "A1"),
            inventory=get_inventory(cfg)
        )

    if "researcher" in agents_config:
        cfg = agents_config["researcher"] or {}
        agents["researcher"] = ResearcherAgent(
            id=cfg.get("id", "R1"),
            inventory=get_inventory(cfg)
        )

    if "trader" in agents_config:
        cfg = agents_config["trader"] or {}
        llm_cfg = trader_llm_config or {}
        agents["trader"] = TraderAgent(
            id=cfg.get("id", "T1"),
            inventory=get_inventory(cfg),
            use_llm=cfg.get("use_llm", llm_cfg.get("use_llm", True)),
            llm_model=cfg.get("llm_model", llm_cfg.get("llm_model", "gpt-4o-mini")),
            log_thoughts=cfg.get("log_thoughts", llm_cfg.get("log_thoughts", True)),
        )

    if "risk" in agents_config:
        cfg = agents_config["risk"] or {}
        agents["risk"] = RiskAgent(
            id=cfg.get("id", "M1"),
            inventory=get_inventory(cfg)
        )

    if "evaluator" in agents_config:
        cfg = agents_config["evaluator"] or {}
        agents["evaluator"] = EvaluatorAgent(id=cfg.get("id", "E1"))

    return agents


def build_agents_from_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Build agents from YAML configuration file.

    Args:
        path: Path to YAML config file

    Returns:
        Dict mapping agent type to agent instance
    """
    filepath = Path(path)
    if not filepath.exists():
        return {}

    config = yaml.safe_load(filepath.read_text()) or {}
    agents_config = config.get("agents", {})

    if not isinstance(agents_config, dict):
        return {}

    # Extract trader LLM config from top-level
    trader_llm_config = {
        "use_llm": config.get("trader_use_llm", True),
        "llm_model": config.get("trader_llm_model",
                                config.get("news", {}).get("llm_model", "gpt-4o-mini")),
        "log_thoughts": config.get("trader_log_thoughts", True),
    }

    return build_agents(agents_config, trader_llm_config=trader_llm_config)
