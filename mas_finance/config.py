# mas_finance/config.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import os, yaml

@dataclass
class DataCfg:
    exchange_id: str = "kraken"
    timeframe: str = "4h"
    max_days: int = 30
    out_dir: str = "data"
    offline_prices_csv: str | None = None
    offline_news_jsonl: str | None = None

@dataclass
class NewsCfg:
    use_llm_news: bool = True
    news_query: str = "bitcoin OR BTC"
    search_provider: str = "serpapi"
    llm_model: str = "gpt-4o-mini"
    max_news_per_stream: int = 10
    max_search_results_per_stream: int = 50
    require_published_signal: bool = True

@dataclass
class AppCfg:
    # top-level app knobs
    symbol: str = "BTCUSD.PERP"
    timeframe: str = "4h"  # CLI still allowed to override
    data: DataCfg = DataCfg()
    news: NewsCfg = NewsCfg()

def _deep_update(obj: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(obj.get(k), dict):
            obj[k] = _deep_update(obj[k], v)
        else:
            obj[k] = v
    return obj

def load_config(path: str | Path | None) -> AppCfg:
    cfg = AppCfg()  # defaults
    if path:
        p = Path(path)
        if p.exists():
            data = yaml.safe_load(p.read_text()) or {}
            base = asdict(cfg)
            merged = _deep_update(base, data)
            # reconstruct dataclasses
            cfg = AppCfg(
                symbol=merged.get("symbol", cfg.symbol),
                timeframe=merged.get("timeframe", cfg.timeframe),
                data=DataCfg(**merged.get("data", {})),
                news=NewsCfg(**merged.get("news", {})),
            )
    # env overrides (optional)
    cfg.data.exchange_id = os.getenv("EXCHANGE_ID", cfg.data.exchange_id)
    cfg.news.search_provider = os.getenv("NEWS_PROVIDER", cfg.news.search_provider)
    return cfg

# --- Orchestrator input shape (typed, fixes Pylance complaints)
from dataclasses import dataclass

@dataclass
class OrchestratorInput:
    symbol: str
    interval: str
    appcfg: AppCfg
