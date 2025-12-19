# Pipeline - Core data pipeline functionality

from .data_pipeline import run_pipeline, run_multi_asset_pipeline, run_pipeline_auto
from .multi_asset import MultiAssetLoader, load_all_assets, load_bybit_csv, DEFAULT_SYMBOLS
from .cross_features import (
    generate_market_context,
    add_market_context_to_asset,
    MarketContext,
    btc_dominance,
    altcoin_momentum,
    eth_btc_ratio,
    cross_oi_delta,
    aggregate_funding,
    risk_on_off,
    market_volatility,
    cross_correlation,
)
from .schemas import PriceBar, NewsItem

__all__ = [
    # Data pipeline
    "run_pipeline",
    "run_multi_asset_pipeline",
    "run_pipeline_auto",
    # Multi-asset loader
    "MultiAssetLoader",
    "load_all_assets",
    "load_bybit_csv",
    "DEFAULT_SYMBOLS",
    # Cross-asset features
    "generate_market_context",
    "add_market_context_to_asset",
    "MarketContext",
    "btc_dominance",
    "altcoin_momentum",
    "eth_btc_ratio",
    "cross_oi_delta",
    "aggregate_funding",
    "risk_on_off",
    "market_volatility",
    "cross_correlation",
    # Schemas
    "PriceBar",
    "NewsItem",
]
