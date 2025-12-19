# News data fetching and processing module
from .llm_prompt_search import search_micro_macro

# Multi-asset queries
from .multi_asset_queries import (
    ASSET_QUERIES,
    get_queries_for_asset,
    get_queries_for_multi_asset,
    detect_event_type,
)

# Source credibility
from .sources import (
    get_source_tier,
    get_source_weight,
    get_source_name,
    filter_by_credibility,
    sort_by_credibility,
)

# Enrichment
from .enrichment import (
    EnrichedNewsItem,
    enrich_with_llm,
    enrich_batch,
    calculate_aggregate_sentiment,
)

# Aggregation
from .aggregation import (
    NewsCluster,
    NewsDigest,
    aggregate_news,
    deduplicate_by_content,
)

__all__ = [
    # Search
    "search_micro_macro",
    # Multi-asset
    "ASSET_QUERIES",
    "get_queries_for_asset",
    "get_queries_for_multi_asset",
    "detect_event_type",
    # Sources
    "get_source_tier",
    "get_source_weight",
    "get_source_name",
    "filter_by_credibility",
    "sort_by_credibility",
    # Enrichment
    "EnrichedNewsItem",
    "enrich_with_llm",
    "enrich_batch",
    "calculate_aggregate_sentiment",
    # Aggregation
    "NewsCluster",
    "NewsDigest",
    "aggregate_news",
    "deduplicate_by_content",
]
