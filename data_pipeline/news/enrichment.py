"""LLM-based news enrichment for extracting trading signals.

Enriches raw news items with:
- Sentiment analysis (bullish/bearish/neutral)
- Event type classification
- Entity extraction
- Impact scoring
- Trading implications
"""
from __future__ import annotations
import os
import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from .sources import get_source_tier, get_source_weight, get_source_name
from .multi_asset_queries import detect_event_type

# Optional OpenAI import
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


@dataclass
class EnrichedNewsItem:
    """A news item enriched with trading signals."""
    # Original fields
    title: str
    url: str
    snippet: str
    published_at: datetime

    # Enriched fields
    sentiment: str = "neutral"  # bullish, bearish, neutral
    sentiment_score: float = 0.0  # -1.0 to 1.0
    sentiment_magnitude: float = 0.0  # 0.0 to 1.0 (strength)
    event_type: str = "general"
    entities: List[str] = field(default_factory=list)
    affected_assets: List[str] = field(default_factory=list)
    impact_score: float = 0.5  # 0.0 to 1.0
    impact_timeframe: str = "short_term"  # immediate, short_term, long_term
    relevance_score: float = 0.5  # 0.0 to 1.0
    trading_implication: str = ""
    key_numbers: Dict[str, Any] = field(default_factory=dict)

    # Source metadata
    source_name: str = ""
    source_tier: int = 5
    source_weight: float = 0.3

    # Processing metadata
    enrichment_model: str = ""
    enrichment_confidence: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        if isinstance(d["published_at"], datetime):
            d["published_at"] = d["published_at"].isoformat()
        return d

    @classmethod
    def from_raw(cls, raw: Dict[str, Any]) -> "EnrichedNewsItem":
        """Create from raw news item dict."""
        url = raw.get("url", "")
        return cls(
            title=raw.get("title", ""),
            url=url,
            snippet=raw.get("snippet", "") or raw.get("summary", ""),
            published_at=raw.get("published_at", datetime.now(timezone.utc)),
            source_name=get_source_name(url),
            source_tier=get_source_tier(url),
            source_weight=get_source_weight(url),
            event_type=detect_event_type(raw.get("title", ""), raw.get("snippet", "")),
        )


ENRICHMENT_PROMPT = """Analyze this crypto news and extract trading signals.

TITLE: {title}
SNIPPET: {snippet}
SOURCE: {source_name}
PUBLISHED: {published_at}

Return JSON only:
{{
  "sentiment": "bullish" | "bearish" | "neutral",
  "sentiment_score": -1.0 to 1.0 (negative=bearish, positive=bullish),
  "sentiment_magnitude": 0.0 to 1.0 (strength of sentiment),
  "event_type": "etf_flow" | "whale_move" | "exchange" | "regulation" | "hack" | "macro" | "mining" | "defi" | "adoption" | "general",
  "affected_assets": ["BTC", "ETH", ...],
  "entities": ["BlackRock", "SEC", "Binance", ...],
  "impact_score": 0.0 to 1.0 (market impact potential),
  "impact_timeframe": "immediate" | "short_term" | "long_term",
  "trading_implication": "One sentence trading insight",
  "key_numbers": {{"amount_usd": null, "percentage": null, "count": null}},
  "confidence": 0.0 to 1.0
}}

Rules:
- ETF inflows/institutional = bullish
- Hacks/exploits/lawsuits = bearish
- Regulation clarity = context-dependent
- Whale accumulation = bullish, whale selling = bearish
- Be conservative with extreme scores
"""


def _create_openai_client() -> Optional[Any]:
    """Create OpenAI client if available."""
    if not OPENAI_AVAILABLE:
        return None

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None

    base_url = os.getenv("OPENAI_API_BASE")
    if base_url:
        base_url = base_url.rstrip("/")
        if base_url.endswith("/chat/completions"):
            base_url = base_url.rsplit("/", 1)[0]
        return OpenAI(api_key=key, base_url=base_url)

    return OpenAI(api_key=key)


def enrich_with_llm(
    item: EnrichedNewsItem,
    model: str = "gpt-4o-mini",
    client: Optional[Any] = None,
) -> EnrichedNewsItem:
    """
    Enrich a single news item using LLM.

    Args:
        item: EnrichedNewsItem to enrich
        model: LLM model to use
        client: OpenAI client (created if not provided)

    Returns:
        Enriched item with sentiment and signals
    """
    if client is None:
        client = _create_openai_client()

    if client is None:
        # Fallback to rule-based enrichment
        return _enrich_rule_based(item)

    prompt = ENRICHMENT_PROMPT.format(
        title=item.title,
        snippet=item.snippet[:500],
        source_name=item.source_name,
        published_at=item.published_at.isoformat() if isinstance(item.published_at, datetime) else str(item.published_at),
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a crypto market analyst. Respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
            max_tokens=500,
        )

        result_text = response.choices[0].message.content or "{}"
        result = json.loads(result_text)

        # Update item with LLM results
        item.sentiment = result.get("sentiment", "neutral")
        item.sentiment_score = float(result.get("sentiment_score", 0.0))
        item.sentiment_magnitude = float(result.get("sentiment_magnitude", 0.5))
        item.event_type = result.get("event_type", item.event_type)
        item.affected_assets = result.get("affected_assets", [])
        item.entities = result.get("entities", [])
        item.impact_score = float(result.get("impact_score", 0.5))
        item.impact_timeframe = result.get("impact_timeframe", "short_term")
        item.trading_implication = result.get("trading_implication", "")
        item.key_numbers = result.get("key_numbers", {})
        item.enrichment_model = model
        item.enrichment_confidence = float(result.get("confidence", 0.5))

    except Exception as e:
        print(f"[Enrichment] LLM failed for '{item.title[:50]}...': {e}")
        item = _enrich_rule_based(item)

    return item


def _enrich_rule_based(item: EnrichedNewsItem) -> EnrichedNewsItem:
    """
    Fallback rule-based enrichment.

    Args:
        item: EnrichedNewsItem to enrich

    Returns:
        Enriched item with basic signals
    """
    text = f"{item.title} {item.snippet}".lower()

    # Simple sentiment detection
    bullish_words = ["bullish", "surge", "rally", "soar", "gain", "rise", "up", "high",
                     "inflow", "buy", "accumulate", "adoption", "approval", "positive"]
    bearish_words = ["bearish", "crash", "dump", "plunge", "drop", "fall", "down", "low",
                     "outflow", "sell", "hack", "exploit", "lawsuit", "ban", "negative"]

    bull_count = sum(1 for w in bullish_words if w in text)
    bear_count = sum(1 for w in bearish_words if w in text)

    if bull_count > bear_count:
        item.sentiment = "bullish"
        item.sentiment_score = min(0.8, 0.3 + bull_count * 0.1)
    elif bear_count > bull_count:
        item.sentiment = "bearish"
        item.sentiment_score = max(-0.8, -0.3 - bear_count * 0.1)
    else:
        item.sentiment = "neutral"
        item.sentiment_score = 0.0

    item.sentiment_magnitude = abs(item.sentiment_score)

    # Detect affected assets
    assets = []
    if "bitcoin" in text or "btc" in text:
        assets.append("BTC")
    if "ethereum" in text or "eth" in text:
        assets.append("ETH")
    if "solana" in text or "sol" in text:
        assets.append("SOL")
    if "dogecoin" in text or "doge" in text:
        assets.append("DOGE")
    if "ripple" in text or "xrp" in text:
        assets.append("XRP")
    item.affected_assets = assets or ["BTC"]  # Default to BTC

    item.enrichment_model = "rule_based"
    item.enrichment_confidence = 0.3

    return item


def enrich_batch(
    items: List[Dict[str, Any]],
    model: str = "gpt-4o-mini",
    max_workers: int = 5,
    use_llm: bool = True,
) -> List[EnrichedNewsItem]:
    """
    Enrich a batch of news items.

    Args:
        items: List of raw news item dicts
        model: LLM model to use
        max_workers: Max parallel workers for LLM calls
        use_llm: Whether to use LLM (False = rule-based only)

    Returns:
        List of EnrichedNewsItem
    """
    # Convert to EnrichedNewsItem
    enriched = [EnrichedNewsItem.from_raw(item) for item in items]

    if not use_llm:
        return [_enrich_rule_based(item) for item in enriched]

    client = _create_openai_client()
    if client is None:
        print("[Enrichment] No OpenAI client, using rule-based enrichment")
        return [_enrich_rule_based(item) for item in enriched]

    # Parallel LLM enrichment
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(enrich_with_llm, item, model, client): i
            for i, item in enumerate(enriched)
        }

        for future in as_completed(futures):
            idx = futures[future]
            try:
                results.append((idx, future.result()))
            except Exception as e:
                print(f"[Enrichment] Failed: {e}")
                results.append((idx, _enrich_rule_based(enriched[idx])))

    # Sort by original order
    results.sort(key=lambda x: x[0])
    return [r[1] for r in results]


def calculate_aggregate_sentiment(items: List[EnrichedNewsItem]) -> Dict[str, Any]:
    """
    Calculate aggregate sentiment from enriched items.

    Args:
        items: List of EnrichedNewsItem

    Returns:
        Aggregate sentiment metrics
    """
    if not items:
        return {
            "overall_sentiment": "neutral",
            "weighted_score": 0.0,
            "bullish_count": 0,
            "bearish_count": 0,
            "neutral_count": 0,
            "confidence": 0.0,
        }

    # Weight by source credibility and recency
    weighted_scores = []
    for item in items:
        weight = item.source_weight * item.enrichment_confidence
        weighted_scores.append(item.sentiment_score * weight)

    avg_score = sum(weighted_scores) / len(weighted_scores) if weighted_scores else 0.0

    bullish = sum(1 for i in items if i.sentiment == "bullish")
    bearish = sum(1 for i in items if i.sentiment == "bearish")
    neutral = sum(1 for i in items if i.sentiment == "neutral")

    if avg_score > 0.2:
        overall = "bullish"
    elif avg_score < -0.2:
        overall = "bearish"
    else:
        overall = "neutral"

    return {
        "overall_sentiment": overall,
        "weighted_score": avg_score,
        "bullish_count": bullish,
        "bearish_count": bearish,
        "neutral_count": neutral,
        "confidence": sum(i.enrichment_confidence for i in items) / len(items),
    }
