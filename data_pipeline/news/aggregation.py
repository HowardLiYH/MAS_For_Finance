"""News aggregation and clustering.

Aggregates enriched news items into a digestible market summary,
grouping similar stories and extracting key narratives.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
from collections import Counter

from .enrichment import EnrichedNewsItem, calculate_aggregate_sentiment


@dataclass
class NewsCluster:
    """A cluster of related news items."""
    event_type: str
    items: List[EnrichedNewsItem]
    primary_headline: str
    sentiment: str
    sentiment_score: float
    affected_assets: List[str]
    source_count: int
    tier1_sources: int

    @property
    def weight(self) -> float:
        """Calculate cluster importance weight."""
        # More sources = more important
        source_factor = min(1.0, self.source_count / 5)
        # Tier 1 sources boost importance
        tier1_factor = min(1.0, self.tier1_sources / 2)
        # Extreme sentiment = more actionable
        sentiment_factor = abs(self.sentiment_score)

        return (source_factor * 0.4 + tier1_factor * 0.3 + sentiment_factor * 0.3)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "headline": self.primary_headline,
            "sentiment": self.sentiment,
            "sentiment_score": self.sentiment_score,
            "affected_assets": self.affected_assets,
            "source_count": self.source_count,
            "tier1_sources": self.tier1_sources,
            "weight": self.weight,
        }


@dataclass
class NewsDigest:
    """Aggregated news digest for trading decisions."""
    # Time window
    from_dt: datetime
    to_dt: datetime

    # Overall metrics
    total_items: int = 0
    overall_sentiment: str = "neutral"
    sentiment_score: float = 0.0
    sentiment_trend: str = "stable"  # improving, worsening, stable

    # Breakdown
    bullish_count: int = 0
    bearish_count: int = 0
    neutral_count: int = 0

    # Top narratives
    dominant_narratives: List[str] = field(default_factory=list)
    key_events: List[NewsCluster] = field(default_factory=list)

    # By asset
    asset_sentiment: Dict[str, float] = field(default_factory=dict)

    # Source quality
    tier1_percentage: float = 0.0
    avg_source_credibility: float = 0.0

    # Notable numbers (aggregated)
    notable_numbers: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["from_dt"] = self.from_dt.isoformat()
        d["to_dt"] = self.to_dt.isoformat()
        d["key_events"] = [e.to_dict() for e in self.key_events]
        return d

    def to_prompt_text(self) -> str:
        """Format digest for LLM prompt."""
        lines = [
            f"NEWS DIGEST ({self.from_dt.strftime('%Y-%m-%d %H:%M')} to {self.to_dt.strftime('%Y-%m-%d %H:%M')} UTC)",
            f"├── Overall Sentiment: {self.sentiment_score:+.2f} ({self.overall_sentiment.upper()})",
            f"├── Trend: {self.sentiment_trend}",
            f"├── Coverage: {self.total_items} articles ({self.tier1_percentage:.0%} tier-1 sources)",
            f"│",
        ]

        if self.dominant_narratives:
            lines.append("├── DOMINANT NARRATIVES:")
            for i, narrative in enumerate(self.dominant_narratives[:3], 1):
                lines.append(f"│   {i}. {narrative}")
            lines.append("│")

        if self.key_events:
            lines.append("├── KEY EVENTS:")
            for event in self.key_events[:5]:
                emoji = "🟢" if event.sentiment == "bullish" else "🔴" if event.sentiment == "bearish" else "⚪"
                lines.append(f"│   {emoji} [{event.event_type.upper()}] {event.primary_headline}")
            lines.append("│")

        if self.asset_sentiment:
            lines.append("├── ASSET SENTIMENT:")
            for asset, score in sorted(self.asset_sentiment.items(), key=lambda x: x[1], reverse=True):
                indicator = "↑" if score > 0.1 else "↓" if score < -0.1 else "→"
                lines.append(f"│   {asset}: {score:+.2f} {indicator}")
            lines.append("│")

        if self.notable_numbers:
            lines.append("├── NOTABLE NUMBERS:")
            for key, value in self.notable_numbers.items():
                if value:
                    lines.append(f"│   • {key}: {value}")

        lines.append("└──")

        return "\n".join(lines)


def cluster_by_event_type(items: List[EnrichedNewsItem]) -> Dict[str, List[EnrichedNewsItem]]:
    """
    Group news items by event type.

    Args:
        items: List of enriched news items

    Returns:
        Dict mapping event_type to list of items
    """
    clusters: Dict[str, List[EnrichedNewsItem]] = {}

    for item in items:
        event_type = item.event_type
        if event_type not in clusters:
            clusters[event_type] = []
        clusters[event_type].append(item)

    return clusters


def deduplicate_by_content(
    items: List[EnrichedNewsItem],
    similarity_threshold: float = 0.7,
) -> List[EnrichedNewsItem]:
    """
    Deduplicate news items by content similarity.

    Uses simple word overlap for efficiency.

    Args:
        items: List of enriched news items
        similarity_threshold: Minimum similarity to consider duplicate

    Returns:
        Deduplicated list
    """
    if not items:
        return []

    def tokenize(text: str) -> set:
        words = text.lower().split()
        # Remove common words
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for"}
        return {w for w in words if w not in stopwords and len(w) > 2}

    def similarity(s1: set, s2: set) -> float:
        if not s1 or not s2:
            return 0.0
        intersection = len(s1 & s2)
        union = len(s1 | s2)
        return intersection / union if union > 0 else 0.0

    # Sort by source credibility (keep higher quality)
    sorted_items = sorted(items, key=lambda x: x.source_weight, reverse=True)

    unique = []
    seen_tokens = []

    for item in sorted_items:
        tokens = tokenize(f"{item.title} {item.snippet[:100]}")

        is_duplicate = False
        for seen in seen_tokens:
            if similarity(tokens, seen) >= similarity_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            unique.append(item)
            seen_tokens.append(tokens)

    return unique


def create_news_cluster(
    event_type: str,
    items: List[EnrichedNewsItem],
) -> NewsCluster:
    """
    Create a NewsCluster from a group of related items.

    Args:
        event_type: Event type for this cluster
        items: List of related news items

    Returns:
        NewsCluster summary
    """
    if not items:
        return NewsCluster(
            event_type=event_type,
            items=[],
            primary_headline="",
            sentiment="neutral",
            sentiment_score=0.0,
            affected_assets=[],
            source_count=0,
            tier1_sources=0,
        )

    # Pick primary headline (from most credible source)
    sorted_items = sorted(items, key=lambda x: x.source_weight, reverse=True)
    primary = sorted_items[0]

    # Aggregate sentiment
    sentiment_data = calculate_aggregate_sentiment(items)

    # Aggregate affected assets
    all_assets: List[str] = []
    for item in items:
        all_assets.extend(item.affected_assets)
    asset_counts = Counter(all_assets)
    top_assets = [asset for asset, _ in asset_counts.most_common(3)]

    return NewsCluster(
        event_type=event_type,
        items=items,
        primary_headline=primary.title,
        sentiment=sentiment_data["overall_sentiment"],
        sentiment_score=sentiment_data["weighted_score"],
        affected_assets=top_assets,
        source_count=len(items),
        tier1_sources=sum(1 for i in items if i.source_tier == 1),
    )


def aggregate_news(
    items: List[EnrichedNewsItem],
    from_dt: Optional[datetime] = None,
    to_dt: Optional[datetime] = None,
    previous_sentiment: Optional[float] = None,
) -> NewsDigest:
    """
    Aggregate news items into a digest.

    Args:
        items: List of enriched news items
        from_dt: Start of time window
        to_dt: End of time window
        previous_sentiment: Previous period's sentiment for trend

    Returns:
        NewsDigest summary
    """
    now = datetime.now(timezone.utc)
    from_dt = from_dt or (now - timedelta(hours=24))
    to_dt = to_dt or now

    if not items:
        return NewsDigest(from_dt=from_dt, to_dt=to_dt)

    # Deduplicate
    unique_items = deduplicate_by_content(items)

    # Overall sentiment
    sentiment_data = calculate_aggregate_sentiment(unique_items)

    # Sentiment trend
    if previous_sentiment is not None:
        diff = sentiment_data["weighted_score"] - previous_sentiment
        if diff > 0.1:
            trend = "improving"
        elif diff < -0.1:
            trend = "worsening"
        else:
            trend = "stable"
    else:
        trend = "stable"

    # Cluster by event type
    clusters_by_type = cluster_by_event_type(unique_items)
    key_events = []
    for event_type, cluster_items in clusters_by_type.items():
        if cluster_items:
            cluster = create_news_cluster(event_type, cluster_items)
            key_events.append(cluster)

    # Sort clusters by weight
    key_events.sort(key=lambda x: x.weight, reverse=True)

    # Dominant narratives
    event_type_counts = Counter(item.event_type for item in unique_items)
    narratives = [
        f"{event_type} ({count} articles)"
        for event_type, count in event_type_counts.most_common(3)
    ]

    # Asset sentiment
    asset_sentiment: Dict[str, List[float]] = {}
    for item in unique_items:
        for asset in item.affected_assets:
            if asset not in asset_sentiment:
                asset_sentiment[asset] = []
            asset_sentiment[asset].append(item.sentiment_score)

    avg_asset_sentiment = {
        asset: sum(scores) / len(scores)
        for asset, scores in asset_sentiment.items()
        if scores
    }

    # Source quality
    tier1_count = sum(1 for i in unique_items if i.source_tier == 1)
    tier1_pct = tier1_count / len(unique_items) if unique_items else 0.0
    avg_credibility = sum(i.source_weight for i in unique_items) / len(unique_items) if unique_items else 0.0

    # Notable numbers (aggregate)
    notable = {}
    for item in unique_items:
        for key, value in item.key_numbers.items():
            if value and key not in notable:
                notable[key] = value

    return NewsDigest(
        from_dt=from_dt,
        to_dt=to_dt,
        total_items=len(unique_items),
        overall_sentiment=sentiment_data["overall_sentiment"],
        sentiment_score=sentiment_data["weighted_score"],
        sentiment_trend=trend,
        bullish_count=sentiment_data["bullish_count"],
        bearish_count=sentiment_data["bearish_count"],
        neutral_count=sentiment_data["neutral_count"],
        dominant_narratives=narratives,
        key_events=key_events[:5],
        asset_sentiment=avg_asset_sentiment,
        tier1_percentage=tier1_pct,
        avg_source_credibility=avg_credibility,
        notable_numbers=notable,
    )
