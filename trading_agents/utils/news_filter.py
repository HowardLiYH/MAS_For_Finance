"""News filtering utilities."""
from __future__ import annotations
from typing import List, Dict, Any
from datetime import datetime, timezone


def basic_filter(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter out items without titles."""
    return [item for item in items if item.get("title")]


def enhanced_filter(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate and filter short titles."""
    out = []
    seen = set()

    for item in items:
        url = item.get("url")
        title = item.get("title", "").strip()

        if not url or url in seen:
            continue

        seen.add(url)

        if len(title) < 5:
            continue

        out.append(item)

    return out


def integrated_filter(
    items: List[Dict[str, Any]],
    *,
    from_dt: datetime,
    to_dt: datetime,
) -> List[Dict[str, Any]]:
    """Filter by date range."""
    out = []

    for item in items:
        ts = item.get("published_at")

        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except Exception:
                ts = None

        if not ts or ts.tzinfo is None:
            continue

        ts = ts.astimezone(timezone.utc)

        if from_dt <= ts <= to_dt:
            out.append(item)

    return out


def filter_news_3_stage(
    items: List[Dict[str, Any]],
    *,
    from_dt: datetime,
    to_dt: datetime,
) -> List[Dict[str, Any]]:
    """
    Three-stage news filtering pipeline.

    1. Basic filter: Remove items without titles
    2. Enhanced filter: Deduplicate and filter short titles
    3. Integrated filter: Filter by date range
    """
    return integrated_filter(
        enhanced_filter(basic_filter(items)),
        from_dt=from_dt,
        to_dt=to_dt,
    )
