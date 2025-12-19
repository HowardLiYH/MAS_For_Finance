"""Web search via Bocha AI API.

Bocha is a Chinese search API that provides web search results with
structured data including titles, URLs, snippets, and publication dates.

API Documentation: https://github.com/BochaAI/bocha-search-mcp
"""
import os
import json
import requests
import datetime as dt
from typing import List, Dict, Any, Optional
from dateutil import parser as dtparser

BOCHA_ENDPOINT = "https://api.bochaai.com/v1/web-search"


def _calculate_freshness(from_dt: dt.datetime, to_dt: dt.datetime) -> str:
    """
    Calculate the freshness parameter based on date range.
    
    Bocha supports: noLimit, oneDay, oneWeek, oneMonth, oneYear
    or specific dates: YYYY-MM-DD, YYYY-MM-DD..YYYY-MM-DD
    """
    now = dt.datetime.now(dt.timezone.utc)
    days_back = (now - from_dt).days
    
    if days_back <= 1:
        return "oneDay"
    elif days_back <= 7:
        return "oneWeek"
    elif days_back <= 30:
        return "oneMonth"
    elif days_back <= 365:
        return "oneYear"
    else:
        return "noLimit"


def _parse_bocha_date(date_str: str) -> Optional[dt.datetime]:
    """
    Parse Bocha's ISO format date string.
    
    Bocha returns dates like: "2025-12-17T05:00:00+08:00"
    """
    if not date_str:
        return None
    
    try:
        parsed = dtparser.parse(date_str)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.astimezone(dt.timezone.utc)
    except Exception:
        return None


def search_news_bocha(
    query: str,
    from_dt: dt.datetime,
    to_dt: dt.datetime,
    limit: int = 50,
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Search news via Bocha AI API.
    
    Args:
        query: Search query string
        from_dt: Start of date range (UTC)
        to_dt: End of date range (UTC)
        limit: Maximum number of results (1-50)
        api_key: Bocha API key (or use BOCHA_API_KEY env var)
        
    Returns:
        List of dicts with keys: url, title, snippet, published_at
    """
    api_key = api_key or os.getenv("BOCHA_API_KEY")
    if not api_key:
        raise RuntimeError("BOCHA_API_KEY not set")
    
    freshness = _calculate_freshness(from_dt, to_dt)
    
    payload = {
        "query": query,
        "summary": False,
        "freshness": freshness,
        "count": min(50, max(1, limit)),
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    try:
        response = requests.post(
            BOCHA_ENDPOINT,
            headers=headers,
            data=json.dumps(payload),
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"[Bocha] Request failed: {e}")
        return []
    except json.JSONDecodeError as e:
        print(f"[Bocha] JSON decode failed: {e}")
        return []
    
    # Extract results from Bocha response structure
    # Response: {"code": 200, "data": {"webPages": {"value": [...]}}}
    try:
        web_pages = data.get("data", {}).get("webPages", {}).get("value", [])
    except (AttributeError, TypeError):
        print(f"[Bocha] Unexpected response structure: {data}")
        return []
    
    rows = []
    for item in web_pages:
        url = item.get("url")
        title = item.get("name", "")
        snippet = item.get("snippet", "")
        date_str = item.get("datePublished") or item.get("dateLastCrawled")
        
        ts = _parse_bocha_date(date_str)
        
        if not url:
            continue
        
        # If we have a valid timestamp, filter by date range
        if ts:
            if from_dt <= ts <= to_dt:
                rows.append({
                    "url": url,
                    "title": title,
                    "snippet": snippet,
                    "published_at": ts,
                })
        else:
            # If no timestamp, include but mark as None
            # The caller can decide whether to use it
            rows.append({
                "url": url,
                "title": title,
                "snippet": snippet,
                "published_at": None,
            })
        
        if len(rows) >= limit:
            break
    
    return rows


def search_news_bocha_ai(
    query: str,
    limit: int = 10,
    api_key: Optional[str] = None,
    answer: bool = False,
) -> Dict[str, Any]:
    """
    Search using Bocha AI Search API (with modal cards).
    
    This is the enhanced version that returns structured data
    for weather, stocks, etc.
    
    Args:
        query: Search query string
        limit: Maximum number of results
        api_key: Bocha API key
        answer: Whether to include LLM summary
        
    Returns:
        Full response dict including modal cards
    """
    api_key = api_key or os.getenv("BOCHA_API_KEY")
    if not api_key:
        raise RuntimeError("BOCHA_API_KEY not set")
    
    payload = {
        "query": query,
        "answer": answer,
        "stream": False,
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    try:
        response = requests.post(
            "https://api.bochaai.com/v1/ai-search",
            headers=headers,
            data=json.dumps(payload),
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"[Bocha AI] Request failed: {e}")
        return {}


class BochaSearchProvider:
    """
    Bocha search provider class for compatibility with existing code.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("BOCHA_API_KEY")
    
    def search(
        self,
        query: str,
        from_dt: dt.datetime,
        to_dt: dt.datetime,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Search for news articles."""
        return search_news_bocha(
            query=query,
            from_dt=from_dt,
            to_dt=to_dt,
            limit=limit,
            api_key=self.api_key,
        )

