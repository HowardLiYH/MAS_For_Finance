"""Source credibility scoring for news articles.

Assigns credibility tiers to news sources based on their reputation
and reliability for crypto/financial news.
"""
from typing import Dict, Optional
from urllib.parse import urlparse


# Source credibility tiers
# Tier 1: Major financial news (highest credibility)
# Tier 2: Established crypto media
# Tier 3: General tech/news
# Tier 4: Social media / blogs
# Tier 5: Unknown sources

SOURCE_TIERS: Dict[str, int] = {
    # Tier 1: Institutional Financial News (weight: 1.0)
    "bloomberg.com": 1,
    "reuters.com": 1,
    "wsj.com": 1,
    "ft.com": 1,
    "cnbc.com": 1,
    "marketwatch.com": 1,
    "barrons.com": 1,
    "forbes.com": 1,
    "businessinsider.com": 1,
    "yahoo.com": 1,  # Yahoo Finance

    # Tier 2: Established Crypto Media (weight: 0.85)
    "coindesk.com": 2,
    "theblock.co": 2,
    "cointelegraph.com": 2,
    "decrypt.co": 2,
    "blockworks.co": 2,
    "coinmarketcap.com": 2,
    "coingecko.com": 2,
    "messari.io": 2,
    "defiant.io": 2,
    "unchainedcrypto.com": 2,

    # Tier 3: General Tech/News (weight: 0.7)
    "techcrunch.com": 3,
    "wired.com": 3,
    "theverge.com": 3,
    "arstechnica.com": 3,
    "vice.com": 3,
    "bbc.com": 3,
    "nytimes.com": 3,
    "theguardian.com": 3,
    "washingtonpost.com": 3,

    # Tier 4: Crypto Aggregators / Social (weight: 0.5)
    "twitter.com": 4,
    "x.com": 4,
    "reddit.com": 4,
    "medium.com": 4,
    "substack.com": 4,
    "bitcoinmagazine.com": 4,
    "u.today": 4,
    "ambcrypto.com": 4,
    "newsbtc.com": 4,
    "beincrypto.com": 4,

    # Tier 5: Unknown/Low credibility (weight: 0.3)
    # Default for unrecognized sources
}

# Credibility weights by tier
TIER_WEIGHTS = {
    1: 1.0,
    2: 0.85,
    3: 0.7,
    4: 0.5,
    5: 0.3,
}


def extract_domain(url: str) -> str:
    """
    Extract the base domain from a URL.

    Args:
        url: Full URL string

    Returns:
        Base domain (e.g., "bloomberg.com")
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        # Remove www prefix
        if domain.startswith("www."):
            domain = domain[4:]

        # Handle subdomains (keep main domain)
        parts = domain.split(".")
        if len(parts) > 2:
            # Keep last two parts (e.g., "finance.yahoo.com" -> "yahoo.com")
            domain = ".".join(parts[-2:])

        return domain
    except Exception:
        return ""


def get_source_tier(url: str) -> int:
    """
    Get the credibility tier for a news source.

    Args:
        url: URL of the news article

    Returns:
        Tier number (1-5, lower is more credible)
    """
    domain = extract_domain(url)
    return SOURCE_TIERS.get(domain, 5)


def get_source_weight(url: str) -> float:
    """
    Get the credibility weight for a news source.

    Args:
        url: URL of the news article

    Returns:
        Weight between 0.0 and 1.0
    """
    tier = get_source_tier(url)
    return TIER_WEIGHTS.get(tier, 0.3)


def get_source_name(url: str) -> str:
    """
    Get a readable source name from URL.

    Args:
        url: URL of the news article

    Returns:
        Readable source name
    """
    domain = extract_domain(url)

    # Map domains to readable names
    NAME_MAP = {
        "bloomberg.com": "Bloomberg",
        "reuters.com": "Reuters",
        "wsj.com": "Wall Street Journal",
        "ft.com": "Financial Times",
        "cnbc.com": "CNBC",
        "coindesk.com": "CoinDesk",
        "theblock.co": "The Block",
        "cointelegraph.com": "Cointelegraph",
        "decrypt.co": "Decrypt",
        "twitter.com": "Twitter/X",
        "x.com": "Twitter/X",
        "reddit.com": "Reddit",
    }

    if domain in NAME_MAP:
        return NAME_MAP[domain]

    # Capitalize domain
    return domain.replace(".com", "").replace(".co", "").title()


def filter_by_credibility(
    items: list,
    min_tier: int = 4,
    url_key: str = "url",
) -> list:
    """
    Filter news items by source credibility.

    Args:
        items: List of news items (dicts)
        min_tier: Minimum tier to include (1-5)
        url_key: Key for URL in item dict

    Returns:
        Filtered list of items
    """
    return [
        item for item in items
        if get_source_tier(item.get(url_key, "")) <= min_tier
    ]


def sort_by_credibility(
    items: list,
    url_key: str = "url",
    descending: bool = True,
) -> list:
    """
    Sort news items by source credibility.

    Args:
        items: List of news items (dicts)
        url_key: Key for URL in item dict
        descending: If True, most credible first

    Returns:
        Sorted list of items
    """
    return sorted(
        items,
        key=lambda x: get_source_weight(x.get(url_key, "")),
        reverse=descending,
    )


def calculate_aggregate_credibility(items: list, url_key: str = "url") -> float:
    """
    Calculate aggregate credibility score for a set of news items.

    Args:
        items: List of news items
        url_key: Key for URL in item dict

    Returns:
        Aggregate credibility score (0.0 to 1.0)
    """
    if not items:
        return 0.0

    weights = [get_source_weight(item.get(url_key, "")) for item in items]
    return sum(weights) / len(weights)
