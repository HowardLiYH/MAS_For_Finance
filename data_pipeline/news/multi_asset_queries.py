"""Multi-asset query templates for news search.

Provides asset-specific search queries for micro (market-specific)
and macro (economic/regulatory) news streams.
"""
from typing import Dict, List

# Asset-specific query templates
ASSET_QUERIES: Dict[str, Dict[str, List[str]]] = {
    "BTC": {
        "micro": [
            '(bitcoin OR BTC) (ETF flows OR "ETF inflow" OR "ETF outflow" OR BlackRock OR Fidelity)',
            '(bitcoin OR BTC) (whale OR "large transfer" OR "wallet movement")',
            '(bitcoin OR BTC) (mining OR hashrate OR difficulty OR halving)',
            '(bitcoin OR BTC) (liquidation OR "short squeeze" OR "long squeeze" OR funding)',
            '(bitcoin OR BTC) (exchange OR binance OR coinbase OR kraken) (withdrawal OR deposit)',
            '(bitcoin OR BTC) ("open interest" OR futures OR perpetual OR derivatives)',
            '(bitcoin OR BTC) (MicroStrategy OR Tesla OR "corporate treasury")',
        ],
        "macro": [
            '(bitcoin OR BTC) (Federal Reserve OR FOMC OR "interest rate" OR Powell)',
            '(bitcoin OR BTC) (SEC OR Gensler OR regulation OR "crypto bill")',
            '(bitcoin OR BTC) (CPI OR inflation OR "consumer price")',
            '(bitcoin OR BTC) (Treasury OR yields OR bonds OR "USD liquidity")',
            '(bitcoin OR BTC) (geopolitics OR sanctions OR "safe haven")',
            '(bitcoin OR BTC) (El Salvador OR "legal tender" OR adoption)',
        ],
    },
    "ETH": {
        "micro": [
            '(ethereum OR ETH) (staking OR "validator" OR "beacon chain" OR withdrawals)',
            '(ethereum OR ETH) ("layer 2" OR L2 OR Arbitrum OR Optimism OR Base)',
            '(ethereum OR ETH) (gas OR fees OR burn OR EIP-1559)',
            '(ethereum OR ETH) (DeFi OR TVL OR "total value locked")',
            '(ethereum OR ETH) (NFT OR OpenSea OR marketplace)',
            '(ethereum OR ETH) (whale OR "large transfer" OR foundation)',
            '(ethereum OR ETH) (ETF OR "spot ETF" OR BlackRock)',
        ],
        "macro": [
            '(ethereum OR ETH) (SEC OR securities OR lawsuit OR regulation)',
            '(ethereum OR ETH) (Vitalik OR "Ethereum Foundation" OR roadmap)',
            '(ethereum OR ETH) ("proof of stake" OR merge OR upgrade)',
            '(ethereum OR ETH) (institutional OR "enterprise adoption")',
        ],
    },
    "SOL": {
        "micro": [
            '(solana OR SOL) (network OR outage OR TPS OR performance)',
            '(solana OR SOL) (NFT OR "Magic Eden" OR marketplace)',
            '(solana OR SOL) (DeFi OR TVL OR Jupiter OR Raydium)',
            '(solana OR SOL) (whale OR "large transfer" OR foundation)',
            '(solana OR SOL) (meme OR memecoin OR "pump fun")',
            '(solana OR SOL) (Firedancer OR validator OR upgrade)',
        ],
        "macro": [
            '(solana OR SOL) (FTX OR Alameda OR unlock OR bankruptcy)',
            '(solana OR SOL) (ETF OR "spot ETF" OR institutional)',
            '(solana OR SOL) (regulation OR SEC)',
        ],
    },
    "DOGE": {
        "micro": [
            '(dogecoin OR DOGE) (Elon OR Musk OR Tesla OR Twitter OR X)',
            '(dogecoin OR DOGE) (whale OR "large transfer")',
            '(dogecoin OR DOGE) (payment OR merchant OR adoption)',
            '(dogecoin OR DOGE) (mining OR hashrate)',
        ],
        "macro": [
            '(dogecoin OR DOGE) (regulation OR SEC)',
            '(dogecoin OR DOGE) (retail OR "meme coin" OR sentiment)',
        ],
    },
    "XRP": {
        "micro": [
            '(ripple OR XRP) (ODL OR "on-demand liquidity" OR remittance)',
            '(ripple OR XRP) (whale OR "large transfer" OR escrow)',
            '(ripple OR XRP) (partnership OR bank OR "financial institution")',
            '(ripple OR XRP) (listing OR exchange)',
        ],
        "macro": [
            '(ripple OR XRP) (SEC OR lawsuit OR Hinman OR ruling)',
            '(ripple OR XRP) (regulation OR "programmatic sales")',
            '(ripple OR XRP) (IPO OR "Ripple Labs")',
        ],
    },
}

# General crypto market queries (applies to all assets)
GENERAL_CRYPTO_QUERIES = {
    "micro": [
        'crypto ("market cap" OR "total market" OR dominance)',
        'crypto (stablecoin OR USDT OR USDC OR Tether OR Circle)',
        'crypto (exchange OR CEX OR DEX) volume',
    ],
    "macro": [
        'crypto (regulation OR "crypto bill" OR Congress OR Senate)',
        'crypto (Fed OR FOMC OR "interest rates" OR liquidity)',
        'crypto (institutional OR "hedge fund" OR "asset manager")',
        'crypto (CBDC OR "central bank digital currency")',
    ],
}


def get_queries_for_asset(
    asset: str,
    stream: str = "micro",
    include_general: bool = True,
) -> List[str]:
    """
    Get search queries for a specific asset.

    Args:
        asset: Asset symbol (BTC, ETH, SOL, DOGE, XRP)
        stream: "micro" or "macro"
        include_general: Whether to include general crypto queries

    Returns:
        List of search query strings
    """
    asset = asset.upper()
    queries = []

    # Asset-specific queries
    if asset in ASSET_QUERIES:
        queries.extend(ASSET_QUERIES[asset].get(stream, []))
    else:
        # Fallback for unknown assets
        queries.append(f'({asset}) (price OR news OR update)')

    # Add general crypto queries
    if include_general:
        queries.extend(GENERAL_CRYPTO_QUERIES.get(stream, []))

    return queries


def get_queries_for_multi_asset(
    assets: List[str],
    stream: str = "micro",
    max_per_asset: int = 4,
) -> List[str]:
    """
    Get queries for multiple assets, balanced across all.

    Args:
        assets: List of asset symbols
        stream: "micro" or "macro"
        max_per_asset: Max queries per asset

    Returns:
        Combined list of queries
    """
    queries = []

    for asset in assets:
        asset_queries = get_queries_for_asset(asset, stream, include_general=False)
        queries.extend(asset_queries[:max_per_asset])

    # Add some general queries
    queries.extend(GENERAL_CRYPTO_QUERIES.get(stream, [])[:3])

    return queries


# Event type detection patterns
EVENT_PATTERNS = {
    "etf_flow": ["ETF", "inflow", "outflow", "BlackRock", "Fidelity", "Grayscale"],
    "whale_move": ["whale", "large transfer", "wallet", "dormant"],
    "exchange": ["exchange", "listing", "delisting", "binance", "coinbase"],
    "regulation": ["SEC", "regulation", "lawsuit", "Gensler", "Congress"],
    "hack": ["hack", "exploit", "breach", "stolen", "vulnerability"],
    "macro": ["Fed", "FOMC", "CPI", "inflation", "interest rate"],
    "mining": ["mining", "hashrate", "difficulty", "halving"],
    "defi": ["DeFi", "TVL", "yield", "liquidity", "protocol"],
    "nft": ["NFT", "OpenSea", "marketplace", "collection"],
    "adoption": ["adoption", "payment", "merchant", "partnership"],
}


def detect_event_type(title: str, snippet: str) -> str:
    """
    Detect the event type from news title and snippet.

    Args:
        title: News title
        snippet: News snippet/summary

    Returns:
        Event type string
    """
    text = f"{title} {snippet}".lower()

    scores = {}
    for event_type, patterns in EVENT_PATTERNS.items():
        score = sum(1 for p in patterns if p.lower() in text)
        if score > 0:
            scores[event_type] = score

    if scores:
        return max(scores, key=scores.get)

    return "general"
