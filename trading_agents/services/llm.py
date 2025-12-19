"""LLM service for Trader agent decision-making."""
from __future__ import annotations
import os
import json
import re
from typing import Dict, Any, List, Optional
from openai import OpenAI


def _create_openai_client() -> OpenAI:
    """Create OpenAI client with environment variables."""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")

    base_url = os.getenv("OPENAI_API_BASE")
    if base_url:
        base_url = base_url.rstrip("/")
        if base_url.endswith("/chat/completions") or base_url.endswith("/responses"):
            base_url = re.sub(r"/(chat/completions|responses)$", "", base_url)
        return OpenAI(api_key=key, base_url=base_url)
    else:
        return OpenAI(api_key=key)


def format_news_digest(news_digest: Optional[Dict[str, Any]]) -> str:
    """
    Format news digest for LLM prompt.

    Args:
        news_digest: NewsDigest.to_dict() output or None

    Returns:
        Formatted string for prompt
    """
    if not news_digest:
        return "No news digest available."

    lines = [
        f"NEWS DIGEST ({news_digest.get('total_items', 0)} articles)",
        f"├── Overall Sentiment: {news_digest.get('sentiment_score', 0):+.2f} ({news_digest.get('overall_sentiment', 'neutral').upper()})",
        f"├── Trend: {news_digest.get('sentiment_trend', 'stable')}",
        f"├── Source Quality: {news_digest.get('tier1_percentage', 0):.0%} tier-1 sources",
        "│",
    ]

    # Dominant narratives
    narratives = news_digest.get("dominant_narratives", [])
    if narratives:
        lines.append("├── DOMINANT NARRATIVES:")
        for i, narrative in enumerate(narratives[:3], 1):
            lines.append(f"│   {i}. {narrative}")
        lines.append("│")

    # Key events
    events = news_digest.get("key_events", [])
    if events:
        lines.append("├── KEY EVENTS:")
        for event in events[:5]:
            emoji = "🟢" if event.get("sentiment") == "bullish" else "🔴" if event.get("sentiment") == "bearish" else "⚪"
            lines.append(f"│   {emoji} [{event.get('event_type', 'general').upper()}] {event.get('headline', '')[:80]}")
        lines.append("│")

    # Asset sentiment
    asset_sentiment = news_digest.get("asset_sentiment", {})
    if asset_sentiment:
        lines.append("├── ASSET SENTIMENT:")
        for asset, score in sorted(asset_sentiment.items(), key=lambda x: x[1], reverse=True):
            indicator = "↑" if score > 0.1 else "↓" if score < -0.1 else "→"
            lines.append(f"│   {asset}: {score:+.2f} {indicator}")

    lines.append("└──")

    return "\n".join(lines)


def format_news_items_legacy(news_items: List[Dict[str, Any]]) -> str:
    """
    Format raw news items for LLM prompt (legacy format).

    Args:
        news_items: List of raw news item dicts

    Returns:
        Formatted string for prompt
    """
    if not news_items:
        return "No relevant news items available."

    return "\n".join([
        f"- [{item.get('source', 'unknown')}] {item.get('title', '')}: {item.get('summary', '')[:200]}"
        for item in news_items[:20]
    ])


def generate_trading_proposal(
    execution_style: str,
    research_summary: Dict[str, Any],
    news_items: List[Dict[str, Any]],
    price_data_summary: Dict[str, Any],
    current_price: float,
    model: str = "gpt-4o-mini",
    news_digest: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, Any], str]:
    """
    Generate trading proposal using LLM.

    Args:
        execution_style: Trading style (e.g., "Aggressive_Market")
        research_summary: Research output from Researcher agent
        news_items: List of news items (legacy format)
        price_data_summary: Current price data summary
        current_price: Current market price
        model: LLM model to use
        news_digest: Optional NewsDigest dict (enhanced format)

    Returns:
        tuple: (parsed_proposal_dict, thought_process_text)
    """
    # Use enhanced news digest if available, otherwise legacy format
    if news_digest:
        news_text = format_news_digest(news_digest)
    else:
        news_text = format_news_items_legacy(news_items)

    # Format price data
    price_summary = f"""
Current Price: ${current_price:,.2f}
Recent Range: ${price_data_summary.get('low', current_price):,.2f} - ${price_data_summary.get('high', current_price):,.2f}
Volume: {price_data_summary.get('volume', 0):,.0f}
"""

    # Format research summary
    research_text = f"""
Market State: {research_summary.get('market_state', 'unknown')}
Recommendation: {research_summary.get('recommendation', 'HOLD')}
Confidence: {research_summary.get('confidence', 0.5):.2%}
Forecast (8h): {research_summary.get('forecast', {}).get('8h', 0.0):.4%}
Forecast (24h): {research_summary.get('forecast', {}).get('24h', 0.0):.4%}
Risk Metrics: {json.dumps(research_summary.get('risk', {}), indent=2)}
"""

    prompt = f"""You are a professional cryptocurrency trader specializing in BTC perpetual futures.

EXECUTION STYLE: {execution_style}

CURRENT MARKET DATA:
{price_summary}

RESEARCH SUMMARY:
{research_text}

{news_text}

Based on the above information, provide a trading proposal in JSON format:
{{
  "direction": "LONG" or "SHORT",
  "position_size": float (0.0 to 1.0),
  "leverage": float (1.0 to 10.0),
  "order_type": "MARKET" or "LIMIT",
  "entry_price": float,
  "take_profit": float,
  "stop_loss": float,
  "execution_expired_time": "ISO8601 datetime" or null,
  "reasoning": "Brief explanation including news impact assessment"
}}

TRADING RULES:
- Position size proportional to confidence (max 0.5)
- Conservative leverage (1-5x recommended)
- Realistic take profit (1-5% for 4h timeframe)
- Tight stop loss (0.5-2% for 4h timeframe)
- LONG: take_profit > entry_price > stop_loss
- SHORT: stop_loss > entry_price > take_profit

NEWS INTEGRATION RULES:
- Bullish news sentiment (>0.3) supports LONG positions
- Bearish news sentiment (<-0.3) supports SHORT positions
- High-impact events (ETF flows, regulation) should influence position sizing
- Low source quality (<50% tier-1) = reduce confidence
- If news contradicts technicals, reduce position size

Return ONLY valid JSON.
"""

    try:
        client = _create_openai_client()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a professional cryptocurrency trader. Respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        response_text = response.choices[0].message.content or "{}"
        thought_process = response_text

        try:
            proposal = json.loads(response_text)
        except json.JSONDecodeError as e:
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                proposal = json.loads(json_match.group(1))
            else:
                raise ValueError(f"Failed to parse LLM response: {e}")

        return proposal, thought_process

    except Exception as e:
        print(f"⚠️ LLM call failed: {e}, using fallback")
        return _fallback_proposal(research_summary, current_price, news_digest), f"LLM failed: {e}"


def _fallback_proposal(
    research_summary: Dict[str, Any],
    current_price: float,
    news_digest: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Fallback rule-based proposal."""
    direction = "LONG" if research_summary.get("recommendation") == "BUY" else "SHORT"
    confidence = research_summary.get("confidence", 0.5)

    # Adjust confidence based on news sentiment
    if news_digest:
        news_sentiment = news_digest.get("sentiment_score", 0)
        if (direction == "LONG" and news_sentiment > 0) or (direction == "SHORT" and news_sentiment < 0):
            confidence = min(1.0, confidence + abs(news_sentiment) * 0.1)
        elif (direction == "LONG" and news_sentiment < -0.3) or (direction == "SHORT" and news_sentiment > 0.3):
            confidence = max(0.1, confidence - 0.2)

    position_size = 0.2 if confidence < 0.6 else 0.5
    leverage = 3.0 if confidence >= 0.6 else 2.0

    forecast_24h = research_summary.get("forecast", {}).get("24h", 0.005)
    risk_q05 = research_summary.get("risk", {}).get("q05", -0.01)
    risk_q95 = research_summary.get("risk", {}).get("q95", 0.01)

    if direction == "LONG":
        take_profit = current_price * (1 + abs(forecast_24h) * 2)
        stop_loss = current_price * (1 + risk_q05)
    else:
        take_profit = current_price * (1 - abs(forecast_24h) * 2)
        stop_loss = current_price * (1 + risk_q95)

    return {
        "direction": direction,
        "position_size": position_size,
        "leverage": leverage,
        "order_type": "MARKET",
        "entry_price": current_price,
        "take_profit": take_profit,
        "stop_loss": stop_loss,
        "execution_expired_time": None,
        "reasoning": f"Fallback rule-based logic (confidence: {confidence:.2f})"
    }
