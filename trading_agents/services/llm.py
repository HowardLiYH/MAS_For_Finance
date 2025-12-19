"""LLM service for Trader agent decision-making."""
from __future__ import annotations
import os
import json
import re
from typing import Dict, Any
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


def generate_trading_proposal(
    execution_style: str,
    research_summary: Dict[str, Any],
    news_items: list[Dict[str, Any]],
    price_data_summary: Dict[str, Any],
    current_price: float,
    model: str = "gpt-4o-mini",
) -> tuple[Dict[str, Any], str]:
    """
    Generate trading proposal using LLM.

    Args:
        execution_style: Trading style (e.g., "Aggressive_Market")
        research_summary: Research output from Researcher agent
        news_items: List of news items
        price_data_summary: Current price data summary
        current_price: Current market price
        model: LLM model to use

    Returns:
        tuple: (parsed_proposal_dict, thought_process_text)
    """
    # Format news items
    if news_items:
        news_text = "\n".join([
            f"- [{item.get('source', 'unknown')}] {item.get('title', '')}: {item.get('summary', '')[:200]}"
            for item in news_items[:20]
        ])
    else:
        news_text = "No relevant news items available."

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

RELEVANT NEWS:
{news_text}

Provide a trading proposal in JSON format:
{{
  "direction": "LONG" or "SHORT",
  "position_size": float (0.0 to 1.0),
  "leverage": float (1.0 to 10.0),
  "order_type": "MARKET" or "LIMIT",
  "entry_price": float,
  "take_profit": float,
  "stop_loss": float,
  "execution_expired_time": "ISO8601 datetime" or null,
  "reasoning": "Brief explanation"
}}

CONSTRAINTS:
- Position size proportional to confidence (max 0.5)
- Conservative leverage (1-5x recommended)
- Realistic take profit (1-5% for 4h timeframe)
- Tight stop loss (0.5-2% for 4h timeframe)
- LONG: take_profit > entry_price > stop_loss
- SHORT: stop_loss > entry_price > take_profit

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
        return _fallback_proposal(research_summary, current_price), f"LLM failed: {e}"


def _fallback_proposal(research_summary: Dict[str, Any], current_price: float) -> Dict[str, Any]:
    """Fallback rule-based proposal."""
    direction = "LONG" if research_summary.get("recommendation") == "BUY" else "SHORT"
    confidence = research_summary.get("confidence", 0.5)
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
        "reasoning": "Fallback rule-based logic"
    }
