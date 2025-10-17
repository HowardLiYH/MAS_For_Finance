"""LLM‑prompted Micro/Macro news search.
1) LLM plans micro/macro queries (JSON).
2) SerpAPI executes queries with time filters.
3) Strict post-filter by [from_dt, to_dt].
"""
from __future__ import annotations
import os, json, datetime as dt
from typing import List, Dict, Any
from openai import OpenAI
from pydantic import BaseModel, Field
from core.schemas import NewsItem
from .providers.search_serpapi import search_news_serpapi
import re

DEFAULT_MICRO = [
    '(bitcoin OR BTC) ("ETF" OR "exchange-traded fund")',
    '(bitcoin OR BTC) (liquidations OR "short squeeze" OR funding OR perps)',
    '(bitcoin OR BTC) (binance OR coinbase OR kraken OR "major exchange")',
    '(bitcoin OR BTC) (miners OR hashrate OR difficulty OR halving)',
    '(bitcoin OR BTC) (order book OR flows OR "open interest")',
    '(bitcoin OR BTC) (perpetual futures OR "BTCUSD perpetual")',
]

DEFAULT_MACRO = [
    '(bitcoin OR BTC) (Federal Reserve OR FOMC OR "interest rates")',
    '(bitcoin OR BTC) (CPI OR inflation OR "PCE")',
    '(bitcoin OR BTC) (SEC OR ETF approvals OR regulation)',
    '(bitcoin OR BTC) (Treasury OR yields OR "UST" OR "liquidity")',
    '(bitcoin OR BTC) (geopolitics OR sanctions OR BRICS OR "USD liquidity")',
    '(bitcoin OR BTC) (macro data OR jobs OR unemployment OR GDP)',
]

def _massage_queries(items: list[str], *, stream: str) -> list[str]:
    """Coerce LLM outputs into workable Google-News queries."""
    out = []
    for s in items or []:
        s0 = s.strip()
        # drop explicit date phrases the LLM sometimes adds
        s0 = re.sub(r"\bfrom\s+\d{4}-\d{2}-\d{2}\s+to\s+\d{4}-\d{2}-\d{2}\b", "", s0, flags=re.I)
        # ensure the bitcoin anchor is present
        if "bitcoin" not in s0.lower() and "btc" not in s0.lower():
            s0 = f"(bitcoin OR BTC) {s0}".strip()
        # if still looks like a sentence (no boolean cues), fall back to themed defaults
        if not re.search(r'\b(AND|OR|site:)\b', s0, flags=re.I):
            out.append(s0)  # keep it; provider will still try it
        else:
            out.append(s0)
    # Guarantee a healthy base set
    base = DEFAULT_MICRO if stream == "micro" else DEFAULT_MACRO
    if len(out) < 4:  # too few or too weak — backfill with strong defaults
        out = (out + base)[:8]
    return out


class QueryPlan(BaseModel):
    """
    BaseModel from pydantic is used here simply for checking element
    with the class.
    """
    micro: List[str] = Field(default_factory=list)
    macro: List[str] = Field(default_factory=list)

def _client() -> OpenAI:
    """
    Create an OpenAI client with provided key
    """
    key = os.getenv("OPENAI_API_KEY")
    if not key: raise RuntimeError("❌ OPENAI_API_KEY not set")
    print(f"Using provided OpenAI Key: {str(key)[:12]}xxxxxxx.....")

    base_url = os.getenv("OPENAI_API_BASE")
    if not base_url: raise RuntimeError("❌ OPENAI_BASE_URL not set")
    print(f"Using provided Base Url: {str(base_url)[:12]}xxxxxxx.....")

    base_url = base_url.rstrip("/")
    # guard against people pasting a full endpoint by mistake
    if base_url.endswith("/chat/completions") or base_url.endswith("/responses"):
        # strip the extra endpoint path
        base_url = re.sub(r"/(chat/completions|responses)$", "", base_url)
    return OpenAI(api_key=key, base_url=base_url)

def _llm_plan(topic: str, from_dt: dt.datetime, to_dt: dt.datetime, model: str) -> QueryPlan:
    prompt = f"""
    Plan *boolean web-search queries* to find BTC-related news STRICTLY within:
    FROM: {from_dt.isoformat()}
    TO:   {to_dt.isoformat()}

    Return JSON ONLY: {{"micro":[...], "macro":[...]}} where each item is a *search query string*:
    - Every query MUST contain: (bitcoin OR BTC)
    - No dates in the query text (the caller handles the time window)
    - Use boolean/keyword style, not sentences; no commentary
    - 6–10 queries per stream

    micro should cover: price/action, ETF flows, miners, exchanges, liquidations, funding, perps.
    macro should cover: Fed, CPI/inflation, SEC actions, Treasury, yields, geopolitics, USD liquidity.

    TOPIC: "{topic}"
    """

    print(f"Feeding the prompt:\n")
    print("-----------------------------------------------------------------")
    print(f"{prompt}")
    print("-----------------------------------------------------------------\n")
    # Tempreature = 0 means deterministic output, no randomness involved
    resp = _client().chat.completions.create(model=model, messages=[{"role":"user","content":prompt}], temperature=0, response_format={"type": "json_object"})
    print("\n-----------------------------------------------------------------\n")
    print(f"🕵️ Let's have a look at the response:\n {resp}")
    print("\n-----------------------------------------------------------------\n")
    # Pick the frist response
    txt = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(txt)
    except Exception as e:
        print(f"❌ Error when processing LLM Response:{e}\n News are set to empty.....")
        data = {"micro": [], "macro": []}
    return QueryPlan(**data)

def _to_items(rows: List[Dict[str, Any]], symbol_hint: str) -> List[NewsItem]:
    out = []
    for r in rows:
        ts = r.get("published_at")
        if ts:  # already normalized to UTC
            out.append(NewsItem(source="web", title=r.get("title",""), url=r.get("url",""),
                                published_at=ts, tickers=[f"{symbol_hint}-USD"], summary=r.get("snippet","")))
    return out

def search_micro_macro(*, topic: str, from_dt: dt.datetime, to_dt: dt.datetime,
                                      max_per_stream: int, provider: str="serpapi",
                                      llm_model: str="gpt-4o-mini", symbol_hint: str="BTC") -> Dict[str, List[NewsItem]]:


    plan = _llm_plan(topic, from_dt, to_dt, llm_model)

    def run(queries: List[str], stream: str) -> List[Dict[str, Any]]:
        seen, acc = set(), []
        qs = _massage_queries(queries, stream=stream)
        for q in qs:
            rows = search_news_serpapi(q, from_dt, to_dt, limit=50) if provider == "serpapi" else []
            for r in rows:
                u = r.get("url")
                if u and u not in seen:
                    seen.add(u); acc.append(r)
            if len(acc) >= max_per_stream * 2:
                break
        # If still sparse, append strong defaults not already tried
        if len(acc) < max_per_stream:
            defaults = DEFAULT_MICRO if stream == "micro" else DEFAULT_MACRO
            extra = [x for x in defaults if x not in qs]
            for q in extra:
                rows = search_news_serpapi(q, from_dt, to_dt, limit=50) if provider == "serpapi" else []
                for r in rows:
                    u = r.get("url")
                    if u and u not in seen:
                        seen.add(u); acc.append(r)
                if len(acc) >= max_per_stream * 2:
                    break
        return acc

    micro_rows = run(plan.micro, "micro")
    macro_rows = run(plan.macro, "macro")
    micro = _to_items(micro_rows, symbol_hint)[:max_per_stream]
    macro = _to_items(macro_rows, symbol_hint)[:max_per_stream]
    return {"micro": micro, "macro": macro}
