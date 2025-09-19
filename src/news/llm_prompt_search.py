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
    Plan news web-search queries to find BTC-related news STRICTLY within:
    FROM: {from_dt.isoformat()}
    TO:   {to_dt.isoformat()}

    Return JSON ONLY: {{"micro":[...], "macro":[...]}}
    - micro: BTC price/action, ETF flows, miners, exchanges, liquidations, funding, perps.
    - macro: Fed, CPI/inflation, SEC actions, Treasury, yields, geopolitics, USD liquidity.
    Rules: 6–10 queries per stream; specific & time-aware; no commentary.
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

    def run(queries: List[str]) -> List[Dict[str, Any]]:
        """
        Here the function is tuned to Serpapi sepcifcially.
        Can later change to Google Customized Search API (Free Tier: 100/day)
        """
        seen, acc = set(), []
        for q in queries:
            # Search with Serpapi
            rows = search_news_serpapi(q, from_dt, to_dt, limit=50) if provider=="serpapi" else []
            for r in rows:
                # include urls
                u = r.get("url")
                if u and u not in seen:
                    seen.add(u); acc.append(r)
            if len(acc) >= max_per_stream*2: break
        return acc

    micro_rows = run(plan.micro)
    macro_rows = run(plan.macro)
    micro = _to_items(micro_rows, symbol_hint)[:max_per_stream]
    macro = _to_items(macro_rows, symbol_hint)[:max_per_stream]
    return {"micro": micro, "macro": macro}
