
"""Phase 1.1 core pipeline with CCXT prices + (optional) LLM dual‑stream news.

- Supports following news modes:
LLM prompt + real OpenAI + SerpAPI (if keys available)
TODO: Modify here and see if this varibable should be deleted
  -> set use_llm_news_prompt=True (and use_llm_news_real or OPENAI_API_KEY present)

- Writes:
  data/prices_<SYMBOL>_<TF>.csv
  data/news_micro.json
  data/news_macro.json
"""
from __future__ import annotations

import os, json, math
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, timezone

import pandas as pd
import news.llm_prompt_search as llmps

try:
    import ccxt
except Exception:
    ccxt = None

from dateutil import parser as dtparser
from core.schemas import PriceBar, NewsItem

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def _read_csv_prices(path: str | Path) -> list[PriceBar]:
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True)
    elif "time" in df.columns:
        ts = pd.to_datetime(df["time"], unit="ms", utc=True)
    else:
        raise ValueError("CSV must contain a 'timestamp' or 'time' column")
    cols = {c.lower(): c for c in df.columns}
    def pick(name): return df[cols[name]]
    out: list[PriceBar] = []
    for i in range(len(df)):
        out.append(PriceBar(
            timestamp=_ensure_utc(ts.iloc[i].to_pydatetime()),
            open=float(pick("open").iloc[i]),
            high=float(pick("high").iloc[i]),
            low=float(pick("low").iloc[i]),
            close=float(pick("close").iloc[i]),
            volume=float(pick("volume").iloc[i]),
        ))
    return out

def _fetch_ccxt_ohlcv(exchange_id: str, symbol: str, timeframe: str, since_ms: int, limit: int = 1000) -> list[list]:
    if ccxt is None:
        raise RuntimeError("ccxt not available")
    ex_class = getattr(ccxt, exchange_id, None)
    if ex_class is None:
        raise ValueError(f"Unknown exchange_id '{exchange_id}'")
    ex = ex_class({'enableRateLimit': True})
    if not hasattr(ex, 'fetchOHLCV'):
        raise RuntimeError(f"Exchange {exchange_id} lacks fetchOHLCV")
    return ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)

def _prices_from_ccxt(exchange_id: str, symbol: str, timeframe: str, max_days: int) -> list[PriceBar]:
    end = _utcnow()
    start = end - timedelta(days=max_days)
    since_ms = int(start.timestamp() * 1000)
    raw = _fetch_ccxt_ohlcv(exchange_id, symbol, timeframe, since_ms)
    bars: list[PriceBar] = []
    for tms, o, h, l, c, v in raw:
        bars.append(PriceBar(
            timestamp=_ensure_utc(datetime.fromtimestamp(tms/1000, tz=timezone.utc)),
            open=float(o), high=float(h), low=float(l), close=float(c), volume=float(v)
        ))
    return bars

def _news_from_jsonl(path: str | Path, start: datetime, end: datetime, max_items: int = 50) -> list[NewsItem]:
    items: list[NewsItem] = []
    p = Path(path)
    if not p.exists():
        return items
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            obj = json.loads(line)
            ts = dtparser.isoparse(obj.get("published_at") or obj.get("publishedAt") or obj.get("date"))
            ts = _ensure_utc(ts)
            if not (start <= ts <= end): continue
            items.append(NewsItem(
                source=obj.get("source") or obj.get("domain") or "offline",
                title=obj.get("title",""),
                url=obj.get("url",""),
                published_at=ts,
                tickers=obj.get("tickers") or obj.get("symbols") or [],
                summary=obj.get("summary") or obj.get("snippet") or ""
            ))
            if len(items) >= max_items: break
        except Exception:
            continue
    return items

def _stub_llm_prompt_news(candidate_urls_path: str | Path, start: datetime, end: datetime,
                          max_items: int, stream: str) -> list[NewsItem]:
    urls = []
    try:
        urls = [ln.strip() for ln in Path(candidate_urls_path).read_text(encoding="utf-8").splitlines()
                if ln.strip() and not ln.strip().startswith("#")]
    except Exception:
        pass
    n = min(max_items, len(urls)) if urls else max_items
    if n <= 0: return []
    # Evenly space within window, most recent first
    span = max(1.0, (end - start).total_seconds())
    items: list[NewsItem] = []
    for i in range(n):
        ts = end - timedelta(seconds=(span * i / max(1, n)))
        items.append(NewsItem(
            source="LLM-Prompt-Stub",
            title=f"[{stream.upper()}] Synthesized item #{i+1}",
            url=urls[i] if i < len(urls) else f"https://example.com/{stream}/{i+1}",
            published_at=_ensure_utc(ts),
            tickers=["BTC-USD"],
            summary=f"Placeholder summary for {stream}."
        ))
    return items

def run_pipeline(
    # General info on underlying
    symbol: str,
    timeframe: str,
    max_days: int,
    out_dir: str,
    offline_prices_csv: Optional[str] = None,
    offline_news_jsonl: Optional[str] = None,
    news_lookback_days: int = 14,
    exchange_id: str = "kraken",
    news_query: str = "bitcoin OR BTC",

    # LLM knobs
    use_llm_news: bool = True,
    max_news_per_stream: int = 10,
    news_candidate_urls_path: Optional[str] = None,
    require_published_signal: bool = True,
    search_provider: str = "serpapi",
    llm_model: str = "gpt-4o-mini",
    max_search_results_per_stream: int = 50,
) -> Dict[str, Any]:
    """Main entry for the pipeline; returns a summary dict."""
    outd = Path(out_dir); outd.mkdir(parents=True, exist_ok=True)

    ########################## Prices #####################################
    print(f"⚙️ Forming Price Data 📈.............")
    """
        Note: We are currently reading off local offline data first.

        Moving forward should set a variable to determine which dataset,
        online or offline, we want to have priority first
    """
    try:
        if offline_prices_csv and Path(offline_prices_csv).exists():
            print(f"👷 Collecting Offline data from location: '{offline_prices_csv}' .............")
            prices = _read_csv_prices(offline_prices_csv)
            print(f"✅ Offline {symbol} {timeframe} Dataset on {max_days} days has been used")
        else:
            print(f"👷 Collecting Online data from cryptoexchange: {exchange_id}.............")
            prices = _prices_from_ccxt(exchange_id, symbol, timeframe, max_days)
            print(f"✅ Online {symbol} {timeframe} Dataset on {max_days} days has been obtained")
    except Exception as e:
        print(f"❌ We encountered an error here: {e}")

    # Build output filename
    price_csv = outd / f"prices_{symbol.replace('/','-')}_{timeframe}.csv"

    # Open the file in write mode 'w' and start formating
    with open(price_csv, "w", encoding="utf-8") as f:
        f.write("timestamp,open,high,low,close,volume\n")
        for b in prices:
            f.write(f"{b.timestamp.isoformat()},{b.open},{b.high},{b.low},{b.close},{b.volume}\n")

    ########################## News #####################################
    print("\n")
    # Time now
    end_dt = _utcnow()
    # Calculate which date to start
    start_dt = end_dt - timedelta(days=news_lookback_days)
    news_micro: list[NewsItem] = []; news_macro: list[NewsItem] = []
    print("⚙️ Forming News Data 📰.............")

    if use_llm_news:
        openai_key_present = bool(os.getenv("OPENAI_API_KEY"))
        if openai_key_present:
            # NEW (module import avoids Pylance/Pylint symbol resolution issues)
            try:
                res = llmps.search_micro_macro(
                    topic=news_query,                 # 👈 pass your topic/query
                    from_dt=start_dt,
                    to_dt=end_dt,
                    max_per_stream=max_news_per_stream,
                    provider=search_provider,
                    llm_model=llm_model,
                    symbol_hint="BTC-USD",
                )
                news_micro = res.get("micro", [])
                news_macro = res.get("macro", [])
            except Exception as e:
                print(f"❌ LLM prompt-search failed: {e}")
                news_micro = _stub_llm_prompt_news(news_candidate_urls_path or "", start_dt, end_dt, max_news_per_stream, "micro")
                news_macro = _stub_llm_prompt_news(news_candidate_urls_path or "", start_dt, end_dt, max_news_per_stream, "macro")
        else:
            print(f"❌ OPENAI_API_KEY NOT FOUND ")
            news_micro = _stub_llm_prompt_news(news_candidate_urls_path or "", start_dt, end_dt, max_news_per_stream, "micro")
            news_macro = _stub_llm_prompt_news(news_candidate_urls_path or "", start_dt, end_dt, max_news_per_stream, "macro")
    else:
        print(f"🟡 'use_llm_news' is set to {use_llm_news}. Attempting offline JSON file.......")
        items = _news_from_jsonl(offline_news_jsonl or "", start_dt, end_dt, max_items=max_news_per_stream*2)
        half = max(1, min(max_news_per_stream, len(items)//2 or 1))
        news_micro = items[:half]; news_macro = items[half:half+max_news_per_stream]

    print("\n-----------------------------------------------------------------")
    print("🌛 Here we are the micro news we obatined with Titles:")
    for i,j in enumerate(news_micro, 1):
        print(f"【{i}】 {j.title}\n")
    print("-----------------------------------------------------------------\n")

    print("\n-----------------------------------------------------------------")
    print("🌞 Here we are the macro news we obatined with Titles:")
    for i,j in enumerate(news_macro, 1):
        print(f"【{i}】 {j.title}\n")
    print("-----------------------------------------------------------------\n")


    micro_path = outd / "news_micro.json"; macro_path = outd / "news_macro.json"
    with open(micro_path,"w",encoding="utf-8") as f:
        json.dump([n.model_dump(mode="json") for n in news_micro], f, default=str, indent=2)
    with open(macro_path,"w",encoding="utf-8") as f:
        json.dump([n.model_dump(mode="json") for n in news_macro], f, default=str, indent=2)

    return {"prices_csv": str(price_csv),
            "news_micro_json": str(micro_path),
            "news_macro_json": str(macro_path),
            "num_prices": len(prices),
            "num_news_micro": len(news_micro),
            "num_news_macro": len(news_macro),
            "window_start": start_dt.isoformat(),
            "window_end": end_dt.isoformat()}
