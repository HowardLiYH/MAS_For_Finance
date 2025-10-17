
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List
from datetime import datetime, timedelta, timezone
import pandas as pd

from ..agents.analyst import AnalystAgent
from ..agents.researcher import ResearcherAgent
from ..agents.trader import TraderAgent
from ..agents.risk import RiskManagerAgent
from ..agents.evaluator import EvaluatorAgent
from ..dto.types import ResearchSummary, ExecutionSummary, RiskReview
# from ..data.providers import fetch_price, fetch_news
from ..tools.news_filter import filter_news_3_stage
from ..config import OrchestratorInput

# --- make bundled src/ importable and load .env (for OPENAI_API_KEY / SERPAPI_KEY)
from pathlib import Path
import sys, os, json
ROOT = Path(__file__).resolve().parents[2]      # .../MAS_Final_With_Agents
SRC  = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass

# --- import your Phase-1.1 pipeline directly
from core.data_pipeline import run_pipeline

def iterate_once(cfg: OrchestratorInput) -> Dict[str, Any]:
    """
    Orchestrator entrypoint.
    - CLI provides: cfg.symbol, cfg.interval
    - All other knobs come from cfg.appcfg (unified config)
    """
    print("\n================= MAS ITERATION START =================")

    # --- pull unified config
    appcfg = cfg.appcfg
    dcfg, ncfg = appcfg.data, appcfg.news   # data/news sections

    # ================= Time window =================
    # Allow optional start/end on cfg; else compute from dcfg.max_days
    end_str   = getattr(cfg, "end", None)
    start_str = getattr(cfg, "start", None)

    if end_str:
        end = datetime.fromisoformat(end_str).astimezone(timezone.utc)
    else:
        end = datetime.now(tz=timezone.utc).replace(minute=0, second=0, microsecond=0)

    if start_str:
        start = datetime.fromisoformat(start_str).astimezone(timezone.utc)
    else:
        start = end - timedelta(days=dcfg.max_days)

    print(f"[SETUP] symbol={cfg.symbol} window=[{start.isoformat()} → {end.isoformat()}], interval={cfg.interval}")

    # ================= Fetch BTC Price & News via src pipeline =================
    def _normalize_symbol(s: str) -> str:
        u = s.upper().replace("USDT", "USD")
        return "BTC/USD" if ("BTC" in u and "USD" in u) else s

    provider = ncfg.search_provider
    if provider == "serpapi" and not os.getenv("SERPAPI_KEY"):
        print("⚠️ SERPAPI_KEY is missing — news will be sparse/empty.")

    days = max(1, int((end - start).total_seconds() // 86400) + 1)

    out = run_pipeline(
        # price path
        symbol=_normalize_symbol(cfg.symbol),
        timeframe=cfg.interval,
        max_days=days,                              # or dcfg.max_days; both OK with explicit window
        out_dir=str(ROOT / dcfg.out_dir),
        offline_prices_csv=dcfg.offline_prices_csv,
        offline_news_jsonl=dcfg.offline_news_jsonl,
        news_lookback_days=getattr(ncfg, "news_lookback_days", 14),
        exchange_id=dcfg.exchange_id,

        # news path
        news_query=ncfg.news_query,
        use_llm_news=ncfg.use_llm_news,
        max_news_per_stream=ncfg.max_news_per_stream,
        search_provider=provider,
        llm_model=ncfg.llm_model,
        max_search_results_per_stream=ncfg.max_search_results_per_stream,
        require_published_signal=ncfg.require_published_signal,
    )

    # Load prices from CSV produced by pipeline
    price_df = pd.read_csv(out["prices_csv"], parse_dates=["timestamp"])
    price_df.set_index(pd.to_datetime(price_df["timestamp"], utc=True), inplace=True)
    print(f"[DATA] Price bars: {len(price_df)} (head={price_df.index[0]}, tail={price_df.index[-1]})")

    # Load news JSONs produced by pipeline, then run your 3-stage filter
    raw_news = []
    for key in ("news_micro_json", "news_macro_json"):
        fpath = out.get(key)
        if fpath and Path(fpath).exists():
            with open(fpath, "r") as f:
                raw_news += json.load(f)

    max_news = getattr(cfg, "max_news", ncfg.max_news_per_stream)
    news_items = filter_news_3_stage(raw_news, from_dt=start, to_dt=end)[: max_news]
    print(f"[DATA] News items: {len(news_items)} (after filter)")

    # ================= Agents =================
    analyst = AnalystAgent(id="A1")
    researcher = ResearcherAgent(id="R1")
    trader = TraderAgent(id="T1")
    risk = RiskManagerAgent(id="M1")
    evaluator = EvaluatorAgent(id="E1")

    # A: Analyst
    features, trend = analyst.run(price_df)
    # R: Researcher
    research = researcher.run(features, trend)
    # T: Trader (proposal)
    exec1 = trader.run(research, news_items, price_df)

    # M: Risk with single regeneration
    review1 = risk.run(exec1, price_df)
    regen_attempted = False
    if review1.verdict == "soft_fail" and not regen_attempted:
        print("[ORCH] Soft-fail envelope received → adjusting order & regenerating once")
        es = exec1.__dict__.copy()
        if "max_size" in review1.envelope:
            es["position_size"] = min(es["position_size"], review1.envelope["max_size"])
        if "max_leverage" in review1.envelope:
            es["leverage"] = min(es["leverage"], review1.envelope["max_leverage"])
        exec2 = exec1
        exec2.position_size = es["position_size"]; exec2.leverage = es["leverage"]
        regen_attempted = True
        review2 = risk.run(exec2, price_df, regen_attempted=True)
        final_exec, final_review = exec2, review2
    else:
        final_exec, final_review = exec1, review1

    # E: Evaluator (placeholder)
    scores = evaluator.score({"exec": final_exec.__dict__, "risk": final_review.__dict__}, benchmarks=None)

    print("================= MAS ITERATION END =================\n")
    return {
        "features_shape": tuple(features.shape),
        "trend_shape": tuple(trend.shape),
        "research": research.__dict__,
        "execution": final_exec.__dict__,
        "risk_review": final_review.__dict__,
        "scores": scores.__dict__,
    }
