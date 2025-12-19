"""Workflow engine for orchestrating the multi-agent trading system."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import sys
import os

import pandas as pd
from dateutil import parser as dtparser

# Setup paths and environment
ROOT = Path(__file__).resolve().parent.parent
DATA_PIPELINE = ROOT / "data_pipeline"
if DATA_PIPELINE.exists() and str(DATA_PIPELINE) not in sys.path:
    sys.path.insert(0, str(DATA_PIPELINE))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass

# Local imports
from .config import AppConfig, OrchestratorInput, load_config, build_agents
from .config.loader import build_agents_from_yaml
from .models import ResearchSummary, ExecutionSummary, RiskReview
from .inventory import load_all_methods
from .services.metrics import PerformanceTracker
from .services.events import EventBus, TradingEvent, EventTypes
from .optimization import KnowledgeTransfer, InventoryPruner
from .utils.news_filter import filter_news_3_stage

# Agent imports
from .agents.analyst import AnalystAgent
from .agents.researcher import ResearcherAgent
from .agents.trader import TraderAgent
from .agents.risk import RiskAgent
from .agents.evaluator import EvaluatorAgent
from .agents.admin import AdminAgent, AdminConfig

# Paper trading imports (optional)
try:
    from .services.bybit_client import BybitTestnetClient
    from .services.order_manager import OrderManager
    from .services.positions import PositionTracker
    PAPER_TRADING_AVAILABLE = True
except ImportError:
    PAPER_TRADING_AVAILABLE = False

# Data pipeline import
try:
    from pipeline.data_pipeline import run_pipeline, run_multi_asset_pipeline, run_pipeline_auto
except ImportError:
    run_pipeline = None
    run_multi_asset_pipeline = None
    run_pipeline_auto = None

DEFAULT_CONFIG_PATH = ROOT / "configs" / "multi_asset.yaml"


def _normalize_symbol(symbol: str) -> str:
    """Normalize symbol format."""
    upper = symbol.upper().replace("USDT", "USD")
    return "BTC/USD" if ("BTC" in upper and "USD" in upper) else symbol


def _get_inventory_methods(agent) -> Dict[str, List[str]]:
    """Extract inventory method names from an agent."""
    if not hasattr(agent, 'inventory'):
        return {}

    methods = {}
    for pool, method_entries in agent.inventory.items():
        method_names = []
        for entry in method_entries:
            if hasattr(entry, 'name'):
                method_names.append(entry.name)
            elif isinstance(entry, tuple) and len(entry) > 0:
                inst = entry[0]
                if hasattr(inst, 'name'):
                    method_names.append(inst.name)
        if method_names:
            methods[pool] = method_names
    return methods


class WorkflowEngine:
    """
    Orchestrates the multi-agent trading workflow.

    Manages:
    - Agent lifecycle and configuration
    - Data pipeline execution
    - Performance tracking
    - Knowledge transfer and inventory pruning
    """

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        config_path: Optional[Path] = None,
        enable_paper_trading: bool = False,
        bybit_api_key: Optional[str] = None,
        bybit_api_secret: Optional[str] = None,
    ):
        """
        Initialize the workflow engine.

        Args:
            config: Application configuration
            config_path: Path to YAML config file (used if config not provided)
            enable_paper_trading: Enable Bybit testnet paper trading
            bybit_api_key: Bybit testnet API key (or use env var)
            bybit_api_secret: Bybit testnet API secret (or use env var)
        """
        # Load configuration
        if config:
            self.config = config
        else:
            path = config_path or DEFAULT_CONFIG_PATH
            self.config = load_config(path)

        # Load inventory methods
        load_all_methods()

        # Initialize agents
        self.agents = self._load_agents()

        # Initialize tracking and optimization
        self.tracker = PerformanceTracker()
        self.knowledge_transfer = KnowledgeTransfer(
            tracker=self.tracker,
            transfer_frequency=self.config.learning.knowledge_transfer_frequency,
            trader_transfer_frequency=self.config.learning.trader_transfer_frequency,
        )
        self.pruner = InventoryPruner(
            tracker=self.tracker,
            pruning_frequency=self.config.learning.pruning_frequency,
        )

        # Ensure evaluator has tracker
        if self.agents.get("evaluator"):
            self.agents["evaluator"].tracker = self.tracker

        # Initialize event bus for system-wide communication
        self.event_bus = EventBus()

        # Initialize Admin Agent
        admin_config = AdminConfig(
            max_drawdown_pct=getattr(self.config, 'admin_max_drawdown_pct', 10.0),
            daily_loss_limit_pct=getattr(self.config, 'admin_daily_loss_pct', 5.0),
            slack_webhook=os.getenv("SLACK_WEBHOOK"),
        )
        self.admin = AdminAgent(
            id="ADMIN",
            tracker=self.tracker,
            event_bus=self.event_bus,
            config=admin_config,
        )
        self.agents["admin"] = self.admin

        # Initialize paper trading if enabled
        self.paper_trading_enabled = False
        self.bybit_client: Optional[BybitTestnetClient] = None
        self.order_manager: Optional[OrderManager] = None
        self.position_tracker: Optional[PositionTracker] = None

        if enable_paper_trading and PAPER_TRADING_AVAILABLE:
            api_key = bybit_api_key or os.getenv("BYBIT_TESTNET_KEY")
            api_secret = bybit_api_secret or os.getenv("BYBIT_TESTNET_SECRET")

            if api_key and api_secret:
                self.bybit_client = BybitTestnetClient(api_key, api_secret)
                self.order_manager = OrderManager(self.bybit_client, self.event_bus)
                self.position_tracker = PositionTracker(self.bybit_client, self.event_bus)
                self.paper_trading_enabled = True
                print("[WORKFLOW] Paper trading enabled (Bybit Testnet)")
            else:
                print("[WORKFLOW] Paper trading disabled: missing API credentials")

        # State
        self.iteration = 0

    def _load_agents(self) -> Dict[str, Any]:
        """Load agents from configuration."""
        config_path = DEFAULT_CONFIG_PATH

        try:
            if config_path.exists():
                agents = build_agents_from_yaml(config_path)
                if agents:
                    print(f"[WORKFLOW] Loaded agents from {config_path}")
                    return agents
        except Exception as e:
            print(f"[WORKFLOW] Failed to load agents from config: {e}")

        # Fallback to defaults
        print("[WORKFLOW] Using default agents")
        return {
            "analyst": AnalystAgent(id="A1"),
            "researcher": ResearcherAgent(id="R1"),
            "trader": TraderAgent(id="T1"),
            "risk": RiskAgent(id="M1"),
            "evaluator": EvaluatorAgent(id="E1"),
        }

    def run_iteration(self, input_cfg: OrchestratorInput) -> Dict[str, Any]:
        """
        Run one iteration of the trading workflow.

        Pipeline:
        1. Fetch data (prices + news)
        2. Analyst: Extract features and trends
        3. Researcher: Generate forecasts and signals
        4. Trader: Create order proposal
        5. Risk: Validate order
        6. Evaluator: Score performance
        7. (Optionally) Apply learning/pruning

        Args:
            input_cfg: Orchestrator input configuration

        Returns:
            Dict with iteration results
        """
        self.iteration += 1
        print(f"\n{'='*60}")
        print(f"[WORKFLOW] Iteration {self.iteration} START")
        print(f"{'='*60}")

        # Unpack configuration
        config = input_cfg.config
        data_cfg = config.data
        news_cfg = config.news

        # Calculate time window
        if input_cfg.end:
            end = datetime.fromisoformat(input_cfg.end).astimezone(timezone.utc)
        else:
            end = datetime.now(tz=timezone.utc).replace(minute=0, second=0, microsecond=0)

        if input_cfg.start:
            start = datetime.fromisoformat(input_cfg.start).astimezone(timezone.utc)
        else:
            start = end - timedelta(days=data_cfg.max_days)

        print(f"[SETUP] Symbol: {input_cfg.symbol}, Window: {start.date()} → {end.date()}")

        # Fetch data
        price_df, news_items = self._fetch_data(input_cfg, start, end)

        # Apply learning/pruning if due
        self._apply_optimization()

        # Get agents
        analyst = self.agents["analyst"]
        researcher = self.agents["researcher"]
        trader = self.agents["trader"]
        risk = self.agents["risk"]
        evaluator = self.agents["evaluator"]

        # ===== PHASE 1: Analysis & Research =====

        # Analyst
        features, trend = analyst.run(price_df)

        # Researcher
        research = researcher.run(features, trend)

        # Determine news lookback
        trader_lookback = trader.determine_news_lookback_days(
            research, price_df, max_lookback_days=news_cfg.max_lookback_days
        )
        lookback_start = end - timedelta(days=trader_lookback)
        print(f"[TRADER] News lookback: {trader_lookback} days")

        # Filter news
        filtered_news = self._filter_news(news_items, lookback_start, end, news_cfg.max_news_per_stream)

        # ===== PHASE 2: Execution & Risk =====

        # Trader
        execution = trader.run(research, filtered_news, price_df)

        # Emit order created event
        self.event_bus.publish(TradingEvent.now(
            event_type=EventTypes.ORDER_CREATED,
            payload={"order_id": execution.order_id, "direction": execution.direction},
            source="workflow",
        ))

        # Risk check
        risk_review = risk.run(execution, price_df)

        # Emit risk event
        if risk_review.verdict == "hard_fail":
            self.event_bus.publish(TradingEvent.now(
                event_type=EventTypes.RISK_HARD_FAIL,
                payload={"reason": risk_review.notes, "order_id": execution.order_id},
                severity="critical",
                source="risk_agent",
            ))
        elif risk_review.verdict == "soft_fail":
            self.event_bus.publish(TradingEvent.now(
                event_type=EventTypes.RISK_SOFT_FAIL,
                payload={"reason": risk_review.notes, "order_id": execution.order_id},
                severity="warning",
                source="risk_agent",
            ))

        # Handle risk review
        final_execution, final_review = self._handle_risk_review(
            execution, risk_review, risk, price_df
        )

        # ===== Evaluation =====

        # Record trade if approved
        if final_review.approved:
            self._record_trade(evaluator, final_execution, research)

            # Emit order submitted/approved event
            self.event_bus.publish(TradingEvent.now(
                event_type=EventTypes.ORDER_SUBMITTED,
                payload={"order_id": final_execution.order_id, "approved": True},
                source="workflow",
            ))

        # Score
        scores = evaluator.score(
            {"exec": final_execution.__dict__, "risk": final_review.__dict__},
            agent_id=trader.id,
            agent_type="trader",
        )

        # Emit iteration end event
        self.event_bus.publish(TradingEvent.now(
            event_type=EventTypes.ITERATION_END,
            payload={
                "iteration": self.iteration,
                "approved": final_review.approved,
                "direction": final_execution.direction,
            },
            source="workflow",
        ))

        print(f"\n{'='*60}")
        print(f"[WORKFLOW] Iteration {self.iteration} END")
        print(f"{'='*60}\n")

        return {
            "iteration": self.iteration,
            "features_shape": tuple(features.shape),
            "trend_shape": tuple(trend.shape),
            "research": research.__dict__,
            "execution": final_execution.__dict__,
            "risk_review": final_review.__dict__,
            "scores": scores.__dict__,
        }

    def _fetch_data(
        self,
        input_cfg: OrchestratorInput,
        start: datetime,
        end: datetime,
    ) -> tuple[pd.DataFrame, List[dict]]:
        """Fetch price and news data."""
        config = input_cfg.config
        data_cfg = config.data
        news_cfg = config.news

        if run_pipeline is None:
            raise ImportError("Data pipeline not available")

        # Check API keys
        if news_cfg.search_provider == "serpapi" and not os.getenv("SERPAPI_KEY"):
            print("⚠️ SERPAPI_KEY is missing — news may be empty")

        days = max(1, int((end - start).total_seconds() // 86400) + 1)
        news_lookback = min(news_cfg.max_lookback_days, news_cfg.default_lookback_days)

        out = run_pipeline(
            symbol=_normalize_symbol(input_cfg.symbol),
            timeframe=input_cfg.interval,
            max_days=days,
            out_dir=str(ROOT / data_cfg.out_dir),
            offline_prices_csv=data_cfg.offline_prices_csv,
            offline_news_jsonl=data_cfg.offline_news_jsonl,
            news_lookback_days=news_lookback,
            exchange_id=data_cfg.exchange_id,
            news_query=news_cfg.news_query,
            use_llm_news=news_cfg.use_llm_news,
            max_news_per_stream=news_cfg.max_news_per_stream,
            search_provider=news_cfg.search_provider,
            llm_model=news_cfg.llm_model,
            max_search_results_per_stream=news_cfg.max_search_results_per_stream,
            require_published_signal=news_cfg.require_published_signal,
        )

        # Load prices
        price_df = pd.read_csv(out["prices_csv"], parse_dates=["timestamp"])
        price_df.set_index(pd.to_datetime(price_df["timestamp"], utc=True), inplace=True)
        print(f"[DATA] Price bars: {len(price_df)}")

        # Load news
        raw_news: List[dict] = []
        for key in ("news_micro_json", "news_macro_json"):
            fpath = out.get(key)
            if fpath and Path(fpath).exists():
                with open(fpath, "r") as f:
                    raw_news += json.load(f)

        news_items = filter_news_3_stage(raw_news, from_dt=start, to_dt=end)
        print(f"[DATA] News items: {len(news_items)}")

        return price_df, news_items

    def _filter_news(
        self,
        news_items: List[Any],
        start: datetime,
        end: datetime,
        max_items: int,
    ) -> List[Dict[str, Any]]:
        """Filter news items by time range."""
        filtered = []
        for item in news_items:
            if hasattr(item, 'published_at'):
                pub_date = item.published_at
            elif isinstance(item, dict):
                pub_str = item.get('published_at', end.isoformat())
                pub_date = dtparser.isoparse(pub_str)
            else:
                continue

            if pub_date >= start:
                filtered.append(item)

        return filtered[:max_items]

    def _handle_risk_review(
        self,
        execution: ExecutionSummary,
        review: RiskReview,
        risk_agent: RiskAgent,
        price_df: pd.DataFrame,
    ) -> tuple[ExecutionSummary, RiskReview]:
        """Handle risk review, including regeneration for soft_fail."""

        # Hard fail: abort immediately
        if review.verdict == "hard_fail":
            print("[WORKFLOW] HARD FAIL: Order aborted")
            return execution, review

        # Soft fail: attempt one regeneration
        if review.verdict == "soft_fail":
            print("[WORKFLOW] Soft fail: Attempting adjustment")

            # Apply envelope adjustments
            if "max_size" in review.envelope:
                execution.position_size = min(
                    execution.position_size,
                    review.envelope["max_size"]
                )
            if "max_leverage" in review.envelope:
                execution.leverage = min(
                    execution.leverage,
                    review.envelope["max_leverage"]
                )

            # Re-check
            review2 = risk_agent.run(execution, price_df, regen_attempted=True)
            return execution, review2

        return execution, review

    def _record_trade(
        self,
        evaluator: EvaluatorAgent,
        execution: ExecutionSummary,
        research: ResearchSummary,
    ):
        """Record trade for performance tracking."""
        inventory_methods = {
            "analyst": _get_inventory_methods(self.agents["analyst"]),
            "researcher": _get_inventory_methods(self.agents["researcher"]),
            "trader": _get_inventory_methods(self.agents["trader"]),
            "risk": _get_inventory_methods(self.agents["risk"]),
        }

        evaluator.record_trade(
            order_id=execution.order_id,
            agent_id=self.agents["trader"].id,
            agent_type="trader",
            execution_summary=execution,
            research_summary=research.__dict__,
            inventory_methods_used=inventory_methods,
        )

    def _apply_optimization(self):
        """Apply knowledge transfer and inventory pruning if due."""
        agents_dict = {
            self.agents["analyst"].id: self.agents["analyst"],
            self.agents["researcher"].id: self.agents["researcher"],
            self.agents["trader"].id: self.agents["trader"],
            self.agents["risk"].id: self.agents["risk"],
        }

        try:
            agents_dict = self.knowledge_transfer.transfer_all_types(
                agents_dict, self.iteration
            )
        except Exception as e:
            print(f"[WORKFLOW] Knowledge transfer skipped: {e}")

        try:
            agents_dict = self.pruner.prune_all_types(agents_dict, self.iteration)
        except Exception as e:
            print(f"[WORKFLOW] Inventory pruning skipped: {e}")

    def run_multi_asset_iteration(self, input_cfg: OrchestratorInput) -> Dict[str, Any]:
        """
        Run one iteration for all assets in multi-asset mode.

        Processes each symbol sequentially with shared market context.

        Args:
            input_cfg: Orchestrator input configuration

        Returns:
            Dict with results for all assets
        """
        self.iteration += 1
        print(f"\n{'='*60}")
        print(f"[WORKFLOW] Multi-Asset Iteration {self.iteration} START")
        print(f"{'='*60}")

        config = input_cfg.config
        data_cfg = config.data
        news_cfg = config.news

        # Calculate time window
        if input_cfg.end:
            end = datetime.fromisoformat(input_cfg.end).astimezone(timezone.utc)
        else:
            end = datetime.now(tz=timezone.utc).replace(minute=0, second=0, microsecond=0)

        if input_cfg.start:
            start = datetime.fromisoformat(input_cfg.start).astimezone(timezone.utc)
        else:
            start = end - timedelta(days=data_cfg.max_days)

        symbols = input_cfg.active_symbols
        print(f"[SETUP] Symbols: {symbols}, Window: {start.date()} → {end.date()}")

        # Fetch multi-asset data
        multi_data = self._fetch_multi_asset_data(input_cfg, start, end)

        assets = multi_data.get("assets", {})
        market_ctx = multi_data.get("market_context")
        news_items = self._load_news_items(multi_data)

        # Apply learning/pruning if due
        self._apply_optimization()

        # Process each symbol
        results = {}
        for symbol in symbols:
            if symbol not in assets:
                print(f"[WORKFLOW] Skipping {symbol}: no data")
                continue

            print(f"\n--- Processing {symbol} ---")

            asset_result = self._process_single_asset(
                symbol=symbol,
                price_df=assets[symbol],
                news_items=news_items,
                market_ctx=market_ctx,
                end=end,
                news_cfg=news_cfg,
            )
            results[symbol] = asset_result

        print(f"\n{'='*60}")
        print(f"[WORKFLOW] Multi-Asset Iteration {self.iteration} END")
        print(f"{'='*60}\n")

        return {
            "iteration": self.iteration,
            "mode": "multi_asset",
            "symbols": list(results.keys()),
            "results": results,
            "market_context_available": market_ctx is not None,
        }

    def _fetch_multi_asset_data(
        self,
        input_cfg: OrchestratorInput,
        start: datetime,
        end: datetime,
    ) -> Dict[str, Any]:
        """Fetch multi-asset price and news data."""
        config = input_cfg.config
        data_cfg = config.data
        news_cfg = config.news

        if run_multi_asset_pipeline is None:
            raise ImportError("Multi-asset data pipeline not available")

        return run_multi_asset_pipeline(
            symbols=data_cfg.symbols,
            bybit_csv_dir=data_cfg.bybit_csv_dir,
            out_dir=str(ROOT / data_cfg.out_dir),
            start_date=start.isoformat() if start else None,
            end_date=end.isoformat() if end else None,
            news_lookback_days=news_cfg.default_lookback_days,
            news_query=news_cfg.news_query,
            use_llm_news=news_cfg.use_llm_news,
            max_news_per_stream=news_cfg.max_news_per_stream,
            search_provider=news_cfg.search_provider,
            llm_model=news_cfg.llm_model,
            add_cross_features=data_cfg.add_cross_features,
        )

    def _load_news_items(self, multi_data: Dict[str, Any]) -> List[dict]:
        """Load news items from multi-asset pipeline output."""
        raw_news: List[dict] = []
        for key in ("news_micro_json", "news_macro_json"):
            fpath = multi_data.get(key)
            if fpath and Path(fpath).exists():
                with open(fpath, "r") as f:
                    raw_news += json.load(f)
        return raw_news

    def _process_single_asset(
        self,
        symbol: str,
        price_df: pd.DataFrame,
        news_items: List[dict],
        market_ctx: Any,
        end: datetime,
        news_cfg: Any,
    ) -> Dict[str, Any]:
        """Process a single asset through the agent pipeline."""
        # Get agents
        analyst = self.agents["analyst"]
        researcher = self.agents["researcher"]
        trader = self.agents["trader"]
        risk = self.agents["risk"]
        evaluator = self.agents["evaluator"]

        # ===== PHASE 1: Analysis & Research =====
        features, trend = analyst.run(price_df)
        research = researcher.run(features, trend)

        # Determine news lookback
        trader_lookback = trader.determine_news_lookback_days(
            research, price_df, max_lookback_days=news_cfg.max_lookback_days
        )
        lookback_start = end - timedelta(days=trader_lookback)

        # Filter news
        filtered_news = self._filter_news(
            news_items, lookback_start, end, news_cfg.max_news_per_stream
        )

        # ===== PHASE 2: Execution & Risk =====
        execution = trader.run(research, filtered_news, price_df)
        risk_review = risk.run(execution, price_df)

        # Handle risk review
        final_execution, final_review = self._handle_risk_review(
            execution, risk_review, risk, price_df
        )

        # ===== Evaluation =====
        if final_review.approved:
            self._record_trade(evaluator, final_execution, research)

        scores = evaluator.score(
            {"exec": final_execution.__dict__, "risk": final_review.__dict__},
            agent_id=trader.id,
            agent_type="trader",
        )

        return {
            "symbol": symbol,
            "features_shape": tuple(features.shape),
            "trend_shape": tuple(trend.shape),
            "research": research.__dict__,
            "execution": final_execution.__dict__,
            "risk_review": final_review.__dict__,
            "scores": scores.__dict__,
        }


# Convenience function for CLI
def run_single_iteration(
    symbol: str,
    interval: str,
    config_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run a single iteration of the workflow."""
    engine = WorkflowEngine(config_path=config_path)

    input_cfg = OrchestratorInput(
        symbol=symbol,
        interval=interval,
        config=engine.config,
    )

    return engine.run_iteration(input_cfg)


def run_multi_asset(
    config_path: Optional[Path] = None,
    symbols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run a multi-asset iteration of the workflow.

    Args:
        config_path: Path to config file (should have multi_asset: true)
        symbols: Optional list of symbols to override config

    Returns:
        Dict with results for all assets
    """
    config_path = config_path or (ROOT / "configs" / "multi_asset.yaml")
    engine = WorkflowEngine(config_path=config_path)

    input_cfg = OrchestratorInput(
        symbol="MULTI",
        interval=engine.config.timeframe,
        config=engine.config,
        symbols=symbols,
    )

    return engine.run_multi_asset_iteration(input_cfg)


def run_paper_trading(
    config_path: Optional[Path] = None,
    symbols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run a multi-asset iteration with paper trading enabled.

    Requires BYBIT_TESTNET_KEY and BYBIT_TESTNET_SECRET env vars.

    Args:
        config_path: Path to config file
        symbols: Optional list of symbols to trade

    Returns:
        Dict with results for all assets
    """
    if not PAPER_TRADING_AVAILABLE:
        raise ImportError("Paper trading requires aiohttp. Install with: pip install aiohttp")

    config_path = config_path or (ROOT / "configs" / "multi_asset.yaml")
    engine = WorkflowEngine(
        config_path=config_path,
        enable_paper_trading=True,
    )

    if not engine.paper_trading_enabled:
        raise ValueError("Paper trading not enabled. Check BYBIT_TESTNET_KEY and BYBIT_TESTNET_SECRET env vars.")

    input_cfg = OrchestratorInput(
        symbol="MULTI",
        interval=engine.config.timeframe,
        config=engine.config,
        symbols=symbols,
    )

    return engine.run_multi_asset_iteration(input_cfg)


def get_admin_status(engine: WorkflowEngine) -> Dict[str, Any]:
    """Get status of the admin agent."""
    if engine.admin:
        return engine.admin.get_status()
    return {"error": "Admin agent not initialized"}


def send_performance_report(engine: WorkflowEngine, lookback_days: int = 30) -> Dict[str, bool]:
    """Send a performance report via the admin agent."""
    if engine.admin:
        return engine.admin.send_performance_report(lookback_days)
    return {"error": True}
