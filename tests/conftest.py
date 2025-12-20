"""Pytest fixtures for PopAgent testing.

Provides mock data generators and test utilities for validating
the multi-agent trading workflow without external dependencies.
"""
from __future__ import annotations

import json
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

# Add project root to path
import sys
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Mock Data Generators
# =============================================================================

@pytest.fixture
def mock_price_data() -> pd.DataFrame:
    """Generate synthetic 4h OHLCV price data for testing."""
    np.random.seed(42)

    n_bars = 100  # 100 bars of 4h data (~16 days)
    start_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    # Generate realistic price movement
    initial_price = 42000.0  # BTC-like price
    returns = np.random.normal(0.0002, 0.015, n_bars)  # Mean ~0, std ~1.5%
    close_prices = initial_price * np.cumprod(1 + returns)

    # Generate OHLCV
    data = []
    for i in range(n_bars):
        close = close_prices[i]
        volatility = abs(np.random.normal(0, 0.008))
        high = close * (1 + volatility)
        low = close * (1 - volatility)
        open_price = close * (1 + np.random.normal(0, 0.003))
        volume = np.random.uniform(1000, 5000) * 1e6

        data.append({
            "timestamp": start_time + timedelta(hours=4 * i),
            "open": open_price,
            "high": max(high, open_price, close),
            "low": min(low, open_price, close),
            "close": close,
            "volume": volume,
            "turnover": volume * close,
        })

    df = pd.DataFrame(data)
    df.set_index("timestamp", inplace=True)
    return df


@pytest.fixture
def mock_multi_asset_data() -> Dict[str, pd.DataFrame]:
    """Generate synthetic price data for 5 assets."""
    np.random.seed(42)

    symbols = ["BTC", "ETH", "SOL", "DOGE", "XRP"]
    base_prices = {"BTC": 42000, "ETH": 2500, "SOL": 100, "DOGE": 0.08, "XRP": 0.55}
    correlations = {"BTC": 1.0, "ETH": 0.85, "SOL": 0.75, "DOGE": 0.5, "XRP": 0.6}

    n_bars = 100
    start_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    # Generate BTC as base
    btc_returns = np.random.normal(0.0002, 0.015, n_bars)

    assets = {}
    for symbol in symbols:
        # Correlated returns
        idio_returns = np.random.normal(0, 0.015 * (1 - correlations[symbol]), n_bars)
        returns = correlations[symbol] * btc_returns + idio_returns

        close_prices = base_prices[symbol] * np.cumprod(1 + returns)

        data = []
        for i in range(n_bars):
            close = close_prices[i]
            volatility = abs(np.random.normal(0, 0.008))
            high = close * (1 + volatility)
            low = close * (1 - volatility)
            open_price = close * (1 + np.random.normal(0, 0.003))

            data.append({
                "timestamp": start_time + timedelta(hours=4 * i),
                "open": open_price,
                "high": max(high, open_price, close),
                "low": min(low, open_price, close),
                "close": close,
                "volume": np.random.uniform(100, 1000) * 1e6,
                "symbol": symbol,
            })

        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        assets[symbol] = df

    return assets


@pytest.fixture
def mock_news_items() -> List[Dict[str, Any]]:
    """Generate synthetic news items for testing."""
    base_time = datetime(2024, 1, 10, 12, 0, 0, tzinfo=timezone.utc)

    news = [
        {
            "title": "Bitcoin ETF Approval Sparks Market Rally",
            "summary": "SEC approves spot Bitcoin ETF, leading to significant price increase.",
            "published_at": (base_time - timedelta(hours=2)).isoformat(),
            "source": "CryptoNews",
            "sentiment": 0.8,
            "relevance": 0.95,
            "category": "regulatory",
        },
        {
            "title": "Ethereum Upgrade Scheduled for Next Month",
            "summary": "Major protocol upgrade promises improved scalability and lower fees.",
            "published_at": (base_time - timedelta(hours=6)).isoformat(),
            "source": "BlockchainDaily",
            "sentiment": 0.6,
            "relevance": 0.85,
            "category": "technology",
        },
        {
            "title": "Federal Reserve Maintains Interest Rates",
            "summary": "Fed holds rates steady, crypto markets respond positively.",
            "published_at": (base_time - timedelta(hours=12)).isoformat(),
            "source": "FinanceWire",
            "sentiment": 0.3,
            "relevance": 0.7,
            "category": "macro",
        },
        {
            "title": "Large Whale Moves 10,000 BTC to Exchange",
            "summary": "On-chain data shows significant Bitcoin transfer to major exchange.",
            "published_at": (base_time - timedelta(hours=24)).isoformat(),
            "source": "OnChainAnalytics",
            "sentiment": -0.2,
            "relevance": 0.8,
            "category": "on-chain",
        },
        {
            "title": "Solana Network Experiences Brief Outage",
            "summary": "Network recovered within 2 hours, minimal impact on ecosystem.",
            "published_at": (base_time - timedelta(hours=48)).isoformat(),
            "source": "CryptoAlerts",
            "sentiment": -0.5,
            "relevance": 0.75,
            "category": "technical",
        },
    ]

    return news


@pytest.fixture
def mock_market_context() -> Dict[str, Any]:
    """Generate mock market context for testing."""
    return {
        "trend": "bullish",
        "volatility": 0.35,
        "regime": "normal",
        "btc_dominance": 0.52,
        "fear_greed_index": 65,
        "funding_rate_avg": 0.0001,
        "oi_change_24h": 0.05,
    }


# =============================================================================
# Selector Workflow Fixtures
# =============================================================================

@pytest.fixture
def selector_workflow_config():
    """Configuration for SelectorWorkflow testing."""
    from trading_agents.population.selector_workflow import SelectorWorkflowConfig

    return SelectorWorkflowConfig(
        population_size=3,  # Smaller for faster tests
        max_methods_per_agent=2,
        transfer_frequency=5,
        transfer_tau=0.1,
        exploration_rate=0.2,
        learning_rate=0.1,
        max_pipeline_samples=10,
    )


@pytest.fixture
def selector_workflow(selector_workflow_config):
    """Create a SelectorWorkflow instance for testing."""
    from trading_agents.population.selector_workflow import SelectorWorkflow

    return SelectorWorkflow(selector_workflow_config)


# =============================================================================
# Agent Fixtures
# =============================================================================

@pytest.fixture
def mock_agents():
    """Create mock agent instances for testing."""
    from trading_agents.agents.analyst import AnalystAgent
    from trading_agents.agents.researcher import ResearcherAgent
    from trading_agents.agents.trader import TraderAgent
    from trading_agents.agents.risk import RiskAgent
    from trading_agents.agents.evaluator import EvaluatorAgent

    return {
        "analyst": AnalystAgent(id="TEST_A1"),
        "researcher": ResearcherAgent(id="TEST_R1"),
        "trader": TraderAgent(id="TEST_T1", use_llm=False),
        "risk": RiskAgent(id="TEST_M1"),
        "evaluator": EvaluatorAgent(id="TEST_E1"),
    }


# =============================================================================
# File Fixtures
# =============================================================================

@pytest.fixture
def fixtures_dir(tmp_path) -> Path:
    """Create a temporary fixtures directory."""
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    return fixtures


@pytest.fixture
def mock_prices_csv(fixtures_dir, mock_price_data) -> Path:
    """Save mock price data to CSV."""
    csv_path = fixtures_dir / "mock_prices.csv"
    mock_price_data.reset_index().to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def mock_news_json(fixtures_dir, mock_news_items) -> Path:
    """Save mock news to JSON."""
    json_path = fixtures_dir / "mock_news.json"
    with open(json_path, "w") as f:
        json.dump(mock_news_items, f, indent=2)
    return json_path


# =============================================================================
# Utility Fixtures
# =============================================================================

@dataclass
class MockExecutionResult:
    """Mock execution result for testing."""
    order_id: str = "TEST_ORDER_001"
    direction: str = "LONG"
    entry_price: float = 42000.0
    take_profit: float = 43000.0
    stop_loss: float = 41000.0
    position_size: float = 0.1
    leverage: float = 5.0
    order_type: str = "MARKET"
    confidence: float = 0.75
    reasoning: str = "Test execution"


@pytest.fixture
def mock_execution():
    """Create a mock execution result."""
    return MockExecutionResult()


@pytest.fixture
def iteration_count():
    """Number of iterations for workflow tests."""
    return 5


# =============================================================================
# Performance Tracking Fixtures
# =============================================================================

@pytest.fixture
def performance_tracker():
    """Create a performance tracker for testing."""
    from trading_agents.services.metrics import PerformanceTracker
    return PerformanceTracker()
