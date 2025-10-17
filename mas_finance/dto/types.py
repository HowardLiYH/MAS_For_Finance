
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

@dataclass
class ResearchSummary:
    market_state: str = "unknown"
    forecast: Dict[str, float] = field(default_factory=dict)  # horizon -> % change
    signals: List[str] = field(default_factory=list)
    risk: Dict[str, float] = field(default_factory=dict)
    recommendation: str = ""
    scenarios: List[str] = field(default_factory=list)
    explainability: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    confidence: float = 0.5
    post_trade_evaluation_keys: List[str] = field(default_factory=lambda: ["Sharpe","PnL","HitRate"])

@dataclass
class ExecutionSummary:
    order_id: str
    ts_create: datetime
    style: str
    order_type: str  # MARKET | LIMIT
    direction: str   # LONG | SHORT
    position_size: float
    leverage: float
    entry_price: float
    take_profit: float
    stop_loss: float
    liquidation_price: Optional[float] = None
    closed_price: Optional[float] = None

@dataclass
class RiskReview:
    verdict: str  # pass | soft_fail | hard_fail
    reasons: List[str] = field(default_factory=list)
    envelope: Dict[str, float] = field(default_factory=dict)  # e.g., {max_size, max_leverage, min_tp_sl_distance}
    approved: bool = False

@dataclass
class AgentScores:
    agent_type: str
    score: float
    metrics: Dict[str, float] = field(default_factory=dict)
    period: str = "latest"
