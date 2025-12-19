"""Data models for the trading system."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime


@dataclass
class ResearchSummary:
    """
    Research output from Researcher Agent.

    Matches architecture JSON schema:
    {
        "Meta": {...},
        "Market_State": "String",
        "Forecast": "Dict[Horizon, Value]",
        "Signals": "List",
        "Risk": {"Constraints": "String", "Confidence": "Float"},
        "Recommendation": {"Scenarios": "String"},
        "Post_trade_evaluation_keys": "List"
    }
    """
    meta: Dict[str, Any] = field(default_factory=dict)
    market_state: str = "unknown"
    forecast: Dict[str, float] = field(default_factory=dict)
    signals: List[str] = field(default_factory=list)
    risk: Dict[str, Any] = field(default_factory=dict)
    recommendation: str = ""
    scenarios: List[str] = field(default_factory=list)
    explainability: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    confidence: float = 0.5
    post_trade_evaluation_keys: List[str] = field(
        default_factory=lambda: ["Sharpe", "PnL", "HitRate"]
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to architecture-compliant JSON structure."""
        return {
            "Meta": self.meta,
            "Market_State": self.market_state,
            "Forecast": self.forecast,
            "Signals": self.signals,
            "Risk": {
                "Constraints": "; ".join(self.constraints) if self.constraints else "",
                "Confidence": self.confidence,
                **{k: v for k, v in self.risk.items() if k not in ("Constraints", "Confidence")}
            },
            "Recommendation": {
                "Scenarios": "; ".join(self.scenarios) if self.scenarios else self.recommendation
            },
            "Explainability": self.explainability,
            "Post_trade_evaluation_keys": self.post_trade_evaluation_keys,
        }


@dataclass
class ExecutionSummary:
    """
    Execution order from Trader Agent.

    Fields match architecture spec:
    - Order_ID, Current_Price, Order_Type, Position_Size
    - Direction, Take_Profit_Price, Stop_Loss_Price
    - Leverage_Size, Liquidation_Price, Execution_Expired_Time
    """
    order_id: str
    timestamp: datetime
    style: str
    order_type: str              # MARKET | LIMIT
    direction: str               # LONG | SHORT
    position_size: float
    leverage: float
    entry_price: float
    take_profit: float
    stop_loss: float
    liquidation_price: Optional[float] = None
    closed_price: Optional[float] = None
    execution_expired_time: Optional[datetime] = None


@dataclass
class RiskReview:
    """
    Risk review from Risk Manager Agent.

    Verdicts:
    - pass: Order is safe to execute
    - soft_fail: Order violates soft rules, can be adjusted
    - hard_fail: Order violates hard rules, must abort
    """
    verdict: str  # pass | soft_fail | hard_fail
    reasons: List[str] = field(default_factory=list)
    envelope: Dict[str, float] = field(default_factory=dict)
    approved: bool = False


@dataclass
class AgentScores:
    """Performance scores for an agent."""
    agent_type: str
    score: float
    metrics: Dict[str, float] = field(default_factory=dict)
    period: str = "latest"
