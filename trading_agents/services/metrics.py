"""Performance tracking and metrics."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path
import numpy as np


@dataclass
class TradeRecord:
    """Record of a single trade execution."""
    order_id: str
    agent_id: str
    agent_type: str
    timestamp: datetime
    execution_summary: Dict[str, Any]
    research_summary: Dict[str, Any]
    inventory_methods_used: Dict[str, List[str]]
    pnl: float = 0.0
    pnl_pct: float = 0.0
    closed: bool = False
    close_reason: Optional[str] = None


@dataclass
class MethodUsageRecord:
    """Record of inventory method usage."""
    pool: str
    method_name: str
    usage_count: int = 0
    last_used: Optional[datetime] = None
    success_count: int = 0
    total_pnl: float = 0.0


class PerformanceTracker:
    """Tracks performance metrics for agents and inventory methods."""

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path(".performance_data")
        self.trades: List[TradeRecord] = []
        self.agent_scores: Dict[str, Dict[str, float]] = defaultdict(lambda: {
            "total_trades": 0,
            "winning_trades": 0,
            "total_pnl": 0.0,
            "total_pnl_pct": 0.0,
        })
        self.method_usage: Dict[str, MethodUsageRecord] = {}

    def _method_key(self, pool: str, name: str) -> str:
        return f"{pool}:{name}"

    def record_trade(
        self,
        order_id: str,
        agent_id: str,
        agent_type: str,
        execution_summary: Dict[str, Any],
        research_summary: Dict[str, Any],
        inventory_methods_used: Dict[str, List[str]],
    ):
        """Record a new trade."""
        trade = TradeRecord(
            order_id=order_id,
            agent_id=agent_id,
            agent_type=agent_type,
            timestamp=datetime.now(tz=timezone.utc),
            execution_summary=execution_summary,
            research_summary=research_summary,
            inventory_methods_used=inventory_methods_used,
        )
        self.trades.append(trade)

        # Update method usage
        for pool, methods in inventory_methods_used.items():
            for method_name in methods:
                key = self._method_key(pool, method_name)
                if key not in self.method_usage:
                    self.method_usage[key] = MethodUsageRecord(
                        pool=pool, method_name=method_name
                    )
                record = self.method_usage[key]
                record.usage_count += 1
                record.last_used = trade.timestamp

    def update_trade_result(
        self,
        order_id: str,
        pnl: float,
        pnl_pct: float,
        close_reason: Optional[str] = None,
    ):
        """Update trade with final PnL results."""
        trade = next((t for t in self.trades if t.order_id == order_id), None)
        if not trade:
            return

        trade.pnl = pnl
        trade.pnl_pct = pnl_pct
        trade.closed = True
        trade.close_reason = close_reason

        # Update agent scores
        agent_key = f"{trade.agent_type}:{trade.agent_id}"
        scores = self.agent_scores[agent_key]
        scores["total_trades"] += 1
        scores["total_pnl"] += pnl
        scores["total_pnl_pct"] += pnl_pct
        if pnl > 0:
            scores["winning_trades"] += 1

        # Update method usage
        for pool, methods in trade.inventory_methods_used.items():
            for method_name in methods:
                key = self._method_key(pool, method_name)
                if key in self.method_usage:
                    record = self.method_usage[key]
                    if pnl > 0:
                        record.success_count += 1
                    record.total_pnl += pnl

    def calculate_agent_metrics(self, agent_id: str, agent_type: str) -> Dict[str, float]:
        """Calculate comprehensive metrics for an agent."""
        agent_key = f"{agent_type}:{agent_id}"
        scores = self.agent_scores[agent_key]

        if scores["total_trades"] == 0:
            return {"Sharpe": 0.0, "PnL": 0.0, "HitRate": 0.0, "MaxDD": 0.0, "CalibECE": 0.0}

        hit_rate = scores["winning_trades"] / scores["total_trades"]

        # Sharpe ratio
        agent_trades = [t for t in self.trades if t.agent_id == agent_id and t.closed]
        if len(agent_trades) > 1:
            pnl_pcts = [t.pnl_pct for t in agent_trades]
            mean_return = np.mean(pnl_pcts)
            std_dev = np.std(pnl_pcts)
            sharpe = (mean_return / std_dev * (2190 ** 0.5)) if std_dev > 0 else 0.0
        else:
            sharpe = 0.0

        # Max drawdown
        if agent_trades:
            cumulative = 0.0
            peak = 0.0
            max_dd = 0.0
            for trade in agent_trades:
                cumulative += trade.pnl_pct
                if cumulative > peak:
                    peak = cumulative
                drawdown = peak - cumulative
                if drawdown > max_dd:
                    max_dd = drawdown
        else:
            max_dd = 0.0

        # Calibration ECE
        calib_ece = 0.0
        if agent_trades:
            buckets = defaultdict(list)
            for trade in agent_trades:
                conf = trade.research_summary.get("confidence", 0.5)
                bucket = int(conf * 10) / 10.0
                buckets[bucket].append(1.0 if trade.pnl > 0 else 0.0)

            ece_sum = 0.0
            total_weight = 0.0
            for bucket, outcomes in buckets.items():
                if outcomes:
                    weight = len(outcomes)
                    ece_sum += abs(bucket - np.mean(outcomes)) * weight
                    total_weight += weight
            calib_ece = ece_sum / total_weight if total_weight > 0 else 0.0

        return {
            "Sharpe": sharpe,
            "PnL": scores["total_pnl"],
            "HitRate": hit_rate,
            "MaxDD": max_dd,
            "CalibECE": calib_ece,
        }

    def get_method_usage_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get usage statistics for all methods."""
        stats = {}
        for key, record in self.method_usage.items():
            success_rate = record.success_count / record.usage_count if record.usage_count > 0 else 0.0
            avg_pnl = record.total_pnl / record.usage_count if record.usage_count > 0 else 0.0
            stats[key] = {
                "pool": record.pool,
                "method_name": record.method_name,
                "usage_count": record.usage_count,
                "success_count": record.success_count,
                "success_rate": success_rate,
                "avg_pnl": avg_pnl,
                "last_used": record.last_used.isoformat() if record.last_used else None,
            }
        return stats

    def get_top_agents(self, agent_type: str, top_n: int = 5) -> List[tuple[str, Dict[str, float]]]:
        """Get top N agents by score."""
        agent_scores = []
        for key, scores in self.agent_scores.items():
            if key.startswith(f"{agent_type}:"):
                agent_id = key.split(":", 1)[1]
                metrics = self.calculate_agent_metrics(agent_id, agent_type)
                score = (
                    metrics["Sharpe"] * 0.3 +
                    metrics["HitRate"] * 0.3 +
                    (metrics["PnL"] / 1000.0) * 0.2 +
                    (1.0 - metrics["MaxDD"]) * 0.1 +
                    (1.0 - metrics["CalibECE"]) * 0.1
                )
                agent_scores.append((agent_id, metrics, score))

        agent_scores.sort(key=lambda x: x[2], reverse=True)
        return [(agent_id, metrics) for agent_id, metrics, _ in agent_scores[:top_n]]
