"""Thought process logging for Trader agent."""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional


def log_thought_process(
    trader_id: str,
    order_id: str,
    thought_process: str,
    inputs: Dict[str, Any],
    outputs: Dict[str, Any],
    log_dir: Optional[Path] = None,
) -> Path:
    """
    Log trader thought process for debugging and reinforcement learning.

    Args:
        trader_id: ID of the trader agent
        order_id: Order ID for this decision
        thought_process: The LLM's reasoning text
        inputs: Input data (research summary, news, price data)
        outputs: Output data (execution summary)
        log_dir: Directory to save logs

    Returns:
        Path to the saved log file
    """
    if log_dir is None:
        log_dir = Path("./logs/thoughts")
    log_dir.mkdir(parents=True, exist_ok=True)

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trader_id": trader_id,
        "order_id": order_id,
        "thought_process": thought_process,
        "inputs": {
            "research_summary": inputs.get("research_summary", {}),
            "news_count": inputs.get("news_count", 0),
            "current_price": inputs.get("current_price"),
        },
        "outputs": outputs,
    }

    # Save individual log file
    log_file = log_dir / f"{order_id}_{trader_id}.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(log_entry, f, indent=2, default=str)

    # Append to daily log file
    daily_log = log_dir / f"daily_{datetime.now(timezone.utc).strftime('%Y%m%d')}.jsonl"
    with open(daily_log, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, default=str) + "\n")

    return log_file
