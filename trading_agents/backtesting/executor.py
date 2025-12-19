"""Order execution simulation for backtesting."""
from __future__ import annotations
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime, timezone
import pandas as pd

from ..models.types import ExecutionSummary


class OrderState(Enum):
    """Order state machine."""
    OPEN = "open"
    FILLED = "filled"
    CLOSED = "closed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


@dataclass
class OrderExecution:
    """Represents an order execution in backtesting."""
    order_id: str
    execution_summary: ExecutionSummary
    state: OrderState = OrderState.OPEN
    filled_price: Optional[float] = None
    filled_time: Optional[datetime] = None
    closed_price: Optional[float] = None
    closed_time: Optional[datetime] = None
    close_reason: Optional[str] = None  # "take_profit", "stop_loss", "eet", "liquidation"
    pnl: float = 0.0
    pnl_pct: float = 0.0


class OrderExecutor:
    """Handles order execution simulation in backtesting."""

    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.open_orders: Dict[str, OrderExecution] = {}
        self.closed_orders: list[OrderExecution] = []

    def calculate_liquidation_price(
        self,
        entry_price: float,
        direction: str,
        leverage: float
    ) -> float:
        """Calculate liquidation price based on leverage."""
        # Simplified: liquidation at ~90% of margin (allowing some buffer)
        if direction == "LONG":
            # Long position liquidates if price drops too much
            margin_ratio = 0.9 / leverage
            return entry_price * (1 - margin_ratio)
        else:  # SHORT
            # Short position liquidates if price rises too much
            margin_ratio = 0.9 / leverage
            return entry_price * (1 + margin_ratio)

    def submit_order(self, execution_summary: ExecutionSummary) -> OrderExecution:
        """Submit a new order for execution."""
        # Calculate liquidation price if not already set
        if execution_summary.liquidation_price is None:
            liquidation_price = self.calculate_liquidation_price(
                execution_summary.entry_price,
                execution_summary.direction,
                execution_summary.leverage
            )
            execution_summary.liquidation_price = liquidation_price

        order_exec = OrderExecution(
            order_id=execution_summary.order_id,
            execution_summary=execution_summary,
            state=OrderState.OPEN,
        )

        self.open_orders[execution_summary.order_id] = order_exec
        return order_exec

    def check_order_fill(
        self,
        order_exec: OrderExecution,
        current_bar: pd.Series,
        current_time: datetime
    ) -> bool:
        """Check if an order should be filled at current bar. Returns True if filled."""
        if order_exec.state != OrderState.OPEN:
            return False

        exec_sum = order_exec.execution_summary

        # Market orders fill immediately at open
        if exec_sum.order_type == "MARKET":
            order_exec.state = OrderState.FILLED
            order_exec.filled_price = current_bar["open"]
            order_exec.filled_time = current_time
            return True

        # Limit orders check if price touched the limit
        if exec_sum.order_type == "LIMIT":
            if exec_sum.direction == "LONG":
                # Long limit: fill if low touched entry_price or below
                if current_bar["low"] <= exec_sum.entry_price:
                    order_exec.state = OrderState.FILLED
                    order_exec.filled_price = exec_sum.entry_price
                    order_exec.filled_time = current_time
                    return True
            else:  # SHORT
                # Short limit: fill if high touched entry_price or above
                if current_bar["high"] >= exec_sum.entry_price:
                    order_exec.state = OrderState.FILLED
                    order_exec.filled_price = exec_sum.entry_price
                    order_exec.filled_time = current_time
                    return True

        return False

    def check_order_close(
        self,
        order_exec: OrderExecution,
        current_bar: pd.Series,
        current_time: datetime
    ) -> Optional[str]:
        """
        Check if an order should be closed (TP/SL/EET/Liquidation).
        Returns close reason if closed, None otherwise.
        """
        if order_exec.state != OrderState.FILLED:
            return None

        exec_sum = order_exec.execution_summary
        filled_price = order_exec.filled_price

        # Check liquidation first (most critical)
        if exec_sum.liquidation_price:
            if exec_sum.direction == "LONG" and current_bar["low"] <= exec_sum.liquidation_price:
                return "liquidation"
            elif exec_sum.direction == "SHORT" and current_bar["high"] >= exec_sum.liquidation_price:
                return "liquidation"

        # Check take profit
        if exec_sum.direction == "LONG" and current_bar["high"] >= exec_sum.take_profit:
            return "take_profit"
        elif exec_sum.direction == "SHORT" and current_bar["low"] <= exec_sum.take_profit:
            return "take_profit"

        # Check stop loss
        if exec_sum.direction == "LONG" and current_bar["low"] <= exec_sum.stop_loss:
            return "stop_loss"
        elif exec_sum.direction == "SHORT" and current_bar["high"] >= exec_sum.stop_loss:
            return "stop_loss"

        # Check EET expiration
        if exec_sum.execution_expired_time and current_time >= exec_sum.execution_expired_time:
            return "eet"

        return None

    def close_order(
        self,
        order_exec: OrderExecution,
        close_reason: str,
        current_bar: pd.Series,
        current_time: datetime
    ):
        """Close an order and calculate PnL."""
        exec_sum = order_exec.execution_summary
        filled_price = order_exec.filled_price

        # Determine close price based on reason
        if close_reason == "take_profit":
            close_price = exec_sum.take_profit
        elif close_reason == "stop_loss":
            close_price = exec_sum.stop_loss
        elif close_reason == "liquidation":
            close_price = exec_sum.liquidation_price or filled_price
        else:  # eet or other
            # Close at current price
            close_price = current_bar["close"]

        order_exec.state = OrderState.CLOSED
        order_exec.closed_price = close_price
        order_exec.closed_time = current_time
        order_exec.close_reason = close_reason

        # Calculate PnL
        if exec_sum.direction == "LONG":
            pnl_pct = (close_price - filled_price) / filled_price
        else:  # SHORT
            pnl_pct = (filled_price - close_price) / filled_price

        # Apply leverage
        pnl_pct *= exec_sum.leverage

        # Calculate absolute PnL
        position_value = self.current_capital * exec_sum.position_size
        pnl = position_value * pnl_pct

        order_exec.pnl = pnl
        order_exec.pnl_pct = pnl_pct

        # Update capital
        self.current_capital += pnl

        # Move from open to closed
        if order_exec.order_id in self.open_orders:
            del self.open_orders[order_exec.order_id]
        self.closed_orders.append(order_exec)

    def process_bar(self, bar: pd.Series, bar_time: datetime):
        """Process a single price bar, checking all open orders."""
        # First, check for fills
        for order_exec in list(self.open_orders.values()):
            if self.check_order_fill(order_exec, bar, bar_time):
                # Order was just filled, now check if it should close immediately
                close_reason = self.check_order_close(order_exec, bar, bar_time)
                if close_reason:
                    self.close_order(order_exec, close_reason, bar, bar_time)

        # Then, check for closes on already-filled orders
        for order_exec in list(self.open_orders.values()):
            if order_exec.state == OrderState.FILLED:
                close_reason = self.check_order_close(order_exec, bar, bar_time)
                if close_reason:
                    self.close_order(order_exec, close_reason, bar, bar_time)
