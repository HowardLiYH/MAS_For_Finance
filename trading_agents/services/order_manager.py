"""Order manager for paper trading lifecycle."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from enum import Enum
import asyncio

from .events import EventBus, TradingEvent, EventTypes
from .bybit_client import BybitTestnetClient, OrderResponse, Position
from ..models import ExecutionSummary


class OrderStatus(Enum):
    """Order status states."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class ManagedOrder:
    """An order being managed by the OrderManager."""
    order_id: str
    symbol: str
    side: str
    order_type: str
    quantity: float
    price: Optional[float]
    take_profit: Optional[float]
    stop_loss: Optional[float]
    leverage: int
    status: OrderStatus
    created_at: datetime
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    bybit_order_id: Optional[str] = None
    execution_summary: Optional[ExecutionSummary] = None
    error_message: Optional[str] = None

    @property
    def is_complete(self) -> bool:
        return self.status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "order_type": self.order_type,
            "quantity": self.quantity,
            "price": self.price,
            "take_profit": self.take_profit,
            "stop_loss": self.stop_loss,
            "leverage": self.leverage,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "filled_qty": self.filled_qty,
            "avg_fill_price": self.avg_fill_price,
            "bybit_order_id": self.bybit_order_id,
        }


class OrderManager:
    """
    Manages order lifecycle for paper trading.

    Responsibilities:
    - Submit orders to Bybit Testnet
    - Track order status
    - Monitor for fills
    - Emit events on state changes
    """

    def __init__(
        self,
        client: BybitTestnetClient,
        event_bus: EventBus,
        max_position_usd: float = 10000,
    ):
        """
        Initialize order manager.

        Args:
            client: Bybit Testnet client
            event_bus: Event bus for publishing events
            max_position_usd: Maximum position size in USD
        """
        self.client = client
        self.event_bus = event_bus
        self.max_position_usd = max_position_usd

        self.pending_orders: Dict[str, ManagedOrder] = {}
        self.filled_orders: Dict[str, ManagedOrder] = {}
        self.order_history: List[ManagedOrder] = []

        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None

    async def submit_order(self, execution: ExecutionSummary) -> ManagedOrder:
        """
        Submit an order from ExecutionSummary.

        Args:
            execution: Execution summary from Trader agent

        Returns:
            ManagedOrder tracking object
        """
        # Create managed order
        order = ManagedOrder(
            order_id=execution.order_id,
            symbol=self._normalize_symbol(execution.symbol if hasattr(execution, 'symbol') else "BTCUSDT"),
            side="Buy" if execution.direction == "long" else "Sell",
            order_type="Market" if execution.order_type == "market" else "Limit",
            quantity=execution.position_size,
            price=execution.entry_price if execution.order_type == "limit" else None,
            take_profit=execution.take_profit,
            stop_loss=execution.stop_loss,
            leverage=execution.leverage,
            status=OrderStatus.PENDING,
            created_at=datetime.now(timezone.utc),
            execution_summary=execution,
        )

        self.pending_orders[order.order_id] = order

        # Emit order created event
        self.event_bus.publish(TradingEvent.now(
            event_type=EventTypes.ORDER_CREATED,
            payload=order.to_dict(),
            source="order_manager",
        ))

        try:
            # Set leverage first
            await self.client.set_leverage(order.symbol, order.leverage)

            # Submit to Bybit
            response = await self.client.place_order(
                symbol=order.symbol,
                side=order.side,
                order_type=order.order_type,
                qty=order.quantity,
                price=order.price,
                take_profit=order.take_profit,
                stop_loss=order.stop_loss,
            )

            order.bybit_order_id = response.order_id
            order.status = OrderStatus.SUBMITTED

            # Emit submitted event
            self.event_bus.publish(TradingEvent.now(
                event_type=EventTypes.ORDER_SUBMITTED,
                payload=order.to_dict(),
                source="order_manager",
            ))

            print(f"[OrderManager] Order submitted: {order.order_id} -> Bybit ID: {response.order_id}")

        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.error_message = str(e)

            # Move to history
            del self.pending_orders[order.order_id]
            self.order_history.append(order)

            # Emit rejected event
            self.event_bus.publish(TradingEvent.now(
                event_type=EventTypes.ORDER_REJECTED,
                payload={**order.to_dict(), "error": str(e)},
                severity="warning",
                source="order_manager",
            ))

            print(f"[OrderManager] Order rejected: {order.order_id} - {e}")

        return order

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a pending order.

        Args:
            order_id: Internal order ID

        Returns:
            True if cancelled successfully
        """
        if order_id not in self.pending_orders:
            return False

        order = self.pending_orders[order_id]

        if order.bybit_order_id:
            success = await self.client.cancel_order(order.symbol, order.bybit_order_id)
            if not success:
                return False

        order.status = OrderStatus.CANCELLED

        # Move to history
        del self.pending_orders[order_id]
        self.order_history.append(order)

        # Emit cancelled event
        self.event_bus.publish(TradingEvent.now(
            event_type=EventTypes.ORDER_CANCELLED,
            payload=order.to_dict(),
            source="order_manager",
        ))

        return True

    async def start_monitoring(self, interval_seconds: int = 5) -> None:
        """
        Start background monitoring of pending orders.

        Args:
            interval_seconds: How often to check order status
        """
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop(interval_seconds))
        print("[OrderManager] Started order monitoring")

    async def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        print("[OrderManager] Stopped order monitoring")

    async def _monitor_loop(self, interval: int) -> None:
        """Background loop to check order status."""
        while self._monitoring:
            try:
                await self._check_pending_orders()
            except Exception as e:
                print(f"[OrderManager] Monitor error: {e}")

            await asyncio.sleep(interval)

    async def _check_pending_orders(self) -> None:
        """Check status of all pending orders."""
        orders_to_check = list(self.pending_orders.values())

        for order in orders_to_check:
            if not order.bybit_order_id:
                continue

            try:
                bybit_order = await self.client.get_order(order.symbol, order.bybit_order_id)

                if not bybit_order:
                    continue

                status = bybit_order.get("orderStatus", "")

                if status == "Filled":
                    order.status = OrderStatus.FILLED
                    order.filled_qty = float(bybit_order.get("cumExecQty", 0))
                    order.avg_fill_price = float(bybit_order.get("avgPrice", 0))

                    # Move to filled orders
                    del self.pending_orders[order.order_id]
                    self.filled_orders[order.order_id] = order
                    self.order_history.append(order)

                    # Emit filled event
                    self.event_bus.publish(TradingEvent.now(
                        event_type=EventTypes.ORDER_FILLED,
                        payload=order.to_dict(),
                        source="order_manager",
                    ))

                    print(f"[OrderManager] Order filled: {order.order_id} @ {order.avg_fill_price}")

                elif status == "PartiallyFilled":
                    order.status = OrderStatus.PARTIALLY_FILLED
                    order.filled_qty = float(bybit_order.get("cumExecQty", 0))

                elif status in ("Cancelled", "Rejected"):
                    order.status = OrderStatus.CANCELLED if status == "Cancelled" else OrderStatus.REJECTED

                    del self.pending_orders[order.order_id]
                    self.order_history.append(order)

            except Exception as e:
                print(f"[OrderManager] Error checking order {order.order_id}: {e}")

    def get_pending_orders(self) -> List[ManagedOrder]:
        """Get all pending orders."""
        return list(self.pending_orders.values())

    def get_filled_orders(self) -> List[ManagedOrder]:
        """Get all filled orders."""
        return list(self.filled_orders.values())

    def get_order(self, order_id: str) -> Optional[ManagedOrder]:
        """Get an order by ID."""
        if order_id in self.pending_orders:
            return self.pending_orders[order_id]
        if order_id in self.filled_orders:
            return self.filled_orders[order_id]
        return None

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol to Bybit format."""
        # Convert various formats to BTCUSDT format
        symbol = symbol.upper()
        symbol = symbol.replace("USD.PERP", "USDT")
        symbol = symbol.replace("/", "")
        symbol = symbol.replace("-", "")

        if not symbol.endswith("USDT") and not symbol.endswith("USD"):
            symbol += "USDT"

        return symbol
