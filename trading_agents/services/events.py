"""Event bus system for system-wide communication."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Any, Callable, Optional
from collections import defaultdict
import asyncio
import threading
import queue


@dataclass
class TradingEvent:
    """A trading system event."""
    event_type: str  # "order_filled", "risk_breach", "pnl_update", etc.
    timestamp: datetime
    payload: Dict[str, Any]
    severity: str = "info"  # info, warning, critical
    source: str = "system"  # Which component generated this event

    @classmethod
    def now(
        cls,
        event_type: str,
        payload: Dict[str, Any],
        severity: str = "info",
        source: str = "system",
    ) -> "TradingEvent":
        """Create an event with current timestamp."""
        return cls(
            event_type=event_type,
            timestamp=datetime.now(timezone.utc),
            payload=payload,
            severity=severity,
            source=source,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "payload": self.payload,
            "severity": self.severity,
            "source": self.source,
        }


# Event type constants
class EventTypes:
    """Standard event types."""
    # Order events
    ORDER_CREATED = "order_created"
    ORDER_SUBMITTED = "order_submitted"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"

    # Risk events
    RISK_PASS = "risk_pass"
    RISK_SOFT_FAIL = "risk_soft_fail"
    RISK_HARD_FAIL = "risk_hard_fail"
    RISK_BREACH = "risk_breach"

    # PnL events
    PNL_UPDATE = "pnl_update"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"

    # System events
    ITERATION_START = "iteration_start"
    ITERATION_END = "iteration_end"
    ALERT_TRIGGERED = "alert_triggered"
    REPORT_GENERATED = "report_generated"

    # Performance events
    DRAWDOWN_WARNING = "drawdown_warning"
    SHARPE_WARNING = "sharpe_warning"
    DAILY_LOSS_LIMIT = "daily_loss_limit"


class EventBus:
    """
    Central event bus for publishing and subscribing to trading events.

    Supports both synchronous and asynchronous event handling.
    Thread-safe for use across multiple components.
    """

    def __init__(self, max_history: int = 1000):
        """
        Initialize the event bus.

        Args:
            max_history: Maximum number of events to keep in history
        """
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._async_subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._wildcard_subscribers: List[Callable] = []
        self._async_wildcard_subscribers: List[Callable] = []
        self._history: List[TradingEvent] = []
        self._max_history = max_history
        self._lock = threading.Lock()
        self._event_queue: queue.Queue = queue.Queue()

    def subscribe(
        self,
        event_type: str,
        callback: Callable[[TradingEvent], None],
    ) -> None:
        """
        Subscribe to a specific event type.

        Args:
            event_type: Event type to subscribe to, or "*" for all events
            callback: Function to call when event is published
        """
        with self._lock:
            if event_type == "*":
                self._wildcard_subscribers.append(callback)
            else:
                self._subscribers[event_type].append(callback)

    def subscribe_async(
        self,
        event_type: str,
        callback: Callable[[TradingEvent], Any],
    ) -> None:
        """
        Subscribe with an async callback.

        Args:
            event_type: Event type to subscribe to, or "*" for all events
            callback: Async function to call when event is published
        """
        with self._lock:
            if event_type == "*":
                self._async_wildcard_subscribers.append(callback)
            else:
                self._async_subscribers[event_type].append(callback)

    def unsubscribe(
        self,
        event_type: str,
        callback: Callable,
    ) -> bool:
        """
        Unsubscribe a callback from an event type.

        Returns:
            True if callback was found and removed
        """
        with self._lock:
            if event_type == "*":
                if callback in self._wildcard_subscribers:
                    self._wildcard_subscribers.remove(callback)
                    return True
                if callback in self._async_wildcard_subscribers:
                    self._async_wildcard_subscribers.remove(callback)
                    return True
            else:
                if callback in self._subscribers[event_type]:
                    self._subscribers[event_type].remove(callback)
                    return True
                if callback in self._async_subscribers[event_type]:
                    self._async_subscribers[event_type].remove(callback)
                    return True
        return False

    def publish(self, event: TradingEvent) -> None:
        """
        Publish an event to all subscribers (synchronous).

        Args:
            event: The event to publish
        """
        with self._lock:
            # Add to history
            self._history.append(event)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

            # Get callbacks
            callbacks = list(self._subscribers.get(event.event_type, []))
            callbacks.extend(self._wildcard_subscribers)

        # Call callbacks outside lock
        for callback in callbacks:
            try:
                callback(event)
            except Exception as e:
                print(f"[EventBus] Error in callback: {e}")

    async def publish_async(self, event: TradingEvent) -> None:
        """
        Publish an event to all subscribers (asynchronous).

        Args:
            event: The event to publish
        """
        # First publish synchronously
        self.publish(event)

        # Then call async subscribers
        with self._lock:
            async_callbacks = list(self._async_subscribers.get(event.event_type, []))
            async_callbacks.extend(self._async_wildcard_subscribers)

        # Call async callbacks
        for callback in async_callbacks:
            try:
                await callback(event)
            except Exception as e:
                print(f"[EventBus] Error in async callback: {e}")

    def get_history(
        self,
        event_type: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 100,
    ) -> List[TradingEvent]:
        """
        Get event history with optional filtering.

        Args:
            event_type: Filter by event type
            severity: Filter by severity
            limit: Maximum number of events to return

        Returns:
            List of events (most recent first)
        """
        with self._lock:
            events = list(reversed(self._history))

        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if severity:
            events = [e for e in events if e.severity == severity]

        return events[:limit]

    def get_events_since(
        self,
        since: datetime,
        event_type: Optional[str] = None,
    ) -> List[TradingEvent]:
        """
        Get events since a specific time.

        Args:
            since: Start time
            event_type: Optional filter by event type

        Returns:
            List of events since the given time
        """
        with self._lock:
            events = [e for e in self._history if e.timestamp >= since]

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        return events

    def clear_history(self) -> None:
        """Clear event history."""
        with self._lock:
            self._history.clear()


# Global event bus instance (optional singleton pattern)
_global_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get or create the global event bus instance."""
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = EventBus()
    return _global_event_bus
