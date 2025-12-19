"""Position tracker for real-time PnL monitoring."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
import asyncio

from .events import EventBus, TradingEvent, EventTypes
from .bybit_client import BybitTestnetClient, Position


@dataclass
class TrackedPosition:
    """A position being tracked for PnL monitoring."""
    symbol: str
    side: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    leverage: int
    liquidation_price: float
    last_update: datetime

    # Alert thresholds
    pnl_alert_pct: float = 5.0  # Alert when PnL moves +/- this %
    last_alert_pnl_pct: float = 0.0

    @property
    def value_usd(self) -> float:
        return self.size * self.current_price

    @property
    def margin_usd(self) -> float:
        return self.value_usd / self.leverage if self.leverage > 0 else self.value_usd

    def should_alert(self) -> bool:
        """Check if PnL change warrants an alert."""
        pnl_change = abs(self.unrealized_pnl_pct - self.last_alert_pnl_pct)
        return pnl_change >= self.pnl_alert_pct

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "size": self.size,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_pnl_pct": self.unrealized_pnl_pct,
            "leverage": self.leverage,
            "liquidation_price": self.liquidation_price,
            "value_usd": self.value_usd,
            "margin_usd": self.margin_usd,
            "last_update": self.last_update.isoformat(),
        }


class PositionTracker:
    """
    Tracks open positions and monitors for PnL changes.

    Responsibilities:
    - Sync positions from Bybit
    - Track real-time unrealized PnL
    - Emit events on significant changes
    - Monitor for liquidation risk
    """

    def __init__(
        self,
        client: BybitTestnetClient,
        event_bus: EventBus,
        pnl_alert_threshold_pct: float = 5.0,
        liquidation_warning_pct: float = 10.0,
    ):
        """
        Initialize position tracker.

        Args:
            client: Bybit Testnet client
            event_bus: Event bus for publishing events
            pnl_alert_threshold_pct: Alert when PnL moves +/- this %
            liquidation_warning_pct: Warn when price is within X% of liquidation
        """
        self.client = client
        self.event_bus = event_bus
        self.pnl_alert_threshold_pct = pnl_alert_threshold_pct
        self.liquidation_warning_pct = liquidation_warning_pct

        self.positions: Dict[str, TrackedPosition] = {}
        self.closed_positions: List[TrackedPosition] = []

        self._tracking = False
        self._track_task: Optional[asyncio.Task] = None

        # Aggregates
        self.total_unrealized_pnl: float = 0.0
        self.total_value_usd: float = 0.0

    async def sync_positions(self) -> None:
        """Sync positions from Bybit."""
        try:
            bybit_positions = await self.client.get_positions()

            current_symbols = set()

            for pos in bybit_positions:
                symbol = pos.symbol
                current_symbols.add(symbol)

                # Calculate unrealized PnL %
                if pos.entry_price > 0:
                    if pos.side == "Buy":
                        pnl_pct = (pos.mark_price - pos.entry_price) / pos.entry_price * 100 * pos.leverage
                    else:
                        pnl_pct = (pos.entry_price - pos.mark_price) / pos.entry_price * 100 * pos.leverage
                else:
                    pnl_pct = 0.0

                if symbol in self.positions:
                    # Update existing position
                    tracked = self.positions[symbol]
                    old_pnl_pct = tracked.unrealized_pnl_pct

                    tracked.current_price = pos.mark_price
                    tracked.unrealized_pnl = pos.unrealized_pnl
                    tracked.unrealized_pnl_pct = pnl_pct
                    tracked.size = pos.size
                    tracked.liquidation_price = pos.liquidation_price
                    tracked.last_update = datetime.now(timezone.utc)

                    # Check for PnL alert
                    if tracked.should_alert():
                        self._emit_pnl_alert(tracked)
                        tracked.last_alert_pnl_pct = tracked.unrealized_pnl_pct

                    # Check for liquidation warning
                    self._check_liquidation_risk(tracked)
                else:
                    # New position
                    tracked = TrackedPosition(
                        symbol=symbol,
                        side=pos.side,
                        size=pos.size,
                        entry_price=pos.entry_price,
                        current_price=pos.mark_price,
                        unrealized_pnl=pos.unrealized_pnl,
                        unrealized_pnl_pct=pnl_pct,
                        leverage=pos.leverage,
                        liquidation_price=pos.liquidation_price,
                        last_update=datetime.now(timezone.utc),
                        pnl_alert_pct=self.pnl_alert_threshold_pct,
                    )
                    self.positions[symbol] = tracked

                    # Emit position opened event
                    self.event_bus.publish(TradingEvent.now(
                        event_type=EventTypes.POSITION_OPENED,
                        payload=tracked.to_dict(),
                        source="position_tracker",
                    ))

            # Check for closed positions
            for symbol in list(self.positions.keys()):
                if symbol not in current_symbols:
                    closed = self.positions.pop(symbol)
                    self.closed_positions.append(closed)

                    # Emit position closed event
                    self.event_bus.publish(TradingEvent.now(
                        event_type=EventTypes.POSITION_CLOSED,
                        payload={
                            **closed.to_dict(),
                            "realized_pnl": closed.unrealized_pnl,  # Now realized
                        },
                        source="position_tracker",
                    ))

            # Update aggregates
            self._update_aggregates()

        except Exception as e:
            print(f"[PositionTracker] Sync error: {e}")

    def _emit_pnl_alert(self, position: TrackedPosition) -> None:
        """Emit an alert for significant PnL change."""
        direction = "gained" if position.unrealized_pnl > 0 else "lost"

        self.event_bus.publish(TradingEvent.now(
            event_type=EventTypes.PNL_UPDATE,
            payload={
                **position.to_dict(),
                "alert_type": "pnl_change",
                "message": f"{position.symbol} {direction} {abs(position.unrealized_pnl_pct):.1f}%",
            },
            severity="warning" if position.unrealized_pnl < 0 else "info",
            source="position_tracker",
        ))

    def _check_liquidation_risk(self, position: TrackedPosition) -> None:
        """Check if position is approaching liquidation."""
        if position.liquidation_price <= 0:
            return

        if position.side == "Buy":
            distance_pct = (position.current_price - position.liquidation_price) / position.current_price * 100
        else:
            distance_pct = (position.liquidation_price - position.current_price) / position.current_price * 100

        if distance_pct <= self.liquidation_warning_pct:
            self.event_bus.publish(TradingEvent.now(
                event_type=EventTypes.RISK_BREACH,
                payload={
                    **position.to_dict(),
                    "alert_type": "liquidation_risk",
                    "distance_pct": distance_pct,
                    "message": f"{position.symbol} is {distance_pct:.1f}% from liquidation!",
                },
                severity="critical",
                source="position_tracker",
            ))

    def _update_aggregates(self) -> None:
        """Update aggregate metrics."""
        self.total_unrealized_pnl = sum(p.unrealized_pnl for p in self.positions.values())
        self.total_value_usd = sum(p.value_usd for p in self.positions.values())

    async def start_tracking(self, interval_seconds: int = 10) -> None:
        """
        Start background position tracking.

        Args:
            interval_seconds: How often to sync positions
        """
        if self._tracking:
            return

        self._tracking = True
        self._track_task = asyncio.create_task(self._track_loop(interval_seconds))
        print("[PositionTracker] Started position tracking")

    async def stop_tracking(self) -> None:
        """Stop background tracking."""
        self._tracking = False
        if self._track_task:
            self._track_task.cancel()
            try:
                await self._track_task
            except asyncio.CancelledError:
                pass
        print("[PositionTracker] Stopped position tracking")

    async def _track_loop(self, interval: int) -> None:
        """Background loop for position syncing."""
        while self._tracking:
            await self.sync_positions()
            await asyncio.sleep(interval)

    def get_position(self, symbol: str) -> Optional[TrackedPosition]:
        """Get a specific position."""
        return self.positions.get(symbol)

    def get_all_positions(self) -> List[TrackedPosition]:
        """Get all open positions."""
        return list(self.positions.values())

    def get_positions_summary(self) -> Dict[str, Any]:
        """Get a summary of all positions."""
        return {
            "count": len(self.positions),
            "total_value_usd": self.total_value_usd,
            "total_unrealized_pnl": self.total_unrealized_pnl,
            "positions": [p.to_dict() for p in self.positions.values()],
        }

    def get_position_for_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get position info for UI/reporting."""
        pos = self.positions.get(symbol)
        return pos.to_dict() if pos else None
