"""Admin Agent for automated reporting and monitoring."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
import threading
import time

from .base import BaseAgent
from ..services.events import EventBus, TradingEvent, EventTypes
from ..services.alerts import AlertManager, Alert
from ..services.notifications import NotificationService, Report
from ..services.reports import ReportGenerator
from ..services.metrics import PerformanceTracker


@dataclass
class AdminConfig:
    """Configuration for Admin Agent."""
    # Alert thresholds
    max_drawdown_pct: float = 10.0
    daily_loss_limit_pct: float = 5.0
    risk_breach_threshold: int = 3

    # Report schedule
    daily_summary_enabled: bool = True
    daily_summary_hour: int = 0  # UTC hour
    weekly_summary_enabled: bool = True
    weekly_summary_day: str = "sunday"  # Day of week

    # Notification settings
    slack_webhook: Optional[str] = None
    email: Optional[str] = None
    console_enabled: bool = True
    log_dir: str = "logs/admin"


class AdminAgent(BaseAgent):
    """
    Admin Agent for automated reporting, monitoring, and alerting.

    Responsibilities:
    - Monitor system health via event bus
    - Check alert rules and send notifications
    - Generate periodic reports (daily/weekly)
    - Track and aggregate system metrics
    """

    def __init__(
        self,
        id: str,
        tracker: PerformanceTracker,
        event_bus: EventBus,
        config: Optional[AdminConfig] = None,
    ):
        super().__init__(id=id)
        self.tracker = tracker
        self.event_bus = event_bus
        self.config = config or AdminConfig()

        # Initialize components
        self.alert_manager = AlertManager()
        self.alert_manager.add_default_rules(
            max_drawdown_pct=self.config.max_drawdown_pct,
            daily_loss_limit_pct=self.config.daily_loss_limit_pct,
            risk_breach_threshold=self.config.risk_breach_threshold,
        )

        self.notifier = NotificationService(
            slack_webhook=self.config.slack_webhook,
            log_dir=self.config.log_dir,
            console_enabled=self.config.console_enabled,
        )

        self.report_generator = ReportGenerator(tracker)

        # State
        self._running = False
        self._scheduler_thread: Optional[threading.Thread] = None
        self._last_daily_report: Optional[datetime] = None
        self._last_weekly_report: Optional[datetime] = None

        # Subscribe to events
        self._subscribe_to_events()

    def _subscribe_to_events(self) -> None:
        """Subscribe to relevant events from the event bus."""
        # Subscribe to all events for monitoring
        self.event_bus.subscribe("*", self._on_event)

        # Subscribe to specific critical events
        self.event_bus.subscribe(EventTypes.RISK_HARD_FAIL, self._on_risk_breach)
        self.event_bus.subscribe(EventTypes.RISK_SOFT_FAIL, self._on_risk_breach)
        self.event_bus.subscribe(EventTypes.POSITION_CLOSED, self._on_position_closed)

    def _on_event(self, event: TradingEvent) -> None:
        """Handle any event - check alerts."""
        # Build context from current state
        context = self._build_alert_context(event)

        # Check all alert rules
        triggered = self.alert_manager.check_all(context)

        # Send notifications for triggered alerts
        for alert in triggered:
            self._send_alert(alert)

    def _on_risk_breach(self, event: TradingEvent) -> None:
        """Handle risk breach events specifically."""
        severity = "critical" if event.event_type == EventTypes.RISK_HARD_FAIL else "warning"

        self.notifier.send(
            title=f"Risk {event.event_type.replace('risk_', '').replace('_', ' ').title()}",
            message=f"Order rejected: {event.payload.get('reason', 'Unknown')}",
            severity=severity,
        )

    def _on_position_closed(self, event: TradingEvent) -> None:
        """Handle position closed events."""
        pnl = event.payload.get("pnl", 0)
        symbol = event.payload.get("symbol", "Unknown")

        # Notify on significant losses
        if pnl < -1000:  # Significant loss threshold
            self.notifier.warning(
                title="Large Loss",
                message=f"Position closed on {symbol} with PnL: ${pnl:,.2f}",
            )

    def _build_alert_context(self, event: TradingEvent) -> Dict[str, Any]:
        """Build context dictionary for alert checking."""
        # Get metrics from tracker
        agent_scores = self.tracker.agent_scores
        trades = self.tracker.trades

        # Calculate current metrics
        recent_trades = [t for t in trades if t.timestamp >= datetime.now(timezone.utc) - timedelta(days=1)]

        total_pnl = sum(t.pnl for t in recent_trades)

        # Count risk breaches in last 24h
        risk_events = self.event_bus.get_events_since(
            datetime.now(timezone.utc) - timedelta(hours=24),
            EventTypes.RISK_HARD_FAIL,
        )

        return {
            "event_type": event.event_type,
            "daily_pnl_pct": total_pnl,  # Simplified - would need proper calculation
            "risk_breach_count_24h": len(risk_events),
            "current_drawdown_pct": self._calculate_current_drawdown(),
            "rolling_sharpe": self._calculate_rolling_sharpe(),
            "consecutive_losses": self._count_consecutive_losses(),
            "positions": {},  # Would be populated from position tracker
            **event.payload,
        }

    def _calculate_current_drawdown(self) -> float:
        """Calculate current drawdown from peak."""
        trades = self.tracker.trades
        if not trades:
            return 0.0

        cumulative = 0
        peak = 0

        for trade in sorted(trades, key=lambda t: t.timestamp):
            cumulative += trade.pnl_pct
            peak = max(peak, cumulative)

        return peak - cumulative

    def _calculate_rolling_sharpe(self, days: int = 30) -> float:
        """Calculate rolling Sharpe ratio."""
        import numpy as np

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        trades = [t for t in self.tracker.trades if t.timestamp >= cutoff]

        if len(trades) < 2:
            return 0.0

        returns = [t.pnl_pct for t in trades]
        if np.std(returns) == 0:
            return 0.0

        return np.mean(returns) / np.std(returns) * np.sqrt(252)

    def _count_consecutive_losses(self) -> int:
        """Count current streak of consecutive losses."""
        trades = sorted(self.tracker.trades, key=lambda t: t.timestamp, reverse=True)

        count = 0
        for trade in trades:
            if trade.pnl < 0:
                count += 1
            else:
                break

        return count

    def _send_alert(self, alert: Alert) -> None:
        """Send an alert notification."""
        self.notifier.send(
            title=alert.title,
            message=alert.message,
            severity=alert.severity,
            metadata=alert.metadata,
        )

        # Publish alert event
        self.event_bus.publish(TradingEvent.now(
            event_type=EventTypes.ALERT_TRIGGERED,
            payload=alert.to_dict(),
            severity=alert.severity,
            source=f"admin:{self.id}",
        ))

    def generate_performance_report(self, lookback_days: int = 30) -> Report:
        """Generate a performance report."""
        return self.report_generator.generate_performance_report(lookback_days)

    def generate_daily_summary(self) -> Report:
        """Generate today's daily summary."""
        return self.report_generator.generate_daily_summary()

    def generate_weekly_summary(self) -> Report:
        """Generate this week's summary."""
        return self.report_generator.generate_weekly_summary()

    def send_performance_report(self, lookback_days: int = 30) -> Dict[str, bool]:
        """Generate and send a performance report."""
        report = self.generate_performance_report(lookback_days)
        results = self.notifier.send_report(report)

        self.event_bus.publish(TradingEvent.now(
            event_type=EventTypes.REPORT_GENERATED,
            payload={"report_type": "performance", "lookback_days": lookback_days},
            source=f"admin:{self.id}",
        ))

        return results

    def send_daily_summary(self) -> Dict[str, bool]:
        """Generate and send today's daily summary."""
        report = self.generate_daily_summary()
        results = self.notifier.send_report(report)
        self._last_daily_report = datetime.now(timezone.utc)

        self.event_bus.publish(TradingEvent.now(
            event_type=EventTypes.REPORT_GENERATED,
            payload={"report_type": "daily"},
            source=f"admin:{self.id}",
        ))

        return results

    def send_weekly_summary(self) -> Dict[str, bool]:
        """Generate and send this week's summary."""
        report = self.generate_weekly_summary()
        results = self.notifier.send_report(report)
        self._last_weekly_report = datetime.now(timezone.utc)

        self.event_bus.publish(TradingEvent.now(
            event_type=EventTypes.REPORT_GENERATED,
            payload={"report_type": "weekly"},
            source=f"admin:{self.id}",
        ))

        return results

    def start_scheduler(self) -> None:
        """Start the background scheduler for periodic reports."""
        if self._running:
            return

        self._running = True
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            daemon=True,
        )
        self._scheduler_thread.start()
        self.log("Scheduler started")

    def stop_scheduler(self) -> None:
        """Stop the background scheduler."""
        self._running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)
        self.log("Scheduler stopped")

    def _scheduler_loop(self) -> None:
        """Background loop for scheduled reports."""
        while self._running:
            now = datetime.now(timezone.utc)

            # Check for daily summary
            if self.config.daily_summary_enabled:
                if self._should_send_daily(now):
                    try:
                        self.send_daily_summary()
                    except Exception as e:
                        self.log(f"Error sending daily summary: {e}")

            # Check for weekly summary
            if self.config.weekly_summary_enabled:
                if self._should_send_weekly(now):
                    try:
                        self.send_weekly_summary()
                    except Exception as e:
                        self.log(f"Error sending weekly summary: {e}")

            # Sleep for a minute before checking again
            time.sleep(60)

    def _should_send_daily(self, now: datetime) -> bool:
        """Check if daily summary should be sent."""
        if now.hour != self.config.daily_summary_hour:
            return False

        if self._last_daily_report:
            # Don't send if we already sent today
            if self._last_daily_report.date() == now.date():
                return False

        return True

    def _should_send_weekly(self, now: datetime) -> bool:
        """Check if weekly summary should be sent."""
        day_names = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        current_day = day_names[now.weekday()]

        if current_day != self.config.weekly_summary_day.lower():
            return False

        if now.hour != self.config.daily_summary_hour:
            return False

        if self._last_weekly_report:
            # Don't send if we already sent this week
            days_since = (now - self._last_weekly_report).days
            if days_since < 6:
                return False

        return True

    def get_status(self) -> Dict[str, Any]:
        """Get current status of the admin agent."""
        return {
            "id": self.id,
            "running": self._running,
            "alert_rules": len(self.alert_manager.rules),
            "recent_alerts": len(self.alert_manager.get_recent_alerts(hours=24)),
            "last_daily_report": self._last_daily_report.isoformat() if self._last_daily_report else None,
            "last_weekly_report": self._last_weekly_report.isoformat() if self._last_weekly_report else None,
        }
