"""Notification service for alerts and reports."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
import json
import os


@dataclass
class Notification:
    """A notification to be sent."""
    title: str
    message: str
    severity: str = "info"  # info, warning, critical
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "message": self.message,
            "severity": self.severity,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class Report:
    """A report to be sent."""
    title: str
    sections: List[Dict[str, Any]]
    report_type: str  # "daily", "weekly", "performance", "alert"
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_markdown(self) -> str:
        """Convert report to markdown format."""
        lines = [f"# {self.title}", ""]
        lines.append(f"*Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M UTC')}*")
        lines.append("")

        for section in self.sections:
            lines.append(f"## {section.get('title', 'Section')}")
            lines.append("")

            if "text" in section:
                lines.append(section["text"])
                lines.append("")

            if "table" in section:
                table = section["table"]
                if table.get("headers") and table.get("rows"):
                    # Header row
                    lines.append("| " + " | ".join(table["headers"]) + " |")
                    lines.append("| " + " | ".join(["---"] * len(table["headers"])) + " |")
                    # Data rows
                    for row in table["rows"]:
                        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
                    lines.append("")

            if "metrics" in section:
                for key, value in section["metrics"].items():
                    lines.append(f"- **{key}**: {value}")
                lines.append("")

        return "\n".join(lines)

    def to_slack_blocks(self) -> List[Dict]:
        """Convert report to Slack Block Kit format."""
        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": self.title}
            },
            {
                "type": "context",
                "elements": [
                    {"type": "mrkdwn", "text": f"Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M UTC')}"}
                ]
            },
            {"type": "divider"}
        ]

        for section in self.sections:
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*{section.get('title', '')}*"}
            })

            if "text" in section:
                blocks.append({
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": section["text"]}
                })

            if "metrics" in section:
                fields = []
                for key, value in section["metrics"].items():
                    fields.append({"type": "mrkdwn", "text": f"*{key}*\n{value}"})
                # Slack allows max 10 fields per section
                for i in range(0, len(fields), 10):
                    blocks.append({
                        "type": "section",
                        "fields": fields[i:i+10]
                    })

        return blocks


class NotificationChannel(ABC):
    """Abstract base class for notification channels."""

    @abstractmethod
    def send(self, notification: Notification) -> bool:
        """Send a notification. Returns True if successful."""
        pass

    @abstractmethod
    def send_report(self, report: Report) -> bool:
        """Send a report. Returns True if successful."""
        pass


class ConsoleChannel(NotificationChannel):
    """Console output channel (always available)."""

    SEVERITY_ICONS = {
        "info": "ℹ️",
        "warning": "⚠️",
        "critical": "🚨",
    }

    def send(self, notification: Notification) -> bool:
        icon = self.SEVERITY_ICONS.get(notification.severity, "📢")
        print(f"\n{icon} [{notification.severity.upper()}] {notification.title}")
        print(f"   {notification.message}")
        return True

    def send_report(self, report: Report) -> bool:
        print("\n" + "=" * 60)
        print(report.to_markdown())
        print("=" * 60 + "\n")
        return True


class SlackChannel(NotificationChannel):
    """Slack webhook channel."""

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        self._http_client = None

    def _get_client(self):
        if self._http_client is None:
            try:
                import requests
                self._http_client = requests
            except ImportError:
                print("[SlackChannel] requests library not installed")
                return None
        return self._http_client

    def send(self, notification: Notification) -> bool:
        client = self._get_client()
        if not client:
            return False

        severity_emoji = {
            "info": ":information_source:",
            "warning": ":warning:",
            "critical": ":rotating_light:",
        }

        payload = {
            "text": f"{severity_emoji.get(notification.severity, '')} *{notification.title}*\n{notification.message}",
        }

        try:
            response = client.post(self.webhook_url, json=payload, timeout=10)
            return response.status_code == 200
        except Exception as e:
            print(f"[SlackChannel] Error sending notification: {e}")
            return False

    def send_report(self, report: Report) -> bool:
        client = self._get_client()
        if not client:
            return False

        payload = {"blocks": report.to_slack_blocks()}

        try:
            response = client.post(self.webhook_url, json=payload, timeout=30)
            return response.status_code == 200
        except Exception as e:
            print(f"[SlackChannel] Error sending report: {e}")
            return False


class FileChannel(NotificationChannel):
    """File-based logging channel."""

    def __init__(self, log_dir: str = "logs/notifications"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def send(self, notification: Notification) -> bool:
        date_str = notification.timestamp.strftime("%Y-%m-%d")
        log_file = os.path.join(self.log_dir, f"notifications_{date_str}.jsonl")

        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(notification.to_dict()) + "\n")
            return True
        except Exception as e:
            print(f"[FileChannel] Error writing notification: {e}")
            return False

    def send_report(self, report: Report) -> bool:
        date_str = report.generated_at.strftime("%Y-%m-%d_%H%M")
        report_file = os.path.join(self.log_dir, f"report_{report.report_type}_{date_str}.md")

        try:
            with open(report_file, "w") as f:
                f.write(report.to_markdown())
            return True
        except Exception as e:
            print(f"[FileChannel] Error writing report: {e}")
            return False


class NotificationService:
    """
    Unified notification delivery service.

    Manages multiple notification channels and routes messages appropriately.
    """

    def __init__(
        self,
        slack_webhook: Optional[str] = None,
        log_dir: Optional[str] = None,
        console_enabled: bool = True,
    ):
        """
        Initialize notification service.

        Args:
            slack_webhook: Slack webhook URL (or use SLACK_WEBHOOK env var)
            log_dir: Directory for file-based logging
            console_enabled: Whether to output to console
        """
        self.channels: Dict[str, NotificationChannel] = {}

        # Always add console if enabled
        if console_enabled:
            self.channels["console"] = ConsoleChannel()

        # Add file channel
        if log_dir:
            self.channels["file"] = FileChannel(log_dir)

        # Add Slack if configured
        webhook = slack_webhook or os.getenv("SLACK_WEBHOOK")
        if webhook:
            self.channels["slack"] = SlackChannel(webhook)

    def add_channel(self, name: str, channel: NotificationChannel) -> None:
        """Add a notification channel."""
        self.channels[name] = channel

    def remove_channel(self, name: str) -> bool:
        """Remove a notification channel."""
        if name in self.channels:
            del self.channels[name]
            return True
        return False

    def send(
        self,
        title: str,
        message: str,
        severity: str = "info",
        channels: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, bool]:
        """
        Send a notification to specified channels.

        Args:
            title: Notification title
            message: Notification message
            severity: Severity level (info, warning, critical)
            channels: List of channel names (None = all channels)
            metadata: Additional metadata

        Returns:
            Dict mapping channel name to success status
        """
        notification = Notification(
            title=title,
            message=message,
            severity=severity,
            metadata=metadata or {},
        )

        target_channels = channels or list(self.channels.keys())
        results = {}

        for channel_name in target_channels:
            if channel_name in self.channels:
                results[channel_name] = self.channels[channel_name].send(notification)
            else:
                results[channel_name] = False

        return results

    def send_report(
        self,
        report: Report,
        channels: Optional[List[str]] = None,
    ) -> Dict[str, bool]:
        """
        Send a report to specified channels.

        Args:
            report: The report to send
            channels: List of channel names (None = all channels)

        Returns:
            Dict mapping channel name to success status
        """
        target_channels = channels or list(self.channels.keys())
        results = {}

        for channel_name in target_channels:
            if channel_name in self.channels:
                results[channel_name] = self.channels[channel_name].send_report(report)
            else:
                results[channel_name] = False

        return results

    def alert(
        self,
        title: str,
        message: str,
        severity: str = "warning",
    ) -> Dict[str, bool]:
        """
        Send an alert (convenience method for warnings/critical).

        Always sends to all channels.
        """
        return self.send(title, message, severity=severity)

    def critical(self, title: str, message: str) -> Dict[str, bool]:
        """Send a critical alert."""
        return self.send(title, message, severity="critical")

    def warning(self, title: str, message: str) -> Dict[str, bool]:
        """Send a warning alert."""
        return self.send(title, message, severity="warning")

    def info(self, title: str, message: str) -> Dict[str, bool]:
        """Send an info notification."""
        return self.send(title, message, severity="info")
