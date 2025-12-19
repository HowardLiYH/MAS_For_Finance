"""Report generation for performance and daily summaries."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional

from .notifications import Report
from .metrics import PerformanceTracker


class ReportGenerator:
    """
    Generates various reports from performance data.
    """

    def __init__(self, tracker: PerformanceTracker):
        self.tracker = tracker

    def generate_performance_report(
        self,
        lookback_days: int = 30,
        symbols: Optional[List[str]] = None,
    ) -> Report:
        """
        Generate a comprehensive performance report.

        Args:
            lookback_days: Number of days to include
            symbols: Optional list of symbols to filter by

        Returns:
            Report object
        """
        metrics = self._calculate_metrics(lookback_days)

        sections = [
            {
                "title": "Performance Summary",
                "metrics": {
                    "Total PnL": f"${metrics.get('total_pnl', 0):,.2f}",
                    "Total PnL %": f"{metrics.get('total_pnl_pct', 0):.2f}%",
                    "Sharpe Ratio": f"{metrics.get('sharpe', 0):.2f}",
                    "Max Drawdown": f"{metrics.get('max_drawdown_pct', 0):.2f}%",
                    "Hit Rate": f"{metrics.get('hit_rate', 0):.1f}%",
                    "Profit Factor": f"{metrics.get('profit_factor', 0):.2f}",
                }
            },
            {
                "title": "Trade Statistics",
                "metrics": {
                    "Total Trades": str(metrics.get("total_trades", 0)),
                    "Winning Trades": str(metrics.get("winning_trades", 0)),
                    "Losing Trades": str(metrics.get("losing_trades", 0)),
                    "Avg Win": f"${metrics.get('avg_win', 0):,.2f}",
                    "Avg Loss": f"${metrics.get('avg_loss', 0):,.2f}",
                    "Largest Win": f"${metrics.get('largest_win', 0):,.2f}",
                    "Largest Loss": f"${metrics.get('largest_loss', 0):,.2f}",
                }
            },
        ]

        # Add per-symbol breakdown if available
        symbol_data = metrics.get("by_symbol", {})
        if symbol_data:
            rows = []
            for sym, data in symbol_data.items():
                rows.append([
                    sym,
                    f"${data.get('pnl', 0):,.2f}",
                    f"{data.get('trades', 0)}",
                    f"{data.get('hit_rate', 0):.1f}%",
                ])
            sections.append({
                "title": "Performance by Symbol",
                "table": {
                    "headers": ["Symbol", "PnL", "Trades", "Hit Rate"],
                    "rows": rows,
                }
            })

        return Report(
            title=f"Performance Report ({lookback_days} Days)",
            sections=sections,
            report_type="performance",
        )

    def generate_daily_summary(self, date: Optional[datetime] = None) -> Report:
        """
        Generate a daily trading summary.

        Args:
            date: Date to generate summary for (default: today)

        Returns:
            Report object
        """
        date = date or datetime.now(timezone.utc)
        date_str = date.strftime("%Y-%m-%d")

        daily_metrics = self._calculate_daily_metrics(date)

        sections = [
            {
                "title": "Daily Overview",
                "metrics": {
                    "Date": date_str,
                    "Trades Executed": str(daily_metrics.get("trades", 0)),
                    "Daily PnL": f"${daily_metrics.get('pnl', 0):,.2f}",
                    "Daily PnL %": f"{daily_metrics.get('pnl_pct', 0):.2f}%",
                    "Win Rate": f"{daily_metrics.get('win_rate', 0):.1f}%",
                }
            },
        ]

        # Add positions section
        positions = daily_metrics.get("positions", [])
        if positions:
            rows = []
            for pos in positions:
                rows.append([
                    pos.get("symbol", ""),
                    pos.get("side", ""),
                    f"${pos.get('size_usd', 0):,.0f}",
                    f"${pos.get('unrealized_pnl', 0):,.2f}",
                ])
            sections.append({
                "title": "Open Positions",
                "table": {
                    "headers": ["Symbol", "Side", "Size", "Unrealized PnL"],
                    "rows": rows,
                }
            })

        # Add alerts section
        alerts = daily_metrics.get("alerts", [])
        if alerts:
            alert_text = "\n".join([f"- [{a['severity']}] {a['title']}" for a in alerts])
            sections.append({
                "title": f"Alerts ({len(alerts)})",
                "text": alert_text,
            })

        return Report(
            title=f"Daily Summary - {date_str}",
            sections=sections,
            report_type="daily",
        )

    def generate_weekly_summary(self, week_end: Optional[datetime] = None) -> Report:
        """
        Generate a weekly trading summary.

        Args:
            week_end: End of week date (default: today)

        Returns:
            Report object
        """
        week_end = week_end or datetime.now(timezone.utc)
        week_start = week_end - timedelta(days=7)

        weekly_metrics = self._calculate_weekly_metrics(week_start, week_end)

        sections = [
            {
                "title": "Weekly Overview",
                "metrics": {
                    "Period": f"{week_start.strftime('%Y-%m-%d')} to {week_end.strftime('%Y-%m-%d')}",
                    "Total Trades": str(weekly_metrics.get("trades", 0)),
                    "Weekly PnL": f"${weekly_metrics.get('pnl', 0):,.2f}",
                    "Weekly PnL %": f"{weekly_metrics.get('pnl_pct', 0):.2f}%",
                    "Win Rate": f"{weekly_metrics.get('win_rate', 0):.1f}%",
                    "Best Day": weekly_metrics.get("best_day", "N/A"),
                    "Worst Day": weekly_metrics.get("worst_day", "N/A"),
                }
            },
        ]

        # Week over week comparison
        prev_week = weekly_metrics.get("previous_week", {})
        if prev_week:
            pnl_change = weekly_metrics.get("pnl", 0) - prev_week.get("pnl", 0)
            sections.append({
                "title": "Week over Week",
                "metrics": {
                    "PnL Change": f"${pnl_change:+,.2f}",
                    "Trade Count Change": f"{weekly_metrics.get('trades', 0) - prev_week.get('trades', 0):+d}",
                }
            })

        # Best and worst performers
        performers = weekly_metrics.get("performers", {})
        if performers.get("best"):
            best = performers["best"]
            sections.append({
                "title": "Best Performer",
                "text": f"**{best.get('symbol', 'N/A')}**: ${best.get('pnl', 0):,.2f} ({best.get('trades', 0)} trades)",
            })
        if performers.get("worst"):
            worst = performers["worst"]
            sections.append({
                "title": "Worst Performer",
                "text": f"**{worst.get('symbol', 'N/A')}**: ${worst.get('pnl', 0):,.2f} ({worst.get('trades', 0)} trades)",
            })

        return Report(
            title=f"Weekly Summary",
            sections=sections,
            report_type="weekly",
        )

    def generate_alert_summary(
        self,
        alerts: List[Dict[str, Any]],
        period_hours: int = 24,
    ) -> Report:
        """
        Generate an alert summary report.

        Args:
            alerts: List of alert dictionaries
            period_hours: Time period covered

        Returns:
            Report object
        """
        critical_count = len([a for a in alerts if a.get("severity") == "critical"])
        warning_count = len([a for a in alerts if a.get("severity") == "warning"])

        sections = [
            {
                "title": "Alert Summary",
                "metrics": {
                    "Period": f"Last {period_hours} hours",
                    "Total Alerts": str(len(alerts)),
                    "Critical": str(critical_count),
                    "Warning": str(warning_count),
                }
            },
        ]

        if alerts:
            rows = []
            for alert in alerts[:20]:  # Limit to 20 most recent
                rows.append([
                    alert.get("triggered_at", "")[:16],  # Trim to minutes
                    alert.get("severity", "").upper(),
                    alert.get("title", ""),
                ])
            sections.append({
                "title": "Recent Alerts",
                "table": {
                    "headers": ["Time", "Severity", "Title"],
                    "rows": rows,
                }
            })

        return Report(
            title="Alert Summary",
            sections=sections,
            report_type="alert",
        )

    def _calculate_metrics(self, lookback_days: int) -> Dict[str, Any]:
        """Calculate performance metrics from tracker data."""
        # Get trades from tracker
        trades = self.tracker.trades

        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        recent_trades = [t for t in trades if t.timestamp >= cutoff]

        if not recent_trades:
            return {"total_trades": 0}

        total_pnl = sum(t.pnl for t in recent_trades)
        winning = [t for t in recent_trades if t.pnl > 0]
        losing = [t for t in recent_trades if t.pnl < 0]

        return {
            "total_trades": len(recent_trades),
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "total_pnl": total_pnl,
            "total_pnl_pct": sum(t.pnl_pct for t in recent_trades),
            "hit_rate": len(winning) / len(recent_trades) * 100 if recent_trades else 0,
            "avg_win": sum(t.pnl for t in winning) / len(winning) if winning else 0,
            "avg_loss": sum(t.pnl for t in losing) / len(losing) if losing else 0,
            "largest_win": max((t.pnl for t in winning), default=0),
            "largest_loss": min((t.pnl for t in losing), default=0),
            "sharpe": self._calculate_sharpe(recent_trades),
            "max_drawdown_pct": self._calculate_max_drawdown(recent_trades),
            "profit_factor": abs(sum(t.pnl for t in winning) / sum(t.pnl for t in losing)) if losing and sum(t.pnl for t in losing) != 0 else 0,
        }

    def _calculate_daily_metrics(self, date: datetime) -> Dict[str, Any]:
        """Calculate metrics for a specific day."""
        start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)

        trades = [t for t in self.tracker.trades if start <= t.timestamp < end]

        return {
            "trades": len(trades),
            "pnl": sum(t.pnl for t in trades),
            "pnl_pct": sum(t.pnl_pct for t in trades),
            "win_rate": len([t for t in trades if t.pnl > 0]) / len(trades) * 100 if trades else 0,
            "positions": [],  # Would be populated from position tracker
            "alerts": [],  # Would be populated from alert manager
        }

    def _calculate_weekly_metrics(
        self,
        week_start: datetime,
        week_end: datetime,
    ) -> Dict[str, Any]:
        """Calculate metrics for a week."""
        trades = [t for t in self.tracker.trades if week_start <= t.timestamp < week_end]

        return {
            "trades": len(trades),
            "pnl": sum(t.pnl for t in trades),
            "pnl_pct": sum(t.pnl_pct for t in trades),
            "win_rate": len([t for t in trades if t.pnl > 0]) / len(trades) * 100 if trades else 0,
        }

    def _calculate_sharpe(self, trades: List) -> float:
        """Calculate Sharpe ratio from trades."""
        if len(trades) < 2:
            return 0.0

        import numpy as np
        returns = [t.pnl_pct for t in trades]
        if np.std(returns) == 0:
            return 0.0

        # Annualize (assuming daily trades)
        return np.mean(returns) / np.std(returns) * np.sqrt(252)

    def _calculate_max_drawdown(self, trades: List) -> float:
        """Calculate maximum drawdown from trades."""
        if not trades:
            return 0.0

        cumulative = 0
        peak = 0
        max_dd = 0

        for trade in sorted(trades, key=lambda t: t.timestamp):
            cumulative += trade.pnl_pct
            peak = max(peak, cumulative)
            drawdown = peak - cumulative
            max_dd = max(max_dd, drawdown)

        return max_dd
