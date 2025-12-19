"""External services module."""
from .llm import generate_trading_proposal
from .metrics import PerformanceTracker

# Event system
from .events import EventBus, TradingEvent, EventTypes, get_event_bus

# Notifications and alerts
from .notifications import NotificationService, Notification, Report
from .alerts import AlertManager, Alert, AlertRule

# Reports
from .reports import ReportGenerator

# Paper trading (optional - requires aiohttp)
try:
    from .bybit_client import BybitTestnetClient, OrderResponse, Position, WalletBalance
    from .order_manager import OrderManager, ManagedOrder, OrderStatus
    from .positions import PositionTracker, TrackedPosition
    PAPER_TRADING_AVAILABLE = True
except ImportError:
    PAPER_TRADING_AVAILABLE = False

__all__ = [
    # LLM
    "generate_trading_proposal",
    # Metrics
    "PerformanceTracker",
    # Events
    "EventBus",
    "TradingEvent",
    "EventTypes",
    "get_event_bus",
    # Notifications
    "NotificationService",
    "Notification",
    "Report",
    # Alerts
    "AlertManager",
    "Alert",
    "AlertRule",
    # Reports
    "ReportGenerator",
    # Paper trading
    "PAPER_TRADING_AVAILABLE",
]

if PAPER_TRADING_AVAILABLE:
    __all__.extend([
        "BybitTestnetClient",
        "OrderResponse",
        "Position",
        "WalletBalance",
        "OrderManager",
        "ManagedOrder",
        "OrderStatus",
        "PositionTracker",
        "TrackedPosition",
    ])
