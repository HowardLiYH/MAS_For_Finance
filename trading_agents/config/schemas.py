"""Configuration schema definitions."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List


# Default symbols for multi-asset mode
DEFAULT_SYMBOLS = ["BTC", "ETH", "SOL", "DOGE", "XRP"]


@dataclass
class DataConfig:
    """Configuration for data ingestion."""
    exchange_id: str = "kraken"
    timeframe: str = "4h"
    max_days: int = 730  # 2 years
    out_dir: str = "data"
    offline_prices_csv: Optional[str] = None
    offline_news_jsonl: Optional[str] = None

    # Multi-asset settings
    multi_asset: bool = False
    symbols: List[str] = field(default_factory=lambda: DEFAULT_SYMBOLS.copy())
    bybit_csv_dir: Optional[str] = None
    add_cross_features: bool = True


@dataclass
class NewsConfig:
    """Configuration for news data."""
    use_llm_news: bool = True
    news_query: str = "bitcoin OR BTC"
    search_provider: str = "serpapi"
    llm_model: str = "gpt-4o-mini"
    max_news_per_stream: int = 10
    max_search_results_per_stream: int = 50
    require_published_signal: bool = True
    max_lookback_days: int = 30
    default_lookback_days: int = 14


@dataclass
class LearningConfig:
    """Configuration for continual learning and optimization."""
    knowledge_transfer_frequency: int = 10  # K rounds
    trader_transfer_frequency: int = 20     # 2K for traders
    pruning_frequency: int = 50             # M rounds
    transfer_ratio: float = 0.5             # Transfer to bottom 50%
    min_usage_threshold: int = 5            # Minimum uses before pruning
    min_success_rate: float = 0.3           # Minimum success rate


@dataclass
class AdminConfigSchema:
    """Configuration for Admin Agent."""
    enabled: bool = True
    max_drawdown_pct: float = 10.0
    daily_loss_limit_pct: float = 5.0
    risk_breach_threshold: int = 3
    daily_summary_enabled: bool = True
    daily_summary_hour: int = 0  # UTC hour
    weekly_summary_enabled: bool = True
    weekly_summary_day: str = "sunday"
    slack_webhook: Optional[str] = None
    email: Optional[str] = None
    log_dir: str = "logs/admin"


@dataclass
class PaperTradingConfig:
    """Configuration for paper trading."""
    enabled: bool = False
    exchange: str = "bybit_testnet"
    api_key: Optional[str] = None  # Can use env var BYBIT_TESTNET_KEY
    api_secret: Optional[str] = None  # Can use env var BYBIT_TESTNET_SECRET
    max_position_usd: float = 10000.0
    default_leverage: int = 5
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT", "SOLUSDT"])


@dataclass
class AppConfig:
    """Main application configuration."""
    symbol: str = "BTCUSD.PERP"
    timeframe: str = "4h"
    data: DataConfig = field(default_factory=DataConfig)
    news: NewsConfig = field(default_factory=NewsConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    admin: AdminConfigSchema = field(default_factory=AdminConfigSchema)
    paper_trading: PaperTradingConfig = field(default_factory=PaperTradingConfig)

    @property
    def is_multi_asset(self) -> bool:
        """Check if multi-asset mode is enabled."""
        return self.data.multi_asset

    @property
    def active_symbols(self) -> List[str]:
        """Get list of active symbols."""
        if self.data.multi_asset:
            return self.data.symbols
        # Extract symbol from single-asset format (e.g., "BTCUSD.PERP" -> "BTC")
        base = self.symbol.upper().replace("USD", "").replace(".PERP", "").replace("USDT", "")
        return [base]

    @property
    def admin_max_drawdown_pct(self) -> float:
        """Get admin max drawdown threshold."""
        return self.admin.max_drawdown_pct

    @property
    def admin_daily_loss_pct(self) -> float:
        """Get admin daily loss limit."""
        return self.admin.daily_loss_limit_pct


@dataclass
class OrchestratorInput:
    """Input for workflow orchestration."""
    symbol: str
    interval: str
    config: AppConfig

    # Optional overrides
    start: Optional[str] = None
    end: Optional[str] = None
    max_news: Optional[int] = None

    # Multi-asset overrides
    symbols: Optional[List[str]] = None

    @property
    def is_multi_asset(self) -> bool:
        """Check if multi-asset mode is enabled."""
        return self.config.is_multi_asset or (self.symbols is not None and len(self.symbols) > 1)

    @property
    def active_symbols(self) -> List[str]:
        """Get list of active symbols."""
        if self.symbols:
            return self.symbols
        return self.config.active_symbols
