"""Scheduler for Paper Trading Mode.

Implements 4-hour iteration scheduling for live paper trading
with Bybit testnet integration.
"""
from __future__ import annotations

import asyncio
import logging
import signal
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Optional, List

try:
    import schedule
except ImportError:
    schedule = None

from ..population import SelectorWorkflow, SelectorWorkflowConfig
from ..services.experiment_logger import ExperimentLogger

logger = logging.getLogger(__name__)


class PaperTradingScheduler:
    """
    Scheduler for running PopAgent iterations at 4-hour intervals.

    Runs at standard 4h bar close times:
    - 00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC

    Usage:
        scheduler = PaperTradingScheduler(
            symbols=["BTC", "ETH"],
            config_path="configs/multi_asset.yaml",
        )
        scheduler.start()  # Blocks until stopped
    """

    # Standard 4-hour bar close times (UTC)
    BAR_CLOSE_HOURS = [0, 4, 8, 12, 16, 20]

    def __init__(
        self,
        symbols: List[str],
        config_path: Optional[Path] = None,
        population_size: int = 5,
        max_methods: int = 3,
        log_dir: str = "logs/experiments",
        on_iteration_complete: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        """
        Initialize the paper trading scheduler.

        Args:
            symbols: List of trading symbols (e.g., ["BTC", "ETH"])
            config_path: Path to config file
            population_size: Number of agents per role
            max_methods: Maximum methods per agent
            log_dir: Directory for experiment logs
            on_iteration_complete: Callback after each iteration
        """
        self.symbols = symbols
        self.config_path = config_path
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.on_iteration_complete = on_iteration_complete
        self._running = False
        self._iteration_count = 0

        # Create workflow config
        self.workflow_config = SelectorWorkflowConfig(
            population_size=population_size,
            max_methods_per_agent=max_methods,
        )

        # Initialize workflow (will be created per symbol)
        self.workflows: Dict[str, SelectorWorkflow] = {}
        self.loggers: Dict[str, ExperimentLogger] = {}

        # Initialize for each symbol
        for symbol in symbols:
            self.workflows[symbol] = SelectorWorkflow(self.workflow_config)
            self.loggers[symbol] = ExperimentLogger(
                experiment_id=f"paper_{symbol}_{datetime.now().strftime('%Y%m%d')}",
                log_dir=str(self.log_dir),
                config={
                    "mode": "paper_trading",
                    "symbol": symbol,
                    "population_size": population_size,
                    "max_methods": max_methods,
                },
            )

    def _get_next_run_time(self) -> datetime:
        """Calculate the next scheduled run time."""
        now = datetime.now(tz=timezone.utc)

        # Find next bar close time
        for hour in self.BAR_CLOSE_HOURS:
            target = now.replace(hour=hour, minute=0, second=0, microsecond=0)
            if target > now:
                return target

        # Next day's first bar
        next_day = now + timedelta(days=1)
        return next_day.replace(hour=0, minute=0, second=0, microsecond=0)

    def _time_until_next_run(self) -> float:
        """Get seconds until next scheduled run."""
        next_run = self._get_next_run_time()
        now = datetime.now(tz=timezone.utc)
        delta = next_run - now
        return max(0, delta.total_seconds())

    async def _fetch_price_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch latest price data from Bybit."""
        try:
            from ..services.bybit_client import BybitClient

            client = BybitClient(testnet=True)

            # Get recent 4h bars
            klines = client.get_klines(
                symbol=f"{symbol}USDT",
                interval="240",  # 4 hours in minutes
                limit=100,
            )

            return {
                "symbol": symbol,
                "klines": klines,
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.error(f"Error fetching price data for {symbol}: {e}")
            return {"symbol": symbol, "error": str(e)}

    async def _fetch_news(self) -> List[Dict[str, Any]]:
        """Fetch recent crypto news."""
        try:
            from ..services.bocha_client import BochaClient

            client = BochaClient()
            news = client.search_news(
                query="cryptocurrency trading",
                freshness="day",
                count=10,
            )
            return news
        except Exception as e:
            logger.warning(f"Error fetching news: {e}")
            return []

    async def _run_iteration_async(self) -> Dict[str, Any]:
        """Run one iteration across all symbols."""
        self._iteration_count += 1
        iteration_start = datetime.now(tz=timezone.utc)

        logger.info(f"\n{'='*60}")
        logger.info(f"📊 PAPER TRADING ITERATION {self._iteration_count}")
        logger.info(f"Time: {iteration_start.isoformat()}")
        logger.info(f"Symbols: {self.symbols}")
        logger.info(f"{'='*60}\n")

        results = {}

        # Fetch news once for all symbols
        news_items = await self._fetch_news()
        news_digest = {"items": news_items, "count": len(news_items)}

        for symbol in self.symbols:
            logger.info(f"\n--- Processing {symbol} ---")

            # Fetch price data
            price_data = await self._fetch_price_data(symbol)

            if "error" in price_data:
                logger.error(f"Skipping {symbol} due to data fetch error")
                results[symbol] = {"error": price_data["error"]}
                continue

            try:
                # Run PopAgent iteration
                import pandas as pd

                klines = price_data.get("klines", [])
                if not klines:
                    logger.warning(f"No kline data for {symbol}")
                    continue

                # Convert to DataFrame
                df = pd.DataFrame(klines)
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                    df.set_index("timestamp", inplace=True)

                # Derive market context
                market_context = self._derive_context(df)

                # Run workflow iteration
                workflow = self.workflows[symbol]
                summary = workflow.run_iteration(
                    price_data=df,
                    market_context=market_context,
                    news_digest=news_digest,
                )

                results[symbol] = {
                    "iteration": summary.iteration,
                    "best_pnl": summary.best_pnl,
                    "avg_pnl": summary.avg_pnl,
                    "best_methods": summary.best_methods,
                    "transfer_performed": summary.transfer_performed,
                }

                logger.info(f"{symbol}: Best PnL = {summary.best_pnl:.4f}, "
                           f"Methods = {summary.best_methods}")

            except Exception as e:
                logger.error(f"Error running iteration for {symbol}: {e}")
                results[symbol] = {"error": str(e)}

        iteration_result = {
            "iteration": self._iteration_count,
            "timestamp": iteration_start.isoformat(),
            "results": results,
            "duration_seconds": (datetime.now(tz=timezone.utc) - iteration_start).total_seconds(),
        }

        # Callback if provided
        if self.on_iteration_complete:
            try:
                self.on_iteration_complete(iteration_result)
            except Exception as e:
                logger.error(f"Error in iteration callback: {e}")

        return iteration_result

    def _derive_context(self, df) -> Dict[str, Any]:
        """Derive market context from price data."""
        import numpy as np

        if len(df) < 20:
            return {"trend": "neutral", "volatility": 0.3, "regime": "normal"}

        closes = df["close"].values if "close" in df.columns else df.iloc[:, 3].values
        closes = [float(c) for c in closes]

        # Trend
        recent_avg = np.mean(closes[-5:])
        older_avg = np.mean(closes[-20:-5])

        if recent_avg > older_avg * 1.02:
            trend = "bullish"
        elif recent_avg < older_avg * 0.98:
            trend = "bearish"
        else:
            trend = "neutral"

        # Volatility
        returns = np.diff(closes) / np.array(closes[:-1])
        volatility = float(np.std(returns[-20:])) if len(returns) >= 20 else 0.02

        # Regime
        if volatility > 0.04:
            regime = "volatile"
        elif volatility < 0.01:
            regime = "quiet"
        else:
            regime = "normal"

        return {
            "trend": trend,
            "volatility": volatility,
            "regime": regime,
            "recent_return": float(closes[-1] / closes[-5] - 1) if len(closes) >= 5 else 0.0,
        }

    def _run_iteration_sync(self):
        """Synchronous wrapper for iteration."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._run_iteration_async())
        finally:
            loop.close()

    def run_once(self) -> Dict[str, Any]:
        """Run a single iteration immediately."""
        return self._run_iteration_sync()

    def start(self):
        """Start the scheduler (blocking)."""
        if schedule is None:
            raise ImportError("schedule package not installed. Run: pip install schedule")

        self._running = True

        # Set up signal handlers
        def signal_handler(signum, frame):
            logger.info("\n🛑 Received shutdown signal. Stopping scheduler...")
            self.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        logger.info(f"\n{'='*60}")
        logger.info("🚀 PAPER TRADING SCHEDULER STARTED")
        logger.info(f"{'='*60}")
        logger.info(f"Symbols: {self.symbols}")
        logger.info(f"Population size: {self.workflow_config.population_size}")
        logger.info(f"Schedule: Every 4 hours (00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC)")

        next_run = self._get_next_run_time()
        logger.info(f"Next run: {next_run.isoformat()}")
        logger.info(f"{'='*60}\n")

        # Schedule at each 4-hour bar close
        for hour in self.BAR_CLOSE_HOURS:
            time_str = f"{hour:02d}:00"
            schedule.every().day.at(time_str).do(self._run_iteration_sync)

        # Run immediately if close to a bar time
        seconds_to_next = self._time_until_next_run()
        if seconds_to_next < 60:  # Within 1 minute of bar close
            logger.info("Running immediate iteration (near bar close time)")
            self._run_iteration_sync()

        # Main loop
        while self._running:
            schedule.run_pending()

            # Sleep until close to next run
            seconds_to_next = self._time_until_next_run()
            sleep_time = min(60, seconds_to_next)  # Check at most every minute

            if seconds_to_next > 60:
                next_run = self._get_next_run_time()
                logger.debug(f"Next iteration at {next_run.isoformat()} ({seconds_to_next/60:.1f} min)")

            asyncio.get_event_loop().run_until_complete(asyncio.sleep(sleep_time))

    def stop(self):
        """Stop the scheduler."""
        self._running = False

        # Finalize loggers
        for symbol, log in self.loggers.items():
            try:
                workflow = self.workflows[symbol]
                log.finalize(workflow.get_best_methods())
                logger.info(f"Finalized logs for {symbol}")
            except Exception as e:
                logger.error(f"Error finalizing logs for {symbol}: {e}")


async def run_paper_trading_async(
    symbols: List[str],
    iterations: int = 1,
    population_size: int = 5,
) -> Dict[str, Any]:
    """
    Run paper trading iterations asynchronously.

    Args:
        symbols: Trading symbols
        iterations: Number of iterations to run
        population_size: Agents per role

    Returns:
        Results from all iterations
    """
    scheduler = PaperTradingScheduler(
        symbols=symbols,
        population_size=population_size,
    )

    results = []
    for i in range(iterations):
        result = await scheduler._run_iteration_async()
        results.append(result)

    return {"iterations": results}


def run_paper_trading(
    symbols: List[str],
    continuous: bool = False,
    population_size: int = 5,
) -> Dict[str, Any]:
    """
    Run paper trading.

    Args:
        symbols: Trading symbols
        continuous: If True, run on 4h schedule; if False, run once
        population_size: Agents per role

    Returns:
        Results from iteration(s)
    """
    scheduler = PaperTradingScheduler(
        symbols=symbols,
        population_size=population_size,
    )

    if continuous:
        scheduler.start()  # Blocks
        return {}
    else:
        return scheduler.run_once()
