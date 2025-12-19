"""Backtesting engine using BackTrader framework."""
from __future__ import annotations
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timezone
import pandas as pd
import numpy as np

try:
    import backtrader as bt
except ImportError:
    bt = None
    print("⚠️ BackTrader not installed. Backtesting will use simplified simulation.")

from .executor import OrderExecutor, OrderExecution
from ..models.types import ExecutionSummary, ResearchSummary
from ..agents.analyst import AnalystAgent
from ..agents.researcher import ResearcherAgent
from ..agents.trader import TraderAgent
from ..agents.risk import RiskManagerAgent


class BacktestEngine:
    """Backtesting engine for MAS trading system."""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission: float = 0.001,  # 0.1% commission
        use_backtrader: bool = True,
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.use_backtrader = use_backtrader and bt is not None
        self.executor = OrderExecutor(initial_capital=initial_capital)
        self.results: Dict[str, Any] = {}

    def run_backtest(
        self,
        price_df: pd.DataFrame,
        agents: Dict[str, Any],
        news_items_fn: Optional[Callable[[datetime, datetime], List[Dict[str, Any]]]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        validation_start: Optional[datetime] = None,
        validation_end: Optional[datetime] = None,
        test_start: Optional[datetime] = None,
        test_end: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Run backtest on historical data.

        Args:
            price_df: Historical price data with datetime index
            agents: Dict with 'analyst', 'researcher', 'trader', 'risk' agents
            news_items_fn: Function to get news items for a time period
            start_date: Start of backtest period
            end_date: End of backtest period
            validation_start: Start of validation period (e.g., August)
            validation_end: End of validation period
            test_start: Start of test period (e.g., September)
            test_end: End of test period
        """
        # Filter price data to backtest period
        if start_date:
            price_df = price_df[price_df.index >= start_date]
        if end_date:
            price_df = price_df[price_df.index <= end_date]

        analyst = agents.get("analyst")
        researcher = agents.get("researcher")
        trader = agents.get("trader")
        risk = agents.get("risk")

        if not all([analyst, researcher, trader, risk]):
            raise ValueError("Missing required agents: analyst, researcher, trader, risk")

        # Process each bar
        for i in range(len(price_df)):
            current_bar = price_df.iloc[i]
            current_time = price_df.index[i]

            # Get historical data up to current bar
            historical_df = price_df.iloc[:i+1]

            if len(historical_df) < 10:  # Need minimum bars for analysis
                continue

            # Process existing orders
            self.executor.process_bar(current_bar, current_time)

            # Run agent pipeline
            try:
                # Analyst
                features, trend = analyst.run(historical_df)

                # Researcher
                research = researcher.run(features, trend)

                # Get news for this period
                news_items = []
                if news_items_fn:
                    # Use trader's determined lookback
                    lookback_days = trader.determine_news_lookback_days(
                        research, historical_df, max_lookback_days=30
                    )
                    news_start = current_time - pd.Timedelta(days=lookback_days)
                    news_items = news_items_fn(news_start, current_time)

                # Trader
                exec_summary = trader.run(research, news_items, historical_df)

                # Risk check
                risk_review = risk.run(exec_summary, historical_df)

                # Submit order if approved
                if risk_review.approved:
                    self.executor.submit_order(exec_summary)
                elif risk_review.verdict == "soft_fail":
                    # Try regeneration once
                    exec_summary.position_size = min(
                        exec_summary.position_size,
                        risk_review.envelope.get("max_size", exec_summary.position_size)
                    )
                    exec_summary.leverage = min(
                        exec_summary.leverage,
                        risk_review.envelope.get("max_leverage", exec_summary.leverage)
                    )
                    risk_review2 = risk.run(exec_summary, historical_df, regen_attempted=True)
                    if risk_review2.approved:
                        self.executor.submit_order(exec_summary)

            except Exception as e:
                print(f"⚠️ Error processing bar at {current_time}: {e}")
                continue

        # Close any remaining open orders at end
        final_bar = price_df.iloc[-1]
        final_time = price_df.index[-1]
        for order_exec in list(self.executor.open_orders.values()):
            if order_exec.state.value == "filled":
                # Force close at final price
                self.executor.close_order(
                    order_exec, "end_of_backtest", final_bar, final_time
                )

        # Calculate metrics
        self.results = self._calculate_metrics(
            price_df, validation_start, validation_end, test_start, test_end
        )

        return self.results

    def _calculate_metrics(
        self,
        price_df: pd.DataFrame,
        validation_start: Optional[datetime] = None,
        validation_end: Optional[datetime] = None,
        test_start: Optional[datetime] = None,
        test_end: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Calculate backtest performance metrics."""
        closed_orders = self.executor.closed_orders

        if not closed_orders:
            return {
                "total_trades": 0,
                "final_capital": self.initial_capital,
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
            }

        # Overall metrics
        total_trades = len(closed_orders)
        final_capital = self.executor.current_capital
        total_return = (final_capital - self.initial_capital) / self.initial_capital

        # PnL statistics
        pnls = [order.pnl for order in closed_orders]
        pnl_pcts = [order.pnl_pct for order in closed_orders]

        winning_trades = [p for p in pnls if p > 0]
        losing_trades = [p for p in pnls if p < 0]

        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0
        avg_win = np.mean(winning_trades) if winning_trades else 0.0
        avg_loss = np.mean(losing_trades) if losing_trades else 0.0

        # Sharpe ratio (annualized)
        if len(pnl_pcts) > 1 and np.std(pnl_pcts) > 0:
            # Assuming 4h bars, ~6 bars per day, ~2190 bars per year
            periods_per_year = 2190
            sharpe_ratio = np.sqrt(periods_per_year) * np.mean(pnl_pcts) / np.std(pnl_pcts)
        else:
            sharpe_ratio = 0.0

        # Maximum drawdown
        capital_curve = [self.initial_capital]
        for order in closed_orders:
            capital_curve.append(capital_curve[-1] + order.pnl)

        if len(capital_curve) > 1:
            running_max = np.maximum.accumulate(capital_curve)
            drawdowns = (capital_curve - running_max) / running_max
            max_drawdown = abs(np.min(drawdowns))
        else:
            max_drawdown = 0.0

        # Period-specific metrics
        validation_metrics = None
        test_metrics = None

        if validation_start and validation_end:
            val_orders = [
                o for o in closed_orders
                if o.filled_time and validation_start <= o.filled_time <= validation_end
            ]
            if val_orders:
                validation_metrics = self._calculate_period_metrics(val_orders)

        if test_start and test_end:
            test_orders = [
                o for o in closed_orders
                if o.filled_time and test_start <= o.filled_time <= test_end
            ]
            if test_orders:
                test_metrics = self._calculate_period_metrics(test_orders)

        return {
            "total_trades": total_trades,
            "final_capital": final_capital,
            "total_return": total_return,
            "total_return_pct": total_return * 100,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "max_drawdown_pct": max_drawdown * 100,
            "win_rate": win_rate,
            "win_rate_pct": win_rate * 100,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": abs(sum(winning_trades) / sum(losing_trades)) if losing_trades else float('inf'),
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "closed_orders": [
                {
                    "order_id": o.order_id,
                    "filled_time": o.filled_time.isoformat() if o.filled_time else None,
                    "closed_time": o.closed_time.isoformat() if o.closed_time else None,
                    "pnl": o.pnl,
                    "pnl_pct": o.pnl_pct,
                    "close_reason": o.close_reason,
                }
                for o in closed_orders
            ],
        }

    def _calculate_period_metrics(self, orders: List[OrderExecution]) -> Dict[str, Any]:
        """Calculate metrics for a specific period."""
        if not orders:
            return {}

        pnls = [o.pnl for o in orders]
        pnl_pcts = [o.pnl_pct for o in orders]

        winning = [p for p in pnls if p > 0]
        losing = [p for p in pnls if p < 0]

        return {
            "trades": len(orders),
            "win_rate": len(winning) / len(orders) if orders else 0.0,
            "total_pnl": sum(pnls),
            "avg_pnl": np.mean(pnls),
            "sharpe_ratio": (
                np.sqrt(2190) * np.mean(pnl_pcts) / np.std(pnl_pcts)
                if len(pnl_pcts) > 1 and np.std(pnl_pcts) > 0 else 0.0
            ),
        }


def setup_validation_test_periods(
    year: int = 2024,
    validation_month: int = 8,  # August
    test_month: int = 9,  # September
) -> Dict[str, datetime]:
    """Set up validation and test periods."""
    validation_start = datetime(year, validation_month, 1, tzinfo=timezone.utc)
    # End of August
    if validation_month == 12:
        validation_end = datetime(year + 1, 1, 1, tzinfo=timezone.utc) - pd.Timedelta(seconds=1)
    else:
        validation_end = datetime(year, validation_month + 1, 1, tzinfo=timezone.utc) - pd.Timedelta(seconds=1)

    test_start = datetime(year, test_month, 1, tzinfo=timezone.utc)
    # End of September
    if test_month == 12:
        test_end = datetime(year + 1, 1, 1, tzinfo=timezone.utc) - pd.Timedelta(seconds=1)
    else:
        test_end = datetime(year, test_month + 1, 1, tzinfo=timezone.utc) - pd.Timedelta(seconds=1)

    return {
        "validation_start": validation_start,
        "validation_end": validation_end,
        "test_start": test_start,
        "test_end": test_end,
    }
