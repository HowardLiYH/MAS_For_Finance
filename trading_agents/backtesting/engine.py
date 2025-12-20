"""Backtesting engine using BackTrader framework.

Supports both single-agent and population-based (PopAgent) backtesting.
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional, Callable, TYPE_CHECKING
from datetime import datetime, timezone
from dataclasses import dataclass, field
import time
import pandas as pd
import numpy as np

try:
    import backtrader as bt
except ImportError:
    bt = None
    print("⚠️ BackTrader not installed. Backtesting will use simplified simulation.")

from .executor import OrderExecutor, OrderExecution
from ..models import ExecutionSummary, ResearchSummary

# Lazy imports to avoid circular dependencies
if TYPE_CHECKING:
    from ..population.selector_workflow import SelectorWorkflow
    from ..services.experiment_logger import ExperimentLogger


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


    def run_population_backtest(
        self,
        price_df: pd.DataFrame,
        selector_workflow: "SelectorWorkflow",
        news_items_fn: Optional[Callable[[datetime, datetime], List[Dict[str, Any]]]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        iteration_bar_count: int = 1,
        logger: Optional["ExperimentLogger"] = None,
    ) -> Dict[str, Any]:
        """
        Run backtest with PopAgent selector workflow.

        At each iteration (every N bars):
        1. All agents select methods from inventory
        2. Sample pipeline combinations
        3. Execute the BEST pipeline through OrderExecutor
        4. Update agent preferences based on actual PnL
        5. Transfer knowledge from best performers

        Args:
            price_df: Historical price data with datetime index
            selector_workflow: SelectorWorkflow instance with agent populations
            news_items_fn: Function to get news items for a time period
            start_date: Start of backtest period
            end_date: End of backtest period
            iteration_bar_count: Number of bars per iteration (default: 1)
            logger: Optional ExperimentLogger for detailed logging

        Returns:
            Dict with backtest results and learning progress
        """
        from ..population.selector_workflow import PipelineResult
        from ..services.experiment_logger import (
            AgentDecisionLog,
            PipelineResultLog,
            DiversityMetrics,
        )

        # Filter price data
        if start_date:
            price_df = price_df[price_df.index >= start_date]
        if end_date:
            price_df = price_df[price_df.index <= end_date]

        print(f"\n{'='*60}")
        print(f"🧬 POPULATION BACKTEST")
        print(f"{'='*60}")
        print(f"Period: {price_df.index[0]} → {price_df.index[-1]}")
        print(f"Bars: {len(price_df)}, Iteration every {iteration_bar_count} bars")
        print(f"Population size: {selector_workflow.config.population_size} per role")
        print(f"{'='*60}\n")

        # Track results
        all_iteration_results = []
        iteration_pnls = []

        # Process bars in iteration chunks
        bar_idx = 10  # Start after minimum lookback
        iteration_num = 0

        while bar_idx < len(price_df):
            iteration_start_time = time.time()
            iteration_num += 1

            # Get data for this iteration
            current_bar = price_df.iloc[bar_idx]
            current_time = price_df.index[bar_idx]
            historical_df = price_df.iloc[:bar_idx + 1]

            # Process existing orders
            self.executor.process_bar(current_bar, current_time)

            # Determine market context from price action
            market_context = self._derive_market_context(historical_df)

            # Get news if available
            news_digest = None
            if news_items_fn:
                lookback_start = current_time - pd.Timedelta(days=7)
                news_items = news_items_fn(lookback_start, current_time)
                news_digest = {"items": news_items, "count": len(news_items)}

            # ====== Run PopAgent Iteration ======

            # 1. Each agent selects methods
            agent_decisions = []
            for role, pop in selector_workflow.populations.items():
                for agent in pop.agents:
                    # Record pre-selection state
                    methods_available = list(agent.inventory)
                    old_prefs = dict(agent.preferences)

                    # Select methods
                    agent.select_methods(market_context)

                    # Log decision
                    if logger:
                        decision = AgentDecisionLog(
                            timestamp=current_time.isoformat(),
                            iteration=iteration_num,
                            agent_id=agent.id,
                            role=role.value,
                            methods_available=methods_available,
                            methods_selected=agent.current_selection,
                            selection_scores=getattr(agent, '_last_scores', {}),
                            preferences=old_prefs,
                            context=market_context,
                            reasoning=getattr(agent, '_last_reasoning', None),
                            exploration_used=getattr(agent, '_used_exploration', False),
                        )
                        agent_decisions.append(decision)

            # 2. Sample pipelines
            pipelines = selector_workflow._sample_pipelines()

            # 3. Evaluate each pipeline (simulated, but we'll execute best one for real)
            pipeline_results = []
            for pipeline in pipelines:
                result = selector_workflow._evaluate_pipeline(
                    pipeline=pipeline,
                    price_data=historical_df,
                    context=market_context,
                    news_digest=news_digest,
                )
                if result:
                    pipeline_results.append(result)

            # 4. Find best pipeline
            if pipeline_results:
                best_result = max(pipeline_results, key=lambda r: r.pnl)
                avg_pnl = np.mean([r.pnl for r in pipeline_results])

                # Execute best pipeline through OrderExecutor
                if best_result.success and best_result.pnl > 0:
                    # Create execution summary from best pipeline
                    exec_summary = self._create_execution_from_pipeline(
                        best_result, current_bar, current_time
                    )
                    if exec_summary:
                        self.executor.submit_order(exec_summary)

                iteration_pnls.append(best_result.pnl)
            else:
                best_result = None
                avg_pnl = 0.0
                iteration_pnls.append(0.0)

            # 5. Update preferences
            selector_workflow._update_preferences(pipeline_results, market_context)

            # 6. Score and transfer knowledge
            selector_workflow._score_and_transfer()

            # 7. Ensure diversity
            for pop in selector_workflow.populations.values():
                pop.ensure_diversity()

            # Calculate diversity metrics
            diversity_metrics = {}
            for role, pop in selector_workflow.populations.items():
                diversity_metrics[role.value] = DiversityMetrics(
                    role=role.value,
                    selection_diversity=pop.calculate_selection_diversity(),
                    preference_entropy=0.0,  # TODO: calculate
                    unique_methods_used=len(pop._get_method_usage()),
                    total_methods_available=len(pop.config.inventory),
                )

            # Log iteration
            iteration_duration_ms = (time.time() - iteration_start_time) * 1000

            if logger:
                pipeline_logs = [
                    PipelineResultLog(
                        pipeline_id=f"pipe_{i}",
                        agents={
                            "analyst": r.analyst_id,
                            "researcher": r.researcher_id,
                            "trader": r.trader_id,
                            "risk": r.risk_id,
                        },
                        methods={
                            "analyst": r.analyst_methods,
                            "researcher": r.researcher_methods,
                            "trader": r.trader_methods,
                            "risk": r.risk_methods,
                        },
                        pnl=r.pnl,
                        sharpe=r.sharpe,
                        success=r.success,
                    )
                    for i, r in enumerate(pipeline_results)
                ]

                iteration_log = logger.create_iteration_log(
                    iteration=iteration_num,
                    market_context=market_context,
                    agent_decisions=agent_decisions,
                    pipeline_results=pipeline_logs,
                    best_pipeline_id=f"pipe_best" if best_result else None,
                    best_pnl=best_result.pnl if best_result else 0.0,
                    avg_pnl=avg_pnl,
                    knowledge_transfer=None,  # TODO: log transfers
                    diversity_metrics=diversity_metrics,
                    iteration_duration_ms=iteration_duration_ms,
                )
                logger.log_iteration(iteration_log)

            # Store result
            all_iteration_results.append({
                "iteration": iteration_num,
                "bar_idx": bar_idx,
                "timestamp": current_time.isoformat(),
                "best_pnl": best_result.pnl if best_result else 0.0,
                "avg_pnl": avg_pnl,
                "pipelines_evaluated": len(pipeline_results),
                "market_context": market_context,
            })

            # Progress
            if iteration_num % 10 == 0:
                print(f"Iteration {iteration_num}: bar={bar_idx}/{len(price_df)}, "
                      f"best_pnl={best_result.pnl if best_result else 0:.4f}, "
                      f"capital={self.executor.current_capital:.2f}")

            # Move to next iteration
            bar_idx += iteration_bar_count

        # Close remaining orders
        final_bar = price_df.iloc[-1]
        final_time = price_df.index[-1]
        for order_exec in list(self.executor.open_orders.values()):
            if order_exec.state.value == "filled":
                self.executor.close_order(order_exec, "end_of_backtest", final_bar, final_time)

        # Calculate final metrics
        backtest_metrics = self._calculate_metrics(price_df)
        learning_progress = selector_workflow.get_learning_progress()

        # Finalize logger
        if logger:
            logger.finalize(selector_workflow.get_best_methods())

        print(f"\n{'='*60}")
        print(f"POPULATION BACKTEST COMPLETE")
        print(f"{'='*60}")
        print(f"Iterations: {iteration_num}")
        print(f"Final Capital: ${self.executor.current_capital:.2f}")
        print(f"Total Return: {backtest_metrics['total_return_pct']:.2f}%")
        print(f"Sharpe Ratio: {backtest_metrics['sharpe_ratio']:.2f}")
        print(f"Learning Improvement: {learning_progress.get('improvement', 0):.4f}")
        print(f"{'='*60}\n")

        return {
            "mode": "population",
            "iterations": iteration_num,
            "backtest_metrics": backtest_metrics,
            "learning_progress": learning_progress,
            "iteration_results": all_iteration_results,
            "best_methods": selector_workflow.get_best_methods(),
            "method_popularity": selector_workflow.get_method_popularity(),
        }

    def _derive_market_context(self, price_df: pd.DataFrame) -> Dict[str, Any]:
        """Derive market context from price action."""
        if len(price_df) < 20:
            return {"trend": "neutral", "volatility": 0.3, "regime": "normal"}

        closes = price_df["close"].values

        # Trend: compare recent to older prices
        recent_avg = np.mean(closes[-5:])
        older_avg = np.mean(closes[-20:-5])

        if recent_avg > older_avg * 1.02:
            trend = "bullish"
        elif recent_avg < older_avg * 0.98:
            trend = "bearish"
        else:
            trend = "neutral"

        # Volatility: standard deviation of returns
        returns = np.diff(closes) / closes[:-1]
        volatility = np.std(returns[-20:]) if len(returns) >= 20 else 0.02

        # Regime
        if volatility > 0.04:
            regime = "volatile"
        elif volatility < 0.01:
            regime = "quiet"
        else:
            regime = "normal"

        return {
            "trend": trend,
            "volatility": float(volatility),
            "regime": regime,
            "recent_return": float((closes[-1] / closes[-5] - 1)) if len(closes) >= 5 else 0.0,
        }

    def _create_execution_from_pipeline(
        self,
        pipeline_result: "PipelineResult",
        current_bar: pd.Series,
        current_time: datetime,
    ) -> Optional[ExecutionSummary]:
        """Create an ExecutionSummary from a pipeline result."""
        try:
            current_price = float(current_bar["close"])

            # Determine direction from PnL expectation
            direction = "LONG" if pipeline_result.pnl > 0 else "SHORT"

            # Set TP/SL based on volatility
            atr_approx = current_price * 0.02  # Approximate 2% ATR

            if direction == "LONG":
                take_profit = current_price + 2 * atr_approx
                stop_loss = current_price - 1 * atr_approx
            else:
                take_profit = current_price - 2 * atr_approx
                stop_loss = current_price + 1 * atr_approx

            return ExecutionSummary(
                order_id=f"pop_{current_time.strftime('%Y%m%d_%H%M')}_{pipeline_result.trader_id}",
                direction=direction,
                order_type="MARKET",
                entry_price=current_price,
                take_profit=take_profit,
                stop_loss=stop_loss,
                position_size=0.05,  # Conservative 5% position
                leverage=3.0,
                confidence=abs(pipeline_result.pnl),
                reasoning=f"PopAgent best pipeline: {pipeline_result.trader_methods}",
            )
        except Exception as e:
            print(f"⚠️ Error creating execution: {e}")
            return None


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
