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
            TransferLog,
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

            # Capture knowledge transfers that just happened
            transfer_log = None
            for role, pop in selector_workflow.populations.items():
                # Check if transfer just happened (iteration is now divisible by frequency)
                if pop.iteration > 0 and pop.iteration % pop.config.transfer_frequency == 0:
                    best = pop.get_best()
                    if best:
                        transfer_log = TransferLog(
                            timestamp=current_time.isoformat(),
                            role=role.value,
                            source_agent_id=best.id,
                            target_agent_ids=[a.id for a in pop.agents if a.id != best.id],
                            transfer_tau=pop.config.transfer_tau,
                            methods_transferred=best.current_selection,
                            source_preferences=dict(best.preferences),
                        )
                        break  # Log first transfer found

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
                    knowledge_transfer=transfer_log,
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


    def run_multi_asset_backtest(
        self,
        price_data: Dict[str, pd.DataFrame],
        selector_workflow: "SelectorWorkflow",
        news_items_fn: Optional[Callable[[datetime, datetime], List[Dict[str, Any]]]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        iteration_bar_count: int = 1,
        logger: Optional["ExperimentLogger"] = None,
    ) -> Dict[str, Any]:
        """
        Run multi-asset backtest with PopAgent selector workflow.

        Includes cross-asset features:
        - Correlation analysis between assets
        - BTC dominance tracking
        - Portfolio-level metrics
        - Cross-asset signal generation

        Args:
            price_data: Dict mapping symbol -> price DataFrame
            selector_workflow: SelectorWorkflow instance with agent populations
            news_items_fn: Function to get news items for a time period
            start_date: Start of backtest period
            end_date: End of backtest period
            iteration_bar_count: Number of bars per iteration (default: 1)
            logger: Optional ExperimentLogger for detailed logging

        Returns:
            Dict with multi-asset backtest results and cross-asset metrics
        """
        from ..population.selector_workflow import PipelineResult
        from ..services.experiment_logger import (
            AgentDecisionLog,
            PipelineResultLog,
            DiversityMetrics,
        )

        symbols = list(price_data.keys())
        print(f"\n{'='*60}")
        print(f"🌐 MULTI-ASSET POPULATION BACKTEST")
        print(f"{'='*60}")
        print(f"Assets: {', '.join(symbols)}")

        # Align all price data to common index
        aligned_data = self._align_multi_asset_data(price_data, start_date, end_date)
        common_index = aligned_data[symbols[0]].index

        print(f"Period: {common_index[0]} → {common_index[-1]}")
        print(f"Bars: {len(common_index)}, Iteration every {iteration_bar_count} bars")
        print(f"Population size: {selector_workflow.config.population_size} per role")
        print(f"{'='*60}\n")

        # Track per-asset executors and results
        asset_executors = {sym: OrderExecutor(initial_capital=self.initial_capital / len(symbols))
                          for sym in symbols}
        all_iteration_results = []
        iteration_pnls = []

        # Track stay-flat decisions (v0.9.1)
        stayed_flat_count = 0
        traded_count = 0

        # Cross-asset tracking
        correlation_history = []
        btc_dominance_history = []

        # Process bars in iteration chunks
        bar_idx = 20  # Start after minimum lookback for cross-asset calculations
        iteration_num = 0

        while bar_idx < len(common_index):
            iteration_start_time = time.time()
            iteration_num += 1

            current_time = common_index[bar_idx]

            # Get historical data for all assets
            historical_data = {sym: aligned_data[sym].iloc[:bar_idx + 1] for sym in symbols}
            current_bars = {sym: aligned_data[sym].iloc[bar_idx] for sym in symbols}

            # Process existing orders for each asset
            for sym in symbols:
                asset_executors[sym].process_bar(current_bars[sym], current_time)

            # ====== Calculate Cross-Asset Features ======
            cross_asset_context = self._calculate_cross_asset_features(
                historical_data, symbols, current_time
            )

            # Derive per-asset market context
            per_asset_context = {}
            for sym in symbols:
                ctx = self._derive_market_context(historical_data[sym])
                ctx["symbol"] = sym
                per_asset_context[sym] = ctx

            # Combine into unified market context
            market_context = {
                "multi_asset": True,
                "symbols": symbols,
                "per_asset": per_asset_context,
                "cross_asset": cross_asset_context,
                "timestamp": current_time.isoformat(),
            }

            # Track cross-asset metrics
            correlation_history.append(cross_asset_context.get("avg_correlation", 0))
            btc_dominance_history.append(cross_asset_context.get("btc_dominance", 0))

            # Get news if available
            news_digest = None
            if news_items_fn:
                lookback_start = current_time - pd.Timedelta(days=7)
                news_items = news_items_fn(lookback_start, current_time)
                news_digest = {"items": news_items, "count": len(news_items)}

            # ====== MODEL PREDICTION (BEFORE trading decision) ======
            # This MUST happen before we decide whether to trade
            # v0.9.8: Feature-aligned learning (RECOMMENDED)
            # v0.9.7: Hybrid learning (deprecated)
            # v0.9.0: Online-only (legacy)
            online_signal = "hold"
            online_confidence = 0.5
            online_details = {}

            # Create combined dataframe for feature extraction
            combined_df = self._create_combined_dataframe(historical_data)
            features = selector_workflow._extract_features(combined_df, market_context)
            selector_workflow.last_features = features

            # Use feature-aligned learner (v0.9.8) - RECOMMENDED
            # Update frequency matches FEATURE TIMESCALE, not model complexity!
            if hasattr(selector_workflow, 'feature_aligned_learner') and selector_workflow.feature_aligned_learner:
                online_signal, online_confidence, online_details = selector_workflow.feature_aligned_learner.predict(features)
                online_details["learning_mode"] = "feature_aligned"

            # Fall back to online-only (legacy)
            elif hasattr(selector_workflow, 'online_models') and selector_workflow.online_models:
                online_signal, online_confidence, online_details = selector_workflow.online_models.get_combined_signal(features)
                online_details["learning_mode"] = "online"

            # Add to market context so agents and logging can see it
            market_context["online_signal"] = online_signal
            market_context["online_confidence"] = online_confidence
            market_context["online_regime"] = online_details.get("regime", "Neutral")
            market_context["online_momentum"] = online_details.get("momentum", 0)
            market_context["online_adjusted_signal"] = online_details.get("adjusted_signal", 0)
            market_context["learning_mode"] = online_details.get("learning_mode", "none")

            # Feature-aligned specific: add per-group predictions
            if online_details.get("learning_mode") == "feature_aligned":
                market_context["fast_pred"] = online_details.get("fast_pred", 0)
                market_context["medium_pred"] = online_details.get("medium_pred", 0)
                market_context["slow_pred"] = online_details.get("slow_pred", 0)

            # ====== Run PopAgent Iteration ======
            agent_decisions = []
            for role, pop in selector_workflow.populations.items():
                for agent in pop.agents:
                    methods_available = list(agent.inventory)
                    old_prefs = dict(agent.preferences)
                    agent.select_methods(market_context)

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

            # Sample and evaluate pipelines
            pipelines = selector_workflow._sample_pipelines()
            pipeline_results = []

            for pipeline in pipelines:
                # For multi-asset, evaluate on primary asset (BTC) or aggregate
                primary_asset = "BTC" if "BTC" in symbols else symbols[0]
                result = selector_workflow._evaluate_pipeline(
                    pipeline=pipeline,
                    price_data=historical_data[primary_asset],
                    context=market_context,
                    news_digest=news_digest,
                )
                if result:
                    pipeline_results.append(result)

            # Find best pipeline and execute across assets
            iteration_pnl = 0.0
            if pipeline_results:
                best_result = max(pipeline_results, key=lambda r: r.pnl)
                avg_pnl = np.mean([r.pnl for r in pipeline_results])

                # ====== SIMPLIFIED TRADING LOGIC (v0.9.4) ======
                # The ONLINE MODEL decides whether to trade (signal) - computed above
                # The TRADER METHODS decide HOW to trade (execution style)
                #
                # Key insight: online_signal and online_confidence were computed
                # BEFORE this block, so we use the values directly (not from context)

                # Simple decision: trade if signal is not "hold"
                should_stay_flat = (online_signal == "hold")

                # Scale position size by confidence (gradual scaling)
                # Full size at confidence >= 0.6, reduced below that
                confidence_multiplier = max(0.3, min(1.0, 0.4 + online_confidence))

                # Log trading decision for debugging
                if iteration_num <= 5 or iteration_num % 100 == 0:
                    print(f"  [Iter {iteration_num}] signal={online_signal}, conf={online_confidence:.2f}, "
                          f"adj_signal={online_details.get('adjusted_signal', 0):.2f}, stay_flat={should_stay_flat}")

                if best_result.success and not should_stay_flat:
                    traded_count += 1
                    for sym in symbols:
                        # Adjust position based on correlation and per-asset context
                        asset_weight = self._calculate_asset_weight(
                            sym, cross_asset_context, per_asset_context[sym]
                        )

                        # Apply confidence multiplier to position size
                        adjusted_weight = asset_weight * confidence_multiplier

                        if adjusted_weight > 0.05:  # Only trade if weight is significant
                            exec_summary = self._create_multi_asset_execution(
                                best_result, current_bars[sym], current_time, sym, adjusted_weight
                            )
                            if exec_summary:
                                asset_executors[sym].submit_order(exec_summary)
                else:
                    # Stayed flat - skipped trading
                    stayed_flat_count += 1

                # Sum PnL across assets
                for sym in symbols:
                    iteration_pnl += asset_executors[sym].current_capital - (self.initial_capital / len(symbols))

                iteration_pnls.append(best_result.pnl)
            else:
                best_result = None
                avg_pnl = 0.0
                iteration_pnls.append(0.0)

            # ====== MODEL UPDATE: Update models with observed return ======
            # Note: Prediction was already done at the start of this iteration
            # Here we UPDATE models with the observed outcome (learning happens here!)
            if bar_idx > 21 and selector_workflow.last_features is not None:
                # Calculate actual return from this bar
                primary_sym = "BTC" if "BTC" in symbols else symbols[0]
                prev_close = aligned_data[primary_sym].iloc[bar_idx - 1]["close"]
                curr_close = current_bars[primary_sym]["close"]
                actual_return = (curr_close / prev_close) - 1.0

                # Update feature-aligned learner (v0.9.8) - RECOMMENDED
                # Each feature group updates at its natural timescale!
                if hasattr(selector_workflow, 'feature_aligned_learner') and selector_workflow.feature_aligned_learner:
                    selector_workflow.feature_aligned_learner.update(
                        selector_workflow.last_features,
                        actual_return
                    )
                # Fall back to online-only (legacy)
                elif hasattr(selector_workflow, 'online_models') and selector_workflow.online_models:
                    selector_workflow.online_models.update_all(
                        selector_workflow.last_features,
                        actual_return
                    )

            # Update preferences and transfer knowledge
            selector_workflow._update_preferences(pipeline_results, market_context)
            selector_workflow._score_and_transfer()

            # Capture knowledge transfers that just happened
            transfer_log = None
            for role, pop in selector_workflow.populations.items():
                # Check if transfer just happened (iteration is now divisible by frequency)
                if pop.iteration > 0 and pop.iteration % pop.config.transfer_frequency == 0:
                    best = pop.get_best()
                    if best:
                        from ..services.experiment_logger import TransferLog
                        transfer_log = TransferLog(
                            timestamp=current_time.isoformat(),
                            role=role.value,
                            source_agent_id=best.id,
                            target_agent_ids=[a.id for a in pop.agents if a.id != best.id],
                            transfer_tau=pop.config.transfer_tau,
                            methods_transferred=best.current_selection,
                            source_preferences=dict(best.preferences),
                        )
                        break  # Log first transfer found

            for pop in selector_workflow.populations.values():
                pop.ensure_diversity()

            # Calculate diversity metrics
            diversity_metrics = {}
            for role, pop in selector_workflow.populations.items():
                diversity_metrics[role.value] = DiversityMetrics(
                    role=role.value,
                    selection_diversity=pop.calculate_selection_diversity(),
                    preference_entropy=0.0,
                    unique_methods_used=len(pop._get_method_usage()),
                    total_methods_available=len(pop.config.inventory),
                )

            iteration_duration_ms = (time.time() - iteration_start_time) * 1000

            # ====== Log iteration to ExperimentLogger ======
            if logger:
                from ..services.experiment_logger import PipelineResultLog

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
                    knowledge_transfer=transfer_log,
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
                "cross_asset_context": cross_asset_context,
            })

            # Progress
            if iteration_num % 10 == 0:
                total_capital = sum(ex.current_capital for ex in asset_executors.values())
                print(f"Iteration {iteration_num}: bar={bar_idx}/{len(common_index)}, "
                      f"best_pnl={best_result.pnl if best_result else 0:.4f}, "
                      f"total_capital={total_capital:.2f}")

            bar_idx += iteration_bar_count

        # Close remaining orders for all assets
        final_time = common_index[-1]
        for sym in symbols:
            final_bar = aligned_data[sym].iloc[-1]
            for order_exec in list(asset_executors[sym].open_orders.values()):
                if order_exec.state.value == "filled":
                    asset_executors[sym].close_order(order_exec, "end_of_backtest", final_bar, final_time)

        # Calculate aggregate metrics
        total_final_capital = sum(ex.current_capital for ex in asset_executors.values())
        total_initial_capital = self.initial_capital

        # Collect all closed orders
        all_closed_orders = []
        for sym, executor in asset_executors.items():
            for order in executor.closed_orders:
                order.symbol = sym  # Tag with symbol
                all_closed_orders.append(order)

        # Calculate per-asset returns
        per_asset_return = {
            sym: (asset_executors[sym].current_capital - (total_initial_capital / len(symbols)))
                 / (total_initial_capital / len(symbols))
            for sym in symbols
        }

        # Cross-asset metrics
        cross_asset_metrics = {
            "avg_correlation": np.mean(correlation_history) if correlation_history else 0,
            "btc_dominance": np.mean(btc_dominance_history) if btc_dominance_history else 0,
            "portfolio_volatility": np.std(iteration_pnls) if iteration_pnls else 0,
            "per_asset_return": per_asset_return,
            "correlation_trend": self._calculate_trend(correlation_history),
        }

        # Combine into backtest metrics
        total_trades = len(all_closed_orders)
        total_return = (total_final_capital - total_initial_capital) / total_initial_capital

        pnls = [o.pnl for o in all_closed_orders]
        winning_trades = [p for p in pnls if p > 0]
        losing_trades = [p for p in pnls if p < 0]

        # Calculate stay-flat rate
        total_decisions = stayed_flat_count + traded_count
        stay_flat_rate = stayed_flat_count / total_decisions if total_decisions > 0 else 0

        backtest_metrics = {
            "total_trades": total_trades,
            "final_capital": total_final_capital,
            "total_return": total_return,
            "total_return_pct": total_return * 100,
            "sharpe_ratio": self._calculate_sharpe(iteration_pnls),
            "max_drawdown": self._calculate_max_drawdown(iteration_pnls, total_initial_capital),
            "max_drawdown_pct": self._calculate_max_drawdown(iteration_pnls, total_initial_capital) * 100,
            "win_rate": len(winning_trades) / total_trades if total_trades > 0 else 0,
            "win_rate_pct": (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0,
            "avg_win": np.mean(winning_trades) if winning_trades else 0,
            "avg_loss": np.mean(losing_trades) if losing_trades else 0,
            "profit_factor": abs(sum(winning_trades) / sum(losing_trades)) if losing_trades else float('inf'),
            # Stay flat metrics (v0.9.1)
            "stayed_flat_count": stayed_flat_count,
            "traded_count": traded_count,
            "stay_flat_rate_pct": stay_flat_rate * 100,
        }

        learning_progress = selector_workflow.get_learning_progress()

        if logger:
            logger.finalize(selector_workflow.get_best_methods())

        print(f"\n{'='*60}")
        print(f"MULTI-ASSET POPULATION BACKTEST COMPLETE")
        print(f"{'='*60}")
        print(f"Iterations: {iteration_num}")
        print(f"Assets: {', '.join(symbols)}")
        print(f"Final Capital: ${total_final_capital:.2f}")
        print(f"Total Return: {backtest_metrics['total_return_pct']:.2f}%")
        print(f"Sharpe Ratio: {backtest_metrics['sharpe_ratio']:.2f}")
        print(f"Avg Correlation: {cross_asset_metrics['avg_correlation']:.3f}")
        print(f"--- Stay Flat Stats (v0.9.1) ---")
        print(f"Stayed Flat: {stayed_flat_count} iterations ({backtest_metrics['stay_flat_rate_pct']:.1f}%)")
        print(f"Traded: {traded_count} iterations ({100 - backtest_metrics['stay_flat_rate_pct']:.1f}%)")
        print(f"{'='*60}\n")

        return {
            "mode": "multi_asset_population",
            "symbols": symbols,
            "iterations": iteration_num,
            "backtest_metrics": backtest_metrics,
            "cross_asset_metrics": cross_asset_metrics,
            "learning_progress": learning_progress,
            "iteration_results": all_iteration_results,
            "best_methods": selector_workflow.get_best_methods(),
            "method_popularity": selector_workflow.get_method_popularity(),
        }

    def _align_multi_asset_data(
        self,
        price_data: Dict[str, pd.DataFrame],
        start_date: Optional[datetime],
        end_date: Optional[datetime],
    ) -> Dict[str, pd.DataFrame]:
        """Align all asset DataFrames to a common index."""
        aligned = {}

        # Find common date range
        all_indices = [df.index for df in price_data.values()]
        common_start = max(idx.min() for idx in all_indices)
        common_end = min(idx.max() for idx in all_indices)

        if start_date and start_date > common_start:
            common_start = start_date
        if end_date and end_date < common_end:
            common_end = end_date

        for sym, df in price_data.items():
            mask = (df.index >= common_start) & (df.index <= common_end)
            aligned[sym] = df.loc[mask].copy()

        return aligned

    def _create_combined_dataframe(
        self,
        historical_data: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Create combined DataFrame from multi-asset historical data.

        Used for feature extraction in online learning.
        """
        if not historical_data:
            return pd.DataFrame()

        # Use first symbol as base
        first_sym = list(historical_data.keys())[0]
        combined = historical_data[first_sym].copy()

        # Add columns from other symbols
        for sym, df in historical_data.items():
            if sym != first_sym:
                for col in ["open", "high", "low", "close", "volume"]:
                    if col in df.columns:
                        combined[f"{sym}_{col}"] = df[col]

        return combined

    def _calculate_cross_asset_features(
        self,
        historical_data: Dict[str, pd.DataFrame],
        symbols: List[str],
        current_time: datetime,
    ) -> Dict[str, Any]:
        """Calculate cross-asset features for market context."""
        features = {}

        # Get returns for correlation
        returns = {}
        for sym in symbols:
            df = historical_data[sym]
            if len(df) > 1:
                returns[sym] = df["close"].pct_change().dropna().values[-20:]  # Last 20 bars

        # Calculate pairwise correlations
        correlations = []
        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i+1:]:
                if sym1 in returns and sym2 in returns:
                    min_len = min(len(returns[sym1]), len(returns[sym2]))
                    if min_len > 5:
                        corr = np.corrcoef(returns[sym1][-min_len:], returns[sym2][-min_len:])[0, 1]
                        if not np.isnan(corr):
                            correlations.append(corr)
                            features[f"corr_{sym1}_{sym2}"] = corr

        features["avg_correlation"] = np.mean(correlations) if correlations else 0

        # BTC dominance (if BTC is in the mix)
        if "BTC" in symbols:
            btc_df = historical_data["BTC"]
            total_volume = sum(historical_data[sym]["volume"].iloc[-1] for sym in symbols)
            btc_volume = btc_df["volume"].iloc[-1]
            features["btc_dominance"] = btc_volume / total_volume if total_volume > 0 else 0.5

            # BTC trend influence
            btc_returns = btc_df["close"].pct_change().dropna().values[-5:]
            features["btc_momentum"] = np.mean(btc_returns) if len(btc_returns) > 0 else 0

        # Sector rotation signals (relative strength)
        relative_strength = {}
        for sym in symbols:
            df = historical_data[sym]
            if len(df) > 5:
                rs = df["close"].iloc[-1] / df["close"].iloc[-5] - 1
                relative_strength[sym] = rs

        if relative_strength:
            avg_rs = np.mean(list(relative_strength.values()))
            for sym, rs in relative_strength.items():
                features[f"relative_strength_{sym}"] = rs - avg_rs

            # Leader/laggard detection
            sorted_rs = sorted(relative_strength.items(), key=lambda x: x[1], reverse=True)
            features["leader"] = sorted_rs[0][0]
            features["laggard"] = sorted_rs[-1][0]

        # Flow analysis (volume changes)
        volume_changes = {}
        for sym in symbols:
            df = historical_data[sym]
            if len(df) > 5:
                recent_vol = df["volume"].iloc[-5:].mean()
                older_vol = df["volume"].iloc[-20:-5].mean() if len(df) > 20 else recent_vol
                volume_changes[sym] = (recent_vol / older_vol - 1) if older_vol > 0 else 0

        features["volume_flow"] = volume_changes

        return features

    def _calculate_asset_weight(
        self,
        symbol: str,
        cross_asset_context: Dict[str, Any],
        asset_context: Dict[str, Any],
    ) -> float:
        """Calculate position weight for an asset based on cross-asset signals."""
        base_weight = 1.0 / len(cross_asset_context.get("volume_flow", {symbol: 1}))

        # Adjust based on relative strength
        rs_key = f"relative_strength_{symbol}"
        if rs_key in cross_asset_context:
            rs = cross_asset_context[rs_key]
            # Overweight leaders, underweight laggards
            base_weight *= (1 + rs * 2)  # Scale factor

        # Reduce weight if high correlation (diversification)
        avg_corr = cross_asset_context.get("avg_correlation", 0.5)
        if avg_corr > 0.8:
            base_weight *= 0.7  # Reduce exposure in high-correlation environment

        # Boost weight for momentum assets
        if asset_context.get("trend") == "bullish":
            base_weight *= 1.2
        elif asset_context.get("trend") == "bearish":
            base_weight *= 0.8

        return max(0, min(1, base_weight))  # Clamp to [0, 1]

    def _create_multi_asset_execution(
        self,
        pipeline_result: "PipelineResult",
        current_bar: pd.Series,
        current_time: datetime,
        symbol: str,
        weight: float,
    ) -> Optional[ExecutionSummary]:
        """Create an ExecutionSummary for a specific asset in multi-asset mode."""
        try:
            current_price = float(current_bar["close"])
            direction = "LONG" if pipeline_result.pnl > 0 else "SHORT"

            atr_approx = current_price * 0.02

            if direction == "LONG":
                take_profit = current_price + 2 * atr_approx
                stop_loss = current_price - 1 * atr_approx
            else:
                take_profit = current_price - 2 * atr_approx
                stop_loss = current_price + 1 * atr_approx

            return ExecutionSummary(
                order_id=f"multi_{symbol}_{current_time.strftime('%Y%m%d_%H%M')}_{pipeline_result.trader_id}",
                direction=direction,
                order_type="MARKET",
                entry_price=current_price,
                take_profit=take_profit,
                stop_loss=stop_loss,
                position_size=0.05 * weight,  # Weighted position
                leverage=3.0,
                confidence=abs(pipeline_result.pnl) * weight,
                reasoning=f"Multi-asset {symbol}: weight={weight:.2f}, methods={pipeline_result.trader_methods}",
            )
        except Exception as e:
            print(f"⚠️ Error creating multi-asset execution for {symbol}: {e}")
            return None

    def _calculate_sharpe(self, pnls: List[float]) -> float:
        """Calculate Sharpe ratio from PnL series."""
        if len(pnls) < 2 or np.std(pnls) == 0:
            return 0.0
        return np.sqrt(2190) * np.mean(pnls) / np.std(pnls)

    def _calculate_max_drawdown(self, pnls: List[float], initial_capital: float) -> float:
        """Calculate maximum drawdown from PnL series."""
        if not pnls:
            return 0.0

        capital_curve = [initial_capital]
        for pnl in pnls:
            capital_curve.append(capital_curve[-1] + pnl)

        running_max = np.maximum.accumulate(capital_curve)
        drawdowns = (np.array(capital_curve) - running_max) / running_max
        return abs(np.min(drawdowns))

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction of a series."""
        if len(values) < 5:
            return "neutral"
        recent = np.mean(values[-5:])
        older = np.mean(values[:-5]) if len(values) > 5 else recent
        if recent > older * 1.05:
            return "increasing"
        elif recent < older * 0.95:
            return "decreasing"
        return "stable"


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
