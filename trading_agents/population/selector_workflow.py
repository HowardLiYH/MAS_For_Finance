"""Selector-based Population Workflow.

This workflow engine uses MethodSelector agents that dynamically choose
which methods to use from their role's inventory.

Key differences from fixed-variant approach:
1. Agents SELECT methods, not locked into fixed strategies
2. Continual learning is about WHICH methods to pick
3. Knowledge transfer shares selection preferences
4. Diversity is about method selection diversity

NEW in v0.9.0: Online Learning Integration
- OnlineModelManager provides true online learning (like hedge funds)
- Models update weights after EVERY observation
- Predictions improve continuously as new data arrives
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import json
import numpy as np
import pandas as pd
from datetime import datetime
from itertools import product
import random

from .selector import (
    MethodSelector,
    SelectorPopulation,
    SelectorPopulationConfig,
    AgentRole,
    SelectionOutcome,
)
from .inventories import (
    ANALYST_INVENTORY,
    RESEARCHER_INVENTORY,
    TRADER_INVENTORY,
    RISK_INVENTORY,
    get_method_info,
)
from ..inventory.online_models import OnlineModelManager
from ..inventory.feature_aligned_learner import FeatureAlignedLearner, FeatureAlignedConfig

# Real execution (v0.9.6)
try:
    from ..execution.pipeline_executor import PipelineExecutor, estimate_pipeline_pnl_from_execution
    HAS_REAL_EXECUTION = True
except ImportError:
    HAS_REAL_EXECUTION = False
    PipelineExecutor = None


@dataclass
class SelectorWorkflowConfig:
    """Configuration for selector-based workflow."""
    population_size: int = 5
    max_methods_per_agent: int = 3
    transfer_frequency: int = 10
    transfer_tau: float = 0.1
    exploration_rate: float = 0.15
    learning_rate: float = 0.1
    max_pipeline_samples: int = 25

    # Online learning settings (v0.9.0)
    use_online_models: bool = True  # Enable online learning
    online_n_features: int = 10  # Number of features for online models
    online_learning_rate: float = 0.001  # SGD learning rate
    online_forgetting_factor: float = 0.99  # Emphasize recent data
    persistence_dir: Optional[str] = None  # Where to save model state

    # Feature-aligned learning (v0.9.8) - RECOMMENDED
    # Update frequency matches FEATURE TIMESCALE, not model complexity!
    use_feature_aligned: bool = True  # Use feature-timescale-aligned learning
    fast_update_frequency: int = 1    # Fast features: every bar
    medium_update_frequency: int = 6  # Medium features: every 6 bars (~daily)
    slow_update_frequency: int = 42   # Slow features: every 42 bars (~weekly)

    # Real execution settings (v0.9.6)
    use_real_execution: bool = True  # Use actual method implementations
    log_execution_details: bool = False  # Log detailed execution info


@dataclass
class PipelineResult:
    """Result of running a single pipeline."""
    analyst_id: str
    researcher_id: str
    trader_id: str
    risk_id: str
    analyst_methods: List[str]
    researcher_methods: List[str]
    trader_methods: List[str]
    risk_methods: List[str]
    pnl: float
    sharpe: float
    success: bool
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agents": {
                "analyst": self.analyst_id,
                "researcher": self.researcher_id,
                "trader": self.trader_id,
                "risk": self.risk_id,
            },
            "methods": {
                "analyst": self.analyst_methods,
                "researcher": self.researcher_methods,
                "trader": self.trader_methods,
                "risk": self.risk_methods,
            },
            "pnl": self.pnl,
            "sharpe": self.sharpe,
            "success": self.success,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class IterationSummary:
    """Summary of one iteration."""
    iteration: int
    best_pnl: float
    avg_pnl: float
    best_pipeline: Dict[str, List[str]]  # role -> methods
    selection_diversity: Dict[str, float]  # role -> diversity
    transfer_performed: bool
    timestamp: datetime = field(default_factory=datetime.now)

    # Online learning fields (v0.9.0)
    online_signal: Optional[str] = None  # "long", "short", "hold"
    online_confidence: float = 0.0

    @property
    def best_methods(self) -> Dict[str, List[str]]:
        """Alias for best_pipeline for backwards compatibility."""
        return self.best_pipeline


class SelectorWorkflow:
    """
    Main workflow engine for PopAgent with method selection.

    Each iteration:
    1. Agents select methods from inventory
    2. Sample pipeline combinations
    3. Evaluate each pipeline
    4. Update agent preferences based on outcomes
    5. Transfer knowledge from best performers
    6. Ensure selection diversity
    """

    def __init__(self, config: Optional[SelectorWorkflowConfig] = None):
        self.config = config or SelectorWorkflowConfig()

        # Create populations for each role
        self.analyst_pop = SelectorPopulation(SelectorPopulationConfig(
            role=AgentRole.ANALYST,
            inventory=ANALYST_INVENTORY,
            population_size=self.config.population_size,
            max_methods_per_agent=self.config.max_methods_per_agent,
            exploration_rate=self.config.exploration_rate,
            learning_rate=self.config.learning_rate,
            transfer_frequency=self.config.transfer_frequency,
            transfer_tau=self.config.transfer_tau,
        ))

        self.researcher_pop = SelectorPopulation(SelectorPopulationConfig(
            role=AgentRole.RESEARCHER,
            inventory=RESEARCHER_INVENTORY,
            population_size=self.config.population_size,
            max_methods_per_agent=self.config.max_methods_per_agent,
            exploration_rate=self.config.exploration_rate,
            learning_rate=self.config.learning_rate,
            transfer_frequency=self.config.transfer_frequency,
            transfer_tau=self.config.transfer_tau,
        ))

        self.trader_pop = SelectorPopulation(SelectorPopulationConfig(
            role=AgentRole.TRADER,
            inventory=TRADER_INVENTORY,
            population_size=self.config.population_size,
            max_methods_per_agent=self.config.max_methods_per_agent,
            exploration_rate=self.config.exploration_rate,
            learning_rate=self.config.learning_rate,
            transfer_frequency=self.config.transfer_frequency,
            transfer_tau=self.config.transfer_tau,
        ))

        self.risk_pop = SelectorPopulation(SelectorPopulationConfig(
            role=AgentRole.RISK,
            inventory=RISK_INVENTORY,
            population_size=self.config.population_size,
            max_methods_per_agent=self.config.max_methods_per_agent,
            exploration_rate=self.config.exploration_rate,
            learning_rate=self.config.learning_rate,
            transfer_frequency=self.config.transfer_frequency,
            transfer_tau=self.config.transfer_tau,
        ))

        self.populations = {
            AgentRole.ANALYST: self.analyst_pop,
            AgentRole.RESEARCHER: self.researcher_pop,
            AgentRole.TRADER: self.trader_pop,
            AgentRole.RISK: self.risk_pop,
        }

        # History
        self.iteration = 0
        self.history: List[IterationSummary] = []
        self.all_results: List[PipelineResult] = []

        # Learning models
        self.online_models: Optional[OnlineModelManager] = None
        self.feature_aligned_learner: Optional[FeatureAlignedLearner] = None
        self.last_prediction: Optional[Dict[str, float]] = None
        self.last_features: Optional[np.ndarray] = None

        # Choose learning approach (priority: feature-aligned > online)
        if self.config.use_feature_aligned:
            # v0.9.8: Feature-aligned learning (RECOMMENDED)
            # Update frequency matches FEATURE TIMESCALE, not model complexity!
            fa_config = FeatureAlignedConfig(
                fast_update_frequency=self.config.fast_update_frequency,
                medium_update_frequency=self.config.medium_update_frequency,
                slow_update_frequency=self.config.slow_update_frequency,
            )
            self.feature_aligned_learner = FeatureAlignedLearner(
                n_features=self.config.online_n_features,
                config=fa_config,
            )
            print("✅ Feature-aligned learning enabled (v0.9.8)")
            print("   Fast features: update every bar | Medium: every 6 bars | Slow: every 42 bars")

        elif self.config.use_online_models:
            # Legacy: online-only learning
            self.online_models = OnlineModelManager(
                n_features=self.config.online_n_features
            )

            # Try to load saved state
            if self.config.persistence_dir:
                self._load_online_model_state()

        # Real Execution (v0.9.6) - Actual method implementations
        self.pipeline_executor = None
        if self.config.use_real_execution and HAS_REAL_EXECUTION and PipelineExecutor:
            self.pipeline_executor = PipelineExecutor()
            print("✅ Real pipeline execution enabled (v0.9.6)")

    def run_iteration(
        self,
        price_data: pd.DataFrame,
        market_context: Optional[Dict[str, Any]] = None,
        news_digest: Optional[Dict[str, Any]] = None,
        actual_return: Optional[float] = None,  # For online learning update
    ) -> IterationSummary:
        """
        Run one iteration of the population-based method selection workflow.

        Online Learning Flow:
        1. Extract features from current data
        2. Get prediction from online models (BEFORE seeing outcome)
        3. Execute pipeline and get results
        4. Update online models with observed outcome (AFTER)
        5. Update agent preferences

        This is how hedge funds operate - models improve with each observation.
        """
        self.iteration += 1
        context = market_context or {}

        # ===== ONLINE LEARNING: Update with previous outcome =====
        if self.online_models and self.last_features is not None and actual_return is not None:
            # Update models with the observed outcome from LAST iteration
            self.online_models.update_all(self.last_features, actual_return)

        # ===== ONLINE LEARNING: Extract features and predict =====
        features = None
        online_signal = None
        online_confidence = 0.0
        online_details = {}

        if self.online_models:
            features = self._extract_features(price_data, context)
            self.last_features = features

            # Get prediction BEFORE seeing this bar's outcome
            online_signal, online_confidence, online_details = (
                self.online_models.get_combined_signal(features)
            )
            self.last_prediction = online_details

            # Add online model predictions to context for agents
            context["online_signal"] = online_signal
            context["online_confidence"] = online_confidence
            context["online_regime"] = online_details.get("regime", "Neutral")
            context["online_volatility"] = online_details.get("volatility", 0.02)

        # 1. Each agent selects methods (now informed by online models)
        for pop in self.populations.values():
            for agent in pop.agents:
                agent.select_methods(context)

        # 2. Sample pipeline combinations
        pipelines = self._sample_pipelines()

        # 3. Evaluate each pipeline
        results: List[PipelineResult] = []
        for pipeline in pipelines:
            result = self._evaluate_pipeline(pipeline, price_data, context, news_digest)
            if result:
                results.append(result)

        # 4. Update agent preferences based on outcomes
        self._update_preferences(results, context)

        # 5. Score agents and transfer knowledge
        self._score_and_transfer()

        # 6. Ensure diversity
        for pop in self.populations.values():
            pop.ensure_diversity()

        # Create summary
        best_result = max(results, key=lambda r: r.pnl) if results else None

        summary = IterationSummary(
            iteration=self.iteration,
            best_pnl=best_result.pnl if best_result else 0.0,
            avg_pnl=np.mean([r.pnl for r in results]) if results else 0.0,
            best_pipeline={
                "analyst": best_result.analyst_methods if best_result else [],
                "researcher": best_result.researcher_methods if best_result else [],
                "trader": best_result.trader_methods if best_result else [],
                "risk": best_result.risk_methods if best_result else [],
            },
            selection_diversity={
                role.value: pop.calculate_selection_diversity()
                for role, pop in self.populations.items()
            },
            transfer_performed=self.analyst_pop.should_transfer(),
        )

        # Add online learning info to summary
        if online_signal:
            summary.online_signal = online_signal
            summary.online_confidence = online_confidence

        self.history.append(summary)
        self.all_results.extend(results)

        return summary

    def _extract_features(
        self,
        price_data: pd.DataFrame,
        context: Dict[str, Any],
    ) -> np.ndarray:
        """Extract features for online models from price data.

        Features:
        1. Returns (1, 5, 10 period)
        2. Volatility (5, 10 period)
        3. Volume ratio
        4. Trend indicator
        5. Context features
        """
        features = np.zeros(self.config.online_n_features)

        try:
            if "close" in price_data.columns and len(price_data) >= 10:
                close = price_data["close"].values

                # Returns
                features[0] = (close[-1] / close[-2] - 1) if len(close) >= 2 else 0
                features[1] = (close[-1] / close[-5] - 1) if len(close) >= 5 else 0
                features[2] = (close[-1] / close[-10] - 1) if len(close) >= 10 else 0

                # Volatility
                if len(close) >= 5:
                    returns = np.diff(close[-6:]) / close[-6:-1]
                    features[3] = np.std(returns)
                if len(close) >= 10:
                    returns = np.diff(close[-11:]) / close[-11:-1]
                    features[4] = np.std(returns)

                # Volume ratio
                if "volume" in price_data.columns:
                    vol = price_data["volume"].values
                    if len(vol) >= 5:
                        features[5] = vol[-1] / np.mean(vol[-5:]) if np.mean(vol[-5:]) > 0 else 1

                # Trend (SMA ratio)
                sma5 = np.mean(close[-5:]) if len(close) >= 5 else close[-1]
                sma10 = np.mean(close[-10:]) if len(close) >= 10 else close[-1]
                features[6] = (sma5 / sma10 - 1) if sma10 > 0 else 0

            # Context features
            features[7] = 1.0 if context.get("trend") == "bullish" else (-1.0 if context.get("trend") == "bearish" else 0.0)
            features[8] = context.get("volatility", 0.02)
            features[9] = 1.0 if context.get("regime") == "volatile" else 0.0

        except Exception as e:
            # Return zeros on error
            pass

        return features

    def _sample_pipelines(self) -> List[Dict[str, MethodSelector]]:
        """Sample pipeline combinations of agents."""
        analysts = self.analyst_pop.agents
        researchers = self.researcher_pop.agents
        traders = self.trader_pop.agents
        risks = self.risk_pop.agents

        # All possible combinations
        all_combos = list(product(analysts, researchers, traders, risks))

        # Sample if too many
        if len(all_combos) > self.config.max_pipeline_samples:
            # Include best agents
            elite_combo = (
                self.analyst_pop.get_best() or analysts[0],
                self.researcher_pop.get_best() or researchers[0],
                self.trader_pop.get_best() or traders[0],
                self.risk_pop.get_best() or risks[0],
            )

            sampled = random.sample(
                [c for c in all_combos if c != elite_combo],
                self.config.max_pipeline_samples - 1
            )
            all_combos = [elite_combo] + sampled

        return [
            {
                "analyst": combo[0],
                "researcher": combo[1],
                "trader": combo[2],
                "risk": combo[3],
            }
            for combo in all_combos
        ]

    def _evaluate_pipeline(
        self,
        pipeline: Dict[str, MethodSelector],
        price_data: pd.DataFrame,
        context: Dict[str, Any],
        news_digest: Optional[Dict[str, Any]],
    ) -> Optional[PipelineResult]:
        """Evaluate a single pipeline combination.

        v0.9.6: Now uses real method execution when enabled.
        """
        try:
            analyst = pipeline["analyst"]
            researcher = pipeline["researcher"]
            trader = pipeline["trader"]
            risk = pipeline["risk"]

            # Get selected methods
            analyst_methods = analyst.current_selection
            researcher_methods = researcher.current_selection
            trader_methods = trader.current_selection
            risk_methods = risk.current_selection

            # ===== USE REAL EXECUTION IF AVAILABLE (v0.9.6) =====
            if self.pipeline_executor is not None:
                result = self.pipeline_executor.execute(
                    analyst_methods=analyst_methods,
                    researcher_methods=researcher_methods,
                    trader_methods=trader_methods,
                    risk_methods=risk_methods,
                    price_data=price_data,
                    context=context,
                    news_digest=news_digest,
                )

                # Calculate actual PnL from price movement if we have enough data
                if len(price_data) >= 2:
                    actual_price_change = (
                        price_data["close"].iloc[-1] / price_data["close"].iloc[-2] - 1
                    )
                    pnl = estimate_pipeline_pnl_from_execution(result, actual_price_change)
                else:
                    pnl = result.expected_pnl

                sharpe = pnl / 0.02 if pnl != 0 else 0.0

                if self.config.log_execution_details:
                    print(f"  Pipeline: {result.action}, conf={result.confidence:.2f}, "
                          f"pnl={pnl:.4f}, time={result.execution_time_ms:.1f}ms")

                return PipelineResult(
                    analyst_id=analyst.id,
                    researcher_id=researcher.id,
                    trader_id=trader.id,
                    risk_id=risk.id,
                    analyst_methods=analyst_methods,
                    researcher_methods=researcher_methods,
                    trader_methods=trader_methods,
                    risk_methods=risk_methods,
                    pnl=pnl,
                    sharpe=sharpe,
                    success=result.action != "hold" and pnl > 0,
                )

            # ===== FALLBACK TO SIMULATED EXECUTION =====
            pnl = self._simulate_pipeline_pnl(
                analyst_methods,
                researcher_methods,
                trader_methods,
                risk_methods,
                price_data,
                context,
            )

            sharpe = pnl / 0.02 if pnl != 0 else 0.0

            return PipelineResult(
                analyst_id=analyst.id,
                researcher_id=researcher.id,
                trader_id=trader.id,
                risk_id=risk.id,
                analyst_methods=analyst_methods,
                researcher_methods=researcher_methods,
                trader_methods=trader_methods,
                risk_methods=risk_methods,
                pnl=pnl,
                sharpe=sharpe,
                success=pnl > 0,
            )

        except Exception as e:
            print(f"Pipeline evaluation error: {e}")
            return None

    def _simulate_pipeline_pnl(
        self,
        analyst_methods: List[str],
        researcher_methods: List[str],
        trader_methods: List[str],
        risk_methods: List[str],
        price_data: pd.DataFrame,
        context: Dict[str, Any],
    ) -> float:
        """
        Simulate PnL based on method selection.

        In production, this would actually execute the methods.
        For research, we can use historical performance or learned simulators.

        NEW: Respects "stay flat" methods to avoid trading during uncertainty.
        """
        # ===== CHECK FOR STAY FLAT / NO TRADE DECISION =====
        stay_flat_methods = {"StayFlat", "WaitForClarity", "ConfidenceThreshold"}
        has_stay_flat = bool(set(trader_methods) & stay_flat_methods)

        # Determine if we should stay flat based on context
        should_stay_flat = False

        if has_stay_flat:
            # Get online model confidence if available
            online_confidence = context.get("online_confidence", 0.5)
            online_signal = context.get("online_signal", "hold")
            volatility = context.get("volatility", 0.02)
            trend = context.get("trend", "neutral")
            regime = context.get("online_regime", "Neutral")

            if "StayFlat" in trader_methods:
                # Stay flat when signals conflict or trend is neutral
                if trend == "neutral" or online_signal == "hold":
                    should_stay_flat = True

            if "WaitForClarity" in trader_methods:
                # Stay flat during high volatility or regime transitions
                if volatility > 0.04 or regime == "Neutral":
                    should_stay_flat = True

            if "ConfidenceThreshold" in trader_methods:
                # Stay flat when confidence is low
                if online_confidence < 0.6:
                    should_stay_flat = True

        # If staying flat, return small positive (preserved capital, no trading cost)
        if should_stay_flat:
            # Small positive for avoiding bad trades, minus opportunity cost
            return 0.0001  # Tiny positive for not losing money

        # ===== NORMAL TRADING LOGIC =====
        # Base PnL from price movement
        if len(price_data) > 1 and "close" in price_data.columns:
            price_return = (
                price_data["close"].iloc[-1] / price_data["close"].iloc[-2] - 1
            )
        else:
            price_return = np.random.normal(0, 0.01)

        # Determine trade direction from context
        trend = context.get("trend", "neutral")
        online_signal = context.get("online_signal", "hold")

        # Trade direction logic
        if "MomentumEntry" in trader_methods:
            # Follow the trend
            if trend == "bullish" or online_signal == "long":
                trade_direction = 1.0  # Long
            elif trend == "bearish" or online_signal == "short":
                trade_direction = -1.0  # Short
            else:
                trade_direction = 0.0  # No clear signal
        elif "ContrarianEntry" in trader_methods:
            # Fade the trend
            if trend == "bearish" or online_signal == "short":
                trade_direction = 1.0  # Long against bearish
            elif trend == "bullish" or online_signal == "long":
                trade_direction = -1.0  # Short against bullish
            else:
                trade_direction = 0.0
        else:
            # Default: follow trend
            trade_direction = 1.0 if trend == "bullish" else (-1.0 if trend == "bearish" else 0.0)

        # If no clear direction, small loss (trading costs)
        if trade_direction == 0.0:
            return -0.0005  # Trading cost without direction

        # PnL = direction * price_return
        base_pnl = trade_direction * price_return

        # Method synergy bonuses
        synergy = 0.0

        # Analyst synergies
        if "HMM_Regime" in analyst_methods and "VolatilityClustering" in analyst_methods:
            synergy += 0.001
        if "RSI" in analyst_methods and "MACD" in analyst_methods:
            synergy += 0.0005

        # Researcher synergies
        if "BootstrapEnsemble" in researcher_methods or "QuantileRegression" in researcher_methods:
            synergy += 0.0005
        if "TemporalFusion" in researcher_methods:
            synergy += 0.001

        # Position sizing improvements
        if "VolatilityScaled" in trader_methods:
            # Reduce position in high vol = smaller losses
            vol = context.get("volatility", 0.02)
            if vol > 0.03:
                base_pnl *= 0.5  # Half size in high vol
        if "KellyCriterion" in trader_methods:
            synergy += 0.0005

        # Risk protection (reduces losses)
        if "MaxDrawdown" in risk_methods or "DailyStopLoss" in risk_methods:
            if base_pnl < -0.02:  # Stop loss kicks in
                base_pnl = -0.02
        if "TrailingStop" in risk_methods:
            if base_pnl < -0.015:
                base_pnl = -0.015

        # Trading costs
        trading_cost = 0.001  # 0.1% round trip

        # Final PnL
        pnl = base_pnl + synergy - trading_cost + np.random.normal(0, 0.002)

        return pnl

    def _update_preferences(
        self,
        results: List[PipelineResult],
        context: Dict[str, Any],
    ) -> None:
        """Update agent preferences based on pipeline results."""
        # Group results by agent
        agent_outcomes: Dict[str, List[Tuple[List[str], float]]] = {}

        for result in results:
            # Analyst
            key = f"analyst:{result.analyst_id}"
            if key not in agent_outcomes:
                agent_outcomes[key] = []
            agent_outcomes[key].append((result.analyst_methods, result.pnl))

            # Researcher
            key = f"researcher:{result.researcher_id}"
            if key not in agent_outcomes:
                agent_outcomes[key] = []
            agent_outcomes[key].append((result.researcher_methods, result.pnl))

            # Trader
            key = f"trader:{result.trader_id}"
            if key not in agent_outcomes:
                agent_outcomes[key] = []
            agent_outcomes[key].append((result.trader_methods, result.pnl))

            # Risk
            key = f"risk:{result.risk_id}"
            if key not in agent_outcomes:
                agent_outcomes[key] = []
            agent_outcomes[key].append((result.risk_methods, result.pnl))

        # Update each agent
        for key, outcomes in agent_outcomes.items():
            role, agent_id = key.split(":")
            pop = self.populations[AgentRole(role)]
            agent = pop.get_agent(agent_id)

            if agent:
                for methods, pnl in outcomes:
                    agent.update_from_outcome(SelectionOutcome(
                        methods_used=methods,
                        reward=pnl,
                        market_context=context,
                    ))

    def _score_and_transfer(self) -> None:
        """Score agents and perform knowledge transfer if needed."""
        for role, pop in self.populations.items():
            # Calculate scores
            scores = {}
            for agent in pop.agents:
                if agent.outcome_history:
                    recent = agent.outcome_history[-20:]
                    avg_reward = np.mean([o.reward for o in recent])
                    scores[agent.id] = avg_reward
                else:
                    scores[agent.id] = 0.0

            pop.update_scores(scores)

            # Transfer if it's time
            if pop.should_transfer():
                pop.transfer_knowledge()

    def get_best_methods(self) -> Dict[str, List[str]]:
        """Get the currently best method selection for each role."""
        result = {}
        for role, pop in self.populations.items():
            best = pop.get_best()
            if best:
                result[role.value] = best.current_selection
            else:
                result[role.value] = []
        return result

    def get_method_popularity(self) -> Dict[str, Dict[str, float]]:
        """Get how popular each method is across the population."""
        result = {}
        for role, pop in self.populations.items():
            usage = pop._get_method_usage()
            total = sum(usage.values()) or 1
            result[role.value] = {m: c / total for m, c in usage.items()}
        return result

    def get_learning_progress(self) -> Dict[str, Any]:
        """Get learning progress metrics."""
        if not self.history:
            return {"message": "No history yet"}

        pnls = [h.best_pnl for h in self.history]

        return {
            "iterations": len(self.history),
            "avg_pnl_first_10": np.mean(pnls[:10]) if len(pnls) >= 10 else np.mean(pnls),
            "avg_pnl_last_10": np.mean(pnls[-10:]),
            "improvement": (np.mean(pnls[-10:]) - np.mean(pnls[:10])) if len(pnls) >= 20 else 0.0,
            "best_methods": self.get_best_methods(),
            "method_popularity": self.get_method_popularity(),
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary."""
        summary = {
            "config": {
                "population_size": self.config.population_size,
                "max_methods_per_agent": self.config.max_methods_per_agent,
                "transfer_frequency": self.config.transfer_frequency,
                "use_online_models": self.config.use_online_models,
            },
            "inventory_sizes": {
                "analyst": len(ANALYST_INVENTORY),
                "researcher": len(RESEARCHER_INVENTORY),
                "trader": len(TRADER_INVENTORY),
                "risk": len(RISK_INVENTORY),
            },
            "iteration": self.iteration,
            "total_pipelines_evaluated": len(self.all_results),
            "learning_progress": self.get_learning_progress(),
            "population_stats": {
                role.value: pop.get_population_stats()
                for role, pop in self.populations.items()
            },
        }

        # Add online learning stats
        if self.online_models:
            summary["online_learning"] = {
                "enabled": True,
                "model_stats": self.online_models.get_state(),
                "last_prediction": self.last_prediction,
            }

        return summary

    def _load_online_model_state(self) -> None:
        """Load online model state from disk."""
        if not self.config.persistence_dir or not self.online_models:
            return

        state_file = Path(self.config.persistence_dir) / "online_models.json"
        if state_file.exists():
            try:
                with open(state_file, "r") as f:
                    state = json.load(f)
                self.online_models.load_state(state)
                print(f"✓ Loaded online model state ({state.get('return_predictor', {}).get('n_updates', 0)} updates)")
            except Exception as e:
                print(f"⚠ Could not load online model state: {e}")

    def save_online_model_state(self) -> None:
        """Save online model state to disk."""
        if not self.config.persistence_dir or not self.online_models:
            return

        state_file = Path(self.config.persistence_dir) / "online_models.json"
        state_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            state = self.online_models.get_state()
            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)
            print(f"✓ Saved online model state to {state_file}")
        except Exception as e:
            print(f"⚠ Could not save online model state: {e}")

    def get_online_model_info(self) -> Dict[str, Any]:
        """Get detailed info about online models."""
        if not self.online_models:
            return {"enabled": False}

        state = self.online_models.get_state()
        return {
            "enabled": True,
            "return_predictor": {
                "updates": state.get("return_predictor", {}).get("n_updates", 0),
                "recent_mae": state.get("return_predictor", {}).get("recent_mae", 0),
            },
            "volatility": {
                "updates": state.get("volatility", {}).get("n_updates", 0),
                "current_vol": np.sqrt(state.get("volatility", {}).get("variance", 0.0001)),
            },
            "regime": {
                "updates": state.get("regime", {}).get("n_updates", 0),
                "current_probs": state.get("regime", {}).get("regime_probs", []),
            },
        }


def create_selector_workflow(
    population_size: int = 5,
    max_methods: int = 3,
    **kwargs
) -> SelectorWorkflow:
    """Factory function to create a selector workflow."""
    config = SelectorWorkflowConfig(
        population_size=population_size,
        max_methods_per_agent=max_methods,
        **kwargs
    )
    return SelectorWorkflow(config)
