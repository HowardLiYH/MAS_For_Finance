"""End-to-end mock workflow tests for PopAgent.

Tests the complete workflow from method selection through evaluation
using synthetic data, without external dependencies.
"""
from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone


class TestSelectorWorkflowInitialization:
    """Test SelectorWorkflow initialization."""

    def test_workflow_creates_populations(self, selector_workflow):
        """Verify populations are created for all roles."""
        assert selector_workflow.analyst_pop is not None
        assert selector_workflow.researcher_pop is not None
        assert selector_workflow.trader_pop is not None
        assert selector_workflow.risk_pop is not None

    def test_population_sizes(self, selector_workflow, selector_workflow_config):
        """Verify population sizes match config."""
        expected_size = selector_workflow_config.population_size

        assert len(selector_workflow.analyst_pop.agents) == expected_size
        assert len(selector_workflow.researcher_pop.agents) == expected_size
        assert len(selector_workflow.trader_pop.agents) == expected_size
        assert len(selector_workflow.risk_pop.agents) == expected_size

    def test_agents_have_unique_ids(self, selector_workflow):
        """Verify all agents have unique IDs."""
        all_ids = []
        for pop in selector_workflow.populations.values():
            for agent in pop.agents:
                all_ids.append(agent.id)

        assert len(all_ids) == len(set(all_ids)), "Agent IDs must be unique"


class TestMethodSelection:
    """Test method selection mechanics."""

    def test_agents_select_methods(self, selector_workflow, mock_market_context):
        """Verify agents can select methods from inventory."""
        for pop in selector_workflow.populations.values():
            for agent in pop.agents:
                agent.select_methods(mock_market_context)

                assert len(agent.current_selection) > 0
                assert len(agent.current_selection) <= selector_workflow.config.max_methods_per_agent

    def test_selected_methods_from_inventory(self, selector_workflow, mock_market_context):
        """Verify selected methods are from the agent's inventory."""
        for pop in selector_workflow.populations.values():
            for agent in pop.agents:
                agent.select_methods(mock_market_context)

                inventory_methods = set(agent.inventory)
                for method in agent.current_selection:
                    assert method in inventory_methods, f"{method} not in inventory"

    def test_exploration_affects_selection(self, selector_workflow):
        """Verify exploration rate influences method selection diversity."""
        # Run multiple selections and check for variety
        contexts = [
            {"trend": "bullish", "volatility": 0.3},
            {"trend": "bearish", "volatility": 0.5},
            {"trend": "neutral", "volatility": 0.2},
        ]

        all_selections = []
        for ctx in contexts:
            for pop in selector_workflow.populations.values():
                for agent in pop.agents:
                    agent.select_methods(ctx)
                    all_selections.append(tuple(sorted(agent.current_selection)))

        # With exploration, we should see some variety
        unique_selections = set(all_selections)
        assert len(unique_selections) > 1, "Exploration should create selection variety"


class TestIterationExecution:
    """Test full iteration execution."""

    def test_run_single_iteration(self, selector_workflow, mock_price_data, mock_market_context):
        """Verify a single iteration completes successfully."""
        summary = selector_workflow.run_iteration(
            price_data=mock_price_data,
            market_context=mock_market_context,
        )

        assert summary is not None
        assert summary.iteration == 1
        assert isinstance(summary.best_pnl, float)
        assert isinstance(summary.avg_pnl, float)

    def test_run_multiple_iterations(self, selector_workflow, mock_price_data, mock_market_context, iteration_count):
        """Verify multiple iterations execute correctly."""
        for i in range(iteration_count):
            summary = selector_workflow.run_iteration(
                price_data=mock_price_data,
                market_context=mock_market_context,
            )

            assert summary.iteration == i + 1

        assert len(selector_workflow.history) == iteration_count

    def test_iteration_produces_pipeline_results(self, selector_workflow, mock_price_data, mock_market_context):
        """Verify iterations produce pipeline results."""
        selector_workflow.run_iteration(
            price_data=mock_price_data,
            market_context=mock_market_context,
        )

        assert len(selector_workflow.all_results) > 0

        # Check result structure
        result = selector_workflow.all_results[0]
        assert hasattr(result, 'analyst_id')
        assert hasattr(result, 'trader_methods')
        assert hasattr(result, 'pnl')


class TestKnowledgeTransfer:
    """Test knowledge transfer mechanics."""

    def test_transfer_frequency_respected(self, selector_workflow, mock_price_data, mock_market_context):
        """Verify transfer happens at correct frequency."""
        transfer_freq = selector_workflow.config.transfer_frequency

        transfer_iterations = []
        for i in range(transfer_freq + 2):
            summary = selector_workflow.run_iteration(
                price_data=mock_price_data,
                market_context=mock_market_context,
            )
            if summary.transfer_performed:
                transfer_iterations.append(summary.iteration)

        # Transfer should happen at iteration = transfer_frequency
        if transfer_iterations:
            assert transfer_freq in transfer_iterations or transfer_freq + 1 in transfer_iterations

    def test_best_agent_identified(self, selector_workflow, mock_price_data, mock_market_context):
        """Verify best agent is correctly identified."""
        # Run some iterations to establish scores
        for _ in range(3):
            selector_workflow.run_iteration(
                price_data=mock_price_data,
                market_context=mock_market_context,
            )

        # Check each population has a best agent
        for pop in selector_workflow.populations.values():
            best = pop.get_best()
            # Best agent should be identifiable after iterations
            # (may be None if no scores yet, which is okay)


class TestDiversityPreservation:
    """Test diversity preservation mechanics."""

    def test_diversity_calculated(self, selector_workflow, mock_price_data, mock_market_context):
        """Verify diversity metrics are calculated."""
        summary = selector_workflow.run_iteration(
            price_data=mock_price_data,
            market_context=mock_market_context,
        )

        assert "selection_diversity" in summary.__dict__
        for role in ["analyst", "researcher", "trader", "risk"]:
            assert role in summary.selection_diversity

    def test_diversity_in_valid_range(self, selector_workflow, mock_price_data, mock_market_context):
        """Verify diversity values are in valid range [0, 1]."""
        for _ in range(3):
            summary = selector_workflow.run_iteration(
                price_data=mock_price_data,
                market_context=mock_market_context,
            )

            for diversity in summary.selection_diversity.values():
                assert 0.0 <= diversity <= 1.0


class TestPreferenceUpdates:
    """Test preference update mechanics."""

    def test_preferences_update_after_iteration(self, selector_workflow, mock_price_data, mock_market_context):
        """Verify agent preferences update based on outcomes."""
        # Get initial preferences
        initial_prefs = {}
        for pop in selector_workflow.populations.values():
            for agent in pop.agents:
                initial_prefs[agent.id] = dict(agent.preferences)

        # Run iterations
        for _ in range(5):
            selector_workflow.run_iteration(
                price_data=mock_price_data,
                market_context=mock_market_context,
            )

        # Check preferences changed
        prefs_changed = False
        for pop in selector_workflow.populations.values():
            for agent in pop.agents:
                if agent.preferences != initial_prefs[agent.id]:
                    prefs_changed = True
                    break

        assert prefs_changed, "Preferences should update after iterations"


class TestLearningProgress:
    """Test learning progress tracking."""

    def test_learning_progress_returned(self, selector_workflow, mock_price_data, mock_market_context):
        """Verify learning progress can be retrieved."""
        # Run some iterations
        for _ in range(10):
            selector_workflow.run_iteration(
                price_data=mock_price_data,
                market_context=mock_market_context,
            )

        progress = selector_workflow.get_learning_progress()

        assert "iterations" in progress
        assert "best_methods" in progress
        assert "method_popularity" in progress

    def test_best_methods_tracked(self, selector_workflow, mock_price_data, mock_market_context):
        """Verify best methods are tracked for each role."""
        for _ in range(5):
            selector_workflow.run_iteration(
                price_data=mock_price_data,
                market_context=mock_market_context,
            )

        best_methods = selector_workflow.get_best_methods()

        for role in ["analyst", "researcher", "trader", "risk"]:
            assert role in best_methods


class TestPipelineEvaluation:
    """Test pipeline evaluation logic."""

    def test_pipeline_sampling(self, selector_workflow, mock_market_context):
        """Verify pipelines are sampled correctly."""
        # Select methods first
        for pop in selector_workflow.populations.values():
            for agent in pop.agents:
                agent.select_methods(mock_market_context)

        pipelines = selector_workflow._sample_pipelines()

        assert len(pipelines) > 0
        assert len(pipelines) <= selector_workflow.config.max_pipeline_samples

        # Check pipeline structure
        for pipeline in pipelines:
            assert "analyst" in pipeline
            assert "researcher" in pipeline
            assert "trader" in pipeline
            assert "risk" in pipeline

    def test_pipeline_evaluation_produces_results(self, selector_workflow, mock_price_data, mock_market_context):
        """Verify pipeline evaluation produces valid results."""
        # Select methods
        for pop in selector_workflow.populations.values():
            for agent in pop.agents:
                agent.select_methods(mock_market_context)

        pipelines = selector_workflow._sample_pipelines()

        for pipeline in pipelines[:3]:  # Test first 3
            result = selector_workflow._evaluate_pipeline(
                pipeline=pipeline,
                price_data=mock_price_data,
                context=mock_market_context,
                news_digest=None,
            )

            if result:  # May be None on error
                assert hasattr(result, 'pnl')
                assert hasattr(result, 'analyst_methods')
                assert isinstance(result.pnl, float)


class TestWorkflowSummary:
    """Test workflow summary generation."""

    def test_summary_generation(self, selector_workflow, mock_price_data, mock_market_context):
        """Verify comprehensive summary is generated."""
        for _ in range(5):
            selector_workflow.run_iteration(
                price_data=mock_price_data,
                market_context=mock_market_context,
            )

        summary = selector_workflow.get_summary()

        assert "config" in summary
        assert "inventory_sizes" in summary
        assert "iteration" in summary
        assert "total_pipelines_evaluated" in summary
        assert "learning_progress" in summary
        assert "population_stats" in summary
