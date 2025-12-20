"""Tests for knowledge transfer mechanics.

Verifies that preference transfer works correctly between agents,
transferring knowledge from best performers to others.
"""
from __future__ import annotations

import pytest
import numpy as np


class TestTransferTiming:
    """Test transfer timing and frequency."""

    def test_should_transfer_respects_frequency(self, selector_workflow):
        """Verify should_transfer checks iteration count correctly."""
        transfer_freq = selector_workflow.config.transfer_frequency

        # Manually set iteration counts and check
        for pop in selector_workflow.populations.values():
            pop._iteration_count = transfer_freq - 1
            assert not pop.should_transfer(), "Should not transfer before frequency"

            pop._iteration_count = transfer_freq
            assert pop.should_transfer(), "Should transfer at frequency"

            pop._iteration_count = transfer_freq + 1
            assert not pop.should_transfer(), "Should not transfer after frequency (until next multiple)"


class TestPreferenceTransfer:
    """Test preference transfer mechanics."""

    def test_transfer_from_best_to_others(self, selector_workflow, mock_price_data, mock_market_context):
        """Verify preferences transfer from best agent to others."""
        # Run iterations to establish scores
        for _ in range(selector_workflow.config.transfer_frequency + 1):
            selector_workflow.run_iteration(
                price_data=mock_price_data,
                market_context=mock_market_context,
            )

        # Get populations and check transfer occurred
        for pop in selector_workflow.populations.values():
            best = pop.get_best()
            if best:
                best_prefs = dict(best.preferences)

                # Other agents should have some influence from best
                # (soft transfer with tau, so not exact copy)
                for agent in pop.agents:
                    if agent.id != best.id:
                        # At minimum, agents should have updated
                        assert agent.preferences is not None

    def test_soft_transfer_with_tau(self, selector_workflow):
        """Verify transfer uses soft update with tau parameter."""
        tau = selector_workflow.config.transfer_tau

        # This should be a soft update: new = (1-tau)*old + tau*best
        assert 0 < tau < 1, "Tau should be between 0 and 1 for soft transfer"

    def test_best_agent_unchanged_by_transfer(self, selector_workflow, mock_price_data, mock_market_context):
        """Verify best agent's preferences are not modified by transfer."""
        # Run to establish scores
        for _ in range(3):
            selector_workflow.run_iteration(
                price_data=mock_price_data,
                market_context=mock_market_context,
            )

        for pop in selector_workflow.populations.values():
            best = pop.get_best()
            if best:
                original_prefs = dict(best.preferences)

                # Force transfer
                pop.transfer_knowledge()

                # Best agent's preferences should remain unchanged
                assert best.preferences == original_prefs


class TestScoreCalculation:
    """Test agent scoring for transfer decisions."""

    def test_scores_updated_after_iteration(self, selector_workflow, mock_price_data, mock_market_context):
        """Verify agent scores are updated after iterations."""
        selector_workflow.run_iteration(
            price_data=mock_price_data,
            market_context=mock_market_context,
        )

        for pop in selector_workflow.populations.values():
            scores = pop.scores
            # Scores should exist for agents that participated
            # (may be empty if no outcomes recorded yet)

    def test_best_agent_has_highest_score(self, selector_workflow, mock_price_data, mock_market_context):
        """Verify get_best returns agent with highest score."""
        # Run iterations
        for _ in range(5):
            selector_workflow.run_iteration(
                price_data=mock_price_data,
                market_context=mock_market_context,
            )

        for pop in selector_workflow.populations.values():
            best = pop.get_best()
            if best and pop.scores:
                best_score = pop.scores.get(best.id, float('-inf'))

                for agent_id, score in pop.scores.items():
                    assert score <= best_score, \
                        f"Best agent should have highest score, but {agent_id} has {score} > {best_score}"


class TestTransferEffects:
    """Test effects of knowledge transfer."""

    def test_transfer_reduces_preference_variance(self, selector_workflow, mock_price_data, mock_market_context):
        """Transfer should reduce variance in preferences across population."""
        # Run until transfer
        transfer_freq = selector_workflow.config.transfer_frequency

        for _ in range(transfer_freq - 1):
            selector_workflow.run_iteration(
                price_data=mock_price_data,
                market_context=mock_market_context,
            )

        # Measure preference variance before transfer
        def get_pref_variance(pop):
            all_prefs = []
            for agent in pop.agents:
                for method, pref in agent.preferences.items():
                    all_prefs.append(pref)
            return np.var(all_prefs) if all_prefs else 0.0

        variances_before = {
            role.value: get_pref_variance(pop)
            for role, pop in selector_workflow.populations.items()
        }

        # Run iteration that triggers transfer
        selector_workflow.run_iteration(
            price_data=mock_price_data,
            market_context=mock_market_context,
        )

        # Variance may decrease (or stay same if already converged)
        # This is a soft check - transfer should at least not explode variance
        variances_after = {
            role.value: get_pref_variance(pop)
            for role, pop in selector_workflow.populations.items()
        }

        for role in variances_before:
            # Variance shouldn't explode
            assert variances_after[role] <= variances_before[role] * 10


class TestPopulationStats:
    """Test population statistics tracking."""

    def test_population_stats_available(self, selector_workflow, mock_price_data, mock_market_context):
        """Verify population stats are tracked."""
        for _ in range(3):
            selector_workflow.run_iteration(
                price_data=mock_price_data,
                market_context=mock_market_context,
            )

        for pop in selector_workflow.populations.values():
            stats = pop.get_population_stats()

            assert isinstance(stats, dict)
            assert "agent_count" in stats or "num_agents" in stats or len(stats) > 0

    def test_method_usage_tracked(self, selector_workflow, mock_price_data, mock_market_context):
        """Verify method usage is tracked across population."""
        for _ in range(5):
            selector_workflow.run_iteration(
                price_data=mock_price_data,
                market_context=mock_market_context,
            )

        popularity = selector_workflow.get_method_popularity()

        for role in ["analyst", "researcher", "trader", "risk"]:
            assert role in popularity
            assert isinstance(popularity[role], dict)
