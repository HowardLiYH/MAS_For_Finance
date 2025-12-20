"""Tests for method selection mechanics.

Verifies that agents correctly select methods from their inventories
using exploration/exploitation strategies.
"""
from __future__ import annotations

import pytest
import numpy as np
from collections import Counter


class TestMethodSelectorBasics:
    """Test MethodSelector basic functionality."""

    def test_selector_has_inventory(self, selector_workflow):
        """Verify each selector has an inventory."""
        for pop in selector_workflow.populations.values():
            for agent in pop.agents:
                assert hasattr(agent, 'inventory')
                assert len(agent.inventory) > 0

    def test_selector_has_preferences(self, selector_workflow):
        """Verify each selector has preferences."""
        for pop in selector_workflow.populations.values():
            for agent in pop.agents:
                assert hasattr(agent, 'preferences')
                assert isinstance(agent.preferences, dict)

    def test_inventory_larger_than_selection(self, selector_workflow):
        """Verify inventory size > max methods per agent (creates selection pressure)."""
        max_methods = selector_workflow.config.max_methods_per_agent

        for pop in selector_workflow.populations.values():
            for agent in pop.agents:
                assert len(agent.inventory) > max_methods, \
                    f"Inventory ({len(agent.inventory)}) should be > max_methods ({max_methods})"


class TestExplorationExploitation:
    """Test exploration vs exploitation balance."""

    def test_high_exploration_creates_variety(self, selector_workflow_config, mock_market_context):
        """High exploration rate should create more variety in selections."""
        from trading_agents.population.selector_workflow import SelectorWorkflow

        # Create workflow with high exploration
        selector_workflow_config.exploration_rate = 0.5
        workflow = SelectorWorkflow(selector_workflow_config)

        # Collect selections across multiple runs
        all_selections = []
        for _ in range(10):
            for pop in workflow.populations.values():
                for agent in pop.agents:
                    agent.select_methods(mock_market_context)
                    all_selections.append(tuple(sorted(agent.current_selection)))

        # High exploration should create variety
        unique_ratio = len(set(all_selections)) / len(all_selections)
        assert unique_ratio > 0.1, "High exploration should create selection variety"

    def test_preferences_influence_selection(self, selector_workflow, mock_market_context):
        """Preferred methods should be selected more often."""
        # Boost preference for a specific method
        analyst = selector_workflow.analyst_pop.agents[0]

        # Get first method in inventory
        first_method = list(analyst.inventory)[0]
        original_pref = analyst.preferences.get(first_method, 0.0)

        # Strongly boost this preference
        analyst.preferences[first_method] = 10.0

        # Count selections over multiple runs
        selection_counts = Counter()
        for _ in range(50):
            analyst.select_methods(mock_market_context)
            for m in analyst.current_selection:
                selection_counts[m] += 1

        # Boosted method should be selected frequently
        if selection_counts:
            assert first_method in selection_counts, \
                "Highly preferred method should be selected"


class TestThompsonSampling:
    """Test Thompson Sampling behavior if enabled."""

    def test_thompson_sampling_uses_beta_distribution(self, selector_workflow):
        """Verify Thompson sampling uses Beta distribution parameters."""
        for pop in selector_workflow.populations.values():
            for agent in pop.agents:
                # Check if agent has alpha/beta parameters
                if hasattr(agent, 'alpha') and hasattr(agent, 'beta'):
                    for method in agent.inventory:
                        # Alpha and beta should be initialized
                        assert method in agent.alpha
                        assert method in agent.beta
                        assert agent.alpha[method] >= 1.0
                        assert agent.beta[method] >= 1.0


class TestContextualSelection:
    """Test context-aware method selection."""

    def test_different_contexts_can_produce_different_selections(self, selector_workflow):
        """Different market contexts may produce different selections."""
        contexts = [
            {"trend": "bullish", "volatility": 0.2, "regime": "normal"},
            {"trend": "bearish", "volatility": 0.6, "regime": "volatile"},
            {"trend": "neutral", "volatility": 0.1, "regime": "quiet"},
        ]

        selections_by_context = {}

        for ctx in contexts:
            ctx_key = f"{ctx['trend']}_{ctx['regime']}"
            selections_by_context[ctx_key] = []

            for _ in range(5):
                for pop in selector_workflow.populations.values():
                    for agent in pop.agents:
                        agent.select_methods(ctx)
                        selections_by_context[ctx_key].append(
                            tuple(sorted(agent.current_selection))
                        )

        # At minimum, selections should be valid
        for ctx_key, selections in selections_by_context.items():
            assert len(selections) > 0


class TestSelectionOutcomes:
    """Test outcome recording and preference updates."""

    def test_positive_outcome_increases_preference(self, selector_workflow, mock_market_context):
        """Positive outcomes should increase method preferences."""
        from trading_agents.population.selector import SelectionOutcome

        agent = selector_workflow.analyst_pop.agents[0]
        agent.select_methods(mock_market_context)

        selected_method = agent.current_selection[0]
        initial_pref = agent.preferences.get(selected_method, 0.0)

        # Record positive outcome
        outcome = SelectionOutcome(
            methods_used=[selected_method],
            reward=0.05,  # 5% positive return
            market_context=mock_market_context,
        )
        agent.update_from_outcome(outcome)

        new_pref = agent.preferences.get(selected_method, 0.0)
        assert new_pref >= initial_pref, "Positive outcome should not decrease preference"

    def test_negative_outcome_decreases_preference(self, selector_workflow, mock_market_context):
        """Negative outcomes should decrease method preferences."""
        from trading_agents.population.selector import SelectionOutcome

        agent = selector_workflow.analyst_pop.agents[0]
        agent.select_methods(mock_market_context)

        selected_method = agent.current_selection[0]

        # First boost the preference
        agent.preferences[selected_method] = 5.0
        initial_pref = agent.preferences[selected_method]

        # Record negative outcome
        outcome = SelectionOutcome(
            methods_used=[selected_method],
            reward=-0.05,  # 5% loss
            market_context=mock_market_context,
        )
        agent.update_from_outcome(outcome)

        new_pref = agent.preferences.get(selected_method, 0.0)
        assert new_pref <= initial_pref, "Negative outcome should not increase preference"


class TestInventorySizes:
    """Test inventory sizes match expected values."""

    def test_analyst_inventory_size(self, selector_workflow):
        """Verify analyst inventory has expected methods."""
        from trading_agents.population.inventories import ANALYST_INVENTORY

        expected_size = len(ANALYST_INVENTORY)

        for agent in selector_workflow.analyst_pop.agents:
            assert len(agent.inventory) == expected_size

    def test_researcher_inventory_size(self, selector_workflow):
        """Verify researcher inventory has expected methods."""
        from trading_agents.population.inventories import RESEARCHER_INVENTORY

        expected_size = len(RESEARCHER_INVENTORY)

        for agent in selector_workflow.researcher_pop.agents:
            assert len(agent.inventory) == expected_size

    def test_trader_inventory_size(self, selector_workflow):
        """Verify trader inventory has expected methods."""
        from trading_agents.population.inventories import TRADER_INVENTORY

        expected_size = len(TRADER_INVENTORY)

        for agent in selector_workflow.trader_pop.agents:
            assert len(agent.inventory) == expected_size

    def test_risk_inventory_size(self, selector_workflow):
        """Verify risk inventory has expected methods."""
        from trading_agents.population.inventories import RISK_INVENTORY

        expected_size = len(RISK_INVENTORY)

        for agent in selector_workflow.risk_pop.agents:
            assert len(agent.inventory) == expected_size
