"""Test recursive removal of dependent symbols."""

from unittest.mock import Mock

import pytest
import sympy as sp

from dqx.common import DQXError
from dqx.orm.repositories import MetricDB
from dqx.provider import MetricProvider


class TestRecursiveSymbolRemoval:
    """Test recursive removal of dependent symbols."""

    @pytest.fixture
    def provider(self) -> MetricProvider:
        """Create a MetricProvider instance for testing."""
        mock_db = Mock(spec=MetricDB)
        return MetricProvider(mock_db, execution_id="test-exec-123")

    def test_remove_symbol_removes_its_dependencies(self, provider: MetricProvider) -> None:
        """Test that removing a symbol also removes its required_metrics."""
        # Create a chain where DoD depends on metrics
        avg_tax = provider.average("tax", dataset="ds1")

        # Create DoD that depends on avg_tax
        # DoD will create lag=0 and lag=1 versions internally
        dod = provider.ext.day_over_day(avg_tax, dataset="ds1")

        # Get the DoD metric to check its dependencies
        dod_metric = provider.get_symbol(dod)
        assert len(dod_metric.required_metrics) == 2  # DoD has 2 dependencies (lag=0 and lag=1)
        dep_symbols = dod_metric.required_metrics

        # Remove the DoD metric
        provider.remove_symbol(dod)

        # The DoD should be removed
        assert dod not in provider.symbols()

        # Its dependencies should also be removed
        for dep in dep_symbols:
            assert dep not in provider.symbols()

        # But the original avg_tax should still exist
        assert avg_tax in provider.symbols()

    def test_remove_symbol_with_shared_dependencies(self, provider: MetricProvider) -> None:
        """Test that dependencies are removed correctly."""
        # Create base metrics
        avg_price = provider.average("price", dataset="ds1")

        # Create two metrics that depend on avg_price
        dod_price = provider.ext.day_over_day(avg_price, dataset="ds1")
        wow_price = provider.ext.week_over_week(avg_price, dataset="ds1")

        # Get the metrics to check dependencies
        dod_metric = provider.get_symbol(dod_price)
        wow_metric = provider.get_symbol(wow_price)

        # DoD creates lag=0 and lag=1 versions
        assert len(dod_metric.required_metrics) == 2
        # WoW creates lag=0 and lag=7 versions
        assert len(wow_metric.required_metrics) == 2

        # Remove DoD
        provider.remove_symbol(dod_price)

        # DoD should be gone
        assert dod_price not in provider.symbols()

        # Its specific dependencies should be gone too
        for dep in dod_metric.required_metrics:
            assert dep not in provider.symbols()

        # But WoW should still exist with its dependencies
        assert wow_price in provider.symbols()

        # The original avg_price should still exist
        assert avg_price in provider.symbols()

    def test_remove_symbol_preserves_shared_dependencies(self, provider: MetricProvider) -> None:
        """Test that shared dependencies are not removed if still needed."""
        # Create a base metric
        avg_tax = provider.average("tax", dataset="ds1")

        # Create two metrics that share the dependency
        dod_tax = provider.ext.day_over_day(avg_tax, dataset="ds1")

        # Create another metric that uses avg_tax directly
        # First, let's manually create it to ensure it's tracked
        sum_tax = provider.sum("tax", dataset="ds1")

        # Create a metric that combines avg and sum (just to ensure tracking)
        _ = avg_tax + sum_tax  # This creates a symbolic expression

        # Remove the DoD metric (not the base average)
        provider.remove_symbol(dod_tax)

        # avg_tax should still exist because it's used elsewhere
        assert avg_tax in provider.symbols()
        assert dod_tax not in provider.symbols()

        # sum_tax should still exist
        assert sum_tax in provider.symbols()

    def test_remove_symbol_with_complex_nested_dependencies(self, provider: MetricProvider) -> None:
        """Test removing symbols with complex nested dependency chains."""
        # Create a complex chain: avg -> DoD -> stddev of DoD
        avg_tax = provider.average("tax", dataset="ds1")
        dod = provider.ext.day_over_day(avg_tax, dataset="ds1")
        stddev_of_dod = provider.ext.stddev(dod, offset=0, n=3, dataset="ds1")

        # Also create a WoW that depends on the same average
        wow = provider.ext.week_over_week(avg_tax, dataset="ds1")

        # Get initial counts
        initial_count = len(provider.metrics)

        # Get the stddev metric to check its dependencies
        stddev_metric = provider.get_symbol(stddev_of_dod)
        # Stddev should have 3 dependencies (offset=0, n=3)
        assert len(stddev_metric.required_metrics) == 3
        stddev_deps = stddev_metric.required_metrics

        # Remove the stddev metric (leaf of the chain)
        provider.remove_symbol(stddev_of_dod)

        # Stddev should be gone
        assert stddev_of_dod not in provider.symbols()

        # Its dependencies should also be removed
        for dep in stddev_deps:
            assert dep not in provider.symbols()

        # But the original metrics should still exist
        assert avg_tax in provider.symbols()
        assert dod in provider.symbols()  # DoD still exists (not removed)
        assert wow in provider.symbols()

        # We removed more than just stddev and its 3 direct dependencies
        # because the recursive removal also removes dependencies of dependencies
        final_count = len(provider.metrics)
        assert final_count < initial_count
        # At minimum we removed stddev + its 3 dependencies = 4 symbols
        assert initial_count - final_count >= 4

    def test_remove_nonexistent_symbol(self, provider: MetricProvider) -> None:
        """Test that removing a non-existent symbol raises DQXError."""
        # Create some metrics
        avg_tax = provider.average("tax", dataset="ds1")

        # Try to remove a symbol that doesn't exist
        fake_symbol = sp.Symbol("x_999")

        # This should raise a DQXError
        with pytest.raises(DQXError, match="Symbol x_999 not found"):
            provider.remove_symbol(fake_symbol)

        # Original metric should still exist
        assert avg_tax in provider.symbols()

    def test_remove_symbol_cleans_required_metrics_references(self, provider: MetricProvider) -> None:
        """Test that removing symbols also cleans up references in other metrics' required_metrics."""
        # Create base metrics
        avg_price = provider.average("price", dataset="ds1")
        avg_tax = provider.average("tax", dataset="ds1")

        # Create a metric that depends on both
        # For this test, we'll create a DoD of avg_price
        dod_price = provider.ext.day_over_day(avg_price, dataset="ds1")

        # Create another independent metric
        sum_tax = provider.sum("tax", dataset="ds1")

        # Remove avg_price and its dependents
        provider.remove_symbol(avg_price)

        # Verify that remaining metrics don't have references to removed symbols
        for metric in provider.metrics:
            assert avg_price not in metric.required_metrics
            assert dod_price not in metric.required_metrics

        # Independent metrics should still exist
        assert avg_tax in provider.symbols()
        assert sum_tax in provider.symbols()
