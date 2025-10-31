"""Test topological sorting functionality in MetricRegistry."""

import pytest
import sympy as sp
from returns.result import Success

from dqx import specs
from dqx.common import DQXError, ExecutionId
from dqx.orm.repositories import InMemoryMetricDB
from dqx.provider import MetricProvider, SymbolicMetric


class TestMetricRegistryTopologicalSort:
    """Test topological sorting functionality in MetricRegistry."""

    @pytest.fixture
    def provider(self) -> MetricProvider:
        """Create a MetricProvider with in-memory database."""
        db = InMemoryMetricDB()
        return MetricProvider(db, ExecutionId("test-exec-123"))

    def test_topological_sort_empty_registry(self, provider: MetricProvider) -> None:
        """Empty registry should remain empty after sort."""
        provider.registry.topological_sort()
        assert provider.registry.metrics == []

    def test_topological_sort_single_metric(self, provider: MetricProvider) -> None:
        """Single metric should remain in place."""
        metric = provider.average("price", dataset="sales")
        provider.registry.topological_sort()

        assert len(provider.registry.metrics) == 1
        assert provider.registry.metrics[0].symbol == metric

    def test_topological_sort_independent_metrics(self, provider: MetricProvider) -> None:
        """Independent metrics should maintain their original order."""
        m1 = provider.sum("revenue", dataset="sales")
        m2 = provider.average("price", dataset="sales")
        m3 = provider.num_rows(dataset="sales")

        provider.registry.topological_sort()

        # Order should be preserved for independent metrics
        symbols = [m.symbol for m in provider.registry.metrics]
        assert symbols == [m1, m2, m3]

    def test_topological_sort_linear_dependency(self, provider: MetricProvider) -> None:
        """Test linear dependency: base → dod."""
        base = provider.average("price", dataset="sales")
        dod = provider.ext.day_over_day(base, dataset="sales")

        provider.registry.topological_sort()

        # Base must come before dod
        symbols = [m.symbol for m in provider.registry.metrics]
        assert symbols.index(base) < symbols.index(dod)

    def test_topological_sort_complex_dependencies(self, provider: MetricProvider) -> None:
        """Test complex multi-level dependencies."""
        # Create base metrics
        sum_revenue = provider.sum("revenue", dataset="sales")
        avg_price = provider.average("price", dataset="sales")

        # Create extended metrics
        dod_sum = provider.ext.day_over_day(sum_revenue, dataset="sales")
        wow_avg = provider.ext.week_over_week(avg_price, dataset="sales")

        # Create stddev of a base metric
        stddev_price = provider.ext.stddev(avg_price, offset=0, n=5, dataset="sales")

        provider.registry.topological_sort()

        # Verify order
        symbols = [m.symbol for m in provider.registry.metrics]

        # Base metrics should come before extended metrics
        assert symbols.index(sum_revenue) < symbols.index(dod_sum)
        assert symbols.index(avg_price) < symbols.index(wow_avg)
        assert symbols.index(avg_price) < symbols.index(stddev_price)

    def test_topological_sort_direct_cycle_detection(self, provider: MetricProvider) -> None:
        """Test detection of direct circular dependency A → B, B → A."""
        # Create metrics manually to introduce cycle
        sym_a = provider.registry._next_symbol()
        sym_b = provider.registry._next_symbol()

        metric_a = SymbolicMetric(
            name="metric_a",
            symbol=sym_a,
            fn=lambda key: Success(1.0),
            metric_spec=specs.NumRows(),
            required_metrics=[sym_b],  # A depends on B
        )

        metric_b = SymbolicMetric(
            name="metric_b",
            symbol=sym_b,
            fn=lambda key: Success(2.0),
            metric_spec=specs.NumRows(),
            required_metrics=[sym_a],  # B depends on A - cycle!
        )

        provider.registry._metrics.extend([metric_a, metric_b])
        provider.registry._symbol_index[sym_a] = metric_a
        provider.registry._symbol_index[sym_b] = metric_b

        with pytest.raises(DQXError, match="Circular dependency detected"):
            provider.registry.topological_sort()

    def test_topological_sort_indirect_cycle_detection(self, provider: MetricProvider) -> None:
        """Test detection of indirect circular dependency A → B → C → A."""
        # Create metrics manually to introduce cycle
        sym_a = provider.registry._next_symbol()
        sym_b = provider.registry._next_symbol()
        sym_c = provider.registry._next_symbol()

        metric_a = SymbolicMetric(
            name="metric_a",
            symbol=sym_a,
            fn=lambda key: Success(1.0),
            metric_spec=specs.NumRows(),
            required_metrics=[sym_b],  # A depends on B
        )

        metric_b = SymbolicMetric(
            name="metric_b",
            symbol=sym_b,
            fn=lambda key: Success(2.0),
            metric_spec=specs.NumRows(),
            required_metrics=[sym_c],  # B depends on C
        )

        metric_c = SymbolicMetric(
            name="metric_c",
            symbol=sym_c,
            fn=lambda key: Success(3.0),
            metric_spec=specs.NumRows(),
            required_metrics=[sym_a],  # C depends on A - cycle!
        )

        provider.registry._metrics.extend([metric_a, metric_b, metric_c])
        provider.registry._symbol_index[sym_a] = metric_a
        provider.registry._symbol_index[sym_b] = metric_b
        provider.registry._symbol_index[sym_c] = metric_c

        with pytest.raises(DQXError) as exc_info:
            provider.registry.topological_sort()

        error_msg = str(exc_info.value)
        assert "Circular dependency detected" in error_msg
        # Should show all metrics involved in the cycle
        assert "metric_a" in error_msg
        assert "metric_b" in error_msg
        assert "metric_c" in error_msg

    def test_topological_sort_with_external_dependencies(self, provider: MetricProvider) -> None:
        """Test metrics with dependencies not in the registry."""
        # Create a metric that depends on a non-existent symbol
        external_symbol = sp.Symbol("external_1")
        sym_a = provider.registry._next_symbol()

        metric_a = SymbolicMetric(
            name="metric_a",
            symbol=sym_a,
            fn=lambda key: Success(1.0),
            metric_spec=specs.NumRows(),
            required_metrics=[external_symbol],  # Depends on external symbol
        )

        provider.registry._metrics.append(metric_a)
        provider.registry._symbol_index[sym_a] = metric_a

        # Should not raise error - external dependencies are ignored
        provider.registry.topological_sort()

        # Metric should still be in the registry
        assert len(provider.registry.metrics) == 1
        assert provider.registry.metrics[0].symbol == sym_a

    def test_topological_sort_preserves_functionality(self, provider: MetricProvider) -> None:
        """Test that sorted metrics still function correctly."""
        # Create a complex dependency graph
        base1 = provider.average("price", dataset="sales")
        base2 = provider.sum("quantity", dataset="sales")
        dod1 = provider.ext.day_over_day(base1, dataset="sales")
        wow2 = provider.ext.week_over_week(base2, dataset="sales")

        # Sort
        provider.registry.topological_sort()

        # Verify all metrics are still accessible
        assert provider.get_symbol(base1) is not None
        assert provider.get_symbol(base2) is not None
        assert provider.get_symbol(dod1) is not None
        assert provider.get_symbol(wow2) is not None

        # Verify dependencies are still correct
        dod_metric = provider.get_symbol(dod1)
        assert len(dod_metric.required_metrics) == 2  # lag+0 and lag+1

    def test_topological_sort_with_shared_dependencies(self, provider: MetricProvider) -> None:
        """Test multiple metrics depending on the same base metric."""
        # One base metric used by multiple extended metrics
        base = provider.average("price", dataset="sales")

        # Multiple metrics depend on the same base
        dod = provider.ext.day_over_day(base, dataset="sales")
        wow = provider.ext.week_over_week(base, dataset="sales")
        stddev = provider.ext.stddev(base, offset=0, n=7, dataset="sales")

        provider.registry.topological_sort()

        # Base should come first
        symbols = [m.symbol for m in provider.registry.metrics]
        base_idx = symbols.index(base)

        # All extended metrics should come after base
        assert base_idx < symbols.index(dod)
        assert base_idx < symbols.index(wow)
        assert base_idx < symbols.index(stddev)

    def test_topological_sort_idempotent(self, provider: MetricProvider) -> None:
        """Test that sorting multiple times produces the same result."""
        # Create metrics
        base = provider.average("price", dataset="sales")
        provider.ext.day_over_day(base, dataset="sales")
        provider.sum("quantity", dataset="sales")

        # Sort once
        provider.registry.topological_sort()
        first_order = [m.symbol for m in provider.registry.metrics]

        # Sort again
        provider.registry.topological_sort()
        second_order = [m.symbol for m in provider.registry.metrics]

        # Should be the same
        assert first_order == second_order

    def assert_topological_order(self, provider: MetricProvider) -> None:
        """Helper to verify topological order is maintained."""
        seen: set[sp.Symbol] = set()

        for metric in provider.registry.metrics:
            # All required metrics should have been seen already
            for req in metric.required_metrics:
                if req in provider.registry._symbol_index:  # Only check if in registry
                    assert req in seen, f"{metric.symbol} depends on {req} which hasn't appeared yet"
            seen.add(metric.symbol)

    def test_topological_sort_maintains_order_invariant(self, provider: MetricProvider) -> None:
        """Test that topological order invariant is maintained after sort."""
        # Create a complex graph
        provider.num_rows(dataset="sales")
        m2 = provider.average("price", dataset="sales")
        m3 = provider.sum("quantity", dataset="sales")

        dod1 = provider.ext.day_over_day(m2, dataset="sales")
        wow1 = provider.ext.week_over_week(m3, dataset="sales")

        # Create a metric that depends on both extended metrics
        sym_combined = provider.registry._next_symbol()
        combined = SymbolicMetric(
            name="combined_metric",
            symbol=sym_combined,
            fn=lambda key: Success(1.0),
            metric_spec=specs.NumRows(),
            required_metrics=[dod1, wow1],
        )
        provider.registry._metrics.append(combined)
        provider.registry._symbol_index[sym_combined] = combined

        # Sort
        provider.registry.topological_sort()

        # Verify the invariant
        self.assert_topological_order(provider)
