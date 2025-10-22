"""Tests for VerificationSuite caching functionality."""

from datetime import date

import pyarrow as pa
import pytest

from dqx.api import Context, VerificationSuite, check
from dqx.common import DQXError, ResultKey
from dqx.extensions.duckds import DuckRelationDataSource
from dqx.orm.repositories import InMemoryMetricDB
from dqx.provider import MetricProvider


class TestSuiteCaching:
    """Test suite for collect_results and collect_symbols caching."""

    def test_collect_results_returns_same_object_reference(self) -> None:
        """Multiple calls to collect_results should return the same cached object."""
        db = InMemoryMetricDB()

        @check(name="Price Check")
        def price_check(mp: MetricProvider, ctx: Context) -> None:
            price = mp.average("price")
            ctx.assert_that(price).where(name="Price is positive").is_positive()

        suite = VerificationSuite([price_check], db, "Test Suite")
        data = pa.table({"price": [10, 20, 30]})
        ds = DuckRelationDataSource.from_arrow(data)
        key = ResultKey(date.today(), {"env": "test"})

        suite.run({"test": ds}, key)

        # Get results twice
        results1 = suite.collect_results()
        results2 = suite.collect_results()

        # Should be the exact same object (not just equal)
        assert results1 is results2

    def test_collect_symbols_returns_same_object_reference(self) -> None:
        """Multiple calls to collect_symbols should return the same cached object."""
        db = InMemoryMetricDB()

        @check(name="Metric Check")
        def metric_check(mp: MetricProvider, ctx: Context) -> None:
            avg_price = mp.average("price")
            sum_quantity = mp.sum("quantity")
            ctx.assert_that(avg_price + sum_quantity).where(name="Combined metric check").is_positive()

        suite = VerificationSuite([metric_check], db, "Test Suite")
        data = pa.table({"price": [10, 20], "quantity": [5, 15]})
        ds = DuckRelationDataSource.from_arrow(data)
        key = ResultKey(date.today(), {"env": "test"})

        suite.run({"test": ds}, key)

        # Get symbols twice
        symbols1 = suite.collect_symbols()
        symbols2 = suite.collect_symbols()

        # Should be the exact same object
        assert symbols1 is symbols2

    def test_caching_works_after_successful_run(self) -> None:
        """Caching should work correctly after a successful suite run."""
        db = InMemoryMetricDB()

        @check(name="Simple Check")
        def simple_check(mp: MetricProvider, ctx: Context) -> None:
            total = mp.sum("value")
            ctx.assert_that(total).where(name="Total is 100").is_eq(100)

        suite = VerificationSuite([simple_check], db, "Test Suite")
        data = pa.table({"value": [25, 25, 25, 25]})
        ds = DuckRelationDataSource.from_arrow(data)
        key = ResultKey(date.today(), {"env": "test"})

        suite.run({"test": ds}, key)

        # First calls - should compute and cache
        results1 = suite.collect_results()
        symbols1 = suite.collect_symbols()

        # Verify we got results
        assert len(results1) == 1
        assert len(symbols1) == 1

        # Second calls - should return cached
        results2 = suite.collect_results()
        symbols2 = suite.collect_symbols()

        # Verify caching
        assert results1 is results2
        assert symbols1 is symbols2

    def test_cache_before_run_raises_error(self) -> None:
        """Attempting to access cache before run() should raise DQXError."""
        db = InMemoryMetricDB()

        @check(name="Never Run")
        def never_run(mp: MetricProvider, ctx: Context) -> None:
            pass

        suite = VerificationSuite([never_run], db, "Test Suite")

        # Should raise error before run
        with pytest.raises(DQXError, match="not been executed"):
            suite.collect_results()

        with pytest.raises(DQXError, match="not been executed"):
            suite.collect_symbols()
