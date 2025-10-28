"""Tests for metrics_by_execution_id functionality."""

from datetime import date

import pyarrow as pa

from dqx.api import Context, MetricProvider, VerificationSuite, check
from dqx.common import ResultKey
from dqx.datasource import DuckRelationDataSource
from dqx.orm.repositories import InMemoryMetricDB


def test_metrics_by_execution_id_basic() -> None:
    """Test basic retrieval of metrics by execution ID."""

    # Create a check that uses multiple metrics
    @check(name="Multi-metric Check")
    def multi_check(mp: MetricProvider, ctx: Context) -> None:
        avg_price = mp.average("price")
        min_price = mp.minimum("price")
        max_price = mp.maximum("price")

        ctx.assert_that(avg_price).where(name="Avg price check").is_positive()
        ctx.assert_that(min_price).where(name="Min price check").is_positive()
        ctx.assert_that(max_price).where(name="Max price check").is_gt(10)

    # Create test data
    table = pa.table({"price": [10.0, 20.0, 30.0]})
    datasource = DuckRelationDataSource.from_arrow(table, "test_data")

    # Run the suite
    db = InMemoryMetricDB()
    suite = VerificationSuite([multi_check], db, "Test Suite")
    key = ResultKey(date.today(), {"env": "test"})
    suite.run([datasource], key)

    # Get the execution ID
    execution_id = suite.execution_id

    # Retrieve metrics by execution ID
    metrics = db.get_by_execution_id(execution_id)

    # Verify we got all metrics
    assert len(metrics) > 0

    # All metrics should have the same execution_id in metadata
    for metric in metrics:
        assert metric.metadata is not None
        assert metric.metadata.execution_id == execution_id

    # Check that we got the expected metrics (at least average, min, max)
    metric_types = {m.spec.metric_type for m in metrics}
    assert "Average" in metric_types
    assert "Minimum" in metric_types
    assert "Maximum" in metric_types


def test_metrics_by_execution_id_with_extended_ops() -> None:
    """Test retrieval includes lagged metrics from extended operations."""

    # Create a check that uses extended metrics
    @check(name="Extended Check")
    def extended_check(mp: MetricProvider, ctx: Context) -> None:
        avg_price = mp.average("price")
        dod_avg = mp.ext.day_over_day(avg_price)

        ctx.assert_that(dod_avg).where(name="DoD check").noop()

    # Create test data for multiple days
    table1 = pa.table({"price": [10.0, 20.0, 30.0]})
    table2 = pa.table({"price": [15.0, 25.0, 35.0]})

    datasource1 = DuckRelationDataSource.from_arrow(table1, "test_data")
    datasource2 = DuckRelationDataSource.from_arrow(table2, "test_data")

    # Run for day 1
    db = InMemoryMetricDB()
    suite1 = VerificationSuite([extended_check], db, "Test Suite")
    key1 = ResultKey(date(2024, 1, 1), {"env": "prod"})
    suite1.run([datasource1], key1)

    # Run for day 2
    suite2 = VerificationSuite([extended_check], db, "Test Suite")
    key2 = ResultKey(date(2024, 1, 2), {"env": "prod"})
    suite2.run([datasource2], key2)

    # Get metrics for the second execution
    execution_id = suite2.execution_id
    metrics = db.get_by_execution_id(execution_id)

    # Should have metrics including the lagged one
    assert len(metrics) > 0

    # Check dates - should have metrics from both current and lag date
    dates = {m.key.yyyy_mm_dd for m in metrics}
    assert date(2024, 1, 2) in dates  # Current date
    assert date(2024, 1, 1) in dates  # Lag date

    # All should have the same execution_id in metadata
    for metric in metrics:
        assert metric.metadata is not None
        assert metric.metadata.execution_id == execution_id


def test_metrics_by_execution_id_not_found() -> None:
    """Test behavior when execution ID doesn't exist."""
    db = InMemoryMetricDB()

    # Try to get metrics for non-existent execution ID
    metrics = db.get_by_execution_id("non-existent-id")

    # Should return empty list
    assert metrics == []


def test_metrics_by_execution_id_isolation() -> None:
    """Test that different executions are properly isolated."""

    @check(name="Simple Check")
    def simple_check(mp: MetricProvider, ctx: Context) -> None:
        avg = mp.average("value")
        ctx.assert_that(avg).where(name="Avg check").is_positive()

    # Create test data
    table = pa.table({"value": [1.0, 2.0, 3.0]})
    datasource = DuckRelationDataSource.from_arrow(table, "test_data")

    db = InMemoryMetricDB()

    # Run suite 1
    suite1 = VerificationSuite([simple_check], db, "Suite 1")
    key1 = ResultKey(date.today(), {"env": "test1"})
    suite1.run([datasource], key1)
    exec_id1 = suite1.execution_id

    # Run suite 2
    suite2 = VerificationSuite([simple_check], db, "Suite 2")
    key2 = ResultKey(date.today(), {"env": "test2"})
    suite2.run([datasource], key2)
    exec_id2 = suite2.execution_id

    # Get metrics for each execution
    metrics1 = db.get_by_execution_id(exec_id1)
    metrics2 = db.get_by_execution_id(exec_id2)

    # Each should only have its own metrics
    assert all(m.metadata is not None and m.metadata.execution_id == exec_id1 for m in metrics1)
    assert all(m.metadata is not None and m.metadata.execution_id == exec_id2 for m in metrics2)

    # Should have different base tags
    assert all(m.key.tags["env"] == "test1" for m in metrics1)
    assert all(m.key.tags["env"] == "test2" for m in metrics2)
