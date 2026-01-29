"""Integration test for execution_id flow through the entire system."""

import datetime
import uuid

import pyarrow as pa

from dqx.api import Context, MetricProvider, VerificationSuite, check
from dqx.common import ResultKey
from dqx.datasource import DuckRelationDataSource
from dqx.orm.repositories import InMemoryMetricDB


def test_execution_id_end_to_end() -> None:
    """Test that execution_id flows through the entire system correctly."""
    # Create database
    db = InMemoryMetricDB()

    # Create a check that uses various metrics
    @check(name="Data Quality Check", datasets=["sales"])
    def quality_check(mp: MetricProvider, ctx: Context) -> None:
        # Basic metrics
        ctx.assert_that(mp.num_rows()).config(name="Has data").is_gt(0)
        ctx.assert_that(mp.average("amount")).config(name="Average amount is positive").is_positive()
        ctx.assert_that(mp.null_count("customer_id")).config(name="No null customer IDs").is_eq(0)

        # Extended metrics
        avg_amount = mp.average("amount")
        dod = mp.ext.day_over_day(avg_amount)
        ctx.assert_that(dod).config(name="Day over day change is reasonable").is_between(-0.5, 0.5)

    # Create suite
    suite = VerificationSuite([quality_check], db, "Sales Quality Suite")

    # Capture execution_id
    execution_id = suite.execution_id

    # Verify it's a valid UUID
    assert execution_id is not None
    assert len(execution_id) == 36  # Standard UUID length
    # Should not raise ValueError
    uuid.UUID(execution_id)

    # Create test data
    data = pa.table(
        {
            "amount": [100.0, 200.0, 150.0, 300.0, 250.0],
            "customer_id": ["C1", "C2", "C3", "C4", "C5"],
            "date": [datetime.date.today()] * 5,
        }
    )
    datasource = DuckRelationDataSource.from_arrow(data, "sales")

    # Run the suite
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={"env": "test"})
    suite.run([datasource], key)

    # Verify metrics were persisted with correct execution_id
    metrics = db.get_by_execution_id(execution_id)
    assert len(metrics) > 0

    # Check that all metrics have the same execution_id
    for metric in metrics:
        assert metric.metadata is not None
        assert metric.metadata.execution_id == execution_id

    # Verify we can retrieve metrics by execution_id
    metric_names = {m.spec.name for m in metrics}
    assert "num_rows()" in metric_names
    assert "average(amount)" in metric_names
    assert "null_count(customer_id)" in metric_names

    # Verify the provider has the execution_id
    assert suite.provider.execution_id == execution_id

    # Verify extended provider also has access to execution_id
    assert suite.provider.ext.execution_id == execution_id

    # Verify analysis reports contain metrics with correct execution_id
    report = suite._analysis_reports
    assert len(report) > 0  # Should have metrics in the report
    # Check that metrics in the report have the correct execution_id
    for metric in report.values():
        assert metric.metadata is not None
        assert metric.metadata.execution_id == execution_id

    # Get metric trace and verify
    trace = suite.metric_trace(db)
    assert trace.num_rows > 0

    # All rows in trace should relate to this execution
    # (The trace is already filtered by execution_id internally)


def test_multiple_suites_different_execution_ids() -> None:
    """Test that different suites have different execution IDs."""
    db = InMemoryMetricDB()

    @check(name="Simple Check", datasets=["data"])
    def simple_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.num_rows()).config(name="Has rows").is_gt(0)

    # Create two suites
    suite1 = VerificationSuite([simple_check], db, "Suite 1")
    suite2 = VerificationSuite([simple_check], db, "Suite 2")

    # They should have different execution IDs
    assert suite1.execution_id != suite2.execution_id

    # Both should be valid UUIDs
    uuid.UUID(suite1.execution_id)
    uuid.UUID(suite2.execution_id)

    # Create test data
    data = pa.table({"x": [1, 2, 3]})
    datasource = DuckRelationDataSource.from_arrow(data, "data")
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})

    # Run both suites
    suite1.run([datasource], key)
    suite2.run([datasource], key)

    # Verify metrics are stored separately
    metrics1 = db.get_by_execution_id(suite1.execution_id)
    metrics2 = db.get_by_execution_id(suite2.execution_id)

    assert len(metrics1) > 0
    assert len(metrics2) > 0

    # No overlap in execution IDs
    for m1 in metrics1:
        assert m1.metadata is not None
        assert m1.metadata.execution_id == suite1.execution_id
        assert m1.metadata.execution_id != suite2.execution_id

    for m2 in metrics2:
        assert m2.metadata is not None
        assert m2.metadata.execution_id == suite2.execution_id
        assert m2.metadata.execution_id != suite1.execution_id


def test_execution_id_in_lazy_retrieval() -> None:
    """Test that execution_id is properly used in lazy retrieval functions."""
    db = InMemoryMetricDB()

    @check(name="Lazy Check", datasets=["test_data"])
    def lazy_check(mp: MetricProvider, ctx: Context) -> None:
        # Create metrics that will be evaluated lazily
        avg_price = mp.average("price", dataset="test_data")
        max_price = mp.maximum("price", dataset="test_data")

        # Extended metrics that depend on base metrics
        dod = mp.ext.day_over_day(avg_price)
        wow = mp.ext.week_over_week(max_price)

        ctx.assert_that(avg_price).config(name="Average price check").is_gt(0)
        ctx.assert_that(dod).config(name="DoD check").is_between(-1, 1)
        ctx.assert_that(wow).config(name="WoW check").is_between(-1, 1)

    suite = VerificationSuite([lazy_check], db, "Lazy Test Suite")
    execution_id = suite.execution_id

    # Create test data
    data = pa.table({"price": [10.0, 20.0, 30.0, 40.0, 50.0], "date": [datetime.date.today()] * 5})
    datasource = DuckRelationDataSource.from_arrow(data, "test_data")

    # Run the suite
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})
    suite.run([datasource], key)

    # Verify all metrics (including extended) have correct execution_id
    metrics = db.get_by_execution_id(execution_id)
    metric_names = {m.spec.name for m in metrics}

    # Should include base metrics
    assert "average(price)" in metric_names
    assert "maximum(price)" in metric_names

    # All should have the same execution_id
    for metric in metrics:
        assert metric.metadata is not None
        assert metric.metadata.execution_id == execution_id


def test_execution_id_with_multiple_datasets() -> None:
    """Test execution_id consistency across multiple datasets."""
    db = InMemoryMetricDB()

    @check(name="Orders Check", datasets=["orders"])
    def orders_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.num_rows()).config(name="Orders exist").is_gt(0)
        ctx.assert_that(mp.sum("total")).config(name="Total is positive").is_positive()

    @check(name="Customers Check", datasets=["customers"])
    def customers_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.num_rows()).config(name="Customers exist").is_gt(0)
        ctx.assert_that(mp.unique_count("customer_id")).config(name="Has unique customers").is_gt(0)

    suite = VerificationSuite([orders_check, customers_check], db, "Multi-Dataset Suite")
    execution_id = suite.execution_id

    # Create test data
    orders_data = pa.table({"order_id": [1, 2, 3], "total": [100.0, 200.0, 300.0]})
    customers_data = pa.table(
        {
            "customer_id": ["C1", "C2", "C3", "C1"],  # C1 is duplicate
            "name": ["Alice", "Bob", "Charlie", "Alice"],
        }
    )

    datasources = [
        DuckRelationDataSource.from_arrow(orders_data, "orders"),
        DuckRelationDataSource.from_arrow(customers_data, "customers"),
    ]

    # Run the suite
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={"test": "multi"})
    from dqx.common import SqlDataSource

    datasources_typed: list[SqlDataSource] = datasources  # type: ignore[assignment]
    suite.run(datasources_typed, key)

    # Verify all metrics across all datasets have the same execution_id
    metrics = db.get_by_execution_id(execution_id)

    # Group by dataset
    from dqx.models import Metric

    metrics_by_dataset: dict[str, list[Metric]] = {}
    for metric in metrics:
        if metric.dataset not in metrics_by_dataset:
            metrics_by_dataset[metric.dataset] = []
        metrics_by_dataset[metric.dataset].append(metric)

    # Should have metrics for both datasets
    assert "orders" in metrics_by_dataset
    assert "customers" in metrics_by_dataset

    # All metrics should have the same execution_id
    for dataset_metrics in metrics_by_dataset.values():
        for metric in dataset_metrics:
            assert metric.metadata is not None
            assert metric.metadata.execution_id == execution_id

    # Verify specific metrics
    orders_metrics = {m.spec.name for m in metrics_by_dataset["orders"]}
    assert "num_rows()" in orders_metrics
    assert "sum(total)" in orders_metrics

    customers_metrics = {m.spec.name for m in metrics_by_dataset["customers"]}
    assert "num_rows()" in customers_metrics
    assert "unique_count(customer_id)" in customers_metrics
