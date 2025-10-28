"""Integration tests for metadata functionality."""

from datetime import date

import pyarrow as pa

from dqx import data
from dqx.api import Context, MetricProvider, VerificationSuite, check
from dqx.common import Metadata, ResultKey
from dqx.datasource import DuckRelationDataSource
from dqx.orm.repositories import InMemoryMetricDB


def test_metadata_persistence_flow() -> None:
    """Test complete metadata flow from VerificationSuite to persistence."""

    @check(name="Metadata Test Check")
    def test_check(mp: MetricProvider, ctx: Context) -> None:
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
    suite = VerificationSuite([test_check], db, "Test Suite")
    key = ResultKey(date.today(), {"env": "test"})
    suite.run([datasource], key)

    # Get the execution ID
    execution_id = suite.execution_id

    # Retrieve metrics by execution ID
    metrics = data.metrics_by_execution_id(db, execution_id)

    # Verify we got all metrics
    assert len(metrics) == 3  # average, minimum, maximum

    # Verify all metrics have correct metadata
    for metric in metrics:
        assert metric.metadata is not None
        assert isinstance(metric.metadata, Metadata)
        assert metric.metadata.execution_id == execution_id
        assert metric.metadata.ttl_hours == 168  # default value

    # Verify metric types
    metric_types = {m.spec.metric_type for m in metrics}
    assert metric_types == {"Average", "Minimum", "Maximum"}


def test_metadata_with_custom_ttl() -> None:
    """Test that custom TTL can be set in metadata."""
    from dqx.analyzer import Analyzer
    from dqx.specs import Average

    # Create analyzer with custom metadata
    custom_metadata = Metadata(execution_id="test-123", ttl_hours=24)
    analyzer = Analyzer(metadata=custom_metadata)

    # Create test data
    table = pa.table({"value": [1.0, 2.0, 3.0]})
    datasource = DuckRelationDataSource.from_arrow(table, "test_data")

    # Analyze
    metrics_by_date = {ResultKey(date.today(), {"env": "test"}): [Average("value")]}
    analyzer.analyze(datasource, metrics_by_date)

    # Persist
    db = InMemoryMetricDB()
    analyzer.report.persist(db)

    # Retrieve and verify
    metrics = data.metrics_by_execution_id(db, "test-123")
    assert len(metrics) == 1
    assert metrics[0].metadata is not None
    assert metrics[0].metadata.ttl_hours == 24


def test_metadata_isolation_between_suites() -> None:
    """Test that metadata is properly isolated between different suite executions."""

    @check(name="Simple Check")
    def simple_check(mp: MetricProvider, ctx: Context) -> None:
        avg = mp.average("value")
        ctx.assert_that(avg).where(name="Value check").is_positive()

    # Create test data
    table = pa.table({"value": [1.0, 2.0, 3.0]})
    datasource = DuckRelationDataSource.from_arrow(table, "test_data")

    db = InMemoryMetricDB()

    # Run first suite
    suite1 = VerificationSuite([simple_check], db, "Suite 1")
    key1 = ResultKey(date.today(), {"env": "test1"})
    suite1.run([datasource], key1)
    exec_id1 = suite1.execution_id

    # Run second suite
    suite2 = VerificationSuite([simple_check], db, "Suite 2")
    key2 = ResultKey(date.today(), {"env": "test2"})
    suite2.run([datasource], key2)
    exec_id2 = suite2.execution_id

    # Retrieve metrics for each execution
    metrics1 = data.metrics_by_execution_id(db, exec_id1)
    metrics2 = data.metrics_by_execution_id(db, exec_id2)

    # Verify isolation
    assert len(metrics1) == 1
    assert len(metrics2) == 1
    assert metrics1[0].metadata is not None
    assert metrics1[0].metadata.execution_id == exec_id1
    assert metrics2[0].metadata is not None
    assert metrics2[0].metadata.execution_id == exec_id2
    assert metrics1[0].key.tags["env"] == "test1"
    assert metrics2[0].key.tags["env"] == "test2"


def test_no_execution_id_tag_injection() -> None:
    """Verify that __execution_id is no longer injected into tags."""

    @check(name="Tag Check")
    def tag_check(mp: MetricProvider, ctx: Context) -> None:
        avg = mp.average("value")
        ctx.assert_that(avg).where(name="Value check").is_positive()

    # Create test data
    table = pa.table({"value": [1.0, 2.0, 3.0]})
    datasource = DuckRelationDataSource.from_arrow(table, "test_data")

    # Run suite
    db = InMemoryMetricDB()
    suite = VerificationSuite([tag_check], db, "Test Suite")
    key = ResultKey(date.today(), {"env": "test", "custom": "tag"})
    suite.run([datasource], key)

    # Query all metrics
    from dqx.orm.repositories import Metric as DBMetric

    all_metrics = db.search(DBMetric.yyyy_mm_dd == date.today())

    # Verify no __execution_id in tags
    for metric in all_metrics:
        assert "__execution_id" not in metric.key.tags
        # But it should be in metadata
        assert metric.metadata is not None
        assert metric.metadata.execution_id == suite.execution_id
