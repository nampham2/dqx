import datetime as dt
from datetime import datetime, timedelta, timezone

import pyarrow as pa

from dqx.api import Context, VerificationSuite, check
from dqx.common import Metadata, ResultKey
from dqx.datasource import DuckRelationDataSource
from dqx.orm.repositories import InMemoryMetricDB
from dqx.provider import MetricProvider


def test_verification_suite_cleanup_expired_metrics() -> None:
    """Test VerificationSuite.cleanup_expired_metrics functionality."""
    db = InMemoryMetricDB()

    # Create a simple check
    @check(name="Test Check")
    def test_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.num_rows()).where(name="Count is positive").is_positive()

    # Create test data
    data = pa.Table.from_pydict({"id": [1, 2, 3]})
    datasource = DuckRelationDataSource.from_arrow(data, "test_data")

    # First run: Create metrics and modify TTL afterwards
    suite1 = VerificationSuite([test_check], db, "Suite 1")
    suite1.run([datasource], ResultKey(dt.date(2024, 10, 30), {}))

    # Get the execution_id from suite1
    execution_id_1 = suite1.execution_id

    # Manually set TTL to 1 hour for suite1's metrics
    with db._mutex:
        session = db.new_session()
        from sqlalchemy import func

        from dqx.orm.repositories import Metric as DBMetric

        # Update TTL for suite1's metrics
        session.query(DBMetric).filter(func.json_extract(DBMetric.meta, "$.execution_id") == execution_id_1).update(
            {"meta": func.json_set(DBMetric.meta, "$.ttl_hours", 1)}, synchronize_session=False
        )
        session.commit()

    # Second run: Create metrics with default TTL (168 hours)
    suite2 = VerificationSuite([test_check], db, "Suite 2")
    suite2.run([datasource], ResultKey(dt.date(2024, 10, 30), {}))

    # Verify we have metrics from both suites by checking stats
    initial_stats = db.get_metrics_stats()
    initial_count = initial_stats.total_metrics
    assert initial_count >= 2  # At least one metric per suite

    # Manually expire metrics from suite1
    with db._mutex:
        session = db.new_session()
        from dqx.orm.repositories import Metric as DBMetric

        # Find metrics from suite1 (those with ttl_hours=1)
        suite1_metrics = session.query(DBMetric).filter(DBMetric.meta["ttl_hours"].as_integer() == 1).all()

        # Update them to be 2 hours old
        old_time = datetime.now(timezone.utc) - timedelta(hours=2)
        for metric in suite1_metrics:
            metric.created = old_time
        session.commit()

    # Run cleanup (could use either suite instance)
    suite2.cleanup_expired_metrics()

    # Verify only suite2's metrics remain
    remaining_stats = db.get_metrics_stats()
    remaining_count = remaining_stats.total_metrics
    assert remaining_count < initial_count

    # Verify remaining metrics are not expired
    stats = db.get_metrics_stats()
    assert stats.expired_metrics == 0


def test_verification_suite_cleanup_no_expired() -> None:
    """Test cleanup_expired_metrics when no metrics are expired."""
    db = InMemoryMetricDB()

    @check(name="Test Check")
    def test_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.num_rows()).where(name="Count is positive").is_positive()

    # Create test data
    data = pa.Table.from_pydict({"id": [1, 2, 3]})
    datasource = DuckRelationDataSource.from_arrow(data, "test_data")

    # Run suite with default TTL (168 hours)
    suite = VerificationSuite([test_check], db, "Test Suite")
    suite.run([datasource], ResultKey(dt.date(2024, 10, 30), {}))

    # Run cleanup immediately - no metrics should be expired
    suite.cleanup_expired_metrics()

    # Verify no metrics were deleted
    stats = db.get_metrics_stats()
    assert stats.total_metrics > 0


def test_verification_suite_cleanup_mixed_ttl() -> None:
    """Test cleanup with metrics having different TTL values."""
    db = InMemoryMetricDB()

    @check(name="Test Check")
    def test_check(mp: MetricProvider, ctx: Context) -> None:
        # This will create multiple metrics
        ctx.assert_that(mp.num_rows()).where(name="Row count check").is_positive()
        ctx.assert_that(mp.sum("value")).where(name="Sum check").is_positive()
        ctx.assert_that(mp.average("value")).where(name="Avg check").is_positive()

    # Create test data
    data = pa.Table.from_pydict({"id": [1, 2, 3], "value": [10.0, 20.0, 30.0]})
    datasource = DuckRelationDataSource.from_arrow(data, "test_data")

    # Run suite
    suite = VerificationSuite([test_check], db, "Test Suite")
    suite.run([datasource], ResultKey(dt.date(2024, 10, 30), {}))

    # Get metrics count before modifications
    initial_stats = db.get_metrics_stats()
    initial_count = initial_stats.total_metrics
    assert initial_count >= 3  # At least 3 metrics from our assertions

    with db._mutex:
        session = db.new_session()
        from dqx.orm.repositories import Metric as DBMetric

        # Set different TTLs and ages for metrics - order by id for deterministic access
        metrics_in_db = session.query(DBMetric).order_by(DBMetric.metric_id).all()

        # First metric: TTL=1 hour, age=2 hours (should be deleted)
        if len(metrics_in_db) > 0:
            metrics_in_db[0].meta = Metadata(ttl_hours=1)
            metrics_in_db[0].created = datetime.now(timezone.utc) - timedelta(hours=2)

        # Second metric: TTL=168 hours (default, not expired)
        if len(metrics_in_db) > 1:
            metrics_in_db[1].meta = Metadata(ttl_hours=168)

        # Third metric: TTL=48 hours, age=1 hour (not expired)
        if len(metrics_in_db) > 2:
            metrics_in_db[2].meta = Metadata(ttl_hours=48)
            metrics_in_db[2].created = datetime.now(timezone.utc) - timedelta(hours=1)

        session.commit()

    # Run cleanup
    suite.cleanup_expired_metrics()

    # Verify remaining metrics
    final_stats = db.get_metrics_stats()
    assert final_stats.total_metrics == initial_count - 1
    assert final_stats.expired_metrics == 0
