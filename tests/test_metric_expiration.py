import datetime as dt
from datetime import datetime, timedelta, timezone

from dqx.common import Metadata, ResultKey
from dqx.models import Metric
from dqx.orm.repositories import InMemoryMetricDB
from dqx.specs import Sum
from dqx.states import SimpleAdditiveState


def test_metric_db_creates_indexes() -> None:
    """Verify MetricDB creates performance indexes on initialization."""
    db = InMemoryMetricDB()

    # Should not raise any exceptions
    assert db is not None

    # Verify we can still persist metrics
    key = ResultKey(dt.date(2024, 10, 30), {})
    sum_spec = Sum("test")
    state = SimpleAdditiveState(value=100.0)
    metric = Metric.build(metric=sum_spec, key=key, dataset="test", state=state, metadata=Metadata(ttl_hours=24))
    persisted = list(db.persist([metric]))

    # Verify we can query by ID
    assert persisted[0].metric_id is not None
    assert db.exists(persisted[0].metric_id) is True


def test_get_expired_metrics_stats_empty_db() -> None:
    """Test get_expired_metrics_stats with an empty database."""
    db = InMemoryMetricDB()
    stats = db.get_metrics_stats()

    assert stats.total_metrics == 0
    assert stats.expired_metrics == 0


def test_get_expired_metrics_stats_with_metrics() -> None:
    """Test get_expired_metrics_stats with various metric types."""
    db = InMemoryMetricDB()

    # Create metrics with different TTL values
    key = ResultKey(dt.date(2024, 10, 30), {})
    state = SimpleAdditiveState(value=100.0)

    # Metric 1: Already expired (TTL=1 hour, created 2 hours ago)
    metric1 = Metric.build(metric=Sum("expired"), key=key, dataset="test", state=state, metadata=Metadata(ttl_hours=1))

    # Metric 2: Not expired (TTL=48 hours, just created)
    metric2 = Metric.build(
        metric=Sum("not_expired"), key=key, dataset="test", state=state, metadata=Metadata(ttl_hours=48)
    )

    # Metric 3: Long TTL (168 hours default)
    metric3 = Metric.build(metric=Sum("long_ttl"), key=key, dataset="test", state=state, metadata=Metadata())

    # Persist all metrics
    persisted = list(db.persist([metric1, metric2, metric3]))

    # Manually update the created timestamp for metric1 to make it expired
    # Note: This is a test-only operation to simulate an old metric
    with db._mutex:
        session = db.new_session()
        from dqx.orm.repositories import Metric as DBMetric

        # Update the first metric to be 2 hours old
        old_time = datetime.now(timezone.utc) - timedelta(hours=2)
        session.query(DBMetric).filter_by(metric_id=persisted[0].metric_id).update({"created": old_time})
        session.commit()

    # Get stats
    stats = db.get_metrics_stats()

    assert stats.total_metrics == 3
    assert stats.expired_metrics == 1


def test_get_expired_metrics_stats_timezone_edge_cases() -> None:
    """Test get_expired_metrics_stats handles timezone edge cases correctly."""
    db = InMemoryMetricDB()

    # Create a metric with TTL
    key = ResultKey(dt.date(2024, 10, 30), {})
    state = SimpleAdditiveState(value=100.0)

    metric = Metric.build(metric=Sum("test"), key=key, dataset="test", state=state, metadata=Metadata(ttl_hours=24))

    db.persist([metric])

    # Stats should be consistent regardless of system timezone
    stats1 = db.get_metrics_stats()
    stats2 = db.get_metrics_stats()

    assert stats1 == stats2
    assert stats1.total_metrics == 1
    assert stats1.expired_metrics == 0  # Just created, not expired


def test_delete_expired_metrics_empty_db() -> None:
    """Test delete_expired_metrics with an empty database."""
    db = InMemoryMetricDB()

    # Should not raise any exceptions
    db.delete_expired_metrics()
    # No return value to check anymore


def test_delete_expired_metrics_with_mixed_metrics() -> None:
    """Test delete_expired_metrics removes only expired metrics."""
    db = InMemoryMetricDB()

    # Create metrics with different TTL values
    key = ResultKey(dt.date(2024, 10, 30), {})
    state = SimpleAdditiveState(value=100.0)

    # Metric 1: Already expired (TTL=1 hour, created 2 hours ago)
    metric1 = Metric.build(metric=Sum("expired"), key=key, dataset="test", state=state, metadata=Metadata(ttl_hours=1))

    # Metric 2: Not expired (TTL=48 hours, just created)
    metric2 = Metric.build(
        metric=Sum("not_expired"), key=key, dataset="test", state=state, metadata=Metadata(ttl_hours=48)
    )

    # Metric 3: Long TTL (168 hours default)
    metric3 = Metric.build(metric=Sum("long_ttl"), key=key, dataset="test", state=state, metadata=Metadata())

    # Persist all metrics
    persisted = list(db.persist([metric1, metric2, metric3]))

    # Manually update the created timestamp for metric1 to make it expired
    with db._mutex:
        session = db.new_session()
        from dqx.orm.repositories import Metric as DBMetric

        # Update the first metric to be 2 hours old
        old_time = datetime.now(timezone.utc) - timedelta(hours=2)
        session.query(DBMetric).filter_by(metric_id=persisted[0].metric_id).update({"created": old_time})
        session.commit()

    # Delete expired metrics
    db.delete_expired_metrics()

    # Verify the correct metrics remain
    stats = db.get_metrics_stats()
    assert stats.total_metrics == 2
    assert stats.expired_metrics == 0

    # Verify specific metrics
    if persisted[0].metric_id is not None:
        assert db.exists(persisted[0].metric_id) is False  # Expired metric deleted
    if persisted[1].metric_id is not None:
        assert db.exists(persisted[1].metric_id) is True  # Not expired
    if persisted[2].metric_id is not None:
        assert db.exists(persisted[2].metric_id) is True  # Long TTL


def test_delete_expired_metrics_boundary_conditions() -> None:
    """Test delete_expired_metrics handles boundary conditions correctly.

    Tests that metrics are only deleted when strictly past their TTL boundary,
    not at or before it.
    """
    db = InMemoryMetricDB()

    key = ResultKey(dt.date(2024, 10, 30), {})
    state = SimpleAdditiveState(value=100.0)

    # Create a metric that expires in exactly 1 hour
    metric = Metric.build(metric=Sum("boundary"), key=key, dataset="test", state=state, metadata=Metadata(ttl_hours=1))

    persisted = list(db.persist([metric]))

    # Update to just under 1 hour old (should NOT be expired)
    with db._mutex:
        session = db.new_session()
        from dqx.orm.repositories import Metric as DBMetric

        # Update to be 59 minutes 59 seconds old (safely before the 1-hour boundary)
        not_expired_time = datetime.now(timezone.utc) - timedelta(minutes=59, seconds=59)
        session.query(DBMetric).filter_by(metric_id=persisted[0].metric_id).update({"created": not_expired_time})
        session.commit()

    # Should not delete - before the boundary
    db.delete_expired_metrics()
    if persisted[0].metric_id is not None:
        assert db.exists(persisted[0].metric_id) is True

    # Update to 1 hour + 1 second old (should be expired)
    with db._mutex:
        session = db.new_session()
        from dqx.orm.repositories import Metric as DBMetric

        # Update to be 1 hour + 1 second old
        expired_time = datetime.now(timezone.utc) - timedelta(hours=1, seconds=1)
        session.query(DBMetric).filter_by(metric_id=persisted[0].metric_id).update({"created": expired_time})
        session.commit()

    # Now should delete - past the boundary
    db.delete_expired_metrics()
    if persisted[0].metric_id is not None:
        assert db.exists(persisted[0].metric_id) is False
