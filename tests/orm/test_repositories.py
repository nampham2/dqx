import datetime as dt

import pytest
from returns.maybe import Nothing, Some
from rich.console import Console

from dqx import specs, states
from dqx.common import Metadata, ResultKey
from dqx.models import Metric
from dqx.orm import repositories
from dqx.orm.repositories import InMemoryMetricDB


@pytest.fixture
def key() -> ResultKey:
    return ResultKey(yyyy_mm_dd=dt.date.fromisoformat("2025-02-04"), tags={})


@pytest.fixture
def metric_1(key: ResultKey) -> Metric:
    metric = specs.Average("page_views")
    return Metric.build(metric, key, state=states.Average(5.2, 10), dataset="test_dataset")


@pytest.fixture
def metric_window(key: ResultKey) -> list[Metric]:
    metric = specs.Average("page_views")
    return [Metric.build(metric, key.lag(_), dataset="test_dataset", state=states.Average(5.2, 10)) for _ in range(10)]


def test_crud(metric_1: Metric) -> None:
    console = Console()
    db = InMemoryMetricDB()
    metric_1 = list(db.persist([metric_1]))[0]
    assert metric_1.metric_id is not None
    console.print(metric_1)

    assert db.exists(metric_1.metric_id)
    db.delete(metric_1.metric_id)
    assert db.exists(metric_1.metric_id) is False


def test_get_metric_value(metric_1: Metric, key: ResultKey) -> None:
    db = InMemoryMetricDB()
    # Add execution_id to the metric
    execution_id = "test-exec-123"
    metric_with_exec_id = Metric.build(
        metric_1.spec,
        metric_1.key,
        dataset=metric_1.dataset,
        state=metric_1.state,
        metadata=Metadata(execution_id=execution_id),
    )
    db.persist([metric_with_exec_id])

    # Test with correct execution_id
    result = db.get_metric(specs.Average("page_views"), key, dataset="test_dataset", execution_id=execution_id)

    # Use pattern matching instead of isinstance
    match result:
        case Some(retrieved_metric):
            # Compare the retrieved metric's attributes, not the object itself
            assert retrieved_metric.spec.metric_type == metric_with_exec_id.spec.metric_type
            assert retrieved_metric.spec.parameters == metric_with_exec_id.spec.parameters
            assert retrieved_metric.key == metric_with_exec_id.key
            assert retrieved_metric.dataset == metric_with_exec_id.dataset
            assert retrieved_metric.value == metric_with_exec_id.value
            assert retrieved_metric.metadata == metric_with_exec_id.metadata
        case _:
            pytest.fail("Expected Some but got Nothing")

    # Test with wrong execution_id
    result_wrong = db.get_metric(specs.Average("page_views"), key, dataset="test_dataset", execution_id="wrong-exec-id")
    assert result_wrong == Nothing


def test_get_metric_window(metric_window: list[Metric], key: ResultKey) -> None:
    db = InMemoryMetricDB()
    execution_id = "test-exec-456"

    # Add execution_id to all metrics in the window
    metrics_with_exec_id = [
        Metric.build(
            m.spec,
            m.key,
            dataset=m.dataset,
            state=m.state,
            metadata=Metadata(execution_id=execution_id),
        )
        for m in metric_window
    ]
    db.persist(metrics_with_exec_id)

    # Test with correct execution_id
    value = db.get_metric_window(
        specs.Average("page_views"), key, lag=1, window=5, dataset="test_dataset", execution_id=execution_id
    ).unwrap()
    assert len(value) == 5
    assert min(value.keys()) == dt.date.fromisoformat("2025-01-30")
    assert max(value.keys()) == dt.date.fromisoformat("2025-02-03")

    # Check values of all metrics in the window
    for metric in value.values():
        assert metric.value == pytest.approx(5.2)

    # Test with wrong execution_id - should return empty
    value_wrong = db.get_metric_window(
        specs.Average("page_views"), key, lag=1, window=5, dataset="test_dataset", execution_id="wrong-exec-id"
    )
    assert value_wrong == Some({})  # Returns Some with empty dict


def test_get_metric_value_missing(key: ResultKey) -> None:
    """Test getting value for non-existent metric returns empty Maybe."""
    db = InMemoryMetricDB()
    spec = specs.Average("non_existent_column")
    execution_id = "test-exec-123"
    result = db.get_metric(spec, key, dataset="test_dataset", execution_id=execution_id)
    assert result == Nothing


def test_get_metric_window_missing(key: ResultKey) -> None:
    """Test getting window for non-existent metric returns Some with empty dict."""
    db = InMemoryMetricDB()
    spec = specs.Average("non_existent_column")
    execution_id = "test-exec-123"
    result = db.get_metric_window(spec, key, lag=1, window=5, dataset="test_dataset", execution_id=execution_id)
    assert result == Some({})


def test_metric_to_spec(metric_1: Metric) -> None:
    """Test that Metric.to_spec() returns the correct MetricSpec."""
    db = InMemoryMetricDB()
    persisted_metric = list(db.persist([metric_1]))[0]

    # Get the database metric directly to test to_spec method
    db_metric = db.new_session().get(repositories.Metric, persisted_metric.metric_id)
    assert db_metric is not None

    spec = db_metric.to_spec()
    assert spec.metric_type == "Average"
    assert spec.parameters == {"column": "page_views"}


def test_get_by_execution_id_basic(key: ResultKey) -> None:
    """Test retrieving metrics by execution ID."""
    db = InMemoryMetricDB()
    execution_id = "test-exec-123"

    # Create metrics with the execution ID
    metric1 = Metric.build(
        specs.Average("page_views"),
        key,
        dataset="test_dataset",
        state=states.Average(5.2, 10),
        metadata=Metadata(execution_id=execution_id),
    )
    metric2 = Metric.build(
        specs.Sum("revenue"),
        key,
        dataset="test_dataset",
        state=states.SimpleAdditiveState(100.0),
        metadata=Metadata(execution_id=execution_id),
    )

    # Persist metrics
    db.persist([metric1, metric2])

    # Retrieve by execution ID
    results = db.get_by_execution_id(execution_id)

    # Verify
    assert len(results) == 2
    assert all(m.metadata and m.metadata.execution_id == execution_id for m in results)
    metric_types = {m.spec.metric_type for m in results}
    assert metric_types == {"Average", "Sum"}


def test_get_by_execution_id_not_found() -> None:
    """Test retrieving with non-existent execution ID returns empty sequence."""
    db = InMemoryMetricDB()
    results = db.get_by_execution_id("non-existent-id")
    assert results == []


def test_get_by_execution_id_isolation(key: ResultKey) -> None:
    """Test that different execution IDs are properly isolated."""
    db = InMemoryMetricDB()
    exec_id1 = "exec-id-1"
    exec_id2 = "exec-id-2"

    # Create metrics for different execution IDs
    metric1 = Metric.build(
        specs.Average("views"),
        key,
        dataset="ds1",
        state=states.Average(10.0, 5),
        metadata=Metadata(execution_id=exec_id1),
    )
    metric2 = Metric.build(
        specs.Average("views"),
        key,
        dataset="ds2",
        state=states.Average(20.0, 5),
        metadata=Metadata(execution_id=exec_id2),
    )

    db.persist([metric1, metric2])

    # Verify isolation
    results1 = db.get_by_execution_id(exec_id1)
    results2 = db.get_by_execution_id(exec_id2)

    assert len(results1) == 1
    assert len(results2) == 1
    assert results1[0].dataset == "ds1"
    assert results2[0].dataset == "ds2"


def test_get_by_execution_id_different_ids(key: ResultKey) -> None:
    """Test that only metrics with matching execution_id are returned."""
    db = InMemoryMetricDB()
    target_id = "target-exec-id"
    other_id = "other-exec-id"

    # Create metrics with different execution IDs
    metric_with_target_id = Metric.build(
        specs.Sum("revenue"),
        key,
        dataset="ds1",
        state=states.SimpleAdditiveState(100.0),
        metadata=Metadata(execution_id=target_id),
    )
    metric_with_other_id = Metric.build(
        specs.Sum("revenue"),
        key,
        dataset="ds2",
        state=states.SimpleAdditiveState(200.0),
        metadata=Metadata(execution_id=other_id),
    )

    db.persist([metric_with_target_id, metric_with_other_id])

    # Only metric with matching execution_id should be returned
    results = db.get_by_execution_id(target_id)
    assert len(results) == 1
    assert results[0].dataset == "ds1"
    assert results[0].metadata and results[0].metadata.execution_id == target_id


# Tests for expiration functionality


def test_get_metrics_stats_empty_db() -> None:
    """Test getting stats from empty database."""
    db = InMemoryMetricDB()
    stats = db.get_metrics_stats()
    assert stats.total_metrics == 0
    assert stats.expired_metrics == 0


def test_get_metrics_stats_no_expired(key: ResultKey) -> None:
    """Test getting stats when no metrics are expired."""
    db = InMemoryMetricDB()

    # Create metrics with long TTL (won't expire)
    metric1 = Metric.build(
        specs.Average("views"),
        key,
        dataset="ds1",
        state=states.Average(10.0, 5),
        metadata=Metadata(ttl_hours=168),  # 7 days
    )
    metric2 = Metric.build(
        specs.Sum("revenue"),
        key,
        dataset="ds2",
        state=states.SimpleAdditiveState(100.0),
        metadata=Metadata(ttl_hours=168),  # 7 days
    )

    db.persist([metric1, metric2])

    stats = db.get_metrics_stats()
    assert stats.total_metrics == 2
    assert stats.expired_metrics == 0


def test_get_metrics_stats_with_expired(key: ResultKey) -> None:
    """Test getting stats when some metrics are expired."""
    from datetime import datetime, timedelta, timezone

    db = InMemoryMetricDB()

    # Create metrics with very short TTL
    metric_expired = Metric.build(
        specs.Average("views"),
        key,
        dataset="ds1",
        state=states.Average(10.0, 5),
        metadata=Metadata(ttl_hours=1),  # 1 hour TTL
    )

    # Create metric with long TTL
    metric_valid = Metric.build(
        specs.Sum("revenue"),
        key,
        dataset="ds2",
        state=states.SimpleAdditiveState(100.0),
        metadata=Metadata(ttl_hours=168),  # 7 days
    )

    # Persist metrics
    persisted = list(db.persist([metric_expired, metric_valid]))

    # Manually update the created timestamp of the first metric to be old
    with db.new_session() as session:
        db_metric = session.get(repositories.Metric, persisted[0].metric_id)
        if db_metric:
            # Set created time to 2 hours ago
            db_metric.created = datetime.now(timezone.utc) - timedelta(hours=2)
            session.commit()

    stats = db.get_metrics_stats()
    assert stats.total_metrics == 2
    assert stats.expired_metrics == 1


def test_delete_expired_metrics_empty_db() -> None:
    """Test deleting expired metrics from empty database."""
    db = InMemoryMetricDB()
    # Should not raise any errors
    db.delete_expired_metrics()

    stats = db.get_metrics_stats()
    assert stats.total_metrics == 0
    assert stats.expired_metrics == 0


def test_delete_expired_metrics(key: ResultKey) -> None:
    """Test deleting expired metrics."""
    from datetime import datetime, timedelta, timezone

    db = InMemoryMetricDB()

    # Create metrics with short TTL
    metric_expired1 = Metric.build(
        specs.Average("views"),
        key,
        dataset="ds1",
        state=states.Average(10.0, 5),
        metadata=Metadata(ttl_hours=1),  # 1 hour TTL
    )
    metric_expired2 = Metric.build(
        specs.Average("clicks"),
        key,
        dataset="ds2",
        state=states.Average(20.0, 5),
        metadata=Metadata(ttl_hours=1),  # 1 hour TTL
    )

    # Create metrics with long TTL
    metric_valid = Metric.build(
        specs.Sum("revenue"),
        key,
        dataset="ds3",
        state=states.SimpleAdditiveState(100.0),
        metadata=Metadata(ttl_hours=168),  # 7 days
    )

    persisted = list(db.persist([metric_expired1, metric_expired2, metric_valid]))
    valid_metric_id = persisted[2].metric_id

    # Manually update the created timestamps of expired metrics to be old
    with db.new_session() as session:
        for i in range(2):  # First two metrics should expire
            db_metric = session.get(repositories.Metric, persisted[i].metric_id)
            if db_metric:
                # Set created time to 2 hours ago (past the 1 hour TTL)
                db_metric.created = datetime.now(timezone.utc) - timedelta(hours=2)
        session.commit()

    # Check stats before deletion
    stats_before = db.get_metrics_stats()
    assert stats_before.total_metrics == 3
    assert stats_before.expired_metrics == 2

    # Delete expired metrics
    db.delete_expired_metrics()

    # Check stats after deletion
    stats_after = db.get_metrics_stats()
    assert stats_after.total_metrics == 1
    assert stats_after.expired_metrics == 0

    # Verify the valid metric still exists
    assert valid_metric_id is not None
    assert db.exists(valid_metric_id)


def test_metric_default_ttl(key: ResultKey) -> None:
    """Test that metrics get default TTL when metadata is not provided."""
    db = InMemoryMetricDB()

    # Create metric without explicit metadata
    metric = Metric.build(
        specs.Average("views"),
        key,
        dataset="ds1",
        state=states.Average(10.0, 5),
    )

    persisted = list(db.persist([metric]))[0]

    # Verify default TTL is applied
    assert persisted.metadata is not None
    assert persisted.metadata.ttl_hours == 168  # 7 days default


def test_expiration_with_mixed_ttl_hours(key: ResultKey) -> None:
    """Test expiration logic with various TTL values."""
    from datetime import datetime, timedelta, timezone

    db = InMemoryMetricDB()

    # Create metrics with different TTL values
    ttl_values = [1, 24, 48, 168, 720]  # 1h, 1d, 2d, 1w, 1month
    metrics = [
        Metric.build(
            specs.Average(f"metric_{i}"),
            key,
            dataset=f"ds_{i}",
            state=states.Average(float(i), 5),
            metadata=Metadata(ttl_hours=ttl),
        )
        for i, ttl in enumerate(ttl_values)
    ]

    persisted = list(db.persist(metrics))

    # Manually expire the first metric (1 hour TTL)
    with db.new_session() as session:
        db_metric = session.get(repositories.Metric, persisted[0].metric_id)
        if db_metric:
            # Set created time to 2 hours ago (past the 1 hour TTL)
            db_metric.created = datetime.now(timezone.utc) - timedelta(hours=2)
            session.commit()

    stats = db.get_metrics_stats()
    assert stats.total_metrics == 5
    assert stats.expired_metrics == 1  # Only the 1-hour TTL metric should be expired
