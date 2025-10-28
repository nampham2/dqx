import datetime as dt

import pytest
from returns.maybe import Nothing, Some
from rich.console import Console

from dqx import specs, states
from dqx.common import DQXError, Metadata, ResultKey
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
    console.print(db.get(metric_1.metric_id))
    assert db.get(metric_1.metric_id) == Some(metric_1)
    assert db.search(repositories.Metric.metric_id == metric_1.metric_id) == [metric_1]
    db.delete(metric_1.metric_id)
    assert db.exists(metric_1.metric_id) is False
    assert db.search(repositories.Metric.metric_id == metric_1.metric_id) == []


def test_search_by_parameter(metric_1: Metric) -> None:
    db = InMemoryMetricDB()
    db.persist([metric_1])
    results = db.search(repositories.Metric.parameters == {"column": "page_views"})
    assert len(results) == 1

    results = db.search(repositories.Metric.parameters == {"column": "impressions"})
    assert len(results) == 0


def test_get_metric_value(metric_1: Metric, key: ResultKey) -> None:
    db = InMemoryMetricDB()
    db.persist([metric_1])
    value = db.get_metric_value(specs.Average("page_views"), key, dataset="test_dataset")
    assert value == Some(pytest.approx(5.2))


def test_get_metric_window(metric_window: list[Metric], key: ResultKey) -> None:
    db = InMemoryMetricDB()
    db.persist(metric_window)
    value = db.get_metric_window(specs.Average("page_views"), key, lag=1, window=5, dataset="test_dataset").unwrap()
    assert len(value) == 5
    assert min(value.keys()) == dt.date.fromisoformat("2025-01-30")
    assert max(value.keys()) == dt.date.fromisoformat("2025-02-03")
    assert min(value.values()) == pytest.approx(5.2)
    assert max(value.values()) == pytest.approx(5.2)


def test_get_missing_metric_by_uuid(key: ResultKey) -> None:
    """Test getting a non-existent metric by UUID returns empty Maybe."""
    db = InMemoryMetricDB()
    import uuid

    non_existent_id = uuid.uuid4()
    result = db.get(non_existent_id)
    assert result == Nothing


def test_get_missing_metric_by_key(key: ResultKey) -> None:
    """Test getting a non-existent metric by ResultKey returns empty Maybe."""
    db = InMemoryMetricDB()
    spec = specs.Average("non_existent_column")
    result = db.get(key, spec)
    assert result == Nothing


def test_get_with_result_key_no_spec(key: ResultKey) -> None:
    """Test that using ResultKey without MetricSpec raises DQXError."""
    db = InMemoryMetricDB()
    with pytest.raises(DQXError, match="MetricSpec must be provided when using ResultKey"):
        db.get(key)  # type: ignore[call-overload]


def test_get_with_unsupported_key_type() -> None:
    """Test that using unsupported key type raises DQXError."""
    db = InMemoryMetricDB()
    with pytest.raises(DQXError, match="Unsupported key type"):
        db.get("invalid_key")  # type: ignore


def test_search_empty_expressions() -> None:
    """Test that searching with no filter expressions raises DQXError."""
    db = InMemoryMetricDB()
    with pytest.raises(DQXError, match="Filter expressions cannot be empty"):
        db.search()


def test_get_metric_value_missing(key: ResultKey) -> None:
    """Test getting value for non-existent metric returns empty Maybe."""
    db = InMemoryMetricDB()
    spec = specs.Average("non_existent_column")
    result = db.get_metric_value(spec, key, dataset="test_dataset")
    assert result == Nothing


def test_get_metric_window_missing(key: ResultKey) -> None:
    """Test getting window for non-existent metric returns Some with empty dict."""
    db = InMemoryMetricDB()
    spec = specs.Average("non_existent_column")
    result = db.get_metric_window(spec, key, lag=1, window=5, dataset="test_dataset")
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


def test_get_metric_window_with_no_scalars_result(key: ResultKey) -> None:
    """Test get_metric_window when session.execute returns None."""
    from unittest.mock import Mock, patch

    db = InMemoryMetricDB()
    spec = specs.Average("test_column")

    # Mock the session to return None from execute()
    with patch.object(db, "new_session") as mock_session:
        mock_session_instance = Mock()
        mock_session_instance.execute.return_value = None
        mock_session.return_value = mock_session_instance

        result = db.get_metric_window(spec, key, lag=1, window=5, dataset="test_dataset")
        assert result == Nothing


# Tests for get_by_execution_id method


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
