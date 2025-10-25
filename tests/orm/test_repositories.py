import datetime as dt

import pytest
from returns.maybe import Nothing, Some
from rich.console import Console

from dqx import specs, states
from dqx.common import DQXError, ResultKey
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
    value = db.get_metric_value(specs.Average("page_views"), key)
    assert value == Some(pytest.approx(5.2))


def test_get_metric_window(metric_window: list[Metric], key: ResultKey) -> None:
    db = InMemoryMetricDB()
    db.persist(metric_window)
    value = db.get_metric_window(specs.Average("page_views"), key, lag=1, window=5).unwrap()
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
    result = db.get_metric_value(spec, key)
    assert result == Nothing


def test_get_metric_window_missing(key: ResultKey) -> None:
    """Test getting window for non-existent metric returns Some with empty dict."""
    db = InMemoryMetricDB()
    spec = specs.Average("non_existent_column")
    result = db.get_metric_window(spec, key, lag=1, window=5)
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
    """Test get_metric_window when session.scalars returns None."""
    from unittest.mock import Mock, patch

    db = InMemoryMetricDB()
    spec = specs.Average("test_column")

    # Mock the session to return None from scalars()
    with patch.object(db, "new_session") as mock_session:
        mock_session_instance = Mock()
        mock_session_instance.scalars.return_value = None
        mock_session.return_value = mock_session_instance

        result = db.get_metric_window(spec, key, lag=1, window=5)
        assert result == Nothing
