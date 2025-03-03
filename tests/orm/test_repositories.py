import datetime as dt

import pytest
from returns.maybe import Some
from rich.console import Console

from dqx import specs, states
from dqx.common import ResultKey
from dqx.models import Metric
from dqx.orm import repositories
from dqx.orm.repositories import InMemoryMetricDB


@pytest.fixture
def key() -> ResultKey:
    return ResultKey(yyyy_mm_dd=dt.date.fromisoformat("2025-02-04"), tags={})


@pytest.fixture
def metric_1(key: ResultKey) -> Metric:
    metric = specs.Average("page_views")
    return Metric.build(metric, key, state=states.Average(5.2, 10))


@pytest.fixture
def metric_window(key: ResultKey) -> list[Metric]:
    metric = specs.Average("page_views")
    return [Metric.build(metric, key.lag(_), state=states.Average(5.2, 10)) for _ in range(10)]


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
