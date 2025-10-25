import datetime as dt

import pytest

from dqx import specs, states
from dqx.common import DQXError, ResultKey
from dqx.models import Metric


@pytest.fixture
def key() -> ResultKey:
    return ResultKey(yyyy_mm_dd=dt.date.fromisoformat("2025-02-04"), tags={})


@pytest.fixture
def metric_1(key: ResultKey) -> Metric:
    metric = specs.Average("page_views")
    return Metric.build(metric, key, state=states.Average(5.2, 10), dataset="test_dataset")


@pytest.fixture
def metric_2(key: ResultKey) -> Metric:
    metric = specs.Average("visitors")
    return Metric.build(metric, key, state=states.Average(2044, 10), dataset="test_dataset")


@pytest.fixture
def metric_3(key: ResultKey) -> Metric:
    metric = specs.NullCount("visitors")
    return Metric.build(metric, key, state=states.Average(2044, 10), dataset="test_dataset")


@pytest.fixture
def metric_4(key: ResultKey) -> Metric:
    metric = specs.Average("page_views")
    return Metric.build(metric, key, state=states.Average(10.0, 5.0), dataset="test_dataset")


def test_metric_merge_success(metric_1: Metric, key: ResultKey) -> None:
    assert metric_1.spec.name == "average(page_views)"
    assert metric_1.key == key
    assert metric_1.value == metric_1.state.value

    merged_metric = metric_1.merge(metric_1)

    assert merged_metric.spec == metric_1.spec
    assert merged_metric.key == key
    assert merged_metric.value == pytest.approx(5.2)


def test_metric_reduce(metric_1: Metric, metric_4: Metric) -> None:
    merged = Metric.reduce([metric_1, metric_4])
    expected_avg = (10 * 5.2 + 5 * 10.0) / 15
    assert merged.value == pytest.approx(expected_avg)


def test_metric_merge_different_types(metric_1: Metric, metric_3: Metric) -> None:
    with pytest.raises(DQXError, match="Cannot merge metrics with different spec"):
        metric_1.merge(metric_3)


def test_metric_merge_different_names(metric_1: Metric, metric_2: Metric) -> None:
    with pytest.raises(DQXError, match="Cannot merge metrics with different spec"):
        metric_1.merge(metric_2)
