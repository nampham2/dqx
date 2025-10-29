"""Unit tests for the create_metric method."""

from typing import Any

import pytest

from dqx import specs
from dqx.orm.repositories import InMemoryMetricDB
from dqx.provider import MetricProvider


def test_create_metric_simple_metrics() -> None:
    """Test create_metric with simple metrics."""
    db = InMemoryMetricDB()
    provider = MetricProvider(db, "test-exec")

    # Test various simple metrics
    avg = provider.create_metric(specs.Average("price"), lag=0, dataset="products")
    assert provider.get_symbol(avg).metric_spec.metric_type == "Average"
    assert provider.get_symbol(avg).lag == 0
    assert provider.get_symbol(avg).dataset == "products"

    sum_metric = provider.create_metric(specs.Sum("revenue"), lag=1, dataset="sales")
    assert provider.get_symbol(sum_metric).metric_spec.metric_type == "Sum"
    assert provider.get_symbol(sum_metric).lag == 1
    assert provider.get_symbol(sum_metric).dataset == "sales"

    num_rows = provider.create_metric(specs.NumRows(), lag=0)
    assert provider.get_symbol(num_rows).metric_spec.metric_type == "NumRows"
    assert provider.get_symbol(num_rows).dataset is None


def test_create_metric_extended_day_over_day() -> None:
    """Test create_metric with DayOverDay extended metrics."""
    db = InMemoryMetricDB()
    provider = MetricProvider(db, "test-exec")

    # Create a DayOverDay metric
    dod_spec = specs.DayOverDay.from_base_spec(specs.Average("price"))
    dod = provider.create_metric(dod_spec, lag=0, dataset="products")

    dod_metric = provider.get_symbol(dod)
    assert isinstance(dod_metric.metric_spec, specs.DayOverDay)
    assert dod_metric.metric_spec.base_spec.metric_type == "Average"
    assert len(dod_metric.required_metrics) == 2  # lag 0 and lag 1

    # Verify the dependencies were created correctly
    for req in dod_metric.required_metrics:
        req_metric = provider.get_symbol(req)
        assert req_metric.metric_spec.metric_type == "Average"


def test_create_metric_extended_week_over_week() -> None:
    """Test create_metric with WeekOverWeek extended metrics."""
    db = InMemoryMetricDB()
    provider = MetricProvider(db, "test-exec")

    # Create a WeekOverWeek metric
    wow_spec = specs.WeekOverWeek.from_base_spec(specs.Maximum("temperature"))
    wow = provider.create_metric(wow_spec, lag=0, dataset="weather")

    wow_metric = provider.get_symbol(wow)
    assert isinstance(wow_metric.metric_spec, specs.WeekOverWeek)
    assert wow_metric.metric_spec.base_spec.metric_type == "Maximum"
    assert len(wow_metric.required_metrics) == 2  # lag 0 and lag 7

    # Verify the dependencies
    for req in wow_metric.required_metrics:
        req_metric = provider.get_symbol(req)
        assert req_metric.metric_spec.metric_type == "Maximum"


def test_create_metric_extended_stddev() -> None:
    """Test create_metric with Stddev extended metrics."""
    db = InMemoryMetricDB()
    provider = MetricProvider(db, "test-exec")

    # Create a Stddev metric
    stddev_spec = specs.Stddev.from_base_spec(specs.Sum("sales"), offset=0, n=5)
    stddev = provider.create_metric(stddev_spec, lag=0, dataset="revenue")

    stddev_metric = provider.get_symbol(stddev)
    assert isinstance(stddev_metric.metric_spec, specs.Stddev)
    assert stddev_metric.metric_spec.base_spec.metric_type == "Sum"
    assert len(stddev_metric.required_metrics) == 5  # n=5

    # Verify the dependencies
    for i, req in enumerate(stddev_metric.required_metrics):
        req_metric = provider.get_symbol(req)
        assert req_metric.metric_spec.metric_type == "Sum"
        assert req_metric.lag == i  # lag 0 through 4


def test_create_metric_nested_extended_metrics() -> None:
    """Test create_metric with nested extended metrics like DayOverDay(DayOverDay(...))."""
    db = InMemoryMetricDB()
    provider = MetricProvider(db, "test-exec")

    # Create nested DoD(DoD(Average))
    base_spec = specs.Average("price")
    dod_spec = specs.DayOverDay.from_base_spec(base_spec)
    dod_dod_spec = specs.DayOverDay.from_base_spec(dod_spec)

    nested = provider.create_metric(dod_dod_spec, lag=0, dataset="products")

    nested_metric = provider.get_symbol(nested)
    assert isinstance(nested_metric.metric_spec, specs.DayOverDay)
    assert isinstance(nested_metric.metric_spec.base_spec, specs.DayOverDay)
    assert nested_metric.metric_spec.base_spec.base_spec.metric_type == "Average"

    # Should have 2 dependencies (both DayOverDay metrics)
    assert len(nested_metric.required_metrics) == 2
    for req in nested_metric.required_metrics:
        req_metric = provider.get_symbol(req)
        assert isinstance(req_metric.metric_spec, specs.DayOverDay)


def test_create_metric_stddev_of_extended_metric() -> None:
    """Test the specific case of Stddev(DayOverDay(...)) that was the original bug."""
    db = InMemoryMetricDB()
    provider = MetricProvider(db, "test-exec")

    # Create Stddev(DoD(Average))
    avg_spec = specs.Average("tax")
    dod_spec = specs.DayOverDay.from_base_spec(avg_spec)
    stddev_spec = specs.Stddev.from_base_spec(dod_spec, offset=0, n=7)

    stddev = provider.create_metric(stddev_spec, lag=0, dataset="tax_data")

    stddev_metric = provider.get_symbol(stddev)
    assert isinstance(stddev_metric.metric_spec, specs.Stddev)
    assert isinstance(stddev_metric.metric_spec.base_spec, specs.DayOverDay)

    # Should have 7 DayOverDay dependencies
    assert len(stddev_metric.required_metrics) == 7
    for req in stddev_metric.required_metrics:
        req_metric = provider.get_symbol(req)
        assert isinstance(req_metric.metric_spec, specs.DayOverDay)
        assert req_metric.metric_spec.base_spec.metric_type == "Average"


def test_create_metric_invalid_type() -> None:
    """Test create_metric with an invalid metric type."""
    db = InMemoryMetricDB()
    provider = MetricProvider(db, "test-exec")

    # Create a mock invalid spec
    class InvalidSpec:
        metric_type = "Invalid"
        is_extended = True
        name = "invalid"
        parameters: dict[str, Any] = {}

    with pytest.raises(ValueError, match="Unsupported extended metric type: Invalid"):
        provider.create_metric(InvalidSpec(), lag=0)  # type: ignore
