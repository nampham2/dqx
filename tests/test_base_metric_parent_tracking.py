"""Tests for base metric parent_symbol tracking after extended metric creation."""

from unittest.mock import Mock

import pytest

from dqx.orm.repositories import MetricDB
from dqx.provider import MetricProvider


@pytest.fixture
def mock_db() -> Mock:
    """Create a mock MetricDB."""
    return Mock(spec=MetricDB)


@pytest.fixture
def provider(mock_db: Mock) -> MetricProvider:
    """Create a MetricProvider with mocked DB."""
    return MetricProvider(mock_db)


def test_base_metric_parent_updated_for_day_over_day(provider: MetricProvider) -> None:
    """Test that base metric's parent_symbol is updated when used in day_over_day."""
    # Create a base metric
    base = provider.sum("revenue")

    # Get the base metric's symbolic data before extended metric creation
    base_metric = provider.get_symbol(base)
    assert base_metric.parent_symbol is None  # Initially no parent

    # Create day_over_day from the base metric
    dod = provider.ext.day_over_day(base)

    # Check that base metric's parent_symbol is now set to the dod symbol
    base_metric_after = provider.get_symbol(base)
    assert base_metric_after.parent_symbol == dod

    # Verify the extended metric itself has no parent
    dod_metric = provider.get_symbol(dod)
    assert dod_metric.parent_symbol is None


def test_base_metric_parent_updated_for_week_over_week(provider: MetricProvider) -> None:
    """Test that base metric's parent_symbol is updated when used in week_over_week."""
    # Create a base metric
    base = provider.average("price")

    # Get the base metric's symbolic data before extended metric creation
    base_metric = provider.get_symbol(base)
    assert base_metric.parent_symbol is None  # Initially no parent

    # Create week_over_week from the base metric
    wow = provider.ext.week_over_week(base)

    # Check that base metric's parent_symbol is now set to the wow symbol
    base_metric_after = provider.get_symbol(base)
    assert base_metric_after.parent_symbol == wow

    # Verify the extended metric itself has no parent
    wow_metric = provider.get_symbol(wow)
    assert wow_metric.parent_symbol is None


def test_base_metric_parent_updated_for_stddev(provider: MetricProvider) -> None:
    """Test that base metric's parent_symbol is updated when used in stddev."""
    # Create a base metric
    base = provider.variance("score")

    # Get the base metric's symbolic data before extended metric creation
    base_metric = provider.get_symbol(base)
    assert base_metric.parent_symbol is None  # Initially no parent

    # Create stddev from the base metric
    stddev = provider.ext.stddev(base, lag=1, n=7)

    # Check that base metric's parent_symbol is now set to the stddev symbol
    base_metric_after = provider.get_symbol(base)
    assert base_metric_after.parent_symbol == stddev

    # Verify the extended metric itself has no parent
    stddev_metric = provider.get_symbol(stddev)
    assert stddev_metric.parent_symbol is None


def test_base_metric_parent_persists_across_get_symbol_calls(provider: MetricProvider) -> None:
    """Test that parent_symbol persists when accessing the same metric multiple times."""
    # Create a base metric and extended metric
    base = provider.minimum("value")
    dod = provider.ext.day_over_day(base)

    # Access the base metric multiple times
    for _ in range(3):
        base_metric = provider.get_symbol(base)
        assert base_metric.parent_symbol == dod


def test_multiple_base_metrics_track_their_extended_metrics(provider: MetricProvider) -> None:
    """Test that multiple base metrics correctly track their respective extended metrics."""
    # Create multiple base metrics
    base1 = provider.sum("revenue")
    base2 = provider.average("price")
    base3 = provider.maximum("score")

    # Create extended metrics
    dod = provider.ext.day_over_day(base1)
    wow = provider.ext.week_over_week(base2)
    stddev = provider.ext.stddev(base3, lag=2, n=5)

    # Verify each base metric points to its extended metric
    assert provider.get_symbol(base1).parent_symbol == dod
    assert provider.get_symbol(base2).parent_symbol == wow
    assert provider.get_symbol(base3).parent_symbol == stddev


def test_lag_dependencies_have_correct_parent(provider: MetricProvider) -> None:
    """Test that lag dependencies created by extended metrics have correct parent_symbol."""
    # Create a base metric and day_over_day
    base = provider.num_rows()
    dod = provider.ext.day_over_day(base)

    # Get children of dod (should include base and lag(1))
    children = provider.get_children(dod)

    # Find the lag dependency (it's not the base metric)
    lag_symbols = [child for child in children if child != base]
    assert len(lag_symbols) == 1

    # Verify the lag dependency has dod as parent
    lag_metric = provider.get_symbol(lag_symbols[0])
    assert lag_metric.parent_symbol == dod
    assert "lag(1)" in lag_metric.name
