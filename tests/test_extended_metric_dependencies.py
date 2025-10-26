"""Test automatic dependency creation for extended metrics."""

import datetime as dt

from dqx.common import ResultKey
from dqx.orm.repositories import InMemoryMetricDB
from dqx.provider import MetricProvider


def test_day_over_day_creates_lag_dependency() -> None:
    """Test that day_over_day automatically creates lag(1) dependency."""
    # GIVEN: A metric provider
    mp = MetricProvider(InMemoryMetricDB())

    # WHEN: Creating a day_over_day metric
    base = mp.maximum("tax")
    dod = mp.ext.day_over_day(base)

    # THEN: The day_over_day metric should have the base metric and lag(1) as children
    children = mp.get_children(dod)
    assert len(children) == 2  # base and lag(1)

    # The base metric should be one of the children
    assert base in children

    # Find the lag symbol (not the base)
    lag_symbol = next((child for child in children if child != base), None)
    assert lag_symbol is not None

    # Verify the lag symbol has correct metadata
    lag_metric = mp.get_symbol(lag_symbol)
    assert "lag(1)" in lag_metric.name
    assert lag_metric.lag == 1


def test_week_over_week_creates_lag_dependency() -> None:
    """Test that week_over_week automatically creates lag(7) dependency."""
    # GIVEN: A metric provider
    mp = MetricProvider(InMemoryMetricDB())

    # WHEN: Creating a week_over_week metric
    base = mp.sum("revenue")
    wow = mp.ext.week_over_week(base)

    # THEN: The week_over_week metric should have the base metric and lag(7) as children
    children = mp.get_children(wow)
    assert len(children) == 2  # base and lag(7)

    # The base metric should be one of the children
    assert base in children

    # Find the lag symbol (not the base)
    lag_symbol = next((child for child in children if child != base), None)
    assert lag_symbol is not None

    # Verify the lag symbol has correct metadata
    lag_metric = mp.get_symbol(lag_symbol)
    assert "lag(7)" in lag_metric.name
    assert lag_metric.lag == 7


def test_dependency_symbols_are_collected() -> None:
    """Test that dependency symbols are registered in the provider."""
    # GIVEN: A metric provider with day_over_day metric
    db = InMemoryMetricDB()
    mp = MetricProvider(db)

    base = mp.maximum("tax")
    dod = mp.ext.day_over_day(base)

    # WHEN: Looking at all registered symbols
    all_symbols = list(mp.symbols())

    # THEN: We should have base, dod, and lag(1) symbols registered
    assert len(all_symbols) == 3

    # Find the lag symbol
    lag_symbol = next((s for s in all_symbols if s != base and s != dod), None)
    assert lag_symbol is not None

    # Verify the lag symbol has correct metadata
    lag_metric = mp.get_symbol(lag_symbol)
    assert "lag(1)" in lag_metric.name
    assert lag_metric.lag == 1

    # Verify the lag metric can be evaluated
    from dqx.evaluator import Evaluator

    key = ResultKey(yyyy_mm_dd=dt.date(2024, 10, 24), tags={})
    evaluator = Evaluator(mp, key, "Test Suite")
    evaluator._metrics = evaluator.collect_metrics(key)
    assert lag_symbol in evaluator.metrics
