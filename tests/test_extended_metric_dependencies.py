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

    # THEN: A lag(1) dependency should be automatically created
    children = mp.get_children(base)
    assert len(children) == 2  # dod and lag(1)

    # Find the lag symbol (not the dod)
    lag_symbol = next((child for child in children if child != dod), None)
    assert lag_symbol is not None

    # Verify the lag symbol has correct metadata
    lag_metric = mp.get_symbol(lag_symbol)
    assert lag_metric.parent_symbol == base
    assert "lag(1)" in lag_metric.name
    assert lag_metric.name == f"lag(1)({base})"


def test_week_over_week_creates_lag_dependency() -> None:
    """Test that week_over_week automatically creates lag(7) dependency."""
    # GIVEN: A metric provider
    mp = MetricProvider(InMemoryMetricDB())

    # WHEN: Creating a week_over_week metric
    base = mp.sum("revenue")
    wow = mp.ext.week_over_week(base)

    # THEN: A lag(7) dependency should be automatically created
    children = mp.get_children(base)
    assert len(children) == 2  # wow and lag(7)

    # Find the lag symbol (not the wow)
    lag_symbol = next((child for child in children if child != wow), None)
    assert lag_symbol is not None

    # Verify the lag symbol has correct metadata
    lag_metric = mp.get_symbol(lag_symbol)
    assert lag_metric.parent_symbol == base
    assert "lag(7)" in lag_metric.name
    assert lag_metric.name == f"lag(7)({base})"


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
    assert lag_metric.name == f"lag(1)({base})"
    assert lag_metric.parent_symbol == base

    # Verify the lag metric can be evaluated
    from dqx.evaluator import Evaluator

    key = ResultKey(yyyy_mm_dd=dt.date(2024, 10, 24), tags={})
    evaluator = Evaluator(mp, key, "Test Suite")
    evaluator._metrics = evaluator.collect_metrics(key)
    assert lag_symbol in evaluator.metrics
