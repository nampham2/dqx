"""Test children tracking for extended metrics."""

from dqx.orm.repositories import InMemoryMetricDB
from dqx.provider import MetricProvider


def test_extended_metrics_children_tracking() -> None:
    """Test that MetricProvider tracks parent-child relationships.

    In the reversed relationship:
    - Extended metrics (day_over_day, week_over_week) are parents
    - Base metrics and lag metrics they depend on are children
    """
    # GIVEN: A metric provider
    mp = MetricProvider(InMemoryMetricDB())

    # WHEN: Creating an extended metric hierarchy
    base = mp.maximum("tax")
    dod = mp.ext.day_over_day(base)

    # THEN: The extended metric (parent) should have children tracked
    children = mp.get_children(dod)
    assert base in children
    assert len(children) == 2  # base and lag(1) dependency

    # Verify the lag(1) dependency is also a child
    lag_symbols = [s for s in children if s != base]
    assert len(lag_symbols) == 1
    lag_metric = mp.get_symbol(lag_symbols[0])
    assert "lag(1)" in lag_metric.name

    # AND: The base metric should have no children (it's a leaf)
    assert mp.get_children(base) == []

    # AND: Creating another extended metric
    wow = mp.ext.week_over_week(base)

    # THEN: The week_over_week should have its own children
    wow_children = mp.get_children(wow)
    assert base in wow_children
    assert len(wow_children) == 2  # base and lag(7) dependency

    # Verify the lag(7) dependency
    lag_symbols = [s for s in wow_children if s != base]
    assert len(lag_symbols) == 1
    lag_metric = mp.get_symbol(lag_symbols[0])
    assert "lag(7)" in lag_metric.name

    # AND: The base metric still has no children
    assert mp.get_children(base) == []
