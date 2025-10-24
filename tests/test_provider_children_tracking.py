"""Test children tracking for extended metrics."""

from dqx.orm.repositories import InMemoryMetricDB
from dqx.provider import MetricProvider


def test_extended_metrics_children_tracking() -> None:
    """Test that MetricProvider tracks parent-child relationships."""
    # GIVEN: A metric provider
    mp = MetricProvider(InMemoryMetricDB())

    # WHEN: Creating an extended metric hierarchy
    base = mp.maximum("tax")
    dod = mp.ext.day_over_day(base)

    # THEN: The parent should have the child tracked
    children = mp.get_children(base)
    assert dod in children
    assert len(children) == 2  # dod and lag(1) dependency

    # Verify the lag(1) dependency is also a child
    lag_symbols = [s for s in children if s != dod]
    assert len(lag_symbols) == 1
    lag_metric = mp.get_symbol(lag_symbols[0])
    assert "lag(1)" in lag_metric.name

    # AND: Creating another child
    wow = mp.ext.week_over_week(base)

    # THEN: All children should be tracked (dod, wow, lag(1), lag(7))
    children = mp.get_children(base)
    assert dod in children
    assert wow in children
    assert len(children) == 4  # dod, wow, lag(1), lag(7)

    # Verify the lag dependencies
    lag_names = []
    for child in children:
        if child not in [dod, wow]:
            lag_metric = mp.get_symbol(child)
            lag_names.append(lag_metric.name)

    assert any("lag(1)" in name for name in lag_names)
    assert any("lag(7)" in name for name in lag_names)

    # AND: The extended metric should have no children
    assert mp.get_children(dod) == []
