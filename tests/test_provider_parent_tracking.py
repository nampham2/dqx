"""Test parent symbol tracking for extended metrics."""

from dqx.orm.repositories import InMemoryMetricDB
from dqx.provider import MetricProvider


def test_extended_metrics_have_parent_symbols() -> None:
    """Test parent-child relationships with reversed hierarchy.

    In the reversed relationship:
    - Extended metrics (day_over_day) are parents (no parent_symbol)
    - Base metrics they depend on are children
    """
    # GIVEN: A metric provider
    mp = MetricProvider(InMemoryMetricDB())

    # WHEN: Creating an extended metric
    base = mp.maximum("tax")
    dod = mp.ext.day_over_day(base)

    # THEN: The extended metric is a parent (has no parent_symbol)
    dod_metric = mp.get_symbol(dod)
    assert hasattr(dod_metric, "parent_symbol")
    assert dod_metric.parent_symbol is None

    # AND: The extended metric has children (base + lag metrics)
    children = mp.get_children(dod)
    assert base in children
    assert len(children) == 2  # base and lag(1)
