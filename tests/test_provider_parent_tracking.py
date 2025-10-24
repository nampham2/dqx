"""Test parent symbol tracking for extended metrics."""

from dqx.orm.repositories import InMemoryMetricDB
from dqx.provider import MetricProvider


def test_extended_metrics_have_parent_symbols() -> None:
    """Test that extended metrics track their parent symbols."""
    # GIVEN: A metric provider
    mp = MetricProvider(InMemoryMetricDB())

    # WHEN: Creating an extended metric
    base = mp.maximum("tax")
    dod = mp.ext.day_over_day(base)

    # THEN: The extended metric should have the base as parent
    dod_metric = mp.get_symbol(dod)
    assert hasattr(dod_metric, "parent_symbol")
    assert dod_metric.parent_symbol == base
