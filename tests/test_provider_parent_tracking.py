"""Test parent symbol tracking for extended metrics."""

from dqx.orm.repositories import InMemoryMetricDB
from dqx.provider import MetricProvider


def test_extended_metrics_have_parent_symbols() -> None:
    """Test parent-child relationships for extended metrics.

    Extended metrics track their dependencies through required_metrics.
    """
    # GIVEN: A metric provider
    mp = MetricProvider(InMemoryMetricDB())

    # WHEN: Creating an extended metric
    base = mp.maximum("tax")
    dod = mp.ext.day_over_day(base)

    # THEN: The extended metric tracks its dependencies
    dod_metric = mp.get_symbol(dod)
    assert hasattr(dod_metric, "required_metrics")
    assert base in dod_metric.required_metrics

    # AND: The extended metric has required metrics (base + lag)
    assert len(dod_metric.required_metrics) == 2  # base and lag(1)

    # Verify that one of them is the base metric
    assert any(req == base for req in dod_metric.required_metrics)
