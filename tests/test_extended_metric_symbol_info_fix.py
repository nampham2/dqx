"""Test that extended metrics display correct names in SymbolInfo."""

import datetime as dt

from dqx.common import ResultKey
from dqx.evaluator import Evaluator
from dqx.orm.repositories import InMemoryMetricDB
from dqx.provider import MetricProvider


def test_extended_metric_symbol_info_displays_correct_name() -> None:
    """Test that SymbolInfo shows 'day_over_day(maximum(tax))' not 'maximum(tax)'."""
    # GIVEN: A metric provider with extended metrics
    db = InMemoryMetricDB()
    mp = MetricProvider(db)

    # WHEN: Creating an extended metric and collecting symbols
    base = mp.maximum("tax")
    dod = mp.ext.day_over_day(base)

    # Create evaluator and collect symbols
    key = ResultKey(yyyy_mm_dd=dt.date(2024, 10, 24), tags={})
    evaluator = Evaluator(mp, key, "Test Suite")
    evaluator._metrics = evaluator.collect_metrics(key)  # Force metric collection
    symbol_infos = evaluator.collect_symbols(dod)

    # THEN: SymbolInfo should show the extended metric name
    dod_info = next((si for si in symbol_infos if si.name == str(dod)), None)
    assert dod_info is not None, f"Could not find symbol {dod} in collected symbols"
    assert dod_info.metric == "day_over_day(maximum(tax))", (
        f"Expected 'day_over_day(maximum(tax))', got '{dod_info.metric}'"
    )
