"""Integration tests for DayOverDay functionality."""

from __future__ import annotations

from datetime import date

import pyarrow as pa

from dqx import data, specs
from dqx.api import Context, VerificationSuite, check
from dqx.common import ResultKey
from dqx.datasource import DuckRelationDataSource
from dqx.display import print_metric_trace
from dqx.orm.repositories import InMemoryMetricDB
from dqx.provider import MetricProvider


def test_day_over_day_integration() -> None:
    """Test the complete DayOverDay flow from provider to spec to storage."""
    # Create in-memory database
    db = InMemoryMetricDB()

    # Define check using day_over_day
    @check(name="Revenue DoD Check")
    def revenue_dod_check(mp: MetricProvider, ctx: Context) -> None:
        # Create base metric and day-over-day metric
        revenue = mp.sum("revenue")
        dod = mp.ext.day_over_day(revenue)

        # Use noop to collect without validation
        ctx.assert_that(dod).config(name="Collect revenue DoD").noop()

    # Create verification suite
    suite = VerificationSuite([revenue_dod_check], db, "DoD Test Suite")

    # Create test data with two days of revenue
    today_data = pa.table({"revenue": [100.0, 200.0, 300.0], "date": ["2024-01-15", "2024-01-15", "2024-01-15"]})
    datasource_today = DuckRelationDataSource.from_arrow(today_data, "revenue_data")

    # Create suite for today
    suite = VerificationSuite([revenue_dod_check], db, "DoD Test Suite")
    key_today = ResultKey(yyyy_mm_dd=date(2024, 1, 15), tags={"test": "dod"})
    suite.run([datasource_today], key_today)

    # Verify metric was created correctly
    # Get the provider from the suite (use today's suite)
    provider = suite.provider

    # Find the DoD symbol (should be the last registered one)
    dod_symbol = None
    for sym_metric in provider.metrics:
        if "dod(" in sym_metric.name:
            dod_symbol = sym_metric.symbol
            break

    assert dod_symbol is not None, "DayOverDay symbol not found"

    # Get the symbolic metric
    dod_metric = provider.get_symbol(dod_symbol)

    # Verify the metric spec
    assert isinstance(dod_metric.metric_spec, specs.DayOverDay)
    assert dod_metric.metric_spec.metric_type == "DayOverDay"
    assert dod_metric.name == "dod(sum(revenue))"

    # Verify the base metric is stored correctly
    dod_spec = dod_metric.metric_spec
    assert dod_spec._base_metric_type == "Sum"
    assert dod_spec._base_parameters == {"column": "revenue"}

    # Verify serialization/deserialization
    state = dod_spec.state()
    serialized = state.serialize()
    deserialized = specs.DayOverDay.deserialize(serialized)
    assert isinstance(deserialized, specs.states.NonMergeable)
    assert deserialized.metric_type == "DayOverDay"

    # Verify the metric is stored in the database
    # The metric should be retrievable by key and spec
    from returns.maybe import Maybe, Some

    stored_metric = db.get_metric(dod_spec, key_today, "revenue_data", suite.execution_id)

    # Use pattern matching for Maybe type
    match stored_metric:
        case Some(stored_value):
            # Pattern matching automatically unwraps Some values
            assert stored_value.spec == dod_spec
            assert stored_value.key == key_today
        case Maybe.empty:
            raise AssertionError("DayOverDay metric not found in database")

    # Verify the suite results (from today's suite)
    results = suite.collect_results()
    assert len(results) == 1
    assert results[0].status == "PASSED"  # noop always succeeds
    assert results[0].assertion == "Collect revenue DoD"

    # Verify required metrics were registered
    assert len(dod_metric.required_metrics) == 2  # Base metric and lag(1) metric

    trace = data.metric_trace(
        db.get_by_execution_id(suite.execution_id),
        suite.execution_id,
        suite.analysis_reports,
        suite.provider.collect_symbols(suite.key),
        suite.provider.registry.symbol_lookup_table(suite.key),
    )
    print_metric_trace(trace)
