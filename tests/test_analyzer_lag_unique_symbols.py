"""
Test that lag operations get unique symbols in analysis reports.

This test verifies the fix for the duplicate symbol bug where
the same metric computed for different dates (e.g., current and lag)
was getting the same symbol.
"""

import datetime
from typing import Any

import pyarrow as pa

from dqx.api import VerificationSuite, check
from dqx.common import ResultKey
from dqx.datasource import DuckRelationDataSource
from dqx.orm.repositories import InMemoryMetricDB
from dqx.provider import MetricProvider
from dqx.specs import Average


class TestLagUniqueSymbols:
    """Test that lag operations receive unique symbols."""

    def test_lag_operations_get_unique_symbols(self) -> None:
        """Test that same metric with different lag dates gets unique symbols."""

        @check(name="Tax Lag Check")
        def tax_check(mp: MetricProvider, ctx: Any) -> None:
            # Same metric (average tax) computed for different dates
            current_tax = mp.average("tax")
            lag1_tax = mp.average("tax", lag=1)
            lag2_tax = mp.average("tax", lag=2)

            # Use all three in assertions
            ctx.assert_that(current_tax).where(name="Current tax positive").is_positive()
            ctx.assert_that(lag1_tax).where(name="Previous day tax positive").is_positive()
            ctx.assert_that(lag2_tax).where(name="Two days ago tax positive").is_positive()

        # Create data with tax values for multiple dates
        data = pa.table(
            {
                "tax": [10.0, 15.0, 20.0, 25.0],
                "date": [
                    datetime.date(2024, 1, 13),
                    datetime.date(2024, 1, 14),
                    datetime.date(2024, 1, 15),
                    datetime.date(2024, 1, 16),
                ],
            }
        )

        # Run suite
        db = InMemoryMetricDB()
        ds = DuckRelationDataSource.from_arrow(data, "taxes")
        suite = VerificationSuite([tax_check], db, "Tax Suite")

        key = ResultKey(datetime.date(2024, 1, 16), {})
        suite.run([ds], key)

        # Get the analysis report
        assert "taxes" in suite.analysis_reports
        report = suite.analysis_reports["taxes"]

        # Collect all symbols
        symbols = list(report.symbol_mapping.values())

        # Should have 3 unique symbols for the 3 average(tax) computations
        assert len(symbols) == 3, f"Expected 3 symbols, got {len(symbols)}: {symbols}"
        assert len(set(symbols)) == 3, f"Expected all unique symbols, got duplicates: {symbols}"

        # All symbols should follow the pattern x_N
        assert all(s.startswith("x_") for s in symbols), f"Invalid symbol format: {symbols}"

        # Verify the metrics are computed for different dates
        metric_keys = list(report.symbol_mapping.keys())
        dates = [key[1].yyyy_mm_dd for key in metric_keys]

        # Should have 3 different dates
        assert len(set(dates)) == 3, f"Expected 3 different dates, got: {dates}"

        # Dates should be current (2024-01-16), lag(1) (2024-01-15), lag(2) (2024-01-14)
        expected_dates = {
            datetime.date(2024, 1, 16),  # current
            datetime.date(2024, 1, 15),  # lag(1)
            datetime.date(2024, 1, 14),  # lag(2)
        }
        assert set(dates) == expected_dates, f"Expected dates {expected_dates}, got {set(dates)}"

    def test_multiple_metrics_with_lag_get_unique_symbols(self) -> None:
        """Test that different metrics with lag also get unique symbols."""

        @check(name="Multi Metric Lag Check")
        def multi_check(mp: MetricProvider, ctx: Any) -> None:
            # Different metrics, some with lag
            avg_price = mp.average("price")
            avg_price_lag = mp.average("price", lag=1)
            sum_quantity = mp.sum("quantity")
            sum_quantity_lag = mp.sum("quantity", lag=1)

            ctx.assert_that(avg_price).where(name="Avg price positive").is_positive()
            ctx.assert_that(avg_price_lag).where(name="Yesterday avg price positive").is_positive()
            ctx.assert_that(sum_quantity).where(name="Sum quantity positive").is_positive()
            ctx.assert_that(sum_quantity_lag).where(name="Yesterday sum quantity positive").is_positive()

        # Create test data
        data = pa.table(
            {
                "price": [100.0, 110.0, 120.0],
                "quantity": [5, 10, 15],
                "date": [
                    datetime.date(2024, 1, 14),
                    datetime.date(2024, 1, 15),
                    datetime.date(2024, 1, 16),
                ],
            }
        )

        db = InMemoryMetricDB()
        ds = DuckRelationDataSource.from_arrow(data, "sales")
        suite = VerificationSuite([multi_check], db, "Multi Metric Suite")

        key = ResultKey(datetime.date(2024, 1, 16), {})
        suite.run([ds], key)

        report = suite.analysis_reports["sales"]
        symbols = list(report.symbol_mapping.values())

        # Should have 4 unique symbols
        assert len(symbols) == 4, f"Expected 4 symbols, got {len(symbols)}"
        assert len(set(symbols)) == 4, f"Expected all unique symbols, got duplicates: {symbols}"

    def test_symbol_mapping_structure(self) -> None:
        """Test that symbol mapping uses (MetricSpec, ResultKey) as key."""

        @check(name="Symbol Structure Check")
        def simple_check(mp: MetricProvider, ctx: Any) -> None:
            avg = mp.average("value")
            avg_lag = mp.average("value", lag=1)
            ctx.assert_that(avg).where(name="Current average").is_positive()
            ctx.assert_that(avg_lag).where(name="Previous average").is_positive()

        data = pa.table(
            {
                "value": [1.0, 2.0, 3.0],
                "date": [
                    datetime.date(2024, 1, 1),
                    datetime.date(2024, 1, 2),
                    datetime.date(2024, 1, 3),
                ],
            }
        )

        db = InMemoryMetricDB()
        ds = DuckRelationDataSource.from_arrow(data, "test_data")
        suite = VerificationSuite([simple_check], db, "Structure Test")

        key = ResultKey(datetime.date(2024, 1, 3), {})
        suite.run([ds], key)

        report = suite.analysis_reports["test_data"]

        # Verify the structure of symbol_mapping keys
        for mapping_key in report.symbol_mapping.keys():
            # Key should be tuple of (MetricSpec, ResultKey)
            assert isinstance(mapping_key, tuple), f"Expected tuple key, got {type(mapping_key)}"
            assert len(mapping_key) == 2, f"Expected 2-element tuple, got {len(mapping_key)}"

            metric_spec, result_key = mapping_key
            assert isinstance(metric_spec, Average), f"Expected Average spec, got {type(metric_spec)}"
            assert isinstance(result_key, ResultKey), f"Expected ResultKey, got {type(result_key)}"

        # Verify we have mappings for both dates
        result_keys = [key[1] for key in report.symbol_mapping.keys()]
        dates = [rk.yyyy_mm_dd for rk in result_keys]
        assert datetime.date(2024, 1, 3) in dates  # Current date
        assert datetime.date(2024, 1, 2) in dates  # Lag(1) date
