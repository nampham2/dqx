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

        # Get the symbols from the provider
        symbol_infos = suite.provider.collect_symbols(key)

        # Filter symbols for average(tax) metrics
        avg_tax_symbols = [s for s in symbol_infos if "average(tax)" in s.metric]

        # Should have 3 symbols for the 3 average(tax) computations
        assert len(avg_tax_symbols) == 3, f"Expected 3 average(tax) symbols, got {len(avg_tax_symbols)}"

        # Extract symbol names
        symbol_names = [s.name for s in avg_tax_symbols]
        assert len(set(symbol_names)) == 3, f"Expected all unique symbols, got duplicates: {symbol_names}"

        # All symbols should follow the pattern x_N
        assert all(s.startswith("x_") for s in symbol_names), f"Invalid symbol format: {symbol_names}"

        # Verify the metrics are computed for different dates
        dates = [s.yyyy_mm_dd for s in avg_tax_symbols]

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

        # Get all symbols from the provider
        symbol_infos = suite.provider.collect_symbols(key)

        # Filter symbols for our metrics (2 averages + 2 sums)
        relevant_symbols = [s for s in symbol_infos if ("average(price)" in s.metric or "sum(quantity)" in s.metric)]

        # Should have 4 unique symbols
        assert len(relevant_symbols) == 4, f"Expected 4 symbols, got {len(relevant_symbols)}"

        # Extract symbol names
        symbol_names = [s.name for s in relevant_symbols]
        assert len(set(symbol_names)) == 4, f"Expected all unique symbols, got duplicates: {symbol_names}"

    def test_symbol_mapping_structure(self) -> None:
        """Test that symbol mapping uses (MetricSpec, ResultKey, dataset) as key."""

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

        # Get all symbols from the provider
        symbol_infos = suite.provider.collect_symbols(key)

        # Filter for average(value) metrics
        avg_value_symbols = [s for s in symbol_infos if "average(value)" in s.metric]

        # Should have 2 symbols (current and lag)
        assert len(avg_value_symbols) == 2, f"Expected 2 symbols, got {len(avg_value_symbols)}"

        # Verify the structure of symbol info
        for symbol_info in avg_value_symbols:
            # Check that each symbol info has the expected attributes
            assert hasattr(symbol_info, "name"), "Missing name attribute"
            assert hasattr(symbol_info, "metric"), "Missing metric attribute"
            assert hasattr(symbol_info, "dataset"), "Missing dataset attribute"
            assert hasattr(symbol_info, "value"), "Missing value attribute"
            assert hasattr(symbol_info, "yyyy_mm_dd"), "Missing yyyy_mm_dd attribute"
            assert hasattr(symbol_info, "tags"), "Missing tags attribute"

            # Verify attribute values
            assert symbol_info.dataset == "test_data", f"Expected dataset 'test_data', got {symbol_info.dataset}"
            assert symbol_info.name.startswith("x_"), f"Invalid symbol name format: {symbol_info.name}"
            assert "average(value)" in symbol_info.metric, f"Unexpected metric: {symbol_info.metric}"

        # Verify we have mappings for both dates
        dates = [s.yyyy_mm_dd for s in avg_value_symbols]
        assert datetime.date(2024, 1, 3) in dates  # Current date
        assert datetime.date(2024, 1, 2) in dates  # Lag(1) date
