"""
Test that all computed symbols appear in analysis reports.

This test verifies the fix for the missing symbols bug where
metrics without explicit dataset assignment were filtered out
from the analysis report.
"""

import datetime
from typing import Any

import pyarrow as pa

from dqx.api import VerificationSuite, check
from dqx.common import ResultKey
from dqx.datasource import DuckRelationDataSource
from dqx.orm.repositories import InMemoryMetricDB
from dqx.provider import MetricProvider


class TestAllSymbolsInReport:
    """Test that all computed symbols appear in analysis reports."""

    def test_all_computed_symbols_in_report(self) -> None:
        """Test that all computed symbols appear in analysis report."""

        @check(name="Orders Check", datasets=["orders"])
        def orders_check(mp: MetricProvider, ctx: Any) -> None:
            # Metrics without explicit dataset (should be imputed to orders)
            null_delivered = mp.null_count("delivered")
            avg_tax = mp.average("tax")

            # Use metrics in assertions
            ctx.assert_that(null_delivered).where(name="Delivered nulls < 100").is_lt(100)
            ctx.assert_that(avg_tax).where(name="Tax positive").is_positive()

        @check(name="Products Check", datasets=["products"])
        def products_check(mp: MetricProvider, ctx: Any) -> None:
            # Metrics with explicit dataset
            avg_price = mp.average("price")
            sum_quantity = mp.sum("quantity")

            # Use metrics in assertions
            ctx.assert_that(avg_price).where(name="Price positive").is_positive()
            ctx.assert_that(sum_quantity).where(name="Quantity positive").is_positive()

        # Create two datasets
        orders_data = pa.table(
            {
                "delivered": [1, None, 1, None, 1],
                "tax": [10.0, 15.0, 20.0, 25.0, 30.0],
            }
        )
        products_data = pa.table(
            {
                "price": [100.0, 200.0, 300.0],
                "quantity": [5, 10, 15],
            }
        )

        db = InMemoryMetricDB()
        ds1 = DuckRelationDataSource.from_arrow(orders_data, "orders")
        ds2 = DuckRelationDataSource.from_arrow(products_data, "products")

        suite = VerificationSuite([orders_check, products_check], db, "Multi Suite")
        key = ResultKey(datetime.date(2024, 1, 15), {})
        suite.run([ds1, ds2], key)

        # Collect all symbols from provider
        provider_symbols = [str(s.symbol) for s in suite.provider.symbolic_metrics]

        # Collect all symbols from all analysis reports
        all_report_symbols: list[str] = []
        for ds_name, report in suite.analysis_reports.items():
            all_report_symbols.extend(report.symbol_mapping.values())

        # All provider symbols should appear in analysis reports
        assert len(provider_symbols) == 4, f"Expected 4 provider symbols, got {len(provider_symbols)}"
        assert len(all_report_symbols) >= 4, f"Expected at least 4 report symbols, got {len(all_report_symbols)}"

        # Check that all provider symbols are in reports
        assert set(provider_symbols).issubset(set(all_report_symbols)), (
            f"Provider symbols {provider_symbols} not all in report symbols {all_report_symbols}"
        )

    def test_metrics_without_dataset_included(self) -> None:
        """Test that metrics without dataset assignment are included."""

        @check(name="No Dataset Check")
        def no_dataset_check(mp: MetricProvider, ctx: Any) -> None:
            # Create metrics without specifying dataset
            rows = mp.num_rows()
            nulls = mp.null_count("id")
            avg = mp.average("value")

            ctx.assert_that(rows).where(name="Has rows").is_positive()
            ctx.assert_that(nulls).where(name="No nulls").is_eq(0)
            ctx.assert_that(avg).where(name="Positive average").is_positive()

        data = pa.table(
            {
                "id": [1, 2, 3, 4, 5],
                "value": [10.0, 20.0, 30.0, 40.0, 50.0],
            }
        )

        db = InMemoryMetricDB()
        ds = DuckRelationDataSource.from_arrow(data, "test_data")
        suite = VerificationSuite([no_dataset_check], db, "No Dataset Suite")

        key = ResultKey(datetime.date(2024, 1, 15), {})
        suite.run([ds], key)

        # Check that all metrics appear in the report
        report = suite.analysis_reports["test_data"]
        assert len(report.symbol_mapping) == 3, f"Expected 3 symbols in report, got {len(report.symbol_mapping)}"

        # Verify all provider metrics are in the report
        provider_metrics = suite.provider.symbolic_metrics
        assert len(provider_metrics) == 3, f"Expected 3 provider metrics, got {len(provider_metrics)}"

    def test_mixed_dataset_assignment(self) -> None:
        """Test mixed scenario with some metrics having dataset and others not."""

        @check(name="Sales Check", datasets=["sales"])
        def sales_check(mp: MetricProvider, ctx: Any) -> None:
            # Metrics without dataset (will be imputed to sales)
            total_rows = mp.num_rows()
            generic_sum = mp.sum("value")

            # Metrics with explicit dataset
            sales_avg = mp.average("amount", dataset="sales")

            ctx.assert_that(total_rows).where(name="Total rows positive").is_positive()
            ctx.assert_that(sales_avg).where(name="Sales avg positive").is_positive()
            ctx.assert_that(generic_sum).where(name="Generic sum positive").is_positive()

        @check(name="Users Check", datasets=["users"])
        def users_check(mp: MetricProvider, ctx: Any) -> None:
            # Metrics with explicit dataset
            users_count = mp.null_count("user_id", dataset="users")

            ctx.assert_that(users_count).where(name="User nulls low").is_lt(10)

        # Create datasets
        sales_data = pa.table(
            {
                "amount": [100.0, 200.0, 300.0],
                "value": [10.0, 20.0, 30.0],
            }
        )
        users_data = pa.table(
            {
                "user_id": [1, 2, None, 4, None],
                "value": [5.0, 10.0, 15.0, 20.0, 25.0],
            }
        )

        db = InMemoryMetricDB()
        ds1 = DuckRelationDataSource.from_arrow(sales_data, "sales")
        ds2 = DuckRelationDataSource.from_arrow(users_data, "users")

        suite = VerificationSuite([sales_check, users_check], db, "Mixed Suite")
        key = ResultKey(datetime.date(2024, 1, 15), {})
        suite.run([ds1, ds2], key)

        # Count total symbols across all reports
        total_symbols = 0
        for report in suite.analysis_reports.values():
            total_symbols += len(report.symbol_mapping)

        # We should have all 4 metrics in the reports
        # Note: total_rows and generic_sum will only be computed for sales dataset
        assert total_symbols >= 4, f"Expected at least 4 total symbols, got {total_symbols}"

        # Verify specific metrics are in correct reports
        sales_report = suite.analysis_reports["sales"]
        users_report = suite.analysis_reports["users"]

        # Check that sales_avg is in sales report
        sales_metrics = [key[0].name for key in sales_report.symbol_mapping.keys()]
        assert any("average(amount)" in m for m in sales_metrics), "average(amount) not found in sales report"

        # Check that users_count is in users report
        users_metrics = [key[0].name for key in users_report.symbol_mapping.keys()]
        assert any("null_count(user_id)" in m for m in users_metrics), "null_count(user_id) not found in users report"

    def test_symbol_appears_in_correct_dataset_report(self) -> None:
        """Test that symbols appear in the correct dataset's analysis report."""

        @check(name="Dataset1 Check", datasets=["dataset1"])
        def dataset1_check(mp: MetricProvider, ctx: Any) -> None:
            # Explicitly assign to dataset1
            ds1_metric = mp.average("col1", dataset="dataset1")

            # No dataset assignment (will be imputed to dataset1)
            generic_metric = mp.null_count("col_common")

            ctx.assert_that(ds1_metric).where(name="DS1 metric positive").is_positive()
            ctx.assert_that(generic_metric).where(name="Generic metric zero").is_eq(0)

        @check(name="Dataset2 Check", datasets=["dataset2"])
        def dataset2_check(mp: MetricProvider, ctx: Any) -> None:
            # Explicitly assign to dataset2
            ds2_metric = mp.sum("col2", dataset="dataset2")

            ctx.assert_that(ds2_metric).where(name="DS2 metric positive").is_positive()

        # Create two datasets with overlapping columns
        data1 = pa.table(
            {
                "col1": [1.0, 2.0, 3.0],
                "col_common": [10, 20, 30],
            }
        )
        data2 = pa.table(
            {
                "col2": [100.0, 200.0, 300.0],
                "col_common": [40, 50, 60],
            }
        )

        db = InMemoryMetricDB()
        ds1 = DuckRelationDataSource.from_arrow(data1, "dataset1")
        ds2 = DuckRelationDataSource.from_arrow(data2, "dataset2")

        suite = VerificationSuite([dataset1_check, dataset2_check], db, "Dataset Assignment Suite")
        key = ResultKey(datetime.date(2024, 1, 15), {})
        suite.run([ds1, ds2], key)

        # Verify ds1_metric is in dataset1 report
        ds1_report = suite.analysis_reports["dataset1"]
        ds1_metrics = [key[0].name for key in ds1_report.symbol_mapping.keys()]
        assert any("average(col1)" in m for m in ds1_metrics), "average(col1) should be in dataset1 report"

        # Verify ds2_metric is in dataset2 report
        ds2_report = suite.analysis_reports["dataset2"]
        ds2_metrics = [key[0].name for key in ds2_report.symbol_mapping.keys()]
        assert any("sum(col2)" in m for m in ds2_metrics), "sum(col2) should be in dataset2 report"

        # Verify generic_metric appears in dataset1 report (where it was imputed)
        assert any("null_count(col_common)" in m for m in ds1_metrics), (
            "null_count(col_common) should be in dataset1 report"
        )
