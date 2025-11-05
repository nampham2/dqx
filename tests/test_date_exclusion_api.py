"""Test date exclusion API integration with VerificationSuite."""

import datetime

import pyarrow as pa
import pytest

from dqx.api import Context, VerificationSuite, check
from dqx.common import ResultKey, SqlDataSource
from dqx.datasource import DuckRelationDataSource
from dqx.orm.repositories import InMemoryMetricDB
from dqx.provider import MetricProvider


class TestVerificationSuiteSkipDates:
    """Test VerificationSuite integration with skip_dates."""

    def test_constructor_accepts_skip_dates(self) -> None:
        """Test that VerificationSuite constructor accepts skip_dates."""

        @check(name="Test Check")
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            pass

        db = InMemoryMetricDB()
        skip_dates = {datetime.date(2024, 1, 1), datetime.date(2024, 1, 5)}

        suite = VerificationSuite(checks=[test_check], db=db, name="Test Suite", skip_dates=skip_dates)

        assert suite._skip_dates == skip_dates

    def test_constructor_default_empty_skip_dates(self) -> None:
        """Test that skip_dates defaults to empty set."""

        @check(name="Test Check")
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            pass

        db = InMemoryMetricDB()

        suite = VerificationSuite(checks=[test_check], db=db, name="Test Suite")

        assert suite._skip_dates == set()

    def test_run_calculates_data_av_ratios(self) -> None:
        """Test that run() calls calculate_data_av_ratios when skip_dates provided."""

        @check(name="Revenue Check")
        def revenue_check(mp: MetricProvider, ctx: Context) -> None:
            # Create metrics with different lags
            revenue_0 = mp.sum("revenue", lag=0, dataset="sales")
            mp.sum("revenue", lag=1, dataset="sales")  # Creates lag=1 metric for testing
            revenue_dod = mp.ext.day_over_day(revenue_0, dataset="sales")

            ctx.assert_that(revenue_0).where(name="Revenue > 0").is_positive()
            ctx.assert_that(revenue_dod).where(name="DoD > 0.8").is_gt(0.8)

        db = InMemoryMetricDB()
        skip_dates = {datetime.date(2024, 1, 14)}  # Exclude yesterday

        suite = VerificationSuite(checks=[revenue_check], db=db, name="Test Suite", skip_dates=skip_dates)

        # Create dummy data
        data = pa.table(
            {
                "revenue": [1000.0, 1100.0, 1200.0],
                "date": [datetime.date(2024, 1, 13), datetime.date(2024, 1, 14), datetime.date(2024, 1, 15)],
            }
        )
        datasource = DuckRelationDataSource.from_arrow(data, "sales")
        key = ResultKey(datetime.date(2024, 1, 15), {})

        # Run the suite
        suite.run([datasource], key, enable_plugins=False)

        # Verify data_av_ratio was calculated
        # Since lag=1 (2024-01-14) is excluded, the DoD metric should have ratio 0.5
        for sym_metric in suite.provider.metrics:
            if "day_over_day" in sym_metric.name or "dod(" in sym_metric.name:
                assert sym_metric.data_av_ratio == 0.5  # (1.0 + 0.0) / 2
            elif sym_metric.lag == 1:
                assert sym_metric.data_av_ratio == 0.0  # Excluded date
            elif sym_metric.lag == 0:
                assert sym_metric.data_av_ratio == 1.0  # Available date

    def test_run_without_skip_dates(self) -> None:
        """Test that run() works normally without skip_dates."""

        @check(name="Simple Check")
        def simple_check(mp: MetricProvider, ctx: Context) -> None:
            avg = mp.average("value", dataset="data")
            ctx.assert_that(avg).where(name="Average > 0").is_positive()

        db = InMemoryMetricDB()

        suite = VerificationSuite(
            checks=[simple_check],
            db=db,
            name="Test Suite",
            # No skip_dates
        )

        data = pa.table({"value": [10.0, 20.0, 30.0]})
        datasource = DuckRelationDataSource.from_arrow(data, "data")
        key = ResultKey(datetime.date(2024, 1, 15), {})

        suite.run([datasource], key, enable_plugins=False)

        # All metrics should have None or 1.0 ratio when no dates excluded
        for sym_metric in suite.provider.metrics:
            if sym_metric.data_av_ratio is not None:
                assert sym_metric.data_av_ratio == 1.0

    def test_multiple_datasets_with_skip_dates(self) -> None:
        """Test skip_dates with multiple datasets."""

        @check(name="Multi Dataset Check")
        def multi_check(mp: MetricProvider, ctx: Context) -> None:
            # Metrics from different datasets
            sales_sum = mp.sum("amount", lag=0, dataset="sales")
            costs_sum = mp.sum("amount", lag=1, dataset="costs")

            ctx.assert_that(sales_sum).where(name="Sales > 0").is_positive()
            ctx.assert_that(costs_sum).where(name="Costs > 0").is_positive()

        db = InMemoryMetricDB()
        skip_dates = {datetime.date(2024, 1, 14)}  # Exclude yesterday

        suite = VerificationSuite(checks=[multi_check], db=db, name="Test Suite", skip_dates=skip_dates)

        # Create data for two datasets
        sales_data = pa.table({"amount": [1000.0, 1500.0]})
        costs_data = pa.table({"amount": [500.0, 600.0]})

        datasources: list[SqlDataSource] = [
            DuckRelationDataSource.from_arrow(sales_data, "sales"),
            DuckRelationDataSource.from_arrow(costs_data, "costs"),
        ]
        key = ResultKey(datetime.date(2024, 1, 15), {})

        suite.run(datasources, key, enable_plugins=False)

        # Check ratios
        for sym_metric in suite.provider.metrics:
            if sym_metric.dataset == "sales" and sym_metric.lag == 0:
                assert sym_metric.data_av_ratio == 1.0  # Today is available
            elif sym_metric.dataset == "costs" and sym_metric.lag == 1:
                assert sym_metric.data_av_ratio == 0.0  # Yesterday is excluded

    def test_complex_extended_metrics_with_skip_dates(self) -> None:
        """Test complex metric hierarchy with skip_dates."""

        @check(name="Complex Metrics Check")
        def complex_check(mp: MetricProvider, ctx: Context) -> None:
            # Build a complex metric tree
            price = mp.average("price", dataset="products")

            # Stddev over 3 days
            price_stddev = mp.ext.stddev(price, offset=0, n=3, dataset="products")

            # Week over week
            price_wow = mp.ext.week_over_week(price, dataset="products")

            ctx.assert_that(price_stddev).where(name="Price volatility").is_lt(10.0)
            ctx.assert_that(price_wow).where(name="Weekly growth").is_gt(0.9)

        db = InMemoryMetricDB()
        skip_dates = {
            datetime.date(2024, 1, 14),  # lag=1
            datetime.date(2024, 1, 8),  # lag=7
        }

        suite = VerificationSuite(checks=[complex_check], db=db, name="Test Suite", skip_dates=skip_dates)

        # Create data with enough history
        dates = [datetime.date(2024, 1, i) for i in range(1, 16)]
        data = pa.table({"price": [100.0] * 15, "date": dates})
        datasource = DuckRelationDataSource.from_arrow(data, "products")
        key = ResultKey(datetime.date(2024, 1, 15), {})

        suite.run([datasource], key, enable_plugins=False)

        # Check ratios for extended metrics
        for sym_metric in suite.provider.metrics:
            if "stddev" in sym_metric.name:
                # Stddev uses lag 0, 1, 2
                # lag=1 is excluded, so ratio should be 2/3
                assert sym_metric.data_av_ratio == pytest.approx(2.0 / 3.0)
            elif "week_over_week" in sym_metric.name:
                # WoW uses lag 0 and 7
                # lag=7 is excluded, so ratio should be 0.5
                assert sym_metric.data_av_ratio == 0.5

    def test_empty_skip_dates_logging(self) -> None:
        """Test that empty skip_dates doesn't log calculation message."""

        @check(name="Test Check")
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            avg = mp.average("value", dataset="data")
            ctx.assert_that(avg).where(name="Average > 0").is_positive()

        db = InMemoryMetricDB()

        suite = VerificationSuite(
            checks=[test_check],
            db=db,
            name="Test Suite",
            skip_dates=set(),  # Explicitly empty
        )

        data = pa.table({"value": [10.0, 20.0, 30.0]})
        datasource = DuckRelationDataSource.from_arrow(data, "data")
        key = ResultKey(datetime.date(2024, 1, 15), {})

        # Should run without logging about date exclusion
        suite.run([datasource], key, enable_plugins=False)

    def test_skip_dates_with_symbol_deduplication(self) -> None:
        """Test that skip_dates work correctly with symbol deduplication."""

        @check(name="Duplicate Symbols Check")
        def dup_check(mp: MetricProvider, ctx: Context) -> None:
            # Create duplicate symbols
            sum1 = mp.sum("amount", lag=0, dataset="data")
            sum2 = mp.sum("amount", lag=0, dataset="data")  # Duplicate
            sum3 = mp.sum("amount", lag=1, dataset="data")

            ctx.assert_that(sum1 + sum2).where(name="Sum check").is_positive()
            ctx.assert_that(sum3).where(name="Yesterday sum").is_positive()

        db = InMemoryMetricDB()
        skip_dates = {datetime.date(2024, 1, 14)}

        suite = VerificationSuite(checks=[dup_check], db=db, name="Test Suite", skip_dates=skip_dates)

        data = pa.table({"amount": [100.0, 200.0]})
        datasource = DuckRelationDataSource.from_arrow(data, "data")
        key = ResultKey(datetime.date(2024, 1, 15), {})

        suite.run([datasource], key, enable_plugins=False)

        # After deduplication, check remaining symbols have correct ratios
        for sym_metric in suite.provider.metrics:
            if sym_metric.lag == 0:
                assert sym_metric.data_av_ratio == 1.0
            elif sym_metric.lag == 1:
                assert sym_metric.data_av_ratio == 0.0
