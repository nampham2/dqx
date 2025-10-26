"""
Test the new functionality for AnalysisReport symbol mappings.

Tests cover:
1. AnalysisReport initialization with symbol_mapping field
2. AnalysisReport.merge() preserving symbol mappings
3. Analyzer accepting symbol_lookup and populating mappings
4. VerificationSuite storing analysis reports
5. print_analysis_report display functionality
"""

import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pyarrow as pa

from dqx.analyzer import AnalysisReport, Analyzer
from dqx.api import VerificationSuite, check
from dqx.common import Metadata, ResultKey
from dqx.datasource import DuckRelationDataSource
from dqx.display import print_analysis_report
from dqx.models import Metric
from dqx.orm.repositories import InMemoryMetricDB

# Import actual metric spec classes
from dqx.specs import Average, MetricSpec, NullCount, Sum
from dqx.states import Average as AverageState
from dqx.states import SimpleAdditiveState


class TestAnalysisReportSymbols:
    """Test AnalysisReport symbol mapping functionality."""

    def test_analysis_report_init_with_symbol_mapping(self) -> None:
        """Test that AnalysisReport initializes with empty symbol_mapping."""
        report = AnalysisReport()
        assert hasattr(report, "symbol_mapping")
        assert report.symbol_mapping == {}

    def test_analysis_report_merge_preserves_symbol_mappings(self) -> None:
        """Test that merge() preserves symbol mappings from both reports."""
        # Create first report with some data and mappings
        spec1 = Sum("col1")
        key1 = ResultKey(datetime.date(2024, 1, 1), {})
        state1 = SimpleAdditiveState(value=100.0)
        metric1 = Metric(spec=spec1, key=key1, state=state1, dataset="default")

        report1 = AnalysisReport({(spec1, key1): metric1})
        report1.symbol_mapping[(spec1, key1)] = "total_col1"

        # Create second report with different data and mappings
        spec2 = Average("col2")
        key2 = ResultKey(datetime.date(2024, 1, 2), {})
        state2 = AverageState(avg=50.0, n=10)
        metric2 = Metric(spec=spec2, key=key2, state=state2, dataset="default")

        report2 = AnalysisReport({(spec2, key2): metric2})
        report2.symbol_mapping[(spec2, key2)] = "average_col2"

        # Merge reports
        merged = report1.merge(report2)

        # Check that merged report has both mappings
        assert len(merged.symbol_mapping) == 2
        assert merged.symbol_mapping[(spec1, key1)] == "total_col1"
        assert merged.symbol_mapping[(spec2, key2)] == "average_col2"

        # Check that data is also merged
        assert len(merged) == 2
        assert (spec1, key1) in merged
        assert (spec2, key2) in merged

    def test_analyzer_with_symbol_lookup(self) -> None:
        """Test that Analyzer accepts symbol_lookup and populates mappings."""
        # Create mock datasource
        mock_ds = MagicMock()
        mock_ds.name = "test_ds"
        mock_ds.cte = MagicMock(return_value="SELECT * FROM test")
        mock_ds.query = MagicMock()
        mock_ds.dialect = "duckdb"

        # Mock query result
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [("2024-01-01", {"sum_col1": 100.0})]
        mock_ds.query.return_value = mock_result

        # Create metric spec and symbol lookup
        spec = Sum("col1")
        key = ResultKey(datetime.date(2024, 1, 1), {})
        symbol_lookup: dict[tuple[MetricSpec, ResultKey], str] = {(spec, key): "total_sales"}

        # Create analyzer with symbol lookup
        analyzer = Analyzer(metadata=Metadata(), symbol_lookup=symbol_lookup)

        # Analyze with the metric
        metrics_by_key = {key: [spec]}

        # Mock the analyze_batch_sql_ops to set the value on the analyzer
        def mock_analyze(ds: Any, ops_by_key: Any) -> None:
            # Set value on the analyzer op
            for key, ops in ops_by_key.items():
                for op in ops:
                    op._value = 100.0

        with patch("dqx.analyzer.analyze_batch_sql_ops", side_effect=mock_analyze):
            report = analyzer.analyze(mock_ds, metrics_by_key)

        # Check that symbol mapping was populated
        assert (spec, key) in report.symbol_mapping
        assert report.symbol_mapping[(spec, key)] == "total_sales"

    def test_verification_suite_stores_analysis_reports(self) -> None:
        """Test that VerificationSuite stores analysis reports by datasource."""

        # Create simple check
        @check(name="Test Check")
        def test_check(mp: Any, ctx: Any) -> None:
            total = mp.sum("value")
            ctx.assert_that(total).where(name="Total positive").is_positive()

        # Create sample data
        data_table = pa.table({"value": [10.0, 20.0, 30.0]})

        # Create suite and run
        db = InMemoryMetricDB()
        ds = DuckRelationDataSource.from_arrow(data_table, "test_data")

        suite = VerificationSuite([test_check], db, "Test Suite")
        key = ResultKey(datetime.date(2024, 1, 1), {})

        # Run the suite
        suite.run([ds], key)

        # Check that analysis reports were stored
        assert hasattr(suite, "_analysis_reports")
        assert isinstance(suite._analysis_reports, dict)
        assert "test_data" in suite._analysis_reports
        assert isinstance(suite._analysis_reports["test_data"], AnalysisReport)

    def test_print_analysis_report(self, capsys: Any) -> None:
        """Test print_analysis_report function displays correctly."""
        # Create test data
        spec1 = Sum("sales")
        spec2 = Average("price")

        key1 = ResultKey(datetime.date(2024, 1, 1), {"env": "prod"})
        key2 = ResultKey(datetime.date(2024, 1, 2), {"env": "test"})

        state1 = SimpleAdditiveState(value=1000.0)
        metric1 = Metric(spec=spec1, key=key1, state=state1, dataset="sales_db")

        state2 = AverageState(avg=50.0, n=10)
        metric2 = Metric(spec=spec2, key=key2, state=state2, dataset="products_db")

        # Create report with symbol mappings
        report = AnalysisReport({(spec1, key1): metric1, (spec2, key2): metric2})
        report.symbol_mapping[(spec1, key1)] = "total_revenue"
        report.symbol_mapping[(spec2, key2)] = "average_price"

        # Print the report
        print_analysis_report({"main": report})

        # Check output contains expected elements
        captured = capsys.readouterr()
        output = captured.out

        assert "Analysis Reports" in output
        assert "2024-01-01" in output
        assert "2024-01-02" in output
        assert "sum" in output
        assert "average" in output
        assert "total_revenue" in output
        assert "average_price" in output
        # Dataset names might be truncated in the display
        assert "sale" in output or "sales_db" in output
        assert "prod" in output or "products_db" in output
        assert "1000.0" in output
        assert "50.0" in output
        # Tags might be truncated in display
        assert "env=" in output

    def test_analysis_report_show_method(self, capsys: Any) -> None:
        """Test AnalysisReport.show() uses print_analysis_report."""
        # Create simple report
        spec = NullCount("id")
        key = ResultKey(datetime.date(2024, 1, 1), {})
        state = SimpleAdditiveState(value=42.0)
        metric = Metric(spec=spec, key=key, state=state, dataset="test_ds")

        report = AnalysisReport({(spec, key): metric})
        report.symbol_mapping[(spec, key)] = "record_count"

        # Call show method
        report.show("my_datasource")

        # Check output
        captured = capsys.readouterr()
        output = captured.out

        assert "Analysis Reports" in output
        assert "record_count" in output
        assert "42.0" in output

    def test_symbol_lookup_integration(self) -> None:
        """Test full integration from suite to symbol mappings."""

        # Create check with symbols
        @check(name="Metrics Check")
        def metrics_check(mp: Any, ctx: Any) -> None:
            # Use descriptive variable names that become symbols
            total_amount = mp.sum("amount")
            unique_users = mp.null_count("user_id")  # Using null_count as an example metric

            ctx.assert_that(total_amount).where(name="Total positive").is_positive()
            ctx.assert_that(unique_users).where(name="Has users").is_gt(0)

        # Create data
        test_data = pa.table({"amount": [100.0, 200.0, 150.0], "user_id": [1, 2, 1]})

        # Run suite
        db = InMemoryMetricDB()
        ds = DuckRelationDataSource.from_arrow(test_data, "transactions")

        suite = VerificationSuite([metrics_check], db, "Test Suite")
        key = ResultKey(datetime.date(2024, 1, 1), {})
        suite.run([ds], key)

        # Get the analysis report
        assert "transactions" in suite._analysis_reports
        report = suite._analysis_reports["transactions"]

        # Check symbol mappings exist
        assert len(report.symbol_mapping) > 0

        # Verify that we have symbol mappings
        symbols = list(report.symbol_mapping.values())
        assert len(symbols) >= 2  # We should have at least 2 symbols

        # The symbols will be x_1, x_2, etc. - not the variable names from the check
        # This is because Python doesn't provide a way to capture local variable names
        # The symbol mapping maps (MetricSpec, ResultKey) -> symbol_name
        assert all(s.startswith("x_") for s in symbols)

        # Verify we have the right metric specs in the mapping
        metric_specs = [key[0] for key in report.symbol_mapping.keys()]
        metric_names = [spec.name for spec in metric_specs]
        assert "sum(amount)" in metric_names
        assert "null_count(user_id)" in metric_names
