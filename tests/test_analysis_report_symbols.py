"""
Test the new functionality for AnalysisReport and symbol lookups.

Tests cover:
1. AnalysisReport functionality without symbol_mapping
2. AnalysisReport.merge() functionality
3. Analyzer with provider symbol lookup
4. VerificationSuite storing analysis reports
5. Symbol lookup integration with provider
"""

import datetime
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pyarrow as pa

from dqx.analyzer import AnalysisReport, Analyzer
from dqx.api import VerificationSuite, check
from dqx.common import ResultKey
from dqx.datasource import DuckRelationDataSource
from dqx.models import Metric
from dqx.orm.repositories import InMemoryMetricDB
from dqx.provider import MetricProvider

# Import actual metric spec classes
from dqx.specs import Average, Sum
from dqx.states import Average as AverageState
from dqx.states import SimpleAdditiveState


class TestAnalysisReportSymbols:
    """Test AnalysisReport functionality and symbol lookups."""

    def test_analysis_report_init_without_symbol_mapping(self) -> None:
        """Test that AnalysisReport works without symbol_mapping attribute."""
        report = AnalysisReport()
        # AnalysisReport no longer has symbol_mapping attribute
        assert not hasattr(report, "symbol_mapping")
        assert len(report) == 0

    def test_analysis_report_merge_functionality(self) -> None:
        """Test that merge() works correctly with metrics."""
        # Create first report with some data
        spec1 = Sum("col1")
        key1 = ResultKey(datetime.date(2024, 1, 1), {})
        state1 = SimpleAdditiveState(value=100.0)
        metric1 = Metric(spec=spec1, key=key1, state=state1, dataset="default")

        report1 = AnalysisReport({(spec1, key1, "default"): metric1})

        # Create second report with different data
        spec2 = Average("col2")
        key2 = ResultKey(datetime.date(2024, 1, 2), {})
        state2 = AverageState(avg=50.0, n=10)
        metric2 = Metric(spec=spec2, key=key2, state=state2, dataset="default")

        report2 = AnalysisReport({(spec2, key2, "default"): metric2})

        # Merge reports
        merged = report1.merge(report2)

        # Check that data is merged
        assert len(merged) == 2
        assert (spec1, key1, "default") in merged
        assert (spec2, key2, "default") in merged

    def test_analyzer_with_symbol_lookup(self) -> None:
        """Test that Analyzer works with provider symbol lookup."""
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

        # Create metric spec
        spec = Sum("col1")
        key = ResultKey(datetime.date(2024, 1, 1), {})

        # Create provider and register a metric
        mock_db = Mock()
        provider = MetricProvider(mock_db, execution_id="test-123")

        # Register a metric - this will create a symbol
        provider.sum("col1", dataset="test_ds")

        # Create analyzer with provider
        analyzer = Analyzer([mock_ds], provider, key, "test-123")

        # Analyze with the metric
        metrics_by_key = {key: [spec]}

        # Mock the analyze_batch_sql_ops to set the value on the analyzer
        def mock_analyze(ds: Any, ops_by_key: Any) -> None:
            # Set value on the analyzer op
            for key, ops in ops_by_key.items():
                for op in ops:
                    op._value = 100.0

        with patch("dqx.analyzer.analyze_batch_sql_ops", side_effect=mock_analyze):
            analyzer.analyze_simple_metrics(mock_ds, metrics_by_key)

        # Check that symbol exists in provider's registry
        symbol_lookup = provider.registry.symbol_lookup_table(key)
        # Should have at least one symbol entry
        assert len(symbol_lookup) > 0
        # The symbol should be in the format "x_1", "x_2" etc
        symbols = list(symbol_lookup.values())
        assert all(str(s).startswith("x_") for s in symbols)

    def test_verification_suite_stores_analysis_reports(self) -> None:
        """Test that VerificationSuite stores analysis reports."""

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
        # _analysis_reports is now an AnalysisReport object, not a dict
        assert isinstance(suite._analysis_reports, AnalysisReport)
        # Check that it contains metrics
        assert len(suite._analysis_reports) > 0
        # Verify it has the expected metric
        metric_keys = list(suite._analysis_reports.keys())
        assert any(key[2] == "test_data" for key in metric_keys)  # key[2] is the dataset name

    def test_symbol_lookup_integration(self) -> None:
        """Test full integration from suite to symbol lookups via provider."""

        # Create check with symbols
        @check(name="Metrics Check")
        def metrics_check(mp: Any, ctx: Any) -> None:
            # Use descriptive variable names that become symbols
            total_amount = mp.sum("amount")
            unique_users = mp.null_count("user_id")  # Using null_count as an example metric

            ctx.assert_that(total_amount).where(name="Total positive").is_positive()
            ctx.assert_that(unique_users).where(name="Has users").is_geq(0)  # Changed to >=0 since null_count can be 0

        # Create data
        test_data = pa.table({"amount": [100.0, 200.0, 150.0], "user_id": [1, 2, 1]})

        # Run suite
        db = InMemoryMetricDB()
        ds = DuckRelationDataSource.from_arrow(test_data, "transactions")

        suite = VerificationSuite([metrics_check], db, "Test Suite")
        key = ResultKey(datetime.date(2024, 1, 1), {})
        suite.run([ds], key)

        # Get the analysis report
        assert hasattr(suite, "_analysis_reports")
        assert isinstance(suite._analysis_reports, AnalysisReport)

        # Check that metrics for "transactions" dataset exist
        transaction_metrics = [(k, v) for k, v in suite._analysis_reports.items() if k[2] == "transactions"]
        assert len(transaction_metrics) > 0

        # Get symbol lookup from provider
        # The suite has a provider property
        provider = suite.provider
        symbol_lookup = provider.registry.symbol_lookup_table(key)

        # Check symbol lookups exist
        assert len(symbol_lookup) > 0

        # Verify that we have symbol mappings
        symbols = list(symbol_lookup.values())
        assert len(symbols) >= 2  # We should have at least 2 symbols

        # The symbols will be x_1, x_2, etc. - not the variable names from the check
        # This is because Python doesn't provide a way to capture local variable names
        # The symbol mapping maps (MetricSpec, ResultKey, dataset) -> symbol_name
        assert all(str(s).startswith("x_") for s in symbols)

        # Verify we have the right metric specs in the mapping
        metric_specs = [key[0] for key in symbol_lookup.keys()]
        metric_names = [spec.name for spec in metric_specs]
        assert "sum(amount)" in metric_names
        assert "null_count(user_id)" in metric_names
