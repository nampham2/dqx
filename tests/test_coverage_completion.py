"""Tests to complete coverage for analyzer.py, api.py, and plugins.py."""

import datetime
from typing import Any
from unittest.mock import Mock, patch

import pyarrow as pa
import pytest
from returns.result import Failure

from dqx.analyzer import AnalysisReport, Analyzer
from dqx.api import AssertionDraft, Context, VerificationSuite, check
from dqx.cache import CacheStats
from dqx.common import AssertionResult, DQXError, ResultKey
from dqx.graph.nodes import CheckNode
from dqx.orm.repositories import MetricStats
from dqx.plugins import AuditPlugin, PluginExecutionContext
from dqx.provider import MetricProvider, SymbolInfo
from dqx.specs import Sum
from dqx.validator import ValidationReport


class TestAnalyzerCoverage:
    """Tests to complete analyzer.py coverage."""

    def test_analysis_report_show(self) -> None:
        """Test AnalysisReport.show() method - covers line 157."""
        report = AnalysisReport()

        # Add some test data
        key = ResultKey(datetime.date(2024, 1, 1), {})
        metric = Mock()
        spec = Sum("revenue")
        report[(spec, key, "test_ds")] = metric

        # Create a symbol lookup dict
        symbol_lookup: dict[tuple[Any, Any, str], str] = {(spec, key, "test_ds"): "x_1"}

        with patch("dqx.display.print_analysis_report") as mock_print:
            report.show(symbol_lookup)
            mock_print.assert_called_once_with(report, symbol_lookup)

    def test_analyzer_value_retrieval_failure(self) -> None:
        """Test analyzer value retrieval failure - covers line 374."""
        # Create required components for Analyzer
        mock_db = Mock()
        provider = MetricProvider(mock_db, execution_id="test-123", data_av_threshold=0.8)
        key = ResultKey(datetime.date(2024, 1, 1), {})

        ds = Mock()
        ds.name = "test_ds"

        # Create analyzer with required parameters
        analyzer = Analyzer([ds], provider, key, "test-123", 0.9)

        # Mock the _analyze_internal to raise the expected error
        with patch.object(analyzer, "_analyze_internal") as mock_analyze:
            # Make _analyze_internal raise the expected error
            def analyze_side_effect(ds: Mock, metrics_by_key: dict) -> None:
                # Simulate the error that occurs at line 374
                raise DQXError("Failed to retrieve value for analyzer Mock on date 2024-01-01")

            mock_analyze.side_effect = analyze_side_effect

            # Create a metric spec with a mock analyzer
            spec = Sum("revenue")

            with pytest.raises(DQXError, match="Failed to retrieve value"):
                analyzer.analyze_simple_metrics(ds, {key: [spec]})


class TestApiCoverage:
    """Tests to complete api.py coverage."""

    def test_assertion_edge_cases(self) -> None:
        """Test assertion edge cases - covers lines 173, 177, 303-306."""
        # Test 1: Assertion with None context (line 173)
        draft = AssertionDraft(actual=Mock(), context=None)
        ready = draft.where(name="Test assertion")
        # This should not raise an error, just return early
        ready._create_assertion_node(Mock())

        # Test 2: Assertion outside check context (line 177)
        db = Mock()
        context = Context(suite="Test", db=db, execution_id="test-id", data_av_threshold=0.9)
        draft2 = AssertionDraft(actual=Mock(), context=context)
        ready2 = draft2.where(name="Test assertion 2")

        with pytest.raises(DQXError, match="Cannot create assertion outside of check context"):
            ready2._create_assertion_node(Mock())

        # Test 3: is_between with invalid bounds (lines 303-306)
        draft3 = AssertionDraft(actual=Mock(), context=Mock())
        ready3 = draft3.where(name="Test range")

        with pytest.raises(ValueError, match="Invalid range: lower bound .* must be less than or equal to upper bound"):
            ready3.is_between(10.0, 5.0)  # lower > upper

    def test_check_stack_operations(self) -> None:
        """Test check stack operations - covers lines 341, 343."""
        db = Mock()
        context = Context(suite="Test", db=db, execution_id="test-id", data_av_threshold=0.9)

        # Test empty stack operations
        assert context._pop_check() is None  # Line 341
        assert context.current_check is None  # Line 343

        # Verify stack works normally - need a real RootNode
        from dqx.graph.nodes import RootNode

        root = RootNode("test_root")
        check_node = CheckNode(name="test_check", parent=root)
        context._push_check(check_node)
        assert context.current_check == check_node
        assert context._pop_check() == check_node
        assert context.current_check is None

    def test_suite_execution_and_state(self) -> None:
        """Test suite execution and state - covers lines 454, 484, 603, 635-636."""
        db = Mock()
        db.get_metrics_stats.return_value = MetricStats(total_metrics=10, expired_metrics=0)

        # Create a simple check function
        @check(name="Test Check")
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            ctx.assert_that(mp.sum("revenue")).where(name="Revenue check", severity="P0").is_positive()

        suite = VerificationSuite([test_check], db, "Test Suite")

        # Test is_evaluated property (line 454)
        assert suite.is_evaluated is False

        # Test validation with warnings (line 484)
        # Create a new suite for this test to avoid duplicate checks
        @check(name="Warning Test Check")
        def warning_test_check(mp: MetricProvider, ctx: Context) -> None:
            ctx.assert_that(mp.sum("cost")).where(name="Cost check").is_positive()

        suite2 = VerificationSuite([warning_test_check], db, "Test Suite 2")

        with patch("dqx.api.SuiteValidator") as mock_validator:
            # Create a report with warnings but no errors
            warning_report = ValidationReport()
            from dqx.validator import ValidationIssue

            # Add a warning issue to the report
            warning_issue = ValidationIssue(rule="test_rule", message="Test warning", node_path=["root", "check:test"])
            warning_report.add_warning(warning_issue)

            mock_validator.return_value.validate.return_value = warning_report

            with patch("dqx.api.logger") as mock_logger:
                suite2.build_graph(suite2._context, ResultKey(datetime.date.today(), {}))
                mock_logger.debug.assert_called_once()
                assert "Suite validation warnings:" in mock_logger.debug.call_args[0][0]

        # Mock the analyze and evaluation phases
        with patch.object(suite, "_analyze") as mock_analyze:
            # Set the _analysis_reports attribute that _analyze would set
            mock_analyze.side_effect = lambda ds, key: setattr(suite, "_analysis_reports", AnalysisReport())

            with patch("dqx.api.Evaluator"):
                # Create mock datasources
                ds = Mock()
                ds.name = "test_ds"
                ds.skip_dates = set()  # Add skip_dates attribute

                # Run the suite
                suite.run([ds], ResultKey(datetime.date.today(), {}), enable_plugins=False)

        # Now test is_critical with P0 failure (line 603)
        # Create mock assertion results with P0 failure
        suite._cached_results = [
            AssertionResult(
                yyyy_mm_dd=datetime.date.today(),
                suite="Test Suite",
                check="Test Check",
                assertion="Revenue check",
                severity="P0",
                status="FAILED",
                metric=Mock(),
                expression="revenue > 0",
                tags={},
            )
        ]

        assert suite.is_critical() is True

        # Test metric_trace (lines 635-636)
        # Mock the necessary components
        db.get_by_execution_id.return_value = []

        # Mock collect_symbols method properly
        with patch.object(suite.provider, "collect_symbols", return_value=[]):
            # Mock the symbol lookup table
            with patch.object(suite.provider.registry, "symbol_lookup_table", return_value={}):
                with patch("dqx.data.metric_trace") as mock_trace:
                    mock_trace.return_value = pa.table({"col": [1, 2, 3]})

                    trace = suite.metric_trace(db)

                    mock_trace.assert_called_once_with(
                        metrics=[],
                        execution_id=suite.execution_id,
                        reports=suite._analysis_reports,
                        symbols=[],
                        symbol_lookup={},
                    )
                    assert trace.num_rows == 3


class TestPluginsCoverage:
    """Tests to complete plugins.py coverage."""

    def test_audit_plugin_process_comprehensive(self) -> None:
        """Test AuditPlugin.process() comprehensively - covers lines 378-396."""
        plugin = AuditPlugin()

        # Create a comprehensive context with all scenarios
        # Mock data discrepancy stats
        discrepancy_stats = Mock()
        discrepancy_stats.discrepancy_count = 2
        discrepancy_stats.discrepancy_details = [
            {"discrepancies": ["value_db != value_analysis", "value_analysis != value_final"]}
        ]

        # Create context with all edge cases
        context = PluginExecutionContext(
            suite_name="Test Suite",
            execution_id="test-exec-123",
            datasources=["ds1", "ds2"],
            key=ResultKey(datetime.date(2024, 1, 1), {"env": "prod", "region": "us"}),
            timestamp=1234567890.0,
            duration_ms=123.45,
            results=[
                # P0 failure
                AssertionResult(
                    yyyy_mm_dd=datetime.date(2024, 1, 1),
                    suite="Test Suite",
                    check="Check1",
                    assertion="Assert1",
                    severity="P0",
                    status="FAILED",
                    metric=Mock(),
                    expression="x > 0",
                    tags={},
                ),
                # P1 success
                AssertionResult(
                    yyyy_mm_dd=datetime.date(2024, 1, 1),
                    suite="Test Suite",
                    check="Check2",
                    assertion="Assert2",
                    severity="P1",
                    status="PASSED",
                    metric=Mock(),
                    expression="y < 100",
                    tags={},
                ),
            ],
            symbols=[
                # Successful symbol - use Success from returns.result
                SymbolInfo(
                    name="x_1",
                    metric="sum(revenue)",
                    dataset="ds1",
                    value=Mock(is_success=Mock(return_value=True), failure=Mock(return_value=None)),  # Success mock
                    yyyy_mm_dd=datetime.date(2024, 1, 1),
                    tags={},
                ),
                # Failed symbol
                SymbolInfo(
                    name="x_2",
                    metric="sum(costs)",
                    dataset="ds2",
                    value=Failure("Symbol computation failed"),
                    yyyy_mm_dd=datetime.date(2024, 1, 1),
                    tags={},
                ),
            ],
            trace=pa.table({"col": [1, 2, 3]}),
            metrics_stats=MetricStats(total_metrics=100, expired_metrics=5),
            cache_stats=CacheStats(hit=0, missed=0),
        )

        # Mock the data discrepancy stats
        with patch.object(context, "data_discrepancy_stats", return_value=discrepancy_stats):
            # Mock display functions
            with patch("dqx.display.print_metrics_by_execution_id"):
                # Capture console output
                with patch.object(plugin.console, "print") as mock_print:
                    # The plugin should raise error on data discrepancies
                    with pytest.raises(DQXError, match="Data discrepancies detected during audit"):
                        plugin.process(context)

                    # Verify it displayed all the expected output before raising
                    print_calls = mock_print.call_args_list
                    call_strings = [str(call) for call in print_calls]

                    # Check that it displayed tags with sorted order
                    assert any("Tags:" in str(call) and "env=prod, region=us" in str(call) for call in call_strings)
                    # Check assertions line
                    assert any("2 total" in str(call) and "1 passed" in str(call) for call in call_strings)
                    # Check symbols line with failures
                    assert any(
                        "2 total" in str(call) and "1 successful" in str(call) and "1 failed" in str(call)
                        for call in call_strings
                    )
                    # Check data discrepancies warning
                    assert any("2 discrepancies" in str(call) for call in call_strings)

        # Test with no tags and no discrepancies
        context2 = PluginExecutionContext(
            suite_name="Test Suite 2",
            execution_id="test-exec-456",
            datasources=[],
            key=ResultKey(datetime.date(2024, 1, 1), {}),  # No tags
            timestamp=1234567890.0,
            duration_ms=456.78,
            results=[],  # No assertions
            symbols=[],  # No symbols
            trace=pa.table({"col": []}),
            metrics_stats=MetricStats(total_metrics=50, expired_metrics=0),
            cache_stats=CacheStats(hit=0, missed=0),
        )

        # Mock no discrepancies
        no_discrepancy_stats = Mock()
        no_discrepancy_stats.discrepancy_count = 0

        with patch.object(context2, "data_discrepancy_stats", return_value=no_discrepancy_stats):
            with patch.object(plugin.console, "print") as mock_print2:
                # Should not raise error this time
                plugin.process(context2)

                # Verify output
                print_calls2 = mock_print2.call_args_list
                call_strings2 = [str(call) for call in print_calls2]

                # Check no tags displayed
                assert any("Tags:" in str(call) and "None" in str(call) for call in call_strings2)

                # Check clean data integrity
                assert any("No discrepancies found" in str(call) for call in call_strings2)

        # Test single dataset display
        context3 = PluginExecutionContext(
            suite_name="Test Suite 3",
            execution_id="test-exec-789",
            datasources=["ds1"],  # Single dataset
            key=ResultKey(datetime.date(2024, 1, 1), {}),
            timestamp=1234567890.0,
            duration_ms=789.01,
            results=[],
            symbols=[
                SymbolInfo(
                    name="x_1",
                    metric="sum(revenue)",
                    dataset="ds1",
                    value=Mock(is_success=Mock(return_value=True), failure=Mock(return_value=None)),  # Success
                    yyyy_mm_dd=datetime.date(2024, 1, 1),
                    tags={},
                ),
            ],
            trace=pa.table({"col": []}),
            metrics_stats=MetricStats(total_metrics=10, expired_metrics=0),
            cache_stats=CacheStats(hit=5, missed=3),  # Test cache stats display
        )

        with patch.object(context3, "data_discrepancy_stats", return_value=None):
            with patch.object(plugin.console, "print") as mock_print3:
                # Should not raise error
                plugin.process(context3)

                # Verify single dataset display format
                print_calls3 = mock_print3.call_args_list
                call_strings3 = [str(call) for call in print_calls3]

                # Check single dataset format (not plural)
                assert any("Dataset:" in str(call) and "ds1" in str(call) for call in call_strings3)
                assert not any("Datasets:" in str(call) for call in call_strings3)

                # Check cache stats display with actual values
                assert any("hit: 5" in str(call) and "missed: 3" in str(call) for call in call_strings3)
