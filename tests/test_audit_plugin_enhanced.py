"""Test enhanced audit plugin behavior with skipped/error assertions."""

import datetime
from io import StringIO
from unittest.mock import Mock, patch

import pyarrow as pa
from returns.result import Failure, Success
from returns.result import Success as MetricSuccess

from dqx.cache import CacheStats
from dqx.common import AssertionResult, EvaluationFailure, ResultKey
from dqx.orm.repositories import MetricStats
from dqx.plugins import AuditPlugin, PluginExecutionContext
from dqx.provider import SymbolInfo


class TestEnhancedAuditPlugin:
    """Test enhanced audit plugin display behavior."""

    def test_audit_plugin_with_skipped_assertions(self) -> None:
        """Test audit plugin correctly displays skipped assertions."""
        plugin = AuditPlugin()

        # Create context with mixed assertion statuses
        context = PluginExecutionContext(
            suite_name="Test Suite",
            execution_id="test-123",
            datasources=["dataset1"],
            key=ResultKey(yyyy_mm_dd=datetime.date(2025, 11, 6), tags={}),
            timestamp=100.0,
            duration_ms=50.0,
            results=[
                AssertionResult(
                    yyyy_mm_dd=datetime.date(2025, 11, 6),
                    suite="Test Suite",
                    check="Test Check",
                    assertion="Check 1",
                    severity="P1",
                    status="PASSED",
                    metric=MetricSuccess(100.0),
                    expression="x > 0",
                    tags={},
                ),
                AssertionResult(
                    yyyy_mm_dd=datetime.date(2025, 11, 6),
                    suite="Test Suite",
                    check="Test Check",
                    assertion="Check 2",
                    severity="P1",
                    status="PASSED",
                    metric=MetricSuccess(50.0),
                    expression="x < 100",
                    tags={},
                ),
                AssertionResult(
                    yyyy_mm_dd=datetime.date(2025, 11, 6),
                    suite="Test Suite",
                    check="Test Check",
                    assertion="Check 3",
                    severity="P0",
                    status="FAILED",
                    metric=MetricSuccess(0.0),
                    expression="x == 50",
                    tags={},
                ),
                AssertionResult(
                    yyyy_mm_dd=datetime.date(2025, 11, 6),
                    suite="Test Suite",
                    check="Test Check",
                    assertion="Check 4",
                    severity="P1",
                    status="SKIPPED",
                    metric=MetricSuccess(0.0),
                    expression="y > 0",
                    tags={},
                ),
                AssertionResult(
                    yyyy_mm_dd=datetime.date(2025, 11, 6),
                    suite="Test Suite",
                    check="Test Check",
                    assertion="Check 5",
                    severity="P1",
                    status="SKIPPED",
                    metric=MetricSuccess(0.0),
                    expression="y < 100",
                    tags={},
                ),
            ],
            symbols=[],
            trace=pa.table({}),
            metrics_stats=MetricStats(total_metrics=0, expired_metrics=0),
            cache_stats=CacheStats(hit=0, missed=0),
        )

        # Capture output
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            plugin.process(context)
            output = mock_stdout.getvalue()

        # Strip ANSI color codes for easier testing
        import re

        clean_output = re.sub(r"\x1b\[[0-9;]*m", "", output)

        # Verify the assertions line includes skipped
        assert "Assertions: 5 total, 2 passed (40.0%), 1 failed (20.0%), 2 skipped (40.0%)" in clean_output
        assert "error" not in output.lower()  # No errors, so shouldn't be shown

    def test_audit_plugin_with_failed_metric_assertions(self) -> None:
        """Test audit plugin correctly displays failed assertions with metric failures."""
        plugin = AuditPlugin()

        # Create context with error assertions
        context = PluginExecutionContext(
            suite_name="Test Suite",
            execution_id="test-123",
            datasources=["dataset1"],
            key=ResultKey(yyyy_mm_dd=datetime.date(2025, 11, 6), tags={}),
            timestamp=100.0,
            duration_ms=50.0,
            results=[
                AssertionResult(
                    yyyy_mm_dd=datetime.date(2025, 11, 6),
                    suite="Test Suite",
                    check="Test Check",
                    assertion="Check 1",
                    severity="P1",
                    status="PASSED",
                    metric=MetricSuccess(100.0),
                    expression="x > 0",
                    tags={},
                ),
                AssertionResult(
                    yyyy_mm_dd=datetime.date(2025, 11, 6),
                    suite="Test Suite",
                    check="Test Check",
                    assertion="Check 2",
                    severity="P0",
                    status="FAILED",
                    metric=MetricSuccess(0.0),
                    expression="x == 50",
                    tags={},
                ),
                AssertionResult(
                    yyyy_mm_dd=datetime.date(2025, 11, 6),
                    suite="Test Suite",
                    check="Test Check",
                    assertion="Check 3",
                    severity="P0",
                    status="FAILED",
                    metric=Failure(
                        [
                            EvaluationFailure(
                                error_message="Error evaluating expression", expression="invalid", symbols=[]
                            )
                        ]
                    ),
                    expression="invalid",
                    tags={},
                ),
            ],
            symbols=[],
            trace=pa.table({}),
            metrics_stats=MetricStats(total_metrics=0, expired_metrics=0),
            cache_stats=CacheStats(hit=0, missed=0),
        )

        # Capture output
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            plugin.process(context)
            output = mock_stdout.getvalue()

        # Strip ANSI color codes for easier testing
        import re

        clean_output = re.sub(r"\x1b\[[0-9;]*m", "", output)

        # Verify the assertions line includes failed
        assert "Assertions: 3 total, 1 passed (33.3%), 2 failed (66.7%)" in clean_output
        assert "skipped" not in output.lower()  # No skipped, so shouldn't be shown

    def test_audit_plugin_only_passed_assertions(self) -> None:
        """Test audit plugin with only passed assertions (no failed/skipped/error)."""
        plugin = AuditPlugin()

        # Create context with only passed assertions
        context = PluginExecutionContext(
            suite_name="Test Suite",
            execution_id="test-123",
            datasources=["dataset1"],
            key=ResultKey(yyyy_mm_dd=datetime.date(2025, 11, 6), tags={}),
            timestamp=100.0,
            duration_ms=50.0,
            results=[
                AssertionResult(
                    yyyy_mm_dd=datetime.date(2025, 11, 6),
                    suite="Test Suite",
                    check="Test Check",
                    assertion="Check 1",
                    severity="P1",
                    status="PASSED",
                    metric=MetricSuccess(100.0),
                    expression="x > 0",
                    tags={},
                ),
                AssertionResult(
                    yyyy_mm_dd=datetime.date(2025, 11, 6),
                    suite="Test Suite",
                    check="Test Check",
                    assertion="Check 2",
                    severity="P1",
                    status="PASSED",
                    metric=MetricSuccess(50.0),
                    expression="x < 100",
                    tags={},
                ),
            ],
            symbols=[],
            trace=pa.table({}),
            metrics_stats=MetricStats(total_metrics=0, expired_metrics=0),
            cache_stats=CacheStats(hit=0, missed=0),
        )

        # Capture output
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            plugin.process(context)
            output = mock_stdout.getvalue()

        # Strip ANSI color codes for easier testing
        import re

        clean_output = re.sub(r"\x1b\[[0-9;]*m", "", output)

        # Verify only passed is shown (no failed/skipped/error)
        assert "Assertions: 2 total, 2 passed (100.0%)" in clean_output
        assert "failed" not in output.lower()
        assert "skipped" not in output.lower()
        assert "error" not in output.lower()

    def test_audit_plugin_no_symbols_when_all_successful(self) -> None:
        """Test that symbols line IS shown when all symbols are successful."""
        plugin = AuditPlugin()

        # Create context with successful symbols
        symbol1 = Mock(spec=SymbolInfo)
        symbol1.value = Success(100.0)
        symbol2 = Mock(spec=SymbolInfo)
        symbol2.value = Success(200.0)

        context = PluginExecutionContext(
            suite_name="Test Suite",
            execution_id="test-123",
            datasources=["dataset1"],
            key=ResultKey(yyyy_mm_dd=datetime.date(2025, 11, 6), tags={}),
            timestamp=100.0,
            duration_ms=50.0,
            results=[],
            symbols=[symbol1, symbol2],
            trace=pa.table({}),
            metrics_stats=MetricStats(total_metrics=0, expired_metrics=0),
            cache_stats=CacheStats(hit=0, missed=0),
        )

        # Capture output
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            plugin.process(context)
            output = mock_stdout.getvalue()

        # Strip ANSI color codes for easier testing
        import re

        clean_output = re.sub(r"\x1b\[[0-9;]*m", "", output)

        # Verify symbols line IS shown with all successful
        assert "Symbols: 2 total, 2 successful (100.0%)" in clean_output

    def test_audit_plugin_shows_symbols_when_failed(self) -> None:
        """Test that symbols line is shown when there are failed symbols."""
        plugin = AuditPlugin()

        # Create context with mixed symbols
        symbol1 = Mock(spec=SymbolInfo)
        symbol1.value = Success(100.0)
        symbol2 = Mock(spec=SymbolInfo)
        symbol2.value = Failure("Error computing symbol")

        context = PluginExecutionContext(
            suite_name="Test Suite",
            execution_id="test-123",
            datasources=["dataset1"],
            key=ResultKey(yyyy_mm_dd=datetime.date(2025, 11, 6), tags={}),
            timestamp=100.0,
            duration_ms=50.0,
            results=[],
            symbols=[symbol1, symbol2],
            trace=pa.table({}),
            metrics_stats=MetricStats(total_metrics=0, expired_metrics=0),
            cache_stats=CacheStats(hit=0, missed=0),
        )

        # Capture output
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            plugin.process(context)
            output = mock_stdout.getvalue()

        # Strip ANSI color codes for easier testing
        import re

        clean_output = re.sub(r"\x1b\[[0-9;]*m", "", output)

        # Verify symbols line IS shown with failed symbols
        assert "Symbols: 2 total, 1 successful (50.0%), 1 failed (50.0%)" in clean_output

    def test_audit_plugin_all_statuses(self) -> None:
        """Test audit plugin with all assertion statuses present."""
        plugin = AuditPlugin()

        # Create context with all statuses
        context = PluginExecutionContext(
            suite_name="Complex Suite",
            execution_id="test-123",
            datasources=["dataset1", "dataset2"],
            key=ResultKey(yyyy_mm_dd=datetime.date(2025, 11, 6), tags={"env": "test"}),
            timestamp=100.0,
            duration_ms=150.0,
            results=[
                # 3 passed
                AssertionResult(
                    yyyy_mm_dd=datetime.date(2025, 11, 6),
                    suite="Complex Suite",
                    check="Check",
                    assertion="Check 1",
                    severity="P1",
                    status="PASSED",
                    metric=MetricSuccess(100.0),
                    expression="x > 0",
                    tags={"env": "test"},
                ),
                AssertionResult(
                    yyyy_mm_dd=datetime.date(2025, 11, 6),
                    suite="Complex Suite",
                    check="Check",
                    assertion="Check 2",
                    severity="P1",
                    status="PASSED",
                    metric=MetricSuccess(50.0),
                    expression="x < 100",
                    tags={"env": "test"},
                ),
                AssertionResult(
                    yyyy_mm_dd=datetime.date(2025, 11, 6),
                    suite="Complex Suite",
                    check="Check",
                    assertion="Check 3",
                    severity="P2",
                    status="PASSED",
                    metric=MetricSuccess(75.0),
                    expression="y > 0",
                    tags={"env": "test"},
                ),
                # 2 failed
                AssertionResult(
                    yyyy_mm_dd=datetime.date(2025, 11, 6),
                    suite="Complex Suite",
                    check="Check",
                    assertion="Check 4",
                    severity="P0",
                    status="FAILED",
                    metric=MetricSuccess(50.0),
                    expression="x == 50",
                    tags={"env": "test"},
                ),
                AssertionResult(
                    yyyy_mm_dd=datetime.date(2025, 11, 6),
                    suite="Complex Suite",
                    check="Check",
                    assertion="Check 5",
                    severity="P0",
                    status="FAILED",
                    metric=MetricSuccess(75.0),
                    expression="y == 50",
                    tags={"env": "test"},
                ),
                # 1 skipped
                AssertionResult(
                    yyyy_mm_dd=datetime.date(2025, 11, 6),
                    suite="Complex Suite",
                    check="Check",
                    assertion="Check 6",
                    severity="P1",
                    status="SKIPPED",
                    metric=MetricSuccess(0.0),
                    expression="z > 0",
                    tags={"env": "test"},
                ),
                # 1 more failed (making 3 total failed)
                AssertionResult(
                    yyyy_mm_dd=datetime.date(2025, 11, 6),
                    suite="Complex Suite",
                    check="Check",
                    assertion="Check 7",
                    severity="P0",
                    status="FAILED",
                    metric=Failure(
                        [EvaluationFailure(error_message="Error in expression", expression="invalid", symbols=[])]
                    ),
                    expression="invalid",
                    tags={"env": "test"},
                ),
            ],
            symbols=[],
            trace=pa.table({}),
            metrics_stats=MetricStats(total_metrics=100, expired_metrics=5),
            cache_stats=CacheStats(hit=100, missed=20),
        )

        # Capture output
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            plugin.process(context)
            output = mock_stdout.getvalue()

        # Strip ANSI color codes for easier testing
        import re

        clean_output = re.sub(r"\x1b\[[0-9;]*m", "", output)

        # Verify all statuses are shown (normalize whitespace to handle potential line wrapping)
        import re

        # Replace any whitespace (including newlines) with a single space
        normalized_output = re.sub(r"\s+", " ", clean_output)
        assert "Assertions: 7 total, 3 passed (42.9%), 3 failed (42.9%), 1 skipped (14.3%)" in normalized_output
        # Verify other elements (use clean_output for consistent testing)
        assert "Tags: env=test" in clean_output
        assert "Datasets:" in clean_output
        assert "- dataset1" in clean_output
        assert "- dataset2" in clean_output
        assert "Metrics Cleanup: 5 expired metrics removed" in clean_output
        assert "Cache Performance: hit: 100, missed: 20 (83.3% hit rate)" in clean_output
