"""Integration tests for the DQX plugin system."""

import time
from datetime import datetime
from unittest.mock import Mock, patch

import pytest
from returns.result import Failure, Success

from dqx.api import Context, VerificationSuite, check
from dqx.common import (
    AssertionResult,
    EvaluationFailure,
    PluginExecutionContext,
    PluginMetadata,
    ResultKey,
    SymbolInfo,
)
from dqx.orm.repositories import MetricDB
from dqx.plugins import PluginManager
from dqx.provider import MetricProvider


class CustomPlugin:
    """Test custom plugin for integration tests."""

    @staticmethod
    def metadata() -> PluginMetadata:
        return PluginMetadata(
            name="custom",
            version="1.0.0",
            author="Test",
            description="Custom test plugin",
        )

    def __init__(self) -> None:
        self.contexts: list[PluginExecutionContext] = []

    def process(self, context: PluginExecutionContext) -> None:
        self.contexts.append(context)


class SlowPlugin:
    """Test slow plugin for timeout tests."""

    @staticmethod
    def metadata() -> PluginMetadata:
        return PluginMetadata(
            name="slow",
            version="1.0.0",
            author="Test",
            description="Slow plugin for timeout test",
        )

    def __init__(self) -> None:
        self.started = False

    def process(self, context: PluginExecutionContext) -> None:
        self.started = True
        time.sleep(2)  # This will timeout


class Plugin1:
    """Test plugin 1 for multiple plugin tests."""

    @staticmethod
    def metadata() -> PluginMetadata:
        return PluginMetadata(name="p1", version="1.0.0", author="Test", description="Plugin 1")

    def __init__(self) -> None:
        self.called = False

    def process(self, context: PluginExecutionContext) -> None:
        self.called = True


class Plugin2:
    """Test plugin 2 for multiple plugin tests."""

    @staticmethod
    def metadata() -> PluginMetadata:
        return PluginMetadata(name="p2", version="1.0.0", author="Test", description="Plugin 2")

    def __init__(self) -> None:
        self.called = False

    def process(self, context: PluginExecutionContext) -> None:
        self.called = True


class FailingPlugin:
    """Test plugin that always fails."""

    @staticmethod
    def metadata() -> PluginMetadata:
        return PluginMetadata(
            name="failing",
            version="1.0.0",
            author="Test",
            description="Plugin that fails",
        )

    def process(self, context: PluginExecutionContext) -> None:
        raise RuntimeError("Plugin error!")


class TestPluginIntegration:
    """Integration tests for plugin system with VerificationSuite."""

    def test_suite_with_default_plugin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that VerificationSuite uses default plugin when none specified."""
        # Mock console to capture output
        mock_console_print = Mock()

        with patch("dqx.plugins.Console") as MockConsole:
            mock_console = Mock()
            mock_console.print = mock_console_print
            MockConsole.return_value = mock_console

            # Create check function
            @check(name="test_check")
            def test_check(mp: MetricProvider, ctx: Context) -> None:
                ctx.assert_that(mp.num_rows()).where(name="x > 0", severity="P0").is_positive()

            # Create suite with mocked DB
            db = Mock(spec=MetricDB)
            suite = VerificationSuite([test_check], db, "TestSuite")

            # Mock the suite's context to return our mocked results
            mock_results = [
                AssertionResult(
                    yyyy_mm_dd=datetime.now().date(),
                    suite="TestSuite",
                    check="test_check",
                    assertion="x > 0",
                    severity="P0",
                    status="OK",
                    metric=Success(1.0),
                    expression="x > 0",
                    tags={},
                )
            ]

            # Patch collect_results to return our mock results
            with patch.object(suite, "collect_results", return_value=mock_results):
                with patch.object(suite, "collect_symbols", return_value=[]):
                    # Mock the graph building and execution
                    suite._is_evaluated = True
                    suite._key = ResultKey(datetime.now().date(), {})
                    suite._execution_start = time.time()
                    # Mock the timer to return a duration
                    with patch.object(suite._analyze_ms, "elapsed_ms", return_value=100.0):
                        # Process plugins
                        suite._process_plugins({"ds1": Mock()})

            # Verify audit plugin was called (console output)
            assert mock_console_print.call_count > 0

            # Check that the header was printed
            call_args = [str(call) for call in mock_console_print.call_args_list]
            output = " ".join(call_args)
            assert "DQX Audit Report" in output
            assert "TestSuite" in output

    def test_suite_with_custom_plugin(self) -> None:
        """Test that VerificationSuite can use custom plugins."""
        # Track plugin calls
        plugin_contexts: list[PluginExecutionContext] = []

        class LocalCustomPlugin:
            @staticmethod
            def metadata() -> PluginMetadata:
                return PluginMetadata(
                    name="custom",
                    version="1.0.0",
                    author="Test",
                    description="Custom test plugin",
                )

            def process(self, context: PluginExecutionContext) -> None:
                plugin_contexts.append(context)

        # Create check function
        @check(name="test_check")
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            ctx.assert_that(mp.average("price")).where(name="y < 100", severity="P1").is_lt(100)

        # Create suite
        db = Mock(spec=MetricDB)
        suite = VerificationSuite([test_check], db, "TestSuite")

        # Register custom plugin
        suite.plugin_manager.clear_plugins()  # Remove any default plugins
        # Create and register a custom plugin instance for this test
        custom_plugin = LocalCustomPlugin()
        suite.plugin_manager._plugins["custom"] = custom_plugin

        # Mock evaluator results
        mock_results = [
            AssertionResult(
                yyyy_mm_dd=datetime.now().date(),
                suite="TestSuite",
                check="test_check",
                assertion="y < 100",
                severity="P1",
                status="FAILURE",
                metric=Failure([EvaluationFailure("Value 100 is not less than 100", "y < 100", [])]),
                expression="y < 100",
                tags={},
            )
        ]

        # Patch collect_results to return our mock results
        with patch.object(suite, "collect_results", return_value=mock_results):
            with patch.object(suite, "collect_symbols", return_value=[]):
                # Mock the graph building and execution
                suite._is_evaluated = True
                suite._key = ResultKey(datetime.now().date(), {})
                suite._execution_start = time.time()
                # Mock the timer to return a duration
                with patch.object(suite._analyze_ms, "elapsed_ms", return_value=100.0):
                    # Process plugins
                    suite._process_plugins({"ds1": Mock()})

        # Verify custom plugin was called
        assert len(plugin_contexts) == 1
        context = plugin_contexts[0]

        # Verify context data
        assert context.suite_name == "TestSuite"
        assert len(context.results) == 1
        assert context.results[0].assertion == "y < 100"
        assert context.results[0].status == "FAILURE"

    def test_plugin_context_methods(self) -> None:
        """Test PluginExecutionContext convenience methods work correctly."""
        # Create test data
        results = [
            AssertionResult(
                yyyy_mm_dd=datetime.now().date(),
                suite="TestSuite",
                check="check1",
                assertion="a1",
                severity="P0",
                status="OK",
                metric=Success(1.0),
                expression="x > 0",
                tags={"env": "prod"},
            ),
            AssertionResult(
                yyyy_mm_dd=datetime.now().date(),
                suite="TestSuite",
                check="check1",
                assertion="a2",
                severity="P0",
                status="FAILURE",
                metric=Failure([EvaluationFailure("Assertion failed", "x < 100", [])]),
                expression="x < 100",
                tags={"env": "prod"},
            ),
            AssertionResult(
                yyyy_mm_dd=datetime.now().date(),
                suite="TestSuite",
                check="check2",
                assertion="a3",
                severity="P1",
                status="FAILURE",
                metric=Failure([EvaluationFailure("Assertion failed", "y > 0", [])]),
                expression="y > 0",
                tags={"env": "dev"},
            ),
        ]

        symbols = [
            SymbolInfo(
                name="x",
                metric="avg(price)",
                dataset="ds1",
                value=Success(50.0),
                yyyy_mm_dd=datetime.now().date(),
                suite="TestSuite",
                tags={},
            ),
            SymbolInfo(
                name="y",
                metric="sum(quantity)",
                dataset="ds2",
                value=Failure("Calc failed"),
                yyyy_mm_dd=datetime.now().date(),
                suite="TestSuite",
                tags={},
            ),
        ]

        # Create context
        context = PluginExecutionContext(
            suite_name="TestSuite",
            datasources=["ds1", "ds2"],
            key=ResultKey(datetime.now().date(), {"region": "US"}),
            timestamp=time.time(),
            duration_ms=150.75,
            results=results,
            symbols=symbols,
        )

        # Test assertion methods
        assert context.total_assertions() == 3
        assert context.passed_assertions() == 1
        assert context.failed_assertions() == 2
        assert context.assertion_pass_rate() == pytest.approx(33.33, rel=0.01)

        # Test failures by severity
        failures_by_sev = context.failures_by_severity()
        assert failures_by_sev == {"P0": 1, "P1": 1}

        # Test symbol methods
        assert context.total_symbols() == 2
        assert context.failed_symbols() == 1

        # Test filtering methods - manually filter
        prod_results = [r for r in context.results if r.tags.get("env") == "prod"]
        assert len(prod_results) == 2

        check1_results = [r for r in context.results if r.check == "check1"]
        assert len(check1_results) == 2

        p0_results = [r for r in context.results if r.severity == "P0"]
        assert len(p0_results) == 2

        # Test combined filters
        prod_failures = [r for r in context.results if r.status == "FAILURE" and r.tags.get("env") == "prod"]
        assert len(prod_failures) == 1
        assert prod_failures[0].assertion == "a2"

    def test_plugin_timeout_integration(self) -> None:
        """Test that plugin timeouts work in integration."""
        # Track if plugin started
        plugin_started = False

        class SlowPlugin:
            @staticmethod
            def metadata() -> PluginMetadata:
                return PluginMetadata(
                    name="slow",
                    version="1.0.0",
                    author="Test",
                    description="Slow plugin for timeout test",
                )

            def process(self, context: PluginExecutionContext) -> None:
                nonlocal plugin_started
                plugin_started = True
                time.sleep(2)  # This will timeout

        # Create check function
        @check(name="test")
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            ctx.assert_that(mp.num_rows()).where(name="always_true").is_positive()

        # Create suite
        db = Mock(spec=MetricDB)
        suite = VerificationSuite([test_check], db, "TestSuite")

        # Replace plugin manager with short timeout
        suite._plugin_manager = PluginManager(timeout_seconds=1)
        suite._plugin_manager._plugins = {"slow": SlowPlugin()}

        # Mock the execution
        with patch.object(suite, "collect_results", return_value=[]):
            with patch.object(suite, "collect_symbols", return_value=[]):
                # Mock the graph building and execution
                suite._is_evaluated = True
                suite._key = ResultKey(datetime.now().date(), {})
                suite._execution_start = time.time()
                # Mock the timer to return a duration
                with patch.object(suite._analyze_ms, "elapsed_ms", return_value=100.0):
                    # Process plugins - should timeout
                    suite._process_plugins({"ds1": Mock()})

        # Verify plugin started but was terminated
        assert plugin_started

    def test_multiple_plugins_integration(self) -> None:
        """Test that multiple plugins can process results."""
        plugin1_called = False
        plugin2_called = False

        class Plugin1:
            @staticmethod
            def metadata() -> PluginMetadata:
                return PluginMetadata(name="p1", version="1.0.0", author="Test", description="Plugin 1")

            def process(self, context: PluginExecutionContext) -> None:
                nonlocal plugin1_called
                plugin1_called = True

        class Plugin2:
            @staticmethod
            def metadata() -> PluginMetadata:
                return PluginMetadata(name="p2", version="1.0.0", author="Test", description="Plugin 2")

            def process(self, context: PluginExecutionContext) -> None:
                nonlocal plugin2_called
                plugin2_called = True

        # Create check function
        @check(name="test")
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            ctx.assert_that(mp.num_rows()).where(name="test").is_positive()

        # Create suite
        db = Mock(spec=MetricDB)
        suite = VerificationSuite([test_check], db, "TestSuite")

        # Configure plugins
        suite.plugin_manager.clear_plugins()
        # Create plugin instances and register them directly
        plugin1 = Plugin1()
        plugin2 = Plugin2()
        suite.plugin_manager._plugins["p1"] = plugin1
        suite.plugin_manager._plugins["p2"] = plugin2

        # Mock the execution
        with patch.object(suite, "collect_results", return_value=[]):
            with patch.object(suite, "collect_symbols", return_value=[]):
                # Mock the graph building and execution
                suite._is_evaluated = True
                suite._key = ResultKey(datetime.now().date(), {})
                suite._execution_start = time.time()
                # Mock the timer to return a duration
                with patch.object(suite._analyze_ms, "elapsed_ms", return_value=100.0):
                    # Process plugins
                    suite._process_plugins({"ds1": Mock()})

        # Both plugins should be called
        assert plugin1_called
        assert plugin2_called

    def test_plugin_error_doesnt_fail_suite(self) -> None:
        """Test that plugin errors don't cause suite execution to fail."""

        class FailingPlugin:
            @staticmethod
            def metadata() -> PluginMetadata:
                return PluginMetadata(
                    name="failing",
                    version="1.0.0",
                    author="Test",
                    description="Plugin that fails",
                )

            def process(self, context: PluginExecutionContext) -> None:
                raise RuntimeError("Plugin error!")

        # Create check function
        @check(name="test")
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            ctx.assert_that(mp.num_rows()).where(name="test").is_positive()

        # Create suite
        db = Mock(spec=MetricDB)
        suite = VerificationSuite([test_check], db, "TestSuite")

        # Configure failing plugin
        suite.plugin_manager.clear_plugins()
        suite.plugin_manager._plugins["failing"] = FailingPlugin()

        # Mock evaluator to return success
        mock_results = [
            AssertionResult(
                yyyy_mm_dd=datetime.now().date(),
                suite="TestSuite",
                check="test",
                assertion="test",
                severity="P1",
                status="OK",
                metric=Success(1.0),
                expression="test",
                tags={},
            )
        ]

        # Mock the execution and plugin processing
        with patch.object(suite, "collect_results", return_value=mock_results):
            with patch.object(suite, "collect_symbols", return_value=[]):
                # Mock the graph building and execution
                suite._is_evaluated = True
                suite._key = ResultKey(datetime.now().date(), {})
                suite._execution_start = time.time()
                # Mock the timer to return a duration
                with patch.object(suite._analyze_ms, "elapsed_ms", return_value=100.0):
                    # Process plugins - should not raise
                    suite._process_plugins({"ds1": Mock()})

                # Get results
                results = suite.collect_results()

        # Verify results were returned successfully
        assert len(results) == 1
        assert results[0].status == "OK"
