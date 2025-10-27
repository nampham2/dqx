"""Test the enable_plugins parameter in VerificationSuite.run()."""

from datetime import datetime
from typing import Any
from unittest.mock import Mock, patch

from dqx.api import Context, VerificationSuite, check
from dqx.common import (
    PluginMetadata,
    ResultKey,
)
from dqx.graph.base import NodeVisitor
from dqx.orm.repositories import MetricDB
from dqx.plugins import PluginExecutionContext
from dqx.provider import MetricProvider


class TestPlugin:
    """Test plugin for enable/disable testing."""

    called = False  # Class variable to track calls

    @staticmethod
    def metadata() -> PluginMetadata:
        return PluginMetadata(
            name="test",
            version="1.0.0",
            author="Test",
            description="Test plugin",
        )

    def process(self, context: PluginExecutionContext) -> None:
        TestPlugin.called = True


def test_enable_plugins_true_by_default() -> None:
    """Test that plugins are enabled by default."""
    plugin_called = False

    class LocalTestPlugin:
        @staticmethod
        def metadata() -> PluginMetadata:
            return PluginMetadata(
                name="test",
                version="1.0.0",
                author="Test",
                description="Test plugin",
            )

        def process(self, context: PluginExecutionContext) -> None:
            nonlocal plugin_called
            plugin_called = True

    # Create check
    @check(name="test_check")
    def test_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.num_rows()).where(name="Has rows").is_positive()

    # Create suite
    db = Mock(spec=MetricDB)
    suite = VerificationSuite([test_check], db, "TestSuite")

    # Register test plugin directly
    suite.plugin_manager.clear_plugins()
    suite.plugin_manager._plugins["test"] = LocalTestPlugin()

    # Mock data source
    datasource = Mock()
    datasource.name = "test_ds"
    datasource.query.return_value.fetchone.return_value = (1,)

    # Mock the analyzer and evaluator behavior
    with patch("dqx.api.Analyzer") as MockAnalyzer:
        mock_analyzer = Mock()
        mock_analyzer.report = Mock()
        MockAnalyzer.return_value = mock_analyzer

        with patch("dqx.api.Evaluator") as MockEvaluator:
            mock_evaluator = Mock()

            # Create a side effect that only sets attributes when called with evaluator
            original_bfs = suite._context._graph.bfs

            def mock_bfs(visitor: NodeVisitor) -> None:
                # Call original for non-evaluator visitors (like validation)
                if not isinstance(visitor, Mock) or visitor != mock_evaluator:
                    return original_bfs(visitor)

                # For evaluator, set the required attributes
                from returns.result import Success

                for assertion in suite._context._graph.assertions():
                    assertion._result = "OK"
                    assertion._metric = Success(1.0)

            # Patch bfs method
            with patch.object(suite._context._graph, "bfs", side_effect=mock_bfs):
                MockEvaluator.return_value = mock_evaluator

                # Mock the metric trace method to return empty table
                import pyarrow as pa

                with patch.object(suite, "metric_trace", return_value=pa.table({})):
                    # Run suite - plugins should be enabled by default
                    key = ResultKey(datetime.now().date(), {})
                    suite.run([datasource], key)

    # Plugin should have been called
    assert plugin_called


def test_enable_plugins_false_disables_plugins() -> None:
    """Test that enable_plugins=False disables plugin execution."""
    plugin_called = False

    class LocalTestPlugin:
        @staticmethod
        def metadata() -> PluginMetadata:
            return PluginMetadata(
                name="test",
                version="1.0.0",
                author="Test",
                description="Test plugin",
            )

        def process(self, context: PluginExecutionContext) -> None:
            nonlocal plugin_called
            plugin_called = True

    # Create check
    @check(name="test_check")
    def test_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.num_rows()).where(name="Has rows").is_positive()

    # Create suite
    db = Mock(spec=MetricDB)
    suite = VerificationSuite([test_check], db, "TestSuite")

    # Register test plugin directly
    suite.plugin_manager.clear_plugins()
    suite.plugin_manager._plugins["test"] = LocalTestPlugin()

    # Mock data source
    datasource = Mock()
    datasource.name = "test_ds"
    datasource.query.return_value.fetchone.return_value = (1,)

    # Mock the analyzer and evaluator behavior
    with patch("dqx.api.Analyzer") as MockAnalyzer:
        mock_analyzer = Mock()
        mock_analyzer.report = Mock()
        MockAnalyzer.return_value = mock_analyzer

        with patch("dqx.api.Evaluator") as MockEvaluator:
            mock_evaluator = Mock()

            # Create a side effect that only sets attributes when called with evaluator
            original_bfs = suite._context._graph.bfs

            def mock_bfs(visitor: NodeVisitor) -> None:
                # Call original for non-evaluator visitors (like validation)
                if not isinstance(visitor, Mock) or visitor != mock_evaluator:
                    return original_bfs(visitor)

                # For evaluator, set the required attributes
                from returns.result import Success

                for assertion in suite._context._graph.assertions():
                    assertion._result = "OK"
                    assertion._metric = Success(1.0)

                # Patch bfs method
                with patch.object(suite._context._graph, "bfs", side_effect=mock_bfs):
                    MockEvaluator.return_value = mock_evaluator

                    # Mock the metric trace method to return empty table
                    import pyarrow as pa

                    with patch.object(suite, "metric_trace", return_value=pa.table({})):
                        # Run suite with plugins disabled
                        key = ResultKey(datetime.now().date(), {})
                        suite.run([datasource], key, enable_plugins=False)

    # Plugin should NOT have been called
    assert not plugin_called


def test_enable_plugins_true_explicit() -> None:
    """Test that enable_plugins=True explicitly enables plugins."""
    plugin_called = False

    class LocalTestPlugin:
        @staticmethod
        def metadata() -> PluginMetadata:
            return PluginMetadata(
                name="test",
                version="1.0.0",
                author="Test",
                description="Test plugin",
            )

        def process(self, context: PluginExecutionContext) -> None:
            nonlocal plugin_called
            plugin_called = True

    # Create check
    @check(name="test_check")
    def test_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.num_rows()).where(name="Has rows").is_positive()

    # Create suite
    db = Mock(spec=MetricDB)
    suite = VerificationSuite([test_check], db, "TestSuite")

    # Register test plugin directly
    suite.plugin_manager.clear_plugins()
    suite.plugin_manager._plugins["test"] = LocalTestPlugin()

    # Mock data source
    datasource = Mock()
    datasource.name = "test_ds"
    datasource.query.return_value.fetchone.return_value = (1,)

    # Mock the analyzer and evaluator behavior
    with patch("dqx.api.Analyzer") as MockAnalyzer:
        mock_analyzer = Mock()
        mock_analyzer.report = Mock()
        MockAnalyzer.return_value = mock_analyzer

        with patch("dqx.api.Evaluator") as MockEvaluator:
            mock_evaluator = Mock()

            # Create a side effect that only sets attributes when called with evaluator
            original_bfs = suite._context._graph.bfs

            def mock_bfs(visitor: NodeVisitor) -> None:
                # Call original for non-evaluator visitors (like validation)
                if not isinstance(visitor, Mock) or visitor != mock_evaluator:
                    return original_bfs(visitor)

                # For evaluator, set the required attributes
                from returns.result import Success

                for assertion in suite._context._graph.assertions():
                    assertion._result = "OK"
                    assertion._metric = Success(1.0)

            # Patch bfs method
            with patch.object(suite._context._graph, "bfs", side_effect=mock_bfs):
                MockEvaluator.return_value = mock_evaluator

                # Mock the metric trace method to return empty table
                import pyarrow as pa

                with patch.object(suite, "metric_trace", return_value=pa.table({})):
                    # Run suite with plugins explicitly enabled
                    key = ResultKey(datetime.now().date(), {})
                    suite.run([datasource], key, enable_plugins=True)

    # Plugin should have been called
    assert plugin_called


def test_process_plugins_not_called_when_disabled() -> None:
    """Test that _process_plugins is not called when plugins are disabled."""

    # Create check
    @check(name="test_check")
    def test_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.num_rows()).where(name="Has rows").is_positive()

    # Create suite
    db = Mock(spec=MetricDB)
    suite = VerificationSuite([test_check], db, "TestSuite")

    # Mock data source
    datasource = Mock()
    datasource.name = "test_ds"
    datasource.query.return_value.fetchone.return_value = (1,)

    # Mock process_plugins to track if it's called
    process_plugins_called = False
    original_process_plugins = suite._process_plugins

    def mock_process_plugins(datasources: list[Any]) -> None:
        nonlocal process_plugins_called
        process_plugins_called = True
        original_process_plugins(datasources)

    suite._process_plugins = mock_process_plugins  # type: ignore

    # Mock the analyzer and evaluator behavior
    with patch("dqx.api.Analyzer") as MockAnalyzer:
        mock_analyzer = Mock()
        mock_analyzer.report = Mock()
        MockAnalyzer.return_value = mock_analyzer

        with patch("dqx.api.Evaluator") as MockEvaluator:
            mock_evaluator = Mock()

            # Create a side effect that only sets attributes when called with evaluator
            original_bfs = suite._context._graph.bfs

            def mock_bfs(visitor: NodeVisitor) -> None:
                # Call original for non-evaluator visitors (like validation)
                if not isinstance(visitor, Mock) or visitor != mock_evaluator:
                    return original_bfs(visitor)

                # For evaluator, set the required attributes
                from returns.result import Success

                for assertion in suite._context._graph.assertions():
                    assertion._result = "OK"
                    assertion._metric = Success(1.0)

            # Patch bfs method
            with patch.object(suite._context._graph, "bfs", side_effect=mock_bfs):
                MockEvaluator.return_value = mock_evaluator

                # Mock the metric trace method to return empty table
                import pyarrow as pa

                with patch.object(suite, "metric_trace", return_value=pa.table({})):
                    # Run suite with plugins disabled
                    key = ResultKey(datetime.now().date(), {})
                    suite.run([datasource], key, enable_plugins=False)

    # _process_plugins should NOT have been called
    assert not process_plugins_called
