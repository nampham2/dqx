"""Test the enable_plugins parameter in VerificationSuite.run()."""

from datetime import datetime
from typing import Any

import pyarrow as pa
import pytest

from dqx.api import Context, VerificationSuite, check
from dqx.common import (
    PluginMetadata,
    ResultKey,
)
from dqx.datasource import DuckRelationDataSource
from dqx.orm.repositories import InMemoryMetricDB
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


@pytest.fixture
def test_data() -> pa.Table:
    """Create test data for the tests."""
    return pa.table({"value": [1, 2, 3, 4, 5], "category": ["A", "B", "A", "B", "C"]})


@pytest.fixture
def test_db() -> InMemoryMetricDB:
    """Create a test database."""
    return InMemoryMetricDB()


def test_enable_plugins_true_by_default(test_data: pa.Table, test_db: InMemoryMetricDB) -> None:
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
    suite = VerificationSuite([test_check], test_db, "TestSuite")

    # Register test plugin directly
    suite.plugin_manager.clear_plugins()
    suite.plugin_manager._plugins["test"] = LocalTestPlugin()

    # Create data source
    datasource = DuckRelationDataSource.from_arrow(test_data, "test_ds")

    # Run suite - plugins should be enabled by default
    key = ResultKey(datetime.now().date(), {})
    suite.run([datasource], key)

    # Plugin should have been called
    assert plugin_called


def test_enable_plugins_false_disables_plugins(test_data: pa.Table, test_db: InMemoryMetricDB) -> None:
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
    suite = VerificationSuite([test_check], test_db, "TestSuite")

    # Register test plugin directly
    suite.plugin_manager.clear_plugins()
    suite.plugin_manager._plugins["test"] = LocalTestPlugin()

    # Create data source
    datasource = DuckRelationDataSource.from_arrow(test_data, "test_ds")

    # Run suite with plugins disabled
    key = ResultKey(datetime.now().date(), {})
    suite.run([datasource], key, enable_plugins=False)

    # Plugin should NOT have been called
    assert not plugin_called


def test_enable_plugins_true_explicit(test_data: pa.Table, test_db: InMemoryMetricDB) -> None:
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
    suite = VerificationSuite([test_check], test_db, "TestSuite")

    # Register test plugin directly
    suite.plugin_manager.clear_plugins()
    suite.plugin_manager._plugins["test"] = LocalTestPlugin()

    # Create data source
    datasource = DuckRelationDataSource.from_arrow(test_data, "test_ds")

    # Run suite with plugins explicitly enabled
    key = ResultKey(datetime.now().date(), {})
    suite.run([datasource], key, enable_plugins=True)

    # Plugin should have been called
    assert plugin_called


def test_process_plugins_not_called_when_disabled(test_data: pa.Table, test_db: InMemoryMetricDB) -> None:
    """Test that _process_plugins is not called when plugins are disabled."""

    # Create check
    @check(name="test_check")
    def test_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.num_rows()).where(name="Has rows").is_positive()

    # Create suite
    suite = VerificationSuite([test_check], test_db, "TestSuite")

    # Create data source
    datasource = DuckRelationDataSource.from_arrow(test_data, "test_ds")

    # Mock process_plugins to track if it's called
    process_plugins_called = False
    original_process_plugins = suite._process_plugins

    def mock_process_plugins(datasources: list[Any]) -> None:
        nonlocal process_plugins_called
        process_plugins_called = True
        original_process_plugins(datasources)

    suite._process_plugins = mock_process_plugins  # type: ignore

    # Run suite with plugins disabled
    key = ResultKey(datetime.now().date(), {})
    suite.run([datasource], key, enable_plugins=False)

    # _process_plugins should NOT have been called
    assert not process_plugins_called


def test_plugin_receives_correct_context(test_data: pa.Table, test_db: InMemoryMetricDB) -> None:
    """Test that plugins receive the correct execution context."""
    captured_context = None

    class CaptureContextPlugin:
        @staticmethod
        def metadata() -> PluginMetadata:
            return PluginMetadata(
                name="capture",
                version="1.0.0",
                author="Test",
                description="Captures execution context",
            )

        def process(self, context: PluginExecutionContext) -> None:
            nonlocal captured_context
            captured_context = context

    # Create check with assertions that will pass
    @check(name="test_check")
    def test_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.num_rows()).where(name="Row count check", severity="P0").is_eq(5.0)
        ctx.assert_that(mp.sum("value")).where(name="Sum check", severity="P1").is_eq(15.0)

    # Create suite
    suite = VerificationSuite([test_check], test_db, "TestSuite")

    # Register capture plugin
    suite.plugin_manager.clear_plugins()
    suite.plugin_manager._plugins["capture"] = CaptureContextPlugin()

    # Create data source
    datasource = DuckRelationDataSource.from_arrow(test_data, "test_ds")

    # Run suite
    key = ResultKey(datetime(2024, 1, 1).date(), {"env": "test"})
    suite.run([datasource], key)

    # Verify captured context
    assert captured_context is not None
    assert captured_context.suite_name == "TestSuite"
    assert captured_context.execution_id == suite.execution_id
    assert captured_context.datasources == ["test_ds"]
    assert captured_context.key.yyyy_mm_dd == datetime(2024, 1, 1).date()
    assert captured_context.key.tags == {"env": "test"}
    assert len(captured_context.results) == 2
    assert all(r.status == "OK" for r in captured_context.results)
    assert captured_context.metrics_stats.total_metrics >= 0
    assert isinstance(captured_context.trace, pa.Table)
