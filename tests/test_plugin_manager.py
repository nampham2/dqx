"""Tests for the PluginManager class."""

import time
from datetime import datetime

import pytest
from returns.result import Failure, Success

from dqx.common import (
    AssertionResult,
    EvaluationFailure,
    PluginExecutionContext,
    PluginMetadata,
    ResultKey,
    SymbolInfo,
)
from dqx.plugins import AuditPlugin, PluginManager


class TestPluginManager:
    """Test cases for PluginManager."""

    def test_plugin_manager_initialization(self) -> None:
        """Test that PluginManager initializes correctly."""
        manager = PluginManager()

        # Should have at least the built-in audit plugin
        plugins = manager.get_plugins()
        assert "audit" in plugins
        assert isinstance(plugins["audit"], AuditPlugin)

    def test_plugin_manager_custom_timeout(self) -> None:
        """Test PluginManager with custom timeout."""
        manager = PluginManager(_timeout_seconds=10)
        assert manager._timeout == 10

    def test_get_metadata(self) -> None:
        """Test getting metadata from all plugins."""
        manager = PluginManager()
        metadata = manager.get_metadata()

        # Check audit plugin metadata
        assert "audit" in metadata
        audit_meta = metadata["audit"]
        assert audit_meta.name == "audit"
        assert audit_meta.version == "1.0.0"
        assert audit_meta.author == "DQX Team"
        assert "console_output" in audit_meta.capabilities

    def test_process_all_with_no_plugins(self) -> None:
        """Test process_all when no plugins are loaded."""
        manager = PluginManager()
        manager._plugins = {}  # Clear plugins

        # Create a minimal context
        context = PluginExecutionContext(
            suite_name="Test",
            datasources=[],
            key=ResultKey(datetime.now().date(), {}),
            timestamp=time.time(),
            duration_ms=0.0,
            results=[],
            symbols=[],
        )

        # Should not raise any errors
        manager.process_all(context)

    def test_process_all_with_plugin_error(self) -> None:
        """Test process_all handles plugin errors gracefully."""

        # Create a failing plugin
        class FailingPlugin:
            @staticmethod
            def metadata() -> PluginMetadata:
                return PluginMetadata(
                    name="failing",
                    version="1.0.0",
                    author="Test",
                    description="A plugin that fails",
                )

            def process(self, context: PluginExecutionContext) -> None:
                raise RuntimeError("Plugin failed!")

        manager = PluginManager()

        # Create a proper context
        context = PluginExecutionContext(
            suite_name="Test",
            datasources=[],
            key=ResultKey(datetime.now().date(), {}),
            timestamp=time.time(),
            duration_ms=0.0,
            results=[],
            symbols=[],
        )

        # Clear the plugins and add only our failing plugin
        manager._plugins = {"failing": FailingPlugin()}

        # Should not raise an exception even when plugin fails
        manager.process_all(context)  # Should complete without raising

    def test_process_all_with_timeout(self) -> None:
        """Test process_all handles plugin timeout."""

        # Create a slow plugin
        class SlowPlugin:
            @staticmethod
            def metadata() -> PluginMetadata:
                return PluginMetadata(
                    name="slow",
                    version="1.0.0",
                    author="Test",
                    description="A plugin that takes too long",
                )

            def process(self, context: PluginExecutionContext) -> None:
                time.sleep(2)  # Sleep longer than timeout

        manager = PluginManager(_timeout_seconds=1)

        # Create a proper context
        context = PluginExecutionContext(
            suite_name="Test",
            datasources=[],
            key=ResultKey(datetime.now().date(), {}),
            timestamp=time.time(),
            duration_ms=0.0,
            results=[],
            symbols=[],
        )

        # Clear the plugins and add only our slow plugin
        manager._plugins = {"slow": SlowPlugin()}

        # Track the start time
        start_time = time.time()

        # Should not raise an exception and should terminate within timeout
        manager.process_all(context)

        # Should have taken approximately 1 second (the timeout), not 2 seconds
        elapsed = time.time() - start_time
        assert 0.9 < elapsed < 1.5  # Allow some margin for timing

    def test_process_all_success(self) -> None:
        """Test successful plugin execution."""

        # Track if plugin was called
        plugin_called = False

        # Create a successful plugin
        class SuccessPlugin:
            @staticmethod
            def metadata() -> PluginMetadata:
                return PluginMetadata(
                    name="success",
                    version="1.0.0",
                    author="Test",
                    description="A successful plugin",
                )

            def process(self, context: PluginExecutionContext) -> None:
                nonlocal plugin_called
                plugin_called = True
                # Do some work
                assert context.suite_name == "Test Suite"

        manager = PluginManager()

        # Create a proper context
        context = PluginExecutionContext(
            suite_name="Test Suite",
            datasources=["ds1"],
            key=ResultKey(datetime.now().date(), {}),
            timestamp=time.time(),
            duration_ms=100.0,
            results=[],
            symbols=[],
        )

        # Clear the plugins and add only our success plugin
        manager._plugins = {"success": SuccessPlugin()}

        # Process all plugins
        manager.process_all(context)

        # Verify plugin was actually called
        assert plugin_called

    def test_plugin_discovery_handles_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that plugin discovery handles errors gracefully."""

        # Mock entry_points to raise an exception
        def mock_entry_points(group: str | None = None) -> None:
            raise RuntimeError("Discovery failed")

        monkeypatch.setattr("importlib.metadata.entry_points", mock_entry_points)

        # Should not raise an exception
        manager = PluginManager()

        # Manager should still have the built-in audit plugin
        assert "audit" in manager._plugins

    def test_plugin_discovery_loads_external_plugins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that external plugins are discovered and loaded."""
        # Track if plugin was instantiated
        plugin_created = False

        # Create a mock plugin that properly implements ResultProcessor
        class ExternalPlugin:
            def __init__(self) -> None:
                nonlocal plugin_created
                plugin_created = True

            @staticmethod
            def metadata() -> PluginMetadata:
                return PluginMetadata(
                    name="external",
                    version="2.0.0",
                    author="External",
                    description="External plugin",
                )

            def process(self, context: PluginExecutionContext) -> None:
                pass

        # Create a mock entry point
        class MockEntryPoint:
            def __init__(self, name: str, plugin_class: type) -> None:
                self.name = name
                self._plugin_class = plugin_class

            def load(self) -> type:
                return self._plugin_class

        # Mock entry_points to return our mock plugin
        def mock_entry_points(group: str | None = None) -> list:
            if group == "dqx.plugins":
                return [MockEntryPoint("external", ExternalPlugin)]
            return []

        monkeypatch.setattr("importlib.metadata.entry_points", mock_entry_points)

        # Create manager - this will trigger plugin loading
        manager = PluginManager()

        # Verify plugin was created
        assert plugin_created

        # Should have both audit and external plugins
        assert "audit" in manager._plugins
        assert "external" in manager._plugins

        # Verify external plugin metadata
        metadata = manager.get_metadata()
        assert metadata["external"].name == "external"
        assert metadata["external"].version == "2.0.0"

        # Verify the plugin implements the protocol
        assert hasattr(manager._plugins["external"], "process")
        assert hasattr(manager._plugins["external"].__class__, "metadata")

    def test_plugin_discovery_rejects_invalid_plugins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that plugins not implementing ResultProcessor are rejected."""

        # Create an invalid plugin (doesn't implement protocol)
        class InvalidPlugin:
            def __init__(self) -> None:
                pass

        # Create a mock entry point
        class MockEntryPoint:
            def __init__(self, name: str, plugin_class: type) -> None:
                self.name = name
                self._plugin_class = plugin_class

            def load(self) -> type:
                return self._plugin_class

        # Mock entry_points to return our invalid plugin
        def mock_entry_points(group: str | None = None) -> list:
            if group == "dqx.plugins":
                return [MockEntryPoint("invalid", InvalidPlugin)]
            return []

        monkeypatch.setattr("importlib.metadata.entry_points", mock_entry_points)

        # Create manager
        manager = PluginManager()

        # Should only have audit plugin, not the invalid one
        assert "audit" in manager._plugins
        assert "invalid" not in manager._plugins

    def test_plugin_discovery_handles_load_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that plugin load errors are handled gracefully."""

        # Create a mock entry point that fails to load
        class FailingEntryPoint:
            def __init__(self) -> None:
                self.name = "failing"

            def load(self) -> None:
                raise ImportError("Failed to import plugin")

        # Mock entry_points to return our failing entry point
        def mock_entry_points(group: str | None = None) -> list:
            if group == "dqx.plugins":
                return [FailingEntryPoint()]
            return []

        monkeypatch.setattr("importlib.metadata.entry_points", mock_entry_points)

        # Should not raise an exception
        manager = PluginManager()

        # Should still have audit plugin
        assert "audit" in manager._plugins
        assert "failing" not in manager._plugins

    def test_plugin_implements_protocol(self) -> None:
        """Test that plugins must implement ResultProcessor protocol."""
        # This is a compile-time check, but we can verify at runtime
        manager = PluginManager()

        for name, plugin in manager.get_plugins().items():
            # Check that plugin has required methods
            assert hasattr(plugin, "process")
            assert callable(plugin.process)
            assert hasattr(plugin.__class__, "metadata")
            assert callable(plugin.__class__.metadata)

            # Verify it has the required protocol methods
            assert hasattr(plugin, "process")
            assert hasattr(plugin.__class__, "metadata")


class TestAuditPlugin:
    """Test cases for the built-in AuditPlugin."""

    def test_audit_plugin_metadata(self) -> None:
        """Test AuditPlugin metadata."""
        metadata = AuditPlugin.metadata()
        assert metadata.name == "audit"
        assert metadata.version == "1.0.0"
        assert metadata.author == "DQX Team"
        assert "console_output" in metadata.capabilities
        assert "statistics" in metadata.capabilities

    def test_audit_plugin_process(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test AuditPlugin processes context and displays output."""
        plugin = AuditPlugin()

        # Track console print calls
        print_calls: list[tuple[tuple, dict]] = []

        def mock_print(*args: object, **kwargs: object) -> None:
            print_calls.append((args, kwargs))

        monkeypatch.setattr(plugin.console, "print", mock_print)

        # Create test data
        results = [
            AssertionResult(
                yyyy_mm_dd=datetime.now().date(),
                suite="Test Suite",
                check="check1",
                assertion="assertion1",
                severity="P0",
                status="OK",
                metric=Success(1.0),
                expression="x > 0",
                tags={},
            ),
            AssertionResult(
                yyyy_mm_dd=datetime.now().date(),
                suite="Test Suite",
                check="check1",
                assertion="assertion2",
                severity="P1",
                status="FAILURE",
                metric=Failure([EvaluationFailure("Assertion failed", "x < 100", [])]),
                expression="x < 100",
                tags={},
            ),
        ]

        symbols = [
            SymbolInfo(
                name="x_1",
                metric="average(price)",
                dataset="ds1",
                value=Success(50.0),
                yyyy_mm_dd=datetime.now().date(),
                suite="Test Suite",
                tags={},
            )
        ]

        context = PluginExecutionContext(
            suite_name="Test Suite",
            datasources=["ds1", "ds2"],
            key=ResultKey(datetime.now().date(), {"env": "prod"}),
            timestamp=time.time(),
            duration_ms=250.5,
            results=results,
            symbols=symbols,
        )

        # Process the context
        plugin.process(context)

        # Verify console was called
        assert len(print_calls) > 0

        # Convert print calls to strings for easier searching
        print_strings = [str(args) for args, kwargs in print_calls]
        all_output = " ".join(print_strings)

        # Should print suite info
        assert "Test Suite" in all_output
        assert "250.5" in all_output  # Duration
        assert "ds1, ds2" in all_output

    def test_audit_plugin_empty_results(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test AuditPlugin with no results."""
        plugin = AuditPlugin()

        # Track console print calls
        print_calls: list[tuple[tuple, dict]] = []

        def mock_print(*args: object, **kwargs: object) -> None:
            print_calls.append((args, kwargs))

        monkeypatch.setattr(plugin.console, "print", mock_print)

        context = PluginExecutionContext(
            suite_name="Empty Suite",
            datasources=[],
            key=ResultKey(datetime.now().date(), {}),
            timestamp=time.time(),
            duration_ms=10.0,
            results=[],
            symbols=[],
        )

        # Should not raise any errors
        plugin.process(context)

        # Should still print header
        assert len(print_calls) > 0

    def test_audit_plugin_with_statistics(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test AuditPlugin displays statistics correctly."""
        plugin = AuditPlugin()

        # Track console print calls and table rendering
        print_calls: list[tuple[tuple, dict]] = []
        table_adds: list[tuple] = []

        def mock_print(*args: object, **kwargs: object) -> None:
            print_calls.append((args, kwargs))

        def mock_add_row(*args: object) -> None:
            table_adds.append(args)

        monkeypatch.setattr(plugin.console, "print", mock_print)

        # Create context with mixed results
        results = [
            # P0 passes
            AssertionResult(
                yyyy_mm_dd=datetime.now().date(),
                suite="Test Suite",
                check="check1",
                assertion="a1",
                severity="P0",
                status="OK",
                metric=Success(1.0),
                expression="x > 0",
                tags={},
            ),
            AssertionResult(
                yyyy_mm_dd=datetime.now().date(),
                suite="Test Suite",
                check="check1",
                assertion="a2",
                severity="P0",
                status="OK",
                metric=Success(2.0),
                expression="x > 1",
                tags={},
            ),
            # P0 failure
            AssertionResult(
                yyyy_mm_dd=datetime.now().date(),
                suite="Test Suite",
                check="check2",
                assertion="a3",
                severity="P0",
                status="FAILURE",
                metric=Failure([EvaluationFailure("Failed", "x < 0", [])]),
                expression="x < 0",
                tags={},
            ),
            # P1 passes
            AssertionResult(
                yyyy_mm_dd=datetime.now().date(),
                suite="Test Suite",
                check="check3",
                assertion="a4",
                severity="P1",
                status="OK",
                metric=Success(3.0),
                expression="y > 0",
                tags={},
            ),
            # P2 failure
            AssertionResult(
                yyyy_mm_dd=datetime.now().date(),
                suite="Test Suite",
                check="check4",
                assertion="a5",
                severity="P2",
                status="FAILURE",
                metric=Failure([EvaluationFailure("Failed", "z < 0", [])]),
                expression="z < 0",
                tags={},
            ),
        ]

        # Create symbols with mixed results
        symbols = [
            SymbolInfo(
                name="x_1",
                metric="average(price)",
                dataset="ds1",
                value=Success(50.0),
                yyyy_mm_dd=datetime.now().date(),
                suite="Test Suite",
                tags={},
            ),
            SymbolInfo(
                name="x_2",
                metric="sum(quantity)",
                dataset="ds1",
                value=Failure("Computation failed"),
                yyyy_mm_dd=datetime.now().date(),
                suite="Test Suite",
                tags={},
            ),
            SymbolInfo(
                name="x_3",
                metric="count(*)",
                dataset="ds2",
                value=Success(100.0),
                yyyy_mm_dd=datetime.now().date(),
                suite="Test Suite",
                tags={},
            ),
        ]

        context = PluginExecutionContext(
            suite_name="Test Suite",
            datasources=["ds1", "ds2"],
            key=ResultKey(datetime.now().date(), {"env": "test"}),
            timestamp=time.time(),
            duration_ms=500.0,
            results=results,
            symbols=symbols,
        )

        # Process the context
        plugin.process(context)

        # Verify statistics were printed
        assert len(print_calls) > 0

        # The plugin prints multiple things, including Tables
        # Let's check for the expected content

        # Find all string arguments in print calls
        all_strings = []
        for args, kwargs in print_calls:
            for arg in args:
                if isinstance(arg, str):
                    all_strings.append(arg)

        # Join all strings
        output_text = " ".join(all_strings)

        # Check header
        assert "DQX Audit Report" in output_text
        assert "Test Suite" in output_text
        assert "500.0" in output_text  # Duration

        # The actual statistics are in Table objects, so we can't check them as strings
        # But we can verify that tables were created by checking the number of print calls
        # We expect:
        # 1. Empty line
        # 2. Header
        # 3. Suite name
        # 4. Date
        # 5. Duration
        # 6. Datasets
        # 7. Empty line
        # 8. Summary table
        # 9. Empty line (before severity table)
        # 10. Severity table
        # 11. Empty line (before symbol table)
        # 12. Symbol table
        # 13. Empty line
        # 14. Footer
        # 15. Empty line
        assert len(print_calls) >= 10  # At minimum
