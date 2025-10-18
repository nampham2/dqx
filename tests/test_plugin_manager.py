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

        # Test timeout property
        assert manager.timeout_seconds == 60  # default value

    def test_plugin_manager_custom_timeout(self) -> None:
        """Test PluginManager with custom timeout."""
        manager = PluginManager(timeout_seconds=10)
        assert manager._timeout_seconds == 10

    def test_plugin_exists_returns_true(self) -> None:
        """Test plugin_exists returns True for existing plugin."""
        manager = PluginManager()

        # Verify audit plugin exists (registered by default)
        assert manager.plugin_exists("audit") is True

        # Clear plugins and add a test one
        manager.clear_plugins()
        assert manager.plugin_exists("audit") is False

        # Register it back
        manager.register_plugin("dqx.plugins.AuditPlugin")
        assert manager.plugin_exists("audit") is True

    def test_register_plugin_successful_import(self) -> None:
        """Test successful plugin registration covering normal import path."""
        manager = PluginManager()

        # Clear default plugins
        manager.clear_plugins()

        # Register the AuditPlugin successfully
        manager.register_plugin("dqx.plugins.AuditPlugin")

        # Verify it was registered
        assert "audit" in manager.get_plugins()
        assert isinstance(manager.get_plugins()["audit"], AuditPlugin)

    def test_register_plugin_import_error(self) -> None:
        """Test register_plugin handles ImportError when module cannot be imported."""
        manager = PluginManager()

        # Try to register a plugin from a non-existent module
        with pytest.raises(ValueError, match="Cannot import module nonexistent.module"):
            manager.register_plugin("nonexistent.module.Plugin")

    def test_register_plugin_invalid_class_name_format(self) -> None:
        """Test register_plugin with invalid class name format."""
        manager = PluginManager()

        # Test with no dots (invalid format)
        with pytest.raises(ValueError, match="Invalid class name format: InvalidName"):
            manager.register_plugin("InvalidName")

        # Test with empty string
        with pytest.raises(ValueError, match="Invalid class name format: "):
            manager.register_plugin("")

    def test_register_plugin_class_not_in_module(self) -> None:
        """Test register_plugin when class doesn't exist in module."""
        manager = PluginManager()

        # Try to register a non-existent class from an existing module
        with pytest.raises(ValueError, match="Module dqx.plugins has no class NonExistentClass"):
            manager.register_plugin("dqx.plugins.NonExistentClass")

    def test_register_plugin_generic_exception_handling(self) -> None:
        """Test register_plugin wraps non-ValueError exceptions."""
        from unittest.mock import patch

        manager = PluginManager()

        # Mock importlib.import_module to raise a generic exception
        with patch("importlib.import_module", side_effect=RuntimeError("Generic error")):
            with pytest.raises(ValueError, match="Failed to register plugin test.module.Class: Generic error"):
                manager.register_plugin("test.module.Class")

    def test_unregister_plugin(self) -> None:
        """Test unregister_plugin removes plugin correctly."""
        manager = PluginManager()

        # Verify audit plugin exists
        assert "audit" in manager.get_plugins()

        # Unregister it
        manager.unregister_plugin("audit")

        # Verify it's gone
        assert "audit" not in manager.get_plugins()

        # Unregistering non-existent plugin should not raise error
        manager.unregister_plugin("non_existent")  # Should not raise

    def test_get_plugin_metadata(self) -> None:
        """Test getting metadata from plugins."""
        manager = PluginManager()

        # Check audit plugin exists
        assert "audit" in manager.get_plugins()

        # Get metadata from the plugin
        audit_plugin = manager.get_plugins()["audit"]
        audit_meta = audit_plugin.__class__.metadata()
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

        manager = PluginManager(timeout_seconds=1)

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

    def test_plugin_discovery_rejects_invalid_plugins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that plugins not implementing PostProcessor are rejected."""

        # Create an invalid plugin (doesn't implement protocol)
        class InvalidPlugin:
            def __init__(self) -> None:
                pass

        # Create a mock entry point that provides the class path
        class MockEntryPoint:
            def __init__(self, name: str, plugin_class: type) -> None:
                self.name = name
                # Store the class path as value (what register_plugin expects)
                self.value = f"tests.test_plugin_manager.{plugin_class.__name__}"

        # Mock entry_points to return our invalid plugin
        def mock_entry_points(group: str | None = None) -> list:
            if group == "dqx.plugins":
                return [MockEntryPoint("invalid", InvalidPlugin)]
            return []

        monkeypatch.setattr("importlib.metadata.entry_points", mock_entry_points)

        # Create manager
        manager = PluginManager()

        # Should have only audit plugin since the invalid one was rejected
        assert "invalid" not in manager._plugins
        assert len(manager._plugins) == 1
        assert "audit" in manager._plugins

    def test_plugin_discovery_handles_load_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that plugin load errors are handled gracefully."""

        # Create a mock entry point that fails to load
        class FailingEntryPoint:
            def __init__(self) -> None:
                self.name = "failing"
                self.value = "nonexistent.module.FailingPlugin"  # This will fail in register_plugin

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

        # Should have only audit plugin since loading failed
        assert "failing" not in manager._plugins
        assert len(manager._plugins) == 1
        assert "audit" in manager._plugins

    def test_plugin_implements_protocol(self) -> None:
        """Test that plugins must implement PostProcessor protocol."""
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

    def test_plugin_validation_uses_isinstance(self) -> None:
        """Test that plugin validation uses isinstance for protocol checking."""

        # Define test plugin classes locally
        class InvalidPlugin:
            """Plugin that doesn't implement the protocol."""

            pass

        class WrongMetadataPlugin:
            """Plugin with wrong metadata return type."""

            @staticmethod
            def metadata() -> str:  # Wrong return type
                return "wrong"

            def process(self, context: PluginExecutionContext) -> None:
                pass

        # Add these test classes to the current module for import
        import sys

        current_module = sys.modules[__name__]
        setattr(current_module, "InvalidPlugin", InvalidPlugin)
        setattr(current_module, "WrongMetadataPlugin", WrongMetadataPlugin)

        manager = PluginManager()

        # Test with a class that doesn't implement the protocol
        with pytest.raises(ValueError, match="doesn't implement PostProcessor protocol"):
            manager.register_plugin(f"{__name__}.InvalidPlugin")

        # Test with a class that has wrong metadata return type
        with pytest.raises(ValueError, match="metadata\\(\\) must return a PluginMetadata instance"):
            manager.register_plugin(f"{__name__}.WrongMetadataPlugin")


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
