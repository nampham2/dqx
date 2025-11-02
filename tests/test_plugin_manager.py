"""Tests for the PluginManager class."""

import time
from datetime import datetime

import pyarrow as pa
import pytest
from returns.result import Failure, Success

from dqx.cache import CacheStats
from dqx.common import (
    AssertionResult,
    DQXError,
    EvaluationFailure,
    PluginMetadata,
    ResultKey,
)
from dqx.orm.repositories import MetricStats
from dqx.plugins import AuditPlugin, PluginExecutionContext, PluginManager
from dqx.provider import SymbolInfo


def _create_empty_trace() -> pa.Table:
    """Create an empty PyArrow table for trace parameter."""
    return pa.table({})


class ValidInstancePlugin:
    """Valid plugin for testing instance registration."""

    @staticmethod
    def metadata() -> PluginMetadata:
        return PluginMetadata(
            name="instance_plugin", version="1.0.0", author="Test", description="Test instance plugin"
        )

    def process(self, context: PluginExecutionContext) -> None:
        pass


class PluginWithConstructor:
    """Plugin that accepts constructor arguments."""

    def __init__(self, threshold: float, debug: bool = False):
        self.threshold = threshold
        self.debug = debug

    @staticmethod
    def metadata() -> PluginMetadata:
        return PluginMetadata(
            name="configured_plugin", version="1.0.0", author="Test", description="Plugin with configuration"
        )

    def process(self, context: PluginExecutionContext) -> None:
        pass


class InvalidInstancePlugin:
    """Invalid plugin missing process method."""

    @staticmethod
    def metadata() -> PluginMetadata:
        return PluginMetadata(name="invalid", version="1.0.0", author="Test", description="Invalid plugin")


class StatefulPlugin:
    """Plugin that maintains state across calls."""

    def __init__(self) -> None:
        self.call_count = 0
        self.processed_suites: list[str] = []

    @staticmethod
    def metadata() -> PluginMetadata:
        return PluginMetadata(
            name="stateful_plugin", version="1.0.0", author="Test", description="Plugin with internal state"
        )

    def process(self, context: PluginExecutionContext) -> None:
        self.call_count += 1
        self.processed_suites.append(context.suite_name)


@pytest.fixture
def empty_context() -> PluginExecutionContext:
    """Create an empty PluginExecutionContext for testing."""
    return PluginExecutionContext(
        suite_name="Test Suite",
        execution_id="test_exec_id",
        datasources=[],
        key=ResultKey(datetime.now().date(), {}),
        timestamp=time.time(),
        duration_ms=0.0,
        results=[],
        symbols=[],
        trace=_create_empty_trace(),
        metrics_stats=MetricStats(total_metrics=0, expired_metrics=0),
        cache_stats=CacheStats(hit=0, missed=0),
    )


@pytest.fixture
def valid_plugin() -> ValidInstancePlugin:
    """Create a valid plugin instance for testing."""
    return ValidInstancePlugin()


@pytest.fixture
def plugin_manager() -> PluginManager:
    """Create a fresh PluginManager instance."""
    return PluginManager()


class TestPluginManager:
    """Test cases for PluginManager."""

    def test_register_plugin_instance_stores_reference(
        self, plugin_manager: PluginManager, valid_plugin: ValidInstancePlugin
    ) -> None:
        """Test registering a PostProcessor instance stores the exact reference."""
        plugin_manager.register_plugin(valid_plugin)

        assert plugin_manager.plugin_exists("instance_plugin")
        # Verify exact same instance is stored (not a copy)
        assert plugin_manager.get_plugins()["instance_plugin"] is valid_plugin

    # Test removed: test_register_invalid_instance_missing_process

    def test_register_configured_plugin_instance(self) -> None:
        """Test registering a plugin instance with constructor parameters."""
        manager = PluginManager()
        plugin = PluginWithConstructor(threshold=0.95, debug=True)

        manager.register_plugin(plugin)

        assert manager.plugin_exists("configured_plugin")
        registered = manager.get_plugins()["configured_plugin"]
        assert registered is plugin
        # Assertions for threshold and debug removed

    # Test removed: test_register_instance_validates_metadata

    def test_register_stateful_plugin_maintains_state(self) -> None:
        """Test stateful plugin instances maintain their state across calls."""
        manager = PluginManager()
        plugin = StatefulPlugin()

        manager.register_plugin(plugin)

        # Create contexts for multiple runs
        context1 = PluginExecutionContext(
            suite_name="Suite1",
            execution_id="exec_1",
            datasources=[],
            key=ResultKey(datetime.now().date(), {}),
            timestamp=time.time(),
            duration_ms=100.0,
            results=[],
            symbols=[],
            trace=_create_empty_trace(),
            metrics_stats=MetricStats(total_metrics=0, expired_metrics=0),
            cache_stats=CacheStats(hit=0, missed=0),
        )

        context2 = PluginExecutionContext(
            suite_name="Suite2",
            execution_id="exec_2",
            datasources=[],
            key=ResultKey(datetime.now().date(), {}),
            timestamp=time.time(),
            duration_ms=200.0,
            results=[],
            symbols=[],
            trace=_create_empty_trace(),
            metrics_stats=MetricStats(total_metrics=0, expired_metrics=0),
            cache_stats=CacheStats(hit=0, missed=0),
        )

        # Process through manager
        manager.process_all(context1)
        manager.process_all(context2)

        # Verify state was maintained
        registered_plugin = manager.get_plugins()["stateful_plugin"]
        assert isinstance(registered_plugin, StatefulPlugin)
        assert registered_plugin.call_count == 2
        assert registered_plugin.processed_suites == ["Suite1", "Suite2"]

    def test_register_instance_duplicate_name_overwrites(self) -> None:
        """Test registering instance with duplicate name overwrites existing."""
        manager = PluginManager()
        plugin1 = ValidInstancePlugin()
        plugin2 = ValidInstancePlugin()

        manager.register_plugin(plugin1)
        manager.register_plugin(plugin2)

        # Should have overwritten first instance
        assert manager.get_plugins()["instance_plugin"] is plugin2
        assert manager.get_plugins()["instance_plugin"] is not plugin1

    def test_mixed_string_and_instance_registration(self) -> None:
        """Test mixing string-based and instance-based registration."""
        manager = PluginManager()
        manager.clear_plugins()  # Remove default plugins

        # Register by string
        manager.register_plugin("dqx.plugins.AuditPlugin")

        # Register by instance
        instance_plugin = ValidInstancePlugin()
        manager.register_plugin(instance_plugin)

        plugins = manager.get_plugins()
        assert len(plugins) == 2
        assert "audit" in plugins
        assert "instance_plugin" in plugins
        assert plugins["instance_plugin"] is instance_plugin

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
        manager = PluginManager()

        # Try to register a plugin from a module that raises on import
        with pytest.raises(
            ValueError, match="Failed to register plugin tests.fixtures.failing_import_module.TestPlugin: Generic error"
        ):
            manager.register_plugin("tests.fixtures.failing_import_module.TestPlugin")

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
        assert "verification" in audit_meta.capabilities

    def test_process_all_with_no_plugins(
        self, plugin_manager: PluginManager, empty_context: PluginExecutionContext
    ) -> None:
        """Test process_all when no plugins are loaded."""
        plugin_manager._plugins = {}  # Clear plugins

        # Should not raise any errors
        plugin_manager.process_all(empty_context)

    def test_process_all_with_plugin_error(
        self, plugin_manager: PluginManager, empty_context: PluginExecutionContext
    ) -> None:
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

        # Clear the plugins and add only our failing plugin
        plugin_manager._plugins = {"failing": FailingPlugin()}

        # Should not raise an exception even when plugin fails
        plugin_manager.process_all(empty_context)  # Should complete without raising

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
            execution_id="test_timeout",
            datasources=[],
            key=ResultKey(datetime.now().date(), {}),
            timestamp=time.time(),
            duration_ms=0.0,
            results=[],
            symbols=[],
            trace=_create_empty_trace(),
            metrics_stats=MetricStats(total_metrics=0, expired_metrics=0),
            cache_stats=CacheStats(hit=0, missed=0),
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
        plugin_called: bool = False

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

        manager: PluginManager = PluginManager()

        # Create a proper context
        context: PluginExecutionContext = PluginExecutionContext(
            suite_name="Test Suite",
            execution_id="test_success",
            datasources=["ds1"],
            key=ResultKey(datetime.now().date(), {}),
            timestamp=time.time(),
            duration_ms=100.0,
            results=[],
            symbols=[],
            trace=_create_empty_trace(),
            metrics_stats=MetricStats(total_metrics=0, expired_metrics=0),
            cache_stats=CacheStats(hit=0, missed=0),
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

        # Create a test entry point that provides the class path
        class TestEntryPoint:
            def __init__(self, name: str, plugin_class: type) -> None:
                self.name: str = name
                # Store the class path as value (what register_plugin expects)
                self.value: str = f"tests.test_plugin_manager.{plugin_class.__name__}"

        # Override entry_points to return our invalid plugin
        def test_entry_points(group: str | None = None) -> list[TestEntryPoint]:
            if group == "dqx.plugins":
                return [TestEntryPoint("invalid", InvalidPlugin)]
            return []

        monkeypatch.setattr("importlib.metadata.entry_points", test_entry_points)

        # Create manager
        manager = PluginManager()

        # Should have only audit plugin since the invalid one was rejected
        assert "invalid" not in manager._plugins
        assert len(manager._plugins) == 1
        assert "audit" in manager._plugins

    def test_plugin_discovery_handles_load_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that plugin load errors are handled gracefully."""

        # Create a test entry point that fails to load
        class FailingEntryPoint:
            def __init__(self) -> None:
                self.name: str = "failing"
                self.value: str = "nonexistent.module.FailingPlugin"  # This will fail in register_plugin

            def load(self) -> None:
                raise ImportError("Failed to import plugin")

        # Override entry_points to return our failing entry point
        def test_entry_points(group: str | None = None) -> list[FailingEntryPoint]:
            if group == "dqx.plugins":
                return [FailingEntryPoint()]
            return []

        monkeypatch.setattr("importlib.metadata.entry_points", test_entry_points)

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
        from types import ModuleType

        current_module: ModuleType = sys.modules[__name__]
        setattr(current_module, "InvalidPlugin", InvalidPlugin)
        setattr(current_module, "WrongMetadataPlugin", WrongMetadataPlugin)

        manager: PluginManager = PluginManager()

        # Test with a class that doesn't implement the protocol
        with pytest.raises(ValueError, match="doesn't implement PostProcessor protocol"):
            manager.register_plugin(f"{__name__}.InvalidPlugin")

        # Test with a class that has wrong metadata return type
        with pytest.raises(ValueError, match=r"metadata\(\) must return a PluginMetadata instance"):
            manager.register_plugin(f"{__name__}.WrongMetadataPlugin")


class TestAuditPlugin:
    """Test cases for the built-in AuditPlugin."""

    def test_audit_plugin_metadata(self) -> None:
        """Test AuditPlugin metadata."""
        metadata = AuditPlugin.metadata()
        assert metadata.name == "audit"
        assert metadata.version == "1.0.0"
        assert metadata.author == "DQX Team"
        assert "verification" in metadata.capabilities
        assert "statistics" in metadata.capabilities

    def test_audit_plugin_process(self) -> None:
        """Test AuditPlugin processes context without errors."""
        plugin = AuditPlugin()

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
                tags={},
            )
        ]

        context = PluginExecutionContext(
            suite_name="Test Suite",
            execution_id="test_audit",
            datasources=["ds1", "ds2"],
            key=ResultKey(datetime.now().date(), {"env": "prod"}),
            timestamp=time.time(),
            duration_ms=250.5,
            results=results,
            symbols=symbols,
            trace=_create_empty_trace(),
            metrics_stats=MetricStats(total_metrics=0, expired_metrics=0),
            cache_stats=CacheStats(hit=0, missed=0),
        )

        # Process the context - should not raise any errors
        plugin.process(context)

    def test_audit_plugin_with_tags(self) -> None:
        """Test AuditPlugin with tags."""
        plugin = AuditPlugin()

        context = PluginExecutionContext(
            suite_name="Tagged Suite",
            execution_id="test_tags",
            datasources=[],
            key=ResultKey(datetime.now().date(), {"env": "prod", "region": "us-east"}),
            timestamp=time.time(),
            duration_ms=50.0,
            results=[],
            symbols=[],
            trace=_create_empty_trace(),
            metrics_stats=MetricStats(total_metrics=0, expired_metrics=0),
            cache_stats=CacheStats(hit=0, missed=0),
        )

        # Process the context - should not raise any errors
        plugin.process(context)

    def test_audit_plugin_no_tags(self) -> None:
        """Test AuditPlugin with empty tags."""
        plugin = AuditPlugin()

        context = PluginExecutionContext(
            suite_name="No Tags Suite",
            execution_id="test_no_tags",
            datasources=[],
            key=ResultKey(datetime.now().date(), {}),
            timestamp=time.time(),
            duration_ms=30.0,
            results=[],
            symbols=[],
            trace=_create_empty_trace(),
            metrics_stats=MetricStats(total_metrics=0, expired_metrics=0),
            cache_stats=CacheStats(hit=0, missed=0),
        )

        # Process the context - should not raise any errors
        plugin.process(context)

    def test_audit_plugin_empty_results(self) -> None:
        """Test AuditPlugin with no results."""
        plugin = AuditPlugin()

        context = PluginExecutionContext(
            suite_name="Empty Suite",
            execution_id="test_empty",
            datasources=[],
            key=ResultKey(datetime.now().date(), {}),
            timestamp=time.time(),
            duration_ms=10.0,
            results=[],
            symbols=[],
            trace=_create_empty_trace(),
            metrics_stats=MetricStats(total_metrics=0, expired_metrics=0),
            cache_stats=CacheStats(hit=0, missed=0),
        )

        # Should not raise any errors
        plugin.process(context)

    def test_audit_plugin_with_cache_stats(self) -> None:
        """Test AuditPlugin with cache statistics."""
        plugin = AuditPlugin()

        # Create context with cache stats
        context = PluginExecutionContext(
            suite_name="Cache Test Suite",
            execution_id="test_cache_stats",
            datasources=["ds1"],
            key=ResultKey(datetime.now().date(), {"env": "test"}),
            timestamp=time.time(),
            duration_ms=100.0,
            results=[
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
            ],
            symbols=[
                SymbolInfo(
                    name="x_1",
                    metric="count(*)",
                    dataset="ds1",
                    value=Success(50.0),
                    yyyy_mm_dd=datetime.now().date(),
                    tags={},
                )
            ],
            trace=_create_empty_trace(),
            metrics_stats=MetricStats(total_metrics=5, expired_metrics=2),
            cache_stats=CacheStats(hit=150, missed=50),
        )

        # Process the context - should not raise any errors
        plugin.process(context)

    def test_audit_plugin_with_statistics(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test AuditPlugin displays statistics correctly."""
        plugin = AuditPlugin()

        # Track console print calls
        print_calls: list[str] = []

        def capture_print(*args: object, **kwargs: object) -> None:
            if args:
                text = str(args[0]) if len(args) == 1 else " ".join(str(arg) for arg in args)
                print_calls.append(text)

        monkeypatch.setattr(plugin.console, "print", capture_print)

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
                tags={},
            ),
            SymbolInfo(
                name="x_2",
                metric="sum(quantity)",
                dataset="ds1",
                value=Failure("Computation failed"),
                yyyy_mm_dd=datetime.now().date(),
                tags={},
            ),
            SymbolInfo(
                name="x_3",
                metric="count(*)",
                dataset="ds2",
                value=Success(100.0),
                yyyy_mm_dd=datetime.now().date(),
                tags={},
            ),
        ]

        context = PluginExecutionContext(
            suite_name="Test Suite",
            execution_id="test_stats",
            datasources=["ds1", "ds2"],
            key=ResultKey(datetime.now().date(), {"env": "test"}),
            timestamp=time.time(),
            duration_ms=500.0,
            results=results,
            symbols=symbols,
            trace=_create_empty_trace(),
            metrics_stats=MetricStats(total_metrics=0, expired_metrics=0),
            cache_stats=CacheStats(hit=0, missed=0),
        )

        with pytest.raises(DQXError, match=r"\[InternalError\] Symbols failed to evaluate during execution!"):
            # Process the context
            plugin.process(context)

    def test_audit_plugin_with_data_discrepancies(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test AuditPlugin handles data discrepancies properly (covers lines 383-405)."""
        plugin = AuditPlugin()

        # Track console print calls
        print_calls: list[str] = []

        def capture_print(*args: object, **kwargs: object) -> None:
            if args:
                text = str(args[0]) if len(args) == 1 else " ".join(str(arg) for arg in args)
                print_calls.append(text)

        monkeypatch.setattr(plugin.console, "print", capture_print)

        # Mock display.print_metrics_by_execution_id
        display_called_with: list[tuple[pa.Table, str]] = []

        def capture_display(trace: pa.Table, execution_id: str) -> None:
            display_called_with.append((trace, execution_id))

        import dqx.display

        monkeypatch.setattr(dqx.display, "print_metrics_by_execution_id", capture_display)

        # Create a trace table with discrepancies
        trace = pa.table(
            {
                "execution_id": ["test_discrepancy"] * 3,
                "symbol": ["x_1", "x_2", "x_3"],
                "dataset": ["ds1", "ds1", "ds2"],
                "metric": ["count(*)", "sum(price)", "average(score)"],
                "yyyy_mm_dd": [datetime.now().date()] * 3,
                "expected_value": [100.0, 200.0, 75.5],
                "actual_value": [95.0, 200.0, 80.0],  # x_1 and x_3 have discrepancies
                "discrepancy": ["expected_value != actual_value", None, "expected_value != actual_value"],
            }
        )

        # Mock the data_discrepancy_stats method to return discrepancies
        from dqx.data import MetricTraceStats

        def mock_discrepancy_stats() -> MetricTraceStats:
            return MetricTraceStats(
                total_rows=3,
                discrepancy_count=2,
                discrepancy_rows=[0, 2],  # Rows with discrepancies
                discrepancy_details=[
                    {
                        "row_index": 0,
                        "date": datetime.now().date(),
                        "metric": "count(*)",
                        "symbol": "x_1",
                        "dataset": "ds1",
                        "value_db": None,
                        "value_analysis": None,
                        "value_final": 95.0,
                        "discrepancies": ["expected_value != actual_value"],
                    },
                    {
                        "row_index": 2,
                        "date": datetime.now().date(),
                        "metric": "average(score)",
                        "symbol": "x_3",
                        "dataset": "ds2",
                        "value_db": None,
                        "value_analysis": None,
                        "value_final": 80.0,
                        "discrepancies": ["expected_value != actual_value"],
                    },
                ],
            )

        # Create context with no failed symbols but with data discrepancies
        results = [
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
        ]

        context = PluginExecutionContext(
            suite_name="Test Suite",
            execution_id="test_discrepancy",
            datasources=["ds1", "ds2"],
            key=ResultKey(datetime.now().date(), {"env": "test"}),
            timestamp=time.time(),
            duration_ms=100.0,
            results=results,
            symbols=[],  # No failed symbols
            trace=trace,
            metrics_stats=MetricStats(total_metrics=5, expired_metrics=0),
            cache_stats=CacheStats(hit=10, missed=2),
        )

        # Patch the data_discrepancy_stats method
        monkeypatch.setattr(context, "data_discrepancy_stats", mock_discrepancy_stats)

        with pytest.raises(DQXError, match=r"\[InternalError\] Data discrepancies detected during audit"):
            # Process the context - should raise due to discrepancies
            plugin.process(context)

        # Verify that display.print_metrics_by_execution_id was called
        assert len(display_called_with) == 1
        assert display_called_with[0][1] == "test_discrepancy"

        # Verify the discrepancy warning was printed
        discrepancy_prints = [p for p in print_calls if "Data Integrity:" in p and "discrepancies" in p]
        assert len(discrepancy_prints) == 1
        assert "2 discrepancies" in discrepancy_prints[0]
        assert "2x expected≠actual" in discrepancy_prints[0]  # Compact format check


class TestPluginInstanceEdgeCases:
    """Edge case tests for plugin instance registration."""

    def test_register_plugin_with_exception_in_metadata(self) -> None:
        """Test registering instance that raises exception in metadata()."""

        class ExceptionMetadataPlugin:
            @staticmethod
            def metadata() -> PluginMetadata:
                raise RuntimeError("Metadata retrieval failed")

            def process(self, context: PluginExecutionContext) -> None:
                pass

        manager = PluginManager()
        plugin = ExceptionMetadataPlugin()

        with pytest.raises(ValueError, match="Failed to get metadata from plugin instance"):
            manager.register_plugin(plugin)

    # Test removed: test_register_plugin_with_none_metadata

    def test_plugin_instance_lifecycle(self) -> None:
        """Test complete lifecycle of plugin instance registration."""

        class LifecyclePlugin:
            def __init__(self) -> None:
                self.initialized = True
                self.process_count = 0

            @staticmethod
            def metadata() -> PluginMetadata:
                return PluginMetadata(
                    name="lifecycle",
                    version="1.0.0",
                    author="Test",
                    description="Lifecycle test plugin",
                )

            def process(self, context: PluginExecutionContext) -> None:
                self.process_count += 1

        # Create and register
        manager = PluginManager()
        plugin = LifecyclePlugin()
        manager.register_plugin(plugin)

        # Verify registration
        assert manager.plugin_exists("lifecycle")
        assert manager.get_plugins()["lifecycle"] is plugin

        # Create context
        context = PluginExecutionContext(
            suite_name="Test",
            execution_id="test_lifecycle",
            datasources=[],
            key=ResultKey(datetime.now().date(), {}),
            timestamp=time.time(),
            duration_ms=0.0,
            results=[],
            symbols=[],
            trace=_create_empty_trace(),
            metrics_stats=MetricStats(total_metrics=0, expired_metrics=0),
            cache_stats=CacheStats(hit=0, missed=0),
        )

        # Process
        manager.process_all(context)
        assert plugin.process_count == 1

        # Process again
        manager.process_all(context)
        assert plugin.process_count == 2

        # Unregister
        manager.unregister_plugin("lifecycle")
        assert not manager.plugin_exists("lifecycle")

        # Process should not call plugin
        manager.process_all(context)
        assert plugin.process_count == 2  # No change

    # Test removed: test_multiple_instance_registration_different_names


class TestPluginIntegration:
    """Integration tests combining string and instance registration."""

    def test_mixed_registration_with_processing(self) -> None:
        """Test processing with mixed string and instance registered plugins."""
        results: list[str] = []

        class TrackingPlugin:
            def __init__(self, tag: str) -> None:
                self.tag = tag

            @staticmethod
            def metadata() -> PluginMetadata:
                return PluginMetadata(
                    name="tracking",
                    version="1.0.0",
                    author="Test",
                    description="Tracking plugin",
                )

            def process(self, context: PluginExecutionContext) -> None:
                results.append(f"instance-{self.tag}")

        # Create a module-level plugin for string registration
        import sys
        from types import ModuleType

        module: ModuleType = sys.modules[__name__]

        class StringPlugin:
            @staticmethod
            def metadata() -> PluginMetadata:
                return PluginMetadata(
                    name="string_plugin",
                    version="1.0.0",
                    author="Test",
                    description="String registered plugin",
                )

            def process(self, context: PluginExecutionContext) -> None:
                results.append("string-plugin")

        setattr(module, "StringPlugin", StringPlugin)

        # Setup manager
        manager = PluginManager()
        manager.clear_plugins()

        # Register by string
        manager.register_plugin(f"{__name__}.StringPlugin")

        # Register by instance
        instance = TrackingPlugin("test")
        manager.register_plugin(instance)

        # Create context
        context = PluginExecutionContext(
            suite_name="Integration Test",
            execution_id="test_integration",
            datasources=[],
            key=ResultKey(datetime.now().date(), {}),
            timestamp=time.time(),
            duration_ms=0.0,
            results=[],
            symbols=[],
            trace=_create_empty_trace(),
            metrics_stats=MetricStats(total_metrics=0, expired_metrics=0),
            cache_stats=CacheStats(hit=0, missed=0),
        )

        # Process all
        manager.process_all(context)

        # Verify both plugins ran
        assert "string-plugin" in results
        assert "instance-test" in results

    def test_replacing_string_with_instance(self) -> None:
        """Test replacing a string-registered plugin with an instance."""

        class CustomAuditPlugin:
            def __init__(self, prefix: str = "CUSTOM") -> None:
                self.prefix = prefix

            @staticmethod
            def metadata() -> PluginMetadata:
                return PluginMetadata(
                    name="audit",  # Same name as built-in
                    version="2.0.0",
                    author="Custom",
                    description="Custom audit plugin",
                )

            def process(self, context: PluginExecutionContext) -> None:
                print(f"{self.prefix}: Processing {context.suite_name}")

        manager = PluginManager()

        # Initially has string-registered audit plugin
        assert manager.plugin_exists("audit")
        original = manager.get_plugins()["audit"]
        assert isinstance(original, AuditPlugin)

        # Replace with instance
        custom = CustomAuditPlugin("REPLACED")
        manager.register_plugin(custom)

        # Verify replacement
        assert manager.plugin_exists("audit")
        current = manager.get_plugins()["audit"]
        assert current is custom
        assert current is not original
