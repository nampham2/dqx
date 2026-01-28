"""Type checking tests for plugin system.

This file contains tests that verify type annotations work correctly
for the plugin system. These tests use mypy's reveal_type functionality
and other type-checking features.
"""

from typing import TYPE_CHECKING, cast

import pytest

from dqx.common import PluginMetadata
from dqx.plugins import PluginExecutionContext, PluginManager, PostProcessor

if TYPE_CHECKING:
    # Type checking utilities
    pass  # type: ignore[attr-defined]


class TypedPlugin:
    """Properly typed plugin for testing."""

    @staticmethod
    def metadata() -> PluginMetadata:
        return PluginMetadata(
            name="typed",
            version="1.0.0",
            author="Test",
            description="Typed plugin",
        )

    def process(self, context: PluginExecutionContext) -> None:
        # Verify context attributes are properly typed
        suite_name: str = context.suite_name
        datasources: list[str] = context.datasources
        duration: float = context.duration_ms

        # These would fail type checking if types were wrong
        assert isinstance(suite_name, str)
        assert isinstance(datasources, list)
        assert isinstance(duration, float)


class TestPluginTypeChecking:
    """Tests focused on type checking aspects of the plugin system."""

    def test_plugin_manager_register_overloads(self) -> None:
        """Test that register_plugin overloads work correctly."""
        manager = PluginManager()

        # String registration
        manager.register_plugin("dqx.plugins.AuditPlugin")

        # Instance registration
        plugin = TypedPlugin()
        manager.register_plugin(plugin)

        # The following would fail type checking:
        # manager.register_plugin(123)  # type: ignore[call-overload]
        # manager.register_plugin([])   # type: ignore[call-overload]

    def test_postprocessor_protocol_typing(self) -> None:
        """Test PostProcessor protocol type checking."""
        plugin = TypedPlugin()

        # Verify it's recognized as PostProcessor
        assert isinstance(plugin, PostProcessor)

        # Verify protocol methods exist with correct signatures
        metadata = plugin.metadata()
        assert isinstance(metadata, PluginMetadata)

        # The protocol requires these exact signatures
        # Wrong signatures would fail at type checking time

    def test_plugin_metadata_typing(self) -> None:
        """Test PluginMetadata type annotations."""
        metadata = PluginMetadata(
            name="test",
            version="1.0.0",
            author="Test Author",
            description="Test Description",
            capabilities={"feature1", "feature2"},
        )

        # Verify all fields have correct types
        assert isinstance(metadata.name, str)
        assert isinstance(metadata.version, str)
        assert isinstance(metadata.author, str)
        assert isinstance(metadata.description, str)
        assert isinstance(metadata.capabilities, set)

    def test_plugin_execution_context_typing(self) -> None:
        """Test PluginExecutionContext type annotations."""
        import time
        from datetime import datetime

        import pyarrow as pa

        from dqx.cache import CacheStats
        from dqx.common import ResultKey
        from dqx.orm.repositories import MetricStats

        context = PluginExecutionContext(
            suite_name="Test Suite",
            execution_id="test_exec_id",
            datasources=["ds1", "ds2"],
            key=ResultKey(datetime.now().date(), {"env": "test"}),
            timestamp=time.time(),
            duration_ms=100.5,
            results=[],
            symbols=[],
            trace=pa.table({}),
            metrics_stats=MetricStats(total_metrics=0, expired_metrics=0),
            cache_stats=CacheStats(hit=0, missed=0),
        )

        # Verify method return types
        total: int = context.total_assertions()
        passed: int = context.passed_assertions()
        failed: int = context.failed_assertions()
        pass_rate: float = context.assertion_pass_rate()

        assert isinstance(total, int)
        assert isinstance(passed, int)
        assert isinstance(failed, int)
        assert isinstance(pass_rate, float)

        # Dictionary return types
        by_severity: dict[str, int] = context.failures_by_severity()
        assert isinstance(by_severity, dict)

    def test_get_plugins_return_type(self) -> None:
        """Test that get_plugins returns correctly typed dictionary."""
        manager = PluginManager()
        plugins: dict[str, PostProcessor] = manager.get_plugins()

        # Verify the dictionary is properly typed
        assert isinstance(plugins, dict)
        for name, plugin in plugins.items():
            assert isinstance(name, str)
            assert isinstance(plugin, PostProcessor)

    def test_timeout_property_typing(self) -> None:
        """Test timeout_seconds property type."""
        manager = PluginManager(timeout_seconds=30.5)

        # Property should return float
        timeout: float = manager.timeout_seconds
        assert isinstance(timeout, float)
        assert timeout == 30.5

    def test_plugin_validation_with_invalid_types(self) -> None:
        """Test that invalid plugin types are caught."""
        manager = PluginManager()

        # These should be caught by type checking
        invalid_plugins = [
            123,
            [],
            {},
            None,
            lambda x: x,
        ]

        for invalid in invalid_plugins:
            with pytest.raises((ValueError, AttributeError)):
                manager.register_plugin(invalid)  # type: ignore[call-overload]

    def test_runtime_checkable_protocol(self) -> None:
        """Test that PostProcessor is runtime checkable."""
        # Valid plugin
        plugin = TypedPlugin()
        assert isinstance(plugin, PostProcessor)

        # Invalid object
        class NotAPlugin:
            pass

        not_plugin = NotAPlugin()
        assert not isinstance(not_plugin, PostProcessor)

    def test_type_inference_in_plugin_usage(self) -> None:
        """Test type inference works correctly when using plugins."""
        manager = PluginManager()

        # Register a typed plugin
        plugin = TypedPlugin()
        manager.register_plugin(plugin)

        # Get it back - type should be inferred as PostProcessor
        retrieved = manager.get_plugins()["typed"]

        # Should be able to call protocol methods without cast
        metadata = retrieved.metadata()
        assert metadata.name == "typed"

        # If we know the concrete type, we can cast
        typed_plugin = cast(TypedPlugin, retrieved)
        assert typed_plugin is plugin


class TestMultiPluginTypeChecking:
    """Tests for type checking of multi-plugin registration overloads."""

    def test_overload_single_string(self) -> None:
        """Test single string registration type checking."""
        manager = PluginManager()

        # Should accept single string
        manager.register_plugin("dqx.plugins.AuditPlugin")

        # Type checker should infer correct signature

    def test_overload_single_instance(self) -> None:
        """Test single instance registration type checking."""
        manager = PluginManager()

        # Should accept single instance
        plugin = TypedPlugin()
        manager.register_plugin(plugin)

        # Type checker should infer correct signature

    def test_overload_multiple_mixed(self) -> None:
        """Test multiple mixed-type plugins type checking."""
        manager = PluginManager()

        # Should accept multiple plugins of mixed types
        plugin = TypedPlugin()
        manager.register_plugin("dqx.plugins.AuditPlugin", plugin, "dqx.plugins.AuditPlugin")

        # Type checker should accept variadic args

    def test_invalid_type_rejected(self) -> None:
        """Test that invalid types are rejected by type checker."""
        manager = PluginManager()

        # These would fail type checking:
        # manager.register_plugin(123)  # type: ignore[call-overload]
        # manager.register_plugin([])   # type: ignore[call-overload]
        # manager.register_plugin({})   # type: ignore[call-overload]

        # Runtime validation also catches these
        with pytest.raises((ValueError, AttributeError)):
            manager.register_plugin(123)  # type: ignore[call-overload]

    def test_list_argument_rejected(self) -> None:
        """Test that passing a list directly is rejected by type checker."""
        manager = PluginManager()

        # This would fail type checking (list is not valid arg type):
        # plugins_list = ["dqx.plugins.AuditPlugin", TypedPlugin()]
        # manager.register_plugin(plugins_list)  # type: ignore[call-overload]

        # But unpacking works:
        plugins_typed: list[str | TypedPlugin] = ["dqx.plugins.AuditPlugin", TypedPlugin()]
        manager.register_plugin(*plugins_typed)

    def test_mixed_types_accepted(self) -> None:
        """Test that mixed string and instance types are accepted."""
        manager = PluginManager()

        plugin1 = TypedPlugin()
        plugin2 = TypedPlugin()

        # All valid combinations
        manager.clear_plugins()
        manager.register_plugin("dqx.plugins.AuditPlugin", plugin1)

        manager.clear_plugins()
        manager.register_plugin(plugin1, "dqx.plugins.AuditPlugin")

        manager.clear_plugins()
        manager.register_plugin("dqx.plugins.AuditPlugin", plugin1, plugin2)

        manager.clear_plugins()
        manager.register_plugin(plugin1, plugin2, "dqx.plugins.AuditPlugin")

    def test_empty_call_rejected_by_mypy(self) -> None:
        """Test that empty calls are rejected by mypy at type-check time.

        This test documents that register_plugin() with no arguments is
        forbidden by the type signature. MyPy will catch this at static
        analysis time.
        """
        # This would fail type checking (no arguments provided):
        # manager = PluginManager()
        # manager.register_plugin()  # type: ignore[call-overload]

        # The implementation signature requires first argument:
        # def register_plugin(self, plugin: str | PostProcessor, *plugins: ...) -> None

        # Therefore empty calls are impossible at both type-check and runtime level
        pass

    def test_unpacking_with_correct_types(self) -> None:
        """Test unpacking works with properly typed sequences."""
        manager = PluginManager()

        # Tuple with correct types
        plugins_tuple: tuple[str, TypedPlugin] = ("dqx.plugins.AuditPlugin", TypedPlugin())
        manager.register_plugin(*plugins_tuple)

        # List with correct types
        manager.clear_plugins()
        plugins_list: list[str | TypedPlugin] = ["dqx.plugins.AuditPlugin", TypedPlugin()]
        manager.register_plugin(*plugins_list)

    def test_return_type_is_none(self) -> None:
        """Test that register_plugin returns None (no return value)."""
        manager = PluginManager()

        # Return type should be None
        result = manager.register_plugin("dqx.plugins.AuditPlugin")
        assert result is None

        # Multi-plugin also returns None
        result2 = manager.register_plugin("dqx.plugins.AuditPlugin", TypedPlugin())
        assert result2 is None
