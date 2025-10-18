"""Tests for plugin public API methods."""

import time
from datetime import datetime

import pytest

from dqx.common import (
    PluginExecutionContext,
    PluginMetadata,
    ResultKey,
)
from dqx.plugins import PluginManager


class TestPluginPublicAPI:
    """Test cases for PluginManager public API."""

    def test_register_plugin_valid(self) -> None:
        """Test registering a valid plugin."""
        manager = PluginManager()

        # Create a valid plugin
        class ValidPlugin:
            @staticmethod
            def metadata() -> PluginMetadata:
                return PluginMetadata(
                    name="test",
                    version="1.0.0",
                    author="Test",
                    description="Test plugin",
                )

            def process(self, context: PluginExecutionContext) -> None:
                pass

        plugin = ValidPlugin()
        manager.register_plugin("test", plugin)

        # Verify plugin was registered
        assert "test" in manager.get_plugins()
        assert manager.get_plugins()["test"] is plugin

    def test_register_plugin_invalid(self) -> None:
        """Test registering an invalid plugin raises ValueError."""
        manager = PluginManager()

        # Create an invalid plugin (missing methods)
        class InvalidPlugin:
            pass

        plugin = InvalidPlugin()

        with pytest.raises(ValueError, match="Invalid plugin: bad_plugin does not implement ResultProcessor protocol"):
            manager.register_plugin("bad_plugin", plugin)

        # Verify plugin was not registered
        assert "bad_plugin" not in manager.get_plugins()

    def test_register_plugin_missing_process(self) -> None:
        """Test registering a plugin without process method."""
        manager = PluginManager()

        # Plugin with metadata but no process
        class PartialPlugin:
            @staticmethod
            def metadata() -> PluginMetadata:
                return PluginMetadata(
                    name="partial",
                    version="1.0.0",
                    author="Test",
                    description="Partial plugin",
                )

        plugin = PartialPlugin()

        with pytest.raises(ValueError):
            manager.register_plugin("partial", plugin)

    def test_register_plugin_missing_metadata(self) -> None:
        """Test registering a plugin without metadata method."""
        manager = PluginManager()

        # Plugin with process but no metadata
        class PartialPlugin:
            def process(self, context: PluginExecutionContext) -> None:
                pass

        plugin = PartialPlugin()

        with pytest.raises(ValueError):
            manager.register_plugin("partial", plugin)

    def test_unregister_plugin_existing(self) -> None:
        """Test unregistering an existing plugin."""
        manager = PluginManager()

        # Verify audit plugin exists
        assert "audit" in manager.get_plugins()

        # Unregister it
        manager.unregister_plugin("audit")

        # Verify it was removed
        assert "audit" not in manager.get_plugins()

    def test_unregister_plugin_nonexistent(self) -> None:
        """Test unregistering a non-existent plugin (should not error)."""
        manager = PluginManager()

        # Should not raise an error
        manager.unregister_plugin("nonexistent")

        # Verify nothing changed
        assert "audit" in manager.get_plugins()

    def test_clear_plugins(self) -> None:
        """Test clearing all plugins."""
        manager = PluginManager()

        # Add a custom plugin
        class TestPlugin:
            @staticmethod
            def metadata() -> PluginMetadata:
                return PluginMetadata(
                    name="test",
                    version="1.0.0",
                    author="Test",
                    description="Test plugin",
                )

            def process(self, context: PluginExecutionContext) -> None:
                pass

        manager.register_plugin("test", TestPlugin())

        # Verify we have plugins
        assert len(manager.get_plugins()) > 0

        # Clear all plugins
        manager.clear_plugins()

        # Verify all plugins were removed
        assert len(manager.get_plugins()) == 0

    def test_plugin_lifecycle(self) -> None:
        """Test complete plugin lifecycle: register, use, unregister."""
        manager = PluginManager()

        # Track if plugin was called
        plugin_called = False

        # Create a plugin that tracks calls
        class TrackerPlugin:
            @staticmethod
            def metadata() -> PluginMetadata:
                return PluginMetadata(
                    name="tracker",
                    version="1.0.0",
                    author="Test",
                    description="Tracking plugin",
                )

            def process(self, context: PluginExecutionContext) -> None:
                nonlocal plugin_called
                plugin_called = True

        # Register the plugin
        tracker = TrackerPlugin()
        manager.register_plugin("tracker", tracker)

        # Create context and process
        context = PluginExecutionContext(
            suite_name="Test",
            datasources=[],
            key=ResultKey(datetime.now().date(), {}),
            timestamp=time.time(),
            duration_ms=0.0,
            results=[],
            symbols=[],
        )

        manager.process_all(context)

        # Verify plugin was called
        assert plugin_called

        # Unregister and reset flag
        manager.unregister_plugin("tracker")
        plugin_called = False

        # Process again
        manager.process_all(context)

        # Verify plugin was NOT called this time
        assert not plugin_called


class TestPluginValidation:
    """Test plugin validation edge cases."""

    def test_plugin_with_non_static_metadata(self) -> None:
        """Test plugin with instance method metadata is rejected."""
        manager = PluginManager()

        # Plugin with instance method metadata
        class BadPlugin:
            def metadata(self) -> PluginMetadata:
                return PluginMetadata(
                    name="bad",
                    version="1.0.0",
                    author="Test",
                    description="Bad plugin",
                )

            def process(self, context: PluginExecutionContext) -> None:
                pass

        plugin = BadPlugin()

        # Should still work because we check callability
        manager.register_plugin("bad", plugin)
        assert "bad" in manager.get_plugins()

    def test_plugin_metadata_not_callable(self) -> None:
        """Test plugin with metadata as attribute is rejected."""
        manager = PluginManager()

        # Plugin with metadata as attribute
        class BadPlugin:
            metadata = PluginMetadata(
                name="bad",
                version="1.0.0",
                author="Test",
                description="Bad plugin",
            )

            def process(self, context: PluginExecutionContext) -> None:
                pass

        plugin = BadPlugin()

        with pytest.raises(ValueError):
            manager.register_plugin("bad", plugin)

    def test_plugin_process_not_callable(self) -> None:
        """Test plugin with process as attribute is rejected."""
        manager = PluginManager()

        # Plugin with process as attribute
        class BadPlugin:
            @staticmethod
            def metadata() -> PluginMetadata:
                return PluginMetadata(
                    name="bad",
                    version="1.0.0",
                    author="Test",
                    description="Bad plugin",
                )

            process = "not a method"

        plugin = BadPlugin()

        with pytest.raises(ValueError):
            manager.register_plugin("bad", plugin)
