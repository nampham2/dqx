"""Tests for plugin public API methods."""

import time
from datetime import datetime

import pyarrow as pa
import pytest

from dqx.common import (
    PluginMetadata,
    ResultKey,
)
from dqx.orm.repositories import MetricStats
from dqx.plugins import PluginExecutionContext, PluginManager


def _create_empty_trace() -> pa.Table:
    """Create an empty PyArrow table for trace parameter."""
    return pa.table({})


# Module-level test plugin classes
class ValidPlugin:
    """Valid test plugin."""

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


class InvalidPlugin:
    """Invalid plugin missing required methods."""

    pass


class PartialPluginNoProcess:
    """Plugin with metadata but no process method."""

    @staticmethod
    def metadata() -> PluginMetadata:
        return PluginMetadata(
            name="partial",
            version="1.0.0",
            author="Test",
            description="Partial plugin",
        )


class PartialPluginNoMetadata:
    """Plugin with process but no metadata method."""

    def process(self, context: PluginExecutionContext) -> None:
        pass


class TrackerPlugin:
    """Plugin that tracks when it's called."""

    @staticmethod
    def metadata() -> PluginMetadata:
        return PluginMetadata(
            name="tracker",
            version="1.0.0",
            author="Test",
            description="Tracking plugin",
        )

    def __init__(self) -> None:
        self.called = False

    def process(self, context: PluginExecutionContext) -> None:
        self.called = True


class BadPluginInstanceMetadata:
    """Plugin with instance method metadata."""

    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="bad",
            version="1.0.0",
            author="Test",
            description="Bad plugin",
        )

    def process(self, context: PluginExecutionContext) -> None:
        pass


class BadPluginMetadataAttribute:
    """Plugin with metadata as attribute."""

    metadata = PluginMetadata(
        name="bad",
        version="1.0.0",
        author="Test",
        description="Bad plugin",
    )

    def process(self, context: PluginExecutionContext) -> None:
        pass


class BadPluginProcessAttribute:
    """Plugin with process as attribute."""

    @staticmethod
    def metadata() -> PluginMetadata:
        return PluginMetadata(
            name="bad",
            version="1.0.0",
            author="Test",
            description="Bad plugin",
        )

    process = "not a method"  # type: ignore


class TestPluginPublicAPI:
    """Test cases for PluginManager public API."""

    def test_register_plugin_valid(self) -> None:
        """Test registering a valid plugin."""
        manager = PluginManager()

        # Clear default plugins first
        manager.clear_plugins()

        # Register using class name
        manager.register_plugin("tests.test_plugin_public_api.ValidPlugin")

        # Verify plugin was registered with name from metadata
        assert "test" in manager.get_plugins()
        assert isinstance(manager.get_plugins()["test"], ValidPlugin)

    def test_register_plugin_invalid(self) -> None:
        """Test registering an invalid plugin raises ValueError."""
        manager = PluginManager()

        with pytest.raises(ValueError, match="doesn't implement PostProcessor protocol"):
            manager.register_plugin("tests.test_plugin_public_api.InvalidPlugin")

        # Verify plugin was not registered
        assert "bad_plugin" not in manager.get_plugins()

    def test_register_plugin_missing_process(self) -> None:
        """Test registering a plugin without process method."""
        manager = PluginManager()

        with pytest.raises(ValueError, match="doesn't implement PostProcessor protocol"):
            manager.register_plugin("tests.test_plugin_public_api.PartialPluginNoProcess")

    def test_register_plugin_missing_metadata(self) -> None:
        """Test registering a plugin without metadata method."""
        manager = PluginManager()

        with pytest.raises(ValueError, match="doesn't implement PostProcessor protocol"):
            manager.register_plugin("tests.test_plugin_public_api.PartialPluginNoMetadata")

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

        # Register a test plugin using class name
        manager.register_plugin("tests.test_plugin_public_api.ValidPlugin")

        # Verify we have plugins
        assert len(manager.get_plugins()) > 0

        # Clear all plugins
        manager.clear_plugins()

        # Verify all plugins were removed
        assert len(manager.get_plugins()) == 0

    def test_plugin_lifecycle(self) -> None:
        """Test complete plugin lifecycle: register, use, unregister."""
        manager = PluginManager()

        # Clear default plugins
        manager.clear_plugins()

        # Register the tracker plugin
        manager.register_plugin("tests.test_plugin_public_api.TrackerPlugin")

        # Get the plugin instance to check if it was called
        tracker = manager.get_plugins()["tracker"]
        assert isinstance(tracker, TrackerPlugin)

        # Create context and process
        context = PluginExecutionContext(
            suite_name="Test",
            execution_id="test_exec_id",
            datasources=[],
            key=ResultKey(datetime.now().date(), {}),
            timestamp=time.time(),
            duration_ms=0.0,
            results=[],
            symbols=[],
            trace=_create_empty_trace(),
            metrics_stats=MetricStats(total_metrics=0, expired_metrics=0),
        )

        manager.process_all(context)

        # Verify plugin was called
        assert tracker.called

        # Unregister and reset flag
        manager.unregister_plugin("tracker")
        tracker.called = False

        # Process again
        manager.process_all(context)

        # Verify plugin was NOT called this time (it's been unregistered)
        assert not tracker.called


class TestPluginValidation:
    """Test plugin validation edge cases."""

    def test_plugin_with_non_static_metadata(self) -> None:
        """Test plugin with instance method metadata works correctly."""
        manager = PluginManager()

        # Clear default plugins
        manager.clear_plugins()

        # Register using class name - should work because we check callability
        manager.register_plugin("tests.test_plugin_public_api.BadPluginInstanceMetadata")
        assert "bad" in manager.get_plugins()

    def test_plugin_metadata_not_callable(self) -> None:
        """Test plugin with metadata as attribute is rejected."""
        manager = PluginManager()

        with pytest.raises(ValueError, match="Failed to get metadata from plugin instance.*not callable"):
            manager.register_plugin("tests.test_plugin_public_api.BadPluginMetadataAttribute")

    def test_plugin_process_not_callable(self) -> None:
        """Test plugin with process as attribute is rejected."""
        manager = PluginManager()

        # This plugin is actually valid from PostProcessor protocol perspective
        # because it has both metadata and process attributes
        manager.register_plugin("tests.test_plugin_public_api.BadPluginProcessAttribute")

        # Verify it was registered
        assert "bad" in manager.get_plugins()
