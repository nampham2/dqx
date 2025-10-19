"""Integration tests for the DQX plugin system."""

import time

from dqx.common import (
    PluginExecutionContext,
    PluginMetadata,
)


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
