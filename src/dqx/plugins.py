"""Plugin system for DQX result processing."""

import importlib
import importlib.metadata
import logging
from typing import Protocol, overload, runtime_checkable

from rich.console import Console

from dqx.common import (
    PluginExecutionContext,
    PluginMetadata,
)
from dqx.timer import TimeLimitExceededError, TimeLimiting

logger = logging.getLogger(__name__)

# Hard time limit for plugin execution
PLUGIN_TIMEOUT_SECONDS = 60


@runtime_checkable
class PostProcessor(Protocol):
    """Protocol for DQX post-processor plugins."""

    @staticmethod
    def metadata() -> PluginMetadata:
        """Return plugin metadata."""
        ...

    def process(self, context: PluginExecutionContext) -> None:
        """
        Process validation results after suite execution.

        Args:
            context: Execution context with results, suite name, and convenience methods
        """
        ...


class PluginManager:
    """Manages DQX result processor plugins."""

    def __init__(self, *, timeout_seconds: float = PLUGIN_TIMEOUT_SECONDS) -> None:
        """
        Initialize the plugin manager.

        Args:
            _timeout_seconds: Timeout in seconds for plugin execution.
                            Defaults to PLUGIN_TIMEOUT_SECONDS (60).
        """
        self._plugins: dict[str, PostProcessor] = {}
        self._timeout_seconds: float = timeout_seconds

        # Register built-in plugins
        self.register_plugin("dqx.plugins.AuditPlugin")

    @property
    def timeout_seconds(self) -> float:
        """Get the plugin execution timeout in seconds."""
        return self._timeout_seconds

    def get_plugins(self) -> dict[str, PostProcessor]:
        """
        Get all loaded plugins.

        Returns:
            Dictionary mapping plugin names to plugin instances
        """
        return self._plugins

    def plugin_exists(self, name: str) -> bool:
        """
        Check if a plugin with the given name is registered.

        Args:
            name: Name of the plugin to check

        Returns:
            True if the plugin exists, False otherwise
        """
        return name in self._plugins

    @overload
    def register_plugin(self, plugin: str) -> None:
        """Register a plugin by its fully qualified class name."""
        ...

    @overload
    def register_plugin(self, plugin: PostProcessor) -> None:
        """Register a plugin by passing a PostProcessor instance directly."""
        ...

    def register_plugin(self, plugin: str | PostProcessor) -> None:
        """
        Register a plugin either by class name or PostProcessor instance.

        Args:
            plugin: Either a fully qualified class name string or a PostProcessor instance

        Raises:
            ValueError: If the plugin is invalid or doesn't implement PostProcessor
        """
        if isinstance(plugin, str):
            self._register_from_class(plugin)
        else:
            self._register_from_instance(plugin)

    def unregister_plugin(self, name: str) -> None:
        """
        Remove a plugin by name.

        Args:
            name: Name of the plugin to remove
        """
        if name in self._plugins:
            del self._plugins[name]
            logger.info(f"Unregistered plugin: {name}")

    def _register_from_class(self, class_name: str) -> None:
        """Register a plugin from a class name string (existing logic)."""
        try:
            # Move ALL existing register_plugin logic here unchanged
            # Parse the class name
            parts = class_name.rsplit(".", 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid class name format: {class_name}")

            module_name, cls_name = parts

            # Import the module
            try:
                module = importlib.import_module(module_name)
            except ImportError as e:
                raise ValueError(f"Cannot import module {module_name}: {e}")

            # Get the class
            if not hasattr(module, cls_name):
                raise ValueError(f"Module {module_name} has no class {cls_name}")

            plugin_class = getattr(module, cls_name)

            # Instantiate the plugin
            plugin = plugin_class()

            # Register the instance
            self._register_from_instance(plugin)
        except Exception as e:
            # Re-raise ValueError, let other exceptions propagate
            if not isinstance(e, ValueError):
                raise ValueError(f"Failed to register plugin {class_name}: {e}")
            raise

    def _register_from_instance(self, plugin: PostProcessor) -> None:
        """Register a PostProcessor instance directly."""
        # Use isinstance to check protocol implementation
        if not isinstance(plugin, PostProcessor):
            raise ValueError(f"Plugin instance {type(plugin).__name__} doesn't implement PostProcessor protocol")

        # Validate metadata returns correct type
        try:
            metadata = plugin.metadata()
        except Exception as e:
            raise ValueError(f"Failed to get metadata from plugin instance: {e}")

        if not isinstance(metadata, PluginMetadata):
            raise ValueError(
                f"Plugin instance's metadata() must return a PluginMetadata instance, got {type(metadata).__name__}"
            )

        plugin_name = metadata.name

        # Store the plugin instance
        self._plugins[plugin_name] = plugin
        logger.info(f"Registered plugin: {plugin_name} (instance)")

    def clear_plugins(self) -> None:
        """Remove all registered plugins."""
        self._plugins.clear()
        logger.info("Cleared all plugins")

    def process_all(self, context: PluginExecutionContext) -> None:
        """
        Process results through all loaded plugins with time limits.

        Args:
            context: Execution context with results, suite name, and convenience methods
        """
        if not self._plugins:
            logger.debug("No plugins loaded, skipping plugin processing")
            return

        logger.info(f"Processing results through {len(self._plugins)} plugin(s)")

        for name, plugin in self._plugins.items():
            try:
                # Hard time limit for plugin execution
                with TimeLimiting(int(self.timeout_seconds)) as timer:
                    plugin.process(context)

                logger.info(f"Plugin {name} processed results in {timer.elapsed_ms():.2f} ms")

            except TimeLimitExceededError:
                logger.error(f"Plugin {name} exceeded {self.timeout_seconds}s time limit")
            except Exception as e:
                # Log error but don't fail the entire suite
                logger.error(f"Plugin {name} failed during processing: {e}")


class AuditPlugin:
    """
    DQX built-in audit plugin for tracking suite execution.

    This plugin provides basic auditing functionality including:
    - Execution timing and performance metrics
    - Text-based result statistics with Rich color markup
    - Tag display with proper formatting
    - Symbol execution tracking
    """

    @staticmethod
    def metadata() -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="audit",
            version="1.0.0",
            author="DQX Team",
            description="Display execution audit report in text format with Rich color markup",
            capabilities={"console_output", "statistics"},
        )

    def __init__(self) -> None:
        """Initialize the audit plugin."""
        self.console = Console()

    def process(self, context: PluginExecutionContext) -> None:
        """
        Process and display validation results.

        Args:
            context: Execution context with results and convenience methods
        """
        self.console.print()
        self.console.print("[bold blue]═══ DQX Audit Report ═══[/bold blue]")
        self.console.print(f"[cyan]Suite:[/cyan] {context.suite_name}")
        self.console.print(f"[cyan]Date:[/cyan] {context.key.yyyy_mm_dd}")

        # Tags handling
        if context.key.tags:
            sorted_tags = ", ".join(f"{k}={v}" for k, v in sorted(context.key.tags.items()))
            self.console.print(f"[cyan]Tags:[/cyan] {sorted_tags}")
        else:
            self.console.print("[cyan]Tags:[/cyan] none")

        self.console.print(f"[cyan]Duration:[/cyan] {f'{context.duration_ms:.2f}ms'}", highlight=False)

        if context.datasources:
            self.console.print(f"[cyan]Datasets:[/cyan] {', '.join(context.datasources)}")

        # Calculate statistics
        total = context.total_assertions()
        passed = context.passed_assertions()
        failed = context.failed_assertions()
        pass_rate = (passed / total * 100) if total > 0 else 0.0

        self.console.print()
        self.console.print("[cyan]Execution Summary:[/cyan]")

        # Assertions line
        if total > 0:
            self.console.print(
                f"  Assertions: {total} total, [green]{passed} passed ({pass_rate:.1f}%)[/green], [red]{failed} failed ({100 - pass_rate:.1f}%)[/red]"
            )
        else:
            self.console.print("  Assertions: 0 total, 0 passed (0.0%), 0 failed (0.0%)")

        # Symbols line (only if symbols exist)
        if context.symbols:
            successful_symbols = context.total_symbols() - context.failed_symbols()
            failed_symbols = context.failed_symbols()
            self.console.print(
                f"  Symbols: {context.total_symbols()} total, {successful_symbols} successful, {failed_symbols} failed"
            )

        self.console.print("[bold blue]══════════════════════[/bold blue]")
        self.console.print()
