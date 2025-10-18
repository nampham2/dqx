"""Plugin system for DQX result processing."""

import importlib.metadata
import logging
from typing import Protocol, runtime_checkable

from rich import box
from rich.console import Console
from rich.table import Table

from dqx.common import (
    PluginExecutionContext,
    PluginMetadata,
)
from dqx.timer import TimeLimitExceededError, TimeLimiting

logger = logging.getLogger(__name__)

# Hard time limit for plugin execution
PLUGIN_TIMEOUT_SECONDS = 60


@runtime_checkable
class ResultProcessor(Protocol):
    """Protocol for DQX result processor plugins."""

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
        self._plugins: dict[str, ResultProcessor] = {}
        self._timeout_seconds: float = timeout_seconds
        self._discover_plugins()

    @property
    def timeout_seconds(self) -> float:
        """Get the plugin execution timeout in seconds."""
        return self._timeout_seconds

    def _discover_plugins(self) -> None:
        """Discover and load plugins from entry points."""
        try:
            # Discover all plugins in the "dqx.plugins" group
            entry_points = importlib.metadata.entry_points(group="dqx.plugins")

            for ep in entry_points:
                try:
                    logger.info(f"Loading plugin: {ep.name}")

                    # Load the plugin class
                    plugin_class = ep.load()

                    # Instantiate the plugin
                    plugin = plugin_class()

                    # Store the plugin
                    self._plugins[ep.name] = plugin

                    # Get metadata
                    metadata = plugin_class.metadata()
                    logger.info(f"Loaded plugin: {metadata.name} v{metadata.version}")

                except Exception:
                    logger.warning(f"Plugin not available: {ep.name}")

        except Exception as e:
            logger.error(f"Failed to discover plugins: {e}")

    def get_plugins(self) -> dict[str, ResultProcessor]:
        """
        Get all loaded plugins.

        Returns:
            Dictionary mapping plugin names to plugin instances
        """
        return self._plugins

    def get_metadata(self) -> dict[str, PluginMetadata]:
        """
        Get all plugin metadata by calling their static methods.

        Returns:
            Dictionary mapping plugin names to metadata
        """
        return {name: plugin.__class__.metadata() for name, plugin in self._plugins.items()}

    def plugin_name_exists(self, name: str) -> bool:
        """
        Check if a plugin with the given name is registered.

        Args:
            name: Name of the plugin to check

        Returns:
            True if the plugin exists, False otherwise
        """
        return name in self._plugins

    def register_plugin(self, name: str, plugin: object) -> None:
        """
        Register a plugin manually.

        Args:
            name: Unique name for the plugin
            plugin: Plugin instance that implements ResultProcessor protocol

        Raises:
            ValueError: If plugin doesn't implement ResultProcessor protocol
        """
        # Check if the plugin name already exists
        self._plugins[name] = plugin  # type: ignore[assignment]
        logger.info(f"Registered plugin: {name}")

    def unregister_plugin(self, name: str) -> None:
        """
        Remove a plugin by name.

        Args:
            name: Name of the plugin to remove
        """
        if name in self._plugins:
            del self._plugins[name]
            logger.info(f"Unregistered plugin: {name}")

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

                logger.info(f"Plugin {name} processed results in {timer.elapsed_ms():.2f}ms")

            except TimeLimitExceededError:
                logger.error(f"Plugin {name} exceeded {self.timeout_seconds}s time limit")
            except Exception as e:
                # Log error but don't fail the entire suite
                logger.error(f"Plugin {name} failed during processing: {e}")


class AuditPlugin:
    """
    DQX built-in audit plugin for tracking suite execution.

    This plugin provides basic auditing functionality including:
    - Execution timing
    - Result statistics with Rich table display
    - Performance metrics with colors
    """

    @staticmethod
    def metadata() -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="audit",
            version="1.0.0",
            author="DQX Team",
            description="Display execution audit report with Rich tables",
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
        # Use context methods for statistics
        total = context.total_assertions()
        passed = context.passed_assertions()
        failed = context.failed_assertions()
        pass_rate = context.assertion_pass_rate()

        # Display header
        self.console.print()
        self.console.print("[bold blue]═══ DQX Audit Report ═══[/bold blue]")
        self.console.print(f"[cyan]Suite:[/cyan] {context.suite_name}")
        self.console.print(f"[cyan]Date:[/cyan] {context.key.yyyy_mm_dd}")
        self.console.print(f"[cyan]Duration:[/cyan] {context.duration_ms:.2f}ms")
        if context.datasources:
            self.console.print(f"[cyan]Datasets:[/cyan] {', '.join(context.datasources)}")
        self.console.print()

        # Create summary table using context methods
        summary_table = Table(title="Execution Summary", box=box.ROUNDED)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Count", justify="right")
        summary_table.add_column("Rate", justify="right")

        summary_table.add_row("Total Assertions", str(total), "")

        if total > 0:
            summary_table.add_row(
                "[green]Passed ✓[/green]", f"[green]{passed}[/green]", f"[green]{pass_rate:.1f}%[/green]"
            )
            summary_table.add_row("[red]Failed ✗[/red]", f"[red]{failed}[/red]", f"[red]{100 - pass_rate:.1f}%[/red]")
        else:
            summary_table.add_row("Passed ✓", "0", "0.0%")
            summary_table.add_row("Failed ✗", "0", "0.0%")

        self.console.print(summary_table)

        # Show failures by severity if any
        failures_by_sev = context.failures_by_severity()
        if failures_by_sev:
            self.console.print()
            severity_table = Table(title="Failures by Severity", box=box.ROUNDED)
            severity_table.add_column("Severity", style="bold")
            severity_table.add_column("Count", justify="right")

            # Color code by severity
            severity_colors = {"P0": "red bold", "P1": "orange1", "P2": "yellow", "P3": "blue", "P4": "grey50"}

            for severity, count in sorted(failures_by_sev.items()):
                color = severity_colors.get(severity, "white")
                severity_table.add_row(f"[{color}]{severity}[/{color}]", f"[{color}]{count}[/{color}]")

            self.console.print(severity_table)

        # Show symbol statistics if any
        if context.symbols:
            successful_symbols = context.total_symbols() - context.failed_symbols()
            failed_symbols = context.failed_symbols()

            self.console.print()
            symbol_table = Table(title="Symbol Statistics", box=box.ROUNDED)
            symbol_table.add_column("Metric", style="cyan")
            symbol_table.add_column("Count", justify="right")

            symbol_table.add_row("Total Symbols", str(context.total_symbols()))
            symbol_table.add_row("[green]Successful[/green]", f"[green]{successful_symbols}[/green]")
            symbol_table.add_row("[red]Failed[/red]", f"[red]{failed_symbols}[/red]")

            self.console.print(symbol_table)

        self.console.print()
        self.console.print("[bold blue]══════════════════════[/bold blue]")
        self.console.print()
