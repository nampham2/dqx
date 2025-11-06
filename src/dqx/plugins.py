"""Plugin system for DQX result processing."""

import importlib
import importlib.metadata
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, overload, runtime_checkable

import pyarrow as pa
from returns.pipeline import is_successful
from rich.console import Console

from dqx.common import (
    AssertionResult,
    DQXError,
    PluginMetadata,
    ResultKey,
)
from dqx.provider import SymbolInfo
from dqx.timer import TimeLimitExceededError, TimeLimiting

if TYPE_CHECKING:
    from dqx.cache import CacheStats
    from dqx.data import MetricTraceStats
    from dqx.orm.repositories import MetricStats

logger = logging.getLogger(__name__)

# Hard time limit for plugin execution
PLUGIN_TIMEOUT_SECONDS = 60


@dataclass
class PluginExecutionContext:
    """Execution context passed to plugins."""

    suite_name: str
    execution_id: str
    datasources: list[str]
    key: ResultKey
    timestamp: float
    duration_ms: float
    results: list[AssertionResult]
    symbols: list[SymbolInfo]
    trace: pa.Table
    metrics_stats: "MetricStats"
    cache_stats: "CacheStats"

    def total_assertions(self) -> int:
        """Total number of assertions."""
        return len(self.results)

    def failed_assertions(self) -> int:
        """Number of failed assertions."""
        return sum(1 for r in self.results if r.status in ("FAILED", "FAILURE"))

    def passed_assertions(self) -> int:
        """Number of passed assertions."""
        return sum(1 for r in self.results if r.status == "PASSED")

    def skipped_assertions(self) -> int:
        """Number of skipped assertions."""
        return sum(1 for r in self.results if r.status == "SKIPPED")

    def error_assertions(self) -> int:
        """Number of error assertions."""
        return sum(1 for r in self.results if r.status == "ERROR")

    def assertion_pass_rate(self) -> float:
        """Pass rate as percentage (0-100)."""
        if not self.results:
            return 100.0
        return (self.passed_assertions() / len(self.results)) * 100

    def total_symbols(self) -> int:
        """Total number of symbols."""
        return len(self.symbols)

    def failed_symbols(self) -> int:
        """Number of symbols with failed computations."""
        return sum(1 for s in self.symbols if not is_successful(s.value))

    def assertions_by_severity(self) -> dict[str, int]:
        """Count of assertions grouped by severity."""
        from collections import Counter

        return dict(Counter(r.severity for r in self.results))

    def failures_by_severity(self) -> dict[str, int]:
        """Count of failures grouped by severity."""
        from collections import Counter

        failures = [r for r in self.results if r.status in ("FAILED", "FAILURE")]
        return dict(Counter(f.severity for f in failures))

    def data_discrepancy_stats(self) -> "MetricTraceStats | None":
        """Get data discrepancy statistics from trace table."""
        if self.trace is None or self.trace.num_rows == 0:
            return None
        from dqx.data import metric_trace_stats

        return metric_trace_stats(self.trace)


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
            capabilities={"verification", "statistics"},
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
            self.console.print("[cyan]Tags:[/cyan] None")

        self.console.print(f"[cyan]Duration:[/cyan] {f'{context.duration_ms:.2f}ms'}", highlight=False)

        if context.datasources:
            if len(context.datasources) == 1:
                # Single dataset - singular label and inline display
                self.console.print(f"[cyan]Dataset:[/cyan] {context.datasources[0]}")
            else:
                # Multiple datasets - plural label and bulleted list
                self.console.print("[cyan]Datasets:[/cyan]")
                for dataset in context.datasources:
                    self.console.print(f"  - {dataset}")

        # Calculate statistics
        total = context.total_assertions()
        passed = context.passed_assertions()
        failed = context.failed_assertions()
        skipped = context.skipped_assertions()
        error = context.error_assertions()

        self.console.print()
        self.console.print("[cyan]Execution Summary:[/cyan]")

        # Assertions line
        if total > 0:
            # Build the assertions line dynamically
            assertion_parts = [f"{total} total"]

            # Always show passed
            pass_rate = (passed / total * 100) if total > 0 else 0.0
            assertion_parts.append(f"[green]{passed} passed ({pass_rate:.1f}%)[/green]")

            # Only show failed if > 0
            if failed > 0:
                fail_rate = failed / total * 100
                assertion_parts.append(f"[red]{failed} failed ({fail_rate:.1f}%)[/red]")

            # Only show skipped if > 0
            if skipped > 0:
                skip_rate = skipped / total * 100
                assertion_parts.append(f"[yellow]{skipped} skipped ({skip_rate:.1f}%)[/yellow]")

            # Only show error if > 0
            if error > 0:
                error_rate = error / total * 100
                assertion_parts.append(f"[red]{error} error ({error_rate:.1f}%)[/red]")

            self.console.print(f"  Assertions: {', '.join(assertion_parts)}")
        else:
            self.console.print("  Assertions: 0 total, 0 passed (0.0%), 0 failed (0.0%)")

        # Symbols line (show whenever symbols are present)
        if context.symbols:
            total_symbols = context.total_symbols()
            successful_symbols = total_symbols - context.failed_symbols()
            failed_symbols = context.failed_symbols()

            if failed_symbols > 0:
                # Show both successful and failed when there are failures
                success_rate = (successful_symbols / total_symbols * 100) if total_symbols > 0 else 0.0
                fail_rate = (failed_symbols / total_symbols * 100) if total_symbols > 0 else 0.0
                self.console.print(
                    f"  Symbols: {total_symbols} total, [green]{successful_symbols} successful ({success_rate:.1f}%)[/green], [red]{failed_symbols} failed ({fail_rate:.1f}%)[/red]"
                )
            else:
                # Show only successful when all symbols succeeded
                self.console.print(
                    f"  Symbols: {total_symbols} total, [green]{total_symbols} successful (100.0%)[/green]"
                )

        # Metrics cleanup line (if any metrics were cleaned up)
        self.console.print(
            f"  Metrics Cleanup: [green]{context.metrics_stats.expired_metrics} expired metrics removed[/green]"
        )

        # Cache performance line
        stats = context.cache_stats
        if stats.hit + stats.missed > 0:
            hit_rate = stats.hit_ratio() * 100
            self.console.print(
                f"  Cache Performance: [green]hit: {stats.hit}[/green], [red]missed: {stats.missed}[/red] ({hit_rate:.1f}% hit rate)"
            )
        else:
            self.console.print("  Cache Performance: [green]hit: 0[/green], [red]missed: 0[/red]")

        # Data discrepancy line
        discrepancy_stats = context.data_discrepancy_stats()
        if discrepancy_stats:
            if discrepancy_stats.discrepancy_count == 0:
                self.console.print("  Data Integrity: [green]✓ No discrepancies found[/green]")
            else:
                # Count discrepancy types
                from collections import Counter

                discrepancy_types: Counter[str] = Counter()
                for detail in discrepancy_stats.discrepancy_details:
                    for discrepancy in detail["discrepancies"]:
                        discrepancy_types[discrepancy] += 1

                # Format the discrepancy summary compactly
                type_summary = ", ".join(
                    f"{count}x {disc.replace('_', '').replace('value', '').replace(' != ', '≠')}"
                    for disc, count in discrepancy_types.most_common()
                )
                self.console.print(
                    f"  Data Integrity: [yellow]⚠️  {discrepancy_stats.discrepancy_count} discrepancies ({type_summary})[/yellow]"
                )
                from dqx import display

                display.print_metrics_by_execution_id(context.trace, context.execution_id)
                raise DQXError("[InternalError] Data discrepancies detected during audit")

        self.console.print("[bold blue]══════════════════════[/bold blue]")
        self.console.print()
