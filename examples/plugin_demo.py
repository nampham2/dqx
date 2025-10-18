#!/usr/bin/env python3
"""
DQX Plugin System Demo

This example demonstrates:
1. Creating a verification suite with plugin support
2. Using the built-in audit plugin
3. Creating and using custom plugins
4. Plugin execution timing and error handling
"""

import time
from datetime import datetime

from returns.result import Failure, Success

from dqx.common import (
    AssertionResult,
    EvaluationFailure,
    PluginExecutionContext,
    PluginMetadata,
    ResultKey,
    SymbolInfo,
)
from dqx.plugins import PluginManager


# Custom plugin example 1: JSON reporter
class JSONReporterPlugin:
    """Plugin that outputs results as JSON."""

    @staticmethod
    def metadata() -> PluginMetadata:
        return PluginMetadata(
            name="json_reporter",
            version="1.0.0",
            author="Example Author",
            description="Outputs validation results as JSON",
            capabilities={"json_output"},
        )

    def process(self, context: PluginExecutionContext) -> None:
        """Process and output results as JSON."""
        import json

        report = {
            "suite": context.suite_name,
            "date": context.key.yyyy_mm_dd.isoformat(),
            "duration_ms": context.duration_ms,
            "datasources": context.datasources,
            "summary": {
                "total_assertions": context.total_assertions(),
                "passed": context.passed_assertions(),
                "failed": context.failed_assertions(),
                "pass_rate": context.assertion_pass_rate(),
            },
            "failures_by_severity": context.failures_by_severity(),
        }

        print("\n=== JSON Report ===")
        print(json.dumps(report, indent=2))


# Custom plugin example 2: Metrics collector
class MetricsCollectorPlugin:
    """Plugin that collects and displays metrics."""

    @staticmethod
    def metadata() -> PluginMetadata:
        return PluginMetadata(
            name="metrics_collector",
            version="1.0.0",
            author="Metrics Team",
            description="Collects and displays validation metrics",
            capabilities={"metrics", "statistics"},
        )

    def process(self, context: PluginExecutionContext) -> None:
        """Collect and display metrics."""
        print("\n=== Metrics Report ===")
        print(f"Execution Time: {context.duration_ms:.2f}ms")

        # Calculate average assertion time
        if context.total_assertions() > 0:
            avg_time = context.duration_ms / context.total_assertions()
            print(f"Average Assertion Time: {avg_time:.2f}ms")

        # Show symbol statistics
        if context.symbols:
            print("\nSymbol Statistics:")
            print(f"  Total Symbols: {context.total_symbols()}")
            print(f"  Failed Symbols: {context.failed_symbols()}")

            # Group symbols by dataset
            by_dataset: dict[str, list[SymbolInfo]] = {}
            for symbol in context.symbols:
                dataset = symbol.dataset or "unknown"
                if dataset not in by_dataset:
                    by_dataset[dataset] = []
                by_dataset[dataset].append(symbol)

            for dataset, symbols in by_dataset.items():
                print(f"  Dataset '{dataset}': {len(symbols)} symbols")


def demo_plugin_system() -> None:
    """Demonstrate the plugin system."""
    print("DQX Plugin System Demo")
    print("=" * 50)

    # Create plugin manager
    manager = PluginManager()

    # Add custom plugins
    manager._plugins["json_reporter"] = JSONReporterPlugin()
    manager._plugins["metrics_collector"] = MetricsCollectorPlugin()

    print(f"\nLoaded {len(manager.get_plugins())} plugins:")
    for name, metadata in manager.get_metadata().items():
        print(f"  - {name} v{metadata.version}: {metadata.description}")

    # Create sample validation results
    results = [
        AssertionResult(
            yyyy_mm_dd=datetime.now().date(),
            suite="ProductQuality",
            check="PriceValidation",
            assertion="price_positive",
            severity="P0",
            status="OK",
            metric=Success(99.5),
            expression="price > 0",
            tags={"env": "prod"},
        ),
        AssertionResult(
            yyyy_mm_dd=datetime.now().date(),
            suite="ProductQuality",
            check="PriceValidation",
            assertion="price_reasonable",
            severity="P1",
            status="OK",
            metric=Success(95.2),
            expression="price < 10000",
            tags={"env": "prod"},
        ),
        AssertionResult(
            yyyy_mm_dd=datetime.now().date(),
            suite="ProductQuality",
            check="InventoryValidation",
            assertion="stock_non_negative",
            severity="P0",
            status="FAILURE",
            metric=Failure([EvaluationFailure("Negative stock found", "stock >= 0", [])]),
            expression="stock >= 0",
            tags={"env": "prod"},
        ),
        AssertionResult(
            yyyy_mm_dd=datetime.now().date(),
            suite="ProductQuality",
            check="DescriptionValidation",
            assertion="description_not_empty",
            severity="P2",
            status="OK",
            metric=Success(100.0),
            expression="length(description) > 0",
            tags={"env": "prod"},
        ),
    ]

    # Create sample symbols
    symbols = [
        SymbolInfo(
            name="avg_price",
            metric="average(price)",
            dataset="products",
            value=Success(250.75),
            yyyy_mm_dd=datetime.now().date(),
            suite="ProductQuality",
            tags={"env": "prod"},
        ),
        SymbolInfo(
            name="total_stock",
            metric="sum(stock)",
            dataset="inventory",
            value=Success(15420.0),
            yyyy_mm_dd=datetime.now().date(),
            suite="ProductQuality",
            tags={"env": "prod"},
        ),
        SymbolInfo(
            name="null_descriptions",
            metric="count_if(description IS NULL)",
            dataset="products",
            value=Failure("Query execution failed"),
            yyyy_mm_dd=datetime.now().date(),
            suite="ProductQuality",
            tags={"env": "prod"},
        ),
    ]

    # Create execution context
    context = PluginExecutionContext(
        suite_name="ProductQuality",
        datasources=["products", "inventory"],
        key=ResultKey(datetime.now().date(), {"env": "prod"}),
        timestamp=time.time(),
        duration_ms=342.5,
        results=results,
        symbols=symbols,
    )

    # Process through all plugins
    print("\n" + "=" * 50)
    print("Processing results through plugins...")
    print("=" * 50)

    manager.process_all(context)

    print("\n" + "=" * 50)
    print("Plugin processing complete!")


def demo_plugin_error_handling() -> None:
    """Demonstrate plugin error handling."""
    print("\n\nPlugin Error Handling Demo")
    print("=" * 50)

    # Create a plugin that fails
    class FailingPlugin:
        @staticmethod
        def metadata() -> PluginMetadata:
            return PluginMetadata(
                name="failing_plugin",
                version="1.0.0",
                author="Test",
                description="A plugin that demonstrates error handling",
            )

        def process(self, context: PluginExecutionContext) -> None:
            raise RuntimeError("Simulated plugin failure!")

    # Create manager with custom timeout
    manager = PluginManager(_timeout_seconds=2)
    manager._plugins["failing"] = FailingPlugin()

    # Create minimal context
    context = PluginExecutionContext(
        suite_name="ErrorTest",
        datasources=[],
        key=ResultKey(datetime.now().date(), {}),
        timestamp=time.time(),
        duration_ms=10.0,
        results=[],
        symbols=[],
    )

    print("\nProcessing with failing plugin...")
    manager.process_all(context)

    print("\nNote: The suite continues despite plugin failure!")


def demo_plugin_timeout() -> None:
    """Demonstrate plugin timeout handling."""
    print("\n\nPlugin Timeout Demo")
    print("=" * 50)

    # Create a slow plugin
    class SlowPlugin:
        @staticmethod
        def metadata() -> PluginMetadata:
            return PluginMetadata(
                name="slow_plugin",
                version="1.0.0",
                author="Test",
                description="A plugin that takes too long",
            )

        def process(self, context: PluginExecutionContext) -> None:
            print("Starting slow operation...")
            time.sleep(3)  # This will timeout
            print("This won't be printed due to timeout")

    # Create manager with 1 second timeout
    manager = PluginManager(_timeout_seconds=1)
    manager._plugins = {}  # Clear default plugins
    manager._plugins["slow"] = SlowPlugin()

    # Create minimal context
    context = PluginExecutionContext(
        suite_name="TimeoutTest",
        datasources=[],
        key=ResultKey(datetime.now().date(), {}),
        timestamp=time.time(),
        duration_ms=10.0,
        results=[],
        symbols=[],
    )

    print("\nProcessing with slow plugin (1s timeout)...")
    start = time.time()
    manager.process_all(context)
    elapsed = time.time() - start

    print(f"\nActual execution time: {elapsed:.2f}s (limited by timeout)")


if __name__ == "__main__":
    # Run all demos
    demo_plugin_system()
    demo_plugin_error_handling()
    demo_plugin_timeout()

    print("\n" + "=" * 50)
    print("All demos complete!")
