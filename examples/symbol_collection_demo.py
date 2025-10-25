"""Demonstration of the symbol collection feature in DQX.

This example shows how to:
1. Use get_symbol with string inputs
2. Collect symbol information from expressions
3. Access enhanced SymbolInfo with context fields
"""

from datetime import date

from returns.result import Success

from dqx.common import ResultKey
from dqx.evaluator import Evaluator
from dqx.orm.repositories import InMemoryMetricDB
from dqx.provider import MetricProvider, SymbolInfo
from dqx.specs import Average


def main() -> None:
    # Set up the database and provider
    db = InMemoryMetricDB()
    provider = MetricProvider(db)

    # Create a result key with context
    key = ResultKey(yyyy_mm_dd=date(2025, 1, 13), tags={"env": "prod", "region": "us-east"})

    # Register some metrics
    avg_price = provider.average("price", dataset="orders")
    sum_quantity = provider.sum("quantity", dataset="orders")
    avg_rating = provider.average("rating", dataset="reviews")

    # For demo purposes, we'll simulate having metrics already evaluated
    # In a real scenario, these would come from analyzing data sources

    # Create an evaluator
    evaluator = Evaluator(provider, key, "Demo Suite")

    print("=== Symbol Collection Demo ===\n")

    # Example 1: Get symbol by string
    print("1. Getting symbol by string:")
    symbol_metric = provider.get_symbol("x_1")  # Using string instead of Symbol
    print(f"   Symbol x_1: {symbol_metric.name} ({symbol_metric.metric_spec})")
    print(f"   Dataset: {symbol_metric.dataset}")
    print()

    # Example 2: Collect symbols from an expression
    print("2. Collecting symbols from expression:")
    expr = avg_price * 2 + sum_quantity / 100
    symbols = evaluator.collect_symbols(expr)

    for symbol_info in symbols:
        print(f"   Symbol: {symbol_info.name}")
        print(f"   - Metric: {symbol_info.metric}")
        print(f"   - Dataset: {symbol_info.dataset}")
        print(f"   - Value: {symbol_info.value}")
        print(f"   - Date: {symbol_info.yyyy_mm_dd}")
        print(f"   - Tags: {symbol_info.tags}")
        print()

    # Example 3: Complex expression with multiple datasets
    print("3. Complex expression across datasets:")
    complex_expr = (avg_price + 10) * avg_rating / sum_quantity
    symbols = evaluator.collect_symbols(complex_expr)

    # Group by dataset
    by_dataset: dict[str | None, list[SymbolInfo]] = {}
    for si in symbols:
        if si.dataset not in by_dataset:
            by_dataset[si.dataset] = []
        by_dataset[si.dataset].append(si)

    for dataset, dataset_symbols in by_dataset.items():
        print(f"   Dataset '{dataset}':")
        for si in dataset_symbols:
            print(f"     - {si.name}: {si.metric} = {si.value}")
    print()

    # Example 4: Symbol collection for debugging
    print("4. Using symbol collection for debugging:")

    # Simulate a failure
    failed_symbol = provider.metric(Average("invalid_column"), dataset="orders")
    evaluator._metrics = {
        avg_price: Success(150.5),
        sum_quantity: Success(1000),
        avg_rating: Success(4.5),
        failed_symbol: Success(0.0),  # This would normally be a Failure
    }

    debug_expr = avg_price + failed_symbol
    debug_symbols = evaluator.collect_symbols(debug_expr)

    print("   Symbols in failing expression:")
    for si in debug_symbols:
        status = "✓" if isinstance(si.value, Success) else "✗"
        print(f"   {status} {si.name}: {si.metric} = {si.value}")


if __name__ == "__main__":
    main()
