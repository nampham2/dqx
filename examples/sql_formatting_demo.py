#!/usr/bin/env python3
"""Demo script to showcase SQL formatting in DQX analyzer."""

import datetime
import logging

import pyarrow as pa
from rich.console import Console

from dqx.analyzer import Analyzer
from dqx.common import ResultKey
from dqx.extensions.duckds import DuckRelationDataSource
from dqx.specs import Average, DuplicateCount, Maximum, MetricSpec, Minimum, NullCount, NumRows, Sum

# Enable debug logging to see SQL queries
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console = Console()


def main() -> None:
    """Demonstrate SQL formatting with various metrics."""
    console.print("\n[bold blue]DQX SQL Formatting Demo[/bold blue]\n")
    console.print("[dim]This demo shows how SQL queries are automatically formatted in DQX[/dim]\n")

    # Create sample data
    table = pa.table(
        {
            "product_id": [1, 2, 3, 4, 5, 1, 2, 3, None, 5],
            "price": [10.5, 20.0, 15.5, None, 30.0, 10.5, 20.0, 16.0, 25.0, 30.0],
            "quantity": [100, 200, 150, 300, None, 100, 200, 150, 300, 50],
            "category": ["A", "B", "A", "B", "C", "A", "B", "A", "B", "C"],
            "discount": [0.1, 0.2, None, 0.15, 0.0, 0.1, 0.2, 0.05, 0.15, 0.0],
        }
    )

    # Create data source
    ds = DuckRelationDataSource.from_arrow(table)

    # Define metrics to analyze
    metrics: list[MetricSpec] = [
        NumRows(),
        Average("price"),
        Sum("quantity"),
        Minimum("price"),
        Maximum("price"),
        NullCount("product_id"),
        NullCount("discount"),
    ]

    console.print("[yellow]Running analyzer with multiple metrics...[/yellow]")
    console.print("[dim]The SQL query will be formatted and shown in the debug logs below[/dim]\n")

    # Create analyzer and run analysis
    analyzer = Analyzer()
    key = ResultKey(yyyy_mm_dd=datetime.date(2024, 10, 17), tags={"demo": "sql_formatting"})

    # The SQL query will be logged with debug level
    report = analyzer.analyze(ds, metrics, key)

    console.print("\n[green]Analysis Results:[/green]")
    for (metric, _), result in report.items():
        console.print(f"  {metric.name}: {result.value}")

    # Demonstrate with duplicate count for complex SQL
    console.print("\n[yellow]Running analyzer with DuplicateCount metric...[/yellow]")
    console.print("[dim]This generates a more complex SQL query with multiple columns[/dim]\n")

    dup_metrics: list[MetricSpec] = [
        DuplicateCount(["product_id", "category"]),
        DuplicateCount(["price", "quantity", "discount"]),
    ]

    # Run another analysis with duplicate count
    analyzer2 = Analyzer()
    key2 = ResultKey(yyyy_mm_dd=datetime.date(2024, 10, 17), tags={"demo": "duplicate_count"})
    report2 = analyzer2.analyze(ds, dup_metrics, key2)

    console.print("\n[green]Duplicate Count Results:[/green]")
    for (metric, _), result in report2.items():
        console.print(f"  {metric.name}: {result.value}")

    # The formatted SQL will be visible in the debug logs
    console.print("\n[bold green]✓ Demo complete![/bold green]")
    console.print("\n[dim]Review the debug logs above to see the formatted SQL queries.[/dim]")
    console.print("[dim]Notice how the SQL is properly indented and formatted for readability.[/dim]\n")


if __name__ == "__main__":
    main()
