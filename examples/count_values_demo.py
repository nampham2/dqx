#!/usr/bin/env python3
"""Demo script showcasing the CountValues functionality in DQX.

This example demonstrates how to use the CountValues operation to count
occurrences of specific values in columns, including:
- Single value counting (string and integer)
- Multiple value counting
- Integration with VerificationSuite and assertions
- Handling special characters and edge cases
"""

import datetime

import pyarrow as pa
from rich.console import Console
from rich.table import Table

from dqx import specs
from dqx.api import Context, VerificationSuite, check
from dqx.common import ResultKey
from dqx.datasource import DuckRelationDataSource
from dqx.orm.repositories import InMemoryMetricDB
from dqx.provider import MetricProvider

console = Console()


def print_section(title: str) -> None:
    """Print a formatted section header."""
    console.print(f"\n[bold blue]{'=' * 60}[/bold blue]")
    console.print(f"[bold green]{title}[/bold green]")
    console.print(f"[bold blue]{'=' * 60}[/bold blue]\n")


def main() -> None:
    """Run the CountValues demo."""
    # Create sample dataset with various scenarios
    data = pa.table(
        {
            "status": ["active", "inactive", "active", "pending", "active", "inactive", None, "ACTIVE"],
            "priority": [1, 2, 1, 3, 1, 2, 4, 1],
            "region": ["US", "EU", "US", "APAC", "US", "EU", "Other", "US"],
            "category": ["A", "B", "A", "C", "A", "B", "D", "A"],
            "name": ["Smith", "O'Brien", "Jones", "Smith", "Lee", "O'Brien", "Chen", "Smith"],
            "score": [100, 200, 100, 300, 100, 200, None, 100],
        }
    )

    print_section("Sample Data")

    # Display the data in a nice table
    table = Table(title="Test Dataset")
    for column in data.schema.names:
        table.add_column(column, style="cyan")

    for i in range(len(data)):
        row = [str(data[column][i].as_py()) for column in data.schema.names]
        table.add_row(*row)

    console.print(table)

    # Create datasource
    ds = DuckRelationDataSource.from_arrow(data)

    # Set up metric database and key
    db = InMemoryMetricDB()
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={"demo": "count_values"})

    print_section("Example 1: Basic Single Value Counting")

    @check(name="Single Value Counts")
    def single_value_check(mp: MetricProvider, ctx: Context) -> None:
        # Count string values
        ctx.assert_that(mp.count_values("status", "active")).where(
            name="Count of 'active' status (case-sensitive)"
        ).is_eq(3)

        # Count integer values
        ctx.assert_that(mp.count_values("priority", 1)).where(name="Count of priority 1").is_eq(4)

        # Special characters in values
        ctx.assert_that(mp.count_values("name", "O'Brien")).where(name="Count of O'Brien (with apostrophe)").is_eq(2)

    suite1 = VerificationSuite([single_value_check], db, "Single Value Suite")
    console.print("[yellow]Running single value checks...[/yellow]")
    suite1.run({"demo": ds}, key)

    print_section("Example 2: Multiple Value Counting")

    @check(name="Multiple Value Counts")
    def multiple_value_check(mp: MetricProvider, ctx: Context) -> None:
        # Count multiple string values
        ctx.assert_that(mp.count_values("region", ["US", "EU"])).where(name="Count of US or EU regions").is_eq(5)

        # Count multiple integer values
        ctx.assert_that(mp.count_values("priority", [1, 2, 3])).where(name="Count of priority levels 1-3").is_eq(7)

        # Multiple categories
        ctx.assert_that(mp.count_values("category", ["A", "B", "C"])).where(name="Count of main categories").is_eq(6)

    suite2 = VerificationSuite([multiple_value_check], db, "Multiple Value Suite")
    console.print("[yellow]Running multiple value checks...[/yellow]")
    suite2.run({"demo": ds}, key)

    print_section("Example 3: Using CountValues Spec Directly")

    @check(name="Direct Spec Usage")
    def spec_check(mp: MetricProvider, ctx: Context) -> None:
        # Create specs directly
        active_count_spec = specs.CountValues("status", "active")
        high_score_spec = specs.CountValues("score", [100, 200])

        ctx.assert_that(mp.metric(active_count_spec)).where(name="Using CountValues spec for active status").is_eq(3)

        ctx.assert_that(mp.metric(high_score_spec)).where(name="Count of scores 100 or 200").is_eq(
            6
        )  # Note: nulls are ignored

    suite3 = VerificationSuite([spec_check], db, "Direct Spec Suite")
    console.print("[yellow]Running direct spec checks...[/yellow]")
    suite3.run({"demo": ds}, key)

    print_section("Example 4: Edge Cases and Validation")

    @check(name="Edge Case Demo")
    def edge_case_check(mp: MetricProvider, ctx: Context) -> None:
        # Case sensitivity matters
        ctx.assert_that(mp.count_values("status", "ACTIVE")).where(
            name="Count of 'ACTIVE' (uppercase) - case sensitive"
        ).is_eq(1)

        # Nulls are not counted
        ctx.assert_that(mp.count_values("score", 100)).where(name="Count of score 100 (nulls ignored)").is_eq(4)

        # Empty result
        ctx.assert_that(mp.count_values("region", "Antarctica")).where(name="Count of non-existent value").is_eq(0)

    suite4 = VerificationSuite([edge_case_check], db, "Edge Case Suite")
    console.print("[yellow]Running edge case checks...[/yellow]")
    suite4.run({"demo": ds}, key)

    print_section("Example 5: Complex Business Rules")

    @check(name="Business Rule Validation")
    def business_rules_check(mp: MetricProvider, ctx: Context) -> None:
        # Ensure majority of items are from main regions
        ctx.assert_that(mp.count_values("region", ["US", "EU", "APAC"])).where(
            name="Main regions should be at least 75% of total"
        ).is_geq(6)  # 6 out of 8 records

        # Check for data quality - too many pending items
        ctx.assert_that(mp.count_values("status", "pending")).where(
            name="Pending items should be less than 25%"
        ).is_leq(2)

        # Validate priority distribution
        ctx.assert_that(mp.count_values("priority", [1, 2])).where(
            name="High priority items (1 or 2) should be majority"
        ).is_gt(4)

    suite5 = VerificationSuite([business_rules_check], db, "Business Rules Suite")
    console.print("[yellow]Running business rule checks...[/yellow]")
    suite5.run({"demo": ds}, key)

    print_section("Summary")
    console.print("""
CountValues is a powerful operation for validating categorical data:

✓ Count occurrences of single values (strings or integers)
✓ Count occurrences of multiple values using lists
✓ Handle special characters in values properly
✓ Ignore null values in counts
✓ Case-sensitive matching
✓ Integration with assertion framework for data quality rules

Use cases:
- Validate distribution of categorical values
- Ensure required values are present
- Check for data quality issues
- Monitor changes in categorical distributions
- Validate business rules about categorical data
""")


if __name__ == "__main__":
    main()
