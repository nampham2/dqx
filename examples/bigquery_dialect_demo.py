#!/usr/bin/env python3
"""
BigQuery dialect demonstration for DQX.

This example shows how to use the BigQuery dialect to generate
SQL compatible with Google BigQuery's syntax.
"""

from datetime import date
from typing import Any

from dqx.dialect import BigQueryDialect, get_dialect
from dqx.models import BatchCTEData, ResultKey
from dqx.ops import (
    Average,
    DuplicateCount,
    First,
    Maximum,
    Minimum,
    NegativeCount,
    NullCount,
    NumRows,
    Sum,
    Variance,
)


def demo_basic_translations() -> None:
    """Demonstrate basic SQL operation translations."""
    print("=== Basic BigQuery SQL Translations ===\n")

    dialect = BigQueryDialect()

    # Basic aggregations
    ops: list[Any] = [
        NumRows(),
        Average("price"),
        Sum("quantity"),
        Minimum("price"),
        Maximum("price"),
        Variance("sales"),
    ]

    for op in ops:
        sql = dialect.translate_sql_op(op)
        print(f"{op.name}:")
        print(f"  {sql}\n")


def demo_advanced_translations() -> None:
    """Demonstrate advanced SQL operation translations."""
    print("\n=== Advanced BigQuery SQL Translations ===\n")

    dialect = BigQueryDialect()

    # Advanced operations
    ops: list[Any] = [
        First("created_at"),  # Uses MIN for deterministic results
        NullCount("email"),
        NegativeCount("profit"),
        DuplicateCount(["user_id"]),
        DuplicateCount(["user_id", "product_id"]),
    ]

    for op in ops:
        sql = dialect.translate_sql_op(op)
        print(f"{op.name}:")
        print(f"  {sql}\n")


def demo_cte_query() -> None:
    """Demonstrate CTE query building."""
    print("\n=== CTE Query Building ===\n")

    dialect = BigQueryDialect()

    # Build a CTE query
    cte_sql = "SELECT * FROM `project.dataset.sales` WHERE date = '2024-01-01'"
    expressions = [
        dialect.translate_sql_op(NumRows()),
        dialect.translate_sql_op(Average("price")),
        dialect.translate_sql_op(Sum("quantity")),
    ]

    query = dialect.build_cte_query(cte_sql, expressions)
    print("Generated CTE query:")
    print(query)


def demo_batch_query() -> None:
    """Demonstrate batch query with STRUCT."""
    print("\n\n=== Batch Query with STRUCT ===\n")

    dialect = BigQueryDialect()

    # Create batch data for multiple dates
    cte_data = []

    # Day 1
    key1 = ResultKey(yyyy_mm_dd=date(2024, 1, 1), tags={"region": "US"})
    ops1: list[Any] = [NumRows(), Average("revenue"), NullCount("customer_id")]
    cte_data.append(
        BatchCTEData(key=key1, cte_sql="SELECT * FROM `project.dataset.sales` WHERE date = '2024-01-01'", ops=ops1)
    )

    # Day 2
    key2 = ResultKey(yyyy_mm_dd=date(2024, 1, 2), tags={"region": "US"})
    ops2: list[Any] = [Minimum("price"), Maximum("price"), DuplicateCount(["order_id", "product_id"])]
    cte_data.append(
        BatchCTEData(key=key2, cte_sql="SELECT * FROM `project.dataset.sales` WHERE date = '2024-01-02'", ops=ops2)
    )

    # Generate batch query
    batch_query = dialect.build_batch_cte_query(cte_data)
    print("Generated batch query with STRUCT:")
    print(batch_query)
    print("\n" + "=" * 50 + "\n")

    # Show the result structure
    print("Result structure:")
    print("- date: STRING")
    print("- values: STRUCT<")
    print("    x_1_num_rows: FLOAT64,")
    print("    x_2_average_revenue: FLOAT64,")
    print("    x_3_null_count_customer_id: FLOAT64,")
    print("    x_4_minimum_price: FLOAT64,")
    print("    x_5_maximum_price: FLOAT64,")
    print("    x_6_duplicate_count_order_id_product_id: FLOAT64")
    print("  >")


def demo_dialect_registry() -> None:
    """Demonstrate dialect registry usage."""
    print("\n\n=== Dialect Registry ===\n")

    # Get dialect from registry
    dialect = get_dialect("bigquery")
    print(f"Retrieved dialect: {dialect.name}")
    print(f"Dialect type: {type(dialect).__name__}")

    # Show it works the same
    op = Average("revenue")
    sql = dialect.translate_sql_op(op)
    print("\nTranslation example:")
    print(f"  {op.name}: {sql}")


def main() -> None:
    """Run all demonstrations."""
    print("BigQuery Dialect Demonstration")
    print("=" * 50)

    demo_basic_translations()
    demo_advanced_translations()
    demo_cte_query()
    demo_batch_query()
    demo_dialect_registry()

    print("\n\nKey differences from DuckDB dialect:")
    print("1. Uses FLOAT64 instead of DOUBLE for type casting")
    print("2. Uses backticks (`) for column aliases instead of single quotes")
    print("3. Uses COUNTIF instead of COUNT_IF")
    print("4. Uses VAR_SAMP instead of VARIANCE")
    print("5. Uses MIN instead of FIRST for deterministic results")
    print("6. Uses STRUCT instead of MAP for batch queries")
    print("7. Properly formats BigQuery table references (project.dataset.table)")


if __name__ == "__main__":
    main()
