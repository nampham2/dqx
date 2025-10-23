#!/usr/bin/env python3
"""Demonstrate BigQuery dialect usage in DQX."""

from datetime import date

from dqx.dialect import get_dialect
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
    SqlOp,
    Sum,
    Variance,
)

print("BigQuery Dialect Demonstration")
print("=" * 50)

# Get the BigQuery dialect
bigquery = get_dialect("bigquery")

print("=== Basic BigQuery SQL Translations ===\n")

# Demonstrate basic SQL translations
ops_basic: list[SqlOp] = [
    NumRows(),
    Average("price"),
    Sum("quantity"),
    Minimum("price"),
    Maximum("price"),
    Variance("sales"),
]

for op in ops_basic:
    sql = bigquery.translate_sql_op(op)
    print(f"{op}:")
    print(f"  {sql}\n")

print("\n=== Advanced BigQuery SQL Translations ===\n")

# Demonstrate advanced translations
ops_advanced: list[SqlOp] = [
    First("created_at"),
    NullCount("email"),
    NegativeCount("profit"),
    DuplicateCount(["user_id"]),
    DuplicateCount(["user_id", "product_id"]),
]

for op in ops_advanced:
    sql = bigquery.translate_sql_op(op)
    print(f"{op}:")
    print(f"  {sql}\n")

print("\n=== CTE Query Building ===\n")

# Build a CTE query
cte_sql = "SELECT * FROM `project.dataset.sales` WHERE date = '2024-01-01'"
expressions = [
    bigquery.translate_sql_op(NumRows()),
    bigquery.translate_sql_op(Average("price")),
    bigquery.translate_sql_op(Sum("quantity")),
]

query = bigquery.build_cte_query(cte_sql, expressions)
print(f"Generated CTE query:\n{query}\n")

print("\n=== Batch Query with STRUCT ===\n")

# Create batch CTE data
batch_data = [
    BatchCTEData(
        key=ResultKey(yyyy_mm_dd=date(2024, 1, 1), tags={}),
        cte_sql="SELECT * FROM `project.dataset.sales` WHERE date = '2024-01-01'",
        ops=[NumRows(), Average("revenue"), NullCount("customer_id")],
    ),
    BatchCTEData(
        key=ResultKey(yyyy_mm_dd=date(2024, 1, 2), tags={}),
        cte_sql="SELECT * FROM `project.dataset.sales` WHERE date = '2024-01-02'",
        ops=[Minimum("price"), Maximum("price"), DuplicateCount(["order_id", "product_id"])],
    ),
]

batch_query = bigquery.build_batch_cte_query(batch_data)
print(f"Generated batch query with STRUCT:\n{batch_query}")

print("\n" + "=" * 50 + "\n")
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

print("\n\n=== Dialect Registry ===\n")

# Show that we can retrieve the dialect by name
dialect = get_dialect("bigquery")
print(f"Retrieved dialect: {dialect.name}")
print(f"Dialect type: {type(dialect).__name__}")

# Verify it translates correctly
test_op = Average("revenue")
sql = dialect.translate_sql_op(test_op)
print("\nTranslation example:")
print(f"  {test_op}: {sql}")

print("\n\nKey differences from DuckDB dialect:")
print("1. Uses FLOAT64 instead of DOUBLE for type casting")
print("2. Uses backticks (`) for column aliases instead of single quotes")
print("3. Uses COUNTIF instead of COUNT_IF")
print("4. Uses VAR_SAMP instead of VARIANCE")
print("5. Uses MIN instead of FIRST for deterministic results")
print("6. Uses STRUCT instead of MAP for batch queries")
print("7. Properly formats BigQuery table references (project.dataset.table)")
