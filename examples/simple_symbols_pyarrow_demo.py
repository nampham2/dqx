#!/usr/bin/env python3
"""
Simple demo of symbols_to_pyarrow_table function.

This example creates SymbolInfo objects directly and shows various
PyArrow operations on them.
"""

import datetime as dt

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from returns.result import Failure, Success

from dqx.data import symbols_to_pyarrow_table
from dqx.display import print_symbols
from dqx.provider import SymbolInfo


def main() -> None:
    """Run simple symbols to PyArrow table demo."""
    print("=== Simple Symbols to PyArrow Table Demo ===\n")

    # Create sample symbols representing different scenarios
    symbols = [
        # Successful metrics
        SymbolInfo(
            name="sales_completeness",
            metric="completeness(amount)",
            dataset="sales",
            value=Success(0.92),
            yyyy_mm_dd=dt.date(2024, 3, 15),
            tags={"env": "prod", "region": "us-west"},
        ),
        SymbolInfo(
            name="avg_sale_amount",
            metric="average(amount)",
            dataset="sales",
            value=Success(525.75),
            yyyy_mm_dd=dt.date(2024, 3, 15),
            tags={"env": "prod", "region": "us-west"},
        ),
        SymbolInfo(
            name="null_sales_count",
            metric="null_count(amount)",
            dataset="sales",
            value=Success(10.0),
            yyyy_mm_dd=dt.date(2024, 3, 15),
            tags={"env": "prod", "region": "us-west"},
        ),
        # Failed metric
        SymbolInfo(
            name="max_sale_amount",
            metric="maximum(amount)",
            dataset="sales_archive",
            value=Failure("Table not found: sales_archive"),
            yyyy_mm_dd=dt.date(2024, 3, 15),
            tags={"env": "prod", "region": "us-west"},
        ),
        # Different date
        SymbolInfo(
            name="user_count",
            metric="num_rows()",
            dataset="users",
            value=Success(15420.0),
            yyyy_mm_dd=dt.date(2024, 3, 14),
            tags={"env": "staging"},
        ),
    ]

    # Display using print_symbols
    print("1. Display using print_symbols (which uses PyArrow internally):")
    print_symbols(symbols)

    # Convert to PyArrow table for analysis
    print("\n2. Convert to PyArrow table for analysis:")
    table = symbols_to_pyarrow_table(symbols)
    print(f"Table has {table.num_rows} rows and {table.num_columns} columns")
    print(f"Schema: {table.schema}")

    # Filter and analyze
    print("\n3. Filter successful metrics only:")
    success_mask = pc.is_valid(table["Value"])
    success_table = table.filter(success_mask)

    print(f"Found {success_table.num_rows} successful metrics:")
    for i in range(success_table.num_rows):
        print(f"  - {success_table['Symbol'][i].as_py()}: {success_table['Value'][i].as_py()}")

    # Aggregate by date
    print("\n4. Group by date:")
    date_groups = table.group_by(["Date"]).aggregate([("Symbol", "count_distinct"), ("Value", "mean")])

    for i in range(date_groups.num_rows):
        date = date_groups["Date"][i].as_py()
        count = date_groups["Symbol_count_distinct"][i].as_py()
        avg = date_groups["Value_mean"][i].as_py()
        print(f"  {date}: {count} metrics, avg value: {avg:.2f}" if avg else f"  {date}: {count} metrics")

    # Export to Parquet
    print("\n5. Export to Parquet:")
    parquet_path = ".tmp/simple_symbols.parquet"
    pq.write_table(table, parquet_path)
    print(f"Exported to: {parquet_path}")

    # Query with PyArrow compute
    print("\n6. Advanced PyArrow queries:")

    # Find metrics with values above threshold
    threshold = 100
    high_value_mask = pc.and_(pc.is_valid(table["Value"]), pc.greater(table["Value"], threshold))
    high_value_table = table.filter(high_value_mask)

    print(f"\nMetrics with values > {threshold}:")
    for i in range(high_value_table.num_rows):
        symbol = high_value_table["Symbol"][i].as_py()
        value = high_value_table["Value"][i].as_py()
        metric = high_value_table["Metric"][i].as_py()
        print(f"  {symbol} ({metric}): {value}")

    # Export to CSV for Excel/Google Sheets
    print("\n7. Export to CSV:")
    import pyarrow.csv as csv

    csv_path = ".tmp/simple_symbols.csv"
    csv.write_csv(table, csv_path)
    print(f"Exported to: {csv_path}")
    print("You can open this file in Excel or Google Sheets for further analysis.")

    # Show PyArrow's ability to handle large datasets
    print("\n8. Performance demonstration:")
    # Create many symbols
    large_symbols = []
    for i in range(1000):
        large_symbols.append(
            SymbolInfo(
                name=f"metric_{i}",
                metric=f"average(col_{i % 10})",
                dataset=f"dataset_{i % 5}",
                value=Success(float(i * 10.5)) if i % 20 != 0 else Failure(f"Error {i}"),
                yyyy_mm_dd=dt.date(2024, 3, 15) - dt.timedelta(days=i % 30),
                tags={"batch": str(i // 100)},
            )
        )

    large_table = symbols_to_pyarrow_table(large_symbols)
    print(f"Created table with {large_table.num_rows:,} rows")

    # Fast aggregations
    stats = {
        "Total rows": large_table.num_rows,
        "Successful": pc.sum(pc.cast(pc.is_valid(large_table["Value"]), pa.int64())).as_py(),
        "Failed": pc.sum(pc.cast(pc.is_valid(large_table["Error"]), pa.int64())).as_py(),
        "Unique dates": pc.count_distinct(large_table["Date"]).as_py(),
        "Unique datasets": pc.count_distinct(large_table["Dataset"]).as_py(),
    }

    print("\nStatistics (computed efficiently with PyArrow):")
    for key, value in stats.items():
        print(f"  {key}: {value:,}")


if __name__ == "__main__":
    main()
