#!/usr/bin/env python3
"""
Demo of symbols_to_pyarrow_table function.

This example shows how to transform SymbolInfo objects from collect_symbols()
into a PyArrow table for further processing or export.
"""

import datetime as dt

import pyarrow.compute as pc
import pyarrow.parquet as pq
from returns.result import Failure, Success

from dqx.data import symbols_to_pyarrow_table
from dqx.provider import SymbolInfo


def main() -> None:
    """Run symbols to PyArrow table demo."""
    print("=== Symbols to PyArrow Table Demo ===\n")

    # Create sample symbols with mixed success/failure cases
    symbols = [
        SymbolInfo(
            name="x_1",
            metric="average(price)",
            dataset="sales",
            value=Success(125.50),
            yyyy_mm_dd=dt.date(2024, 1, 26),
            tags={"env": "prod", "region": "us-east"},
        ),
        SymbolInfo(
            name="x_2",
            metric="sum(quantity)",
            dataset="inventory",
            value=Success(10500.0),
            yyyy_mm_dd=dt.date(2024, 1, 26),
            tags={"env": "prod"},
        ),
        SymbolInfo(
            name="x_3",
            metric="null_count(user_id)",
            dataset=None,  # No dataset
            value=Failure("Table not found: users"),
            yyyy_mm_dd=dt.date(2024, 1, 26),
            tags={},
        ),
        SymbolInfo(
            name="x_4",
            metric="completeness(email)",
            dataset="customers",
            value=Success(0.952),
            yyyy_mm_dd=dt.date(2024, 1, 25),
            tags={"env": "staging"},
        ),
        SymbolInfo(
            name="x_10",  # Test numeric ordering
            metric="num_rows()",
            dataset="orders",
            value=Success(42000.0),
            yyyy_mm_dd=dt.date(2024, 1, 25),
            tags={},
        ),
    ]

    # Convert to PyArrow table
    table = symbols_to_pyarrow_table(symbols)

    # Display schema
    print("Table Schema:")
    print(table.schema)
    print()

    # Display table contents
    print("Table Contents:")
    print(f"Number of rows: {table.num_rows}")
    print("\nSample data (first 5 rows):")
    for i in range(min(5, table.num_rows)):
        row_data = []
        for col_name in table.schema.names:
            value = table[col_name][i].as_py()
            row_data.append(f"{col_name}: {value}")
        print(f"Row {i}: {', '.join(row_data)}")
    print()

    # Demonstrate filtering capabilities
    print("=== Filtering Examples ===")

    # Filter successful values only
    success_mask = pc.is_valid(table["Value"])
    success_table = table.filter(success_mask)
    print(f"\nSuccessful metrics ({success_table.num_rows} rows):")
    for i in range(success_table.num_rows):
        print(
            f"  {success_table['Symbol'][i].as_py()}: {success_table['Metric'][i].as_py()} = {success_table['Value'][i].as_py()}"
        )

    # Filter failures only
    failure_mask = pc.is_valid(table["Error"])
    failure_table = table.filter(failure_mask)
    print(f"\nFailed metrics ({failure_table.num_rows} rows):")
    for i in range(failure_table.num_rows):
        print(
            f"  {failure_table['Symbol'][i].as_py()}: {failure_table['Metric'][i].as_py()} - {failure_table['Error'][i].as_py()}"
        )

    # Filter by date
    date_2024_01_26 = dt.date(2024, 1, 26)
    date_mask = pc.equal(table["Date"], date_2024_01_26)
    date_table = table.filter(date_mask)
    print(f"\nMetrics from {date_2024_01_26} ({date_table.num_rows} rows):")
    for i in range(date_table.num_rows):
        value = date_table["Value"][i].as_py()
        error = date_table["Error"][i].as_py()
        result = f"{value}" if value is not None else f"Error: {error}"
        print(f"  {date_table['Symbol'][i].as_py()}: {date_table['Metric'][i].as_py()} = {result}")

    # Demonstrate aggregations
    print("\n=== Aggregation Examples ===")

    # Count by dataset
    print("\nMetrics count by dataset:")
    dataset_counts = table.group_by(["Dataset"]).aggregate([("Symbol", "count")])
    for i in range(dataset_counts.num_rows):
        dataset = dataset_counts["Dataset"][i].as_py()
        count = dataset_counts["Symbol_count"][i].as_py()
        print(f"  {dataset}: {count} metrics")

    # Average of successful values
    valid_values = success_table["Value"]
    avg_value = pc.mean(valid_values)
    print(f"\nAverage of successful values: {avg_value.as_py():.2f}")

    # Export examples
    print("\n=== Export Examples ===")

    # Export to Parquet
    parquet_path = ".tmp/symbols_demo.parquet"
    pq.write_table(table, parquet_path)
    print(f"Exported to Parquet: {parquet_path}")

    # Export to CSV
    import pyarrow.csv as csv

    csv_path = ".tmp/symbols_demo.csv"
    csv.write_csv(table, csv_path)
    print(f"Exported to CSV: {csv_path}")

    # Show how to read back
    print("\nReading back from Parquet:")
    read_table = pq.read_table(parquet_path)
    print(f"Read {read_table.num_rows} rows with schema: {list(read_table.schema.names)}")

    # Demonstrate integration with display function
    print("\n=== Integration with Display Function ===")
    from dqx.display import print_symbols

    print("\nUsing print_symbols (which now uses symbols_to_pyarrow_table internally):")
    print_symbols(symbols)


if __name__ == "__main__":
    main()
