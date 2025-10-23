#!/usr/bin/env python3
"""Demonstration of batch analysis functionality in DQX.

This example shows how to:
1. Analyze multiple dates efficiently with batch analysis
2. Compare single-date analysis vs batch analysis
3. Handle large date ranges with automatic batching
"""

import datetime
import time
from typing import Sequence

import pyarrow as pa

from dqx import specs
from dqx.analyzer import Analyzer
from dqx.common import ResultKey, SqlDataSource
from dqx.datasource import DuckRelationDataSource


def generate_sales_data(start_date: datetime.date, num_days: int) -> pa.Table:
    """Generate sample sales data for multiple days."""
    import random

    random.seed(42)  # For reproducible results

    dates = []
    revenues = []
    quantities = []
    prices = []
    categories = []

    for day_offset in range(num_days):
        date = start_date + datetime.timedelta(days=day_offset)
        # Generate 50-100 records per day
        num_records = random.randint(50, 100)

        for _ in range(num_records):
            dates.append(date.isoformat())
            revenues.append(random.uniform(100, 1000))
            quantities.append(random.randint(1, 50))
            prices.append(random.uniform(10, 200))
            categories.append(random.choice(["Electronics", "Clothing", "Food", "Books"]))

    return pa.table(
        {
            "yyyy_mm_dd": dates,
            "revenue": revenues,
            "quantity": quantities,
            "price": prices,
            "category": categories,
        }
    )


def single_date_analysis(ds: SqlDataSource, metrics: Sequence[specs.MetricSpec], dates: list[datetime.date]) -> float:
    """Analyze dates one by one (traditional approach)."""
    analyzer = Analyzer()
    start_time = time.time()

    for date in dates:
        key = ResultKey(date, {})
        # Create fresh metric instances for each date
        fresh_metrics = create_metrics(metrics)
        # Analyze one date at a time
        analyzer.analyze(ds, {key: fresh_metrics})

    elapsed = time.time() - start_time
    return elapsed


def batch_analysis(ds: SqlDataSource, metrics: Sequence[specs.MetricSpec], dates: list[datetime.date]) -> float:
    """Analyze multiple dates in batch (new efficient approach)."""
    analyzer = Analyzer()
    start_time = time.time()

    # Build metrics_by_key dictionary with fresh metric instances
    metrics_by_key = {ResultKey(date, {}): create_metrics(metrics) for date in dates}

    # Single batch call
    analyzer.analyze(ds, metrics_by_key)

    elapsed = time.time() - start_time
    return elapsed


def create_metrics(metric_list: Sequence[specs.MetricSpec]) -> list[specs.MetricSpec]:
    """Create fresh instances of metrics to avoid SQL op reuse."""
    fresh_metrics: list[specs.MetricSpec] = []
    for metric in metric_list:
        # Create fresh instance based on metric type
        if isinstance(metric, specs.Sum):
            fresh_metrics.append(specs.Sum(metric.parameters["column"]))
        elif isinstance(metric, specs.Average):
            fresh_metrics.append(specs.Average(metric.parameters["column"]))
        elif isinstance(metric, specs.Minimum):
            fresh_metrics.append(specs.Minimum(metric.parameters["column"]))
        elif isinstance(metric, specs.Maximum):
            fresh_metrics.append(specs.Maximum(metric.parameters["column"]))
        elif isinstance(metric, specs.NullCount):
            fresh_metrics.append(specs.NullCount(metric.parameters["column"]))
        elif isinstance(metric, specs.NumRows):
            fresh_metrics.append(specs.NumRows())
        else:
            # For any other metric types, just use the original
            fresh_metrics.append(metric)
    return fresh_metrics


def main() -> None:
    """Run the batch analysis demonstration."""
    print("DQX Batch Analysis Demo")
    print("=" * 50)
    print("\nNote: This demo uses DuckRelationDataSource which doesn't filter by date.")
    print("In production, use a date-aware data source that implements proper filtering.")

    # Generate sample data for 30 days
    start_date = datetime.date(2024, 1, 1)
    num_days = 30
    print(f"\nGenerating sales data for {num_days} days...")
    table = generate_sales_data(start_date, num_days)
    print(f"Generated {len(table)} total records")

    # Create data source
    ds = DuckRelationDataSource.from_arrow(table)

    # Define metrics to compute
    metrics: list[specs.MetricSpec] = [
        specs.Sum("revenue"),
        specs.Average("price"),
        specs.Minimum("quantity"),
        specs.Maximum("revenue"),
        specs.NullCount("category"),
    ]

    print(f"\nMetrics to compute: {[m.name for m in metrics]}")

    # Test with different date ranges
    test_cases = [
        ("Small (5 days)", 5),
        ("Medium (15 days)", 15),
        ("Large (30 days)", 30),
    ]

    for test_name, days in test_cases:
        print(f"\n{test_name} Test:")
        print("-" * 30)

        dates = [start_date + datetime.timedelta(days=i) for i in range(days)]

        # Run single-date analysis
        single_time = single_date_analysis(ds, metrics, dates)
        print(f"Single-date analysis: {single_time:.3f} seconds")

        # Run batch analysis
        batch_time = batch_analysis(ds, metrics, dates)
        print(f"Batch analysis: {batch_time:.3f} seconds")

        # Calculate speedup
        speedup = single_time / batch_time
        print(f"Speedup: {speedup:.1f}x faster")

    # Demonstrate automatic batching for large date ranges
    print("\n" + "=" * 50)
    print("Automatic Batching Demo (60 days)")
    print("=" * 50)

    # Generate data for 60 days (exceeds DEFAULT_BATCH_SIZE)
    large_table = generate_sales_data(start_date, 60)
    large_ds = DuckRelationDataSource.from_arrow(large_table)

    analyzer = Analyzer()
    dates_60 = [start_date + datetime.timedelta(days=i) for i in range(60)]
    # Create fresh metrics for each date
    metrics_by_key = {ResultKey(date, {}): create_metrics(metrics) for date in dates_60}

    print("\nAnalyzing 60 dates with DEFAULT_BATCH_SIZE=7...")
    print("The analyzer will automatically split this into multiple batches.")

    start_time = time.time()
    report = analyzer.analyze(large_ds, metrics_by_key)
    elapsed = time.time() - start_time

    print(f"\nCompleted in {elapsed:.3f} seconds")
    print(f"Total metrics computed: {len(report)}")
    print(f"Expected: {len(dates_60) * len(metrics)} = {len(dates_60) * len(metrics)}")

    # Show sample results
    print("\nSample Results (first 3 dates):")
    print("-" * 50)
    print("\nNote: DuckRelationDataSource doesn't filter by date, so all dates show total values.")
    print("In a real implementation, you'd use a date-filtered data source.\n")

    for i, date in enumerate(dates_60[:3]):
        print(f"Date: {date}")
        key = ResultKey(date, {})
        # Find matching metrics in report (they have different instances)
        for report_key, metric_result in report.items():
            if report_key[1] == key and report_key[0].name in [m.name for m in metrics]:
                print(f"  {report_key[0].name}: {metric_result.value:.2f}")

    # Demonstrate performance characteristics
    print("\n" + "=" * 50)
    print("Performance Summary")
    print("=" * 50)

    print("\nBatch analysis is most effective when:")
    print("- Analyzing a small number of dates (< 10)")
    print("- The overhead of multiple SQL queries exceeds batch query overhead")
    print("- You need to analyze different metrics for different dates")

    print("\nFor larger date ranges, the batch query becomes more complex,")
    print("which can offset the benefits. The implementation automatically")
    print("splits large batches to maintain optimal performance.")

    print("\n" + "=" * 50)
    print("Demo completed!")


if __name__ == "__main__":
    main()
