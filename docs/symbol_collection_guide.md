# Symbol Collection Guide

## Overview

The symbol collection feature allows you to retrieve all computed metrics (symbols) after running a VerificationSuite. This is useful for:
- Debugging metric calculations
- Exporting metric values for reporting
- Understanding what metrics were computed during validation
- Creating custom reports or visualizations

## Basic Usage

```python
from dqx.api import VerificationSuite, check
from dqx.common import ResultKey
from returns.result import Success, Failure
import datetime


# Define your checks
@check(name="Revenue Checks", datasets=["sales"])
def revenue_checks(mp, ctx):
    ctx.assert_that(mp.sum("amount", dataset="sales")).where(
        name="Total revenue positive"
    ).is_positive()

    ctx.assert_that(mp.average("amount", dataset="sales")).where(
        name="Average transaction value"
    ).is_gt(50)


# Create and run suite
suite = VerificationSuite([revenue_checks], db, "Daily Revenue Suite")
key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={"env": "prod"})
suite.run({"sales": sales_datasource}, key)

# Collect all symbols (automatically sorted by name)
symbols = suite.collect_symbols()

# Process results
for symbol in symbols:
    if isinstance(symbol.value, Success):
        print(f"{symbol.metric}: {symbol.value.unwrap():.2f}")
    else:
        print(f"{symbol.metric}: FAILED - {symbol.value.failure()}")
```

## API Reference

### VerificationSuite.collect_symbols()

Collects all symbols registered during suite setup and returns them with computed values.

**Returns**: `list[SymbolInfo]` - List sorted by symbol name (x_1, x_2, etc.)

**Raises**: `DQXError` if called before `run()`

**Note**: Symbols are automatically sorted by name for consistent ordering across runs.

### SymbolInfo

A dataclass containing complete information about a computed metric:

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Symbol identifier (e.g., "x_1", "x_2") |
| `metric` | `str` | Human-readable metric description (e.g., "average(price)") |
| `dataset` | `str` | Dataset name the metric was computed from |
| `value` | `Result[float, str]` | Computation result - Success(float) or Failure(error) |
| `yyyy_mm_dd` | `datetime.date` | Date when the metric was evaluated |
| `suite` | `str` | Name of the verification suite |
| `tags` | `dict[str, Any]` | Metadata from ResultKey (defaults to empty dict) |

### MetricProvider.get_symbol()

Enhanced to accept string symbol names for easier debugging:

```python
# Get by Symbol object (existing)
symbol = mp.average("price")
metric = mp.get_symbol(symbol)

# Get by string name (new)
metric = mp.get_symbol("x_1")
```

**String format**: Must be `"x_N"` where N is a positive integer.

## Practical Examples

### Filtering and Grouping

```python
from returns.result import Success

# Get only successful computations
successful = [s for s in symbols if isinstance(s.value, Success)]

# Get symbols for specific dataset
orders_symbols = [s for s in symbols if s.dataset == "orders"]

# Get specific metric types
averages = [s for s in symbols if "average" in s.metric.lower()]

# Group by dataset
from collections import defaultdict

by_dataset = defaultdict(list)
for sym in symbols:
    by_dataset[sym.dataset].append(sym)

for dataset, dataset_symbols in sorted(by_dataset.items()):
    print(f"{dataset}: {len(dataset_symbols)} metrics")
```

### Export to DataFrame

```python
import pandas as pd
from returns.result import Success, Failure

# Convert to DataFrame
df = pd.DataFrame(
    [
        {
            "symbol": s.name,
            "metric": s.metric,
            "dataset": s.dataset,
            "value": s.value.unwrap() if isinstance(s.value, Success) else None,
            "error": str(s.value.failure()) if isinstance(s.value, Failure) else None,
            "date": s.yyyy_mm_dd,
            "suite": s.suite,
            **s.tags,  # Expand tags as columns
        }
        for s in symbols
    ]
)

# Export to CSV
df.to_csv("metrics_report.csv", index=False)

# Filter failed metrics
failed_df = df[df["error"].notna()]
print(f"Found {len(failed_df)} failed metrics")
```

### Create Summary Report

```python
from returns.result import Success


def create_summary_report(symbols: list[SymbolInfo]) -> dict:
    """Create a summary report from collected symbols."""
    total = len(symbols)
    successful = sum(1 for s in symbols if isinstance(s.value, Success))
    failed = total - successful

    # Group by dataset
    by_dataset = {}
    for s in symbols:
        if s.dataset not in by_dataset:
            by_dataset[s.dataset] = {"total": 0, "successful": 0}
        by_dataset[s.dataset]["total"] += 1
        if isinstance(s.value, Success):
            by_dataset[s.dataset]["successful"] += 1

    return {
        "total_symbols": total,
        "successful": successful,
        "failed": failed,
        "success_rate": successful / total if total > 0 else 0,
        "by_dataset": by_dataset,
        "evaluation_date": symbols[0].yyyy_mm_dd if symbols else None,
        "suite": symbols[0].suite if symbols else None,
    }


# Usage
report = create_summary_report(symbols)
print(f"Success rate: {report['success_rate']:.1%}")
```

### Integration with Monitoring

```python
from returns.result import Success


def send_metrics_to_monitoring(symbols: list[SymbolInfo], client):
    """Send computed metrics to monitoring system."""
    for symbol in symbols:
        if isinstance(symbol.value, Success):
            # Send metric value
            client.send_metric(
                name=f"dqx.{symbol.suite}.{symbol.metric}",
                value=symbol.value.unwrap(),
                tags={"dataset": symbol.dataset, "symbol": symbol.name, **symbol.tags},
                timestamp=symbol.yyyy_mm_dd,
            )
        else:
            # Send failure event
            client.send_event(
                title=f"DQX Metric Failure: {symbol.metric}",
                text=str(symbol.value.failure()),
                tags={"dataset": symbol.dataset, "suite": symbol.suite, **symbol.tags},
                alert_type="error",
            )
```

## Best Practices

1. **Always check evaluation status**: Ensure the suite has been run before collecting symbols
2. **Handle failures gracefully**: Check `isinstance(value, Success)` before unwrapping values
3. **Use consistent naming**: The sorted output ensures predictable symbol ordering
4. **Export for analysis**: Convert symbols to DataFrames or CSV for further analysis
5. **Monitor trends**: Track metric values over time by storing symbols with their evaluation dates

## Troubleshooting

### Common Issues

**DQXError: Cannot collect symbols before suite execution**
- Ensure you've called `suite.run()` before `collect_symbols()`

**Invalid symbol string format**
- Symbol strings must match pattern `x_N` where N is a positive integer
- Examples: `x_1` ✓, `x_42` ✓, `x_abc` ✗, `y_1` ✗

**Empty symbol list**
- Check that your checks actually register metrics (not just assertions)
- Metrics are created by calling MetricProvider methods (e.g., `mp.average()`)

## Summary

The symbol collection feature provides comprehensive access to all computed metrics after suite execution. With automatic sorting, enhanced string support, and complete context information, it enables powerful debugging, reporting, and monitoring capabilities for your data quality pipelines.
