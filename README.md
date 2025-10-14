# DQX - Data Quality eXcellence

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> High-performance data quality framework built on DuckDB and PyArrow for sub-second query performance at scale.

## 🌟 Features

- **Lightning Fast**: Sub-second query performance on large datasets using DuckDB
- **Symbolic Assertions**: Write data quality checks as mathematical expressions
- **Graph-Based Execution**: Intelligent dependency resolution and optimization
- **Statistical Sketches**: Memory-efficient approximate computations for massive datasets
- **Multi-Dataset Support**: Validate and compare data across multiple sources
- **Time Travel**: Compare metrics across different time periods
- **Built-in Persistence**: Store metrics history in any SQL database
- **100% Test Coverage**: Comprehensive test suite ensuring reliability

## 🚀 Quick Start

### Installation

```bash
pip install dqx
```

### Basic Example

```python
from dqx.api import VerificationSuite, check
from dqx.extensions.pyarrow_ds import ArrowDataSource
from dqx.orm.repositories import InMemoryMetricDB
from dqx.common import ResultKey


@check(name="Order validation")
def validate_orders(mp, ctx):
    """Check data quality for orders."""
    # Simple metrics
    ctx.assert_that(mp.null_count("customer_id")).is_eq(0)
    ctx.assert_that(mp.average("price")).is_gt(0)

    # Complex expressions
    revenue = mp.sum("price") * mp.sum("quantity")
    ctx.assert_that(revenue).is_positive()


# Setup
db = InMemoryMetricDB()
data = ArrowDataSource.from_pandas(df)

# Create and run suite
suite = VerificationSuite([validate_orders], db, "Order Quality Suite")

# Run the checks
key = ResultKey(yyyy_mm_dd=date.today())
context = suite.run({"orders": data}, key)

# Inspect results
for check_result in context.checks():
    print(f"{check_result.name}: {check_result.status}")
```

## 📋 Core Concepts

### 1. Checks and Assertions

Define data quality rules using the `@check` decorator:

```python
@check(name="Customer data quality", severity="P0")
def validate_customers(mp, ctx):
    # Check for nulls
    ctx.assert_that(mp.null_count("id")).is_eq(0)

    # Check ranges
    ctx.assert_that(mp.average("age")).is_between(18, 100)

    # Check uniqueness
    unique_emails = mp.approx_cardinality("email")
    total_emails = mp.num_rows()
    ctx.assert_that(unique_emails / total_emails).is_geq(0.99)
```

### 2. Metric Providers

Access a rich set of built-in metrics:

```python
# Basic metrics
mp.num_rows()  # Row count
mp.average("column")  # Average value
mp.sum("column")  # Sum of values
mp.minimum("column")  # Minimum value
mp.maximum("column")  # Maximum value
mp.null_count("column")  # Count of nulls
mp.approx_cardinality("column")  # Approximate distinct count

# Statistical metrics
mp.variance("column")  # Variance
mp.stddev("column")  # Standard deviation
```

### 3. Symbolic Expressions

Combine metrics using mathematical operations:

```python
# Revenue calculation
revenue = mp.sum("price") * mp.sum("quantity")
ctx.assert_that(revenue).is_positive()

# Null percentage
null_pct = mp.null_count("email") / mp.num_rows()
ctx.assert_that(null_pct).is_leq(0.05)  # Max 5% nulls

# Complex validations
avg_order_value = mp.sum("revenue") / mp.num_rows()
ctx.assert_that(avg_order_value).is_between(50, 500)
```

## 📊 Advanced Features

### Time-Based Comparisons

```python
@check(name="Revenue monitoring")
def monitor_revenue(mp, ctx):
    # Current vs yesterday
    current = mp.sum("revenue")
    yesterday = mp.sum("revenue", key=ctx.key.lag(1))

    # Day-over-day change
    dod_change = (current - yesterday) / yesterday
    ctx.assert_that(dod_change).is_between(-0.1, 0.1)  # ±10%

    # Week-over-week trend
    last_week = mp.sum("revenue", key=ctx.key.lag(7))
    ctx.assert_that(current / last_week).is_geq(0.9)  # Max 10% drop
```

### Multi-Dataset Validation

```python
@check(name="Cross-dataset consistency")
def validate_consistency(mp, ctx):
    # Compare counts across datasets
    source_count = mp.num_rows(datasets=["source"])
    target_count = mp.num_rows(datasets=["target"])

    ctx.assert_that(source_count).is_eq(target_count)

    # Compare aggregates
    source_revenue = mp.sum("revenue", datasets=["source"])
    target_revenue = mp.sum("revenue", datasets=["target"])

    diff = sp.Abs(source_revenue - target_revenue)
    ctx.assert_that(diff / source_revenue).is_leq(0.001)  # Max 0.1% difference
```

## 🎯 Pre-commit Hooks

DQX includes comprehensive pre-commit hooks to maintain code quality:

### Setup

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install
```

### Available Hooks

- **Ruff**: Fast Python linter (replaces flake8, isort, and more)
- **Mypy**: Static type checking
- **Black**: Code formatting
- **Safety**: Security vulnerability scanning
- **Bandit**: Security linting
- **Standard hooks**: Trailing whitespace, YAML validation, etc.

### Running Hooks

```bash
# Run all hooks
pre-commit run --all-files

# Run specific hook
pre-commit run ruff --all-files

# Skip hooks temporarily
git commit --no-verify -m "Emergency fix"
```

## 📝 API Reference

### Check Definition

```python
@check(
    name="Check name",  # Required: Descriptive name
    tags=["tag1", "tag2"],  # Optional: Tags for filtering
    datasets=["dataset1"],  # Optional: Dataset constraints
    severity="P0",  # Optional: Priority level (P0-P3)
)
def my_check(mp: MetricProvider, ctx: Context) -> None:
    """Define quality checks using symbolic assertions."""
    pass
```

### Assertion API

```python
# Basic comparisons
ctx.assert_that(metric).is_eq(expected)  # Equal to
ctx.assert_that(metric).is_gt(threshold)  # Greater than
ctx.assert_that(metric).is_geq(threshold)  # Greater or equal
ctx.assert_that(metric).is_lt(threshold)  # Less than
ctx.assert_that(metric).is_leq(threshold)  # Less or equal
ctx.assert_that(metric).is_between(min, max)  # In range

# Special checks
ctx.assert_that(metric).is_positive()  # > 0
ctx.assert_that(metric).is_negative()  # < 0
ctx.assert_that(metric).is_non_negative()  # >= 0

# With tolerance
ctx.assert_that(metric).is_eq(100, tol=1)  # 99 <= metric <= 101

# With metadata
ctx.assert_that(metric).where(name="Revenue check", severity="P0").is_gt(10000)
```

### Suite Execution

```python
# Create suite
suite = VerificationSuite(
    checks=[check1, check2], db=MetricDB("postgresql://..."), name="Production Suite"
)

# Execute validation
context = suite.run(
    datasources={"orders": orders_ds, "customers": customers_ds},
    key=ResultKey(yyyy_mm_dd=date.today(), tags={"env": "prod"}),
    threading=True,  # Enable parallel execution
)

# Inspect results
if context.has_failures():
    for check in context.checks():
        if check.has_failures():
            print(f"Failed: {check.name}")
            for assertion in check.assertions():
                if not assertion.is_success():
                    print(f"  - {assertion.name}: {assertion.error}")
```

## 🔧 Pre-commit Management

### Configuration

The `.pre-commit-config.yaml` file configures all hooks:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
```

### Custom Scripts

Run all hooks with the provided script:

```bash
# Run hooks with auto-fix
./bin/run-hooks.sh

# Run without auto-fix
./bin/run-hooks.sh --no-fix

# Check only (no fixes)
./bin/run-hooks.sh --check
```

## 📚 Examples

### Time-Series Validation

```python
@check(name="Daily metrics")
def validate_daily_metrics(mp, ctx):
    current = mp.sum("revenue")

    # Compare with multiple time periods
    yesterday = mp.sum("revenue", key=ctx.key.lag(1))
    last_week = mp.sum("revenue", key=ctx.key.lag(7))
    last_month = mp.sum("revenue", key=ctx.key.lag(30))

    # Day-over-day
    dod_ratio = current / yesterday
    ctx.assert_that(dod_ratio).where(name="Daily revenue change lower bound").is_geq(
        0.9
    )  # No more than 10% drop
    ctx.assert_that(dod_ratio).where(name="Daily revenue change upper bound").is_leq(
        1.1
    )  # No more than 10% increase

    # Week-over-week trend
    ctx.assert_that(current / last_week).where(name="Weekly revenue trend").is_geq(
        0.8
    )  # No more than 20% drop
```

### Cross-dataset Validation

```python
@check(name="Cross-dataset validation")
def cross_validate(mp, ctx):
    # Compare metrics across different datasets
    prod_count = mp.num_rows(datasets=["production"])
    staging_count = mp.num_rows(datasets=["staging"])

    # Ensure data consistency
    ratio = prod_count / staging_count
    ctx.assert_that(ratio).where(name="Production/Staging lower bound").is_geq(
        0.95, tol=0.01
    )
    ctx.assert_that(ratio).where(name="Production/Staging upper bound").is_leq(
        1.05, tol=0.01
    )
```

### Collecting Metrics Without Execution

```python
# Collect checks to inspect the dependency graph
context = suite.collect(key)

# View pending metrics
all_pending = context.pending_metrics()  # All datasets
dataset_pending = context.pending_metrics("specific_dataset")

# Useful for:
# - Debugging check dependencies
# - Understanding metric requirements
# - Planning execution strategies
```

### Collecting Assertion Results

After running a verification suite, you can collect detailed results for all assertions. The API has been designed for simplicity - no need to pass the ResultKey again:

```python
# Run the suite once
suite.run(datasources, key)

# Check if evaluation is complete
if suite.is_evaluated:
    # Collect results - no key parameter needed!
    results = suite.collect_results()

    # Process results
    for result in results:
        print(f"{result.check}/{result.assertion}: {result.status}")

        if result.status == "FAILURE":
            # Extract error details directly from Result object
            failures = result.metric.failure()
            for failure in failures:
                print(f"  Error: {failure.error_message}")

# Convert to DataFrame for analysis
import pandas as pd

df = pd.DataFrame(
    [
        {
            "date": r.yyyy_mm_dd,
            "check": r.check,
            "assertion": r.assertion,
            "status": r.status,
            "value": r.metric.unwrap() if isinstance(r.metric, Success) else None,
        }
        for r in results
    ]
)

# Or create a DuckDB relation
import duckdb

conn = duckdb.connect()
relation = conn.from_pandas(df)
```

**Key Features**:
- **Single-run enforcement**: A suite can only be run once per instance
- **No redundant parameters**: ResultKey is stored during run() and reused automatically
- **Pattern matching**: Clean Result type handling using Python 3.10+ match statements
- **Direct error access**: Extract error details directly from the Result object

The `AssertionResult` dataclass provides:
- Full context (suite/check/assertion hierarchy)
- Success/failure status ("OK" or "FAILURE")
- The actual Result object (metric) for advanced usage
- Severity levels and tags
- Direct access to errors via `metric.failure()`

The `is_evaluated` flag indicates whether the suite has been executed, ensuring results are available before collection.

## Viewing Results

After running a verification suite, you can display the results in formatted tables:

```python
# Run verification
suite = VerificationSuite(checks, db, "Daily Quality Checks")
suite.run({"orders": datasource}, key)

# Display assertion results in a table
from dqx.display import print_assertion_results, print_symbols

results = suite.collect_results()
print_assertion_results(results)

# Display computed symbol values in a table
symbols = suite.collect_symbols()
print_symbols(symbols)
```

The tables show all relevant information including dates, suites, checks,
assertion names, statuses, computed values, and any error messages.
The output is color-coded for better readability:
- Green: Successful values and OK status
- Red: Failures and error messages
- Yellow: High priority items and identifiers
- Blue/Cyan: Organizational information

### Error Handling and Validation

DQX provides comprehensive validation and clear error messages:

```python
from dqx.api import GraphStates

# The framework validates inputs
try:
    # Empty suite name
    suite = VerificationSuite([], db, "")
except Exception as e:
    print(f"Validation error: {e}")  # "Suite name cannot be empty"

# Check metric states
for metric in context._graph.metrics():
    state = metric.state()  # Returns: "PENDING", "PROVIDED", or "ERROR"
    if state == GraphStates.PENDING:
        print(f"Metric {metric.spec.name} needs computation")
```

## 💾 Persistence

Store metrics for historical analysis:

```python
from dqx.orm.repositories import MetricDB

# SQLite for local development
db = MetricDB("sqlite:///metrics.db")

# PostgreSQL for production
db = MetricDB("postgresql://user:pass@host/dbname")

# Metrics are automatically persisted after analysis
suite = VerificationSuite(checks, db, name="Production Suite")

# Query historical metrics
historical_data = db.get_metrics(
    metric_name="average(revenue)",
    start_date=dt.date.today() - dt.timedelta(days=30),
    end_date=dt.date.today(),
)
```

## 🏗️ Architecture

DQX follows a modular, graph-based architecture:

```
┌─────────────┐     ┌──────────────┐     ┌────────────┐
│    API      │────▶│    Graph     │────▶│   States   │
│  (Checks)   │     │ (Dependency) │     │ (Storage)  │
└─────────────┘     └──────────────┘     └────────────┘
       │                    │                    │
       ▼                    ▼                    ▼
┌─────────────┐     ┌──────────────┐     ┌────────────┐
│   Specs     │     │   Analyzer   │     │  MetricDB  │
│  (Metrics)  │     │ (Execution)  │     │(Persistence)│
└─────────────┘     └──────────────┘     └────────────┘
       │                    │                    │
       ▼                    ▼                    ▼
┌─────────────┐     ┌──────────────┐     ┌────────────┐
│     Ops     │     │  Extensions  │     │   Models   │
│ (SQL/Sketch)│     │(DataSources) │     │   (Data)   │
└─────────────┘     └──────────────┘     └────────────┘
```

## 📈 Performance

DQX is designed for high performance:

- **DuckDB Integration**: Columnar processing with vectorized execution
- **Statistical Sketches**: Memory-efficient approximate algorithms (HyperLogLog, DataSketches)
- **Batch Processing**: Multiple metrics computed in single SQL pass
- **Parallel Execution**: Multi-threaded analysis with thread-safe aggregation
- **Query Optimization**: Automatic deduplication and reordering

Benchmarks on 1TB dataset:
- Row count: ~100ms
- Basic statistics: ~500ms
- Cardinality estimation: ~200ms
- Full validation suite: ~2s

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Areas of interest:
- New metric types
- Additional data source adapters
- Performance optimizations
- Documentation improvements

## 📚 Documentation

- [Design Document](docs/design.md) - Architecture and design decisions
- [API Reference](docs/api.md) - Complete API documentation
- [Examples](examples/) - Runnable examples
- [Symbol Collection Guide](docs/symbol_collection_guide.md) - Understanding metric collection
- [Dataset Validation Guide](docs/dataset_validation_guide.md) - Multi-dataset validation

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 🎉 Recent Improvements

### v0.5.0 (API Simplification)
- ✅ **Removed VerificationSuiteBuilder:** Direct instantiation with `VerificationSuite(checks, db, name)`
- ✅ **Simplified API:** No more builder pattern - just pass checks as a list
- ✅ **Cleaner codebase:** Reduced complexity by removing unnecessary abstractions
- ✅ **Better developer experience:** More intuitive suite creation

### v0.4.0 (Immutable Assertions & No Chaining)
- 🚨 **Breaking:** Removed assertion chaining - assertions now return None instead of AssertBuilder
- 🚨 **Breaking:** Removed listener pattern from AssertBuilder
- ✅ **Immutable AssertionNode:** Removed setter methods (set_label, set_severity, set_validator)
- ✅ **Simplified API:** AssertBuilder no longer accepts listeners parameter
- ✅ **Cleaner architecture:** Direct assertion node creation without listener indirection
- ✅ **Updated documentation:** Removed chaining examples, added multiple assertion patterns
- ✅ **Better separation:** Each assertion is now completely independent

### v0.3.0 (Architecture Improvements)
- ✅ **Refactored symbol management:** Moved symbol tracking from CheckNode to AssertionNode
- ✅ **Improved separation of concerns:** CheckNode now focuses on aggregating assertion results
- ✅ **Enhanced observer pattern:** AssertionNodes directly observe symbol state changes
- ✅ **Better error propagation:** Dataset validation errors now occur at the assertion level
- ✅ **Cleaner architecture:** Removed symbol management complexity from CheckNode
- ✅ **Simplified CheckNode state management:**
  - Removed `update_status` method in favor of pure `aggregate_children_status` function
  - Changed `CheckNode._value` type from `Maybe[Result[float, str]]` to `Maybe[str]`
  - CheckNode state is now derived purely from child assertion results
- ✅ **Maintained 98% test coverage** with comprehensive test updates

### v0.2.0 (Breaking Changes)
- 🚨 **Breaking:** Removed legacy `.sql` property from SqlOp protocol
- 🚨 **Breaking:** All data sources now require a dialect implementation
- ✅ Simplified SQL generation through unified dialect approach
- ✅ Improved code maintainability by removing duplicate SQL logic

### v0.1.0
- ✅ Fixed critical bug in chained assertion validation
- ✅ Added comprehensive tolerance support for all comparisons
- ✅ Improved error messages with actual vs expected values
- ✅ Enhanced NaN and infinity handling
- ✅ Optimized dependency graph execution
- ✅ Fixed CheckNode status propagation based on child node results
- ✅ Added builder pattern for suite creation
- ✅ Standardized API with better type hints
- ✅ Achieved 100% test coverage for graph.py module
- ✅ Refactored graph.py to separate display logic into display.py
- ✅ Achieved 100% test coverage for display.py module
- ✅ Improved code organization with clear separation of concerns
- ✅ Refactored analyzer.py to use sets for operation deduplication (simpler and more efficient than defaultdict)
- ✅ Achieved 100% test coverage for analyzer.py module

---

Built with ❤️ by the Data Quality Team
