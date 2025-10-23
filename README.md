# DQX

Write quality checks as mathematical expressions. Validate data efficiently. Get instant feedback.

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Start Here

```bash
pip install dqx
```

```python
import pyarrow as pa

from dqx.api import check, VerificationSuite, MetricProvider, Context
from dqx.extensions.duckds import DuckRelationDataSource
from dqx.orm.repositories import InMemoryMetricDB
from dqx.common import ResultKey


@check("Orders have prices")
def validate_orders(mp: MetricProvider, ctx: Context) -> None:
    ctx.assert_that(mp.average("price")).where(name="Price check").is_positive()


# Example data
data = pa.Table.from_pydict({"price": [10.5, 24.99, 5.0, 100.0]})

# Run it
db = InMemoryMetricDB()
suite = VerificationSuite([validate_orders], db, "Quick validation")
suite.run({"orders": DuckRelationDataSource.from_arrow(data)}, ResultKey())
# ✓ Orders have prices: OK
```

Done. You just validated your first dataset.

## Three Things to Know

### 1. Checks = Functions
```python
from dqx.api import check, MetricProvider, Context


@check("Customer data is complete")
def validate_customers(mp: MetricProvider, ctx: Context) -> None:
    ctx.assert_that(mp.null_count("email")).where(name="Email completeness").is_zero()
```

### 2. Metrics = Measurements
```python
mp.average("price")  # → 24.99
mp.null_count("email")  # → 0
mp.approx_cardinality("id")  # → 10,847
```

### 3. Assertions = Rules
```python
ctx.assert_that(metric).where(name="Rule description").is_positive()
ctx.assert_that(metric).where(name="Range check").is_between(0, 100)
ctx.assert_that(metric).where(name="Exact match").is_eq(expected)
```

## Common Patterns

### Monitor Daily Changes
```python
from dqx.api import check, MetricProvider, Context


@check("Revenue stability")
def monitor_revenue(mp: MetricProvider, ctx: Context) -> None:
    today = mp.sum("revenue")
    yesterday = mp.sum("revenue", key=ctx.key.lag(1))

    change = (today - yesterday) / yesterday
    ctx.assert_that(change).where(name="Daily change", severity="P0").is_between(
        -0.1, 0.1
    )  # ±10%
```

### Compare Datasets
```python
from dqx.api import check, MetricProvider, Context


@check("Prod matches staging")
def compare_environments(mp: MetricProvider, ctx: Context) -> None:
    prod = mp.sum("revenue", dataset="production")
    staging = mp.sum("revenue", dataset="staging")

    ctx.assert_that(prod).where(name="Environment match").is_eq(staging, tol=0.01)
```

### Detect Anomalies
```python
from dqx.api import check, MetricProvider, Context


@check("No payment spikes")
def detect_anomalies(mp: MetricProvider, ctx: Context) -> None:
    avg = mp.average("payment_amount")
    max_payment = mp.maximum("payment_amount")

    ctx.assert_that(max_payment / avg).where(name="Spike detection").is_lt(100)
```

## Real Example

```python
from datetime import date

import pyarrow as pa
import sympy as sp

from dqx.api import check, VerificationSuite, MetricProvider, Context
from dqx.common import ResultKey
from dqx.display import print_assertion_results
from dqx.extensions.duckds import DuckRelationDataSource
from dqx.orm.repositories import MetricDB


# Your validation logic
@check("E-commerce data quality")
def validate_ecommerce(mp: MetricProvider, ctx: Context) -> None:
    # Revenue integrity
    calculated_revenue = mp.sum("price") * mp.sum("quantity")
    reported_revenue = mp.sum("revenue")
    error_rate = sp.Abs(calculated_revenue - reported_revenue) / reported_revenue
    ctx.assert_that(error_rate).where(name="Revenue accuracy").is_lt(0.001)

    # No nulls in critical fields
    ctx.assert_that(mp.null_count("order_id")).where(
        name="Order ID completeness"
    ).is_eq(0)
    ctx.assert_that(mp.null_count("customer_id")).where(
        name="Customer ID completeness"
    ).is_eq(0)

    # Reasonable ranges
    ctx.assert_that(mp.average("price")).where(name="Average price range").is_between(
        10, 1000
    )
    ctx.assert_that(mp.maximum("quantity")).where(name="Max quantity limit").is_leq(100)


# Set up data
# In practice, you might read from parquet: data = pa.parquet.read_table("orders.parquet")
data = pa.Table.from_pydict(
    {
        "order_id": list(range(1001, 1051)),  # 50 orders
        "customer_id": [200 + i % 10 for i in range(50)],
        "price": [10.0 + i * 2.5 for i in range(50)],
        "quantity": [1 + i % 5 for i in range(50)],
        "revenue": [(10.0 + i * 2.5) * (1 + i % 5) for i in range(50)],
    }
)
datasource = DuckRelationDataSource.from_arrow(data)

# Run validation
db = MetricDB("postgresql://localhost/metrics")
suite = VerificationSuite([validate_ecommerce], db, "Daily e-commerce validation")
key = ResultKey(yyyy_mm_dd=date.today())
suite.run({"orders": datasource}, key)

# Check results
results = suite.collect_results()
print_assertion_results(results)
```

## API Reference

### Define Checks
```python
from dqx.api import check, MetricProvider, Context


@check(
    name="Required name",  # What this validates
    severity="P0",  # P0-P3 (optional)
    datasets=["orders"],  # Limit to specific datasets (optional)
)
def my_check(mp: MetricProvider, ctx: Context) -> None:
    """Your validation logic here."""
    pass
```

### Available Metrics
```python
# Counts
mp.num_rows()  # Total rows
mp.count("column")  # Non-null count
mp.null_count("column")  # Null count
mp.approx_cardinality("column")  # Distinct values (fast, approximate)

# Statistics
mp.sum("column")  # Sum
mp.average("column")  # Mean
mp.minimum("column")  # Min
mp.maximum("column")  # Max
mp.variance("column")  # Variance
mp.stddev("column")  # Standard deviation

# Time-based
mp.sum("revenue", key=ctx.key.lag(1))  # Yesterday
mp.sum("revenue", key=ctx.key.lag(7))  # Last week

# Multi-dataset
mp.sum("amount", dataset="production")  # Specific dataset
```

### Batch Analysis

Analyze multiple dates efficiently in a single operation:

```python
from dqx.analyzer import Analyzer
from dqx.common import ResultKey
from dqx import specs

# Define metrics
metrics = [specs.Sum("revenue"), specs.Average("price"), specs.NullCount("customer_id")]

# Analyze multiple dates at once
analyzer = Analyzer()
dates = [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)]
metrics_by_key = {ResultKey(d): metrics for d in dates}

# Single batch operation instead of multiple analyze() calls
report = analyzer.analyze_batch(datasource, metrics_by_key)

# Different metrics per date
custom_metrics = {
    ResultKey(date(2024, 1, 1)): [specs.Sum("revenue"), specs.Average("price")],
    ResultKey(date(2024, 1, 2)): [specs.Maximum("quantity"), specs.NumRows()],
}
report = analyzer.analyze_batch(datasource, custom_metrics)
```

Large date ranges are automatically split into optimal batches for performance.

### Make Assertions
```python
# Every assertion needs a name
draft = ctx.assert_that(metric).where(name="Description")

# Then apply your rule
draft.is_eq(value)  # Equals
draft.is_gt(value)  # Greater than
draft.is_geq(value)  # Greater or equal
draft.is_lt(value)  # Less than
draft.is_leq(value)  # Less or equal
draft.is_between(min, max)  # In range
draft.is_positive()  # > 0
draft.is_zero()  # = 0

# With tolerance
draft.is_eq(100, tol=1)  # 99 ≤ x ≤ 101
```

### Run Validation
```python
# Create suite
suite = VerificationSuite(
    checks=[check1, check2], db=MetricDB("connection_string"), name="Suite name"
)

# Execute
key = ResultKey(yyyy_mm_dd=date.today())
context = suite.run(datasources={"name": datasource}, key=key)

# Inspect results
context.has_failures()  # True if any check failed
context.checks()  # List of check results
```

## How It Works

DQX compiles your checks into a dependency graph, generates optimized SQL, and executes everything in a single pass. Results flow back through the graph to evaluate assertions. You get comprehensive validation without writing SQL or managing complex pipelines.

```
Your Checks → Dependency Graph → Optimized SQL → DuckDB → Results
```

## Data Sources

### Built-in Support
```python
# PyArrow Tables
from dqx.extensions.duckds import DuckRelationDataSource
import pyarrow as pa

# From dict
data = pa.Table.from_pydict({"col": [1, 2, 3]})
ds = DuckRelationDataSource.from_arrow(data)

# From parquet file
data = pa.parquet.read_table("file.parquet")
ds = DuckRelationDataSource.from_arrow(data)

# DuckDB
from dqx.extensions.duck_ds import DuckDataSource

ds = DuckDataSource(conn, "SELECT * FROM table")
```

### Custom Sources
Implement the protocol:
```python
class MyDataSource:
    name = "custom"
    dialect = "duckdb"

    @property
    def cte(self) -> str:
        return "SELECT * FROM my_table"
```

## Persistence

Store metrics for trending and analysis:

```python
# Local development
db = InMemoryMetricDB()
db = MetricDB("sqlite:///metrics.db")

# Production
db = MetricDB("postgresql://user:pass@host/db")

# Metrics automatically persist after validation
suite = VerificationSuite(checks, db, "Production")
suite.run(data, key)

# Query history
from datetime import date, timedelta

metrics = db.get_metrics(
    metric_name="sum(revenue)",
    start_date=date.today() - timedelta(days=30),
    end_date=date.today(),
)
```

## Display Results

```python
from dqx.display import print_assertion_results, print_symbols

# After running validation
results = suite.collect_results()
print_assertion_results(results)  # Formatted table with colors

symbols = suite.collect_symbols()
print_symbols(symbols)  # Show all computed values
```

## Plugin System

DQX includes a plugin system for extending validation result processing. Plugins can generate reports, send notifications, collect metrics, or perform custom processing after validation completes.

### Built-in Audit Plugin

The audit plugin displays a Rich-formatted summary of validation results and is enabled by default:

```python
# Audit plugin enabled by default
suite = VerificationSuite("MyDataQuality")

# Explicitly control plugin usage
suite = VerificationSuite("MyDataQuality", enable_plugins=True)
suite = VerificationSuite("MyDataQuality", enable_plugins=False)
```

### Creating Custom Plugins

```python
from dqx.common import PluginExecutionContext, PluginMetadata


class MyPlugin:
    @staticmethod
    def metadata() -> PluginMetadata:
        return PluginMetadata(
            name="my_plugin",
            version="1.0.0",
            author="Your Name",
            description="What your plugin does",
        )

    def process(self, context: PluginExecutionContext) -> None:
        """Process validation results."""
        if context.failed_assertions() > 0:
            # Send alerts, write reports, etc.
            print(
                f"Alert: {context.suite_name} has {context.failed_assertions()} failures!"
            )
```

### Plugin Examples

See the [Plugin System Documentation](docs/plugin_system.md) for complete examples including:
- JSON report generation
- Slack notifications
- Metrics collection
- Error handling and timeouts

The [plugin demo](examples/plugin_demo.py) shows a complete working implementation.

## Development

### Setup
```bash
git clone https://github.com/yourusername/dqx.git
cd dqx
./bin/setup-dev-env.sh
```

### Test
```bash
# Run all checks
./bin/run-hooks.sh

# Run specific tests
pytest tests/test_api.py -v

# Check coverage
pytest --cov=dqx --cov-report=html
```

### Conventional Commits

This project uses [conventional commits](https://www.conventionalcommits.org/) for all commit messages. This enables:
- Automated changelog generation
- Semantic versioning
- Clear commit history

#### Commit Format
```
<type>(<scope>): <subject>

<body>

<footer>
```

#### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Test additions or fixes
- `build`: Build system changes
- `ci`: CI/CD changes
- `chore`: Other changes

#### Examples
```bash
# Simple commit
git commit -m "feat(analyzer): add query optimization for large datasets"

# Breaking change
git commit -m "feat(api)!: change assertion interface to require where() clause"

# With body
git commit -m "fix(evaluator): handle division by zero in percentage calculations

This fixes the issue where percentage calculations would fail when the
denominator is zero. Now returns NaN instead of raising an exception.

Closes #123"
```

#### Commit Creation and Validation

You have two ways to create and validate commits:

```bash
# Option 1: Use commitizen for interactive commit creation
uv run cz commit

# Option 2: Use regular git commit (validation happens automatically)
git commit -m "feat(api): add new validation method"

# Check if your last commit follows the convention
./bin/run-hooks.sh --check-commit
```

### Pre-commit Hooks

This project uses pre-commit hooks to maintain code quality. Hooks run automatically on `git commit` and check:

- **Code formatting** (ruff)
- **Type checking** (mypy)
- **Commit message format** (commitizen)
- **File quality** (trailing whitespace, file endings)
- **Security** (no private keys)
- **Syntax validation** (Python, YAML, JSON)

If a commit is blocked:
1. Read the error message
2. Fix the issues (many are auto-fixed)
3. Re-stage changed files: `git add -u`
4. Commit again

For emergencies only:
```bash
# Skip all hooks (not recommended)
git commit --no-verify -m "emergency fix"

# Skip specific hook
SKIP=mypy git commit -m "wip: debugging"
```

### Contribute

1. Fork the repository
2. Create a feature branch
3. Write tests first (TDD)
4. Make your changes
5. Run `./bin/run-hooks.sh`
6. Commit with conventional format
7. Submit a pull request

**We need**: Better error messages, more data sources, faster algorithms.

## Advanced Usage

### Cross-Time Analysis
```python
from dqx.api import check, MetricProvider, Context


@check("Trend analysis")
def analyze_trend(mp: MetricProvider, ctx: Context) -> None:
    # Get metrics for multiple time periods
    current = mp.sum("sales")
    daily_values = [mp.sum("sales", key=ctx.key.lag(i)) for i in range(1, 8)]

    # Calculate week-over-week growth
    last_week = sum(daily_values)
    growth = (current * 7 - last_week) / last_week

    ctx.assert_that(growth).where(name="Weekly growth").is_between(-0.2, 0.5)
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Questions?

- 📐 [Design Document](docs/design.md) - Architecture and design decisions

---

DQX makes data validation expressive and reliable. Stop writing SQL. Start validating data.
