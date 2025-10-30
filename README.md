# DQX - Data Quality Excellence

Data quality as code. Works with your warehouse, scales with your needs.

[![Tests](https://github.com/nampham2/dqx/actions/workflows/test.yml/badge.svg)](https://github.com/nampham2/dqx/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/nampham2/dqx/branch/main/graph/badge.svg)](https://codecov.io/gh/nampham2/dqx)
[![Documentation Status](https://readthedocs.org/projects/dqx/badge/?version=latest)](https://dqx.readthedocs.io/en/latest/?badge=latest)
[![CodeRabbit Pull Request Reviews](https://img.shields.io/coderabbit/prs/github/nampham2/dqx?utm_source=oss&utm_medium=github&utm_campaign=nampham2%2Fdqx&labelColor=171717&color=FF570A&link=https%3A%2F%2Fcoderabbit.ai&label=CodeRabbit+Reviews)](https://coderabbit.ai)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Why DQX?

- **Write validation logic as testable Python functions** - No more complex SQL scripts scattered across your codebase
- **Execute efficiently on any SQL backend** - DuckDB, BigQuery, Snowflake, or your existing data warehouse
- **No clusters or complex infrastructure needed** - Runs wherever your data lives
- **Integrates seamlessly with existing workflows** - Drop it into your current pipeline

## Quick Start

```bash
pip install dqlib
```

Define your data quality checks as Python functions:

```python
import pyarrow as pa
from dqx.api import check, VerificationSuite, MetricProvider, Context
from dqx.common import ResultKey
from dqx.datasource import DuckRelationDataSource
from dqx.orm.repositories import InMemoryMetricDB


# Define your validation rules
@check(name="Revenue integrity")
def validate_revenue(mp: MetricProvider, ctx: Context) -> None:
    # Verify reported revenue is positive
    reported = mp.sum("revenue")
    ctx.assert_that(reported).where(
        name="Revenue is positive", severity="P0"
    ).is_positive()

    # Check average transaction size is reasonable
    avg_revenue = mp.average("revenue")
    ctx.assert_that(avg_revenue).where(
        name="Average transaction size", severity="P1"
    ).is_between(10, 100)


# Your own metric store
db = InMemoryMetricDB()
suite = VerificationSuite([validate_revenue], db, "Daily validation")

# Data comes from your warehouse
data = pa.Table.from_pydict(
    {"price": [10.5, 20.0, 15.5], "quantity": [2, 1, 3], "revenue": [21.0, 20.0, 46.5]}
)
datasource = DuckRelationDataSource.from_arrow(data)

# Validate your data
suite.run([datasource], ResultKey())
# ✓ Revenue integrity: OK
```

## Real-World Examples

### 1. Data Completeness

#### Monitor critical fields aren't missing

```python
@check(name="Customer data quality")
def check_completeness(mp: MetricProvider, ctx: Context) -> None:
    # Flag if more than 5% of emails are missing
    null_rate = mp.null_count("email") / mp.num_rows()
    ctx.assert_that(null_rate).where(name="Email completeness", severity="P0").is_lt(
        0.05
    )

    # Ensure all orders have customer IDs
    ctx.assert_that(mp.null_count("customer_id")).where(
        name="Customer ID required", severity="P0"
    ).is_eq(0)
```

### 2. Revenue Integrity

#### Catch calculation errors in financial data

```python
@check(name="Financial accuracy")
def validate_financials(mp: MetricProvider, ctx: Context) -> None:
    # Verify totals match across systems
    total_revenue = mp.sum("revenue")
    total_collected = mp.sum("payments")

    ctx.assert_that(total_collected / total_revenue).where(
        name="Payment collection rate", severity="P1"
    ).is_between(
        0.95, 1.05
    )  # 5% tolerance

    # Check for negative prices
    ctx.assert_that(mp.minimum("price")).where(
        name="No negative prices", severity="P0"
    ).is_geq(0)
```

### 3. Trend Monitoring

#### Alert on unexpected metric changes

```python
@check(name="Business metrics stability")
def monitor_trends(mp: MetricProvider, ctx: Context) -> None:
    # Alert on significant daily changes
    daily_change = mp.sum("revenue") / mp.sum("revenue", lag=1)
    ctx.assert_that(daily_change).where(
        name="Daily revenue stability", severity="P0"
    ).is_between(
        0.8, 1.2
    )  # ±20% change

    # Track week-over-week growth
    wow_change = mp.sum("revenue") / mp.sum("revenue", lag=7)
    ctx.assert_that(wow_change).where(
        name="Weekly revenue trend", severity="P1"
    ).is_geq(
        0.95
    )  # Allow 5% decline
```

### 4. Cross-Dataset Validation

#### Ensure consistency across environments

```python
@check(name="Production vs Staging", datasets=["production", "staging"])
def validate_environments(mp: MetricProvider, ctx: Context) -> None:
    # Compare row counts
    prod_count = mp.num_rows(dataset="production")
    staging_count = mp.num_rows(dataset="staging")

    ctx.assert_that(prod_count).where(name="Row count match", severity="P1").is_between(
        staging_count - 100, staging_count + 100
    )  # Allow 100 row difference

    # Verify key metrics align
    prod_revenue = mp.sum("revenue", dataset="production")
    staging_revenue = mp.sum("revenue", dataset="staging")

    ctx.assert_that((prod_revenue - staging_revenue) / prod_revenue).where(
        name="Revenue consistency", severity="P0"
    ).is_lt(
        0.01
    )  # Less than 1% difference
```

### 5. Data Quality SLAs

#### Track quality metrics with severity levels

```python
@check(name="Data quality SLAs")
def enforce_slas(mp: MetricProvider, ctx: Context) -> None:
    # P0: Critical - No duplicate transactions
    ctx.assert_that(mp.duplicate_count(["transaction_id"])).where(
        name="Transaction uniqueness", severity="P0"
    ).is_eq(0)

    # P1: High - Recent activity
    recent_count = mp.count_values("status", "active")
    total_count = mp.num_rows()
    active_rate = recent_count / total_count

    ctx.assert_that(active_rate).where(
        name="Active record percentage", severity="P1"
    ).is_gt(
        0.5
    )  # At least 50% active

    # P2: Medium - Cardinality checks
    unique_users = mp.unique_count("user_id")
    ctx.assert_that(unique_users).where(
        name="Active user threshold", severity="P2"
    ).is_gt(1000)
```

## Quick Reference

### Available Metrics

| Metric | Description | Example |
|--------|-------------|---------|
| `num_rows()` | Total row count | `mp.num_rows()` |
| `sum(col)` | Sum of values | `mp.sum("revenue")` |
| `average(col)` | Mean value | `mp.average("price")` |
| `minimum(col)` / `maximum(col)` | Min/max values | `mp.minimum("age")` |
| `first(col)` | First value in column | `mp.first("timestamp")` |
| `variance(col)` | Statistical variance | `mp.variance("score")` |
| `null_count(col)` | Count of null values | `mp.null_count("email")` |
| `duplicate_count([cols])` | Count of duplicate rows | `mp.duplicate_count(["id"])` |
| `count_values(col, val)` | Count specific values | `mp.count_values("status", "active")` |
| `unique_count(col)` | Distinct value count | `mp.unique_count("user_id")` |

### Extended Metrics

| Metric | Description | Example |
|--------|-------------|---------|
| `ext.day_over_day(metric)` | Day-over-day change | `mp.ext.day_over_day(mp.sum("revenue"))` |
| `ext.week_over_week(metric)` | Week-over-week change | `mp.ext.week_over_week(mp.average("price"))` |
| `ext.stddev(metric, offset, n)` | Standard deviation over window | `mp.ext.stddev(mp.sum("sales"), offset=0, n=7)` |

### Available Assertions

| Assertion | Description | Example |
|-----------|-------------|---------|
| `is_eq(value, tol)` | Equals with tolerance | `.is_eq(100, tol=0.01)` |
| `is_between(min, max)` | In range (inclusive) | `.is_between(0, 100)` |
| `is_positive()` | Greater than zero | `.is_positive()` |
| `is_zero()` | Equals zero | `.is_zero()` |
| `is_negative()` | Less than zero | `.is_negative()` |
| `is_gt(val)` / `is_geq(val)` | Greater than (or equal) | `.is_gt(0.95)` |
| `is_lt(val)` / `is_leq(val)` | Less than (or equal) | `.is_lt(0.05)` |
| `noop()` | No validation (collect only) | `.noop()` |

## License

MIT License. See [LICENSE](LICENSE) for details.
