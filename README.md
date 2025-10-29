# DQX - Data Quality Excellence

Transform data validation into mathematical expressions. Find issues before they find you.

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation Status](https://readthedocs.org/projects/dqx/badge/?version=latest)](https://dqx.readthedocs.io/en/latest/?badge=latest)

Write quality checks as code, not SQL. Get instant feedback on millions of rows.

## Why DQX?

### Before: Complex SQL that's hard to maintain
```sql
WITH metrics AS (
  SELECT
    COUNT(*) as total,
    SUM(CASE WHEN price IS NULL THEN 1 ELSE 0 END) as null_prices,
    AVG(price) as avg_price
  FROM orders
)
SELECT
  CASE
    WHEN null_prices::float / total > 0.05 THEN 'FAIL: Too many nulls'
    WHEN avg_price < 10 THEN 'FAIL: Prices too low'
    ELSE 'PASS'
  END as result
FROM metrics;
```

### After: Clear, testable validation logic
```python
@check(name="Order quality")
def validate_orders(mp: MetricProvider, ctx: Context) -> None:
    null_rate = mp.null_count("price") / mp.num_rows()
    ctx.assert_that(null_rate).where(name="Price completeness").is_leq(0.05)
    ctx.assert_that(mp.average("price")).where(name="Price reasonableness").is_geq(10.0)
```

## Quick Start

```bash
pip install dqx
```

Then validate your data:

```python
import pyarrow as pa
import sympy as sp
from dqx.api import check, VerificationSuite, MetricProvider, Context
from dqx.common import ResultKey
from dqx.datasource import DuckRelationDataSource
from dqx.orm.repositories import InMemoryMetricDB


# Your business rules as code
@check(name="Revenue integrity")
def validate_revenue(mp: MetricProvider, ctx: Context) -> None:
    # Catch calculation errors
    calculated = mp.sum("price") * mp.sum("quantity")
    reported = mp.sum("revenue")
    error_rate = sp.Abs(calculated - reported) / reported

    ctx.assert_that(error_rate).where(
        name="Revenue calculation accuracy", severity="P0"
    ).is_lt(
        0.001
    )  # Less than 0.1% error


# Load your data
data = pa.Table.from_pydict(
    {"price": [10.5, 20.0, 15.5], "quantity": [2, 1, 3], "revenue": [21.0, 20.0, 46.5]}
)

# Run validation
db = InMemoryMetricDB()
suite = VerificationSuite([validate_revenue], db, "Daily validation")
suite.run([DuckRelationDataSource.from_arrow(data)], ResultKey())
# ✓ Revenue integrity: OK
```

## What DQX Solves

Real examples from production data pipelines:

### 📊 Data Completeness
*"Are critical fields populated?"*

```python
@check(name="Customer data quality")
def check_completeness(mp: MetricProvider, ctx: Context) -> None:
    # Flag if more than 5% of customer IDs are missing
    null_rate = mp.null_count("customer_id") / mp.num_rows()
    ctx.assert_that(null_rate).where(
        name="Customer ID completeness", severity="P0"
    ).is_lt(0.05)
```

### 📈 Trend Monitoring
*"Did revenue drop unexpectedly?"*

```python
@check(name="Revenue stability")
def monitor_trends(mp: MetricProvider, ctx: Context) -> None:
    # Alert on >20% daily revenue changes
    daily_change = mp.sum("revenue") / mp.sum("revenue", lag=1)
    ctx.assert_that(daily_change).where(
        name="Daily revenue stability", severity="P0"
    ).is_between(0.8, 1.2)

    # Track week-over-week growth
    wow_change = mp.sum("revenue") / mp.sum("revenue", lag=7)
    ctx.assert_that(wow_change).where(name="Weekly revenue trend").is_geq(
        0.95
    )  # Allow 5% decline
```

### 🔍 Data Integrity
*"Are there duplicate transactions?"*

```python
@check(name="Transaction integrity")
def check_integrity(mp: MetricProvider, ctx: Context) -> None:
    # No duplicate transaction IDs allowed
    ctx.assert_that(mp.duplicate_count(["transaction_id"])).where(
        name="Transaction uniqueness", severity="P0"
    ).is_eq(0)

    # Validate business rules
    ctx.assert_that(mp.minimum("quantity")).where(
        name="Positive quantities only"
    ).is_positive()
```

### ⚖️ Cross-Dataset Validation
*"Do staging and production match?"*

```python
@check(name="Environment consistency", datasets=["staging", "production"])
def compare_environments(mp: MetricProvider, ctx: Context) -> None:
    # Ensure staging matches production
    prod_total = mp.sum("revenue", dataset="production")
    staging_total = mp.sum("revenue", dataset="staging")

    ctx.assert_that(prod_total).where(name="Prod-staging revenue match").is_eq(
        staging_total, tol=0.01
    )  # 1% tolerance
```

## Core Concepts

Just three simple ideas:

1. **Checks** = Your validation logic (decorated functions)
2. **Metrics** = What you measure (`sum`, `average`, `null_count`)
3. **Assertions** = Your rules (`is_between`, `is_positive`, `is_eq`)

## Common Patterns

### Validate Percentages
```python
# Check null rate
null_rate = mp.null_count("email") / mp.num_rows()
ctx.assert_that(null_rate).where(name="Email completeness").is_lt(0.1)  # <10%

# Check category distribution
fraud_rate = mp.count_values("status", "fraud") / mp.num_rows()
ctx.assert_that(fraud_rate).where(name="Fraud rate", severity="P0").is_lt(0.001)

# Check cardinality
unique_users = mp.unique_count("user_id")
total_orders = mp.num_rows()
ctx.assert_that(unique_users / total_orders).where(name="User diversity").is_gt(
    0.3
)  # >30% unique users
```

### Monitor Complex Metrics
```python
# Standard deviation over time
tax_volatility = mp.ext.stddev(mp.average("tax_rate"), lag=1, n=30)
ctx.assert_that(tax_volatility).where(name="Tax rate stability").is_lt(0.05)

# Multi-step calculations
conversion_rate = mp.count_values("converted", True) / mp.count_values("visited", True)
ctx.assert_that(conversion_rate).where(name="Conversion rate").is_between(0.02, 0.10)
```

### Detect Anomalies
```python
# Spike detection
max_payment = mp.maximum("payment_amount")
avg_payment = mp.average("payment_amount")
spike_ratio = max_payment / avg_payment

ctx.assert_that(spike_ratio).where(name="Payment spike detection", severity="P1").is_lt(
    100
)  # Max should be <100x average
```

## Development & Tooling

### Setup
```bash
git clone <repo>
cd dqx
./bin/setup-dev-env.sh  # One-time setup
```

### Daily Workflow
```bash
# Run tests with coverage
uv run pytest --cov=dqx

# Check code quality (auto-fixes many issues)
uv run hooks

# Commit with conventional format
uv run cz commit  # Interactive commit helper

# Clean up temporary files
uv run cleanup
```

### Key Tools
- **uv** - Fast Python package & environment manager
- **pytest** - Testing with 100% coverage requirement
- **ruff** - Lightning-fast Python linter
- **mypy** - Static type checking
- **commitizen** - Conventional commit enforcement

## Quick Reference

### Metrics Available
| Metric | Description | Example |
|--------|-------------|---------|
| `num_rows()` | Total row count | Basic count |
| `null_count(col)` | Null values in column | Data completeness |
| `sum(col)` | Sum of values | Totals |
| `average(col)` | Mean value | Averages |
| `minimum(col)` / `maximum(col)` | Min/max values | Range checks |
| `duplicate_count([cols])` | Duplicate rows | Uniqueness |
| `count_values(col, value)` | Count specific values | Categories |
| `unique_count(col)` | Distinct values | Cardinality |

### Assertions Available
| Assertion | Description | Example |
|-----------|-------------|---------|
| `is_eq(value, tol)` | Equals (with tolerance) | Exact match |
| `is_between(min, max)` | In range (inclusive) | Valid ranges |
| `is_positive()` / `is_zero()` | Sign checks | Amounts |
| `is_gt(val)` / `is_geq(val)` | Greater than (or equal) | Thresholds |
| `is_lt(val)` / `is_leq(val)` | Less than (or equal) | Limits |

### Time-based Analysis
```python
# Yesterday's value
mp.sum("revenue", lag=1)

# Last week's value
mp.sum("revenue", lag=7)

# Day-over-day change
mp.ext.day_over_day(mp.sum("revenue"))

# Week-over-week change
mp.ext.week_over_week(mp.sum("revenue"))
```

## Installation Options

```bash
# Basic installation
pip install dqx

# With PostgreSQL support
pip install "dqx[postgres]"

# Development setup
pip install "dqx[dev]"
```

## Learn More

- [Examples](examples/) - Real-world validation scenarios
- [Design Document](docs/design.md) - Architecture decisions
- [Plugin System](docs/plugin_system.md) - Extend DQX functionality

## License

MIT License. See [LICENSE](LICENSE) for details.

---

*DQX makes data quality a first-class concern. Stop writing SQL. Start validating data.*
