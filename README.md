# DQX - Data Quality eXcellence

A high-performance, scalable data quality framework built on DuckDB and PyArrow for fast, efficient data validation and monitoring.

## 🚀 Features

- **Blazing Fast**: Powered by DuckDB's analytical engine for sub-second query performance
- **Memory Efficient**: Statistical sketching algorithms (HyperLogLog, DataSketches) for large datasets
- **Declarative API**: Intuitive symbolic expressions for data quality checks with fluent assertion chaining
- **Batch Processing**: Multi-threaded analysis with configurable chunks for TB-scale data
- **Dependency Graph**: Smart execution planning with automatic metric deduplication
- **Extensible**: Plugin architecture for custom metrics and data sources
- **Production Ready**: Built-in persistence, comprehensive error handling, and detailed failure messages

## 📦 Installation

```bash
# Clone the repository
git clone git@gitlab.booking.com:npham/dqx.git
cd dqx

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### Requirements

- Python 3.11 or 3.12
- Dependencies are automatically installed, including:
  - DuckDB ≥ 1.3.2 (analytical engine)
  - PyArrow ≥ 21.0.0 (columnar data processing)
  - DataSketches ≥ 5.2.0 (probabilistic data structures)
  - SymPy ≥ 1.14.0 (symbolic mathematics)
  - SQLAlchemy ≥ 2.0.43 (database abstraction)
  - Returns ≥ 0.26.0 (functional programming utilities)

## 🎯 Quick Start

```python
import datetime as dt
import pyarrow as pa
from dqx.api import VerificationSuiteBuilder, check
from dqx.extensions.pyarrow_ds import ArrowDataSource
from dqx.orm.repositories import InMemoryMetricDB
from dqx.common import ResultKey

# Define your data quality check
@check(label="Order validation", tags=["critical"])
def validate_orders(mp, ctx):
    # Check for null values
    ctx.assert_that(mp.null_count("customer_id")).is_eq(0)
    
    # Validate price ranges with tolerance
    ctx.assert_that(mp.minimum("price")).is_geq(0.0, tol=0.01)
    ctx.assert_that(mp.average("price")).is_gt(10.0)
    
    # Chain multiple assertions on the same metric
    ctx.assert_that(mp.average("quantity")).on(
        label="Quantity should be reasonable"
    ).is_gt(0).is_leq(100)
    
    # Check data volume
    ctx.assert_that(mp.num_rows()).is_geq(100)

# Create your data
data = pa.table({
    "customer_id": [1, 2, 3, 4, 5],
    "price": [15.99, 23.50, 45.00, 12.75, 89.99],
    "quantity": [2, 1, 3, 1, 2]
})

# Build and run verification suite
db = InMemoryMetricDB()
suite = (VerificationSuiteBuilder("Order Quality Suite", db)
         .add_check(validate_orders)
         .build())

# Run the checks
data_source = ArrowDataSource(data)
key = ResultKey(yyyy_mm_dd=dt.date.today(), tags={"env": "prod"})
context = suite.run({"orders": data_source}, key)

# Inspect results
for assertion in context._graph.assertions():
    if assertion._value:
        print(f"{assertion.label}: {assertion._value}")
```

## 📊 Available Metrics

### Basic Statistics
- `num_rows()` - Total row count
- `average(column)` - Mean value
- `sum(column)` - Sum of values
- `minimum(column)` - Minimum value
- `maximum(column)` - Maximum value
- `variance(column)` - Sample variance
- `first(column)` - First non-null value

### Data Quality
- `null_count(column)` - Count of null values
- `approx_cardinality(column)` - Estimated unique values using HyperLogLog

### Custom Metrics
```python
from dqx import specs

# Use built-in metric specs
negative_values = mp.metric(specs.NegativeCount("balance"))
ctx.assert_that(negative_values).is_eq(0)  # No negative balances allowed

# Access metrics with time offsets
yesterday_avg = mp.average("score", key=ctx.key.lag(1))
```

### Extended Metrics
- `ext.day_over_day(metric, key)` - Compare metrics across days
- `ext.stddev(metric, lag, n, key)` - Standard deviation over time windows

## 🔍 Writing Data Quality Checks

### Using the Check Decorator

```python
@check(
    label="Data completeness check",
    tags=["completeness", "critical"],
    datasets=["main_table"]  # Optional: specify required datasets
)
def check_completeness(mp, ctx):
    """Ensure key columns have minimal null values."""
    total_rows = mp.num_rows()
    
    for column in ["id", "name", "email"]:
        null_percentage = mp.null_count(column) / total_rows
        ctx.assert_that(null_percentage).on(
            label=f"{column} null percentage",
            severity="P0"  # Critical severity
        ).is_leq(0.05)  # Max 5% nulls
```

### Assertion Methods

All comparison methods support an optional `tol` parameter for floating-point tolerance:

```python
# Basic comparisons
ctx.assert_that(metric).is_eq(100, tol=0.1)       # Equal within tolerance
ctx.assert_that(metric).is_gt(0)                  # Greater than
ctx.assert_that(metric).is_geq(0, tol=1e-6)       # Greater than or equal
ctx.assert_that(metric).is_lt(1000)               # Less than
ctx.assert_that(metric).is_leq(1000, tol=0.01)    # Less than or equal

# Special checks
ctx.assert_that(metric).is_positive()              # Value > 0
ctx.assert_that(metric).is_negative()              # Value < 0

# Configure severity and labels
ctx.assert_that(metric).on(
    label="Critical business rule",
    severity="P0"  # Severity levels: "P0", "P1", "P2", "P3"
).is_geq(0)
```

### Assertion Chaining

DQX supports fluent assertion chaining for multiple validations on the same metric:

```python
# Validate a ratio is within acceptable bounds
ratio = mp.average("price") / mp.average("tax")
ctx.assert_that(ratio).on(
    label="Price/tax ratio check"
).is_geq(0.95).is_leq(1.05)

# Complex validation with multiple conditions
ctx.assert_that(mp.sum("revenue")).on(
    label="Revenue validation",
    severity="P0"
).is_positive().is_lt(1000000).is_geq(10000)

# Each chained assertion creates a separate validation node
# This provides:
# - Independent failure tracking
# - Granular error messages
# - Clear visibility into which conditions failed
```

## 📁 Data Sources

### PyArrow Tables

```python
from dqx.extensions.pyarrow_ds import ArrowDataSource

# Single table
ds = ArrowDataSource(your_arrow_table)

# Run checks
suite.run({"dataset_name": ds}, key)
```

### Batch Processing

```python
from dqx.extensions.pyarrow_ds import ArrowBatchDataSource

# Process multiple Parquet files efficiently
batch_ds = ArrowBatchDataSource.from_parquets([
    "data/file1.parquet",
    "data/file2.parquet",
    "data/file3.parquet"
])

# Enable multi-threading for parallel processing
suite.run({"large_dataset": batch_ds}, key, threading=True)
```

### Custom Data Sources

Implement the `SqlDataSource` protocol:

```python
from dqx.common import SqlDataSource, Analyzer
import duckdb

class CustomDataSource:
    name = "custom_source"
    analyzer_class = Analyzer  # Your analyzer implementation
    
    @property
    def cte(self) -> str:
        """SQL fragment for WITH clause."""
        return "SELECT * FROM my_custom_table WHERE date = '2024-01-01'"
        
    def query(self, query: str) -> duckdb.DuckDBPyRelation:
        """Execute query against this data source."""
        conn = duckdb.connect()
        # Set up your custom table/view
        return conn.execute(query).fetchdf()
```

## 🚀 Advanced Usage

### Time-based Comparisons

```python
@check
def monitor_trends(mp, ctx):
    # Compare metrics across time periods
    current = mp.average("revenue")
    yesterday = mp.average("revenue", key=ctx.key.lag(1))
    last_week = mp.average("revenue", key=ctx.key.lag(7))
    
    # Day-over-day check
    ctx.assert_that(current / yesterday).on(
        label="Daily revenue change"
    ).is_geq(0.9).is_leq(1.1)  # ±10% change allowed
    
    # Week-over-week trend
    ctx.assert_that(current / last_week).on(
        label="Weekly revenue trend"
    ).is_geq(0.8)  # No more than 20% drop
```

### Cross-dataset Validation

```python
@check
def cross_validate(mp, ctx):
    # Compare metrics across different datasets
    prod_count = mp.num_rows(datasets=["production"])
    staging_count = mp.num_rows(datasets=["staging"])
    
    # Ensure data consistency
    ratio = prod_count / staging_count
    ctx.assert_that(ratio).on(
        label="Production/Staging data consistency"
    ).is_geq(0.95, tol=0.01).is_leq(1.05, tol=0.01)
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

### Error Handling and Validation

DQX provides comprehensive validation and clear error messages:

```python
from dqx.api import GraphStates

# The framework validates inputs
try:
    # Empty suite name
    suite = VerificationSuiteBuilder("", db).build()
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
    end_date=dt.date.today()
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
│ (SQL/Sketch)│     │(DataSources) │     │  (Schema)  │
└─────────────┘     └──────────────┘     └────────────┘
```

### Key Components

- **API**: High-level verification suites and check definitions
- **Graph**: Dependency tracking and execution planning
- **Specs**: Metric specifications (row count, averages, etc.)
- **Analyzer**: SQL generation and execution engine
- **Ops**: Low-level SQL and sketch operations
- **States**: Serializable metric states for distributed processing
- **Extensions**: Data source adapters (PyArrow, etc.)
- **Models**: Data models and ORM definitions
- **MetricDB**: Persistence layer for historical metrics

## 🛠️ Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_api.py

# Run with coverage
uv run pytest --cov=dqx

# Run tests in parallel
uv run pytest -n auto
```

### Code Quality

DQX maintains high code quality standards:

```bash
# Linting with Ruff
uv run ruff check src/          # Check for issues
uv run ruff check src/ --fix    # Auto-fix issues

# Type checking with MyPy
uv run mypy src/

# Format code
uv run ruff format src/ tests/

# Run all quality checks
uv run ruff check src/ tests/ && uv run ruff format src/ tests/ && uv run mypy src/
```

### Development Setup

```bash
# Install development dependencies
uv sync --all-extras

# Install pre-commit hooks (optional)
pre-commit install
```

### Commit Standards

We use [Commitizen](https://commitizen-tools.github.io/commitizen/) for standardized commits:

```bash
# Create a commit interactively
cz commit  # or: cz c

# Bump version automatically
cz bump

# Commit format examples:
# feat: add batch processing support
# fix: resolve chained assertion validation
# docs: update installation instructions
# refactor: simplify analyzer logic
# test: add coverage for edge cases
# perf: optimize SQL deduplication
# chore: update dependencies
```

## 🔧 Performance Optimization

DQX is optimized for performance through:

1. **SQL Optimization**
   - Batches multiple metrics into single SQL queries
   - Deduplicates redundant computations
   - Leverages DuckDB's columnar processing

2. **Memory Efficiency**
   - HyperLogLog for cardinality estimation (99.9% accuracy, <1% memory)
   - DataSketches for quantiles and histograms
   - Streaming algorithms for large datasets

3. **Parallel Processing**
   - Multi-threaded batch analysis
   - Concurrent data source processing
   - Lock-free metric aggregation

4. **Smart Caching**
   - Reuses computed metrics across checks
   - Dependency graph minimizes redundant work
   - Persistent metric storage for historical analysis

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🗺️ Roadmap

- [x] Fluent assertion chaining
- [x] Comprehensive error messages
- [x] Builder pattern for suite creation
- [ ] Streaming data source support
- [ ] Web UI for monitoring and alerting
- [ ] Data catalog integrations (Hive, Glue, Unity)
- [ ] ML-based anomaly detection
- [ ] Complex event processing
- [ ] Kubernetes operators for cloud-native deployment

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Write tests for new functionality
4. Ensure all tests pass (`uv run pytest`)
5. Run code quality checks:
   ```bash
   uv run ruff check src/ tests/
   uv run mypy src/
   uv run ruff format src/ tests/
   ```
6. Commit your changes (`cz commit`)
7. Push to your branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Development Guidelines

- Write comprehensive docstrings (Google style)
- Add type hints to all functions
- Maintain test coverage above 90%
- Follow existing code patterns
- Update documentation as needed

## 📞 Support

- Create an issue on GitLab for bug reports or feature requests
- Check existing issues before creating new ones
- Provide minimal reproducible examples for bugs
- Join discussions in merge requests

## 🏆 Recent Improvements

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
