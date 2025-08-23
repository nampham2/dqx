# DQX - Data Quality eXcellence

A high-performance, scalable data quality framework built on DuckDB and PyArrow for fast, efficient data validation and monitoring.

## 🚀 Features

- **Blazing Fast**: Powered by DuckDB's analytical engine
- **Memory Efficient**: Statistical sketching algorithms for large datasets
- **Declarative API**: Intuitive symbolic expressions for data quality checks
- **Batch Processing**: Multi-threaded analysis with configurable chunks
- **Extensible**: Plugin architecture for custom metrics and data sources
- **Production Ready**: Built-in persistence, monitoring, and comprehensive error handling

## 📦 Installation

```bash
# Clone the repository
git clone <repository-url>
cd dqx

# Install with uv (recommended)
uv install

# Or with pip
pip install -e .
```

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
    
    # Validate price ranges
    ctx.assert_that(mp.minimum("price")).is_geq(0.0)
    ctx.assert_that(mp.average("price")).is_gt(10.0)
    
    # Check data volume
    ctx.assert_that(mp.num_rows()).is_geq(100)

# Create your data
data = pa.table({
    "customer_id": [1, 2, 3, 4, 5],
    "price": [15.99, 23.50, 45.00, 12.75, 89.99]
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
context._graph.inspect()
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
- `approx_cardinality(column)` - Estimated unique values (HyperLogLog)

### Extended Metrics
- `ext.day_over_day(metric, key)` - Compare metrics across days
- `ext.stddev(metric, lag, n, key)` - Standard deviation over time windows

## 🔍 Writing Data Quality Checks

### Using the Check Decorator

```python
@check(
    label="Data completeness check",
    tags=["completeness", "critical"],
    datasets=["main_table"]  # Optional: specify which datasets this check applies to
)
def check_completeness(mp, ctx):
    """Ensure key columns have minimal null values."""
    total_rows = mp.num_rows()
    
    for column in ["id", "name", "email"]:
        null_percentage = mp.null_count(column) / total_rows
        ctx.assert_that(null_percentage).on(
            label=f"{column} should have <5% nulls"
        ).is_leq(0.05)
```

### Assertion Methods

```python
# Comparison operators
ctx.assert_that(metric).is_eq(100)      # Equal to
ctx.assert_that(metric).is_gt(0)        # Greater than
ctx.assert_that(metric).is_geq(0)       # Greater than or equal
ctx.assert_that(metric).is_lt(1000)     # Less than
ctx.assert_that(metric).is_leq(1000)    # Less than or equal

# Special checks
ctx.assert_that(metric).is_positive()   # Value > 0
ctx.assert_that(metric).is_negative()   # Value < 0

# Configure severity and labels
ctx.assert_that(metric).on(
    label="Critical business rule",
    severity=SeverityLevel.ERROR
).is_geq(0)
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
from dqx.common import SqlDataSource

class CustomDataSource:
    name = "custom"
    analyzer_class = Analyzer
    
    @property
    def cte(self) -> str:
        return "SELECT * FROM my_custom_table"
        
    def query(self, query: str) -> duckdb.DuckDBPyRelation:
        return duckdb.query(query)
```

## 🚀 Advanced Usage

### Time-based Comparisons

```python
@check
def monitor_trends(mp, ctx):
    # Compare current metrics with previous day
    current = mp.average("revenue")
    previous = mp.average("revenue", key=ctx.key.lag(1))
    
    # Revenue shouldn't drop more than 10%
    ctx.assert_that(current / previous).on(
        label="Day-over-day revenue drop"
    ).is_geq(0.9)
```

### Cross-dataset Validation

```python
@check
def cross_validate(mp, ctx):
    # Compare metrics across different datasets
    prod_count = mp.num_rows(datasets=["production"])
    staging_count = mp.num_rows(datasets=["staging"])
    
    # Ensure counts are similar (within 5%)
    ratio = prod_count / staging_count
    ctx.assert_that(ratio).on(
        label="Production/Staging consistency"
    ).is_geq(0.95).is_leq(1.05)
```

### Verification Suite Builder

```python
# Use the builder pattern for complex suites
suite = (VerificationSuiteBuilder("Data Quality Suite", db)
         .add_check(volume_check)
         .add_check(completeness_check)
         .add_checks([consistency_check, trend_check])
         .build())
```

## 💾 Persistence

Store metrics for historical analysis:

```python
from dqx.orm.repositories import MetricDB

# Configure persistent storage
db = MetricDB(connection_string="sqlite:///metrics.db")

# Metrics are automatically persisted
suite = VerificationSuite(checks, db, name="Production Suite")
```

## 🏗️ Architecture

DQX follows a modular architecture:

```
┌─────────────┐     ┌──────────────┐     ┌────────────┐
│   API       │────▶│  Analyzer    │────▶│  States    │
│  (Checks)   │     │  (Execution) │     │ (Storage)  │
└─────────────┘     └──────────────┘     └────────────┘
       │                    │                    │
       ▼                    ▼                    ▼
┌─────────────┐     ┌──────────────┐     ┌────────────┐
│   Specs     │     │     Ops      │     │ MetricDB   │
│  (Metrics)  │     │ (SQL/Sketch) │     │(Persistence)│
└─────────────┘     └──────────────┘     └────────────┘
```

- **API**: High-level verification suites and check definitions
- **Specs**: Metric specifications (row count, averages, etc.)
- **Ops**: SQL and sketch-based operations for analysis
- **States**: Serializable metric states for merging
- **Analyzer**: Execution engine for data operations
- **Extensions**: Data source adapters

## 🛠️ Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_api.py

# Run with coverage
uv run pytest --cov=dqx
```

### Code Quality

DQX maintains high standards with automated tooling:

```bash
# Linting with Ruff
uv run ruff check          # Check for issues
uv run ruff check --fix    # Auto-fix issues

# Type checking with MyPy
uv run mypy src/

# Format code
uv run ruff format

# Run all quality checks
uv run ruff check && uv run ruff format && uv run mypy src/
```

### Commit Standards

We use [Commitizen](https://commitizen-tools.github.io/commitizen/) for standardized commits:

```bash
# Create a commit interactively
cz commit  # or: cz c

# Bump version
cz bump

# Format examples:
# feat: add batch processing support
# fix: resolve cardinality sketch serialization
# docs: update installation instructions
# refactor: simplify analyzer logic
# test: add coverage for edge cases
```

## 🔧 Performance Optimization

DQX is optimized for performance through:

- **SQL Deduplication**: Batches and deduplicates SQL operations
- **Sketch Algorithms**: Memory-efficient approximate computations
- **Parallel Processing**: Multi-threaded batch analysis
- **State Merging**: Combines partial results from distributed processing
- **Smart Caching**: Reuses computed metrics across checks

## 📦 Dependencies

Core dependencies:
- **DuckDB**: High-performance analytical database
- **PyArrow**: Columnar data processing
- **DataSketches**: Probabilistic data structures
- **SymPy**: Symbolic mathematics
- **SQLAlchemy**: Database abstraction
- **Returns**: Functional programming utilities

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🗺️ Roadmap

- [ ] Streaming data source support
- [ ] Web UI for monitoring and alerting
- [ ] Data catalog integrations
- [ ] ML-based anomaly detection
- [ ] Complex event processing
- [ ] Cloud-native deployment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Add tests for new functionality
4. Ensure all tests pass (`uv run pytest`)
5. Run code quality checks (`uv run ruff check && uv run mypy src/`)
6. Commit your changes (`cz commit`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📞 Support

- Create an issue on GitHub
- Check the documentation
- Join community discussions
