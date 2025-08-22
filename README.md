# DQX - Data Quality eXecution Engine

A high-performance, scalable data quality framework built on DuckDB and PyArrow for fast, efficient data validation and monitoring.

## Overview

DQX (`D`ata `Q`uality e`X`cellent) is a modern data quality framework designed for production-scale data validation. It provides a declarative API for defining data quality checks, supports multiple data sources, and leverages statistical sketching algorithms for memory-efficient analysis of large datasets.

### Key Features

- **High Performance**: Built on DuckDB for blazing-fast analytical queries
- **Scalable**: Supports batch processing with threading and memory-efficient sketching algorithms
- **Declarative**: Intuitive API for defining data quality checks using symbolic expressions
- **Multi-Source**: Support for PyArrow tables, Parquet files, BigQuery, and more
- **Extensible**: Plugin architecture for custom metrics and data sources
- **Production Ready**: Built-in persistence, monitoring, and error handling

## Architecture

DQX follows a modular architecture with clear separation of concerns:

- **Specs**: Metric specifications (row count, averages, cardinality, etc.)
- **Ops**: SQL and sketch-based operations for data analysis
- **States**: Serializable states for metric computation and merging
- **Analyzer**: Execution engine that runs operations against data sources
- **API**: High-level verification suite and check definitions
- **Extensions**: Data source adapters and custom functionality

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd dqx

# Install with uv (recommended)
uv install

# Or with pip
pip install -e .
```

### Basic Usage

```python
import datetime as dt
from dqx.api import VerificationSuite, check
from dqx.extensions.pyarrow_ds import ArrowDataSource
from dqx.orm.repositories import InMemoryMetricDB
from dqx.common import ResultKey

# Define data quality checks
@check(datasets=["orders"])
def order_validation(mp, ctx):
    # Check for null values
    ctx.assert_that(mp.null_count("customer_id")).is_eq(0)
    
    # Validate price ranges
    ctx.assert_that(mp.minimum("price")).is_gt(0)
    ctx.assert_that(mp.average("price")).is_geq(10.0)
    
    # Check data freshness (day-over-day comparison)
    ctx.assert_that(mp.ext.day_over_day(specs.NumRows())).is_geq(0.8)

# Set up data source and run checks
db = InMemoryMetricDB()
data_source = ArrowDataSource(your_arrow_table)
key = ResultKey(yyyy_mm_dd=dt.date.today(), tags={"environment": "prod"})

suite = VerificationSuite([order_validation], db, name="Order Quality Suite")
context = suite.run({"orders": data_source}, key)

# View results
context._graph.inspect()
```

## Supported Metrics

DQX provides a comprehensive set of built-in metrics:

### Basic Statistics
- `NumRows()` - Row count
- `Average(column)` - Mean value
- `Sum(column)` - Sum of values
- `Minimum(column)` - Minimum value
- `Maximum(column)` - Maximum value
- `Variance(column)` - Sample variance
- `First(column)` - First non-null value

### Data Quality Metrics
- `NullCount(column)` - Count of null values
- `NegativeCount(column)` - Count of negative values
- `ApproxCardinality(column)` - Estimated unique values using HyperLogLog

### Advanced Metrics
- `ext.day_over_day(metric)` - Day-over-day comparison
- `ext.stddev(metric, lag, window)` - Standard deviation over time windows

## Data Sources

DQX supports multiple data source types through its extension system:

### PyArrow
```python
from dqx.extensions.pyarrow_ds import ArrowDataSource, ArrowBatchDataSource

# Single table
ds = ArrowDataSource(arrow_table)

# Batch processing for large datasets
batch_ds = ArrowBatchDataSource.from_parquets(["file1.parquet", "file2.parquet"])
```

### BigQuery
```python
from dqx.extensions.bigquery_ds import BigQueryDataSource

ds = BigQueryDataSource(
    project="my-project",
    dataset="my_dataset", 
    table="my_table"
)
```

### Custom Data Sources
Implement the `DuckDataSource` protocol:

```python
class CustomDataSource:
    name = "custom"
    analyzer_class = Analyzer
    
    @property
    def cte(self) -> str:
        return "SELECT * FROM my_custom_table"
        
    def query(self, query: str) -> duckdb.DuckDBPyRelation:
        return duckdb.query(query)
```

## Advanced Features

### Batch Processing
For large datasets, DQX supports memory-efficient batch processing:

```python
# Enable threading for parallel batch processing
context = suite.run(data_sources, key, threading=True)
```

### Time-based Analysis
Create checks that compare metrics across time:

```python
@check
def trend_analysis(mp, ctx):
    current = mp.average("revenue")
    previous = mp.average("revenue", key=ctx.key.lag(1))
    
    # Revenue should not drop more than 10%
    ctx.assert_that(current / previous).is_geq(0.9)
```

### Cross-dataset Validation
Compare metrics across different datasets:

```python
@check
def cross_dataset_consistency(mp, ctx):
    prod_count = mp.num_rows(datasets=["production"])
    staging_count = mp.num_rows(datasets=["staging"])
    
    # Counts should be similar (within 5%)
    ratio = prod_count / staging_count
    ctx.assert_that(sp.Abs(ratio - 1.0)).is_lt(0.05)
```

### Custom Assertions
DQX provides flexible assertion methods:

```python
ctx.assert_that(metric).is_eq(100)           # Equal to
ctx.assert_that(metric).is_gt(0)             # Greater than
ctx.assert_that(metric).is_geq(0)            # Greater than or equal
ctx.assert_that(metric).is_lt(1000)          # Less than  
ctx.assert_that(metric).is_leq(1000)         # Less than or equal
ctx.assert_that(metric).is_positive()        # Positive values
ctx.assert_that(metric).is_negative()        # Negative values
```

## Performance

DQX is optimized for performance through several techniques:

- **SQL Optimization**: Deduplicates and batches SQL operations
- **Sketch Algorithms**: Memory-efficient approximate algorithms for cardinality
- **Batch Processing**: Processes large datasets in configurable chunks
- **Parallel Execution**: Multi-threaded analysis for independent operations
- **State Merging**: Combines partial results from distributed processing

## Persistence

Metrics can be persisted for historical analysis and trend detection:

```python
from dqx.orm.repositories import MetricDB

# Configure persistence
db = MetricDB(connection_string="sqlite:///metrics.db")

# Results are automatically persisted
suite = VerificationSuite(checks, db, name="Production Suite")
```

## Development

### Running Tests
```bash
uv run pytest
```

### Code Quality
```bash
# Run linting
uv run ruff check

# Auto-fix issues
uv run ruff check --fix

# Type checking
uv run mypy src/
```

### Commit Messages

This project uses [Commitizen](https://commitizen-tools.github.io/commitizen/) for standardized commit messages and automated versioning.

```bash
# Install commitizen (if not already installed)
pip install commitizen

# Create a commit interactively
cz commit

# Or use the shorthand
cz c

# Bump version and create changelog
cz bump

# Check version
cz version
```

**Commit Message Format:**
- `feat:` - New features
- `fix:` - Bug fixes  
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

Example:
```bash
feat(analyzer): add support for parallel batch processing
fix(specs): resolve cardinality sketch serialization issue  
docs(readme): update installation instructions
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality  
4. Ensure all tests pass
5. Submit a pull request

## Dependencies

- **DuckDB**: High-performance analytical database engine
- **PyArrow**: Columnar data processing
- **DataSketches**: Probabilistic data structures
- **SymPy**: Symbolic mathematics for expressions
- **SQLAlchemy**: Database abstraction layer
- **Returns**: Functional programming utilities

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Roadmap

- [ ] Support for streaming data sources
- [ ] Web UI for monitoring and alerting  
- [ ] Integration with data catalogs
- [ ] Machine learning-based anomaly detection
- [ ] Support for complex event processing
- [ ] Cloud-native deployment options

## Support

For questions and support:
- Create an issue on GitHub
- Check the documentation
- Join our community discussions
