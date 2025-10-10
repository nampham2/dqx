# DQX - Data Quality eXcellence

A high-performance, scalable data quality framework built on DuckDB and PyArrow for fast, efficient data validation and monitoring.

## 🚀 Features

- **Blazing Fast**: Powered by DuckDB's analytical engine for sub-second query performance
- **Memory Efficient**: Statistical sketching algorithms (HyperLogLog, DataSketches) for large datasets
- **Declarative API**: Intuitive symbolic expressions for data quality checks
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
@check(name="Order validation", tags=["critical"])
def validate_orders(mp, ctx):
    # Check for null values
    ctx.assert_that(mp.null_count("customer_id")).is_eq(0)

    # Validate price ranges with tolerance
    ctx.assert_that(mp.minimum("price")).is_geq(0.0, tol=0.01)
    ctx.assert_that(mp.average("price")).is_gt(10.0)

    # Multiple assertions on the same metric
    ctx.assert_that(mp.average("quantity")).where(
        name="Quantity should be reasonable"
    ).is_gt(0)
    ctx.assert_that(mp.average("quantity")).is_leq(100)

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
        print(f"{assertion.name}: {assertion._value}")
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
    name="Data completeness check",
    tags=["completeness", "critical"],
    datasets=["main_table"]  # Optional: specify required datasets
)
def check_completeness(mp, ctx):
    """Ensure key columns have minimal null values."""
    total_rows = mp.num_rows()

    for column in ["id", "name", "email"]:
        null_percentage = mp.null_count(column) / total_rows
        ctx.assert_that(null_percentage).where(
            name=f"{column} null percentage",
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
ctx.assert_that(metric).where(
    name="Critical business rule",
    severity="P0"  # Severity levels: "P0", "P1", "P2", "P3"
).is_geq(0)
```

### Multiple Assertions

To perform multiple validations on the same metric, create separate assertions:

```python
# Validate a ratio is within acceptable bounds
ratio = mp.average("price") / mp.average("tax")
ctx.assert_that(ratio).where(
    name="Price/tax ratio lower bound"
).is_geq(0.95)
ctx.assert_that(ratio).where(
    name="Price/tax ratio upper bound"
).is_leq(1.05)

# Complex validation with multiple conditions
revenue = mp.sum("revenue")
ctx.assert_that(revenue).where(
    name="Revenue is positive",
    severity="P0"
).is_positive()
ctx.assert_that(revenue).where(
    name="Revenue upper limit"
).is_lt(1000000)
ctx.assert_that(revenue).where(
    name="Revenue lower limit"
).is_geq(10000)

# Each assertion is evaluated independently, providing:
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
from dqx.common import SqlDataSource
import duckdb

class CustomDataSource:
    name = "custom_source"
    dialect = "duckdb"  # SQL dialect to use

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
    dod_ratio = current / yesterday
    ctx.assert_that(dod_ratio).where(
        name="Daily revenue change lower bound"
    ).is_geq(0.9)  # No more than 10% drop
    ctx.assert_that(dod_ratio).where(
        name="Daily revenue change upper bound"
    ).is_leq(1.1)  # No more than 10% increase

    # Week-over-week trend
    ctx.assert_that(current / last_week).where(
        name="Weekly revenue trend"
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
    ctx.assert_that(ratio).where(
        name="Production/Staging lower bound"
    ).is_geq(0.95, tol=0.01)
    ctx.assert_that(ratio).where(
        name="Production/Staging upper bound"
    ).is_leq(1.05, tol=0.01)
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

## 📊 Graph Implementation

DQX uses a sophisticated graph-based architecture to manage dependencies between checks, metrics, and assertions. The graph implementation follows a **Composite Pattern** with a **Visitor Pattern** for traversal, enabling efficient execution planning and dependency resolution.

### Graph Hierarchy

The graph is organized in a hierarchical structure with the following node types:

```
┌─────────────────────────────────────────────────────────────┐
│                         RootNode                            │
│              (Top-level verification suite)                 │
└─────────────────────────┬───────────────────────────────────┘
                          │ contains
                          ▼
         ┌─────────────────────────────────────┐
         │           CheckNode                 │
         │    (Data quality check)             │
         └──────────┬─────────────┬────────────┘
                    │             │ contains
         contains   │             ▼
                    │   ┌─────────────────────┐
                    │   │   SymbolNode        │
                    │   │ (Computed symbols)  │
                    │   └──────────┬──────────┘
                    │              │ contains
                    ▼              ▼
       ┌─────────────────────┐  ┌─────────────────────┐
       │   AssertionNode     │  │    MetricNode       │
       │ (Validation rules)  │  │ (Metric to compute) │
       └─────────────────────┘  └──────────┬──────────┘
                                           │ contains
                                           ▼
                                ┌─────────────────────┐
                                │   AnalyzerNode      │
                                │ (SQL/computation)   │
                                └─────────────────────┘
```

### Node Types

#### 1. **RootNode** (Composite)
- The top-level container for all verification checks
- Manages the entire graph and provides traversal methods
- Handles propagation of dataset information through the graph

#### 2. **CheckNode** (Composite)
- Represents an individual data quality check
- Contains assertions that define the check logic
- Can be tagged and labeled for organization
- Aggregates child assertion results to determine overall check status
- Does NOT directly manage symbols (delegated to AssertionNodes)

#### 3. **AssertionNode** (Leaf, SymbolStateObserver)
- Represents a specific validation rule to be evaluated
- Uses symbolic expressions (SymPy) for flexible comparisons
- Supports custom validators and severity levels
- Evaluates to Success or Failure based on computed values
- **Implements SymbolStateObserver** to track symbol state changes
- Validates dataset availability for symbols used in expressions
- Registers itself as observer for all symbols in its expression

#### 4. **SymbolNode** (Composite)
- Represents a computed value that can be used in assertions
- Links to one or more MetricNodes that provide the actual data
- Manages the retrieval function for accessing computed values

#### 5. **MetricNode** (Composite)
- Represents a metric that needs to be computed from data
- Tracks metric state: PENDING → PROVIDED/ERROR
- Links to AnalyzerNode for actual computation logic

#### 6. **AnalyzerNode** (Leaf)
- Contains the actual computation logic (SQL operations)
- Executed by the analyzer engine against data sources

### Graph Features

#### Dependency Resolution
The graph automatically resolves dependencies between metrics and assertions:
- Metrics are computed only once, even if used by multiple assertions
- The execution order is determined by the dependency graph
- Failed dependencies propagate failures to dependent nodes

#### State Management
Each node maintains its state throughout the execution:
- **PENDING**: Waiting for computation
- **PROVIDED**: Successfully computed/evaluated
- **ERROR**: Computation or validation failed

#### Symbol Management and Observer Pattern
The graph implements an observer pattern for symbol state notifications:
- **AssertionNode** implements `SymbolStateObserver` to track symbol states
- When a symbol's state changes (ready, success, error), assertions are notified
- Assertions validate that required datasets are available for their symbols
- Symbol errors are propagated to assertions, ensuring clear failure tracking

#### Traversal Methods
The RootNode provides convenient methods for graph traversal:
```python
# Get all assertions in the graph
for assertion in graph.assertions():
    print(f"{assertion.name}: {assertion._value}")

# Get all pending metrics for a dataset
for metric in graph.pending_metrics("orders"):
    print(f"Pending: {metric.spec.name}")

# Get all checks and their status
for check in graph.checks():
    print(f"{check.name}: {check._value}")
```

#### Dataset Propagation and Validation
The graph supports multi-dataset validation with validation at the assertion level:
1. Datasets are propagated from checks to assertions
2. Assertions validate that their symbols have required datasets available
3. Dataset mismatches result in clear error messages at the assertion level
4. CheckNode aggregates assertion results to determine overall check status

### Example Graph Construction

When you define a check like this:
```python
@check(name="Price validation")
def validate_prices(mp, ctx):
    ctx.assert_that(mp.average("price")).is_gt(0)
    ctx.assert_that(mp.maximum("price")).is_leq(1000)
```

The framework automatically constructs this graph structure:
```
RootNode
└── CheckNode("Price validation")
    ├── SymbolNode(average_price)
    │   └── MetricNode(Average("price"))
    │       └── AnalyzerNode(SQL: AVG(price))
    ├── SymbolNode(max_price)
    │   └── MetricNode(Maximum("price"))
    │       └── AnalyzerNode(SQL: MAX(price))
    ├── AssertionNode(average_price > 0)
    └── AssertionNode(max_price <= 1000)
```

This graph structure enables:
- Efficient execution (each metric computed once)
- Clear dependency tracking
- Granular error reporting
- Flexible execution strategies

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

# Run only tests marked with 'demo' tag
uv run pytest -m demo
```

This is useful for running a subset of tests that demonstrate specific functionality or are used for demo purposes. Tests can be marked with the `@pytest.mark.demo` decorator to include them in this test run.

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

### Pre-commit Hooks

This project uses pre-commit hooks to maintain code quality. Hooks run automatically before each commit.

#### Quick Setup

```bash
# Run the setup script (recommended)
./bin/setup-dev-env.sh

# Or manually install hooks
uv run pre-commit install
```

#### What Gets Checked

- **Code formatting**: Ruff automatically formats Python code
- **Linting**: Ruff checks for code quality issues
- **Type checking**: MyPy validates type annotations
- **Debug detection**: Catches forgotten print/breakpoint statements
- **File quality**: Trailing whitespace, file endings, large files
- **Security**: Detects accidentally committed private keys
- **Syntax validation**: Python, YAML, TOML, JSON files

#### Manual Usage

```bash
# Run on all files
./bin/run-hooks.sh --all

# Run on specific files
./bin/run-hooks.sh src/dqx/api.py tests/test_api.py

# Run only formatting hooks (fast)
./bin/run-hooks.sh --fix

# Skip slow hooks like mypy
./bin/run-hooks.sh --fast

# Skip hooks temporarily (not recommended)
git commit --no-verify -m "emergency fix"
```

#### Fixing Issues

If pre-commit blocks your commit:

1. **Read the error message** - it tells you exactly what's wrong
2. **Let hooks auto-fix** - many issues are fixed automatically
3. **Review changes** - check what was fixed with `git diff`
4. **Re-stage files** - `git add` the fixed files
5. **Commit again** - your commit should now succeed

Example:
```bash
$ git commit -m "feat: add new feature"
ruff.....................................................................Failed
- hook id: ruff-check
- exit code: 1
  Fixed 1 error:
  - src/dqx/new_feature.py:
    1 × I001 (unsorted-imports)

# Ruff fixed the import order. Check and re-commit:
$ git add src/dqx/new_feature.py
$ git commit -m "feat: add new feature"
```

#### VS Code Integration

VS Code will automatically use the same formatting rules if you have the Ruff extension installed:

1. Install the Ruff extension (ID: `charliermarsh.ruff`)
2. Your code will be formatted on save
3. Linting errors will appear inline

#### Performance Tips

- For faster commits during development:
  ```bash
  SKIP=mypy git commit -m "wip: quick save"  # Skip type checking
  ./bin/run-hooks.sh --fast                   # Run without mypy
  ```

- Pre-commit caches results, so subsequent runs on unchanged files are instant

#### CI/CD Integration

Pre-commit runs automatically in CI on pull requests. To run the same checks locally:
```bash
./bin/run-hooks.sh --all
```

### Development Setup

```bash
# Install development dependencies
uv sync --all-extras

# Install pre-commit hooks (recommended)
./bin/setup-dev-env.sh
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

## 📋 Contribution Guidelines

### Logging Best Practices

#### Lazy Rendering of Expensive Log Messages

When logging messages that require expensive computations (e.g., serialization, string formatting, or complex calculations), always use lazy evaluation to avoid performance overhead when the log level is disabled:

```python
# ❌ BAD: Expensive operation always executed
logger.debug(f"Processing data: {expensive_to_string_operation()}")
logger.debug("Stats: {}".format(json.dumps(large_dict, indent=2)))

# ✅ GOOD: Use isEnabledFor guard
if logger.isEnabledFor(logging.DEBUG):
    logger.debug("Processing data: %s", expensive_to_string_operation())

# ✅ GOOD: Use built-in % formatting (evaluated lazily)
logger.debug("Processing %d records from %s", len(records), dataset_name)
logger.debug("Metric value: %.4f", compute_expensive_metric())
```

#### Key Principles

1. **Use `%` formatting instead of f-strings or `.format()` for log messages**
   - The `%` formatting is lazily evaluated only when the log level is active
   - f-strings and `.format()` are always evaluated, even if logging is disabled

2. **Guard expensive computations with `isEnabledFor`**
   ```python
   # For complex debug information
   if logger.isEnabledFor(logging.DEBUG):
       debug_info = {
           "metrics": [m.to_dict() for m in metrics],
           "graph": graph.to_json(),
           "state": analyzer.get_debug_state()
       }
       logger.debug("Analysis state: %s", json.dumps(debug_info, indent=2))
   ```

3. **Common patterns in DQX**
   ```python
   # Logging in the analyzer
   logger.debug("Executing %d operations for dataset '%s'", len(ops), dataset_name)

   # Logging metric computations
   logger.debug("Computing metric %s with spec %r", metric_name, metric_spec)

   # Logging graph traversal
   if logger.isEnabledFor(logging.DEBUG):
       logger.debug("Graph structure:\n%s", graph.pretty_print())
   ```

4. **Performance impact**
   - Lazy logging can significantly improve performance in production where debug logging is typically disabled
   - Especially important in hot paths like the analyzer loop or graph traversal
   - Critical for operations that process large datasets or run frequently

### Other Guidelines

- Follow PEP 8 and the existing code style
- Write comprehensive tests for all new features
- Update documentation when adding new functionality
- Use type hints for all function signatures
- Keep functions focused and single-purpose
- Document complex algorithms and design decisions

## 🏆 Recent Improvements

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
