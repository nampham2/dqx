# DQX - Data Quality eXcellence: Detailed Design Document

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [Symbolic Metrics](#symbolic-metrics)
5. [Design Patterns](#design-patterns)
6. [Design Decisions](#design-decisions)
7. [Dataset Specification and Validation](#dataset-specification-and-validation)
8. [API Design](#api-design)
9. [Severity Levels in Checks and Assertions](#severity-levels-in-checks-and-assertions)
10. [Code Usage and Examples](#code-usage-and-examples)
11. [Performance Considerations](#performance-considerations)
12. [Extension Points](#extension-points)
13. [Future Improvements and Roadmap](#future-improvements-and-roadmap)

## Overview

DQX is a high-performance data quality framework built on DuckDB and PyArrow, designed to provide fast, scalable, and declarative data validation capabilities. The framework employs a graph-based architecture to manage dependencies between data quality checks, metrics, and assertions efficiently.

### Key Design Goals
- **Performance**: Sub-second query performance on large datasets
- **Scalability**: Handle TB-scale data through batch processing and statistical sketching
- **Declarative API**: Intuitive symbolic expressions for data quality checks
- **Extensibility**: Plugin architecture for custom metrics and data sources
- **Production-Ready**: Built-in persistence, error handling, and monitoring

## Architecture

### High-Level Architecture

```mermaid
graph TB
    subgraph "User Interface Layer"
        API[Verification API]
        Check[Check Decorator]
        Builder[Suite Builder]
    end

    subgraph "Orchestration Layer"
        Graph[Dependency Graph]
        Context[Execution Context]
        Suite[Verification Suite]
    end

    subgraph "Computation Layer"
        Provider[Metric Provider]
        Analyzer[SQL Analyzer]
        Specs[Metric Specs]
    end

    subgraph "Data Layer"
        DS[Data Sources]
        DB[Metric Database]
        States[Metric States]
    end

    subgraph "Foundation Layer"
        Ops[SQL Operations]
        Sketches[Statistical Sketches]
    end

    API --> Suite
    Check --> Context
    Builder --> Suite

    Suite --> Graph
    Suite --> Provider
    Context --> Graph

    Provider --> Specs
    Graph --> Analyzer
    Specs --> Ops

    Analyzer --> DS
    Provider --> DB
    States --> DB
```

### Component Interaction Flow

```mermaid
sequenceDiagram
    participant User
    participant Suite
    participant Graph
    participant Provider
    participant Analyzer
    participant DataSource
    participant MetricDB

    User->>Suite: run(datasources, key)
    Suite->>Provider: Create checks
    Suite->>Graph: Build dependency graph

    loop For each dataset
        Suite->>Graph: Get pending metrics
        Suite->>Analyzer: analyze(datasource, metrics)
        Analyzer->>DataSource: Execute SQL queries
        DataSource-->>Analyzer: Results
        Analyzer->>MetricDB: Persist metrics
        Analyzer-->>Suite: Analysis complete
    end

    Suite->>Graph: Evaluate assertions
    Graph->>Provider: Retrieve metric values
    Provider->>MetricDB: Get stored metrics
    MetricDB-->>Provider: Metric values
    Provider-->>Graph: Computed values
    Graph-->>Suite: Evaluation results
    Suite-->>User: Context with results
```

## Core Components

### 1. Graph System (`graph.py`)

The graph system implements a hierarchical node structure using Composite and Visitor patterns:

```mermaid
graph TD
    Root["RootNode<br/>(Verification Suite)"]
    Check["CheckNode<br/>(Quality Check)"]
    Assertion["AssertionNode<br/>(Validation)"]
    Symbol["SymbolNode<br/>(Expression)"]
    Metric["MetricNode<br/>(Computation)"]
    Analyzer["AnalyzerNode<br/>(SQL Execution)"]

    Root --> Check
    Check --> Assertion
    Check --> Symbol
    Symbol --> Metric
    Metric --> Analyzer
```

#### Node Types and Responsibilities

- **RootNode**: Top-level container managing the entire verification suite
- **CheckNode**: Represents individual data quality checks with labels and tags
- **AssertionNode**: Leaf nodes containing validation rules
- **SymbolNode**: Computed values that can be used in assertions
- **MetricNode**: Metrics to be computed from data
- **AnalyzerNode**: SQL operations for metric computation

### 2. API Layer (`api.py`)

The API layer provides the user-facing interface with several key components:

```mermaid
graph TB
    Suite[VerificationSuite]
    Builder[VerificationBuilder]
    Context[Context]
    Assert[SymbolicAssert]
    States[GraphStates]

    Builder --> Suite
    Suite --> Context
    Context --> Assert
    Context --> States
```

### 3. Metric Specifications (`specs.py`)

Metric specifications define the available metrics and their computation logic:

```mermaid
classDiagram
    class MetricSpec {
        <<Protocol>>
        +metric_type: MetricType
        +name() str
        +analyzers() Sequence[Op]
    }

    class NumRows {
        +metric_type: "NumRows"
    }

    class First {
        +column: str
        +metric_type: "First"
    }

    class Average {
        +column: str
        +metric_type: "Average"
    }

    class Variance {
        +column: str
        +metric_type: "Variance"
    }

    class Minimum {
        +column: str
        +metric_type: "Minimum"
    }

    class Maximum {
        +column: str
        +metric_type: "Maximum"
    }

    class Sum {
        +column: str
        +metric_type: "Sum"
    }

    class NullCount {
        +column: str
        +metric_type: "NullCount"
    }

    class NegativeCount {
        +column: str
        +metric_type: "NegativeCount"
    }

    class ApproxCardinality {
        +column: str
        +metric_type: "ApproxCardinality"
    }

    MetricSpec <|.. NumRows
    MetricSpec <|.. First
    MetricSpec <|.. Average
    MetricSpec <|.. Variance
    MetricSpec <|.. Minimum
    MetricSpec <|.. Maximum
    MetricSpec <|.. Sum
    MetricSpec <|.. NullCount
    MetricSpec <|.. NegativeCount
    MetricSpec <|.. ApproxCardinality
```

### 4. Provider System (`provider.py`)

The provider system manages metric retrieval and symbolic metric creation:

```mermaid
graph LR
    MP[MetricProvider] --> SM[SymbolicMetric]
    MP --> DB[MetricDB]
    SM --> MS[MetricSpec]
    SM --> RKP[ResultKeyProvider]
```

### 5. Symbol Table System (`symbol_table.py`)

The symbol table manages the registration and evaluation of symbols used in data quality checks:

```mermaid
classDiagram
    class SymbolEntry {
        +symbolic_metric: SymbolicMetric
        +dataset: str | None
        +result_key: ResultKey | None
        +state: SymbolState
        +value: Maybe[Result[float, str]]
        +symbol: Symbol (property)
        +name: str (property)
        +retrieval_fn: RetrievalFn (property)
        +dependencies: list[Dependency] (property)
        +metric_spec: MetricSpec | None (property)
        +ops: list[Op] (property)
    }

    class SymbolTable {
        -_entries: dict[Symbol, SymbolEntry]
        -_by_dataset: dict[str, list[Symbol]]
        -_by_metric: dict[str, list[Symbol]]
        -_evaluation_order: list[Symbol]
        -_by_check: dict[str, list[Symbol]]
        +register(entry: SymbolEntry)
        +register_from_provider(symbolic_metric, key)
        +evaluate_symbol(symbol, key): Result[float, str]
        +get_pending(dataset?): list[SymbolEntry]
        +get_ready(dataset?): list[SymbolEntry]
    }

    class SymbolicMetric {
        +name: str
        +symbol: Symbol
        +fn: RetrievalFn
        +key_provider: ResultKeyProvider
        +dependencies: list[tuple[MetricSpec, ResultKeyProvider]]
        +datasets: list[str]
    }

    SymbolEntry --> SymbolicMetric
    SymbolTable --> SymbolEntry
```

#### Design Decision: Elimination of Redundancy

**Previous Design Issue**: The `SymbolEntry` class originally stored `metric_spec` and `ops` fields directly, duplicating information already available through `symbolic_metric.dependencies`.

**Solution**: Implemented computed properties using Python's `@property` decorator:

```python
@property
def metric_spec(self) -> MetricSpec | None:
    """Get the metric spec from dependencies."""
    if self.symbolic_metric.dependencies:
        return self.symbolic_metric.dependencies[0][0]
    return None

@property
def ops(self) -> list[Op]:
    """Get the analyzer operations from metric spec."""
    if self.metric_spec:
        return list(self.metric_spec.analyzers)
    return []
```

**Benefits**:
- **Single Source of Truth**: Data is stored only in `SymbolicMetric.dependencies`
- **Backward Compatibility**: Existing code using `entry.metric_spec` and `entry.ops` continues to work
- **Data Consistency**: Eliminates possibility of `metric_spec` and `ops` being out of sync with dependencies
- **Reduced Memory**: Slight reduction in memory usage by avoiding redundant storage
- **Cleaner Constructor**: `SymbolEntry` constructor is simpler with fewer parameters

This refactoring demonstrates the principle of **DRY (Don't Repeat Yourself)** and improves maintainability without breaking existing APIs.

## Symbolic Metrics

### Introduction to Symbolic Metrics

Symbolic metrics are a core abstraction in DQX that bridge the gap between data quality metrics and mathematical expressions. By leveraging SymPy (Python's symbolic mathematics library), DQX allows users to compose complex data quality assertions using simple mathematical operations on metrics. Each metric is represented as a symbolic variable that can be combined, transformed, and evaluated lazily.

### Architecture of Symbolic Metrics

```mermaid
graph TB
    subgraph "Symbolic Layer"
        MP[MetricProvider]
        SM[SymbolicMetric]
        Symbol[SymPy Symbol]
        Expr[Symbolic Expression]
    end

    subgraph "Computation Layer"
        RF[RetrievalFn]
        Compute[Compute Module]
        DB[(MetricDB)]
    end

    subgraph "Evaluation"
        Key[ResultKey]
        Value[Float Value]
    end

    MP -->|creates| SM
    SM -->|contains| Symbol
    SM -->|references| RF
    Symbol -->|combines into| Expr
    RF -->|uses| Compute
    Compute -->|queries| DB
    Expr -->|evaluate with| Key
    Key -->|produces| Value
```

### Creation of Symbolic Metrics

Symbolic metrics are created through the `MetricProvider` class, which acts as a factory for generating symbolic representations of metrics:

```python
@dataclass
class SymbolicMetric:
    name: str                          # Human-readable name
    symbol: sp.Symbol                  # SymPy symbol (e.g., x_1, x_2)
    fn: RetrievalFn                    # Function to retrieve actual value
    key_provider: ResultKeyProvider    # Handles temporal keys
    dependencies: list[Dependency]     # Metric dependencies
    datasets: list[str]               # Associated datasets
```

#### Creation Process:

1. **Symbol Generation**: Each metric gets a unique SymPy symbol (x_1, x_2, etc.)
   ```python
   def _next_symbol(self, prefix: str = "x") -> sp.Symbol:
       with self._mutex:
           self._curr_index += 1
           return sp.Symbol(f"{prefix}_{self._curr_index}")
   ```

2. **Registration**: The symbol is registered with its metadata
   ```python
   def metric(self, metric: MetricSpec, key: ResultKeyProvider = ResultKeyProvider(),
             datasets: list[str] | None = None) -> sp.Symbol:
       self._register(
           sym := self._next_symbol(),
           name=metric.name,
           fn=partial(compute.simple_metric, self._db, metric, key),
           key=key,
           dependencies=[(metric, key)],
           datasets=datasets or [],
       )
       return sym
   ```

3. **API Methods**: Convenience methods create common metrics
   ```python
   # Basic metric creation
   avg_price = mp.average("price")        # Returns sp.Symbol('x_1')
   total_items = mp.sum("quantity")       # Returns sp.Symbol('x_2')

   # Complex expressions
   revenue = avg_price * total_items      # Returns x_1 * x_2
   ```

### How Symbol Values are Collected

The collection of symbol values follows a lazy evaluation pattern:

```mermaid
sequenceDiagram
    participant User
    participant Context
    participant Provider
    participant SymbolicMetric
    participant Compute
    participant MetricDB

    User->>Context: assert_that(symbol)
    Context->>Provider: evaluate(symbol, key)
    Provider->>SymbolicMetric: get retrieval function
    SymbolicMetric->>Compute: simple_metric(db, spec, key)
    Compute->>MetricDB: get_metric_value(spec, key)
    MetricDB-->>Compute: value or None
    Compute-->>Provider: Result[float, str]
    Provider-->>Context: evaluation result
```

#### Value Collection Process:

1. **Symbol Resolution**: When an assertion needs evaluation, the symbol is resolved to its `SymbolicMetric`
   ```python
   def evaluate(self, symbol: sp.Symbol, key: ResultKey) -> Result[float, str]:
       return self._symbol_index[symbol].fn(key)
   ```

2. **Retrieval Function Execution**: The stored retrieval function is called
   ```python
   def simple_metric(db: MetricDB, metric: MetricSpec, key_provider: ResultKeyProvider,
                    nominal_key: ResultKey) -> Result[float, str]:
       key = key_provider.create(nominal_key)
       value = db.get_metric_value(metric, key)
       return maybe_to_result(value, f"Metric {metric.name} not found")
   ```

3. **Complex Expression Evaluation**: For composite expressions, SymPy handles the mathematical operations
   ```python
   # If expression is: x_1 * x_2 + x_3
   # SymPy evaluates each symbol and computes the result
   ```

### Benefits of Symbolic Metrics

#### 1. **Declarative Mathematical Operations**
Users can express complex data quality rules using natural mathematical syntax:
```python
# Natural expression of business rules
revenue = mp.sum("price") * mp.sum("quantity")
profit_margin = (revenue - mp.sum("cost")) / revenue

ctx.assert_that(profit_margin).is_gt(0.2)  # 20% minimum margin
```

#### 2. **Lazy Evaluation**
Metrics are only computed when needed, optimizing performance:
```python
# These create symbols, not compute values
metric1 = mp.average("col1")  # No computation yet
metric2 = mp.average("col2")  # No computation yet

# Only computed if assertion fails or inspection requested
ctx.assert_that(metric1 / metric2).is_lt(2.0)
```

#### 3. **Composability and Reusability**
Metrics can be composed into higher-level abstractions:
```python
def revenue_per_user(mp: MetricProvider) -> sp.Symbol:
    """Reusable metric definition."""
    total_revenue = mp.sum("revenue")
    unique_users = mp.approx_cardinality("user_id")
    return total_revenue / unique_users

# Use in multiple checks
ctx.assert_that(revenue_per_user(mp)).is_gt(100)
```

#### 4. **Time-Travel Capabilities**
Symbolic metrics support temporal operations elegantly:
```python
current = mp.average("response_time")
yesterday = mp.average("response_time", key=ctx.key.lag(1))
weekly_avg = mp.average("response_time", key=ctx.key.lag(7))

# Express temporal rules symbolically
daily_change = (current - yesterday) / yesterday
ctx.assert_that(daily_change).is_between(-0.1, 0.1)  # ±10% change
```

#### 5. **Type Safety and Expression Validation**
SymPy provides mathematical validation of expressions:
```python
# SymPy ensures mathematical correctness
sqrt_variance = sp.sqrt(mp.variance("amount"))  # Valid
log_count = sp.log(mp.num_rows())              # Valid for positive values
```

### Simplicity Through Abstraction

Symbolic metrics dramatically simplify complex data quality checks:

#### Before (Traditional Approach):
```python
# Manual computation and comparison
total_revenue = db.query("SELECT SUM(revenue) FROM sales WHERE date = ?", date)
total_cost = db.query("SELECT SUM(cost) FROM sales WHERE date = ?", date)
if total_revenue is None or total_cost is None:
    raise ValueError("Missing metrics")
profit = total_revenue - total_cost
margin = profit / total_revenue if total_revenue > 0 else 0
if margin < 0.2:
    raise AssertionError(f"Profit margin {margin} below threshold")
```

#### After (Symbolic Metrics):
```python
revenue = mp.sum("revenue")
cost = mp.sum("cost")
margin = (revenue - cost) / revenue
ctx.assert_that(margin).is_gt(0.2)
```

The symbolic approach:
- Eliminates boilerplate code
- Handles null checks automatically
- Provides better error messages
- Enables metric reuse across checks

### Advanced Features

#### 1. **Extended Metrics**
The `ExtendedMetricProvider` adds derived metrics:
```python
# Day-over-day calculations
dod_change = mp.ext.day_over_day(specs.Sum("revenue"))

# Statistical calculations over time windows
revenue_stddev = mp.ext.stddev(
    specs.Sum("revenue"),
    lag=30,  # Start 30 days ago
    n=30     # 30-day window
)
```

#### 2. **Custom Symbolic Functions**
Users can create custom symbolic operations:
```python
def zscore(value: sp.Symbol, mean: sp.Symbol, stddev: sp.Symbol) -> sp.Symbol:
    """Calculate z-score symbolically."""
    return (value - mean) / stddev

current_revenue = mp.sum("revenue")
avg_revenue = mp.average("revenue", key=ctx.key.lag(30))
std_revenue = mp.ext.stddev(specs.Sum("revenue"), lag=30, n=30)

ctx.assert_that(zscore(current_revenue, avg_revenue, std_revenue)).is_between(-3, 3)
```

### Drawbacks and Limitations

#### 1. **Debugging Complexity**
Symbolic expressions can be harder to debug than direct computations:
```python
# This creates: ((x_1 - x_2) / x_3) * x_4
complex_metric = ((mp.sum("a") - mp.sum("b")) / mp.average("c")) * mp.num_rows()

# Debugging requires understanding symbol mapping:
# x_1 -> sum(a), x_2 -> sum(b), x_3 -> average(c), x_4 -> num_rows
```

**Mitigation**: Use descriptive intermediate variables and logging:
```python
total_a = mp.sum("a")
total_b = mp.sum("b")
avg_c = mp.average("c")
rows = mp.num_rows()
complex_metric = ((total_a - total_b) / avg_c) * rows
```

#### 2. **Performance Overhead**
Symbolic manipulation adds computational overhead:
- Symbol creation and registration
- Expression parsing and simplification
- Indirect function calls during evaluation

**Mitigation**: DQX optimizes by:
- Caching computed metric values
- Batch processing multiple metrics
- Reusing symbols across assertions

#### 3. **Limited Mathematical Operations**
Not all mathematical operations are meaningful for data metrics:
```python
# Mathematically valid but semantically questionable
sin_of_count = sp.sin(mp.num_rows())  # What does this mean?

# Operations limited by metric nature
sqrt_negative = sp.sqrt(mp.sum("profit"))  # Fails if profit < 0
```

#### 4. **Memory Overhead**
Each symbolic metric maintains metadata:
- Symbol-to-metric mapping
- Retrieval function closures
- Dependency tracking

For suites with thousands of metrics, this can add up.

#### 5. **Learning Curve**
Users need to understand:
- Symbolic vs. concrete evaluation
- SymPy expression syntax
- Lazy evaluation semantics

### Best Practices

1. **Use Intermediate Variables for Clarity**
   ```python
   # Good: Clear variable names
   daily_revenue = mp.sum("revenue")
   daily_orders = mp.num_rows()
   revenue_per_order = daily_revenue / daily_orders

   # Avoid: Inline complex expressions
   ctx.assert_that(mp.sum("revenue") / mp.num_rows()).is_gt(100)
   ```

2. **Handle Division by Zero**
   ```python
   denominator = mp.num_rows()
   # Use conditional expressions
   safe_average = sp.Piecewise(
       (mp.sum("value") / denominator, denominator > 0),
       (0, True)
   )
   ```

3. **Document Complex Expressions**
   ```python
   # Calculate customer lifetime value (CLV)
   # CLV = (Average Order Value) × (Purchase Frequency) × (Customer Lifetime)
   aov = mp.sum("revenue") / mp.num_rows()
   frequency = mp.num_rows() / mp.approx_cardinality("customer_id")
   lifetime_days = 365  # Assumed 1-year lifetime
   clv = aov * frequency * lifetime_days
   ```

### Future Enhancements

Potential improvements to the symbolic metric system:

1. **Symbolic Differentiation**: Automatic sensitivity analysis
2. **Constraint Solving**: Find metric values that satisfy constraints
3. **Symbolic Optimization**: Minimize/maximize metric expressions
4. **Probabilistic Reasoning**: Handle uncertainty in metrics
5. **Visual Expression Builder**: GUI for creating complex expressions

### 6. Analyzer System (`analyzer.py`)

The analyzer executes SQL operations and manages batch processing:

```mermaid
graph TD
    A[Analyzer] --> B[Batch Processing]
    A --> S[SQL Generation]
    A --> D[DuckDB Execution]
    B --> T[Threading Pool]
    S --> O[Operation Deduplication]
```

## Design Patterns

### 1. Composite Pattern
Used in the graph system to create a tree structure of nodes:
- **CompositeNode**: Can have children (RootNode, CheckNode, SymbolNode, MetricNode)
- **LeafNode**: Cannot have children (AssertionNode, AnalyzerNode)

### 2. Visitor Pattern
Enables traversal of the graph without modifying node classes:
```python
class NodeVisitor(Protocol):
    def visit(self, node: BaseNode) -> Any: ...

class GraphTraverser(NodeVisitor):
    def visit(self, node: BaseNode) -> None:
        # Process node and continue traversal
```

### 3. Builder Pattern
Used for constructing verification suites:
```python
suite = (VerificationSuiteBuilder("Suite Name", db)
         .add_check(check1)
         .add_checks([check2, check3])
         .build())
```

### 4. Strategy Pattern
Metric specifications act as strategies for computing different metrics:
- Each MetricSpec defines its own computation strategy
- Analyzers are selected based on the metric type

### 5. Protocol Pattern
Python protocols define interfaces without inheritance:
```python
@runtime_checkable
class MetricSpec(Protocol):
    metric_type: MetricType
    @property
    def name(self) -> str: ...
    @property
    def analyzers(self) -> Sequence[ops.Op]: ...
```

## Design Decisions

### 1. Graph-Based Architecture
**Decision**: Use a dependency graph to manage checks, metrics, and assertions.

**Rationale**:
- Enables efficient execution planning
- Avoids redundant metric computation
- Provides clear dependency tracking
- Supports complex validation scenarios

### 2. Symbolic Expressions
**Decision**: Use SymPy for symbolic expression handling in assertions.

**Rationale**:
- Enables mathematical operations on metrics
- Provides a declarative API
- Supports complex formulas
- Allows for symbolic simplification

### 3. Statistical Sketching
**Decision**: Use HyperLogLog and DataSketches for approximate computations.

**Rationale**:
- Memory-efficient for large datasets
- Provides configurable accuracy (99.9%)
- Enables distributed computation
- Supports incremental updates

### 4. SQL-Based Computation
**Decision**: Generate SQL queries for metric computation using DuckDB.

**Rationale**:
- Leverages DuckDB's columnar processing
- Enables batch computation
- Provides consistent computation semantics
- Supports various data formats

### 5. Fluent Assertion API
**Decision**: Implement chainable assertion methods.

**Rationale**:
- Improves readability
- Enables multiple validations on same metric
- Provides granular error reporting
- Supports progressive validation

### 6. Separation of Concerns
**Decision**: Separate display logic from graph logic.

**Rationale**:
- Improves maintainability
- Enables different display formats
- Reduces coupling
- Simplifies testing

## Dataset Specification and Validation

### Overview

DQX supports flexible dataset specification at multiple levels to enable both simple single-dataset validations and complex cross-dataset comparisons. The framework employs a hybrid approach that maintains backward compatibility while providing clear rules for dataset usage.

### Dataset Definition Levels

Datasets can be specified at three levels in the validation hierarchy:

1. **Check Level** - Using the `@check(datasets=["ds1", "ds2"])` decorator
2. **Symbol/Metric Level** - Using `mp.average("column", datasets=["ds1"])`
3. **Assertion Level** - Implicitly inherited from parent check

### The Hybrid Approach

The hybrid approach provides sensible defaults while maintaining flexibility for complex scenarios:

#### Rules for Single Dataset

When only one dataset is provided to `suite.run()`:
- **No explicit dataset specification required** at any level
- All metrics automatically use the single available dataset
- Existing single-dataset suites work without modification

```python
# Single dataset - simple and clean
@check
def validate_orders(mp: MetricProvider, ctx: Context) -> None:
    # No dataset specification needed
    avg_price = mp.average("price")
    total_orders = mp.num_rows()

    ctx.assert_that(avg_price).is_gt(0)
    ctx.assert_that(total_orders).is_gt(1000)

# Run with single dataset
suite.run({"orders": orders_ds}, key)
```

#### Rules for Multiple Datasets

When multiple datasets are provided to `suite.run()`:
- **Explicit dataset specification required** at check or metric level
- Check-level datasets act as a filter/constraint
- Metric-level datasets override check-level specifications
- Clear error messages guide users when specification is missing

```python
# Multiple datasets - explicit specification required

# Option 1: Specify at check level (constrains entire check)
@check(datasets=["production"])
def validate_production_data(mp: MetricProvider, ctx: Context) -> None:
    # All metrics in this check use "production" dataset
    avg_price = mp.average("price")
    null_count = mp.null_count("customer_id")

    ctx.assert_that(avg_price).is_gt(0)
    ctx.assert_that(null_count).is_eq(0)

# Option 2: Specify at metric level (for cross-dataset checks)
@check  # No dataset constraint - can access all datasets
def compare_environments(mp: MetricProvider, ctx: Context) -> None:
    # Explicit dataset specification for each metric
    prod_revenue = mp.sum("revenue", datasets=["production"])
    staging_revenue = mp.sum("revenue", datasets=["staging"])

    # Cross-dataset comparison
    revenue_diff = sp.Abs(prod_revenue - staging_revenue)
    ctx.assert_that(revenue_diff / prod_revenue).where(
        name="Production-Staging revenue difference"
    ).is_lt(0.01)  # Less than 1% difference

# Run with multiple datasets
suite.run({
    "production": prod_ds,
    "staging": staging_ds
}, key)
```

### Dataset Propagation and Validation

The framework validates dataset consistency at collection time using the DatasetImputationVisitor:

```mermaid
graph TD
    Visitor[DatasetImputationVisitor<br/>available_datasets: list[str]]
    Check[CheckNode<br/>datasets: list[str] or None]
    Symbol[SymbolicMetric<br/>datasets: list[str]]
    Assertion[AssertionNode<br/>datasets: list[str]]

    Visitor -->|validates/imputes| Check
    Visitor -->|processes symbols in| Assertion
    Symbol -->|validates against| Check
```

#### Validation Rules

1. **Check Validation**: If a check specifies datasets, they must be a subset of provided datasets
2. **Symbol Validation**: If a symbol specifies a dataset, it must be present in the check's datasets (or provided datasets if check has none)
3. **Assertion Inheritance**: Assertions inherit datasets from their parent check
4. **Consistency Check**: All dataset references are validated before execution begins

### Error Handling and Messages

The framework provides clear, actionable error messages:

```python
# Error when dataset not specified with multiple datasets
DQXError: "Metric 'average(price)' requires explicit dataset specification when multiple datasets are provided.
          Available datasets: ['production', 'staging'].
          Specify using: mp.average('price', datasets=['production'])"

# Error when invalid dataset specified
DQXError: "Check 'validate_data' specifies dataset 'testing' which is not provided.
          Available datasets: ['production', 'staging']"

# Error when symbol dataset doesn't match check constraint
DQXError: "Metric in check 'production_only' requests dataset 'staging',
          but check is constrained to datasets: ['production']"
```

### Common Patterns and Best Practices

#### 1. Single-Purpose Checks
Constrain checks to specific datasets for clarity:

```python
@check(datasets=["orders"], name="Order validation")
def validate_orders(mp: MetricProvider, ctx: Context) -> None:
    # All metrics automatically use "orders" dataset
    ctx.assert_that(mp.null_count("order_id")).is_eq(0)
    ctx.assert_that(mp.average("amount")).is_gt(0)
```

#### 2. Cross-Dataset Validation
Leave check unconstrained and specify at metric level:

```python
@check(name="Data consistency check")
def validate_migration(mp: MetricProvider, ctx: Context) -> None:
    for column in ["user_id", "amount", "timestamp"]:
        source_nulls = mp.null_count(column, datasets=["source"])
        target_nulls = mp.null_count(column, datasets=["target"])

        ctx.assert_that(source_nulls).where(
            name=f"Source {column} nulls"
        ).is_eq(target_nulls)
```

#### 3. Reusable Checks with Dataset Parameters
Create parameterized checks for reuse:

```python
def create_completeness_check(dataset_name: str, threshold: float = 0.95):
    @check(datasets=[dataset_name], name=f"{dataset_name} completeness")
    def check_completeness(mp: MetricProvider, ctx: Context) -> None:
        for column in ["id", "created_at", "status"]:
            completeness = 1 - (mp.null_count(column) / mp.num_rows())
            ctx.assert_that(completeness).where(
                name=f"{column} completeness",
                severity="P1"
            ).is_geq(threshold)

    return check_completeness

# Create checks for different datasets
prod_completeness = create_completeness_check("production", 0.99)
staging_completeness = create_completeness_check("staging", 0.95)
```

#### 4. Dynamic Dataset Discovery
Handle variable datasets programmatically:

```python
@check
def validate_all_sources(mp: MetricProvider, ctx: Context) -> None:
    # This check adapts to whatever datasets are provided
    # Note: Requires accessing context internals - use with caution
    available_datasets = list(ctx._datasources.keys())

    for dataset in available_datasets:
        row_count = mp.num_rows(datasets=[dataset])
        ctx.assert_that(row_count).where(
            name=f"{dataset} has data"
        ).is_gt(0)
```

### Migration Strategy

For existing codebases, migration is straightforward:

#### Phase 1: Identify Usage Patterns
```bash
# Find checks that might need dataset specification
grep -r "@check" . | grep -v "datasets="

# Find metric calls that might need datasets
grep -r "mp\." . | grep -E "(average|sum|null_count|num_rows)"
```

#### Phase 2: Update Multi-Dataset Checks
1. Add dataset specifications to checks that should be constrained
2. Add dataset parameters to cross-dataset metrics
3. Run validation to catch any missed specifications

#### Phase 3: Add Validation Tests
```python
def test_dataset_specifications():
    """Ensure all checks properly specify datasets."""
    suite = VerificationSuite(all_checks, db, "test")

    # Test with multiple datasets should not raise errors
    try:
        suite.run({"ds1": ds1, "ds2": ds2}, key)
    except DQXError as e:
        if "requires explicit dataset specification" in str(e):
            pytest.fail(f"Missing dataset specification: {e}")
```

### Future Enhancements

While the current hybrid approach balances simplicity and flexibility, future enhancements could include:

1. **Dataset Aliases**: Map logical names to physical datasets
   ```python
   suite.run({
       "production": prod_ds,
       "production_mirror": prod_ds,  # Same physical dataset, different logical name
   }, key)
   ```

2. **Dataset Groups**: Define groups for related datasets
   ```python
   @check(dataset_groups=["transactional"])
   def validate_transactional_data(mp, ctx):
       # Runs on all datasets tagged as "transactional"
   ```

3. **Conditional Dataset Selection**: Dynamic dataset selection based on context
   ```python
   @check(dataset_selector=lambda ctx: ctx.get_active_regions())
   def validate_regional_data(mp, ctx):
       # Dynamically selects datasets based on active regions
   ```

4. **Dataset-Aware Caching**: Optimize metric computation across datasets
   ```python
   # Automatically reuse computations when same metric needed across datasets
   for ds in ["ds1", "ds2", "ds3"]:
       avg = mp.average("price", datasets=[ds])  # Cached after first computation
   ```

### Summary

The hybrid approach to dataset specification provides:
- **Simplicity**: Single-dataset cases work without any dataset specification
- **Flexibility**: Multi-dataset scenarios support both constrained and cross-dataset checks
- **Clarity**: Explicit specification requirements prevent ambiguity
- **Compatibility**: Existing code continues to work for single-dataset cases
- **Extensibility**: Clear patterns for future enhancements

This design ensures DQX can handle everything from simple single-table validations to complex multi-system data quality checks while maintaining an intuitive API.

## API Design

```mermaid
graph TD
    Suite[Verification Suite<br/>Top-level container]
    Check1[Check 1<br/>Quality validation]
    Check2[Check 2<br/>Quality validation]
    CheckN[Check N<br/>...]

    Assert1[Assertion 1<br/>Validation rule]
    Assert2[Assertion 2<br/>Validation rule]
    AssertM[Assertion M<br/>...]

    Op1[SQL Op 1<br/>Metric computation]
    Op2[SQL Op 2<br/>Metric computation]
    OpK[SQL Op K<br/>...]

    DS1[DataSource 1<br/>orders]
    DS2[DataSource 2<br/>customers]
    DSN[DataSource N<br/>...]

    Suite -->|contains| Check1
    Suite -->|contains| Check2
    Suite -->|contains| CheckN

    Check1 -->|has| Assert1
    Check1 -->|has| Assert2
    Check2 -->|has| AssertM

    Assert1 -->|uses| Op1
    Assert1 -->|uses| Op2
    Assert2 -->|uses| OpK

    Op1 -->|depends on| DS1
    Op2 -->|depends on| DS2
    OpK -->|depends on| DSN
```

### 1. Check Definition API

```python
@check(name="Order validation", tags=["critical"], datasets=["orders"])
def validate_orders(mp: MetricProvider, ctx: Context) -> None:
    """Define data quality checks using symbolic expressions."""
    # Simple assertions
    ctx.assert_that(mp.null_count("customer_id")).is_eq(0)

    # Chained assertions
    ctx.assert_that(mp.average("price")).where(
        name="Price validation",
        severity="P0"
    ).is_gt(0).is_lt(1000)

    # Complex expressions
    revenue = mp.sum("price") * mp.sum("quantity")
    ctx.assert_that(revenue).is_positive()
```

### 2. Suite Building API

```python
# Builder pattern for suite construction
suite = (VerificationSuiteBuilder("Production Suite", db)
         .add_check(validate_orders)
         .add_check(validate_customers)
         .add_checks([check_inventory, check_shipping])
         .build())

# Run verification
context = suite.run(
    datasources={"orders": order_ds, "customers": customer_ds},
    key=ResultKey(yyyy_mm_dd=date.today(), tags={"env": "prod"}),
    threading=True
)
```

### 3. Metric Provider API

```python
# Basic metrics
mp.num_rows()
mp.average("column")
mp.sum("column")
mp.minimum("column")
mp.maximum("column")
mp.null_count("column")
mp.approx_cardinality("column")

# Time-based metrics
mp.average("revenue", key=ctx.key.lag(1))  # Yesterday
mp.average("revenue", key=ctx.key.lag(7))  # Last week

# Custom metrics
mp.metric(specs.NegativeCount("balance"))
```

## Severity Levels in Checks and Assertions

### Severity Level Definitions

DQX supports four severity levels for data quality assertions, following a standard priority system:

- **P0 (Critical)**: Business-critical failures requiring immediate attention
  - Examples: Missing primary keys, revenue calculation errors, data loss
  - Impact: Can halt data pipelines or cause significant business disruption

- **P1 (High)**: Important issues affecting data reliability
  - Examples: Data freshness violations, significant statistical anomalies
  - Impact: May affect downstream analytics or reporting accuracy

- **P2 (Medium)**: Quality issues with moderate impact
  - Examples: Data format inconsistencies, minor completeness issues
  - Impact: Could affect specific use cases or require data cleaning

- **P3 (Low)**: Minor issues or warnings
  - Examples: Deprecated field usage, optimization opportunities
  - Impact: Informational, may indicate technical debt

### Setting Severity on Assertions

Severity is set using the `.where()` method when creating assertions:

```python
@check(name="Critical validations")
def validate_critical_fields(mp, ctx):
    # P0: No null primary keys
    ctx.assert_that(mp.null_count("id")).where(
        name="Primary key completeness",
        severity="P0"
    ).is_eq(0)

    # P1: Email format validation
    invalid_emails = mp.metric(specs.RegexNonMatchCount("email", r"^[^@]+@[^@]+\.[^@]+$"))
    ctx.assert_that(invalid_emails).where(
        name="Email format validation",
        severity="P1"
    ).is_leq(mp.num_rows() * 0.01)  # Max 1% invalid

    # P2: Optional field completeness
    ctx.assert_that(mp.null_count("phone_number") / mp.num_rows()).where(
        name="Phone number completeness",
        severity="P2"
    ).is_leq(0.3)  # Max 30% missing

    # P3: Performance warning
    ctx.assert_that(mp.average("response_time_ms")).where(
        name="Average response time",
        severity="P3"
    ).is_leq(500)
```

### Severity Propagation Model

The current severity propagation follows a bottom-up binary failure model:

```mermaid
graph BT
    A1[Assertion P0: Failed] --> C1[Check: Failed]
    A2[Assertion P1: Success] --> C1
    A3[Assertion P3: Failed] --> C1

    A4[Assertion P2: Success] --> C2[Check: Success]
    A5[Assertion P3: Success] --> C2

    C1 --> S[Suite: Contains Failures]
    C2 --> S
```

#### Current Behavior:
1. **Assertion Level**: Each assertion can have a severity level, but currently, any assertion failure (regardless of severity) causes its parent check to fail
2. **Check Level**: A check fails if ANY of its assertions fail, collecting all failure messages
3. **Suite Level**: The suite reports all check statuses but doesn't aggregate based on severity

#### Key Characteristics:
- **Binary Failure**: Any assertion failure causes check failure (no severity-based thresholds)
- **All Severities Equal**: P0 and P3 failures have the same effect on check status
- **No Severity Display**: Severity is stored but not shown in the inspection output

### Best Practices for Severity Assignment

1. **Be Consistent Across Teams**
   - Establish organization-wide guidelines for severity levels
   - Document severity criteria in your data quality runbooks
   - Review and calibrate severity assignments quarterly

2. **Consider Business Impact**
   ```python
   # P0: Direct revenue impact
   ctx.assert_that(mp.sum("transaction_amount")).where(
       name="Daily revenue total",
       severity="P0"
   ).is_geq(expected_daily_min)

   # P1: Affects key metrics
   ctx.assert_that(mp.approx_cardinality("user_id")).where(
       name="Active user count",
       severity="P1"
   ).is_geq(expected_users * 0.95)

   # P2: Data quality metrics
   ctx.assert_that(mp.null_count("category") / mp.num_rows()).where(
       name="Category completeness",
       severity="P2"
   ).is_leq(0.05)
   ```

3. **Use Severity for Alerting**
   - Configure monitoring to page on-call for P0 failures
   - Send email alerts for P1 failures
   - Log P2/P3 failures for batch review

4. **Progressive Severity Escalation**
   - Start new assertions at P2/P3 during development
   - Escalate to P1/P0 after validating thresholds
   - Document reasoning for severity changes

### Future Enhancement Opportunities

While the current implementation treats all severities equally for failure propagation, future enhancements could include:

1. **Severity-Aware Check Status**
   - Allow checks to pass with only P2/P3 failures
   - Implement configurable failure thresholds per severity

2. **Enhanced Display**
   - Show severity indicators in the inspection tree
   - Color-code assertions by severity level
   - Summary statistics by severity

3. **Suite-Level Policies**
   ```python
   # Hypothetical future API
   suite = (VerificationSuiteBuilder("Production Suite", db)
            .with_failure_policy(
                fail_on=["P0", "P1"],  # Only fail suite on P0/P1
                warn_on=["P2", "P3"]   # Generate warnings for P2/P3
            )
            .build())
   ```

4. **Severity-Based Execution**
   - Run only critical checks in fast-path validations
   - Schedule comprehensive checks including P2/P3 less frequently

## Code Usage and Examples

### Example 1: E-commerce Data Quality

```python
from dqx.api import VerificationSuiteBuilder, check
from dqx.extensions.pyarrow_ds import ArrowDataSource
from dqx.orm.repositories import MetricDB
from dqx.common import ResultKey
import datetime as dt

# Define comprehensive e-commerce checks
@check(name="Order integrity", tags=["critical", "orders"])
def check_order_integrity(mp, ctx):
    """Ensure order data integrity."""
    # No null order IDs
    ctx.assert_that(mp.null_count("order_id")).is_eq(0)

    # Valid order amounts
    ctx.assert_that(mp.minimum("total_amount")).is_geq(0)
    ctx.assert_that(mp.maximum("total_amount")).is_leq(100000)

    # Reasonable order quantities
    avg_items = mp.average("item_count")
    ctx.assert_that(avg_items).where(
        name="Average items per order"
    ).is_gt(0).is_lt(50)

@check(name="Customer validation", tags=["critical", "customers"])
def check_customer_data(mp, ctx):
    """Validate customer information."""
    # Unique customer IDs
    total_customers = mp.num_rows()
    unique_customers = mp.approx_cardinality("customer_id")
    ctx.assert_that(unique_customers / total_customers).where(
        name="Customer ID uniqueness"
    ).is_geq(0.99, tol=0.01)

    # Email completeness
    email_nulls = mp.null_count("email")
    ctx.assert_that(email_nulls / total_customers).where(
        name="Email completeness",
        severity="P1"
    ).is_leq(0.05)  # Max 5% missing emails

@check(name="Revenue monitoring", tags=["monitoring"])
def monitor_revenue_trends(mp, ctx):
    """Monitor revenue trends over time."""
    current_revenue = mp.sum("revenue")
    yesterday_revenue = mp.sum("revenue", key=ctx.key.lag(1))

    # Day-over-day change
    dod_change = (current_revenue - yesterday_revenue) / yesterday_revenue
    ctx.assert_that(dod_change).where(
        name="Day-over-day revenue change"
    ).is_geq(-0.3).is_leq(0.5)  # -30% to +50% change allowed

    # Minimum daily revenue threshold
    ctx.assert_that(current_revenue).where(
        name="Daily revenue threshold",
        severity="P0"
    ).is_geq(10000)

# Create and run the suite
db = MetricDB("postgresql://user:pass@localhost/metrics")

suite = (VerificationSuiteBuilder("E-commerce Quality Suite", db)
         .add_check(check_order_integrity)
         .add_check(check_customer_data)
         .add_check(monitor_revenue_trends)
         .build())

# Load data
orders_ds = ArrowDataSource.from_parquet("data/orders.parquet")
customers_ds = ArrowDataSource.from_parquet("data/customers.parquet")
revenue_ds = ArrowDataSource.from_parquet("data/daily_revenue.parquet")

# Run verification
result = suite.run(
    datasources={
        "orders": orders_ds,
        "customers": customers_ds,
        "revenue": revenue_ds
    },
    key=ResultKey(yyyy_mm_dd=dt.date.today(), tags={"env": "prod"}),
    threading=True
)

# Inspect results
for assertion in result._graph.assertions():
    if assertion._value:
        status = "✓" if assertion._value.unwrap().is_ok() else "✗"
        print(f"{status} {assertion.name}: {assertion._value}")
```

### Example 2: Cross-Dataset Validation

```python
@check(name="Data consistency", datasets=["source", "target"])
def validate_data_migration(mp, ctx):
    """Ensure data consistency between source and target systems."""
    # Row count consistency
    source_count = mp.num_rows(datasets=["source"])
    target_count = mp.num_rows(datasets=["target"])

    ctx.assert_that(target_count / source_count).where(
        name="Row count ratio"
    ).is_geq(0.99, tol=0.001).is_leq(1.01, tol=0.001)

    # Revenue consistency
    source_revenue = mp.sum("revenue", datasets=["source"])
    target_revenue = mp.sum("revenue", datasets=["target"])

    revenue_diff = sp.Abs(source_revenue - target_revenue)
    ctx.assert_that(revenue_diff).where(
        name="Revenue difference",
        severity="P0"
    ).is_leq(0.01)  # Max $0.01 difference
```

### Example 3: Batch Processing Large Datasets

```python
from dqx.extensions.pyarrow_ds import ArrowBatchDataSource

# Process multiple large Parquet files
batch_ds = ArrowBatchDataSource.from_parquets([
    "s3://bucket/data/2024-01-*.parquet",  # Glob pattern
    "s3://bucket/data/2024-02-*.parquet",
])

@check(name="Large dataset validation")
def validate_large_dataset(mp, ctx):
    """Validate large dataset with statistical methods."""
    # Use approximate algorithms for efficiency
    cardinality = mp.approx_cardinality("user_id")
    ctx.assert_that(cardinality).where(
        name="Unique users"
    ).is_geq(1_000_000)

    # Statistical validation
    avg_transaction = mp.average("amount")
    variance = mp.variance("amount")
    std_dev = sp.sqrt(variance)

    # Check for outliers (values beyond 3 standard deviations)
    ctx.assert_that(mp.maximum("amount")).is_leq(avg_transaction + 3 * std_dev)
    ctx.assert_that(mp.minimum("amount")).is_geq(avg_transaction - 3 * std_dev)

# Run with multi-threading for performance
suite.run({"transactions": batch_ds}, key, threading=True)
```

### Example 4: Custom Metric Implementation

```python
from dqx.specs import MetricSpec
from dqx.ops import Op
from dqx.states import State

class PercentileMetric:
    """Custom percentile metric implementation."""
    metric_type = "Percentile"

    def __init__(self, column: str, percentile: float):
        self._column = column
        self._percentile = percentile
        self._analyzers = (PercentileOp(column, percentile),)

    @property
    def name(self) -> str:
        return f"percentile({self._column}, {self._percentile})"

    @property
    def parameters(self) -> dict:
        return {"column": self._column, "percentile": self._percentile}

    @property
    def analyzers(self) -> Sequence[Op]:
        return self._analyzers

    def state(self) -> State:
        return PercentileState(
            value=self._analyzers[0].value(),
            percentile=self._percentile
        )

# Use custom metric in checks
@check
def check_response_times(mp, ctx):
    p95 = mp.metric(PercentileMetric("response_time", 0.95))
    p99 = mp.metric(PercentileMetric("response_time", 0.99))

    ctx.assert_that(p95).is_leq(1000)  # 95th percentile < 1s
    ctx.assert_that(p99).is_leq(2000)  # 99th percentile < 2s
```

## Performance Considerations

### 1. Query Optimization
- **Batch Processing**: Multiple metrics computed in single SQL query
- **Deduplication**: Redundant operations eliminated at planning stage
- **Columnar Processing**: Leverages DuckDB's vectorized execution

### 2. Memory Efficiency
- **Statistical Sketches**: HyperLogLog for cardinality (1.5KB per sketch)
- **Streaming Algorithms**: Process data in chunks for large datasets
- **Lazy Evaluation**: Metrics computed only when needed

### 3. Parallel Execution
- **Multi-threading**: Concurrent processing of data chunks
- **Independent Datasets**: Parallel analysis across datasets
- **Lock-free Aggregation**: Thread-safe metric combination

### 4. Caching Strategy
- **Metric Reuse**: Computed metrics shared across checks
- **Persistent Storage**: Historical metrics cached in database
- **Dependency Resolution**: Minimal recomputation on changes

## Extension Points

### 1. Custom Data Sources
Implement the `SqlDataSource` protocol:

```python
class CustomDataSource:
    name = "custom_source"
    dialect = "duckdb"  # or your custom dialect

    @property
    def cte(self) -> str:
        """SQL Common Table Expression."""
        return "SELECT * FROM custom_table"

    def query(self, query: str) -> duckdb.DuckDBPyRelation:
        """Execute query against data source."""
        # Custom query execution logic
```

### 2. Custom Dialects
Implement custom SQL dialects for different databases:

```python
from dqx.dialect import Dialect

class PostgreSQLDialect:
    name = "postgresql"

    def translate_sql_op(self, op: SqlOp) -> str:
        """Translate operations to PostgreSQL syntax."""
        # Implement SQL translation

    def build_cte_query(self, cte_sql: str, expressions: list[str]) -> str:
        """Build formatted CTE query."""
        from dqx.dialect import build_cte_query
        return build_cte_query(cte_sql, expressions)
```

### 3. Custom Metric Types
Add new metric specifications:

```python
class CustomMetric:
    metric_type = "Custom"

    def __init__(self, **params):
        # Initialize metric

    @property
    def analyzers(self) -> Sequence[Op]:
        # Return required operations
```

### 4. Custom Validators
Create custom validation logic:

```python
def custom_validator(value: float) -> bool:
    """Custom validation function."""
    # Implement validation logic
    return True

ctx.assert_that(metric).where(
    validator=SymbolicValidator("custom", custom_validator)
)
```

## Future Improvements and Roadmap

### Phase 1: Core Enhancements (Q1 2025)
- [ ] **Streaming Support**: Real-time data quality monitoring
  - Implement streaming data sources
  - Support incremental metric updates
  - Add windowed aggregations

- [ ] **Advanced Metrics**: Expand metric library
  - Entropy and information metrics
  - ML-based anomaly scores
  - Custom quantile sketches

- [ ] **Performance Optimizations**
  - GPU acceleration for large-scale computations
  - Distributed execution with Ray/Dask
  - Adaptive query optimization

### Phase 2: Integration & UI (Q2 2025)
- [ ] **Web Dashboard**: Visual monitoring interface
  - Real-time metric visualization
  - Alert configuration UI
  - Historical trend analysis

- [ ] **Data Catalog Integration**
  - Apache Hive metastore support
  - AWS Glue catalog integration
  - Databricks Unity Catalog support

- [ ] **Workflow Integration**
  - Apache Airflow operators
  - Prefect tasks
  - DBT integration

### Phase 3: Advanced Features (Q3 2025)
- [ ] **ML-Powered Quality**: Intelligent anomaly detection
  - Auto-learn normal data patterns
  - Predictive quality alerts
  - Root cause analysis

- [ ] **Complex Event Processing**
  - Multi-stream correlations
  - Temporal pattern matching
  - Causal inference

- [ ] **Data Lineage Tracking**
  - Automatic dependency discovery
  - Impact analysis
  - Quality propagation

### Phase 4: Enterprise & Cloud (Q4 2025)
- [ ] **Kubernetes Native**: Cloud-native deployment
  - Kubernetes operators
  - Helm charts
  - Auto-scaling policies

- [ ] **Multi-tenancy Support**
  - Resource isolation
  - Role-based access control
  - Audit logging

- [ ] **Cloud Provider Integration**
  - AWS native services
  - GCP BigQuery support
  - Azure Synapse integration

### Long-term Vision
- **Self-Healing Data**: Automatic data quality remediation
- **Federated Quality**: Cross-organization quality standards
- **Quality-as-Code**: GitOps for data quality rules
- **AI Assistant**: Natural language quality check generation

## Contributing

We welcome contributions! Key areas for contribution:

1. **New Metric Types**: Extend the metric library
   - Statistical metrics (median, mode, percentiles)
   - Text analysis metrics (pattern matching, regex counts)
   - Time-series specific metrics

2. **Data Source Adapters**: Support more data sources
   - Apache Spark DataFrames
   - Pandas DataFrames
   - Cloud data warehouses (Snowflake, Redshift)
   - Streaming sources (Kafka, Kinesis)

3. **Performance Improvements**: Optimize existing functionality
   - Query optimization strategies
   - Memory usage reduction
   - Parallel processing enhancements

4. **Documentation**: Improve and expand documentation
   - Tutorial notebooks
   - Best practices guide
   - Performance tuning guide

5. **Testing**: Enhance test coverage
   - Edge case testing
   - Performance benchmarks
   - Integration tests

## Conclusion

DQX represents a modern approach to data quality management, combining high-performance computation with an intuitive API. The graph-based architecture provides flexibility and efficiency, while the use of DuckDB and statistical sketching algorithms ensures scalability to large datasets.

The framework's design emphasizes:
- **Developer Experience**: Clean, declarative API with fluent assertions
- **Performance**: Optimized SQL generation and parallel execution
- **Extensibility**: Clear extension points for custom metrics and data sources
- **Production Readiness**: Comprehensive error handling and persistence

As data quality becomes increasingly critical in modern data pipelines, DQX provides a solid foundation for building reliable data quality checks that scale with your data infrastructure.

For more information, contribute, or report issues, visit the project repository at [gitlab.booking.com/npham/dqx](https://gitlab.booking.com/npham/dqx).
