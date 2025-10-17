# DQX System Patterns

## Architecture Overview

### Layered Architecture
```
┌─────────────────────────────────────┐
│          User API Layer             │  @check, MetricProvider, Context
├─────────────────────────────────────┤
│       Graph Execution Layer         │  Graph, Nodes, Visitors
├─────────────────────────────────────┤
│        Analysis Layer               │  Analyzer, MetricSpecs
├─────────────────────────────────────┤
│      SQL Generation Layer           │  Dialect, CTE Builder
├─────────────────────────────────────┤
│        Data Source Layer            │  SqlDataSource Protocol
└─────────────────────────────────────┘
```

## Core Design Patterns

### 1. Dependency Graph Pattern
The system builds a directed acyclic graph (DAG) of dependencies:
```
RootNode
├── CheckNode("Order Validation")
│   ├── AssertionNode("Price check")
│   │   └── SymbolNode("x_1: average(price)")
│   └── AssertionNode("Max check")
│       └── SymbolNode("x_2: max(price)")
└── CheckNode("Customer Validation")
    └── AssertionNode("Email check")
        └── SymbolNode("x_3: null_count(email)")
```

### 2. Visitor Pattern
Different visitors traverse the graph for different purposes:
- **SuiteValidator**: Validates graph structure and finds issues
- **DatasetImputer**: Infers dataset associations
- **Evaluator**: Evaluates assertions after metric computation
- **SymbolCollector**: Collects all symbols for display

### 3. Protocol-Based Extensions
Uses Python protocols for clean extension points:
```python
class SqlDataSource(Protocol):
    name: str
    dialect: str

    @property
    def cte(self) -> str: ...


class MetricSpec(Protocol):
    def to_sql(self, table_alias: str, dialect: Dialect) -> str: ...
```

### 4. Symbolic Expression Pattern
Leverages SymPy for mathematical expressions:
```python
# Metrics are symbols
x_1 = mp.average("price")  # Creates Symbol("x_1")
x_2 = mp.sum("quantity")  # Creates Symbol("x_2")

# Combine with math
total_revenue = x_1 * x_2  # SymPy expression
```

### 5. Builder Pattern
Used for constructing complex objects:
- **AssertionDraft → AssertionReady**: Two-stage assertion building
- **Graph Builder**: Constructs dependency graph incrementally
- **CTE Builder**: Builds optimized SQL queries

## Component Relationships

### Graph Module (`src/dqx/graph/`)
- **base.py**: Abstract base classes (BaseNode, CompositeNode)
- **nodes.py**: Concrete node types (RootNode, CheckNode, AssertionNode)
- **traversal.py**: Graph class with BFS traversal
- **visitors.py**: Visitor implementations

### Core Components
- **api.py**: User-facing API (VerificationSuite, Context, decorators)
- **analyzer.py**: Converts metrics to SQL and executes queries
- **evaluator.py**: Evaluates assertions using computed metrics
- **provider.py**: MetricProvider and SymbolicMetric management

### Extension Points
- **specs.py**: Metric specifications (SumOp, AverageOp, etc.)
- **dialect.py**: SQL dialect implementations
- **validator.py**: Suite validation logic

## Key Workflows

### 1. Check Registration Flow
```
@check decorator
    → _create_check()
    → CheckNode created & added to graph
    → Check function executed in context
    → Assertions create AssertionNodes
```

### 2. Analysis Flow
```
VerificationSuite.run()
    → Build graph
    → Impute datasets
    → Group metrics by date
    → Analyzer.analyze()
        → Generate SQL with CTEs
        → Execute on data source
        → Store results in DB
```

### 3. Evaluation Flow
```
Graph.bfs(Evaluator)
    → Visit each assertion
    → Evaluate symbolic expression
    → Apply validator function
    → Set assertion status (OK/FAILURE)
```

## Data Flow

### Metric Registration
1. User calls `mp.average("col")`
2. MetricProvider creates SymbolicMetric with:
   - Unique symbol (x_1, x_2, etc.)
   - MetricSpec instance
   - Dataset association
   - ResultKey provider

### SQL Generation
1. Analyzer groups metrics by dataset/date
2. Builds CTE for data source
3. Adds SELECT for each metric
4. Executes single query per dataset/date

### Result Propagation
1. Query results stored in MetricDB
2. Evaluator retrieves values by symbol
3. SymPy evaluates expressions
4. Validators check constraints
5. Results collected for reporting

## Error Handling Patterns

### Validation Errors
- Graph validation before execution
- Clear error messages with context
- Warnings for non-critical issues

### Execution Errors
- SQL errors wrapped with context
- Metric computation failures tracked
- Partial results still accessible

### Assertion Failures
- Each failure includes:
  - Expression that failed
  - Actual computed value
  - Expected constraint
  - Symbol values involved

## Performance Optimizations

### Single-Pass Execution
- All metrics for a dataset/date in one query
- CTE-based query structure
- Columnar computation in database

### Lazy Evaluation
- Symbols only computed when needed
- Graph built incrementally
- Metrics grouped by execution context

### Memory Efficiency
- Streaming results from database
- No data materialization in Python
- Constant memory usage

## Extension Examples

### Custom Metric
```python
class MedianOp(MetricSpec):
    def __init__(self, column: str):
        self.column = column

    def to_sql(self, table_alias: str, dialect: Dialect) -> str:
        return (
            f"PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {table_alias}.{self.column})"
        )
```

### Custom Validator
```python
def is_unique(value: float) -> bool:
    return abs(value - 1.0) < EPSILON


validator = SymbolicValidator("is unique", is_unique)
```

### Custom Data Source
```python
class PostgresDataSource:
    name = "postgres"
    dialect = "postgresql"

    @property
    def cte(self) -> str:
        return f"SELECT * FROM {self.table}"
```

## Design Principles Applied

### SOLID Principles
- **Single Responsibility**: Each class has one clear purpose
- **Open/Closed**: Extensible via protocols, closed for modification
- **Liskov Substitution**: All nodes/visitors follow base contracts
- **Interface Segregation**: Small, focused protocols
- **Dependency Inversion**: Depend on protocols, not implementations

### DRY (Don't Repeat Yourself)
- Metric specs reused across checks
- Common validation logic in base classes
- Shared SQL generation logic

### YAGNI (You Aren't Gonna Need It)
- No premature optimization
- Features added only when needed
- Simple solutions preferred
