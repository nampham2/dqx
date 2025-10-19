# DQX System Patterns

## Architecture Overview

DQX follows a graph-based architecture where data quality checks are represented as a directed acyclic graph (DAG) of dependencies. The system uses several key design patterns to achieve extensibility, testability, and performance.

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│    API      │────▶│  Graph/Node  │────▶│  Analyzer   │
│  (api.py)   │     │  Structure   │     │ (analyzer.py)│
└─────────────┘     └──────────────┘     └─────────────┘
                            │                     │
                            ▼                     ▼
                    ┌──────────────┐     ┌─────────────┐
                    │  Evaluator   │     │   Plugins   │
                    │(evaluator.py)│     │(plugins.py) │
                    └──────────────┘     └─────────────┘
```

## Core Design Patterns

### 1. Graph Pattern
The entire system is built around a graph structure that represents the dependencies between checks, assertions, and metrics.

```text
# Node Hierarchy
RootNode
├── CheckNode (name="revenue_check")
│   ├── AssertionNode (expr="x_1 > 0")
│   │   └── SymbolNode (symbol="x_1", metric="sum(revenue)")
│   └── AssertionNode (expr="x_2 < 1000")
│       └── SymbolNode (symbol="x_2", metric="avg(revenue)")
└── CheckNode (name="user_check")
    └── AssertionNode (expr="x_3 > 100")
        └── SymbolNode (symbol="x_3", metric="count(user_id)")
```

**Key Benefits:**
- Clear dependency tracking
- Optimal execution order
- Easy visualization
- Extensible structure

### 2. Visitor Pattern
Used extensively for graph traversal and operations without modifying node classes.

```python
# Base Visitor Protocol
class GraphVisitor(Protocol):
    def visit(self, node: BaseNode) -> None: ...


# Concrete Visitors
class ComputeVisitor:
    """Computes metrics by visiting symbol nodes"""


class DisplayVisitor:
    """Displays graph structure"""


class DatasetImputationVisitor:
    """Infers dataset associations"""


class CompositeValidationVisitor:
    """Runs multiple validators in single pass"""

    def __init__(self, validators: list[GraphVisitor]):
        self.validators = validators
```

**Visitor Types:**
- **ComputeVisitor**: Executes SQL queries
- **DisplayVisitor**: Renders graph structure
- **ValidationVisitor**: Validates graph consistency
- **DatasetImputationVisitor**: Infers missing datasets
- **CompositeValidationVisitor**: Combines multiple validators

### 3. Builder Pattern
Two-stage assertion building ensures all assertions have meaningful names.

```python
# Stage 1: Create draft
draft = ctx.assert_that(mp.sum("revenue"))

# Stage 2: Add context and create assertion
assertion = draft.where(name="Revenue is positive").is_positive()
```

**Benefits:**
- Enforces naming convention
- Fluent interface
- Type safety at each stage
- Clear error messages

### 4. Protocol-Based Extensions
All extension points use Python protocols (PEP 544) instead of inheritance.

```python
# Core Protocols
class SqlDataSource(Protocol):
    name: str
    dialect: str

    def cte(self, nominal_date: datetime.date) -> str: ...
    def query(
        self, query: str, nominal_date: datetime.date
    ) -> duckdb.DuckDBPyRelation: ...


class Analyzer(Protocol):
    def analyze(
        self, ds: SqlDataSource, metrics: Sequence[MetricSpec], key: ResultKey
    ) -> AnalysisReport: ...


class MetricSpec(Protocol):
    def to_sql(self, table_alias: str, dialect: Dialect) -> str: ...
```

**Advantages:**
- No inheritance required
- Static type checking
- Clear contracts
- Easy testing

### 5. Plugin System Pattern
Extensible plugin architecture for result processing and custom behaviors.

```python
# Plugin Protocol
class PostProcessor(Protocol):
    @staticmethod
    def metadata() -> PluginMetadata: ...

    def process(self, context: PluginExecutionContext) -> None: ...


# Plugin Manager
class PluginManager:
    def __init__(self, timeout: float = 60.0):
        self._plugins: dict[str, PostProcessor] = {}
        self._timeout = timeout

    def register(self, plugin: PostProcessor) -> None:
        """Register plugin with validation"""
        metadata = plugin.metadata()
        self._plugins[metadata.name] = plugin

    def execute_all(self, context: PluginExecutionContext) -> None:
        """Execute all plugins with timeout protection"""
        for plugin in self._plugins.values():
            with time_limit(self._timeout):
                plugin.process(context)
```

**Plugin Features:**
- **Metadata**: Each plugin provides name, version, author, description
- **Capabilities**: Plugins declare their capabilities (e.g., "reporting")
- **Execution Context**: Rich context with results, symbols, and helpers
- **Time Limits**: 60-second timeout prevents hanging
- **Built-in Plugins**: AuditPlugin for execution reports

**Example Plugin:**
```python
class AuditPlugin:
    @staticmethod
    def metadata() -> PluginMetadata:
        return PluginMetadata(
            name="audit",
            version="1.0.0",
            author="DQX Team",
            description="Displays execution audit report",
            capabilities={"reporting", "display"},
        )

    def process(self, context: PluginExecutionContext) -> None:
        # Create rich tables with results
        # Display execution statistics
        # Show timing information
        pass
```

### 6. Result Type Pattern
Using the Returns library for functional error handling.

```python
from returns.result import Result, Success, Failure


# Instead of exceptions
def compute_metric() -> Result[float, str]:
    if error:
        return Failure("Error message")
    return Success(42.0)


# Chain operations
result = compute_metric().bind(validate_range).map(lambda x: x * 100)
```

**Benefits:**
- Explicit error handling
- Composable operations
- Type-safe error propagation
- No hidden exceptions

### 7. CTE-Based SQL Generation
All SQL queries use Common Table Expressions for clarity and performance.

```sql
WITH
t1 AS (
    SELECT price, user_id
    FROM sales
    WHERE yyyy_mm_dd = '2024-01-01'
),
x_1 AS (SELECT SUM(price) AS value FROM t1),
x_2 AS (SELECT AVG(price) AS value FROM t1),
x_3 AS (SELECT COUNT(DISTINCT user_id) AS value FROM t1)
SELECT
    'x_1' AS symbol, value FROM x_1
    UNION ALL
    'x_2' AS symbol, value FROM x_2
    UNION ALL
    'x_3' AS symbol, value FROM x_3
```

**Advantages:**
- Single query per dataset/date
- Readable structure
- Reusable subqueries
- Optimal execution plans

## Error Handling Patterns

### EvaluationFailure Pattern
Rich error context for debugging failed evaluations.

```python
@dataclass
class EvaluationFailure:
    """Detailed failure information"""

    error_message: str  # Overall error
    expression: str  # Failed expression
    symbols: list[SymbolInfo]  # Symbol details


@dataclass
class SymbolInfo:
    """Symbol metadata and value"""

    name: str  # e.g., "x_1"
    metric: str  # e.g., "sum(revenue)"
    dataset: str | None  # Source dataset
    value: Result[float, str]  # Computed value
    yyyy_mm_dd: datetime.date
    suite: str
    tags: Tags = field(default_factory=dict)
```

**Usage in Assertions:**
```python
# Assertion results include rich error info
result = AssertionResult(
    metric=Failure(
        [
            EvaluationFailure(
                error_message="Division by zero",
                expression="x_1 / x_2",
                symbols=[
                    SymbolInfo(name="x_1", value=Success(100)),
                    SymbolInfo(name="x_2", value=Success(0)),
                ],
            )
        ]
    )
)
```

### Validation Error Pattern
Structured validation with different severity levels.

```python
@dataclass
class ValidationReport:
    errors: list[str]  # Must fix before execution
    warnings: list[str]  # Should address but not blocking

    def to_json(self) -> dict[str, Any]:
        """Export for tooling integration"""
        return {
            "errors": self.errors,
            "warnings": self.warnings,
            "valid": not self.errors,
        }
```

## Component Relationships

### Dependency Flow
```
User Code → API → Graph Builder → Validator → Analyzer → Evaluator → Plugins
                        ↓              ↓          ↓          ↓          ↓
                     nodes.py    validator.py analyzer.py evaluator.py plugins.py
```

### Data Flow
```
DataSource → SQL Query → DuckDB → Results → Evaluation → Display/Storage
                ↓                    ↓           ↓            ↓
            dialect.py          analyzer.py  evaluator.py  display.py
```

## State Management Patterns

### Graph Building State
```python
class RootNode:
    def __init__(self):
        self._graph_built = False  # Defensive flag
        self.datasets: set[str] = set()  # For imputation

    @property
    def graph(self) -> Graph:
        if not self._graph_built:
            raise DQXError("Graph not built yet")
        return self._graph
```

### Symbol Registry Pattern
```python
class SymbolRegistry:
    """Manages symbol generation and tracking"""

    def __init__(self):
        self._counter = 0
        self._symbols: dict[str, MetricSpec] = {}

    def create_symbol(self, metric: MetricSpec) -> str:
        self._counter += 1
        symbol = f"x_{self._counter}"
        self._symbols[symbol] = metric
        return symbol
```

## Performance Patterns

### Lazy Evaluation
- Metrics computed only when needed
- Graph built on first access
- Results cached after computation

### Single-Pass Execution
- All metrics for a dataset/date in one query
- No repeated scans
- Optimal query plans

### Plugin Timeout Pattern
```python
def time_limit(seconds: float):
    """Context manager for time-limited execution"""

    def handler(signum, frame):
        raise TimeoutError(f"Exceeded {seconds}s limit")

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)
```

## Testing Patterns

### Protocol Testing
```python
def test_implements_protocol():
    """Verify class implements protocol correctly"""
    assert isinstance(MyDataSource(), SqlDataSource)
```

### Result Testing
```python
def test_success_case():
    result = compute_metric()
    assert isinstance(result, Success)
    assert result.unwrap() == 42.0


def test_failure_case():
    result = compute_metric()
    assert isinstance(result, Failure)
    assert "error" in result.failure()
```

### Graph Testing
```python
def test_graph_structure():
    suite = VerificationSuite("test")
    # Build graph
    graph = suite.graph

    # Verify structure
    assert len(graph.nodes) == expected
    assert graph.is_dag()
```

## Future Patterns Under Consideration

### Stream Processing Pattern
For real-time validation (not yet implemented).

### Circuit Breaker Pattern
For handling repeated failures gracefully.

### Event Sourcing Pattern
For audit trail and time travel debugging.

### Saga Pattern
For multi-step validation workflows with compensation.

## Anti-Patterns to Avoid

### ❌ Mutable Shared State
All nodes and specs are immutable after creation.

### ❌ Inheritance Hierarchies
Use protocols and composition instead.

### ❌ Hidden Side Effects
All operations explicit in return types.

### ❌ Synchronous Blocking
Future async support will be non-blocking.

### ❌ Tight Coupling
Components interact through protocols only.
