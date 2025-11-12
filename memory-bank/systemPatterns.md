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

## Development Methodology

### 3D Methodology (Design-Driven Development)

**Key principle**: Think before you build, build with intention, ship with confidence.

#### Quick 3D Reminders
- **Run `uv run pytest` at end of every phase** - 100% pass required
- **Update implementation progress checkboxes** as you complete each phase
- **Follow Example Creation Guidelines** for comprehensive examples
- **Apply Unit Testing Guidelines** for thorough test coverage

### KISS/YAGNI Design Principles

**KISS (Keep It Simple, Stupid)** & **YAGNI (You Aren't Gonna Need It)**: Balance engineering rigor with practical simplicity.

#### AI Decision-Making Guidelines

🎯 **START SIMPLE, EVOLVE THOUGHTFULLY**

For design decisions, AI coders should:
1. **Default to simplest solution** that meets current requirements
2. **Document complexity trade-offs** when proposing alternatives
3. **Present options** when multiple approaches have merit
4. **Justify complexity** only when immediate needs require it

🤖 **AI CAN DECIDE** (choose simplest):
- Data structure choice (dict vs class vs dataclass)
- Function organization (single file vs module split)
- Error handling level (basic vs comprehensive)
- Documentation depth (minimal vs extensive)

👤 **PRESENT TO HUMAN** (let them choose):
- Architecture patterns (monolith vs microservices)
- Framework choices (custom vs third-party)
- Performance optimizations (simple vs complex)
- Extensibility mechanisms (hardcoded vs configurable)

⚖️ **COMPLEXITY JUSTIFICATION TEMPLATE**:
"Proposing [complex solution] over [simple solution] because:
- Current requirement: [specific need]
- Simple approach limitation: [concrete issue]
- Complexity benefit: [measurable advantage]
- Alternative: [let human decide vs simpler approach]"

#### Incremental Complexity Strategy

📈 **EVOLUTION PATH** (add complexity only when needed):

Phase 1: Hardcoded → Phase 2: Configurable → Phase 3: Extensible

Example:
- Phase 1: `return "Hello, World!"`
- Phase 2: `return f"Hello, {name}!"`
- Phase 3: `return formatter.format(greeting_template, name)`

🔄 **WHEN TO EVOLVE**:
- Phase 1→2: When second use case appears
- Phase 2→3: When third different pattern emerges
- Never evolve: If usage remains stable

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

### 6. Result Type Pattern (Returns Library)

The Returns library provides functional error handling with Result and Maybe types. This pattern is fundamental to DQX's error handling strategy.

**Documentation**: Read https://returns.readthedocs.io/en/latest/pages/result.html before writing code with returns.

#### Core Types

```python
from returns.result import Result, Success, Failure
from returns.maybe import Maybe, Some, Nothing
```

#### Result Type Usage

Used for operations that can fail with an error message:

```python
def compute_metric() -> Result[float, str]:
    if error_condition:
        return Failure("Descriptive error message")
    return Success(computed_value)
```

#### Maybe Type Usage

Used for values that might not exist:

```python
def get_from_cache(key: CacheKey) -> Maybe[Metric]:
    if key in cache:
        return Some(cache[key])
    return Nothing
```

#### Pattern Matching - The ONLY Way

**CRITICAL**: Always use pattern matching. NEVER use isinstance, hasattr, or direct attribute access.

```python
# ✅ CORRECT - Pattern matching for Maybe
match maybe_metric:
    case Some(metric):
        # Use metric value here
        process(metric)
    case Nothing():
        # Handle absence
        return default_value

# ✅ CORRECT - Pattern matching for Result
match computation_result:
    case Success(value):
        return value * 2
    case Failure(error):
        logger.error(f"Computation failed: {error}")
        return Failure(error)

# ❌ WRONG - NEVER DO THIS!
if isinstance(maybe_value, Some):  # Anti-pattern!
    value = maybe_value.unwrap()

# ❌ WRONG - NEVER DO THIS!
if hasattr(result, 'unwrap'):  # Anti-pattern!
    value = result.unwrap()
```

#### Helper Functions for Result Checking

For cases where you need to filter or check Result types in list comprehensions or other functional contexts, use the `is_successful` helper from `returns.pipeline`:

```python
from returns.pipeline import is_successful

# ✅ CORRECT - Using is_successful for filtering
failed_symbols = [si for si in symbol_infos if not is_successful(si.value)]
failure_count = sum(1 for s in symbols if not is_successful(s.value))

# The helper works with both Result and Maybe types
assert is_successful(Success(1)) is True
assert is_successful(Failure("error")) is False
assert is_successful(Some(1)) is True
assert is_successful(Nothing) is False
```

This helper is particularly useful in:
- List comprehensions where pattern matching would be verbose
- Counting operations
- Filtering collections
- Boolean checks in functional pipelines

#### Early Return Pattern

```python
def complex_computation(cache: MetricCache) -> Result[float, str]:
    # Early return on failure
    check_result = validate_input()
    match check_result:
        case Failure() as failure:
            return failure
        case Success():
            pass  # Continue processing

    # Main computation
    return Success(computed_value)
```

#### Real Examples from DQX

**Cache Operations (cache.py)**:
```python
def get(self, key: CacheKey) -> Maybe[Metric]:
    with self._lock:
        if key in self._cache:
            return Some(self._cache[key])

    # Try database
    db_result = self._db.get_metric(...)
    match db_result:
        case Some(value):
            self._cache[key] = value
            return Some(value)
        case _:
            return Nothing
```

**Compute Functions (compute.py)**:
```python
def simple_metric(
    metric: MetricSpec, cache: MetricCache, key: CacheKey
) -> Result[float, str]:
    maybe_metric = cache.get(cache_key)

    match maybe_metric:
        case Some(metric_value):
            # Correct: pattern match and extract
            return Success(metric_value.value)
        case _:
            error_msg = f"Metric {metric.name} not found!"
            return Failure(error_msg)
```

#### Chaining Operations

```python
# Functional composition
result = (
    get_metric()
    .bind(validate_range)
    .map(lambda x: x * 100)
    .bind(apply_business_rule)
)

# Handle final result
match result:
    case Success(value):
        logger.info(f"Final value: {value}")
    case Failure(error):
        logger.error(f"Pipeline failed: {error}")
```

#### Common Patterns in DQX

1. **Cache Miss Handling**:
```python
match cache.get(key):
    case Some(metric):
        return Success(metric.value)
    case Nothing():
        return Failure("Not in cache")
```

2. **Validation Chains**:
```python
match validate_dates(ts, expected_dates):
    case Failure() as f:
        return f
    case Success():
        # Continue processing
```

3. **Database Fallback**:
```python
# Cache first, then database
match cache_result:
    case Some(value):
        return Success(value)
    case Nothing():
        # Try database
        match db_result:
            case Some(value):
                cache.put(value)
                return Success(value)
            case Nothing():
                return Failure("Not found")
```

#### Anti-Patterns to Avoid

1. **❌ isinstance Checking**:
```python
# WRONG - breaks functional paradigm
if isinstance(maybe, Some):
    value = maybe.unwrap()
```

2. **❌ hasattr Checking**:
```python
# WRONG - implementation detail leak
if hasattr(result, 'failure'):
    handle_error(result.failure())
```

3. **❌ Try-Except with Result**:
```python
# WRONG - mixing paradigms
try:
    return Success(risky_operation())
except Exception as e:
    return Failure(str(e))
```

4. **❌ Using None Instead of Nothing**:
```python
# WRONG - use Maybe type
def get_value() -> Optional[float]:  # Bad
    return None

# CORRECT
def get_value() -> Maybe[float]:
    return Nothing
```

#### Benefits
- **Explicit Error Handling**: All failure points visible in type signatures
- **Composable**: Chain operations without nested error checking
- **Type-Safe**: Compiler ensures all cases handled
- **No Hidden Exceptions**: All errors propagated explicitly
- **Functional Purity**: No side effects in error handling

#### Known Issues in Codebase
- ~~provider.py currently has incorrect isinstance usage that needs fixing~~ (Fixed)
- ~~evaluator.py and plugins.py had isinstance usage~~ (Fixed - now uses pattern matching and `is_successful`)
- Some older code might still use Optional instead of Maybe

**Remember**: When in doubt, use pattern matching. It's the only correct way to handle Result and Maybe types.

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

### Topological Sort Pattern
The MetricRegistry uses topological sorting to ensure metrics are evaluated in dependency order.

```python
class MetricRegistry:
    def topological_sort(self) -> None:
        """Sort metrics in topological order for evaluation.

        Ensures all required_metrics for a given metric appear
        before that metric in the list.
        """
        # Build dependency graph
        in_degree = {sm.symbol: len(internal_deps) for sm in metrics}
        adjacency = build_adjacency_list()

        # Process in topological order
        queue = deque(metrics_with_no_deps)
        while queue:
            current = queue.popleft()
            result.append(current)
            # Update dependent metrics

        # Detect cycles
        if len(result) != n:
            raise DQXError("Circular dependency detected")
```

**Key Features:**
- **Dependency Resolution**: Simple metrics evaluated before extended metrics
- **Cycle Detection**: Identifies and reports circular dependencies
- **External Dependencies**: Gracefully handles dependencies outside registry
- **In-place Sorting**: Modifies internal metric list for optimal evaluation
- **Clear Error Messages**: Reports specific metrics involved in cycles

**Benefits:**
- Ensures correct evaluation order
- Prevents infinite loops from circular dependencies
- Optimizes computation by respecting dependencies
- Aids debugging with clear dependency visualization

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

## Cache Pattern

The caching system provides high-performance metric retrieval with automatic fallback to database queries.

### Cache Architecture
```python
class MetricCache:
    """Thread-safe cache with performance tracking"""

    def __init__(self, db: MetricDB):
        self._db = db
        self._cache: dict[CacheKey, Metric] = {}
        self._dirty: set[CacheKey] = set()  # Tracks unsaved metrics
        self._lock = RLock()
        self._stats = CacheStats()
```

### Cache Key Design
```python
# Type-safe cache key
CacheKey: TypeAlias = tuple[MetricSpec, ResultKey, DatasetName, ExecutionId]

# Ensures proper isolation by execution context
key = (metric_spec, result_key, dataset, execution_id)
```

### Performance Statistics
```python
@dataclass
class CacheStats:
    """Mutable statistics for efficient tracking"""
    hit: int = 0
    missed: int = 0

    def hit_ratio(self) -> float:
        total = self.hit + self.missed
        return self.hit / total if total > 0 else 0.0

    def record_hit(self) -> None:
        self.hit += 1

    def record_miss(self) -> None:
        self.missed += 1
```

### Lock Optimization Pattern
```python
def get(self, key: CacheKey) -> Maybe[Metric]:
    """Minimal lock contention design"""

    # Quick cache check with lock
    with self._lock:
        if key in self._cache:
            self._stats.record_hit()
            return Some(self._cache[key])

    # DB query without lock (I/O outside critical section)
    db_result = self._db.get_metric(...)

    # Update cache with lock
    with self._lock:
        if db_result:
            self._cache[key] = db_result
        self._stats.record_miss()

    return db_result
```

### TimeSeries Storage Pattern
```python
# Old: Store just values
TimeSeries: TypeAlias = dict[datetime.date, float]

# New: Store full Metric objects for richer context
TimeSeries: TypeAlias = dict[datetime.date, Metric]

# Benefits:
# - Access to metadata
# - Full metric information
# - Better debugging context
```

### Cache Integration with Plugins
```python
@dataclass
class PluginExecutionContext:
    # ... other fields ...
    cache_stats: CacheStats  # Performance metrics

    # Plugin can display cache performance
    def display_cache_stats(self):
        stats = self.cache_stats
        print(f"Cache hits: {stats.hit}, misses: {stats.missed}")
        print(f"Hit ratio: {stats.hit_ratio():.1%}")
```

### Dirty Tracking Pattern
```python
def put(self, metrics: Sequence[Metric], mark_dirty: bool = False):
    """Track which metrics need persistence"""
    with self._lock:
        for metric in metrics:
            key = self._build_key(metric)
            self._cache[key] = metric

            if mark_dirty:
                self._dirty.add(key)  # Needs DB write
            else:
                self._dirty.discard(key)  # Already persisted

def write_back(self) -> int:
    """Batch persist dirty metrics"""
    with self._lock:
        dirty_metrics = [self._cache[k] for k in self._dirty if k in self._cache]
        if dirty_metrics:
            self._db.persist(dirty_metrics)
            self._dirty.clear()
        return len(dirty_metrics)
```

### Cache Benefits
- **Performance**: Reduces redundant DB queries
- **Thread-Safe**: RLock allows recursive locking
- **Monitoring**: Built-in statistics tracking
- **Fallback**: Automatic DB query on miss
- **Batch Operations**: Efficient dirty metric persistence
- **Lock-Free I/O**: Database operations outside critical sections

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

## Testing Patterns and Standards

### Core Testing Philosophy

**Quality First**: Test code quality equals source code quality. Tests should be concise, modern, and maintainable.

### 1. Real Objects Over Mocks

**Principle**: Use real objects whenever possible. Mocks hide bugs and make tests brittle.

```python
# ❌ AVOID - Mock-based testing
def test_with_mock():
    mock_db = Mock()
    mock_db.get_metric.return_value = Some(metric)

# ✅ PREFER - Real object testing
def test_with_real_objects():
    db = MetricDB(":memory:")  # Real in-memory database
    cache = MetricCache(db)     # Real cache with real DB
    provider = MetricProvider(db, ExecutionId())
```

### 2. Type Annotations Required

**All test code MUST have type annotations**:

```python
# ❌ WRONG - Missing type annotations
def test_cache_hit():
    cache = MetricCache(db)
    result = cache.get(key)

# ✅ CORRECT - Full type annotations
def test_cache_hit() -> None:
    cache: MetricCache = MetricCache(db)
    result: Maybe[Metric] = cache.get(key)
```

### 3. Testing Result/Maybe Types

**Always use pattern matching** for Result/Maybe assertions:

```python
# ❌ WRONG - Using isinstance
def test_metric_computation() -> None:
    result = compute_metric(...)
    assert isinstance(result, Success)
    assert result.unwrap() == 42.0

# ✅ CORRECT - Pattern matching
def test_metric_computation() -> None:
    result = compute_metric(...)
    match result:
        case Success(value):
            assert value == 42.0
        case Failure(error):
            pytest.fail(f"Expected Success, got Failure: {error}")

# ✅ CORRECT - Testing failure cases
def test_metric_failure() -> None:
    result = compute_metric(...)
    match result:
        case Failure(error):
            assert "expected error" in error
        case Success(_):
            pytest.fail("Expected Failure, got Success")
```

### 4. Testing Maybe Types

```python
# ✅ Testing Some case
def test_cache_hit() -> None:
    maybe_metric = cache.get(key)
    match maybe_metric:
        case Some(metric):
            assert metric.name == "test_metric"
            assert metric.value == 100.0
        case Nothing():
            pytest.fail("Expected Some, got Nothing")

# ✅ Testing Nothing case
def test_cache_miss() -> None:
    maybe_metric = cache.get(unknown_key)
    match maybe_metric:
        case Nothing():
            pass  # Expected
        case Some(_):
            pytest.fail("Expected Nothing, got Some")
```

### 5. Minimal Tests, Maximal Coverage

**Principle**: Write the fewest tests that provide the most coverage.

```python
# Instead of many specific tests, use parametrized tests:
@pytest.mark.parametrize("column,expected", [
    ("price", 100.0),
    ("quantity", 50.0),
    ("tax", 10.0),
])
def test_average_computation(column: str, expected: float) -> None:
    symbol = provider.average(column)
    result = provider.evaluate(symbol, key)
    match result:
        case Success(value):
            assert value == expected
        case Failure(error):
            pytest.fail(f"Computation failed: {error}")
```

### 6. Test-Driven Development (TDD)

**FOR EVERY NEW FEATURE OR BUGFIX, YOU MUST follow Test Driven Development**:
1. Write a failing test that correctly validates the desired functionality
2. Run the test to confirm it fails as expected
3. Write ONLY enough code to make the failing test pass
4. Run the test to confirm success
5. Refactor if needed while keeping tests green

### 7. Test Organization Pattern

```python
class TestMetricCache:
    """Test cache functionality with real objects."""

    @pytest.fixture
    def cache(self) -> MetricCache:
        """Provide real cache with in-memory DB."""
        db = MetricDB(":memory:")
        return MetricCache(db)

    def test_cache_operations(self, cache: MetricCache) -> None:
        """Test complete cache workflow."""
        # Setup
        metric = Metric(...)
        cache.put(metric)

        # Test retrieval
        result = cache.get(key)
        match result:
            case Some(retrieved):
                assert retrieved == metric
            case Nothing():
                pytest.fail("Cache should contain metric")
```

### 8. Coverage Standards

- **Target**: Maintain or exceed current coverage levels (100%)
- **Meaningful coverage**: Focus on critical paths and edge cases
- **Pragma usage**: Use `# pragma: no cover` only for truly unreachable code

### 9. Test Output Standards

- **Test output MUST BE PRISTINE TO PASS**
- If logs are expected to contain errors, these MUST be captured and tested
- If a test is intentionally triggering an error, we *must* capture and validate that the error output is as expected

### 10. Test Naming Conventions

```python
# Clear, descriptive test names
def test_cache_returns_metric_on_hit() -> None: ...
def test_cache_returns_nothing_on_miss() -> None: ...
def test_provider_falls_back_to_db_on_cache_miss() -> None: ...
```

## Git Workflow Patterns

### Core Git Philosophy

**Quality commits over quantity**: Every commit should be meaningful, well-documented, and follow conventional commit standards.

### 1. Conventional Commit Pattern

**All commits MUST follow conventional commit format**:

```text
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only changes
- `style`: Code style changes (formatting, missing semicolons, etc.)
- `refactor`: Code change that neither fixes a bug nor adds a feature
- `perf`: Performance improvement
- `test`: Adding missing tests or correcting existing tests
- `build`: Changes to build system or external dependencies
- `ci`: Changes to CI configuration files and scripts
- `chore`: Other changes that don't modify src or test files

**Examples**:
```text
feat(cache): add performance statistics tracking

fix(provider): use pattern matching for Maybe types instead of isinstance

docs(memory-bank): add testing standards documentation

refactor(analyzer): simplify SQL generation logic
```

### 2. Commit Workflow Pattern

**Step-by-step process for committing changes**:

1. **Read Context First**:
```bash
git --no-pager diff
git --no-pager status
# Review README.md, related code, and tests
```

2. **Prepare Commit Message**:
   - **< 10 lines** total
   - Focus on **high-level goal**
   - Avoid mundane details
   - Use conventional commit format

3. **For Simple Commits**:
```bash
git add <files>  # Never use -A without checking status
git commit -m "type(scope): description"
```

4. **For Complex Commits**:
```bash
# Create message file
echo "type(scope): description

Body explaining the high-level goal..." > .tmp/commit-msg.txt

# Commit with file
git commit --file .tmp/commit-msg.txt

# Clean up
rm .tmp/commit-msg.txt
```

5. **Always Get Permission**:
   - Show complete commit message
   - Wait for approval before executing
   - Never auto-push

### 3. Branch Management Pattern

**Naming Conventions**:
```text
feature/description-of-feature
bugfix/issue-description
release/version-number
hotfix/urgent-fix-description
```

**Rules**:
- Use lowercase with hyphens
- Be descriptive but concise
- Include issue number if applicable: `feature/123-user-authentication`
- No dots at start or slashes at end
- Case-sensitive (feature ≠ Feature)

**Workflow**:
```bash
# Create feature branch
git checkout -b feature/cache-statistics

# Create bugfix branch
git checkout -b bugfix/pattern-matching-fix

# Create WIP branch when task unclear
git checkout -b wip/exploring-options
```

### 4. Git Command Patterns

**Always use --no-pager**:
```bash
git --no-pager log --oneline -5
git --no-pager diff
git --no-pager status
git --no-pager show HEAD
```

**Safe Add Patterns**:
```bash
# Check first
git --no-pager status

# Add specific files
git add src/dqx/cache.py tests/test_cache.py

# Only use -A after verifying
git --no-pager status
git add -A  # Only if all changes intended
```

**Commit Frequently**:
- Commit after each logical change
- Don't wait for "perfect" state
- Small, focused commits preferred
- Even commit journal/documentation updates

### 5. Pre-commit Hook Pattern

**Never skip pre-commit hooks**:
```bash
# If pre-commit fails
uv run pre-commit run --all-files

# Fix issues and try again
git add <fixed-files>
git commit  # Hooks run automatically
```

**Common Hook Fixes**:
- Ruff formatting: `uv run ruff check --fix`
- Type checking: Fix type annotations
- YAML/Shell: Fix syntax issues

### 6. Git Workflow Integration

**With Development Process**:
1. Create branch following naming convention
2. Make changes following coding standards
3. Commit frequently with conventional commits
4. Run tests and ensure coverage
5. Create PR with clear, high-level description

**Example Full Workflow**:
```bash
# 1. Start new feature
git checkout -b feature/testing-standards

# 2. Make changes
# ... edit files ...

# 3. Check and commit
git --no-pager status
git add memory-bank/systemPatterns.md
git commit -m "docs(memory-bank): add testing standards section"

# 4. Continue work and commit
# ... more edits ...
git add memory-bank/techContext.md
git commit -m "docs(memory-bank): update testing philosophy"

# 5. Push and create PR
git push -u origin feature/testing-standards
gh pr create --title "docs(memory-bank): add comprehensive testing standards" \
             --body "Added testing patterns and philosophy to memory bank

Emphasizes real objects over mocks and pattern matching for Result types."
```

## Coding Standards

### Core Principles

- Follow PEP 8 style guide for Python code
- Use 4-space indentation (no tabs)
- **Type hints required**: `def func(x: int) -> str:` (mandatory)
- Use docstrings for all public modules, classes, and functions
- **Always use f-strings**: `f"Value: {var}"` not `"Value: " + str(var)`

### Modern Type Hints (PEP 604)

```python
# ✅ CORRECT - Modern syntax
def process_data(items: list[str], config: dict[str, int] | None = None) -> str | None:
    return f"Processed {len(items)} items"

# ❌ AVOID - Old syntax
from typing import Dict, List, Optional, Union

def process_data(
    items: List[str], config: Optional[Dict[str, int]] = None
) -> Union[str, None]:
    return "Processed " + str(len(items)) + " items"
```

### Naming Conventions

- Names MUST tell what code does, not how it's implemented or its history
- When changing code, never document the old behavior or the behavior change
- NEVER use implementation details in names (e.g., "ZodValidator", "MCPWrapper", "JSONParser")
- NEVER use temporal/historical context in names (e.g., "NewAPI", "LegacyHandler", "UnifiedTool", "ImprovedInterface", "EnhancedParser")
- NEVER use pattern names unless they add clarity (e.g., prefer "Tool" over "ToolFactory")

### Code Comments

- NEVER add comments explaining that something is "improved", "better", "new", "enhanced", or referencing what it used to be
- NEVER add instructional comments telling developers what to do ("copy this pattern", "use this instead")
- Comments should explain WHAT the code does or WHY it exists, not how it's better than something else
- If you're refactoring, remove old comments - don't add new ones explaining the refactoring
- YOU MUST NEVER remove code comments unless you can PROVE they are actively false
- YOU MUST NEVER add comments about what used to be there or how something has changed
- YOU MUST NEVER refer to temporal context in comments

Examples:
```python
# BAD: This uses Zod for validation instead of manual checking
# BAD: Refactored from the old validation system
# BAD: Wrapper around MCP tool protocol
# GOOD: Executes tools with validated arguments
```

### Writing Code Principles

- When submitting work, verify that you have FOLLOWED ALL RULES
- YOU MUST make the SMALLEST reasonable changes to achieve the desired outcome
- We STRONGLY prefer simple, clean, maintainable solutions over clever or complex ones
- YOU MUST WORK HARD to reduce code duplication, even if the refactoring takes extra effort
- YOU MUST NEVER throw away or rewrite implementations without EXPLICIT permission
- YOU MUST get Nam's explicit approval before implementing ANY backward compatibility
- YOU MUST MATCH the style and formatting of surrounding code
- Fix broken things immediately when you find them. Don't ask permission to fix bugs

### Proactiveness

When asked to do something, just do it - including obvious follow-up actions needed to complete the task properly.
Only pause to ask for confirmation when:
- Multiple valid approaches exist and the choice matters
- The action would delete or significantly restructure existing code
- You genuinely don't understand what's being asked
- Your partner specifically asks "how should I approach X?"

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

### ❌ Mock-Heavy Tests
Tests that mock everything test nothing real.

### ❌ Type-Unsafe Tests
Tests without type annotations are error-prone.

### ❌ isinstance with Result/Maybe
Always use pattern matching for functional types.

### ❌ Over-Engineering
- Abstract base classes for single implementations
- Configuration systems for hardcoded values
- Generic solutions for specific problems
- Premature performance optimizations
- Complex inheritance hierarchies
- Over-flexible APIs with many parameters
- Caching systems without proven performance needs
- Event systems for simple function calls

### ✅ What to Prefer Instead
- Concrete implementations that work
- Hardcoded values that can be extracted later
- Specific solutions for specific problems
- Simple, readable code first
- Composition over inheritance
- Simple function signatures
- Direct computation until performance matters
- Direct function calls for simple interactions
