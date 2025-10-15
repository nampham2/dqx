# DQX System Patterns

## System Architecture Overview

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

## Core Design Patterns

### 1. Graph-Based Dependency Management
**Pattern**: Directed Acyclic Graph (DAG)
**Implementation**: Hierarchical node structure with typed relationships

```
# Node hierarchy with strongly typed parents
RootNode (parent: None)
└── CheckNode (parent: RootNode)
    ├── AssertionNode (parent: CheckNode)
    └── SymbolNode (parent: CheckNode)
        └── MetricNode (parent: SymbolNode)
            └── AnalyzerNode (parent: MetricNode)
```

**Benefits**:
- Automatic dependency resolution
- Efficient execution planning
- Clear visualization of relationships
- Prevents circular dependencies

### 2. Composite Pattern
**Used In**: Graph node system
**Implementation**: BaseNode with CompositeNode and LeafNode variants

```python
class BaseNode(ABC):
    """Abstract base for all nodes"""

    @abstractmethod
    def accept(self, visitor: NodeVisitor) -> Any: ...


class CompositeNode(BaseNode):
    """Nodes that can have children"""

    def add_child(self, child: BaseNode) -> None: ...
    def get_children(self) -> list[BaseNode]: ...


class LeafNode(BaseNode):
    """Terminal nodes without children"""

    pass
```

### 3. Visitor Pattern
**Used In**: Graph traversal and operations
**Implementation**: Double dispatch for extensible operations

```python
class NodeVisitor(Protocol):
    def visit(self, node: BaseNode) -> Any: ...


class GraphTraverser(NodeVisitor):
    def visit(self, node: BaseNode) -> None:
        # Process node
        if isinstance(node, CompositeNode):
            for child in node.get_children():
                child.accept(self)
```

### 4. Direct Construction Pattern
**Used In**: Verification suite construction (v0.5.0+)
**Implementation**: Direct instantiation with all parameters

```python
# Direct instantiation - the only way to create suites
suite = VerificationSuite([check1, check2, check3], db, "Suite Name")
```

**History**: The VerificationSuiteBuilder was removed in v0.5.0 as it added unnecessary complexity without significant benefits. Direct construction is now the only way to create verification suites, providing a cleaner, more maintainable API. The suite no longer accepts tags parameter - tags are now associated with individual checks.

### 5. Two-Stage Builder Pattern
**Used In**: Assertion creation (v0.4.0+)
**Implementation**: Enforces mandatory properties through type system

```python
# Stage 1: AssertionDraft (requires name)
draft = ctx.assert_that(mp.average("price"))

# Stage 2: AssertionReady (all assertion methods available)
ready = draft.where(name="Average price is positive", severity="P1")
ready.is_positive()  # Creates AssertionNode
```

**Benefits**:
- Compile-time safety for required properties
- Clear error messages when name is missing
- Better debugging with descriptive assertion names
- Prevents unnamed assertions in production

**Pattern Structure**:
```
AssertionDraft (incomplete)
    ↓ .where(name=...) [required transition]
AssertionReady (complete)
    ↓ .is_gt(), .is_eq(), etc. [terminal operations]
AssertionNode (immutable result)
```

### 6. Strategy Pattern
**Used In**: Metric specifications and SQL operations
**Implementation**: Each MetricSpec defines computation strategy

```python
@runtime_checkable
class MetricSpec(Protocol):
    metric_type: MetricType

    @property
    def analyzers(self) -> Sequence[Op]: ...
```

### 7. Observer Pattern
**Used In**: Symbol state management
**Implementation**: AssertionNodes observe SymbolNode state changes

```python
class SymbolStateObserver(Protocol):
    def on_symbol_state_change(self, symbol: sp.Symbol, state: SymbolState) -> None: ...


class AssertionNode(LeafNode, SymbolStateObserver):
    def on_symbol_state_change(self, symbol: sp.Symbol, state: SymbolState):
        # React to symbol becoming ready/error
        pass
```

## Key Technical Decisions

### 1. Symbolic Mathematics (SymPy)
**Decision**: Use SymPy for expression handling
**Rationale**:
- Natural mathematical syntax
- Automatic simplification
- Type-safe operations
- Lazy evaluation support

**Trade-offs**:
- Additional dependency
- Learning curve for complex expressions
- Performance overhead for simple operations

### 2. SQL Generation Strategy
**Decision**: Generate SQL dynamically with dialects
**Implementation**:
```python
class Dialect(Protocol):
    name: str

    def translate_sql_op(self, op: SqlOp) -> str: ...
    def build_cte_query(self, cte_sql: str, expressions: list[str]) -> str: ...
```

**Benefits**:
- Support multiple SQL engines
- Optimize for each platform
- Maintain consistent semantics

### 3. Statistical Sketching Algorithms
**Decision**: HyperLogLog for cardinality, DataSketches for distributions
**Implementation Details**:
- HyperLogLog: 99.9% accuracy with 1.5KB memory
- DataSketches: Configurable accuracy/memory trade-off
- Mergeable for distributed computation

### 4. Immutable Node Design
**Decision**: Make graph nodes immutable after construction
**Implementation**: All properties set via constructor, no setters

```python
class AssertionNode:
    def __init__(self, name: str, severity: str, *args, **kwargs):
        self._name = name
        self._severity = severity
        # No set_name() or set_severity() methods
```

**Benefits**:
- Thread safety
- Predictable behavior
- Easier debugging
- Prevents accidental modification

### 5. Protocol-Based Interfaces
**Decision**: Use Python Protocol instead of ABC inheritance
**Rationale**:
- Structural typing flexibility
- No inheritance hierarchy required
- Better for external integrations
- Runtime checkability

## Component Relationships

### 1. API → Graph Flow
```
User Check Function
    ↓
MetricProvider (creates symbols)
    ↓
Context (builds assertions)
    ↓
Graph Construction
    ↓
Dependency Resolution
```

### 2. Graph → Execution Flow
```
Graph.pending_metrics()
    ↓
Analyzer.analyze(datasource, metrics)
    ↓
SQL Generation (with deduplication)
    ↓
DuckDB Execution
    ↓
State Updates in Graph
```

### 3. Symbol → Metric → Value Flow
```
SymbolicMetric (x_1)
    ↓
RetrievalFn (compute.simple_metric)
    ↓
MetricDB lookup
    ↓
Result[float, str]
    ↓
Assertion Evaluation
```

## Critical Implementation Paths

### 1. Check Execution Path
1. Suite.run() called with datasources
2. DatasetImputationVisitor validates datasets
3. For each dataset:
   - Get pending metrics from graph
   - Analyzer generates and executes SQL
   - Results stored in MetricDB
4. Graph evaluates all assertions
5. Context returned with results

### 2. Metric Computation Path
1. MetricSpec defines analyzers (SQL operations)
2. Analyzer deduplicates operations
3. SQL generated based on dialect
4. DuckDB executes queries
5. Results transformed to States
6. States persisted to MetricDB

### 3. Assertion Evaluation Path
1. AssertionNode contains SymPy expression
2. Symbols resolved to values via MetricProvider
3. Expression evaluated with substituted values
4. Validator applied (comparison, custom logic)
5. Result (Success/Failure) propagated to CheckNode

## Performance Optimization Patterns

### 1. Single-Pass Processing
- Combine multiple metrics in single SQL query
- Efficient execution through DuckDB's query engine
- Operation deduplication for performance

### 2. Lazy Evaluation
- Metrics computed only when needed
- Symbolic expressions evaluated on demand
- Graph traversal stops at satisfied dependencies

### 3. Caching Strategy
- Computed metrics stored in MetricDB
- Symbol values cached during evaluation
- Graph state preserved across operations

### 4. Memory Management
- Statistical sketches for large datasets
- Efficient single-pass processing
- Minimal object allocation in hot paths

## Error Handling Patterns

### 1. Result Type Pattern
```python
Result[T, E] = Success[T] | Failure[E]


# Usage throughout codebase
def compute_metric() -> Result[float, str]:
    if success:
        return Success(value)
    else:
        return Failure("Error message")
```

### 2. Graceful Degradation
- Missing metrics return Failure, not exception
- Partial results available even with failures
- Clear error propagation through graph

### 3. Validation Layers
1. API validation (empty names, invalid params)
2. Graph validation (dataset availability)
3. Execution validation (SQL errors)
4. Result validation (NaN, infinity handling)

## Extension Patterns

### 1. Custom Metrics
```python
class CustomMetric:
    metric_type = "Custom"

    @property
    def analyzers(self) -> Sequence[Op]:
        return [CustomOp(self.params)]
```

### 2. Custom Data Sources
```python
class CustomDataSource:
    name = "custom"
    dialect = "duckdb"

    @property
    def cte(self) -> str:
        return "SELECT * FROM custom_table"
```

### 3. Custom Validators
```python
def within_range(value: float, min_val: float, max_val: float) -> bool:
    return min_val <= value <= max_val


ctx.assert_that(metric).where(
    validator=SymbolicValidator("range", partial(within_range, min_val=0, max_val=100))
)
```

## Anti-Patterns to Avoid

### 1. **Mutable Shared State**
- Don't modify nodes after construction
- Don't share mutable objects between nodes
- Don't rely on execution order for state

### 2. **Tight Coupling**
- Don't directly reference implementation classes
- Don't bypass abstraction layers
- Don't mix concerns (e.g., display in computation)

### 3. **Synchronous Blocking**
- Don't block on individual metric computation
- Don't serialize independent operations
- Don't hold locks during I/O

### 4. **Over-Engineering**
- Don't add abstraction without clear benefit
- Don't optimize prematurely
- Don't generalize from single use case

## Future Architecture Considerations

### 1. **Streaming Architecture**
- Event-driven metric updates
- Incremental computation support
- Real-time alerting integration

### 2. **Distributed Execution**
- Partition-aware computation
- Cross-region metric aggregation
- Federated validation

### 3. **Plugin Architecture**
- Dynamic metric loading
- Custom dialect plugins
- Third-party integrations

### 4. **Observability Layer**
- OpenTelemetry integration
- Metric computation tracing
- Performance profiling hooks
