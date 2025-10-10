# Question 1
What's the hierarchy of the graph? One main goal of dqx is to transform the test suite in to a graph, validate the input against the graph, collect metrics and update graph then evaluate the assertions and propagate results to the parent nodes.

# Answer 1
Based on my analysis of the dqx codebase, here's the complete hierarchy of the graph structure:

## Graph Hierarchy

### 1. __RootNode__ (Top Level)

- The root of the verification graph

- Contains multiple `CheckNode` children

- Manages traversal and propagation across the entire graph

- Key responsibilities:

  - Traverse the graph to find specific node types (assertions, checks, metrics, symbols)
  - Propagate dataset information down the hierarchy
  - Mark pending metrics as success/failure

### 2. __CheckNode__ (Second Level)

- Represents a data quality check

- Children can be `AssertionNode` or `SymbolNode`

- Has properties: name, tags, label, datasets

- Key responsibilities:

  - Validate dataset requirements
  - Propagate dataset information to children
  - Update status based on children's success/failure

### 3. __AssertionNode__ and __SymbolNode__ (Third Level)

#### AssertionNode (Leaf)

- Represents an assertion to be evaluated
- Contains: actual expression (sympy), validator, severity, label
- Cannot have children
- Evaluates expressions using symbol values from the graph

#### SymbolNode (Composite)

- Represents a symbol that can be evaluated
- Contains: symbol (sympy), retrieval function, dataset (singular - one symbol maps to one dataset)
- Children are `MetricNode` instances
- States: ready (all metrics provided), success, failure

### 4. __MetricNode__ (Fourth Level)

- Represents a metric to be computed
- Children are `AnalyzerNode` instances
- States: PENDING, PROVIDED, ERROR
- Contains: metric specification, key provider, datasets

### 5. __AnalyzerNode__ (Fifth Level - Leaf)

- Represents an analyzer operation
- Contains the actual analyzer logic
- Cannot have children

## Data Flow Process

1. __Graph Construction__:

   - Checks are decorated functions that create assertions
   - Assertions reference symbols (e.g., `count`, `sum`, `avg`)
   - Symbols depend on metrics
   - Metrics have analyzers that compute values from data

2. __Propagation__ (Top-Down):

   - RootNode receives dataset names
   - CheckNodes validate and propagate datasets to children
   - SymbolNodes propagate to MetricNodes
   - Failed dataset validations mark nodes as failed

3. __Evaluation__ (Bottom-Up):

   - Analyzers compute metrics from datasets
   - Metrics provide values to symbols
   - Symbols evaluate using retrieval functions
   - Assertions evaluate using symbol values
   - CheckNodes aggregate children's status

4. __Result Propagation__:

   - Failed metrics propagate failure to parent symbols
   - Failed symbols propagate to assertions
   - Failed assertions propagate to parent checks
   - Final status bubbles up to root

## Example Flow

```python
@check
def my_check(mp: MetricProvider, ctx: Context):
    # Creates AssertionNode as child of CheckNode
    ctx.assert_that(mp.count("col") > 100).where(name="Count check")

# Graph structure:
# RootNode
# └── CheckNode("my_check")
#     └── AssertionNode(count > 100)
#         └── SymbolNode(count)
#             └── MetricNode(CountMetric)
#                 └── AnalyzerNode(CountAnalyzer)
```

This hierarchical structure enables:

- Efficient dependency resolution
- Clear separation of concerns
- Flexible metric computation and caching
- Comprehensive error propagation
- Easy traversal and inspection

# Question 2
What changes to the tree structure would make it better fit the purpose of dqx ?

# Answer 2
Based on dqx's core purpose of transforming test suites into a graph, validating inputs, collecting metrics, and propagating results, here are several structural improvements that could make the tree better fit its purpose:

## 1. __Explicit Dependency Graph Layer__

Instead of mixing organizational hierarchy with computational dependencies, introduce a separate dependency resolution layer:

```python
class DependencyNode(CompositeNode):
    """Explicit node for managing dependencies between symbols and assertions"""
    def __init__(self, target: AssertionNode | SymbolNode):
        self.target = target
        self.dependencies: list[SymbolNode] = []
        self.is_resolved = False

# New structure:
CheckNode
├── DependencyNode(AssertionNode)
│   └── References: [SymbolNode1, SymbolNode2]
└── SymbolNode (with metrics)
```

__Benefits:__

- Clear separation between logical organization and computational dependencies
- Easier to validate circular dependencies
- Better visualization of data flow

## 2. __Lazy Evaluation Nodes__

Add nodes that defer computation until needed:

```python
class LazyMetricNode(MetricNode):
    """Computes metrics only when requested by symbols"""
    def __init__(self, spec, key_provider, nominal_key):
        super().__init__(spec, key_provider, nominal_key)
        self._computation_promise = None

    def compute_when_needed(self) -> Promise[Metric]:
        if not self._computation_promise:
            self._computation_promise = self._create_computation()
        return self._computation_promise
```

__Benefits:__

- Avoid computing unused metrics
- Better resource utilization
- Support for streaming/incremental computation

## 3. __Multi-Dataset Aware Structure__

Make dataset handling a first-class citizen in the hierarchy:

```python
class DatasetNode(CompositeNode):
    """Groups all computations for a specific dataset"""
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.metrics: list[MetricNode] = []
        self.status = "PENDING"

# New structure:
RootNode
└── CheckNode
    ├── DatasetNode("orders")
    │   ├── MetricNode(count)
    │   └── MetricNode(sum)
    └── DatasetNode("customers")
        ├── MetricNode(count)
        └── MetricNode(distinct)
```

__Benefits:__

- Clearer dataset isolation
- Easier parallel processing per dataset
- Better error isolation (one dataset failure doesn't affect others)

## 4. __Assertion Groups with Shared Context__

Group related assertions that share validation context:

```python
class AssertionGroupNode(CompositeNode):
    """Groups assertions with shared symbols and validation logic"""
    def __init__(self, name: str):
        self.name = name
        self.shared_symbols: dict[str, SymbolNode] = {}
        self.shared_context: dict[str, Any] = {}

# Example:
CheckNode
└── AssertionGroupNode("completeness_checks")
    ├── AssertionNode(null_count == 0)
    ├── AssertionNode(distinct_count > min_threshold)
    └── SharedSymbols: {null_count, distinct_count}
```

__Benefits:__

- Better organization of related assertions
- Shared computation context
- Easier to apply group-level policies (e.g., "fail fast" for critical groups)

## 5. __Result Aggregation Nodes__

Add explicit nodes for result aggregation strategies:

```python
class AggregationNode(CompositeNode):
    """Defines how child results should be combined"""
    def __init__(self, strategy: Literal["all", "any", "threshold"]):
        self.strategy = strategy
        self.threshold = None  # For threshold strategy

    def aggregate(self, results: list[Result]) -> Result:
        # Implement aggregation logic
```

__Benefits:__

- Flexible failure propagation strategies
- Support for partial success scenarios
- Better control over check severity

## 6. __Caching and Memoization Nodes__

Make caching explicit in the structure:

```python
class CachedSymbolNode(SymbolNode):
    """Symbol node with explicit caching behavior"""
    def __init__(self, name, symbol, fn, dataset, cache_key: str):
        super().__init__(name, symbol, fn, dataset)
        self.cache_key = cache_key
        self.cache_ttl = None
        self.cache_conditions = []
    """Symbol node with explicit caching behavior"""
    def __init__(self, name, symbol, fn, dataset, cache_key: str):
        super().__init__(name, symbol, fn, dataset)
        self.cache_key = cache_key
        self.cache_ttl = None
        self.cache_conditions = []
```

__Benefits:__

- Clear cache invalidation rules
- Better debugging of cached vs. fresh computations
- Support for different caching strategies per symbol

## 7. __Error Context Nodes__

Add nodes that capture and enrich error context:

```python
class ErrorContextNode(CompositeNode):
    """Captures error context and recovery strategies"""
    def __init__(self, parent: BaseNode):
        self.parent = parent
        self.error_handlers: list[ErrorHandler] = []
        self.fallback_values: dict[str, Any] = {}
        self.error_trace: list[str] = []
```

__Benefits:__

- Rich error messages with full context
- Support for error recovery strategies
- Better debugging and troubleshooting

## Recommended Combined Structure

The most impactful improvement would be combining several of these concepts:

```javascript
RootNode
└── CheckNode
    ├── AssertionGroupNode("critical_validations")
    │   ├── DependencyNode(AssertionNode)
    │   │   └── References: [CachedSymbolNode]
    │   └── ErrorContextNode
    └── DatasetNode("primary_dataset")
        ├── LazyMetricNode
        └── AggregationNode
```

This structure would:

1. Make dependencies explicit
2. Support efficient computation strategies
3. Provide clear error propagation paths
4. Enable flexible result aggregation
5. Maintain backward compatibility with existing API

Would you like me to elaborate on any of these improvements or discuss implementation strategies for specific ones?

# Question 3
Elaborate on an implementation strategies for the above

# Answer 3
Let me provide detailed implementation strategies for each major improvement, focusing on practical approaches that maintain backward compatibility while enhancing dqx's capabilities.

## 1. __Explicit Dependency Graph Layer - Implementation Strategy__

### Phased Approach

__Phase 1: Add Dependency Tracking (Non-Breaking)__

```python
# Extend existing nodes with dependency tracking
class DependencyMixin:
    """Mixin to add dependency tracking to existing nodes"""
    def __init__(self):
        self._dependencies: set[BaseNode] = set()
        self._dependents: set[BaseNode] = set()
        self._resolution_order: int | None = None

    def add_dependency(self, node: BaseNode) -> None:
        self._dependencies.add(node)
        if hasattr(node, '_dependents'):
            node._dependents.add(self)

    def get_resolution_order(self) -> int:
        """Topological sort order for evaluation"""
        if self._resolution_order is None:
            max_dep_order = max(
                (dep.get_resolution_order() for dep in self._dependencies),
                default=-1
            )
            self._resolution_order = max_dep_order + 1
        return self._resolution_order
```

__Phase 2: Parallel Dependency Resolution__

```python
class DependencyResolver:
    """Resolves dependencies in parallel where possible"""
    def __init__(self, executor: ThreadPoolExecutor):
        self.executor = executor
        self._futures: dict[BaseNode, Future] = {}

    def resolve_async(self, node: BaseNode) -> Future:
        if node in self._futures:
            return self._futures[node]

        # Wait for dependencies
        dep_futures = [
            self.resolve_async(dep)
            for dep in node._dependencies
        ]

        # Schedule node resolution
        future = self.executor.submit(
            self._resolve_node, node, dep_futures
        )
        self._futures[node] = future
        return future
```

## 2. __Lazy Evaluation Nodes - Implementation Strategy__

### Progressive Enhancement

__Step 1: Add Computation Promises__

```python
from abc import abstractmethod
from concurrent.futures import Future
from functools import cached_property

class ComputationPromise(Protocol):
    """Promise for deferred computation"""
    @abstractmethod
    def compute(self) -> Result[float, str]: ...

    @abstractmethod
    def is_ready(self) -> bool: ...

class LazyMetricNode(MetricNode):
    """Metric node with lazy evaluation support"""
    def __init__(self, spec, key_provider, nominal_key):
        super().__init__(spec, key_provider, nominal_key)
        self._computation: ComputationPromise | None = None
        self._compute_on_access = True

    @cached_property
    def value(self) -> Result[float, str]:
        """Compute value only when accessed"""
        if self._computation is None:
            self._computation = self._create_computation()
        return self._computation.compute()

    def _create_computation(self) -> ComputationPromise:
        """Factory method for creating computation promise"""
        return MetricComputation(
            self.spec,
            self.datasets,
            self.eval_key()
        )
```

__Step 2: Batch Computation Optimization__

```python
class BatchComputationManager:
    """Manages batch computation of multiple lazy nodes"""
    def __init__(self):
        self._pending: list[LazyMetricNode] = []
        self._batch_size = 100

    def add_for_computation(self, node: LazyMetricNode) -> None:
        self._pending.append(node)
        if len(self._pending) >= self._batch_size:
            self._compute_batch()

    def _compute_batch(self) -> None:
        """Compute all pending nodes efficiently"""
        # Group by dataset and metric type
        grouped = self._group_by_dataset_and_type(self._pending)

        # Execute batch computations
        for (dataset, metric_type), nodes in grouped.items():
            results = self._batch_compute(dataset, metric_type, nodes)
            for node, result in zip(nodes, results):
                node._set_computed_value(result)

        self._pending.clear()
```

## 3. __Multi-Dataset Aware Structure - Implementation Strategy__

### Incremental Migration

__Stage 1: Dataset Context Manager__

```python
class DatasetContext:
    """Manages dataset-specific computation context"""
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.metrics: dict[str, MetricNode] = {}
        self.symbols: dict[str, SymbolNode] = {}
        self.computation_state = "PENDING"
        self.error_context: list[str] = []

    def add_metric(self, metric: MetricNode) -> None:
        key = f"{metric.spec.name}:{metric.eval_key()}"
        self.metrics[key] = metric

    def mark_failed(self, error: str) -> None:
        self.computation_state = "FAILED"
        self.error_context.append(error)
        # Propagate failure to all metrics
        for metric in self.metrics.values():
            metric.mark_as_failure(f"Dataset {self.dataset_name} failed: {error}")

# Integrate with existing CheckNode
class CheckNode(CompositeNode):
    def __init__(self, name, tags=None, label=None, datasets=None):
        super().__init__()
        # ... existing init code ...
        self._dataset_contexts: dict[str, DatasetContext] = {}

    def get_or_create_dataset_context(self, dataset: str) -> DatasetContext:
        if dataset not in self._dataset_contexts:
            self._dataset_contexts[dataset] = DatasetContext(dataset)
        return self._dataset_contexts[dataset]
```

__Stage 2: Parallel Dataset Processing__

```python
class DatasetExecutor:
    """Executes computations per dataset in parallel"""
    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def execute_checks(
        self,
        root: RootNode,
        datasources: dict[str, SqlDataSource]
    ) -> dict[str, Future[DatasetResult]]:
        futures = {}

        for dataset_name, datasource in datasources.items():
            future = self.executor.submit(
                self._execute_dataset,
                root, dataset_name, datasource
            )
            futures[dataset_name] = future

        return futures

    def _execute_dataset(
        self,
        root: RootNode,
        dataset_name: str,
        datasource: SqlDataSource
    ) -> DatasetResult:
        try:
            # Process metrics for this dataset
            metrics = list(root.pending_metrics(dataset_name))
            analyzer = Analyzer()
            analyzer.analyze(datasource, metrics, threading=True)
            return DatasetResult(dataset_name, "SUCCESS", analyzer.results)
        except Exception as e:
            return DatasetResult(dataset_name, "FAILED", str(e))
```

## 4. __Assertion Groups - Implementation Strategy__

### API-Compatible Enhancement

__Option 1: Decorator-Based Grouping__

```python
from contextlib import contextmanager

class AssertionGroupManager:
    """Manages assertion grouping within checks"""
    def __init__(self):
        self._current_group: AssertionGroupNode | None = None
        self._groups: dict[str, AssertionGroupNode] = {}

    @contextmanager
    def group(self, name: str, **kwargs):
        """Context manager for assertion grouping"""
        prev_group = self._current_group

        # Create or get group
        if name not in self._groups:
            self._groups[name] = AssertionGroupNode(
                name=name,
                severity=kwargs.get('severity'),
                fail_fast=kwargs.get('fail_fast', False)
            )

        self._current_group = self._groups[name]
        try:
            yield self._current_group
        finally:
            self._current_group = prev_group

    def current_group(self) -> AssertionGroupNode | None:
        return self._current_group

# Updated check implementation
@check
def quality_check(mp: MetricProvider, ctx: Context) -> None:
    group_mgr = AssertionGroupManager()

    with group_mgr.group("critical", severity="ERROR", fail_fast=True):
        ctx.assert_that(mp.count("id") > 0).where(name="Has records")
        ctx.assert_that(mp.null_ratio("id") == 0).where(name="No null IDs")

    with group_mgr.group("warnings", severity="WARN"):
        ctx.assert_that(mp.distinct_ratio("category") > 0.1)
```

__Option 2: Builder Pattern__

```python
class AssertionGroupBuilder:
    """Fluent API for building assertion groups"""
    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.assertions: list[SymbolicAssert] = []
        self.group_config = {}

    def assert_that(self, expr: sp.Expr) -> Self:
        assertion = self.ctx.assert_that(expr)
        self.assertions.append(assertion)
        return self

    def with_severity(self, severity: SeverityLevel) -> Self:
        self.group_config['severity'] = severity
        for assertion in self.assertions:
            assertion.where(severity=severity)
        return self

    def fail_fast(self) -> Self:
        self.group_config['fail_fast'] = True
        return self

    def build(self, name: str) -> AssertionGroupNode:
        group = AssertionGroupNode(name, **self.group_config)
        for assertion in self.assertions:
            group.add_child(assertion._node)
        return group
```

## 5. __Result Aggregation - Implementation Strategy__

### Strategy Pattern Implementation

```python
from abc import abstractmethod
from typing import Protocol

class AggregationStrategy(Protocol):
    """Protocol for result aggregation strategies"""
    @abstractmethod
    def aggregate(self, results: list[Result]) -> Result: ...

    @abstractmethod
    def can_short_circuit(self) -> bool: ...

class AllSuccessStrategy(AggregationStrategy):
    """All children must succeed"""
    def aggregate(self, results: list[Result]) -> Result:
        failures = [r for r in results if isinstance(r, Failure)]
        if failures:
            return Failure(f"{len(failures)} checks failed")
        return Success(1.0)

    def can_short_circuit(self) -> bool:
        return True  # Can fail fast on first failure

class ThresholdStrategy(AggregationStrategy):
    """Configurable threshold of success"""
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold

    def aggregate(self, results: list[Result]) -> Result:
        success_rate = sum(1 for r in results if isinstance(r, Success)) / len(results)
        if success_rate >= self.threshold:
            return Success(success_rate)
        return Failure(f"Success rate {success_rate:.1%} below threshold {self.threshold:.1%}")

    def can_short_circuit(self) -> bool:
        return False  # Must evaluate all children

# Integration with CheckNode
class CheckNode(CompositeNode):
    def __init__(self, name, tags=None, label=None, datasets=None,
                 aggregation_strategy: AggregationStrategy | None = None):
        super().__init__()
        # ... existing init ...
        self.aggregation_strategy = aggregation_strategy or AllSuccessStrategy()

    def update_status(self) -> None:
        """Enhanced status update with aggregation strategy"""
        if isinstance(self._value, Some) and isinstance(self._value.unwrap(), Failure):
            return  # Already failed

        child_results = []
        for child in self.children:
            if hasattr(child, '_value') and isinstance(child._value, Some):
                child_results.append(child._value.unwrap())
            elif self.aggregation_strategy.can_short_circuit():
                return  # Child pending, can't determine result yet

        if child_results:
            self._value = Some(self.aggregation_strategy.aggregate(child_results))
```

## 6. __Caching Layer - Implementation Strategy__

### Pluggable Cache Implementation

```python
from datetime import datetime, timedelta
from typing import Protocol

class CacheBackend(Protocol):
    """Protocol for cache implementations"""
    def get(self, key: str) -> Maybe[Result]: ...
    def set(self, key: str, value: Result, ttl: timedelta | None = None) -> None: ...
    def invalidate(self, pattern: str) -> None: ...

class MemoryCacheBackend:
    """In-memory cache implementation"""
    def __init__(self):
        self._cache: dict[str, tuple[Result, datetime | None]] = {}

    def get(self, key: str) -> Maybe[Result]:
        if key in self._cache:
            result, expiry = self._cache[key]
            if expiry is None or datetime.now() < expiry:
                return Some(result)
            del self._cache[key]
        return Nothing

    def set(self, key: str, value: Result, ttl: timedelta | None = None) -> None:
        expiry = datetime.now() + ttl if ttl else None
        self._cache[key] = (value, expiry)

class CachedSymbolNode(SymbolNode):
    """Symbol node with caching capabilities"""
    _cache_backend: CacheBackend = MemoryCacheBackend()

    def __init__(self, name, symbol, fn, dataset, cache_config: dict | None = None):
        super().__init__(name, symbol, fn, dataset)
        self.cache_config = cache_config or {}
        self.cache_key_template = self.cache_config.get(
            'key_template',
            f"symbol:{name}:{{dataset}}:{{time_period}}"
        )

    def evaluate(self, key: ResultKey) -> Result[float, str]:
        cache_key = self._build_cache_key(key)

        # Try cache first
        cached = self._cache_backend.get(cache_key)
        if isinstance(cached, Some):
            self._value = cached
            return cached.unwrap()

        # Compute and cache
        result = super().evaluate(key)
        ttl = timedelta(seconds=self.cache_config.get('ttl_seconds', 3600))
        self._cache_backend.set(cache_key, result, ttl)
        return result
```

## Implementation Roadmap

### Phase 1 (Weeks 1-2): Foundation

1. Implement `DependencyMixin` for existing nodes
2. Add `DatasetContext` to `CheckNode`
3. Create basic `LazyMetricNode` with deferred computation

### Phase 2 (Weeks 3-4): Core Features

1. Implement `AssertionGroupManager` with context manager API
2. Add aggregation strategies to `CheckNode`
3. Create `CacheBackend` protocol and memory implementation

### Phase 3 (Weeks 5-6): Optimization

1. Implement parallel dataset execution
2. Add batch computation for lazy nodes
3. Optimize dependency resolution with topological sorting

### Phase 4 (Weeks 7-8): Polish

1. Add comprehensive error context tracking
2. Implement cache invalidation strategies
3. Create migration utilities for existing code

This phased approach ensures:

- Backward compatibility at each step
- Incremental value delivery
- Easy rollback if issues arise
- Clear testing boundaries

Would you like me to dive deeper into any specific implementation aspect or discuss migration strategies for existing dqx users?

# Question 4
Why do we need topological sort order for evaluation ?

# Answer 4
Great question! Let me explain why topological sort order would be valuable for dqx's evaluation process by examining the current implementation and its limitations.

## Current Implementation (Without Topological Sort)

Looking at the current dqx implementation, the evaluation happens in a somewhat implicit order:

```python
# From Context.evaluate() in api.py:
def evaluate(self, key: ResultKey) -> None:
    """Evaluate all ready symbols and assertions in the graph."""
    for symbol in self._graph.ready_symbols():
        symbol.evaluate(key)
    for assertion in self._graph.assertions():
        assertion.evaluate()
    # Update check node statuses based on their children
    for check in self._graph.checks():
        check.update_status()
```

This works, but has several limitations:

## Why Topological Sort Would Improve dqx

### 1. __Complex Symbol Dependencies__

Consider this scenario:

```python
@check
def complex_check(mp: MetricProvider, ctx: Context) -> None:
    # Symbol A depends on count
    error_rate = mp.count("errors") / mp.count("total")

    # Symbol B depends on Symbol A (indirectly)
    ctx.assert_that(error_rate < 0.05)

    # Symbol C might depend on both
    success_rate = 1 - error_rate
    ctx.assert_that(success_rate > 0.95)
```

Without topological sort, you might try to evaluate `success_rate` before `error_rate` is computed.

### 2. __Parallel Execution Optimization__

With topological sort, you can identify independent nodes at each level:

```javascript
Level 0: [count("errors"), count("total")]  # Can compute in parallel
Level 1: [error_rate]                       # Depends on level 0
Level 2: [success_rate]                     # Depends on level 1
Level 3: [assertion1, assertion2]           # Can evaluate in parallel
```

### 3. __Circular Dependency Detection__

Topological sort naturally detects circular dependencies:

```python
# This would create a circular dependency
@check
def circular_check(mp: MetricProvider, ctx: Context) -> None:
    # A depends on B, B depends on A (through some complex logic)
    # Topological sort would detect this and fail fast
```

### 4. __Efficient Cache Invalidation__

With dependency order, you know exactly what needs recomputation:

```python
# If count("errors") changes, we know to invalidate:
# 1. error_rate (direct dependent)
# 2. success_rate (transitive dependent)
# 3. Related assertions
```

### 5. __Current Implementation Issues__

The current approach has subtle issues:

```python
# From AssertionNode.evaluate():
for symbol_node in symbol_nodes:
    if symbol_node.symbol in self.actual.free_symbols:
        found_symbols.add(symbol_node.symbol)
        if symbol_node.success():
            symbol_table[symbol_node.symbol] = symbol_node._value.unwrap().unwrap()
        elif symbol_node.failure():
            failed_symbols.append(str(symbol_node.symbol))
```

This assumes symbols are already evaluated, but there's no guarantee of order.

## Benefits of Explicit Topological Sort

### 1. __Deterministic Execution__

```python
class TopologicalEvaluator:
    def evaluate(self, root: RootNode, key: ResultKey) -> None:
        # Build dependency graph
        dep_graph = self._build_dependency_graph(root)

        # Get topological order
        eval_order = self._topological_sort(dep_graph)

        # Evaluate in guaranteed order
        for node in eval_order:
            if isinstance(node, MetricNode):
                self._evaluate_metric(node)
            elif isinstance(node, SymbolNode):
                self._evaluate_symbol(node, key)
            elif isinstance(node, AssertionNode):
                self._evaluate_assertion(node)
```

### 2. __Parallel Evaluation by Level__

```python
def evaluate_parallel(self, levels: list[list[BaseNode]]) -> None:
    with ThreadPoolExecutor() as executor:
        for level in levels:
            # All nodes in a level are independent
            futures = [
                executor.submit(self._evaluate_node, node)
                for node in level
            ]
            # Wait for level completion before next level
            for future in futures:
                future.result()
```

### 3. __Smart Re-evaluation__

```python
def smart_reevaluate(self, changed_node: BaseNode) -> None:
    # Only re-evaluate nodes that depend on changed_node
    affected_nodes = self._get_downstream_nodes(changed_node)
    sorted_affected = self._topological_sort(affected_nodes)

    for node in sorted_affected:
        node.invalidate_cache()
        self._evaluate_node(node)
```

### 4. __Better Error Messages__

```python
def evaluate_with_context(self, node: BaseNode) -> Result:
    try:
        return node.evaluate()
    except DependencyError as e:
        # Can provide full dependency chain
        chain = self._get_dependency_chain(node)
        raise DQXError(
            f"Failed to evaluate {node} due to dependency failure:\n"
            f"Dependency chain: {' -> '.join(chain)}\n"
            f"Root cause: {e}"
        )
```

## When It's Most Critical

Topological sort becomes essential when:

1. __Multi-level Dependencies__: Symbols depend on other symbols
2. __Conditional Evaluation__: Some branches should only evaluate if others succeed
3. __Resource Optimization__: Expensive computations that shouldn't be repeated
4. __Streaming/Incremental__: Processing data as it arrives
5. __Distributed Execution__: Computing across multiple machines

## Current dqx Workaround

The current implementation works around this by:

1. Having a strict hierarchy (Assertions → Symbols → Metrics)
2. Evaluating "ready" symbols first (those with all metrics provided)
3. Using the `ready_symbols()` iterator

But this is limited and doesn't handle complex cases well.

Would you like me to show how we could implement topological sorting in dqx while maintaining backward compatibility?

# Question 5
We don't need backward compatibility in this plan.

# Answer 5
Excellent! Without backward compatibility constraints, we can design an optimal graph structure from scratch. Here's a radical redesign that fully embraces dqx's core purpose:

## Clean Slate Design: True Dependency Graph Architecture

### Core Principle: Dependencies as First-Class Citizens

```python
from dataclasses import dataclass
from typing import Protocol, TypeVar, Generic

T = TypeVar('T')

@dataclass
class Computation(Generic[T]):
    """Base computation unit with explicit dependencies"""
    name: str
    dependencies: list['Computation']
    compute: Callable[[dict[str, Any]], T]
    cache_key: str | None = None

class DependencyGraph:
    """Pure dependency graph without hierarchical constraints"""
    def __init__(self):
        self.nodes: dict[str, Computation] = {}
        self.edges: dict[str, set[str]] = {}  # node -> dependencies
        self._topological_order: list[str] | None = None

    def add_computation(self, computation: Computation) -> None:
        self.nodes[computation.name] = computation
        self.edges[computation.name] = {dep.name for dep in computation.dependencies}
        self._topological_order = None  # Invalidate cache

    def evaluate(self, context: EvaluationContext) -> dict[str, Any]:
        """Evaluate all computations in dependency order"""
        results = {}
        for node_name in self._get_topological_order():
            computation = self.nodes[node_name]
            # Gather dependency results
            deps = {dep: results[dep] for dep in self.edges[node_name]}
            # Execute computation
            results[node_name] = computation.compute(deps)
        return results
```

### New Node Architecture

```python
# No more inheritance hierarchy - composition over inheritance
@dataclass
class MetricComputation:
    """Computes a metric from a dataset"""
    metric_spec: MetricSpec
    dataset: str
    analyzer: Analyzer

    def compute(self, datasource: SqlDataSource) -> Metric:
        return self.analyzer.analyze(datasource, self.metric_spec)

@dataclass
class SymbolComputation:
    """Evaluates a symbol from metrics"""
    symbol: sp.Symbol
    retrieval_fn: Callable[[dict[str, Metric]], float]
    required_metrics: list[str]

    def compute(self, metrics: dict[str, Metric]) -> float:
        return self.retrieval_fn(metrics)

@dataclass
class AssertionComputation:
    """Evaluates an assertion from symbols"""
    expression: sp.Expr
    validator: Callable[[float], bool]
    severity: SeverityLevel

    def compute(self, symbol_values: dict[sp.Symbol, float]) -> AssertionResult:
        value = self.expression.subs(symbol_values)
        passed = self.validator(float(value))
        return AssertionResult(
            passed=passed,
            value=value,
            severity=self.severity
        )

@dataclass
class CheckComputation:
    """Aggregates assertion results"""
    name: str
    aggregation_strategy: AggregationStrategy
    assertions: list[str]  # Names of assertion computations

    def compute(self, assertion_results: dict[str, AssertionResult]) -> CheckResult:
        results = [assertion_results[name] for name in self.assertions]
        return self.aggregation_strategy.aggregate(results)
```

### Execution Engine

```python
class ExecutionEngine:
    """Manages computation execution with advanced features"""

    def __init__(self, graph: DependencyGraph):
        self.graph = graph
        self.cache = ComputationCache()
        self.executor = ProcessPoolExecutor()  # True parallelism

    def execute(self, datasources: dict[str, SqlDataSource]) -> ExecutionResult:
        # Phase 1: Build execution plan
        plan = self._build_execution_plan(datasources)

        # Phase 2: Execute with maximum parallelism
        with self.executor:
            results = self._execute_parallel(plan)

        # Phase 3: Aggregate results
        return self._aggregate_results(results)

    def _build_execution_plan(self, datasources: dict[str, SqlDataSource]) -> ExecutionPlan:
        """Creates optimized execution plan with batching and caching"""
        levels = self._compute_dependency_levels()

        plan = ExecutionPlan()
        for level in levels:
            # Group computations by type and dataset for batching
            batches = self._create_batches(level, datasources)
            plan.add_level(batches)

        return plan

    def _execute_parallel(self, plan: ExecutionPlan) -> dict[str, Any]:
        """Execute plan with intelligent parallelism"""
        results = {}

        for level in plan.levels:
            # All batches in a level can run in parallel
            futures = []
            for batch in level.batches:
                if batch.can_use_cache:
                    # Check cache first
                    cached = self.cache.get_batch(batch)
                    if cached:
                        results.update(cached)
                        continue

                # Submit for parallel execution
                future = self.executor.submit(self._execute_batch, batch, results)
                futures.append((batch, future))

            # Collect results
            for batch, future in futures:
                batch_results = future.result()
                results.update(batch_results)
                self.cache.store_batch(batch, batch_results)

        return results
```

### Smart Features Without Legacy Constraints

#### 1. __Streaming Evaluation__

```python
class StreamingEngine(ExecutionEngine):
    """Processes data incrementally as it arrives"""

    def execute_streaming(self, datasource_stream: AsyncIterator[DataChunk]) -> AsyncIterator[Result]:
        async for chunk in datasource_stream:
            # Update only affected computations
            affected = self._determine_affected_computations(chunk)
            partial_results = await self._evaluate_subset(affected, chunk)
            yield partial_results
```

#### 2. __Distributed Execution__

```python
class DistributedEngine(ExecutionEngine):
    """Distributes computation across multiple nodes"""

    def __init__(self, graph: DependencyGraph, cluster: ClusterConfig):
        super().__init__(graph)
        self.cluster = cluster
        self.scheduler = TaskScheduler(cluster)

    def execute(self, datasources: dict[str, SqlDataSource]) -> ExecutionResult:
        # Partition graph for distributed execution
        partitions = self._partition_graph()

        # Schedule tasks across cluster
        tasks = []
        for partition in partitions:
            node = self.scheduler.select_node(partition)
            task = RemoteTask(partition, datasources)
            tasks.append(self.scheduler.submit(node, task))

        # Gather results
        return self._gather_distributed_results(tasks)
```

#### 3. __Adaptive Optimization__

```python
class AdaptiveEngine(ExecutionEngine):
    """Learns from execution patterns to optimize future runs"""

    def __init__(self, graph: DependencyGraph):
        super().__init__(graph)
        self.profiler = ExecutionProfiler()
        self.optimizer = GraphOptimizer()

    def execute(self, datasources: dict[str, SqlDataSource]) -> ExecutionResult:
        # Profile execution
        with self.profiler.profile() as profile:
            results = super().execute(datasources)

        # Learn from this execution
        self.optimizer.update_statistics(profile)

        # Optimize graph for next execution
        if self.optimizer.should_reoptimize():
            self.graph = self.optimizer.optimize(self.graph)

        return results
```

### Clean API Design

```python
class DataQualitySpec:
    """Declarative specification of data quality checks"""

    def __init__(self, name: str):
        self.name = name
        self.graph = DependencyGraph()

    def metric(self, name: str, spec: MetricSpec, dataset: str) -> MetricRef:
        """Define a metric computation"""
        computation = MetricComputation(spec, dataset, spec.analyzer)
        self.graph.add_computation(Computation(name, [], computation.compute))
        return MetricRef(name)

    def symbol(self, name: str, expression: Callable, *metrics: MetricRef) -> SymbolRef:
        """Define a symbol computation"""
        computation = SymbolComputation(
            sp.Symbol(name),
            expression,
            [m.name for m in metrics]
        )
        self.graph.add_computation(
            Computation(name, metrics, computation.compute)
        )
        return SymbolRef(name)

    def assert_that(self, expression: sp.Expr) -> AssertionBuilder:
        """Create an assertion"""
        return AssertionBuilder(self, expression)

    def check(self, name: str, *assertions: AssertionRef) -> CheckRef:
        """Group assertions into a check"""
        computation = CheckComputation(
            name,
            AllMustPassStrategy(),
            [a.name for a in assertions]
        )
        self.graph.add_computation(
            Computation(name, assertions, computation.compute)
        )
        return CheckRef(name)

# Usage - much cleaner!
spec = DataQualitySpec("Order Quality")

# Define metrics
order_count = spec.metric("order_count", CountMetric("order_id"), "orders")
null_count = spec.metric("null_count", NullCountMetric("order_id"), "orders")

# Define symbols
null_ratio = spec.symbol("null_ratio", lambda m: m["null_count"] / m["order_count"], null_count, order_count)

# Define assertions
no_nulls = spec.assert_that(null_ratio == 0).with_severity("ERROR")
has_data = spec.assert_that(order_count > 0).with_severity("ERROR")

# Group into check
spec.check("data_completeness", no_nulls, has_data)

# Execute
engine = AdaptiveEngine(spec.graph)
result = engine.execute({"orders": order_datasource})
```

## Key Improvements

1. __Pure Dependency Graph__: No artificial hierarchy constraints
2. __True Parallelism__: Process pool instead of threads
3. __Intelligent Batching__: Group similar operations
4. __Adaptive Optimization__: Learn from execution patterns
5. __Streaming Support__: Handle real-time data
6. __Distributed Ready__: Scale across clusters
7. __Clean Separation__: Computation definition vs execution
8. __Type Safety__: Strongly typed throughout
9. __Testability__: Easy to test individual computations
10. __Extensibility__: Add new computation types easily

This design fully embraces dqx's goal of transforming test suites into graphs while providing maximum flexibility and performance. Would you like me to elaborate on any specific aspect?

# Question 6
# Answer 6

## Refactoring Analysis: Simplifying Graph.py to 3 Node Types

Based on my analysis of the current implementation, I'll propose several approaches to refactor the graph to have only Root, Check, and Assertion nodes, with a new SymbolTable class managing symbols and their properties.

### Current Structure Overview

- __6 node types__: RootNode, CheckNode, AssertionNode, SymbolNode, MetricNode, AnalyzerNode
- __Complex hierarchy__: Metrics depend on analyzers, symbols depend on metrics
- __Mixed responsibilities__: Node structure handles both graph traversal and data computation

### Proposed SymbolTable Structure

```python
@dataclass
class SymbolEntry:
    symbol: sp.Symbol
    name: str
    datasets: list[str]
    result_key: ResultKey
    metric_spec: MetricSpec | None
    ops: list[Op] | None
    retrieval_fn: RetrievalFn
    value: Maybe[Result[float, str]]
    state: Literal["PENDING", "READY", "PROVIDED", "ERROR"]
    dependencies: list[tuple[MetricSpec, ResultKeyProvider]]

class SymbolTable:
    def __init__(self):
        self._symbols: dict[sp.Symbol, SymbolEntry] = {}

    def register(self, symbol: sp.Symbol, entry: SymbolEntry) -> None: ...
    def get(self, symbol: sp.Symbol) -> SymbolEntry | None: ...
    def get_pending_metrics(self, dataset: str) -> list[MetricSpec]: ...
    def evaluate_ready_symbols(self, key: ResultKey) -> None: ...
```

## Approach 1: Flat Symbol Management

__Description__: Move all symbol, metric, and analyzer logic into SymbolTable. Nodes only track relationships.

__Structure__:

- Root → Check → Assertion (graph only tracks logical grouping)
- SymbolTable manages all computation logic
- Assertions reference symbols by name in SymbolTable

__Pros__:

- Clean separation of concerns (graph structure vs computation)
- Simplified node classes
- Easier to test symbol logic independently
- Better encapsulation of metric/analyzer complexity

__Cons__:

- Loss of visual representation of dependencies in graph
- Need to synchronize between graph and symbol table
- More complex debugging (dependencies not visible in graph)
- Potential for orphaned symbols in table

## Approach 2: Embedded Symbol References

__Description__: Assertion nodes contain symbol references. SymbolTable is populated during graph traversal.

__Structure__:

- Assertion nodes store list of required symbols
- During propagation, populate SymbolTable from assertions
- SymbolTable handles metric computation and state

__Pros__:

- Natural flow from assertions to symbols
- No duplicate symbol management
- Easy to track which assertions use which symbols
- Maintains clear ownership model

__Cons__:

- Assertions become more complex (need to extract symbols)
- Circular dependency risk between graph and symbol table
- Harder to share symbols between assertions

## Approach 3: Hybrid with Symbol Registry

__Description__: Keep minimal symbol info in graph, detailed computation in SymbolTable.

__Structure__:

```python
class AssertionNode:
    actual: sp.Expr
    required_symbols: set[sp.Symbol]  # Just references

class SymbolTable:
    # Full symbol management with metrics, analyzers, etc.
```

__Pros__:

- Best of both worlds (graph shows dependencies, table handles computation)
- Can still traverse to find symbol usage
- Clean API separation
- Flexible symbol sharing

__Cons__:

- Some duplication of symbol tracking
- Need to maintain consistency between graph and table
- More complex initialization process

## Approach 4: Event-Driven Architecture

__Description__: Use events/callbacks to decouple graph from symbol computation.

__Structure__:

- Graph emits events during traversal
- SymbolTable subscribes to events and manages computation
- Loose coupling through event system

__Pros__:

- Maximum flexibility and extensibility
- Easy to add new functionality without changing graph
- Clean separation of concerns
- Testable in isolation

__Cons__:

- More complex to understand and debug
- Potential performance overhead
- Risk of event ordering issues
- Harder to trace execution flow

## Recommendation

I recommend __Approach 2 (Embedded Symbol References)__ with some elements from Approach 3:

1. __Simplicity__: Natural flow from assertions → symbols → metrics
2. __Maintainability__: Clear ownership and minimal duplication
3. __Performance__: Direct access patterns without indirection
4. __Debuggability__: Easy to trace symbol usage

### Implementation Strategy:

1. __Phase 1__: Create SymbolTable class with full metric/analyzer management
2. __Phase 2__: Refactor AssertionNode to extract and register symbols
3. __Phase 3__: Remove SymbolNode, MetricNode, AnalyzerNode classes
4. __Phase 4__: Update propagation logic to work with SymbolTable
5. __Phase 5__: Update visualization to show symbol dependencies

Would you like me to elaborate on any of these approaches or discuss specific implementation details for the recommended approach?
