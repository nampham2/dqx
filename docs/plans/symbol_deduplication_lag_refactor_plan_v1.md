# Symbol Deduplication and Lag Refactor Implementation Plan v1

## Overview

This plan addresses two critical issues in the DQX codebase:
1. **Symbol Duplication**: Multiple symbols representing the same metric/date/dataset combination
2. **Complex Lag Dependencies**: Hidden lag symbol creation causing confusion and analysis report issues

The solution introduces a cleaner architecture by:
- Replacing `ResultKeyProvider` with a simple `lag_offset` field
- Adding `required_metrics` to track complex metric dependencies explicitly
- Implementing symbol deduplication to ensure unique metric representation

## Background

### Current Issues
- Duplicate symbols appear in analysis reports (e.g., multiple `average(price)` entries)
- Lag dependencies are created implicitly, making them hard to track
- `ResultKeyProvider` adds unnecessary complexity to the API
- `parent_symbol` field creates confusing parent-child relationships

### Proposed Solution
- Remove `ResultKeyProvider` completely
- Add `lag_offset: int` field to `SymbolicMetric`
- Replace `parent_symbol` with `required_metrics: list[sp.Symbol]`
- Implement `SymbolDeduplicationVisitor` to identify and merge duplicates
- Update all metric APIs to use `lag: int` parameter instead of `key: ResultKeyProvider`

## Implementation Tasks

### Task Group 1: Core Data Model Updates
**Goal**: Update SymbolicMetric dataclass and remove ResultKeyProvider

#### Task 1.1: Update SymbolicMetric dataclass
**File**: `src/dqx/provider.py`

Current:
```python
@dataclass
class SymbolicMetric:
    name: str
    symbol: sp.Symbol
    fn: RetrievalFn
    key_provider: ResultKeyProvider
    metric_spec: MetricSpec
    dataset: str | None = None
    parent_symbol: sp.Symbol | None = None
```

Change to:
```python
@dataclass
class SymbolicMetric:
    name: str
    symbol: sp.Symbol
    fn: RetrievalFn
    metric_spec: MetricSpec
    lag_offset: int = 0
    dataset: str | None = None
    required_metrics: list[sp.Symbol] = field(default_factory=list)
```

#### Task 1.2: Remove ResultKeyProvider class
**File**: `src/dqx/common.py`

Delete the entire `ResultKeyProvider` class and its imports throughout the codebase.

#### Task 1.3: Write tests for updated data model
**File**: `tests/test_provider.py`

```python
def test_symbolic_metric_has_new_fields():
    """Test that SymbolicMetric has lag_offset and required_metrics."""
    sm = SymbolicMetric(
        name="test",
        symbol=sp.Symbol("x_0"),
        fn=lambda key: Result.success(1.0),
        metric_spec=Average("col"),
        lag_offset=1,
        required_metrics=[sp.Symbol("x_1")]
    )
    assert sm.lag_offset == 1
    assert sm.required_metrics == [sp.Symbol("x_1")]
    assert not hasattr(sm, 'key_provider')
    assert not hasattr(sm, 'parent_symbol')
```

### Task Group 2: Update MetricProvider API
**Goal**: Change all metric creation methods to use lag parameter

#### Task 2.1: Update metric method signatures
**File**: `src/dqx/provider.py`

Update all metric creation methods:
```python
def average(self, column: str, lag: int = 0, dataset: str | None = None) -> sp.Symbol:
    """Create average metric with optional lag."""
    self._register(
        sym := self._next_symbol(),
        name=f"average({column})",
        fn=partial(self._compute_average, column, lag),
        metric_spec=Average(column),
        lag_offset=lag,
        dataset=dataset,
    )
    return sym

# Similar updates for: maximum, minimum, count, sum
```

#### Task 2.2: Update _register method
**File**: `src/dqx/provider.py`

Update signature and implementation:
```python
def _register(
    self,
    symbol: sp.Symbol,
    name: str,
    fn: RetrievalFn,
    metric_spec: MetricSpec,
    lag_offset: int = 0,
    dataset: str | None = None,
    required_metrics: list[sp.Symbol] | None = None,
) -> None:
    """Register a symbolic metric."""
    self._symbols.append(
        SymbolicMetric(
            name=name,
            symbol=symbol,
            fn=fn,
            metric_spec=metric_spec,
            lag_offset=lag_offset,
            dataset=dataset,
            required_metrics=required_metrics or [],
        )
    )
    self._symbol_index[symbol] = self._symbols[-1]
```

#### Task 2.3: Update tests for new API
**File**: `tests/test_provider.py`

```python
def test_metric_creation_with_lag():
    """Test metric creation with lag parameter."""
    mp = MetricProvider(db)
    sym = mp.average("price", lag=1)
    metric = mp.get_symbol(sym)
    assert metric.lag_offset == 1

def test_old_api_no_longer_works():
    """Test that old ResultKeyProvider API raises error."""
    mp = MetricProvider(db)
    with pytest.raises(TypeError):
        mp.average("price", key=ResultKeyProvider().lag(1))

def test_all_metric_methods_accept_lag():
    """Test all metric methods accept lag parameter."""
    mp = MetricProvider(db)

    # Test each metric method
    avg = mp.average("price", lag=1)
    assert mp.get_symbol(avg).lag_offset == 1

    min_val = mp.minimum("price", lag=2)
    assert mp.get_symbol(min_val).lag_offset == 2

    max_val = mp.maximum("price", lag=3)
    assert mp.get_symbol(max_val).lag_offset == 3

    count = mp.count(lag=4)
    assert mp.get_symbol(count).lag_offset == 4

    sum_val = mp.sum("price", lag=5)
    assert mp.get_symbol(sum_val).lag_offset == 5
```

### Task Group 3: Update Extended Metrics
**Goal**: Make extended metrics populate required_metrics field

#### Task 3.1: Update stddev implementation
**File**: `src/dqx/provider.py`

```python
def stddev(self, metric: sp.Symbol, lag: int, n: int) -> sp.Symbol:
    """Create standard deviation metric over time window."""
    base = self.get_symbol(metric)

    # Ensure required lag metrics exist
    required = []
    for i in range(lag, lag + n):
        lag_sym = self._ensure_lagged_symbol(base, i)
        required.append(lag_sym)

    self._register(
        sym := self._next_symbol(),
        name=f"stddev({base.name}, lag={lag}, n={n})",
        fn=partial(compute.stddev, self._db, base.metric_spec, lag, n),
        metric_spec=StdDev(base.metric_spec),
        lag_offset=0,
        required_metrics=required,
    )
    return sym

def _ensure_lagged_symbol(self, base: SymbolicMetric, lag: int) -> sp.Symbol:
    """Get or create a lagged version of a base metric."""
    # Check if it already exists
    for sm in self._symbols:
        if (sm.metric_spec == base.metric_spec and
            sm.lag_offset == lag and
            sm.dataset == base.dataset):
            return sm.symbol

    # Create new lagged metric
    return self._create_metric_from_spec(
        base.metric_spec,
        lag=lag,
        dataset=base.dataset
    )
```

#### Task 3.2: Update day_over_day and week_over_week
**File**: `src/dqx/provider.py`

Remove `_create_lag_dependency` method and update extended metrics to use `required_metrics`.

#### Task 3.3: Test extended metrics
**File**: `tests/test_extended_metric_dependencies.py`

```python
def test_stddev_populates_required_metrics():
    """Test that stddev creates required_metrics."""
    mp = MetricProvider(db)
    base = mp.average("price")
    std = mp.stddev(base, lag=1, n=3)

    std_metric = mp.get_symbol(std)
    assert len(std_metric.required_metrics) == 3
    # Should have symbols for lag 1, 2, and 3

    # Verify each required metric has correct lag
    for i, req_sym in enumerate(std_metric.required_metrics):
        req_metric = mp.get_symbol(req_sym)
        assert req_metric.lag_offset == i + 1

def test_day_over_day_creates_explicit_dependency():
    """Test day_over_day creates explicit lag dependency."""
    mp = MetricProvider(db)
    base = mp.average("price")
    dod = mp.day_over_day(base)

    dod_metric = mp.get_symbol(dod)
    # Should have base metric with lag=1 as dependency
    assert len(dod_metric.required_metrics) == 1

    lag_metric = mp.get_symbol(dod_metric.required_metrics[0])
    assert lag_metric.metric_spec == mp.get_symbol(base).metric_spec
    assert lag_metric.lag_offset == 1

def test_no_hidden_symbol_creation():
    """Test that extended metrics don't create hidden symbols."""
    mp = MetricProvider(db)
    initial_count = len(mp.symbolic_metrics)

    base = mp.average("price")
    dod = mp.day_over_day(base)

    # Should have base + dod + one lag symbol
    assert len(mp.symbolic_metrics) == initial_count + 3

    # Verify all symbols are trackable
    all_symbols = [sm.symbol for sm in mp.symbolic_metrics]
    assert base in all_symbols
    assert dod in all_symbols
```

### Task Group 4: Update Compute Functions
**Goal**: Update compute functions to use lag_offset instead of ResultKeyProvider

#### Task 4.1: Update compute function signatures
**File**: `src/dqx/compute.py`

Change functions that use ResultKeyProvider to accept lag_offset:
```python
def compute_average(db: Database, column: str, lag_offset: int, key: ResultKey) -> Result[float, str]:
    """Compute average with lag applied."""
    effective_key = key.lag(lag_offset)
    # ... rest of implementation
```

#### Task 4.2: Update function implementations
Remove all ResultKeyProvider usage and use `key.lag(lag_offset)` directly.

#### Task 4.3: Test compute functions
**File**: `tests/test_compute.py`

```python
def test_compute_with_lag_offset():
    """Test compute functions use lag_offset correctly."""
    # Test average with lag
    result = compute_average(db, "price", lag_offset=1, key=ResultKey("2024-01-15"))
    # Should compute for 2024-01-14
    assert result.is_success()

    # Test other compute functions
    result = compute_minimum(db, "price", lag_offset=2, key=ResultKey("2024-01-15"))
    # Should compute for 2024-01-13
    assert result.is_success()

def test_lag_offset_date_calculation():
    """Test that lag_offset correctly calculates dates."""
    key = ResultKey("2024-01-15")

    # Test various lag offsets
    effective_key_0 = key.lag(0)
    assert effective_key_0.yyyy_mm_dd == date(2024, 1, 15)

    effective_key_1 = key.lag(1)
    assert effective_key_1.yyyy_mm_dd == date(2024, 1, 14)

    effective_key_7 = key.lag(7)
    assert effective_key_7.yyyy_mm_dd == date(2024, 1, 8)
```

### Task Group 5: Implement Symbol Deduplication
**Goal**: Create visitor to identify and merge duplicate symbols

#### Task 5.1: Add build_deduplication_map to MetricProvider
**File**: `src/dqx/provider.py`

```python
def build_deduplication_map(self, context_key: ResultKey) -> dict[sp.Symbol, sp.Symbol]:
    """Build symbol substitution map for deduplication."""
    groups: dict[tuple[str, str, str | None], list[sp.Symbol]] = {}

    # Group symbols by identity
    for sym_metric in self.symbolic_metrics:
        # Calculate effective date for this symbol
        effective_date = context_key.yyyy_mm_dd - timedelta(days=sym_metric.lag_offset)

        identity = (
            sym_metric.metric_spec.name,
            effective_date.isoformat(),
            sym_metric.dataset
        )

        if identity not in groups:
            groups[identity] = []
        groups[identity].append(sym_metric.symbol)

    # Build substitution map
    substitutions = {}
    for duplicates in groups.values():
        if len(duplicates) > 1:
            # Keep the lowest numbered symbol as canonical
            duplicates_sorted = sorted(
                duplicates,
                key=lambda s: int(s.name.split('_')[1])
            )
            canonical = duplicates_sorted[0]

            for dup in duplicates_sorted[1:]:
                substitutions[dup] = canonical

    return substitutions
```

#### Task 5.2: Create SymbolDeduplicationVisitor
**File**: `src/dqx/graph/visitors/symbol_deduplication.py`

```python
import sympy as sp

from dqx.graph.base import BaseNode
from dqx.graph.nodes import AssertionNode


class SymbolDeduplicationVisitor:
    """Visitor to replace duplicate symbols in graph expressions."""

    def __init__(self, substitutions: dict[sp.Symbol, sp.Symbol]) -> None:
        self.substitutions = substitutions

    def visit(self, node: BaseNode) -> None:
        """Visit a node and apply symbol substitutions."""
        # Only process AssertionNodes
        if isinstance(node, AssertionNode):
            # Replace symbols in the assertion's expression
            node.actual = node.actual.subs(self.substitutions)
```

#### Task 5.3: Update prune_duplicate_symbols to handle required_metrics
**File**: `src/dqx/provider.py`

```python
def prune_duplicate_symbols(self, substitutions: dict[sp.Symbol, sp.Symbol]) -> None:
    """Remove duplicate symbols and update all references."""
    if not substitutions:
        return

    to_remove = set(substitutions.keys())

    # Update required_metrics in remaining symbols
    for sym_metric in self._symbols:
        if sym_metric.required_metrics:
            # Replace any duplicates in required_metrics
            sym_metric.required_metrics = [
                substitutions.get(req, req)
                for req in sym_metric.required_metrics
            ]

    # Remove duplicate symbols
    self._symbols = [
        sm for sm in self._symbols
        if sm.symbol not in to_remove
    ]

    # Remove from index
    for symbol in to_remove:
        del self._symbol_index[symbol]
```

#### Task 5.4: Test deduplication
**File**: `tests/test_symbol_deduplication.py`

```python
def test_deduplication_visitor():
    """Test SymbolDeduplicationVisitor replaces symbols."""
    # Create graph with assertions
    root = RootNode("test")
    check = root.add_check("check1")
    assertion = check.add_assertion(
        sp.Symbol("x_0") + sp.Symbol("x_1"),
        "test",
        lambda x: x > 0
    )

    # Apply deduplication
    substitutions = {sp.Symbol("x_1"): sp.Symbol("x_0")}
    visitor = SymbolDeduplicationVisitor(substitutions)
    root.accept(visitor)

    # Verify substitution
    assert assertion.actual == sp.Symbol("x_0") + sp.Symbol("x_0")

def test_build_deduplication_map():
    """Test MetricProvider builds correct dedup map."""
    mp = MetricProvider(db)
    mp.average("price")  # x_0
    mp.average("price", lag=1)  # x_1 (same as x_0 on different date)

    key = ResultKey("2024-01-15")
    dedup_map = mp.build_deduplication_map(key)

    # Should identify duplicates
    assert len(dedup_map) > 0

def test_prune_duplicate_symbols():
    """Test pruning removes duplicates and updates references."""
    mp = MetricProvider(db)
    x0 = mp.average("price")
    x1 = mp.average("price")  # duplicate

    # Create metric that references x1
    mp._symbols[0].required_metrics = [x1]

    substitutions = {x1: x0}
    mp.prune_duplicate_symbols(substitutions)

    # x1 should be removed
    assert x1 not in [sm.symbol for sm in mp.symbolic_metrics]
    # required_metrics should be updated
    assert mp._symbols[0].required_metrics == [x0]
```

### Task Group 6: Integrate Deduplication into Analyzer
**Goal**: Apply deduplication before analysis

#### Task 6.1: Update Analyzer._build_symbolic_metrics
**File**: `src/dqx/analyzer.py`

```python
def _build_symbolic_metrics(
    self, graph: ComputationGraph, key: ResultKey
) -> tuple[MetricProvider, dict[sp.Symbol, sp.Symbol]]:
    """Build symbolic metrics with deduplication."""
    # ... existing code ...

    # Build deduplication map
    substitutions = metric_provider.build_deduplication_map(key)

    # Apply deduplication to graph
    if substitutions:
        from dqx.graph.visitors import SymbolDeduplicationVisitor

        dedup_visitor = SymbolDeduplicationVisitor(substitutions)
        graph.accept(dedup_visitor)

        # Prune duplicate symbols from provider
        metric_provider.prune_duplicate_symbols(substitutions)

    return metric_provider, substitutions
```

#### Task 6.2: Test analyzer integration
**File**: `tests/test_analyzer.py`

```python
def test_analyzer_deduplicates_symbols():
    """Test analyzer applies deduplication."""
    # Create suite with duplicate symbols
    mp = MetricProvider(db)
    x0 = mp.average("price")
    x1 = mp.average("price", lag=1)

    # Create assertions using both symbols
    suite = Suite("test").check("test_check").assertion(x0 > 0).assertion(x1 > 0)

    # Run analyzer
    analyzer = Analyzer(db)
    result = analyzer.analyze(suite, ResultKey("2024-01-15"))

    # Verify no duplicates in analysis report
    # All average(price) should be consolidated
```

### Task Group 7: Update DatasetValidator
**Goal**: Add validation for required_metrics consistency

#### Task 7.1: Add required_metrics validation
**File**: `src/dqx/validator.py`

```python
def _check_required_metrics_consistency(self, metric: SymbolicMetric) -> list[str]:
    """Check that complex metric and its required metrics have same dataset."""
    errors = []

    # Skip if metric itself has no dataset yet
    if metric.dataset is None:
        return errors

    for req_sym in metric.required_metrics:
        req_metric = self.provider.get_symbol(req_sym)

        # Only check when BOTH have datasets assigned
        if (req_metric.dataset is not None and
            req_metric.dataset != metric.dataset):
            errors.append(
                f"Dataset mismatch: {metric.symbol} has dataset "
                f"'{metric.dataset}' but required metric {req_sym} has "
                f"dataset '{req_metric.dataset}'"
            )

    return errors
```

#### Task 7.2: Update validate method
Call the new consistency check for all metrics with required_metrics.

#### Task 7.3: Test validation
**File**: `tests/test_validator.py`

```python
def test_required_metrics_consistency_validation():
    """Test DatasetValidator checks required_metrics consistency."""
    mp = MetricProvider(db)
    base = mp.average("price")
    std = mp.stddev(base, lag=1, n=3)

    # Assign datasets - this should trigger validation error
    mp.get_symbol(base).dataset = "ds1"
    mp.get_symbol(std).dataset = "ds2"  # Different dataset

    validator = DatasetValidator(mp)
    errors = validator.validate()

    # Should report dataset mismatch
    assert any("Dataset mismatch" in error for error in errors)
    assert any("required metric" in error for error in errors)

def test_required_metrics_validation_same_dataset():
    """Test validation passes when datasets match."""
    mp = MetricProvider(db)
    base = mp.average("price")
    std = mp.stddev(base, lag=1, n=3)

    # Assign same dataset to all
    mp.get_symbol(base).dataset = "ds1"
    mp.get_symbol(std).dataset = "ds1"

    # Also need to assign to required metrics
    for req_sym in mp.get_symbol(std).required_metrics:
        mp.get_symbol(req_sym).dataset = "ds1"

    validator = DatasetValidator(mp)
    errors = validator.validate()

    # Should not have dataset mismatch errors
    assert not any("Dataset mismatch" in error for error in errors)

def test_required_metrics_validation_none_dataset():
    """Test validation skips when dataset is None."""
    mp = MetricProvider(db)
    base = mp.average("price")
    std = mp.stddev(base, lag=1, n=3)

    # Don't assign any datasets (all None)
    validator = DatasetValidator(mp)
    errors = validator.validate()

    # Should not report errors for None datasets
    assert not any("Dataset mismatch" in error for error in errors)
```

### Task Group 8: Update All Tests
**Goal**: Remove ResultKeyProvider usage from all tests

#### Task 8.1: Update test files
Search and replace all ResultKeyProvider usage with lag parameter:
- `tests/test_provider*.py`
- `tests/test_analyzer*.py`
- `tests/test_extended_metric*.py`
- Any other test files using ResultKeyProvider

#### Task 8.2: Run all tests
Ensure all tests pass after refactoring.

### Task Group 9: Final Verification
**Goal**: Ensure code quality and completeness

#### Task 9.1: Run pre-commit checks
```bash
bin/run-hooks.sh --all
```

#### Task 9.2: Run full test suite
```bash
uv run pytest tests/ -v
```

#### Task 9.3: Check code coverage
```bash
uv run pytest tests/ --cov=dqx --cov-report=html
```

## Testing Strategy

1. **Unit Tests**: Test each component in isolation
2. **Integration Tests**: Test complete workflows with deduplication
3. **E2E Tests**: Verify analysis reports have no duplicates
4. **Coverage**: Maintain or improve current coverage levels

## Risk Mitigation

1. **Breaking Changes**: The removal of ResultKeyProvider is a breaking change. Users will need to update their code from `key=ResultKeyProvider().lag(n)` to `lag=n`.
2. **Performance**: Deduplication adds a traversal step, but impact should be minimal.
3. **Compatibility**: Ensure all existing tests pass with new implementation.

## Success Criteria

1. No duplicate symbols in analysis reports
2. All tests passing
3. Cleaner, more intuitive API
4. Explicit tracking of metric dependencies
5. No hidden symbol creation
