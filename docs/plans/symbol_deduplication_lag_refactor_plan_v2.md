# Symbol Deduplication and Lag Refactor Implementation Plan v2

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

#### Task 2.1: Update ALL metric method signatures
**File**: `src/dqx/provider.py`

Update all metric creation methods (not just the ones in the original plan):
```python
def metric(
    self,
    metric: MetricSpec,
    lag: int = 0,
    dataset: str | None = None,
) -> sp.Symbol:
    self._register(
        sym := self._next_symbol(),
        name=metric.name,
        fn=partial(compute.simple_metric, self._db, metric, lag),
        metric_spec=metric,
        lag_offset=lag,
        dataset=dataset,
    )
    return sym

def num_rows(self, lag: int = 0, dataset: str | None = None) -> sp.Symbol:
    return self.metric(specs.NumRows(), lag, dataset)

def first(self, column: str, lag: int = 0, dataset: str | None = None) -> sp.Symbol:
    return self.metric(specs.First(column), lag, dataset)

def average(self, column: str, lag: int = 0, dataset: str | None = None) -> sp.Symbol:
    return self.metric(specs.Average(column), lag, dataset)

def minimum(self, column: str, lag: int = 0, dataset: str | None = None) -> sp.Symbol:
    return self.metric(specs.Minimum(column), lag, dataset)

def maximum(self, column: str, lag: int = 0, dataset: str | None = None) -> sp.Symbol:
    return self.metric(specs.Maximum(column), lag, dataset)

def sum(self, column: str, lag: int = 0, dataset: str | None = None) -> sp.Symbol:
    return self.metric(specs.Sum(column), lag, dataset)

def null_count(self, column: str, lag: int = 0, dataset: str | None = None) -> sp.Symbol:
    return self.metric(specs.NullCount(column), lag, dataset)

def variance(self, column: str, lag: int = 0, dataset: str | None = None) -> sp.Symbol:
    return self.metric(specs.Variance(column), lag, dataset)

def duplicate_count(self, columns: list[str], lag: int = 0, dataset: str | None = None) -> sp.Symbol:
    return self.metric(specs.DuplicateCount(columns), lag, dataset)

def count_values(
    self,
    column: str,
    values: int | str | bool | list[int] | list[str],
    lag: int = 0,
    dataset: str | None = None,
) -> sp.Symbol:
    return self.metric(specs.CountValues(column, values), lag, dataset)
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
    self._metrics.append(
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
    self._symbol_index[symbol] = self._metrics[-1]
```

#### Task 2.3: Remove obsolete methods from ExtendedMetricProvider
**File**: `src/dqx/provider.py`

Remove these methods that are no longer needed:
- `_create_lag_dependency()` - entire method becomes obsolete
- Complex parent-child tracking logic in `day_over_day()`, `stddev()`, and `week_over_week()`
- Any code managing `_children_map`

#### Task 2.4: Update tests for new API
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

    # Test simple metrics
    avg = mp.average("price", lag=1)
    assert mp.get_symbol(avg).lag_offset == 1

    min_val = mp.minimum("price", lag=2)
    assert mp.get_symbol(min_val).lag_offset == 2

    max_val = mp.maximum("price", lag=3)
    assert mp.get_symbol(max_val).lag_offset == 3

    count = mp.num_rows(lag=4)
    assert mp.get_symbol(count).lag_offset == 4

    sum_val = mp.sum("price", lag=5)
    assert mp.get_symbol(sum_val).lag_offset == 5

    # Test additional metrics
    first = mp.first("price", lag=6)
    assert mp.get_symbol(first).lag_offset == 6

    null = mp.null_count("price", lag=7)
    assert mp.get_symbol(null).lag_offset == 7

    var = mp.variance("price", lag=8)
    assert mp.get_symbol(var).lag_offset == 8
```

### Task Group 3: Update Extended Metrics
**Goal**: Make extended metrics populate required_metrics field

#### Task 3.1: Update stddev implementation
**File**: `src/dqx/provider.py`

```python
def stddev(self, metric: sp.Symbol, lag: int, n: int, dataset: str | None = None) -> sp.Symbol:
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
        dataset=dataset,
        required_metrics=required,
    )
    return sym

def _ensure_lagged_symbol(self, base: SymbolicMetric, lag: int) -> sp.Symbol:
    """Get or create a lagged version of a base metric."""
    # Check if it already exists
    for sm in self._metrics:
        if (sm.metric_spec == base.metric_spec and
            sm.lag_offset == lag and
            sm.dataset == base.dataset):
            return sm.symbol

    # Create new lagged metric using the appropriate method
    if isinstance(base.metric_spec, specs.Average):
        return self.average(base.metric_spec.column, lag=lag, dataset=base.dataset)
    elif isinstance(base.metric_spec, specs.Minimum):
        return self.minimum(base.metric_spec.column, lag=lag, dataset=base.dataset)
    elif isinstance(base.metric_spec, specs.Maximum):
        return self.maximum(base.metric_spec.column, lag=lag, dataset=base.dataset)
    elif isinstance(base.metric_spec, specs.Sum):
        return self.sum(base.metric_spec.column, lag=lag, dataset=base.dataset)
    elif isinstance(base.metric_spec, specs.NumRows):
        return self.num_rows(lag=lag, dataset=base.dataset)
    else:
        # Fallback for any other metric spec
        return self.metric(base.metric_spec, lag=lag, dataset=base.dataset)
```

#### Task 3.2: Update day_over_day and week_over_week
**File**: `src/dqx/provider.py`

```python
def day_over_day(self, metric: sp.Symbol, dataset: str | None = None) -> sp.Symbol:
    """Create day-over-day comparison metric."""
    base = self.get_symbol(metric)

    # Create lag dependency
    lag_sym = self._ensure_lagged_symbol(base, 1)

    self._register(
        sym := self._next_symbol(),
        name=f"day_over_day({base.name})",
        fn=partial(compute.day_over_day, self._db, base.metric_spec),
        metric_spec=base.metric_spec,
        lag_offset=0,
        dataset=dataset,
        required_metrics=[metric, lag_sym],
    )
    return sym

def week_over_week(self, metric: sp.Symbol, dataset: str | None = None) -> sp.Symbol:
    """Create week-over-week comparison metric."""
    base = self.get_symbol(metric)

    # Create lag dependency
    lag_sym = self._ensure_lagged_symbol(base, 7)

    self._register(
        sym := self._next_symbol(),
        name=f"week_over_week({base.name})",
        fn=partial(compute.week_over_week, self._db, base.metric_spec),
        metric_spec=base.metric_spec,
        lag_offset=0,
        dataset=dataset,
        required_metrics=[metric, lag_sym],
    )
    return sym
```

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
    # Should have base metric and lag=1 metric as dependencies
    assert len(dod_metric.required_metrics) == 2
    assert base in dod_metric.required_metrics

    # Find the lag metric
    lag_metric = None
    for req_sym in dod_metric.required_metrics:
        if req_sym != base:
            lag_metric = mp.get_symbol(req_sym)
            break

    assert lag_metric is not None
    assert lag_metric.metric_spec == mp.get_symbol(base).metric_spec
    assert lag_metric.lag_offset == 1

def test_week_over_week_creates_explicit_dependency():
    """Test week_over_week creates explicit lag dependency."""
    mp = MetricProvider(db)
    base = mp.average("price")
    wow = mp.week_over_week(base)

    wow_metric = mp.get_symbol(wow)
    # Should have base metric and lag=7 metric as dependencies
    assert len(wow_metric.required_metrics) == 2
    assert base in wow_metric.required_metrics

    # Find the lag metric
    lag_metric = None
    for req_sym in wow_metric.required_metrics:
        if req_sym != base:
            lag_metric = mp.get_symbol(req_sym)
            break

    assert lag_metric is not None
    assert lag_metric.metric_spec == mp.get_symbol(base).metric_spec
    assert lag_metric.lag_offset == 7
```

### Task Group 4: Update Compute Functions
**Goal**: Update compute functions to use lag_offset instead of ResultKeyProvider

#### Task 4.1: Update ALL compute function signatures
**File**: `src/dqx/compute.py`

Update all functions that currently use ResultKeyProvider:
```python
def simple_metric(db: MetricDB, metric: MetricSpec, lag_offset: int, key: ResultKey) -> Result[float, str]:
    """Compute simple metric with lag applied."""
    effective_key = key.lag(lag_offset)
    # ... rest of implementation

def day_over_day(db: MetricDB, metric: MetricSpec, key: ResultKey) -> Result[float, str]:
    """Compute day-over-day comparison."""
    # Note: DoD doesn't take lag_offset because it internally uses lag=1
    # ... implementation

def week_over_week(db: MetricDB, metric: MetricSpec, key: ResultKey) -> Result[float, str]:
    """Compute week-over-week comparison."""
    # Note: WoW doesn't take lag_offset because it internally uses lag=7
    # ... implementation

def stddev(db: MetricDB, metric: MetricSpec, lag: int, n: int, key: ResultKey) -> Result[float, str]:
    """Compute standard deviation over time window."""
    # Note: stddev has its own lag parameter for the window start
    # ... implementation
```

#### Task 4.2: Test compute functions
**File**: `tests/test_compute.py`

```python
def test_compute_with_lag_offset():
    """Test compute functions use lag_offset correctly."""
    # Test simple_metric with lag
    result = simple_metric(db, Average("price"), lag_offset=1, key=ResultKey("2024-01-15"))
    # Should compute for 2024-01-14
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
    """Build symbol substitution map for deduplication.

    This method identifies duplicate symbols that represent the same metric
    computed for the same effective date and dataset. It returns a mapping
    from duplicate symbols to their canonical representatives.

    Args:
        context_key: The analysis date context. Used to calculate effective
                    dates for lagged metrics.

    Returns:
        Dict mapping duplicate symbols to canonical symbols. For example:
        {
            sp.Symbol('x_3'): sp.Symbol('x_1'),  # x_3 is duplicate of x_1
            sp.Symbol('x_5'): sp.Symbol('x_2'),  # x_5 is duplicate of x_2
        }

        The canonical symbol is always the one with the lowest index number.
        Empty dict if no duplicates found.

    Example:
        If we have:
        - x_1: average(price) for 2024-01-15
        - x_2: average(price) with lag=1 for 2024-01-16 (effective: 2024-01-15)
        - x_3: average(price) for 2024-01-15 (duplicate of x_1)

        This returns: {x_3: x_1, x_2: x_1}
    """
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
    """Visitor to replace duplicate symbols in graph expressions.

    This visitor traverses the computation graph and replaces duplicate
    symbols in assertion expressions according to the provided substitution map.
    Only AssertionNode expressions are modified.
    """

    def __init__(self, substitutions: dict[sp.Symbol, sp.Symbol]) -> None:
        """Initialize with symbol substitution map.

        Args:
            substitutions: Map from duplicate symbols to canonical symbols
        """
        self.substitutions = substitutions

    def visit(self, node: BaseNode) -> None:
        """Visit a node and apply symbol substitutions.

        Args:
            node: The graph node to process
        """
        # Only process AssertionNodes
        if isinstance(node, AssertionNode):
            # Replace symbols in the assertion's expression
            node.actual = node.actual.subs(self.substitutions)
```

#### Task 5.3: Update prune_duplicate_symbols
**File**: `src/dqx/provider.py`

```python
def prune_duplicate_symbols(self, substitutions: dict[sp.Symbol, sp.Symbol]) -> None:
    """Remove duplicate symbols from the provider.

    This method removes duplicate symbols from the internal tracking structures.
    It does NOT update required_metrics - that should be done separately by
    the analyzer to maintain separation of concerns.

    Args:
        substitutions: Map from duplicate symbols to canonical symbols
    """
    if not substitutions:
        return

    to_remove = set(substitutions.keys())

    # Remove duplicate symbols
    self._metrics = [
        sm for sm in self._metrics
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

def test_prune_duplicate_symbols_does_not_update_required_metrics():
    """Test pruning removes duplicates but doesn't update required_metrics."""
    mp = MetricProvider(db)
    x0 = mp.average("price")
    x1 = mp.average("price")  # duplicate

    # Create metric that references x1
    mp._metrics[0].required_metrics = [x1]

    substitutions = {x1: x0}
    mp.prune_duplicate_symbols(substitutions)

    # x1 should be removed
    assert x1 not in [sm.symbol for sm in mp.symbolic_metrics]
    # required_metrics should NOT be updated (that's analyzer's job)
    assert mp._metrics[0].required_metrics == [x1]
```

### Task Group 6: Integrate Deduplication into API
**Goal**: Apply deduplication in VerificationSuite._analyze

#### Task 6.1: Add _build_symbolic_metrics to VerificationSuite
**File**: `src/dqx/api.py`

Add these methods to the VerificationSuite class:

```python
def _build_symbolic_metrics(
    self, graph: Graph, provider: MetricProvider, key: ResultKey
) -> tuple[MetricProvider, dict[sp.Symbol, sp.Symbol]]:
    """Build symbolic metrics with deduplication.

    Args:
        graph: The computation graph to apply deduplication to
        provider: The metric provider containing all symbolic metrics
        key: The analysis date context for calculating effective dates

    Returns:
        Tuple of (provider, substitutions) where substitutions maps
        duplicate symbols to canonical symbols
    """
    # Build deduplication map
    substitutions = provider.build_deduplication_map(key)

    # Apply deduplication to graph if there are duplicates
    if substitutions:
        from dqx.graph.visitors import SymbolDeduplicationVisitor

        dedup_visitor = SymbolDeduplicationVisitor(substitutions)
        graph.accept(dedup_visitor)

        # Update required_metrics in remaining symbols
        self._update_required_metrics(provider, substitutions)

        # Prune duplicate symbols from provider
        provider.prune_duplicate_symbols(substitutions)

    return provider, substitutions

def _update_required_metrics(
    self, provider: MetricProvider, substitutions: dict[sp.Symbol, sp.Symbol]
) -> None:
    """Update required_metrics in all symbolic metrics after deduplication.

    Args:
        provider: The metric provider containing symbolic metrics
        substitutions: Map of duplicate symbols to canonical symbols
    """
    for sym_metric in provider.symbolic_metrics:
        if sym_metric.required_metrics:
            # Replace any duplicates in required_metrics
            sym_metric.required_metrics = [
                substitutions.get(req, req)
                for req in sym_metric.required_metrics
            ]
```

#### Task 6.2: Update _analyze method to use deduplication
**File**: `src/dqx/api.py`

Modify the existing `_analyze` method:

```python
def _analyze(self, datasources: list[SqlDataSource], key: ResultKey) -> None:
    # Build symbolic metrics with deduplication FIRST
    provider, substitutions = self._build_symbolic_metrics(
        self._context._graph, self._context.provider, key
    )

    # Log deduplication results if any
    if substitutions:
        logger.info(
            f"Deduplication removed {len(substitutions)} duplicate symbols"
        )

    # ... rest of existing _analyze implementation ...
    # (The existing code for grouping by dataset and analyzing)
```

#### Task 6.3: Test API integration
**File**: `tests/test_api_symbol_deduplication.py`

```python
def test_verification_suite_deduplicates_symbols():
    """Test VerificationSuite applies deduplication during analysis."""
    # Create suite with duplicate symbols
    @check(name="Test check")
    def test_check(mp: MetricProvider, ctx: Context) -> None:
        x0 = mp.average("price")
        x1 = mp.average("price", lag=1)  # Will be duplicate on lag date

        ctx.assert_that(x0 > 0).where(name="x0 positive").is_positive()
        ctx.assert_that(x1 > 0).where(name="x1 positive").is_positive()

    suite = VerificationSuite([test_check], db, "Test")
    datasources = [DuckRelationDataSource.from_arrow(data, "test_ds")]
    suite.run(datasources, ResultKey("2024-01-16"))

    # Verify deduplication occurred
    provider = suite.provider
    symbols = [sm.symbol for sm in provider.symbolic_metrics]

    # Should have deduplicated x1 since it's same as x0 on 2024-01-15
    assert len(symbols) == 1

def test_deduplication_updates_required_metrics():
    """Test that deduplication updates required_metrics references."""
    @check(name="Extended metric check")
    def extended_check(mp: MetricProvider, ctx: Context) -> None:
        base = mp.average("price")
        dod = mp.ext.day_over_day(base)

        ctx.assert_that(dod > 0).where(name="DoD positive").is_positive()

    suite = VerificationSuite([extended_check], db, "Test")
    datasources = [DuckRelationDataSource.from_arrow(data, "test_ds")]
    suite.run(datasources, ResultKey("2024-01-16"))

    # Verify required_metrics were updated after deduplication
    provider = suite.provider
    dod_metric = next(sm for sm in provider.symbolic_metrics
                      if "day_over_day" in sm.name)

    # All required_metrics should point to valid symbols
    for req_sym in dod_metric.required_metrics:
        assert any(sm.symbol == req_sym for sm in provider.symbolic_metrics)
```

### Task Group 7: Update DatasetValidator
**Goal**: Add validation for required_metrics consistency

#### Task 7.1: Add _check_required_metrics_consistency to DatasetValidator
**File**: `src/dqx/validator.py`

Add this method to the DatasetValidator class:

```python
def _check_required_metrics_consistency(
    self, metric: SymbolicMetric
) -> list[str]:
    """Check that complex metric and its required metrics have same dataset.

    Args:
        metric: The symbolic metric to check

    Returns:
        List of error messages (empty if no errors)
    """
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

#### Task 7.2: Update validate method to call new check
**File**: `src/dqx/validator.py`

Modify the existing validate method:

```python
def validate(self) -> list[str]:
    """Validate dataset assignments across all metrics.

    Returns:
        List of validation error messages
    """
    errors = []

    # ... existing validation code ...

    # Add new validation for required_metrics
    for metric in self.provider.symbolic_metrics:
        if metric.required_metrics:  # Only check metrics with dependencies
            consistency_errors = self._check_required_metrics_consistency(metric)
            errors.extend(consistency_errors)

    return errors
```

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

### Task Group 8: Automated Test Updates
**Goal**: Automatically remove ResultKeyProvider usage from all tests

#### Task 8.1: Create automation script
**File**: `scripts/update_resultkeyprovider.py`

```python
#!/usr/bin/env python3
"""Script to automatically update tests from ResultKeyProvider to lag parameter."""

import re
from pathlib import Path

def update_test_file(filepath: Path) -> bool:
    """Update a single test file."""
    content = filepath.read_text()
    original = content

    # Pattern 1: key=ResultKeyProvider() with no lag
    content = re.sub(
        r'key=ResultKeyProvider\(\)',
        'lag=0',
        content
    )

    # Pattern 2: key=ResultKeyProvider().lag(n)
    content = re.sub(
        r'key=ResultKeyProvider\(\)\.lag\((\d+)\)',
        r'lag=\1',
        content
    )

    # Pattern 3: Update method calls that use key parameter
    # mp.average("col", key=...) -> mp.average("col", lag=...)
    content = re.sub(
        r'(mp\.\w+\([^)]*),\s*key=([^,)]+)',
        r'\1, lag=\2',
        content
    )

    # Remove ResultKeyProvider imports
    content = re.sub(
        r'from dqx\.common import ([^)\n]*?)ResultKeyProvider(?:,\s*)?([^)\n]*?)\n',
        lambda m: f"from dqx.common import {m.group(1).strip(', ')}{', ' if m.group(1).strip(', ') and m.group(2).strip(', ') else ''}{m.group(2).strip(', ')}\n"
                  if m.group(1).strip(', ') or m.group(2).strip(', ')
                  else '',
        content
    )

    # Clean up empty imports
    content = re.sub(
        r'from dqx\.common import\s*\n',
        '',
        content
    )

    if content != original:
        filepath.write_text(content)
        return True
    return False

def main():
    """Update all test files."""
    test_dir = Path('tests')
    updated_files = []

    for test_file in test_dir.rglob('test_*.py'):
        if update_test_file(test_file):
            updated_files.append(test_file)

    print(f"Updated {len(updated_files)} test files:")
    for f in updated_files:
        print(f"  - {f}")

if __name__ == "__main__":
    main()
```

#### Task 8.2: Run automation script
```bash
# Make script executable
chmod +x scripts/update_resultkeyprovider.py

# Run the script
uv run python scripts/update_resultkeyprovider.py
```

#### Task 8.3: Manual verification of critical files
After running the script, manually verify these critical test files:
- `tests/test_provider.py`
- `tests/test_analyzer.py`
- `tests/test_extended_metric.py`
- `tests/e2e/test_api_e2e.py`

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

## Migration Guide

For users updating their code:

### Before
```python
mp.average("price", key=ResultKeyProvider().lag(1))
```

### After
```python
mp.average("price", lag=1)
```

### Extended Metrics
Extended metrics no longer need key parameter:
```python
# Before
mp.ext.day_over_day(base_metric, key=ResultKeyProvider())

# After
mp.ext.day_over_day(base_metric)
```
