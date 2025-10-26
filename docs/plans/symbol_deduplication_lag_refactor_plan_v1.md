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

Add test to verify:
- SymbolicMetric has `lag_offset` field
- SymbolicMetric has `required_metrics` field
- No `key_provider` or `parent_symbol` fields exist

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

Test that:
- `mp.average("price", lag=1)` creates metric with lag_offset=1
- Old `key=ResultKeyProvider()` usage is no longer valid
- All metric methods accept lag parameter

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

Verify that:
- stddev populates required_metrics correctly
- day_over_day creates explicit lag dependency
- No hidden symbol creation occurs

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

Ensure compute functions work correctly with lag_offset parameter.

### Task Group 5: Implement Symbol Deduplication
**Goal**: Create visitor to identify and merge duplicate symbols

#### Task 5.1: Create SymbolDeduplicationVisitor
**File**: `src/dqx/graph/visitors/symbol_deduplication.py`

```python
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING

import sympy as sp

from dqx.common import ResultKey

if TYPE_CHECKING:
    from dqx.provider import MetricProvider


class SymbolDeduplicationVisitor:
    """Visitor to identify and merge duplicate symbols."""

    def __init__(self, provider: MetricProvider) -> None:
        self.provider = provider

    def find_duplicates(self, context_key: ResultKey) -> dict[sp.Symbol, sp.Symbol]:
        """Find duplicate symbols and return substitution map."""
        groups: dict[tuple[str, str, str | None], list[sp.Symbol]] = {}

        # Group symbols by identity
        for sym_metric in self.provider.symbolic_metrics:
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

#### Task 5.2: Add prune_duplicate_symbols to MetricProvider
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

#### Task 5.3: Test deduplication
**File**: `tests/test_symbol_deduplication.py`

Create comprehensive tests for deduplication logic.

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

    # Apply deduplication
    from dqx.graph.visitors import SymbolDeduplicationVisitor

    dedup_visitor = SymbolDeduplicationVisitor(metric_provider)
    substitutions = dedup_visitor.find_duplicates(key)

    # Update expressions with substitutions
    if substitutions:
        for node in graph.nodes:
            if hasattr(node, 'assertions'):
                for assertion in node.assertions:
                    assertion.expression = assertion.expression.subs(substitutions)

        # Prune duplicate symbols
        metric_provider.prune_duplicate_symbols(substitutions)

    return metric_provider, substitutions
```

#### Task 6.2: Test analyzer integration
**File**: `tests/test_analyzer.py`

Test that analyzer correctly deduplicates symbols in analysis reports.

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

Test dataset consistency validation for complex metrics.

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
