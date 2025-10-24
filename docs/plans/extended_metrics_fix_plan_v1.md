# Extended Metrics Fix Implementation Plan v1

## Problem Summary

Extended metrics (day_over_day, week_over_week, stddev) have three main issues:

1. **Incorrect Name Display**: When collecting symbols, extended metrics show their base metric name instead of the full extended metric name (e.g., showing "maximum(tax)" instead of "day_over_day(maximum(tax))").

2. **Missing Historical Dependencies**: Extended metrics fail when historical data isn't pre-computed because they don't automatically create the dependency symbols they need.

3. **False Unused Warnings**: Dependency symbols created for extended metrics trigger UnusedSymbolValidator warnings even though they're required.

## Solution Overview

1. Fix the naming issue in `evaluator.py`
2. Add `parent_symbol` tracking to `SymbolicMetric`
3. Implement automatic dependency creation in extended metric methods
4. Update `UnusedSymbolValidator` to skip symbols with parents

## Implementation Tasks

### Task Group 1: Fix Symbol Naming Display (TDD)

**Task 1.1: Write failing test for extended metric name display**
```python
# In tests/test_extended_metric_symbol_info.py
def test_extended_metric_symbol_info_displays_correct_name():
    """Test that SymbolInfo shows 'day_over_day(maximum(tax))' not 'maximum(tax)'."""
    # GIVEN: A metric provider with extended metrics
    db = InMemoryMetricDB()
    mp = MetricProvider(db)

    # WHEN: Creating an extended metric and collecting symbols
    base = mp.maximum("tax")
    dod = mp.ext.day_over_day(base)

    # Create evaluator and collect symbols
    key = ResultKey(yyyy_mm_dd="2024-10-24")
    evaluator = Evaluator(mp, key, "Test Suite")
    evaluator.add_symbols(dod)
    symbol_infos = evaluator.collect_symbols()

    # THEN: SymbolInfo should show the extended metric name
    dod_info = next(si for si in symbol_infos if si.name == str(dod))
    assert dod_info.metric == "day_over_day(maximum(tax))"
```

**Task 1.2: Fix the naming issue in evaluator.py**
- In `src/dqx/evaluator.py`, find the `collect_symbols` method
- Change `metric=str(sm.metric_spec),` to `metric=sm.name,`

**Task 1.3: Verify test passes and run linting**
```bash
uv run pytest tests/test_extended_metric_symbol_info.py::test_extended_metric_symbol_info_displays_correct_name -v
uv run mypy src/dqx/evaluator.py
uv run ruff check --fix src/dqx/evaluator.py
```

### Task Group 2: Add Parent Symbol Tracking (TDD)

**Task 2.1: Write failing test for parent symbol tracking**
```python
# In tests/test_provider.py
def test_extended_metrics_have_parent_symbols():
    """Test that extended metrics track their parent symbols."""
    # GIVEN: A metric provider
    mp = MetricProvider(InMemoryMetricDB())

    # WHEN: Creating an extended metric
    base = mp.maximum("tax")
    dod = mp.ext.day_over_day(base)

    # THEN: The extended metric should have the base as parent
    dod_metric = mp.get_symbol(dod)
    assert hasattr(dod_metric, 'parent_symbol')
    assert dod_metric.parent_symbol == base
```

**Task 2.2: Add parent_symbol field to SymbolicMetric**
```python
# In src/dqx/provider.py, update the SymbolicMetric dataclass
@dataclass
class SymbolicMetric:
    """Metadata for symbolic metrics."""

    name: str
    symbol: sp.Symbol
    fn: RetrievalFn
    key_provider: ResultKeyProvider
    metric_spec: MetricSpec
    dataset: str | None = None
    parent_symbol: sp.Symbol | None = None  # Add this field
```

**Task 2.3: Update ExtendedMetricProvider to set parent relationships**
```python
# In src/dqx/provider.py, update day_over_day method
def day_over_day(
    self,
    metric: sp.Symbol,
    key: ResultKeyProvider = ResultKeyProvider(),
    dataset: str | None = None
) -> sp.Symbol:
    """Create day-over-day change percentage metric."""
    metric_spec = self._resolve_metric_spec(metric)

    sym = self._next_symbol()
    self._provider._register(
        sym,
        name=f"day_over_day({metric_spec.name})",
        fn=partial(compute.day_over_day, self._db, metric_spec, key),
        key=key,
        metric_spec=metric_spec,
        dataset=dataset,
    )

    # Set parent relationship
    extended_metric = self._provider.get_symbol(sym)
    extended_metric.parent_symbol = metric

    return sym
```

**Task 2.4: Run tests and linting**
```bash
uv run pytest tests/test_provider.py::test_extended_metrics_have_parent_symbols -v
uv run mypy src/dqx/provider.py
uv run ruff check --fix src/dqx/provider.py
```

### Task Group 3: Implement Dependency Creation (TDD)

**Task 3.1: Write failing test for automatic dependency creation**
```python
# In tests/test_provider.py
def test_day_over_day_creates_dependency_symbols():
    """Test that day_over_day automatically creates lag_0 and lag_1 symbols."""
    # GIVEN: A metric provider
    mp = MetricProvider(InMemoryMetricDB())
    key = ResultKeyProvider()

    # WHEN: Creating a day-over-day metric
    base = mp.maximum("tax")
    initial_count = len(list(mp.symbols()))

    dod = mp.ext.day_over_day(base, key)

    # THEN: Should have created additional symbols
    final_count = len(list(mp.symbols()))
    assert final_count > initial_count + 1  # More than just base + dod

    # AND: Should have metrics for lag 0 and lag 1
    all_metrics = mp.symbolic_metrics
    lags = {sm.key_provider._lag for sm in all_metrics}
    assert 0 in lags
    assert 1 in lags
```

**Task 3.2: Add helper method for creating metric windows**
```python
# In src/dqx/provider.py, add to ExtendedMetricProvider class
def _create_metric_window(
    self,
    metric: sp.Symbol,
    start_lag: int,
    window_size: int,
    key: ResultKeyProvider,
    dataset: str | None = None,
    parent_symbol: sp.Symbol | None = None
) -> list[sp.Symbol]:
    """Create metric symbols for a time window.

    Args:
        metric: Base metric symbol to create window for
        start_lag: Starting lag (e.g., 0 for current day)
        window_size: Number of days in window
        key: Result key provider
        dataset: Optional dataset name
        parent_symbol: The extended metric symbol that depends on these

    Returns:
        List of symbols for each lag in the window
    """
    metric_spec = self._resolve_metric_spec(metric)
    symbols = []

    for lag in range(start_lag, start_lag + window_size):
        lagged_key = key.lag(lag) if lag > 0 else key
        sym = self._provider.metric(metric_spec, lagged_key, dataset)

        # Set parent if provided
        if parent_symbol:
            symbolic_metric = self._provider.get_symbol(sym)
            symbolic_metric.parent_symbol = parent_symbol

        symbols.append(sym)

    return symbols
```

**Task 3.3: Update day_over_day to create dependencies**
```python
# In src/dqx/provider.py, update day_over_day method
def day_over_day(
    self,
    metric: sp.Symbol,
    key: ResultKeyProvider = ResultKeyProvider(),
    dataset: str | None = None
) -> sp.Symbol:
    """Create day-over-day change percentage metric with automatic dependencies."""
    metric_spec = self._resolve_metric_spec(metric)

    # First create the extended metric symbol
    sym = self._next_symbol()
    self._provider._register(
        sym,
        name=f"day_over_day({metric_spec.name})",
        fn=partial(compute.day_over_day, self._db, metric_spec, key),
        key=key,
        metric_spec=metric_spec,
        dataset=dataset,
    )

    # Set parent relationship to input metric
    extended_metric = self._provider.get_symbol(sym)
    extended_metric.parent_symbol = metric

    # Now create dependency symbols with extended metric as parent
    self._create_metric_window(
        metric,
        start_lag=0,
        window_size=2,
        key=key,
        dataset=dataset,
        parent_symbol=sym
    )

    return sym
```

**Task 3.4: Run tests and fix any issues**
```bash
uv run pytest tests/test_provider.py::test_day_over_day_creates_dependency_symbols -v
uv run mypy src/dqx/provider.py
uv run ruff check --fix src/dqx/provider.py
```

### Task Group 4: Update Other Extended Metrics (TDD)

**Task 4.1: Write tests for week_over_week and stddev**
```python
# In tests/test_provider.py
def test_week_over_week_creates_lag_7_dependency():
    """Test that week_over_week creates lag_0 and lag_7 symbols."""
    mp = MetricProvider(InMemoryMetricDB())
    key = ResultKeyProvider()

    base = mp.sum("price")
    wow = mp.ext.week_over_week(base, key)

    # Verify lag_7 symbol exists
    all_metrics = mp.symbolic_metrics
    lags = {sm.key_provider._lag for sm in all_metrics}
    assert 0 in lags
    assert 7 in lags

def test_stddev_creates_window_dependencies():
    """Test that stddev creates all required window symbols."""
    mp = MetricProvider(InMemoryMetricDB())

    base = mp.average("price")
    stddev = mp.ext.stddev(base, lag=1, n=3)

    # Check we have metrics for lags 1, 2, 3
    all_metrics = mp.symbolic_metrics
    lags = {sm.key_provider._lag for sm in all_metrics}
    assert {1, 2, 3}.issubset(lags)
```

**Task 4.2: Update week_over_week method**
```python
# In src/dqx/provider.py
def week_over_week(
    self,
    metric: sp.Symbol,
    key: ResultKeyProvider = ResultKeyProvider(),
    dataset: str | None = None
) -> sp.Symbol:
    """Create week-over-week change percentage metric with automatic dependencies."""
    metric_spec = self._resolve_metric_spec(metric)

    # Create the extended metric
    sym = self._next_symbol()
    self._provider._register(
        sym,
        name=f"week_over_week({metric_spec.name})",
        fn=partial(compute.week_over_week, self._db, metric_spec, key),
        key=key,
        metric_spec=metric_spec,
        dataset=dataset,
    )

    # Set parent relationship
    extended_metric = self._provider.get_symbol(sym)
    extended_metric.parent_symbol = metric

    # Create specific dependencies (lag 0 and lag 7)
    for lag in [0, 7]:
        lagged_key = key.lag(lag) if lag > 0 else key
        dep_sym = self._provider.metric(metric_spec, lagged_key, dataset)

        dep_metric = self._provider.get_symbol(dep_sym)
        dep_metric.parent_symbol = sym

    return sym
```

**Task 4.3: Update stddev method**
```python
# In src/dqx/provider.py
def stddev(
    self,
    metric: sp.Symbol,
    lag: int,
    n: int,
    key: ResultKeyProvider = ResultKeyProvider(),
    dataset: str | None = None,
) -> sp.Symbol:
    """Create standard deviation metric with automatic dependencies."""
    metric_spec = self._resolve_metric_spec(metric)

    # Create the extended metric
    sym = self._next_symbol()
    self._provider._register(
        sym,
        name=f"stddev({metric_spec.name})",
        fn=partial(compute.stddev, self._db, metric_spec, lag, n, key),
        key=key,
        metric_spec=metric_spec,
        dataset=dataset,
    )

    # Set parent relationship
    extended_metric = self._provider.get_symbol(sym)
    extended_metric.parent_symbol = metric

    # Create dependency symbols for the window
    self._create_metric_window(
        metric,
        start_lag=lag,
        window_size=n,
        key=key,
        dataset=dataset,
        parent_symbol=sym
    )

    return sym
```

**Task 4.4: Run all extended metric tests**
```bash
uv run pytest tests/test_provider.py -k "test_week_over_week_creates_lag_7_dependency or test_stddev_creates_window_dependencies" -v
uv run mypy src/dqx/provider.py
uv run ruff check --fix src/dqx/provider.py
```

### Task Group 5: Update UnusedSymbolValidator (TDD)

**Task 5.1: Write test for validator ignoring non-orphan symbols**
```python
# In tests/test_validator.py
def test_unused_validator_ignores_symbols_with_parents():
    """Test that UnusedSymbolValidator doesn't warn about symbols with parents."""
    # GIVEN: A provider with extended metrics
    mp = MetricProvider(InMemoryMetricDB())
    base = mp.maximum("tax")
    dod = mp.ext.day_over_day(base)

    # AND: A graph with only the dod symbol used
    graph = Graph()
    check = CheckNode("test_check")
    assertion = AssertionNode(
        name="dod_check",
        actual=dod,
        validator=SymbolicValidator("geq", lambda x: x >= 0.5)
    )
    check.add_child(assertion)
    graph.root.add_child(check)

    # WHEN: Running validation
    validator = SuiteValidator()
    report = validator.validate(graph, mp)

    # THEN: Should not warn about dependency metrics
    unused_warnings = [w for w in report.warnings
                      if w.rule == "unused_symbols"]

    # Only the base metric should be warned (not used directly)
    assert len(unused_warnings) == 1
    assert "maximum(tax)" in unused_warnings[0].message
```

**Task 5.2: Update UnusedSymbolValidator to skip symbols with parents**
```python
# In src/dqx/validator.py, update the finalize method
def finalize(self) -> None:
    """Compare defined vs used symbols and generate warnings."""
    # Get all defined symbols from provider
    defined_symbols = set(self._provider.symbols())

    # Find unused symbols
    unused_symbols = defined_symbols - self._used_symbols

    # Generate warnings for each unused symbol (skip if has parent)
    for symbol in unused_symbols:
        metric = self._provider.get_symbol(symbol)

        # Skip symbols that have parents - they're not orphans
        if metric.parent_symbol is not None:
            continue

        # Format: symbol_name ← metric_name
        symbol_repr = f"{symbol} ← {metric.name}"

        self._issues.append(
            ValidationIssue(
                rule=self.name,
                message=f"Unused symbol: {symbol_repr}",
                node_path=["root", symbol_repr],
            )
        )
```

**Task 5.3: Run validator tests**
```bash
uv run pytest tests/test_validator.py::test_unused_validator_ignores_symbols_with_parents -v
uv run mypy src/dqx/validator.py
uv run ruff check --fix src/dqx/validator.py
```

### Task Group 6: Integration Testing and Final Verification

**Task 6.1: Create comprehensive integration test**
```python
# In tests/test_extended_metric_symbol_info.py
def test_extended_metrics_full_integration():
    """Test full integration of extended metrics fixes."""
    # Setup
    db = InMemoryMetricDB()
    mp = MetricProvider(db)

    # Create various extended metrics
    tax_base = mp.maximum("tax")
    tax_dod = mp.ext.day_over_day(tax_base)

    price_base = mp.sum("price")
    price_wow = mp.ext.week_over_week(price_base)

    avg_base = mp.average("revenue")
    avg_stddev = mp.ext.stddev(avg_base, lag=1, n=3)

    # Create graph using only extended metrics
    graph = Graph()
    check = CheckNode("metrics_check")

    for sym, name in [(tax_dod, "tax_dod"), (price_wow, "price_wow"), (avg_stddev, "avg_stddev")]:
        assertion = AssertionNode(
            name=name,
            actual=sym,
            validator=SymbolicValidator("geq", lambda x: x >= 0)
        )
        check.add_child(assertion)

    graph.root.add_child(check)

    # Validate
    validator = SuiteValidator()
    report = validator.validate(graph, mp)

    # Only base metrics should be warned as unused
    unused_warnings = [w for w in report.warnings if w.rule == "unused_symbols"]
    assert len(unused_warnings) == 3

    # Verify names in warnings
    warning_messages = [w.message for w in unused_warnings]
    assert any("maximum(tax)" in msg for msg in warning_messages)
    assert any("sum(price)" in msg for msg in warning_messages)
    assert any("average(revenue)" in msg for msg in warning_messages)

    # No warnings for dependency metrics
    assert not any("lag" in msg.lower() for msg in warning_messages)
```

**Task 6.2: Run all tests to ensure nothing is broken**
```bash
uv run pytest tests/test_extended_metric_symbol_info.py -v
uv run pytest tests/test_provider.py -v
uv run pytest tests/test_validator.py -v
```

**Task 6.3: Run full test suite and pre-commit checks**
```bash
# Run full test suite
uv run pytest tests/ -v

# Check test coverage
uv run pytest tests/ -v --cov=dqx.provider --cov=dqx.evaluator --cov=dqx.validator

# Run pre-commit checks
bin/run-hooks.sh
```

### Task Group 7: Improve print_symbols Display (TDD)

**Task 7.1: Write test for hierarchical symbol display**
```python
# In tests/test_display.py
def test_print_symbols_with_hierarchical_display(capsys):
    """Test that print_symbols shows parent-child relationships with indentation."""
    from dqx.common import SymbolInfo
    from dqx.display import print_symbols
    from returns.result import Success
    from datetime import date

    # Create symbols with parent-child relationships
    symbols = [
        SymbolInfo(
            name="x_2",
            metric="day_over_day(maximum(tax))",
            dataset="sales",
            value=Success(0.15),
            yyyy_mm_dd=date(2024, 10, 24),
            suite="Suite",
            tags={}
        ),
        SymbolInfo(
            name="x_1",
            metric="maximum(tax)",
            dataset="sales",
            value=Success(100.0),
            yyyy_mm_dd=date(2024, 10, 24),
            suite="Suite",
            tags={}
        ),
        SymbolInfo(
            name="x_3",
            metric="maximum(tax) [lag=1]",
            dataset="sales",
            value=Success(87.0),
            yyyy_mm_dd=date(2024, 10, 24),
            suite="Suite",
            tags={}
        ),
    ]

    print_symbols(symbols)
    captured = capsys.readouterr()

    # Check for indentation in output
    assert "x_2" in captured.out
    assert "└─ x_1" in captured.out
    assert "└─ x_3" in captured.out
```

**Task 7.2: Add parent tracking to SymbolInfo**
Since SymbolInfo is likely frozen/immutable, we'll need to handle parent-child relationships during display by analyzing the symbols.

**Task 7.3: Update print_symbols to show hierarchy**
```python
# In src/dqx/display.py
def print_symbols(symbols: list[SymbolInfo], show_dependencies: bool = True) -> None:
    """
    Display symbol values in a formatted table with hierarchical grouping.

    Shows all fields from SymbolInfo objects in a table with columns:
    Date, Suite, Symbol, Metric, Dataset, Value/Error, Tags

    Dependencies are shown indented under their parent symbols.

    Args:
        symbols: List of SymbolInfo objects from collect_symbols()
        show_dependencies: Whether to show dependency symbols (default: True)

    Example:
        >>> suite = VerificationSuite(checks, db, "My Suite")
        >>> suite.run(datasources, key)
        >>> symbols = suite.collect_symbols()
        >>> print_symbols(symbols)
    """
    from returns.result import Failure, Success
    from rich.table import Table

    # Create table with title
    table = Table(title="Symbol Values", show_lines=True)

    # Add columns in specified order
    table.add_column("Date", style="cyan", no_wrap=True)
    table.add_column("Suite", style="blue")
    table.add_column("Symbol", style="yellow", no_wrap=True)
    table.add_column("Metric")
    table.add_column("Dataset", style="magenta")
    table.add_column("Value/Error")
    table.add_column("Tags", style="dim")

    # Group symbols by parent-child relationships
    # Extended metrics contain base metric names in them
    displayed_symbols = set()

    for symbol in symbols:
        if symbol.name in displayed_symbols:
            continue

        # Add parent row
        _add_symbol_row(table, symbol)
        displayed_symbols.add(symbol.name)

        if show_dependencies:
            # Find child symbols (those whose metric appears in parent metric)
            children = [
                s for s in symbols
                if s.name != symbol.name
                and s.name not in displayed_symbols
                and _is_dependency_of(s, symbol)
            ]

            # Add child rows with indentation
            for child in children:
                _add_symbol_row(table, child, indent="  └─ ")
                displayed_symbols.add(child.name)

    # Print table
    console = Console()
    console.print(table)


def _is_dependency_of(child: SymbolInfo, parent: SymbolInfo) -> bool:
    """Check if child symbol is a dependency of parent symbol."""
    # Extract base metric from child (remove lag info)
    base_metric = child.metric.split(" [lag=")[0]

    # Check if base metric appears in parent metric
    return base_metric in parent.metric and (
        "day_over_day" in parent.metric
        or "week_over_week" in parent.metric
        or "stddev" in parent.metric
    )


def _add_symbol_row(
    table: Table,
    symbol: SymbolInfo,
    indent: str = ""
) -> None:
    """Add a symbol row to the table with optional indentation."""
    # Extract value/error using pattern matching with colors
    match symbol.value:
        case Success(value):
            value_display = f"[green]{value}[/green]"
        case Failure(error):
            value_display = f"[red]{error}[/red]"

    # Format tags as key=value pairs
    tags_display = ", ".join(f"{k}={v}" for k, v in symbol.tags.items())
    if not tags_display:
        tags_display = "-"

    # Add row with optional indentation on symbol
    table.add_row(
        symbol.yyyy_mm_dd.isoformat(),
        symbol.suite,
        f"{indent}{symbol.name}",
        symbol.metric,
        symbol.dataset or "-",
        value_display,
        tags_display,
    )
```

**Task 7.4: Update metric names to include lag information**
When creating dependency symbols, include lag information in the metric name for clarity.

**Task 7.5: Run tests and verify display**
```bash
uv run pytest tests/test_display.py::test_print_symbols_with_hierarchical_display -v
uv run mypy src/dqx/display.py
uv run ruff check --fix src/dqx/display.py
```

## Success Criteria

1. Extended metrics display correct names in symbol collection
2. Extended metrics automatically create their required dependencies
3. Dependency symbols don't trigger unused warnings
4. Symbol display shows hierarchical relationships clearly
5. All tests pass with 100% coverage
6. No linting or type checking errors
7. Pre-commit hooks pass

## Notes for Implementation

- Follow TDD strictly: write test first, see it fail, then implement
- Commit after each task group completion
- If any test reveals additional issues, add them to the plan
- The `parent_symbol` field creates a dependency graph that can be used for future enhancements
- The print_symbols improvement provides better visibility into metric relationships
