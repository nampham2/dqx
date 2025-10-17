# Lag Date Handling Fix Implementation Plan v1

## Overview

This plan addresses a critical bug in DQX where metrics with different lags (e.g., `mp.average("tax")` and `mp.average("tax", key=ctx.key.lag(1))`) are computed for the same date instead of their intended dates. This breaks time-series comparisons like day-over-day calculations.

## Problem Statement

### Current Behavior
When a check uses metrics with different lags:
```python
tax_avg = mp.average("tax")  # Should use 2025-01-15
tax_avg_lag = mp.average("tax", key=ctx.key.lag(1))  # Should use 2025-01-14
```

Both metrics are computed for the same date (2025-01-15) because:
1. The analyzer receives a single `ResultKey` and applies it to all metrics
2. The lag information stored in `SymbolicMetric.key_provider` is ignored during analysis
3. The `collect_symbols()` method reports incorrect dates for lagged metrics

### Expected Behavior
- `tax_avg` should be computed for the current date (2025-01-15)
- `tax_avg_lag` should be computed for the previous date (2025-01-14)
- Symbol collection should report the correct effective date for each metric

## Implementation Tasks

### Phase 0: Setup

#### Task 0.1: Create feature branch
```bash
git checkout -b fix-lag-date-handling
```

#### Task 0.2: Commit the implementation plan
```bash
git add docs/plans/lag_date_handling_fix_plan_v1.md
git commit -m "docs: Add implementation plan for lag date handling fix"
```

### Phase 1: Infrastructure Only (No behavior changes)

#### Task 1.1: Update SqlDataSource Protocol
**File**: `src/dqx/common.py`

Change the SqlDataSource protocol to support nominal_date parameter:
```python
@runtime_checkable
class SqlDataSource(Protocol):
    """
    Protocol for SQL data sources that can be analyzed by DQX.

    This protocol defines the interface for adapting various data sources
    (e.g., Arrow tables, BigQuery, DuckDB tables) to work with the DQX
    analysis framework.

    Attributes:
        name: A unique identifier for this data source instance
        dialect: The SQL dialect name used for query generation
    """

    name: str
    dialect: str

    def cte(self, nominal_date: date) -> str:
        """
        Get the Common Table Expression (CTE) for this data source.

        Args:
            nominal_date: The date for which the CTE should filter data.
                         Implementations may ignore this parameter if date
                         filtering is not needed.

        Returns:
            The CTE SQL string
        """
        ...

    def query(self, query: str, nominal_date: date) -> duckdb.DuckDBPyRelation:
        """
        Execute a query against this data source.

        Args:
            query: The SQL query to execute
            nominal_date: The date for which data should be filtered.
                         Implementations may ignore this parameter if date
                         filtering is not needed.

        Returns:
            Query results as a DuckDB relation
        """
        ...
```

#### Task 1.2: Update ArrowDataSource.cte() to accept nominal_date parameter
**File**: `src/dqx/extensions/pyarrow_ds.py`

Update the cte property to a method:
```python
def cte(self, nominal_date: date) -> str:
    """
    Get the CTE for this data source.

    Args:
        nominal_date: The date for filtering

    Returns:
        The CTE SQL string
    """
    return f"SELECT * FROM {self._table_name}"
```

#### Task 1.3: Update ArrowDataSource.query() to accept nominal_date parameter
**File**: `src/dqx/extensions/pyarrow_ds.py`

Update the query method:
```python
def query(self, query: str, nominal_date: date) -> duckdb.DuckDBPyRelation:
    """
    Execute a query against the Arrow data.

    Args:
        query: The SQL query to execute
        nominal_date: The date for filtering

    Returns:
        Query results as a DuckDB relation
    """
    return self._conn.execute(query)
```

#### Task 1.4: Update DuckRelationDataSource.cte() to accept nominal_date parameter
**File**: `src/dqx/extensions/duck_ds.py`

Update the cte property to a method:
```python
def cte(self, nominal_date: date) -> str:
    """
    Get the CTE for this data source.

    Args:
        nominal_date: The date for filtering

    Returns:
        The CTE SQL string
    """
    return f"SELECT * FROM {self._table_name}"
```

#### Task 1.5: Update DuckRelationDataSource.query() to accept nominal_date parameter
**File**: `src/dqx/extensions/duck_ds.py`

Update the query method:
```python
def query(self, query: str, nominal_date: date) -> duckdb.DuckDBPyRelation:
    """
    Execute a query against the DuckDB relation.

    Args:
        query: The SQL query to execute
        nominal_date: The date for filtering

    Returns:
        Query results as a DuckDB relation
    """
    return self._conn.execute(query)
```

#### Task 1.6: Update analyze_sql_ops() signature to accept nominal_date
**File**: `src/dqx/analyzer.py`

Update the analyzer functions to pass the nominal date:

#### Task 1.7: Update analyze_sql_ops() to pass nominal_date to ds.cte() and ds.query()
**File**: `src/dqx/analyzer.py`

In `analyze_sql_ops`:
```python
def analyze_sql_ops(ds: T, ops: Sequence[SqlOp], nominal_date: date) -> None:
    if len(ops) == 0:
        return

    # ... existing deduplication code ...

    # Generate SQL expressions using the dialect
    expressions = [dialect_instance.translate_sql_op(op) for op in distinct_ops]
    sql = dialect_instance.build_cte_query(ds.cte(nominal_date), expressions)

    # Execute the query
    logger.debug(f"SQL Query:\n{sql}")
    result: dict[str, np.ndarray] = ds.query(sql, nominal_date).fetchnumpy()

    # ... rest of the function ...
```

#### Task 1.8: Update analyze_sketch_ops() signature to accept nominal_date
**File**: `src/dqx/analyzer.py`

#### Task 1.9: Update analyze_sketch_ops() to pass nominal_date to ds.cte() and ds.query()
**File**: `src/dqx/analyzer.py`

In `analyze_sketch_ops`:
```python
def analyze_sketch_ops(ds: T, ops: Sequence[SketchOp], batch_size: int, nominal_date: date) -> None:
    if len(ops) == 0:
        return

    # ... existing deduplication code ...

    # Constructing the query
    logger.info(f"Analyzing SketchOps: {distinct_ops}")
    query = textwrap.dedent(f"""\
        WITH source AS ( {ds.cte(nominal_date)})
        SELECT {", ".join(op.sketch_column for op in distinct_ops)} FROM source
        """)

    # Fetch the only the required columns into memory by batch
    sketches = {op: op.create() for op in ops}
    batches = ds.query(query, nominal_date).fetch_arrow_reader(
        batch_size=batch_size,
    )

    # ... rest of the function ...
```

#### Task 1.10: Update Analyzer.analyze() to pass key.yyyy_mm_dd to analyze functions
**File**: `src/dqx/analyzer.py`

Update the `Analyzer.analyze` method to pass the date:
```python
def analyze(
    self,
    ds: SqlDataSource,
    metrics: Sequence[MetricSpec],
    key: ResultKey,
) -> AnalysisReport:
    # ... existing code ...

    # Analyze sql ops - pass the date from key
    sql_ops = [op for op in all_ops if isinstance(op, SqlOp)]
    analyze_sql_ops(ds, sql_ops, key.yyyy_mm_dd)

    # Analyze sketch ops - pass the date from key
    sketch_ops = [op for op in all_ops if isinstance(op, SketchOp)]
    analyze_sketch_ops(ds, sketch_ops, batch_size=100_000, nominal_date=key.yyyy_mm_dd)

    # ... rest of the function ...
```

#### Task 1.11: Update FakeSqlDataSource in test_analyzer.py
**File**: `tests/test_analyzer.py`

Update FakeSqlDataSource:
```python
class FakeSqlDataSource:
    """A test implementation of SqlDataSource protocol."""

    def __init__(
        self,
        name: str = "test_ds",
        cte_str: str = "SELECT * FROM test",
        dialect: str = "duckdb",
        data: dict[str, np.ndarray] | None = None,
    ):
        self.name = name
        self._cte_str = cte_str
        self.dialect = dialect
        self._data = data or {}

    def cte(self, nominal_date: date) -> str:
        """Return the CTE string."""
        return self._cte_str

    def query(self, query: str, nominal_date: date) -> duckdb.DuckDBPyRelation:
        """Return a mock DuckDB relation."""
        mock_relation = Mock(spec=duckdb.DuckDBPyRelation)
        mock_relation.fetchnumpy.return_value = self._data
        mock_relation.fetch_arrow_reader.return_value = FakeRelation(self._data).fetch_arrow_reader(100000)
        return mock_relation
```

#### Task 1.12: Fix test_evaluator_validation.py mock assignments
**File**: `tests/test_evaluator_validation.py`

Replace direct property assignment:
```python
# Old: datasource.cte = "SELECT * FROM test"
# New: Mock the cte method
datasource.cte = Mock(return_value="SELECT * FROM test")
```

#### Task 1.13: Update Context.pending_metrics() return type and implementation
**File**: `src/dqx/api.py`

Change the return type from `Sequence[MetricSpec]` to `Sequence[SymbolicMetric]`:
```python
def pending_metrics(self, dataset: str | None = None) -> Sequence[SymbolicMetric]:
    """
    Get pending metrics for the specified dataset or all datasets if none specified.

    Returns SymbolicMetric objects that contain both the metric specification
    and the key provider with lag information.
    """
    all_metrics = self.provider.symbolic_metrics
    if dataset:
        return [metric for metric in all_metrics if metric.dataset == dataset]
    return all_metrics
```

**Note**: Remove the TODO comment about TTL filtering as it's out of scope.

#### Task 1.14: Add SymbolicMetric import to api.py
**File**: `src/dqx/api.py`

Add the import for SymbolicMetric:
```python
from dqx.provider import MetricProvider, SymbolicMetric
```

#### Task 1.15: Run tests to verify no breaks
```bash
uv run pytest tests/ -x
```

#### Task 1.16: Commit Phase 1
```bash
git add -A
git commit -m "feat: Add infrastructure for nominal_date parameter support

- Update SqlDataSource protocol to accept nominal_date
- Update data source implementations (ArrowDataSource, DuckRelationDataSource)
- Update analyzer functions to pass nominal_date
- Update pending_metrics to return SymbolicMetric objects
- Update test mocks to match new signatures

No behavior changes - implementations currently ignore the parameter."
```

### Phase 2: Implement the Fix

#### Task 2.1: Add defaultdict import to api.py
**File**: `src/dqx/api.py`

Add the import:
```python
from collections import defaultdict
```

#### Task 2.2: Update VerificationSuite.run() to get symbolic metrics
**File**: `src/dqx/api.py`

In the `run()` method, change how we get metrics:
```python
# Get all symbolic metrics for this dataset
symbolic_metrics = self._context.pending_metrics(ds_name)
```

#### Task 2.3: Implement date grouping logic in run()
**File**: `src/dqx/api.py`

Add the grouping logic:
```python
# Group metrics by their effective date
metrics_by_date: dict[ResultKey, list[MetricSpec]] = defaultdict(list)
for sym_metric in symbolic_metrics:
    effective_key = sym_metric.key_provider.create(key)
    metrics_by_date[effective_key].append(sym_metric.metric_spec)
```

#### Task 2.4: Update analyzer loop to analyze each date group
**File**: `src/dqx/api.py`

Replace the single analyze call with a loop:
```python
# Analyze each date group separately
analyzer = Analyzer()
for effective_key, metrics in metrics_by_date.items():
    logger.info(f"Analyzing {len(metrics)} metrics for {effective_key.yyyy_mm_dd}")
    analyzer.analyze(datasource, metrics, effective_key)

# Persist the combined report
analyzer.report.persist(self.provider._db)
```

#### Task 2.5: Update collect_symbols() to calculate effective dates
**File**: `src/dqx/api.py`

In `collect_symbols()`, calculate the effective key:
```python
# Calculate the effective key for this symbol
effective_key = symbolic_metric.key_provider.create(self._key)
```

#### Task 2.6: Update collect_symbols() to use effective dates in SymbolInfo
**File**: `src/dqx/api.py`

Use the effective date when creating SymbolInfo:
```python
symbol_info = SymbolInfo(
    name=str(symbolic_metric.symbol),
    metric=str(symbolic_metric.metric_spec),
    dataset=symbolic_metric.dataset,
    value=value,
    yyyy_mm_dd=effective_key.yyyy_mm_dd,  # Use effective date!
    suite=self._name,
    tags=effective_key.tags,
)
```

#### Task 2.7: Run e2e test to verify fix works
```bash
uv run pytest tests/e2e/test_api_e2e.py::test_verification_suite -xvs
```

#### Task 2.8: Commit Phase 2
```bash
git add -A
git commit -m "fix: Implement lag date handling fix

- Group metrics by effective date in VerificationSuite.run()
- Use effective dates in collect_symbols()
- Metrics with different lags now compute for correct dates

Fixes the bug where lagged metrics were computed for the wrong date."
```

### Phase 3: Add Comprehensive Tests

#### Task 3.1: Write test_pending_metrics_returns_symbolic_metrics()
**File**: `tests/test_api.py`

Add test to verify pending_metrics returns SymbolicMetric objects:
```python
def test_pending_metrics_returns_symbolic_metrics():
    """Test that pending_metrics returns SymbolicMetric objects with key providers."""
    db = InMemoryMetricDB()
    ctx = Context("Test Suite", db)

    # Create metrics with different lags
    mp = ctx.provider
    m1 = mp.average("price")
    m2 = mp.average("price", key=ctx.key.lag(1))

    pending = ctx.pending_metrics()

    assert len(pending) == 2
    assert all(isinstance(m, SymbolicMetric) for m in pending)
    assert pending[0].key_provider._lag == 0
    assert pending[1].key_provider._lag == 1
```

#### Task 3.2: Write test_suite_analyzes_metrics_with_correct_dates()
**File**: `tests/test_api.py`

Add test to verify metrics are analyzed with correct dates:
```python
def test_suite_analyzes_metrics_with_correct_dates(mocker):
    """Test that metrics with different lags are analyzed for correct dates."""
    db = InMemoryMetricDB()

    @check(name="Test Check", datasets=["ds1"])
    def test_check(mp: MetricProvider, ctx: Context) -> None:
        current = mp.average("value")
        lagged = mp.average("value", key=ctx.key.lag(1))
        ctx.assert_that(current / lagged).where(name="Ratio check").is_eq(1.0)

    # Mock the analyzer to track calls
    analyze_spy = mocker.spy(Analyzer, 'analyze')

    suite = VerificationSuite([test_check], db, "Test Suite")
    key = ResultKey(datetime.date(2025, 1, 15), {})

    # Create test data
    data = pa.table({"value": [100.0] * 10})
    ds = ArrowDataSource(data)

    suite.run({"ds1": ds}, key)

    # Verify analyzer was called with different dates
    calls = analyze_spy.call_args_list
    assert len(calls) == 2

    # Extract the keys from the calls
    keys_used = [call[0][2] for call in calls]
    dates_used = {k.yyyy_mm_dd for k in keys_used}

    assert datetime.date(2025, 1, 15) in dates_used
    assert datetime.date(2025, 1, 14) in dates_used
```

#### Task 3.3: Write test_collect_symbols_with_lagged_dates()
**File**: `tests/test_symbol_collection.py`

Add test to verify correct dates in collected symbols:
```python
def test_collect_symbols_with_lagged_dates():
    """Test that collected symbols show correct effective dates for lagged metrics."""
    db = InMemoryMetricDB()

    @check(name="Time Series Check", datasets=["ds1"])
    def time_series_check(mp: MetricProvider, ctx: Context) -> None:
        current = mp.average("value")  # Should be 2025-01-15
        lag1 = mp.average("value", key=ctx.key.lag(1))  # Should be 2025-01-14
        lag2 = mp.average("value", key=ctx.key.lag(2))  # Should be 2025-01-13

        # Create assertions to ensure symbols are registered
        ctx.assert_that(current).where(name="Current check").is_gt(0)
        ctx.assert_that(lag1).where(name="Lag 1 check").is_gt(0)
        ctx.assert_that(lag2).where(name="Lag 2 check").is_gt(0)

    suite = VerificationSuite([time_series_check], db, "Test Suite")
    key = ResultKey(datetime.date(2025, 1, 15), {"env": "test"})

    # Create test data
    data = pa.table({"value": [100.0] * 10})
    ds = ArrowDataSource(data)

    # Run suite
    suite.run({"ds1": ds}, key)

    # Collect symbols
    symbols = suite.collect_symbols()

    # Find the three symbols we created
    current_sym = next(s for s in symbols if "lag" not in str(s.metric))
    lag1_sym = next(s for s in symbols if "lag(1)" in str(s.metric))
    lag2_sym = next(s for s in symbols if "lag(2)" in str(s.metric))

    # Verify dates
    assert current_sym.yyyy_mm_dd == datetime.date(2025, 1, 15)
    assert lag1_sym.yyyy_mm_dd == datetime.date(2025, 1, 14)
    assert lag2_sym.yyyy_mm_dd == datetime.date(2025, 1, 13)

    # Verify tags are preserved
    assert all(s.tags == {"env": "test"} for s in [current_sym, lag1_sym, lag2_sym])
```

#### Task 3.4: Add nominal_date usage documentation
**File**: `src/dqx/extensions/pyarrow_ds.py`

Add documentation showing how future data sources could use nominal_date:
```python
"""
Note: For time-series analysis, data sources can use the nominal_date parameter
to filter data appropriately. The current implementation ignores this parameter,
but custom data sources can leverage it:

Example:
    class DateFilteredArrowSource(ArrowDataSource):
        def __init__(self, table: pa.Table, date_column: str):
            super().__init__(table)
            self.date_column = date_column

        def cte(self, nominal_date: date) -> str:
            # Filter data for specific date in the CTE
            return f'''
                SELECT * FROM {self._table_name}
                WHERE DATE({self.date_column}) = '{nominal_date.isoformat()}'
            '''
"""
```

## Known Limitations

1. **Data Source Date Filtering**: The current fix assumes data sources contain all necessary dates. If a data source needs date-specific filtering, it must be handled at the data source level. A future enhancement could pass the effective date to the data source's CTE method.

2. **Memory Usage**: Analyzing metrics for multiple dates may increase memory usage since each date group creates separate analysis results. This is typically not an issue for day-over-day comparisons but could matter for large windows.

3. **Performance**: Each unique date requires a separate analysis pass. For checks with many different lag values, this could impact performance. Consider consolidating lag values where possible.

#### Task 3.5: Run all new tests
```bash
uv run pytest tests/test_api.py::test_pending_metrics_returns_symbolic_metrics -xvs
uv run pytest tests/test_api.py::test_suite_analyzes_metrics_with_correct_dates -xvs
uv run pytest tests/test_symbol_collection.py::test_collect_symbols_with_lagged_dates -xvs
```

#### Task 3.6: Commit Phase 3
```bash
git add -A
git commit -m "test: Add comprehensive tests for lag date handling

- Test pending_metrics returns SymbolicMetric objects
- Test metrics are analyzed with correct dates
- Test symbol collection reports correct effective dates
- Add documentation for nominal_date usage"
```

### Phase 4: Final Validation

#### Task 4.1: Run complete test suite
```bash
uv run pytest tests/ -x
```

#### Task 4.2: Run pre-commit hooks
```bash
bin/run-hooks.sh --fix
```

#### Task 4.3: Fix any linting/formatting issues
If pre-commit hooks report issues:
```bash
uv run ruff check --fix
uv run mypy src/
```

#### Task 4.4: Final commit
```bash
git add -A
git commit -m "chore: Fix linting and formatting issues"
```

#### Task 4.5: Run existing e2e test to verify fix
```bash
uv run pytest tests/e2e/test_api_e2e.py::test_verification_suite -xvs
```

Verify that symbols x_7 and x_8 now show different dates in the output.

#### Task 4.6: Push branch and create merge request
```bash
git push origin fix-lag-date-handling
```

Create merge request with description explaining the fix.

## Backward Compatibility

This fix maintains backward compatibility:
- The API changes are additive (returning more information, not less)
- Existing code that doesn't use lag will continue to work unchanged
- The fix only affects behavior when lag is explicitly used

## Testing Strategy

1. **Unit Tests**: Test each component in isolation
   - `pending_metrics` returns SymbolicMetric objects
   - Date grouping logic works correctly
   - Symbol collection reports correct dates

2. **Integration Tests**: Test the full flow
   - Metrics with different lags are analyzed for correct dates
   - Symbol collection shows correct effective dates
   - The e2e test passes with correct dates

3. **Manual Testing**: Run the e2e test and inspect output
   - Verify x_7 shows date 2025-01-15
   - Verify x_8 shows date 2025-01-14
   - Verify day-over-day calculations work correctly

## Rollback Plan

If issues arise:
1. Git revert the commit
2. Return to main branch: `git checkout main`
3. The bug will remain but system will be stable

## Success Criteria

1. The "Manual Day Over Day" check computes metrics for correct dates
2. `collect_symbols()` reports correct effective dates for each symbol
3. All existing tests pass
4. No performance regression for checks without lag

## Summary

This plan fixes the lag date handling bug by:
1. Preserving lag information through the analysis pipeline
2. Grouping metrics by their effective date before analysis
3. Reporting correct effective dates in symbol collection

The fix is minimal, focused, and maintains backward compatibility while enabling proper time-series analysis in DQX.
