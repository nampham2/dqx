# Analyzer Batch Implementation Plan v1

## Overview

This plan describes the implementation of batch analysis capability for DQX, allowing multiple dates with different metrics to be analyzed in a single SQL query. This improves performance for time-series analysis by reducing database round trips.

## Background

Currently, the `Analyzer` class processes one `ResultKey` (date) at a time. For time-series analysis, this results in N separate SQL queries for N dates. The BatchAnalyzer feature will process multiple dates in a single query using SQL UNION ALL.

### Key Design Decisions
- Modify the existing `Analyzer` class to support batch operations (not create a separate class)
- Use date-suffixed CTEs (e.g., `source_2024_01_01`, `metrics_2024_01_01`)
- Compute metrics in SELECT statements, then unpivot to rows
- Return the same `AnalysisReport` structure
- **No backward compatibility needed** - breaking changes are acceptable
- Batch size limit of 100 dates per query
- Fail the entire batch if any date fails

## Implementation Tasks

### Task Group 1: Update Protocols and Constants

**Files to modify:**
- `src/dqx/common.py` - Update Analyzer protocol
- `src/dqx/analyzer.py` - Add batch size constant

**Changes:**

1. **Update Analyzer Protocol** (`src/dqx/common.py`):
```python
@runtime_checkable
class Analyzer(Protocol):
    """
    Protocol for data analysis engines that process SQL data sources.
    """

    def analyze(
        self,
        ds: SqlDataSource,
        metrics: Sequence[MetricSpec],
        key: ResultKey,
    ) -> AnalysisReport:
        """Analyze single date - existing method."""
        ...

    def analyze_batch(
        self,
        ds: SqlDataSource,
        metrics_by_key: dict[ResultKey, Sequence[MetricSpec]],
    ) -> AnalysisReport:
        """
        Analyze multiple dates with different metrics in a single SQL query.

        Args:
            ds: The data source to analyze
            metrics_by_key: Dictionary mapping ResultKeys to their metrics

        Returns:
            AnalysisReport containing all computed metrics

        Raises:
            DQXError: If batch is empty or SQL execution fails
        """
        ...
```

2. **Add batch constant** (`src/dqx/analyzer.py`):
```python
# Add at top of file after imports
DEFAULT_BATCH_SIZE = 100  # Maximum dates per SQL query
```

**Tests to write:**
- `tests/test_analyzer.py`: Add test to verify protocol implementation
- Verify existing tests still pass

### Task Group 2: Implement Dialect Support

**Files to modify:**
- `src/dqx/dialect.py` - Add build_batch_cte_query method

**Changes:**

1. **Update Dialect Protocol** (`src/dqx/dialect.py`):
```python
@runtime_checkable
class Dialect(Protocol):
    # ... existing methods ...

    def build_batch_cte_query(
        self,
        cte_data: list[dict[str, Any]]
    ) -> str:
        """Build a batch CTE query for multiple dates.

        Args:
            cte_data: List of dicts with keys:
                - key: ResultKey containing the date
                - cte_sql: CTE SQL for this date
                - expressions: List of SQL expressions (e.g., "CAST(SUM(x) AS DOUBLE) AS 'x_1'")
                - ops: List of SqlOp objects (for reference)

        Returns:
            Complete SQL query with CTEs and UNION ALL

        Example output:
            WITH
              source_2024_01_01 AS (...),
              metrics_2024_01_01 AS (SELECT ... FROM source_2024_01_01)
            SELECT '2024-01-01' as date, 'x_1' as symbol, x_1 as value FROM metrics_2024_01_01
            UNION ALL
            SELECT '2024-01-01' as date, 'x_2' as symbol, x_2 as value FROM metrics_2024_01_01
        """
        ...
```

2. **Implement DuckDBDialect.build_batch_cte_query**:
```python
def build_batch_cte_query(self, cte_data: list[dict[str, Any]]) -> str:
    """Build batch query with date-suffixed CTEs and unpivot."""
    if not cte_data:
        raise ValueError("No CTE data provided")

    cte_parts = []
    unpivot_parts = []

    for data in cte_data:
        key = data['key']
        cte_sql = data['cte_sql']
        expressions = data['expressions']

        # Format date for CTE names (yyyy_mm_dd)
        date_suffix = key.yyyy_mm_dd.strftime('%Y_%m_%d')
        source_cte = f"source_{date_suffix}"
        metrics_cte = f"metrics_{date_suffix}"

        # Add source CTE
        cte_parts.append(f"{source_cte} AS ({cte_sql})")

        # Build metrics CTE with all expressions if any exist
        if expressions:
            # Join expressions with comma for single SELECT
            metrics_select = ", ".join(expressions)
            cte_parts.append(
                f"{metrics_cte} AS (SELECT {metrics_select} FROM {source_cte})"
            )

            # Create unpivot SELECT statements
            date_str = key.yyyy_mm_dd.isoformat()
            for expr in expressions:
                # Extract symbol from expression
                # Format: "CAST(SUM(revenue) AS DOUBLE) AS 'x_1'"
                parts = expr.split(" AS ")
                if len(parts) == 2:
                    symbol = parts[1].strip().strip("'")
                    unpivot_parts.append(
                        f"SELECT '{date_str}' as date, '{symbol}' as symbol, "
                        f"{symbol} as value FROM {metrics_cte}"
                    )

    # Build final query
    if not unpivot_parts:
        raise ValueError("No metrics to compute")

    cte_clause = "WITH\n  " + ",\n  ".join(cte_parts)
    union_clause = "\n".join(f"{'UNION ALL' if i > 0 else ''}\n{part}"
                            for i, part in enumerate(unpivot_parts))

    return f"{cte_clause}\n{union_clause}"
```

**Tests to write:**
- `tests/test_dialect.py`:
  - `test_build_batch_cte_query_single_date`
  - `test_build_batch_cte_query_multiple_dates`
  - `test_build_batch_cte_query_empty`
  - `test_build_batch_cte_query_no_expressions`

### Task Group 3: Implement Batch Analysis Function

**Files to modify:**
- `src/dqx/analyzer.py` - Add analyze_batch_sql_ops function

**Changes:**

1. **Add analyze_batch_sql_ops function**:
```python
def analyze_batch_sql_ops(
    ds: T,
    ops_by_key: dict[ResultKey, list[SqlOp]]
) -> None:
    """Analyze SQL ops for multiple dates in one query.

    Args:
        ds: Data source
        ops_by_key: Dict mapping ResultKey to list of SqlOps

    Raises:
        DQXError: If SQL execution fails
    """
    if not ops_by_key:
        return

    # Dedupe ops per key while preserving order
    distinct_ops_by_key: dict[ResultKey, list[SqlOp]] = {}
    for key, ops in ops_by_key.items():
        seen = set()
        distinct_ops = []
        for op in ops:
            if op not in seen:
                seen.add(op)
                distinct_ops.append(op)
        distinct_ops_by_key[key] = distinct_ops

    # Get dialect and build batch query
    dialect_instance = get_dialect(ds.dialect)

    # Build CTE data for each key
    cte_data = []
    for key, ops in distinct_ops_by_key.items():
        cte_sql = ds.cte(key.yyyy_mm_dd)
        expressions = [dialect_instance.translate_sql_op(op) for op in ops]
        cte_data.append({
            'key': key,
            'cte_sql': cte_sql,
            'expressions': expressions,
            'ops': ops
        })

    # Generate and execute SQL
    sql = dialect_instance.build_batch_cte_query(cte_data)

    # Format SQL for readability
    sql = sqlparse.format(
        sql,
        reindent=True,
        keyword_case="upper",
        identifier_case="lower",
        indent_width=2,
        wrap_after=120,
        comma_first=False,
    )

    logger.debug(f"Batch SQL Query:\n{sql}")

    # Execute query - will raise DQXError on failure
    result: dict[str, np.ndarray] = ds.query(sql, datetime.date.today()).fetchnumpy()

    # Parse results - expecting columns: date, symbol, value
    date_col = result['date']
    symbol_col = result['symbol']
    value_col = result['value']

    # Build lookup map
    value_map: dict[tuple[str, str], float] = {}
    for i in range(len(date_col)):
        date_str = date_col[i]
        symbol = symbol_col[i]
        value = value_col[i]
        value_map[(date_str, symbol)] = value

    # Assign values to all ops (including duplicates)
    for key, ops in ops_by_key.items():
        date_str = key.yyyy_mm_dd.isoformat()
        for op in ops:
            # Find the corresponding distinct op to get sql_col
            for distinct_op in distinct_ops_by_key[key]:
                if op == distinct_op:
                    value = value_map.get((date_str, distinct_op.sql_col))
                    if value is not None:
                        op.assign(value)
                    break
```

**Tests to write:**
- Create mock data source and ops for testing
- Test deduplication logic
- Test value assignment

### Task Group 4: Implement Analyzer.analyze_batch

**Files to modify:**
- `src/dqx/analyzer.py` - Add analyze_batch method

**Changes:**

1. **Add analyze_batch method**:
```python
def analyze_batch(
    self,
    ds: SqlDataSource,
    metrics_by_key: dict[ResultKey, Sequence[MetricSpec]],
) -> AnalysisReport:
    """Analyze multiple dates with different metrics in batch.

    Processes dates in batches of DEFAULT_BATCH_SIZE to avoid
    SQL query length limits.

    Args:
        ds: Data source to analyze
        metrics_by_key: Dict mapping ResultKeys to metrics

    Returns:
        AnalysisReport with all computed metrics

    Raises:
        DQXError: If no metrics provided or SQL execution fails
    """
    logger.info(f"Analyzing batch of {len(metrics_by_key)} keys...")
    self._setup_duckdb()

    if not metrics_by_key:
        raise DQXError("No metrics provided for batch analysis!")

    # Process in batches if needed
    all_reports = []
    items = list(metrics_by_key.items())

    for i in range(0, len(items), DEFAULT_BATCH_SIZE):
        batch = dict(items[i:i + DEFAULT_BATCH_SIZE])
        report = self._analyze_batch_internal(ds, batch)
        all_reports.append(report)

    # Merge all batch reports
    final_report = AnalysisReport()
    for report in all_reports:
        final_report = final_report.merge(report)

    self._report = self._report.merge(final_report)
    return self._report

def _analyze_batch_internal(
    self,
    ds: SqlDataSource,
    metrics_by_key: dict[ResultKey, Sequence[MetricSpec]],
) -> AnalysisReport:
    """Process a single batch of dates.

    Args:
        ds: Data source
        metrics_by_key: Batch of dates to process

    Returns:
        AnalysisReport for this batch
    """
    # Collect all ops with their associated keys
    ops_by_key: dict[ResultKey, list[SqlOp]] = {}

    for key, metrics in metrics_by_key.items():
        all_ops = list(itertools.chain.from_iterable(m.analyzers for m in metrics))
        sql_ops = [op for op in all_ops if isinstance(op, SqlOp)]
        if sql_ops:
            ops_by_key[key] = sql_ops

    if not ops_by_key:
        return AnalysisReport()

    # Batch analyze SQL ops
    analyze_batch_sql_ops(ds, ops_by_key)

    # Build report
    report_data = {}
    for key, metrics in metrics_by_key.items():
        for metric in metrics:
            report_data[(metric, key)] = models.Metric.build(metric, key)

    return AnalysisReport(data=report_data)
```

**Tests to write:**
- `tests/test_analyzer.py`:
  - `test_analyze_batch_single_date`
  - `test_analyze_batch_multiple_dates`
  - `test_analyze_batch_empty`
  - `test_analyze_batch_large_batch` (>100 dates)
  - `test_analyze_batch_sql_failure`

### Task Group 5: Integration Tests

**Files to create:**
- `tests/test_analyzer_batch_integration.py`

**Tests to implement:**

1. **Test with real data**:
```python
def test_batch_analyze_with_real_data():
    """Test batch analysis with actual Arrow data."""
    # Create test data
    data = pa.table({
        'yyyy_mm_dd': [date(2024, 1, 1)] * 5 + [date(2024, 1, 2)] * 5,
        'revenue': [100, 200, 300, 400, 500] * 2,
        'price': [10, 20, 30, 40, 50] * 2,
        'user_id': [1, 2, None, 4, 5] * 2
    })

    ds = ArrowDataSource("test", data)
    analyzer = Analyzer()

    key1 = ResultKey(date(2024, 1, 1), {})
    key2 = ResultKey(date(2024, 1, 2), {})

    report = analyzer.analyze_batch(ds, {
        key1: [Sum("revenue"), Average("price"), NullCount("user_id")],
        key2: [Sum("revenue"), Maximum("price")]
    })

    # Verify values
    assert report[(Sum("revenue"), key1)].value == 1500.0
    assert report[(Average("price"), key1)].value == 30.0
    assert report[(NullCount("user_id"), key1)].value == 1.0
    assert report[(Sum("revenue"), key2)].value == 1500.0
    assert report[(Maximum("price"), key2)].value == 50.0
```

2. **Test with lag operations**
3. **Test with tags**
4. **Test persistence to MetricDB**
5. **Performance comparison test**

### Task Group 6: Documentation and Examples

**Files to create/update:**
- `examples/batch_analyzer_demo.py`
- Update `README.md` with batch analysis example

**Example demo**:
```python
"""Demo of batch analysis for time series data."""
import datetime
from dqx import Analyzer, Sum, Average, ResultKey
from dqx.datasource import ArrowDataSource

# Create sample data
dates = [datetime.date(2024, 1, i) for i in range(1, 8)]
data = create_time_series_data(dates)
ds = ArrowDataSource("sales", data)

# Analyze week of data in one query
analyzer = Analyzer()
metrics_by_key = {
    ResultKey(date, {}): [Sum("revenue"), Average("price")]
    for date in dates
}

report = analyzer.analyze_batch(ds, metrics_by_key)

# Display results
for date in dates:
    key = ResultKey(date, {})
    revenue = report[(Sum("revenue"), key)].value
    avg_price = report[(Average("price"), key)].value
    print(f"{date}: Revenue=${revenue:,.2f}, Avg Price=${avg_price:.2f}")
```

### Task Group 7: Final Verification

1. Run all tests with coverage:
```bash
uv run pytest tests/test_analyzer.py tests/test_dialect.py tests/test_analyzer_batch_integration.py -v --cov=dqx.analyzer --cov=dqx.dialect
```

2. Run pre-commit checks:
```bash
./bin/run-hooks.sh
```

3. Verify mypy passes:
```bash
uv run mypy src/dqx/analyzer.py src/dqx/dialect.py
```

4. Update memory bank files if needed

## Success Criteria

1. All existing tests pass (maintain 100% coverage)
2. Batch analysis produces identical results to sequential analysis
3. Performance improvement of at least 50% for multi-date analysis
4. Clean SQL output with proper formatting
5. Clear error messages on failures

## Notes

- Symbol names (x_1, x_2, etc.) remain unchanged and sequential across all dates
- The existing `AnalysisReport` and `AnalysisReport.merge()` handle combining results
- SQL failures should propagate as DQXError (fail fast)
- No backward compatibility required - this is a new feature
