# Batch SQL Optimization Implementation Plan v2

## Overview

This plan implements a complete replacement of the batch SQL query generation in DQX's dialect layer for DuckDB. The optimization uses DuckDB's MAP feature to reduce the number of rows returned from N*M (dates × metrics) to just N (dates), significantly improving performance for large batch operations.

**IMPORTANT: NO BACKWARD COMPATIBILITY** - This is a breaking change that replaces the existing implementation entirely.

## Current Implementation

The current `build_batch_cte_query` method generates SQL that:
1. Creates separate CTEs for each date with suffixes (source_YYYY_MM_DD_i, metrics_YYYY_MM_DD_i)
2. Uses UNION ALL to unpivot results into (date, symbol, value) format
3. Returns N*M rows where N = number of dates, M = number of metrics

## New Implementation

Replace the entire approach with DuckDB's MAP feature to return all metrics as a single MAP per date:

```sql
WITH
  source_2024_01_01_0 AS (SELECT * FROM sales WHERE date = '2024-01-01'),
  metrics_2024_01_01_0 AS (SELECT SUM(revenue) AS 'x_1', AVG(price) AS 'x_2' FROM source_2024_01_01_0)
SELECT '2024-01-01' as date, MAP {'x_1': "x_1", 'x_2': "x_2"} as values FROM metrics_2024_01_01_0
```

This returns only N rows with all metric values in a MAP column.

## Design Decisions

1. **Complete Replacement**: Replace the existing `build_batch_cte_query` method entirely
2. **KISS Principle**: Simple, direct implementation without feature flags or compatibility modes
3. **Helper Methods**: Extract common logic into private helper methods to follow DRY principle
4. **Focused Testing**: Write tests only for the new MAP-based approach

## Implementation Tasks

### Task 1: Write Failing Tests for MAP-based Implementation

**File**: `tests/test_dialect_batch_optimization.py`

```python
"""Test batch SQL optimization using MAP feature."""

import datetime
from typing import Any

import pytest

from dqx.common import ResultKey
from dqx.dialect import DuckDBDialect
from dqx.models import BatchCTEData
from dqx.ops import Average, Maximum, Minimum, NullCount, Sum


class TestBatchCTEQueryMap:
    """Test MAP-based batch CTE queries."""

    def test_build_batch_cte_query_single_date(self) -> None:
        """Test MAP query with single date."""
        dialect = DuckDBDialect()

        # Create test data
        key = ResultKey(datetime.date(2024, 1, 1), {})
        ops = [
            Sum("revenue"),
            Average("price"),
        ]

        cte_data = [
            BatchCTEData(
                key=key,
                cte_sql="SELECT * FROM sales WHERE yyyy_mm_dd = '2024-01-01'",
                ops=ops
            )
        ]

        # This method will be updated to use MAP
        sql = dialect.build_batch_cte_query(cte_data)

        # Verify MAP structure
        assert "MAP {" in sql
        assert "'x_1': \"x_1\"" in sql  # Sum(revenue)
        assert "'x_2': \"x_2\"" in sql  # Average(price)
        assert "as values FROM" in sql
        assert sql.count("SELECT '2024-01-01' as date") == 1  # Only one row

    def test_build_batch_cte_query_multiple_dates(self) -> None:
        """Test MAP query with multiple dates."""
        dialect = DuckDBDialect()

        # Create test data for 3 dates
        cte_data = []
        for day in [1, 2, 3]:
            key = ResultKey(datetime.date(2024, 1, day), {})
            ops = [
                Sum("revenue"),
                Average("price"),
                NullCount("customer_id"),
            ]

            cte_data.append(
                BatchCTEData(
                    key=key,
                    cte_sql=f"SELECT * FROM sales WHERE yyyy_mm_dd = '2024-01-0{day}'",
                    ops=ops
                )
            )

        sql = dialect.build_batch_cte_query(cte_data)

        # Should have exactly 3 SELECT statements (one per date)
        assert sql.count("SELECT") == 6  # 3 metrics CTEs + 3 final SELECTs
        assert sql.count("UNION ALL") == 2  # 3 results joined by 2 UNION ALLs
        assert sql.count("MAP {") == 3  # One MAP per date

    def test_build_batch_cte_query_preserves_column_aliases(self) -> None:
        """Test that MAP keys use the correct sql_col aliases."""
        dialect = DuckDBDialect()

        key = ResultKey(datetime.date(2024, 1, 1), {})
        ops = [
            Sum("total_sales"),
            Minimum("min_price"),
            Maximum("max_price"),
        ]

        # Get the actual sql_col values
        expected_keys = [op.sql_col for op in ops]

        cte_data = [
            BatchCTEData(
                key=key,
                cte_sql="SELECT * FROM orders",
                ops=ops
            )
        ]

        sql = dialect.build_batch_cte_query(cte_data)

        # Verify each sql_col appears as a MAP key
        for sql_col in expected_keys:
            assert f"'{sql_col}': \"{sql_col}\"" in sql

    def test_build_batch_cte_query_empty_data(self) -> None:
        """Test error handling for empty CTE data."""
        dialect = DuckDBDialect()

        with pytest.raises(ValueError, match="No CTE data provided"):
            dialect.build_batch_cte_query([])

    def test_build_batch_cte_query_no_ops(self) -> None:
        """Test error handling when no ops provided."""
        dialect = DuckDBDialect()

        cte_data = [
            BatchCTEData(
                key=ResultKey(datetime.date(2024, 1, 1), {}),
                cte_sql="SELECT * FROM sales",
                ops=[]  # No ops
            )
        ]

        with pytest.raises(ValueError, match="No metrics to compute"):
            dialect.build_batch_cte_query(cte_data)
```

**Commit**:
```bash
git add tests/test_dialect_batch_optimization.py
git commit -m "test: add failing tests for MAP-based batch query implementation"
```

### Task 2: Implement Helper Methods

**File**: `src/dqx/dialect.py`

Add these helper methods to the `DuckDBDialect` class:

```python
def _build_cte_parts(self, cte_data: list["BatchCTEData"]) -> tuple[list[str], list[tuple[str, list["SqlOp"]]]]:
    """Build CTE parts for batch query.

    Args:
        cte_data: List of BatchCTEData objects

    Returns:
        Tuple of (cte_parts, metrics_info)
        where metrics_info contains (metrics_cte_name, ops) for each CTE with ops

    Raises:
        ValueError: If no CTE data provided
    """
    if not cte_data:
        raise ValueError("No CTE data provided")

    cte_parts = []
    metrics_info: list[tuple[str, list["SqlOp"]]] = []

    for i, data in enumerate(cte_data):
        # Format date for CTE names (yyyy_mm_dd)
        # Include index to ensure unique names even for same date with different tags
        date_suffix = data.key.yyyy_mm_dd.strftime("%Y_%m_%d")
        source_cte = f"source_{date_suffix}_{i}"
        metrics_cte = f"metrics_{date_suffix}_{i}"

        # Add source CTE
        cte_parts.append(f"{source_cte} AS ({data.cte_sql})")

        # Build metrics CTE with all expressions if ops exist
        if data.ops:
            # Translate ops to expressions
            expressions = [self.translate_sql_op(op) for op in data.ops]
            metrics_select = ", ".join(expressions)
            cte_parts.append(f"{metrics_cte} AS (SELECT {metrics_select} FROM {source_cte})")

            # Store metrics info for later use
            metrics_info.append((metrics_cte, list(data.ops)))

    return cte_parts, metrics_info

def _validate_metrics(self, metrics_info: list[tuple[str, list["SqlOp"]]]) -> None:
    """Validate that metrics exist to compute.

    Args:
        metrics_info: List of (metrics_cte_name, ops) tuples

    Raises:
        ValueError: If no metrics to compute
    """
    if not metrics_info:
        raise ValueError("No metrics to compute")
```

**Commit**:
```bash
git add -p src/dqx/dialect.py
git commit -m "refactor: add helper methods for batch CTE query building"
```

### Task 3: Replace build_batch_cte_query Implementation

**File**: `src/dqx/dialect.py`

Replace the existing `build_batch_cte_query` method entirely:

```python
def build_batch_cte_query(self, cte_data: list["BatchCTEData"]) -> str:
    """Build batch CTE query using MAP for DuckDB.

    This method uses DuckDB's MAP feature to return all metrics as a single
    MAP per date, reducing the result from N*M rows to just N rows.

    Args:
        cte_data: List of BatchCTEData objects containing:
            - key: ResultKey with the date
            - cte_sql: CTE SQL for this date
            - ops: List of SqlOp objects to translate

    Returns:
        Complete SQL query with CTEs and MAP-based results

    Example output:
        WITH
          source_2024_01_01_0 AS (...),
          metrics_2024_01_01_0 AS (SELECT ... FROM source_2024_01_01_0)
        SELECT '2024-01-01' as date, MAP {'x_1': "x_1", 'x_2': "x_2"} as values
        FROM metrics_2024_01_01_0
    """
    # Use helper to build CTE parts
    cte_parts, metrics_info = self._build_cte_parts(cte_data)

    # Validate metrics
    self._validate_metrics(metrics_info)

    # Build MAP-based SELECT statements
    map_selects = []
    for i, (data, (metrics_cte, ops)) in enumerate(zip(cte_data, metrics_info)):
        date_str = data.key.yyyy_mm_dd.isoformat()

        # Build MAP entries
        map_entries = [f"'{op.sql_col}': \"{op.sql_col}\"" for op in ops]
        map_expr = "MAP {" + ", ".join(map_entries) + "}"

        map_selects.append(
            f"SELECT '{date_str}' as date, {map_expr} as values FROM {metrics_cte}"
        )

    # Build final query
    cte_clause = "WITH\n  " + ",\n  ".join(cte_parts)
    union_clause = "\n".join(
        f"{'UNION ALL' if i > 0 else ''}\n{select}"
        for i, select in enumerate(map_selects)
    )

    return f"{cte_clause}\n{union_clause}"
```

**Commit**:
```bash
git add -p src/dqx/dialect.py
git commit -m "feat: replace build_batch_cte_query with MAP-based implementation"
```

### Task 4: Add Integration Tests

**File**: `tests/test_dialect_batch_optimization.py`

Add integration tests with actual DuckDB execution:

```python
def test_map_query_with_duckdb_execution() -> None:
    """Test MAP query with actual DuckDB execution."""
    import duckdb
    from dqx import ops
    from dqx.common import ResultKey
    from dqx.dialect import DuckDBDialect
    from dqx.models import BatchCTEData

    # Create test database
    conn = duckdb.connect(":memory:")

    # Create test data for multiple dates
    conn.execute("""
        CREATE TABLE sales AS
        SELECT
            '2024-01-01'::DATE as yyyy_mm_dd,
            100.0 as revenue,
            25.0 as price,
            1 as customer_id
        UNION ALL
        SELECT '2024-01-01'::DATE, 200.0, 30.0, 2
        UNION ALL
        SELECT '2024-01-01'::DATE, 150.0, NULL, NULL
        UNION ALL
        SELECT '2024-01-02'::DATE, 300.0, 40.0, 3
        UNION ALL
        SELECT '2024-01-02'::DATE, 250.0, 35.0, NULL
        UNION ALL
        SELECT '2024-01-03'::DATE, 400.0, 50.0, 4
    """)

    dialect = DuckDBDialect()

    # Create batch data
    cte_data = []
    for day in [1, 2, 3]:
        date = datetime.date(2024, 1, day)
        key = ResultKey(date, {})

        ops_list = [
            ops.Sum("revenue"),
            ops.Average("price"),
            ops.NullCount("customer_id"),
        ]

        cte_data.append(
            BatchCTEData(
                key=key,
                cte_sql=f"SELECT * FROM sales WHERE yyyy_mm_dd = '{date.isoformat()}'",
                ops=ops_list
            )
        )

    # Generate MAP query
    sql = dialect.build_batch_cte_query(cte_data)

    # Execute query
    result = conn.execute(sql).fetchall()

    # Verify results
    assert len(result) == 3  # One row per date

    # Check first date results
    date1, values1 = result[0]
    assert date1 == "2024-01-01"
    assert isinstance(values1, dict)  # DuckDB returns MAP as dict
    # Sum of revenue for 2024-01-01: 100 + 200 + 150 = 450
    assert values1[cte_data[0].ops[0].sql_col] == 450.0
    # Average of price for 2024-01-01: (25 + 30) / 2 = 27.5 (NULL excluded)
    assert values1[cte_data[0].ops[1].sql_col] == 27.5
    # Null count for customer_id: 1
    assert values1[cte_data[0].ops[2].sql_col] == 1.0

    # Check second date results
    date2, values2 = result[1]
    assert date2 == "2024-01-02"
    # Sum of revenue for 2024-01-02: 300 + 250 = 550
    assert values2[cte_data[1].ops[0].sql_col] == 550.0

    conn.close()
```

**Commit**:
```bash
git add -p tests/test_dialect_batch_optimization.py
git commit -m "test: add integration tests for MAP query with DuckDB"
```

### Task 5: Update Analyzer to Use MAP Results

**File**: `src/dqx/analyzer.py`

Update the `analyze_batch_sql_ops` function to handle MAP results directly:

```python
def analyze_batch_sql_ops(
    ds: T,
    ops_by_key: dict[ResultKey, list[SqlOp]],
    nominal_date: date | None = None,
) -> None:
    """Batch analyze SQL ops across multiple keys.

    Args:
        ds: Data source to analyze
        ops_by_key: Dict mapping ResultKey to list of SqlOp objects
        nominal_date: Nominal date for analysis (optional)
    """
    if not ops_by_key:
        return

    # Get dialect instance
    dialect_instance = get_dialect(ds.dialect)

    # Build CTE data
    cte_data = [
        BatchCTEData(key=key, cte_sql=ds.cte(key.yyyy_mm_dd), ops=ops)
        for key, ops in ops_by_key.items()
    ]

    # Generate SQL and execute
    sql = dialect_instance.build_batch_cte_query(cte_data)

    # Execute query and process MAP results
    result = ds.query(sql).fetchall()

    # Process results
    for (date_str, values_map), (key, ops) in zip(result, ops_by_key.items()):
        for op in ops:
            if op.sql_col in values_map:
                op.assign(float(values_map[op.sql_col]))
```

**Commit**:
```bash
git add -p src/dqx/analyzer.py
git commit -m "refactor: simplify analyzer to use MAP results directly"
```

### Task 6: Add Analyzer Tests

**File**: `tests/test_analyzer_batch_optimization.py`

```python
"""Test analyzer integration with batch SQL optimization."""

import datetime
from unittest.mock import Mock

import pytest

from dqx import ops
from dqx.analyzer import analyze_batch_sql_ops
from dqx.common import ResultKey


class TestAnalyzerBatchOptimization:
    """Test analyzer batch processing with MAP results."""

    def test_analyze_batch_sql_ops_with_map_results(self) -> None:
        """Test batch analysis with MAP results."""
        # Create mock data source
        ds = Mock()
        ds.dialect = "duckdb"
        ds.cte = Mock(side_effect=lambda date: f"SELECT * FROM sales WHERE date = '{date}'")

        # Create ops
        ops_by_key = {
            ResultKey(datetime.date(2024, 1, 1), {}): [
                ops.Sum("revenue"),
                ops.Average("price"),
            ],
            ResultKey(datetime.date(2024, 1, 2), {}): [
                ops.Sum("revenue"),
                ops.Average("price"),
            ],
        }

        # Mock query results - MAP format
        mock_results = [
            ("2024-01-01", {"x_1_sum(revenue)": 1000.0, "x_2_average(price)": 25.0}),
            ("2024-01-02", {"x_1_sum(revenue)": 1500.0, "x_2_average(price)": 30.0}),
        ]
        ds.query.return_value.fetchall.return_value = mock_results

        # Execute
        analyze_batch_sql_ops(ds, ops_by_key)

        # Verify values assigned
        key1_ops = ops_by_key[ResultKey(datetime.date(2024, 1, 1), {})]
        assert key1_ops[0].value() == 1000.0  # Sum
        assert key1_ops[1].value() == 25.0    # Average

        key2_ops = ops_by_key[ResultKey(datetime.date(2024, 1, 2), {})]
        assert key2_ops[0].value() == 1500.0  # Sum
        assert key2_ops[1].value() == 30.0    # Average

    def test_analyze_batch_sql_ops_empty_input(self) -> None:
        """Test batch analysis with empty input."""
        ds = Mock()
        analyze_batch_sql_ops(ds, {})

        # Should not call query
        ds.query.assert_not_called()

    def test_analyze_batch_sql_ops_missing_values(self) -> None:
        """Test batch analysis handles missing values gracefully."""
        ds = Mock()
        ds.dialect = "duckdb"
        ds.cte = Mock(return_value="SELECT * FROM sales")

        # Create ops
        op1 = ops.Sum("revenue")
        op2 = ops.Average("price")
        ops_by_key = {
            ResultKey(datetime.date(2024, 1, 1), {}): [op1, op2],
        }

        # Mock results with missing value
        mock_results = [
            ("2024-01-01", {"x_1_sum(revenue)": 1000.0}),  # Missing average
        ]
        ds.query.return_value.fetchall.return_value = mock_results

        # Execute
        analyze_batch_sql_ops(ds, ops_by_key)

        # Verify only available value assigned
        assert op1.value() == 1000.0
        assert op2.value() is None  # Should remain unassigned
```

**Commit**:
```bash
git add tests/test_analyzer_batch_optimization.py
git commit -m "test: add analyzer tests for MAP result processing"
```

## Summary

This implementation plan provides a clean, KISS-compliant solution for optimizing batch SQL queries using DuckDB's MAP feature:

1. **No Backward Compatibility**: Completely replaces the existing `build_batch_cte_query` method
2. **Simple Architecture**: Direct MAP implementation without feature flags or compatibility modes
3. **Clean Code**: Helper methods `_build_cte_parts` and `_validate_metrics` reduce duplication
4. **Focused Testing**: TDD approach with tests only for the new MAP-based implementation
5. **Direct Integration**: Analyzer simply processes MAP results without any feature detection

The implementation follows DQX coding standards and KISS principles:
- Single, clear approach without optional paths
- Type hints throughout
- Comprehensive docstrings
- Proper error handling
- Clean commit messages following conventional commits

## Next Steps

After implementation:
1. Run all tests to ensure nothing is broken
2. Update any existing tests that rely on the old unpivot format
3. Update documentation if needed
