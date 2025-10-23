# Batch SQL Optimization Implementation Plan v1

## Overview

This plan implements an optimization for batch SQL query generation in DQX's dialect layer, specifically for DuckDB. The optimization uses DuckDB's MAP feature to reduce the number of rows returned from N*M (dates × metrics) to just N (dates), significantly improving performance for large batch operations.

## Current Implementation

The current `build_batch_cte_query` method generates SQL that:
1. Creates separate CTEs for each date with suffixes (source_YYYY_MM_DD_i, metrics_YYYY_MM_DD_i)
2. Uses UNION ALL to unpivot results into (date, symbol, value) format
3. Returns N*M rows where N = number of dates, M = number of metrics

Example current output:
```sql
WITH
  source_2024_01_01_0 AS (SELECT * FROM sales WHERE date = '2024-01-01'),
  metrics_2024_01_01_0 AS (SELECT SUM(revenue) AS 'x_1', AVG(price) AS 'x_2' FROM source_2024_01_01_0)
SELECT '2024-01-01' as date, 'x_1' as symbol, "x_1" as value FROM metrics_2024_01_01_0
UNION ALL
SELECT '2024-01-01' as date, 'x_2' as symbol, "x_2" as value FROM metrics_2024_01_01_0
```

## Proposed Optimization

Use DuckDB's MAP feature to return all metrics as a single MAP per date:

```sql
WITH
  source_2024_01_01_0 AS (SELECT * FROM sales WHERE date = '2024-01-01'),
  metrics_2024_01_01_0 AS (SELECT SUM(revenue) AS 'x_1', AVG(price) AS 'x_2' FROM source_2024_01_01_0)
SELECT '2024-01-01' as date, MAP {'x_1': "x_1", 'x_2': "x_2"} as values FROM metrics_2024_01_01_0
```

This returns only N rows with all metric values in a MAP column.

## Design Decisions

1. **New Method**: Add `build_batch_cte_query_map` to `DuckDBDialect` instead of modifying the existing method to maintain backward compatibility
2. **Feature Detection**: Add a `use_map_optimization` parameter to control which approach to use
3. **Helper Methods**: Extract common logic into private helper methods to follow DRY principle
4. **Testing**: Write comprehensive tests using TDD approach

## Implementation Tasks

### Task 1: Write Failing Tests for MAP-based Optimization

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


class TestBatchCTEQueryMapOptimization:
    """Test MAP-based optimization for batch CTE queries."""

    def test_build_batch_cte_query_map_single_date(self) -> None:
        """Test MAP optimization with single date."""
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

        # This method doesn't exist yet - will fail
        sql = dialect.build_batch_cte_query_map(cte_data)

        # Verify MAP structure
        assert "MAP {" in sql
        assert "'x_1': \"x_1\"" in sql  # Sum(revenue)
        assert "'x_2': \"x_2\"" in sql  # Average(price)
        assert "as values FROM" in sql
        assert sql.count("SELECT '2024-01-01' as date") == 1  # Only one row

    def test_build_batch_cte_query_map_multiple_dates(self) -> None:
        """Test MAP optimization with multiple dates."""
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

        sql = dialect.build_batch_cte_query_map(cte_data)

        # Should have exactly 3 SELECT statements (one per date)
        assert sql.count("SELECT") == 6  # 3 metrics CTEs + 3 final SELECTs
        assert sql.count("UNION ALL") == 2  # 3 results joined by 2 UNION ALLs
        assert sql.count("MAP {") == 3  # One MAP per date

    def test_build_batch_cte_query_map_preserves_column_aliases(self) -> None:
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

        sql = dialect.build_batch_cte_query_map(cte_data)

        # Verify each sql_col appears as a MAP key
        for sql_col in expected_keys:
            assert f"'{sql_col}': \"{sql_col}\"" in sql

    def test_build_batch_cte_query_map_empty_data(self) -> None:
        """Test error handling for empty CTE data."""
        dialect = DuckDBDialect()

        with pytest.raises(ValueError, match="No CTE data provided"):
            dialect.build_batch_cte_query_map([])

    def test_build_batch_cte_query_map_no_ops(self) -> None:
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
            dialect.build_batch_cte_query_map(cte_data)

    def test_build_batch_cte_query_with_use_map_parameter(self) -> None:
        """Test backward compatibility with use_map parameter."""
        dialect = DuckDBDialect()

        key = ResultKey(datetime.date(2024, 1, 1), {})
        ops = [Sum("revenue")]

        cte_data = [
            BatchCTEData(
                key=key,
                cte_sql="SELECT * FROM sales",
                ops=ops
            )
        ]

        # Default behavior (use_map=False) - uses existing unpivot approach
        sql_unpivot = dialect.build_batch_cte_query(cte_data, use_map=False)
        assert "UNION ALL" in sql_unpivot
        assert "MAP {" not in sql_unpivot

        # New behavior (use_map=True) - uses MAP optimization
        sql_map = dialect.build_batch_cte_query(cte_data, use_map=True)
        assert "MAP {" in sql_map
        assert sql_map.count("SELECT") < sql_unpivot.count("SELECT")
```

**Commit**:
```bash
git add tests/test_dialect_batch_optimization.py
git commit -m "test: add failing tests for MAP-based batch query optimization"
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

### Task 3: Implement MAP-based Optimization

**File**: `src/dqx/dialect.py`

Add the new MAP-based method and update the existing method:

```python
def build_batch_cte_query_map(self, cte_data: list["BatchCTEData"]) -> str:
    """Build batch CTE query using MAP optimization for DuckDB.

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

def build_batch_cte_query(self, cte_data: list["BatchCTEData"], use_map: bool = False) -> str:
    """Build batch CTE query for multiple dates.

    Args:
        cte_data: List of BatchCTEData objects
        use_map: If True, use MAP optimization (DuckDB only).
                If False, use traditional unpivot approach.

    Returns:
        Complete SQL query
    """
    if use_map:
        return self.build_batch_cte_query_map(cte_data)

    # Original implementation (refactored to use helpers)
    cte_parts, metrics_info = self._build_cte_parts(cte_data)
    self._validate_metrics(metrics_info)

    # Create unpivot SELECT statements
    unpivot_parts = []
    for data, (metrics_cte, ops) in zip(cte_data, metrics_info):
        date_str = data.key.yyyy_mm_dd.isoformat()
        for op in ops:
            unpivot_parts.append(
                f"SELECT '{date_str}' as date, '{op.sql_col}' as symbol, "
                f'"{op.sql_col}" as value FROM {metrics_cte}'
            )

    # Build final query
    cte_clause = "WITH\n  " + ",\n  ".join(cte_parts)
    union_clause = "\n".join(
        f"{'UNION ALL' if i > 0 else ''}\n{part}"
        for i, part in enumerate(unpivot_parts)
    )

    return f"{cte_clause}\n{union_clause}"
```

**Commit**:
```bash
git add -p src/dqx/dialect.py
git commit -m "feat: implement MAP-based batch query optimization for DuckDB"
```

### Task 4: Add Integration Tests

**File**: `tests/test_dialect_batch_optimization.py`

Add integration tests with actual DuckDB execution:

```python
def test_map_optimization_with_duckdb_execution() -> None:
    """Test MAP optimization with actual DuckDB execution."""
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
    sql = dialect.build_batch_cte_query(cte_data, use_map=True)

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

def test_map_vs_unpivot_results_equivalence() -> None:
    """Test that MAP and unpivot approaches produce equivalent results."""
    import duckdb
    from dqx import ops
    from dqx.common import ResultKey
    from dqx.dialect import DuckDBDialect
    from dqx.models import BatchCTEData

    # Create test database
    conn = duckdb.connect(":memory:")

    conn.execute("""
        CREATE TABLE test_data AS
        SELECT
            '2024-01-01'::DATE as date,
            100.0 as value1,
            'A' as category
        UNION ALL
        SELECT '2024-01-01'::DATE, 200.0, 'B'
        UNION ALL
        SELECT '2024-01-02'::DATE, 150.0, 'A'
    """)

    dialect = DuckDBDialect()

    # Create test data
    cte_data = []
    for day in [1, 2]:
        date = datetime.date(2024, 1, day)
        key = ResultKey(date, {})

        ops_list = [
            ops.Sum("value1"),
            ops.Average("value1"),
        ]

        cte_data.append(
            BatchCTEData(
                key=key,
                cte_sql=f"SELECT * FROM test_data WHERE date = '{date.isoformat()}'",
                ops=ops_list
            )
        )

    # Get results with unpivot approach
    sql_unpivot = dialect.build_batch_cte_query(cte_data, use_map=False)
    unpivot_results = conn.execute(sql_unpivot).fetchall()

    # Get results with MAP approach
    sql_map = dialect.build_batch_cte_query(cte_data, use_map=True)
    map_results = conn.execute(sql_map).fetchall()

    # Convert unpivot results to dict format for comparison
    unpivot_dict = {}
    for date, symbol, value in unpivot_results:
        if date not in unpivot_dict:
            unpivot_dict[date] = {}
        unpivot_dict[date][symbol] = value

    # Compare results
    assert len(map_results) == 2  # One row per date

    for date, values_map in map_results:
        # Check that all values match
        for symbol, value in values_map.items():
            assert unpivot_dict[date][symbol] == value

    conn.close()
```

**Commit**:
```bash
git add -p tests/test_dialect_batch_optimization.py
git commit -m "test: add integration tests for MAP optimization with DuckDB"
```

### Task 5: Update Analyzer to Use Optimization

**File**: `src/dqx/analyzer.py`

Update the `analyze_batch_sql_ops` function to support the optimization:

```python
def analyze_batch_sql_ops(
    ds: T,
    ops_by_key: dict[ResultKey, list[SqlOp]],
    nominal_date: date | None = None,
    use_map_optimization: bool = True,
) -> None:
    """Batch analyze SQL ops across multiple keys.

    Args:
        ds: Data source to analyze
        ops_by_key: Dict mapping ResultKey to list of SqlOp objects
        nominal_date: Nominal date for analysis (optional)
        use_map_optimization: Use MAP optimization if dialect supports it (default: True)
    """
    if not ops_by_key:
        return

    # Get dialect instance
    dialect_instance = get_dialect(ds.dialect)

    # Check if dialect supports MAP optimization
    supports_map = (
        hasattr(dialect_instance, "build_batch_cte_query") and
        "use_map" in inspect.signature(dialect_instance.build_batch_cte_query).parameters and
        ds.dialect == "duckdb"  # Currently only DuckDB supports MAP
    )

    # Build CTE data
    cte_data = [
        BatchCTEData(key=key, cte_sql=ds.cte(key.yyyy_mm_dd), ops=ops)
        for key, ops in ops_by_key.items()
    ]

    # Generate and execute SQL
    if supports_map and use_map_optimization:
        sql = dialect_instance.build_batch_cte_query(cte_data, use_map=True)

        # Execute query and process MAP results
        result = ds.query(sql).fetchall()

        # Process results
        for (date_str, values_map), (key, ops) in zip(result, ops_by_key.items()):
            for op in ops:
                if op.sql_col in values_map:
                    op.assign(float(values_map[op.sql_col]))
    else:
        # Use original unpivot approach
        sql = dialect_instance.build_batch_cte_query(cte_data)

        # Process unpivot results (existing logic)
        result = ds.query(sql).fetchnumpy()

        # Build lookup: (date, symbol) -> value
        date_col = result["date"]
        symbol_col = result["symbol"]
        value_col = result["value"]

        value_lookup = {}
        for i in range(len(date_col)):
            date_str = date_col[i]
            symbol = symbol_col[i]
            value = value_col[i]
            value_lookup[(date_str, symbol)] = value

        # Assign values to ops
        for key, ops in ops_by_key.items():
            date_str = key.yyyy_mm_dd.isoformat()
            for op in ops:
                lookup_key = (date_str, op.sql_col)
                if lookup_key in value_lookup:
                    op.assign(float(value_lookup[lookup_key]))
```

Also add the import at the top of the file:
```python
import inspect
```

**Commit**:
```bash
git add -p src/dqx/analyzer.py
git commit -m "feat: add MAP optimization support to analyzer batch processing"
```

### Task 6: Add Analyzer Tests

**File**: `tests/test_analyzer_batch_optimization.py`

```python
"""Test analyzer integration with batch SQL optimization."""

import datetime
from unittest.mock import Mock, patch

import numpy as np
import pytest

from dqx import ops
from dqx.analyzer import analyze_batch_sql_ops
from dqx.common import ResultKey


class TestAnalyzerBatchOptimization:
    """Test analyzer batch processing with MAP optimization."""

    def test_analyze_batch_sql_ops_with_map_optimization(self) -> None:
        """Test batch analysis uses MAP optimization for DuckDB."""
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

        # Patch dialect to have expected methods
        with patch("dqx.analyzer.get_dialect") as mock_get_dialect:
            mock_dialect = Mock()
            mock_dialect.build_batch_cte_query = Mock()

            # Make build_batch_cte_query accept use_map parameter
            def build_batch_cte_query_with_map(cte_data, use_map=False):
                if use_map:
                    return "SQL WITH MAP"
                return "SQL WITHOUT MAP"

            mock_dialect.build_batch_cte_query.side_effect = build_batch_cte_query_with_map
            mock_get_dialect.return_value = mock_dialect

            # Execute
            analyze_batch_sql_ops(ds, ops_by_key, use_map_optimization=True)

            # Verify MAP query was used
            mock_dialect.build_batch_cte_query.assert_called_once()
            call_args = mock_dialect.build_batch_cte_query.call_args
            assert call_args[1]["use_map"] is True

    def test_analyze_batch_sql_ops_fallback_to_unpivot(self) -> None:
        """Test batch analysis falls back to unpivot for non-DuckDB."""
        # Create mock data source with PostgreSQL
        ds = Mock()
        ds.dialect = "postgresql"
        ds.cte = Mock(side_effect=lambda date: f"SELECT * FROM sales WHERE date = '{date}'")

        # Create ops
        key1 = ResultKey(datetime.date(2024, 1, 1), {})
        op1 = ops.Sum("revenue")
        op2 = ops.Average("price")

        ops_by_key = {key1: [op1, op2]}

        # Mock query results - unpivot format
        mock_results = {
            "date": np.array(["2024-01-01", "2024-01-01"]),
            "symbol": np.array([op1.sql_col, op2.sql_col]),
            "value": np.array([1000.0, 25.0]),
        }
        ds.query.return_value.fetchnumpy.return_value = mock_results

        # Execute
        analyze_batch_sql_ops(ds, ops_by_key, use_map_optimization=True)

        # Verify values assigned
        assert op1.value() == 1000.0
        assert op2.value() == 25.0

    def test_analyze_batch_sql_ops_disable_optimization(self) -> None:
        """Test that MAP optimization can be disabled."""
        # Create mock data source
        ds = Mock()
        ds.dialect = "duckdb"
        ds.cte = Mock(return_value="SELECT * FROM sales")

        # Create simple ops
        ops_by_key = {
            ResultKey(datetime.date(2024, 1, 1), {}): [ops.NumRows()],
        }

        # Mock unpivot results
        mock_results = {
            "date": np.array(["2024-01-01"]),
            "symbol": np.array(["x_1_num_rows()"]),
            "value": np.array([100.0]),
        }
        ds.query.return_value.fetchnumpy.return_value = mock_results

        # Execute with optimization disabled
        analyze_batch_sql_ops(ds, ops_by_key, use_map_optimization=False)

        # Should use fetchnumpy (unpivot approach)
        ds.query.return_value.fetchnumpy.assert_called_once()
        ds.query.return_value.fetchall.assert_not_called()
```

**Commit**:
```bash
git add tests/test_analyzer_batch_optimization.py
git commit -m "test: add analyzer tests for MAP optimization integration"
```

### Task 7: Performance Benchmarks

**File**: `examples/batch_sql_optimization_benchmark.py`

```python
"""Benchmark script comparing MAP vs unpivot batch SQL approaches."""

import time
from datetime import date, timedelta

import duckdb
import pandas as pd

from dqx import ops
from dqx.common import ResultKey
from dqx.dialect import DuckDBDialect
from dqx.models import BatchCTEData


def create_test_data(conn: duckdb.DuckDBConnection, num_dates: int, rows_per_date: int) -> None:
    """Create test data for benchmarking."""
    print(f"Creating test data: {num_dates} dates, {rows_per_date} rows per date...")

    # Generate dates
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(num_dates)]

    # Create DataFrame
    data = []
    for d in dates:
        for i in range(rows_per_date):
            data.append({
                "date": d,
                "revenue": 100.0 + i * 10,
                "price": 20.0 + i * 2,
                "quantity": i + 1,
                "customer_id": i % 100 if i % 10 != 0 else None,
            })

    df = pd.DataFrame(data)

    # Load into DuckDB
    conn.execute("CREATE TABLE sales AS SELECT * FROM df")
    print(f"Created {len(df)} total rows")


def benchmark_approach(
    conn: duckdb.DuckDBConnection,
    dialect: DuckDBDialect,
    cte_data: list[BatchCTEData],
    use_map: bool,
    name: str,
) -> float:
    """Benchmark a single approach."""
    print(f"\nBenchmarking {name}...")

    # Generate SQL
    start = time.time()
    sql = dialect.build_batch_cte_query(cte_data, use_map=use_map)
    sql_time = time.time() - start

    print(f"  SQL generation: {sql_time:.4f}s")
    print(f"  SQL length: {len(sql)} characters")

    # Execute query
    start = time.time()
    result = conn.execute(sql).fetchall()
    exec_time = time.time() - start

    print(f"  Query execution: {exec_time:.4f}s")
    print(f"  Result rows: {len(result)}")

    return sql_time + exec_time


def main() -> None:
    """Run performance benchmarks."""
    # Test configurations
    configs = [
        (10, 1000, 5),      # 10 dates, 1K rows/date, 5 metrics
        (50, 1000, 10),     # 50 dates, 1K rows/date, 10 metrics
        (100, 5000, 20),    # 100 dates, 5K rows/date, 20 metrics
    ]

    for num_dates, rows_per_date, num_metrics in configs:
        print(f"\n{'='*60}")
        print(f"Configuration: {num_dates} dates, {rows_per_date} rows/date, {num_metrics} metrics")
        print(f"{'='*60}")

        # Create connection and test data
        conn = duckdb.connect(":memory:")
        create_test_data(conn, num_dates, rows_per_date)

        # Create dialect
        dialect = DuckDBDialect()

        # Create ops (varied metrics)
        metric_ops = []
        if num_metrics >= 1:
            metric_ops.append(ops.Sum("revenue"))
        if num_metrics >= 2:
            metric_ops.append(ops.Average("price"))
        if num_metrics >= 3:
            metric_ops.append(ops.Minimum("quantity"))
        if num_metrics >= 4:
            metric_ops.append(ops.Maximum("quantity"))
        if num_metrics >= 5:
            metric_ops.append(ops.NullCount("customer_id"))

        # Add more ops if needed
        for i in range(5, num_metrics):
            metric_ops.append(ops.Sum(f"revenue"))  # Reuse columns

        # Create batch data
        cte_data = []
        for i in range(num_dates):
            d = date(2024, 1, 1) + timedelta(days=i)
            key = ResultKey(d, {})

            cte_data.append(
                BatchCTEData(
                    key=key,
                    cte_sql=f"SELECT * FROM sales WHERE date = '{d}'",
                    ops=metric_ops.copy(),  # Copy to get new instances
                )
            )

        # Benchmark both approaches
        unpivot_time = benchmark_approach(
            conn, dialect, cte_data, use_map=False, name="Unpivot approach"
        )

        map_time = benchmark_approach(
            conn, dialect, cte_data, use_map=True, name="MAP approach"
        )

        # Calculate improvement
        improvement = (unpivot_time - map_time) / unpivot_time * 100
        speedup = unpivot_time / map_time

        print(f"\nResults:")
        print(f"  Unpivot total time: {unpivot_time:.4f}s")
        print(f"  MAP total time: {map_time:.4f}s")
        print(f"  Improvement: {improvement:.1f}%")
        print(f"  Speedup: {speedup:.2f}x")

        # Row reduction
        unpivot_rows = num_dates * num_metrics
        map_rows = num_dates
        row_reduction = (unpivot_rows - map_rows) / unpivot_rows * 100

        print(f"  Row reduction: {unpivot_rows} → {map_rows} ({row_reduction:.1f}% fewer rows)")

        conn.close()


if __name__ == "__main__":
    main()
```

**Commit**:
```bash
git add examples/batch_sql_optimization_benchmark.py
git commit -m "feat: add performance benchmark for MAP vs unpivot approaches"
```

## Summary

This implementation plan provides a complete solution for optimizing batch SQL queries using DuckDB's MAP feature:

1. **Backward Compatibility**: The existing `build_batch_cte_query` method is preserved with an optional `use_map` parameter
2. **Clean Architecture**: Helper methods `_build_cte_parts` and `_validate_metrics` reduce code duplication
3. **Comprehensive Testing**: TDD approach with unit tests, integration tests, and performance benchmarks
4. **Analyzer Integration**: The `analyze_batch_sql_ops` function automatically uses the optimization when available
5. **Performance Gains**: Reduces result rows from N*M to N, providing significant performance improvements for large batches

The implementation follows DQX coding standards:
- KISS/YAGNI principles - only adds necessary complexity
- Type hints throughout
- Comprehensive docstrings
- Proper error handling
- Clean commit messages following conventional commits

## Next Steps

After implementation:
1. Run all tests to ensure nothing is broken
2. Run the benchmark script to quantify performance improvements
3. Update documentation if needed
4. Consider adding support for other SQL dialects that support similar features
