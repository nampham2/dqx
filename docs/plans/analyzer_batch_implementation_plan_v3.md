# Analyzer Batch Implementation Plan v3

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
- Batch size limit of 7 dates per query (optimized for query performance)
- Fail the entire batch if any date fails

## Implementation Tasks

### Task Group 1: Create Data Structures and Update Protocols

**Files to modify:**
- `src/dqx/models.py` - Add BatchCTEData dataclass
- `src/dqx/common.py` - Update Analyzer protocol
- `src/dqx/analyzer.py` - Add batch size constant

**Changes:**

1. **Add BatchCTEData dataclass** (`src/dqx/models.py`):
```python
from dataclasses import dataclass
from typing import Sequence

@dataclass
class BatchCTEData:
    """Data for building a batch CTE query."""
    key: ResultKey
    cte_sql: str
    ops: Sequence[SqlOp]
```

2. **Update Analyzer Protocol** (`src/dqx/common.py`):
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

3. **Add batch constant** (`src/dqx/analyzer.py`):
```python
# Add at top of file after imports
DEFAULT_BATCH_SIZE = 7  # Maximum dates per SQL query
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
from dqx.models import BatchCTEData

@runtime_checkable
class Dialect(Protocol):
    # ... existing methods ...

    def build_batch_cte_query(
        self,
        cte_data: list[BatchCTEData]
    ) -> str:
        """Build a batch CTE query for multiple dates.

        Args:
            cte_data: List of BatchCTEData objects containing:
                - key: ResultKey with the date
                - cte_sql: CTE SQL for this date
                - ops: List of SqlOp objects to translate

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
def build_batch_cte_query(self, cte_data: list[BatchCTEData]) -> str:
    """Build batch query with date-suffixed CTEs and unpivot."""
    if not cte_data:
        raise ValueError("No CTE data provided")

    cte_parts = []
    unpivot_parts = []

    for data in cte_data:
        # Format date for CTE names (yyyy_mm_dd)
        date_suffix = data.key.yyyy_mm_dd.strftime('%Y_%m_%d')
        source_cte = f"source_{date_suffix}"
        metrics_cte = f"metrics_{date_suffix}"

        # Add source CTE
        cte_parts.append(f"{source_cte} AS ({data.cte_sql})")

        # Build metrics CTE with all expressions if ops exist
        if data.ops:
            # Translate ops to expressions
            expressions = [self.translate_sql_op(op) for op in data.ops]
            metrics_select = ", ".join(expressions)
            cte_parts.append(
                f"{metrics_cte} AS (SELECT {metrics_select} FROM {source_cte})"
            )

            # Create unpivot SELECT statements
            date_str = data.key.yyyy_mm_dd.isoformat()
            for op in data.ops:
                # Use op.sql_col directly for symbol name
                unpivot_parts.append(
                    f"SELECT '{date_str}' as date, '{op.sql_col}' as symbol, "
                    f"{op.sql_col} as value FROM {metrics_cte}"
                )

    # Build final query
    if not unpivot_parts:
        raise ValueError("No metrics to compute")

    cte_clause = "WITH\n  " + ",\n  ".join(cte_parts)
    union_clause = "\n".join(f"{'UNION ALL' if i > 0 else ''}\n{part}"
                            for i, part in enumerate(unpivot_parts))

    return f"{cte_clause}\n{union_clause}"
```

**Tests:**
```python
# tests/test_dialect.py

def test_build_batch_cte_query_single_date():
    """Test building batch CTE query for single date."""
    from dqx.models import BatchCTEData, ResultKey
    from dqx.ops import Sum, Average
    from datetime import date

    dialect = DuckDBDialect()

    # Create test data
    key = ResultKey(date(2024, 1, 1), {})
    ops = [
        Sum("revenue").analyzers[0],  # Should be x_1
        Average("price").analyzers[0]  # Should be x_2
    ]

    cte_data = [
        BatchCTEData(
            key=key,
            cte_sql="SELECT * FROM sales WHERE yyyy_mm_dd = '2024-01-01'",
            ops=ops
        )
    ]

    sql = dialect.build_batch_cte_query(cte_data)

    # Verify structure
    assert "WITH" in sql
    assert "source_2024_01_01 AS (" in sql
    assert "metrics_2024_01_01 AS (" in sql
    assert "SELECT '2024-01-01' as date, 'x_1' as symbol" in sql
    assert "SELECT '2024-01-01' as date, 'x_2' as symbol" in sql
    assert "UNION ALL" in sql


def test_build_batch_cte_query_multiple_dates():
    """Test building batch CTE query for multiple dates."""
    from dqx.models import BatchCTEData, ResultKey
    from dqx.ops import Sum, Maximum
    from datetime import date

    dialect = DuckDBDialect()

    # Create test data for 3 dates
    cte_data = []
    for day in [1, 2, 3]:
        key = ResultKey(date(2024, 1, day), {})
        ops = [Sum("revenue").analyzers[0]]  # x_1, x_2, x_3

        cte_data.append(BatchCTEData(
            key=key,
            cte_sql=f"SELECT * FROM sales WHERE yyyy_mm_dd = '2024-01-0{day}'",
            ops=ops
        ))

    sql = dialect.build_batch_cte_query(cte_data)

    # Verify all dates are included
    assert "source_2024_01_01" in sql
    assert "source_2024_01_02" in sql
    assert "source_2024_01_03" in sql
    assert sql.count("UNION ALL") == 2  # 3 selects, 2 unions


def test_build_batch_cte_query_empty():
    """Test error handling for empty CTE data."""
    dialect = DuckDBDialect()

    with pytest.raises(ValueError, match="No CTE data provided"):
        dialect.build_batch_cte_query([])


def test_build_batch_cte_query_no_ops():
    """Test error handling when no ops provided."""
    from dqx.models import BatchCTEData, ResultKey
    from datetime import date

    dialect = DuckDBDialect()

    cte_data = [
        BatchCTEData(
            key=ResultKey(date(2024, 1, 1), {}),
            cte_sql="SELECT * FROM sales",
            ops=[]  # No ops
        )
    ]

    with pytest.raises(ValueError, match="No metrics to compute"):
        dialect.build_batch_cte_query(cte_data)
```

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
        ops_by_key: Dict mapping ResultKey to list of deduplicated SqlOps

    Raises:
        DQXError: If SQL execution fails
    """
    if not ops_by_key:
        return

    # Get dialect
    dialect_instance = get_dialect(ds.dialect)

    # Build CTE data using dataclass
    cte_data = [
        BatchCTEData(
            key=key,
            cte_sql=ds.cte(key.yyyy_mm_dd),
            ops=ops
        )
        for key, ops in ops_by_key.items()
    ]

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

    # Assign values back to ops
    for key, ops in ops_by_key.items():
        date_str = key.yyyy_mm_dd.isoformat()
        for op in ops:
            value = value_map.get((date_str, op.sql_col))
            if value is not None:
                op.assign(value)
```

**Tests:**
```python
# tests/test_analyzer.py

def test_analyze_batch_sql_ops():
    """Test analyze_batch_sql_ops function."""
    from dqx.analyzer import analyze_batch_sql_ops
    from dqx.models import ResultKey
    from dqx.ops import Sum, Average
    from datetime import date
    import numpy as np

    # Create mock data source
    class MockDataSource:
        dialect = "duckdb"

        def cte(self, date):
            return f"SELECT * FROM table WHERE date = '{date}'"

        def query(self, sql, ref_date):
            # Return mock results
            class MockResult:
                def fetchnumpy(self):
                    return {
                        'date': np.array(['2024-01-01', '2024-01-01', '2024-01-02']),
                        'symbol': np.array(['x_1', 'x_2', 'x_1']),
                        'value': np.array([1500.0, 30.0, 2000.0])
                    }
            return MockResult()

    ds = MockDataSource()

    # Create ops
    key1 = ResultKey(date(2024, 1, 1), {})
    key2 = ResultKey(date(2024, 1, 2), {})

    ops1 = [
        Sum("revenue").analyzers[0],
        Average("price").analyzers[0]
    ]
    ops2 = [Sum("revenue").analyzers[0]]

    ops_by_key = {key1: ops1, key2: ops2}

    # Execute
    analyze_batch_sql_ops(ds, ops_by_key)

    # Verify values assigned
    assert ops1[0].value == 1500.0
    assert ops1[1].value == 30.0
    assert ops2[0].value == 2000.0


def test_analyze_batch_sql_ops_empty():
    """Test analyze_batch_sql_ops with empty input."""
    from dqx.analyzer import analyze_batch_sql_ops

    class MockDataSource:
        dialect = "duckdb"

    ds = MockDataSource()

    # Should not raise error
    analyze_batch_sql_ops(ds, {})


def test_analyze_batch_sql_ops_value_assignment():
    """Test correct value assignment including duplicates."""
    from dqx.analyzer import analyze_batch_sql_ops
    from dqx.models import ResultKey
    from dqx.ops import Sum
    from datetime import date
    import numpy as np

    class MockDataSource:
        dialect = "duckdb"

        def cte(self, date):
            return f"SELECT * FROM table WHERE date = '{date}'"

        def query(self, sql, ref_date):
            class MockResult:
                def fetchnumpy(self):
                    return {
                        'date': np.array(['2024-01-01']),
                        'symbol': np.array(['x_1']),
                        'value': np.array([1500.0])
                    }
            return MockResult()

    ds = MockDataSource()

    # Create same op instance
    sum_op = Sum("revenue").analyzers[0]

    key = ResultKey(date(2024, 1, 1), {})
    ops_by_key = {key: [sum_op]}

    # Execute
    analyze_batch_sql_ops(ds, ops_by_key)

    # Verify value assigned
    assert sum_op.value == 1500.0
```

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
    # Track both all ops and distinct ops from the start
    all_ops_by_key: dict[ResultKey, list[SqlOp]] = {}
    distinct_ops_by_key: dict[ResultKey, list[SqlOp]] = {}

    for key, metrics in metrics_by_key.items():
        all_ops = []
        distinct_ops = []
        seen = set()

        # Extract ops and deduplicate in one pass
        for metric in metrics:
            for analyzer in metric.analyzers:
                if isinstance(analyzer, SqlOp):
                    all_ops.append(analyzer)

                    # Add to distinct list if not seen before
                    if analyzer not in seen:
                        seen.add(analyzer)
                        distinct_ops.append(analyzer)

        if all_ops:
            all_ops_by_key[key] = all_ops
            distinct_ops_by_key[key] = distinct_ops

    if not distinct_ops_by_key:
        return AnalysisReport()

    # Batch analyze SQL ops with deduplicated ops
    analyze_batch_sql_ops(ds, distinct_ops_by_key)

    # Build report (using all ops including duplicates)
    report_data = {}
    for key, metrics in metrics_by_key.items():
        for metric in metrics:
            report_data[(metric, key)] = models.Metric.build(metric, key)

    return AnalysisReport(data=report_data)
```

**Tests:**
```python
# tests/test_analyzer.py

def test_analyze_batch_single_date():
    """Test batch analysis with single date."""
    from dqx import Analyzer, Sum, Average, ResultKey
    from dqx.datasource import ArrowDataSource
    from datetime import date
    import pyarrow as pa

    # Create test data
    data = pa.table({
        'yyyy_mm_dd': [date(2024, 1, 1)] * 5,
        'revenue': [100, 200, 300, 400, 500],
        'price': [10, 20, 30, 40, 50]
    })

    ds = ArrowDataSource("test", data)
    analyzer = Analyzer()

    key = ResultKey(date(2024, 1, 1), {})
    metrics_by_key = {key: [Sum("revenue"), Average("price")]}

    report = analyzer.analyze_batch(ds, metrics_by_key)

    # Verify results
    assert len(report) == 2
    assert report[(Sum("revenue"), key)].value == 1500.0
    assert report[(Average("price"), key)].value == 30.0


def test_analyze_batch_multiple_dates():
    """Test batch analysis with multiple dates."""
    from dqx import Analyzer, Sum, Maximum, ResultKey
    from dqx.datasource import ArrowDataSource
    from datetime import date
    import pyarrow as pa

    # Create test data for 3 dates
    dates = [date(2024, 1, i) for i in range(1, 4)]
    data_rows = []
    for d in dates:
        for i in range(5):
            data_rows.append({
                'yyyy_mm_dd': d,
                'revenue': 100 * d.day + i * 10
            })

    data = pa.table(data_rows)
    ds = ArrowDataSource("test", data)
    analyzer = Analyzer()

    # Different metrics per date
    metrics_by_key = {
        ResultKey(date(2024, 1, 1), {}): [Sum("revenue")],
        ResultKey(date(2024, 1, 2), {}): [Sum("revenue"), Maximum("revenue")],
        ResultKey(date(2024, 1, 3), {}): [Maximum("revenue")]
    }

    report = analyzer.analyze_batch(ds, metrics_by_key)

    # Verify results
    assert len(report) == 4  # Total metrics across dates

    key1 = ResultKey(date(2024, 1, 1), {})
    assert report[(Sum("revenue"), key1)].value == 600.0  # 100+110+120+130+140

    key2 = ResultKey(date(2024, 1, 2), {})
    assert report[(Sum("revenue"), key2)].value == 1100.0  # 200+210+220+230+240
    assert report[(Maximum("revenue"), key2)].value == 240.0

    key3 = ResultKey(date(2024, 1, 3), {})
    assert report[(Maximum("revenue"), key3)].value == 340.0


def test_analyze_batch_empty():
    """Test error handling for empty batch."""
    from dqx import Analyzer
    from dqx.datasource import ArrowDataSource
    from dqx.errors import DQXError
    import pyarrow as pa

    data = pa.table({'yyyy_mm_dd': [], 'revenue': []})
    ds = ArrowDataSource("test", data)
    analyzer = Analyzer()

    with pytest.raises(DQXError, match="No metrics provided"):
        analyzer.analyze_batch(ds, {})


def test_analyze_batch_large_batch():
    """Test batch analysis with more than 7 dates (tests batching)."""
    from dqx import Analyzer, Sum, ResultKey
    from dqx.datasource import ArrowDataSource
    from datetime import date
    import pyarrow as pa

    # Create data for 15 dates (will require 3 batches with size 7)
    dates = [date(2024, 1, i) for i in range(1, 16)]
    data_rows = []
    for d in dates:
        for i in range(5):
            data_rows.append({
                'yyyy_mm_dd': d,
                'revenue': d.day * 100
            })

    data = pa.table(data_rows)
    ds = ArrowDataSource("test", data)
    analyzer = Analyzer()

    # Create metrics for all dates
    metrics_by_key = {
        ResultKey(d, {}): [Sum("revenue")]
        for d in dates
    }

    report = analyzer.analyze_batch(ds, metrics_by_key)

    # Verify all dates processed
    assert len(report) == 15

    # Spot check some values
    key1 = ResultKey(date(2024, 1, 1), {})
    assert report[(Sum("revenue"), key1)].value == 500.0  # 100 * 5

    key8 = ResultKey(date(2024, 1, 8), {})
    assert report[(Sum("revenue"), key8)].value == 4000.0  # 800 * 5

    key15 = ResultKey(date(2024, 1, 15), {})
    assert report[(Sum("revenue"), key15)].value == 7500.0  # 1500 * 5


def test_analyze_batch_sql_failure():
    """Test error propagation from SQL failures."""
    from dqx import Analyzer, Sum, ResultKey
    from dqx.datasource import ArrowDataSource
    from dqx.errors import DQXError
    from datetime import date
    import pyarrow as pa

    # Create test data
    data = pa.table({
        'yyyy_mm_dd': [date(2024, 1, 1)] * 5,
        'revenue': [100, 200, 300, 400, 500]
    })

    ds = ArrowDataSource("test", data)
    analyzer = Analyzer()

    key = ResultKey(date(2024, 1, 1), {})
    # Request metric on non-existent column
    metrics_by_key = {key: [Sum("non_existent_column")]}

    with pytest.raises(DQXError):
        analyzer.analyze_batch(ds, metrics_by_key)


def test_analyze_batch_deduplication():
    """Test that duplicate metrics are handled correctly."""
    from dqx import Analyzer, Sum, ResultKey
    from dqx.datasource import ArrowDataSource
    from datetime import date
    import pyarrow as pa

    # Create test data
    data = pa.table({
        'yyyy_mm_dd': [date(2024, 1, 1)] * 5,
        'revenue': [100, 200, 300, 400, 500]
    })

    ds = ArrowDataSource("test", data)
    analyzer = Analyzer()

    # Request same metric multiple times
    key = ResultKey(date(2024, 1, 1), {})
    metrics_by_key = {key: [Sum("revenue"), Sum("revenue"), Sum("revenue")]}

    report = analyzer.analyze_batch(ds, metrics_by_key)

    # Should have 3 entries in report (one for each metric request)
    assert len(report) == 3
    # All should have the same value
    assert report[(Sum("revenue"), key)].value == 1500.0
```

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

2. **Test with lag operations**:
```python
def test_batch_analyze_with_lag():
    """Test batch analysis with lag operations."""
    # Create data for multiple consecutive days
    dates = [date(2024, 1, i) for i in range(1, 6)]  # Jan 1-5
    data_rows = []
    for d in dates:
        data_rows.extend([
            {'yyyy_mm_dd': d, 'revenue': 100 * (d.day), 'user_id': i}
            for i in range(1, 6)
        ])

    data = pa.table(data_rows)
    ds = ArrowDataSource("test", data)
    analyzer = Analyzer()

    # Analyze current and previous days
    metrics = [Sum("revenue"), Average("revenue")]
    metrics_by_key = {}

    # Add current day and lag days for Jan 3
    current_key = ResultKey(date(2024, 1, 3), {})
    metrics_by_key[current_key] = metrics
    metrics_by_key[current_key.lag(1)] = metrics  # Jan 2
    metrics_by_key[current_key.lag(2)] = metrics  # Jan 1

    report = analyzer.analyze_batch(ds, metrics_by_key)

    # Verify values
    assert report[(Sum("revenue"), current_key)].value == 1500.0  # 300 * 5
    assert report[(Sum("revenue"), current_key.lag(1))].value == 1000.0  # 200 * 5
    assert report[(Sum("revenue"), current_key.lag(2))].value == 500.0  # 100 * 5

    # Verify all 6 metrics present (2 metrics * 3 dates)
    assert len(report) == 6
```

3. **Test with tags**:
```python
def test_batch_analyze_with_tags():
    """Test batch analysis with different tag combinations."""
    # Create data with region and environment tags
    data_rows = []
    for d in [date(2024, 1, 1), date(2024, 1, 2)]:
        for region in ['us', 'eu', 'asia']:
            for env in ['prod', 'staging']:
                for i in range(5):
                    data_rows.append({
                        'yyyy_mm_dd': d,
                        'region': region,
                        'env': env,
                        'revenue': 100 + (10 if region == 'us' else 0),
                        'user_id': i
                    })

    data = pa.table(data_rows)
    ds = ArrowDataSource("test", data)
    analyzer = Analyzer()

    # Create keys with different tag combinations
    keys_and_metrics = [
        # Date 1
        (ResultKey(date(2024, 1, 1), {"region": "us", "env": "prod"}), [Sum("revenue")]),
        (ResultKey(date(2024, 1, 1), {"region": "eu", "env": "prod"}), [Sum("revenue")]),
        (ResultKey(date(2024, 1, 1), {"region": "asia", "env": "staging"}), [Average("revenue")]),
        # Date 2
        (ResultKey(date(2024, 1, 2), {"region": "us", "env": "prod"}), [Sum("revenue")]),
        (ResultKey(date(2024, 1, 2), {"env": "staging"}), [Count()]),  # No region filter
    ]

    metrics_by_key = dict(keys_and_metrics)
    report = analyzer.analyze_batch(ds, metrics_by_key)

    # Verify results
    key1_us = ResultKey(date(2024, 1, 1), {"region": "us", "env": "prod"})
    assert report[(Sum("revenue"), key1_us)].value == 550.0  # 110 * 5

    key1_eu = ResultKey(date(2024, 1, 1), {"region": "eu", "env": "prod"})
    assert report[(Sum("revenue"), key1_eu)].value == 500.0  # 100 * 5

    key2_staging = ResultKey(date(2024, 1, 2), {"env": "staging"})
    assert report[(Count(), key2_staging)].value == 15.0  # 3 regions * 5 rows

    # Verify all requested metrics are present
    assert len(report) == 5
```

4. **Test persistence to MetricDB**:
```python
def test_batch_analyze_persists_correctly():
    """Test that batch results persist correctly to MetricDB."""
    # Create test data
    data = pa.table({
        'yyyy_mm_dd': [date(2024, 1, 1)] * 10 + [date(2024, 1, 2)] * 10 + [date(2024, 1, 3)] * 10,
        'revenue': list(range(100, 110)) * 3,
        'cost': list(range(50, 60)) * 3,
        'user_id': list(range(1, 11)) * 3
    })

    ds = ArrowDataSource("test", data)
    db = DuckMetricDB("test_batch_metrics.db")
    analyzer = Analyzer()

    # Analyze multiple dates with different metrics
    metrics_by_key = {
        ResultKey(date(2024, 1, 1), {"suite": "daily"}): [
            Sum("revenue"), Average("cost"), Count()
        ],
        ResultKey(date(2024, 1, 2), {"suite": "daily"}): [
            Sum("revenue"), Maximum("revenue"), NullCount("user_id")
        ],
        ResultKey(date(2024, 1, 3), {"suite": "daily"}): [
            Average("revenue"), Minimum("cost")
        ]
    }

    # Run batch analysis and persist
    report = analyzer.analyze_batch(ds, metrics_by_key)
    report.persist(db)

    # Verify all metrics were persisted correctly
    for key, metrics in metrics_by_key.items():
        for metric in metrics:
            # Retrieve from database
            retrieved = db.get(key, metric)
            assert retrieved is not None

            # Verify value matches report
            expected = report[(metric, key)]
            assert retrieved.value == expected.value
            assert retrieved.spec == expected.spec
            assert retrieved.key == expected.key

    # Verify count
    all_metrics = db.list()
    assert len(all_metrics) == 8  # Total metrics across all dates

    # Clean up
    db.close()
    os.remove("test_batch_metrics.db")
```

5. **Test performance comparison**:
```python
def test_batch_analyze_performance():
    """Compare batch vs sequential analysis performance."""
    import time
    import random

    # Create larger dataset - 30 days of data
    dates = [date(2024, 1, i) for i in range(1, 31)]
    data_rows = []
    for d in dates:
        for i in range(1000):  # 1000 rows per day
            data_rows.append({
                'yyyy_mm_dd': d,
                'revenue': random.randint(100, 1000),
                'cost': random.randint(50, 500),
                'user_id': i,
                'category': random.choice(['A', 'B', 'C', 'D'])
            })

    data = pa.table(data_rows)
    ds = ArrowDataSource("perf_test", data)

    # Define metrics to compute
    metrics = [
        Sum("revenue"),
        Average("revenue"),
        Maximum("revenue"),
        Minimum("cost"),
        Average("cost"),
        StandardDeviation("revenue"),
        Count(),
        DistinctCount("category")
    ]

    # Time sequential approach
    analyzer1 = Analyzer()
    start_sequential = time.time()
    sequential_reports = []
    for d in dates:
        key = ResultKey(d, {"test": "sequential"})
        report = analyzer1.analyze(ds, metrics, key)
        sequential_reports.append(report)
    sequential_time = time.time() - start_sequential

    # Time batch approach
    analyzer2 = Analyzer()
    start_batch = time.time()
    metrics_by_key = {
        ResultKey(d, {"test": "batch"}): metrics
        for d in dates
    }
    batch_report = analyzer2.analyze_batch(ds, metrics_by_key)
    batch_time = time.time() - start_batch

    # Log performance results
    logger.info(f"Sequential time: {sequential_time:.2f}s")
    logger.info(f"Batch time: {batch_time:.2f}s")
    logger.info(f"Speedup: {sequential_time / batch_time:.1f}x")

    # Verify batch is significantly faster
    assert batch_time < sequential_time * 0.5, \
        f"Batch should be at least 2x faster. Sequential: {sequential_time:.2f}s, Batch: {batch_time:.2f}s"

    # Verify results are identical
    for i, d in enumerate(dates):
        seq_key = ResultKey(d, {"test": "sequential"})
        batch_key = ResultKey(d, {"test": "batch"})

        for metric in metrics:
            seq_value = sequential_reports[i][(metric, seq_key)].value
            batch_value = batch_report[(metric, batch_key)].value
            assert abs(seq_value - batch_value) < 1e-10, \
                f"Values don't match for {metric} on {d}: {seq_value} != {batch_value}"

    # Verify we analyzed all dates
    assert len(batch_report) == len(dates) * len(metrics)
```

### Additional Edge Case Tests

```python
def test_batch_analyze_symbol_ordering():
    """Ensure symbols maintain proper numeric ordering across dates."""
    # Create data with many metrics to test symbol ordering
    data = pa.table({
        'yyyy_mm_dd': [date(2024, 1, 1)] * 5 + [date(2024, 1, 2)] * 5,
        **{f'col_{i}': list(range(5)) * 2 for i in range(15)}  # 15 columns
    })

    ds = ArrowDataSource("test", data)
    analyzer = Analyzer()

    # Create many metrics to ensure we get x_1 through x_15+
    metrics = [Sum(f"col_{i}") for i in range(15)]

    report = analyzer.analyze_batch(ds, {
        ResultKey(date(2024, 1, 1), {}): metrics[:10],  # x_1 to x_10
        ResultKey(date(2024, 1, 2), {}): metrics[5:]   # x_11 to x_20
    })

    # Verify all metrics computed correctly
    key1 = ResultKey(date(2024, 1, 1), {})
    key2 = ResultKey(date(2024, 1, 2), {})

    for i in range(10):
        assert report[(metrics[i], key1)].value == 10.0  # sum(0,1,2,3,4)

    for i in range(5, 15):
        assert report[(metrics[i], key2)].value == 10.0


def test_batch_analyze_duplicate_metrics():
    """Test same metric requested for same date (should dedupe)."""
    data = pa.table({
        'yyyy_mm_dd': [date(2024, 1, 1)] * 5,
        'revenue': [100, 200, 300, 400, 500]
    })

    ds = ArrowDataSource("test", data)
    analyzer = Analyzer()

    # Request same metric multiple times
    key = ResultKey(date(2024, 1, 1), {})
    metrics_by_key = {key: [Sum("revenue"), Sum("revenue"), Sum("revenue")]}

    report = analyzer.analyze_batch(ds, metrics_by_key)

    # Should only compute once but report should have 3 entries
    assert len(report) == 3
    assert report[(Sum("revenue"), key)].value == 1500.0


def test_batch_analyze_empty_batch():
    """Test batch analysis with empty metrics."""
    data = pa.table({
        'yyyy_mm_dd': [date(2024, 1, 1)] * 5,
        'revenue': [100, 200, 300, 400, 500]
    })

    ds = ArrowDataSource("test", data)
    analyzer = Analyzer()

    # Empty batch
    with pytest.raises(DQXError, match="No metrics provided"):
        analyzer.analyze_batch(ds, {})


def test_batch_analyze_sql_failure():
    """Test batch analysis handles SQL errors gracefully."""
    data = pa.table({
        'yyyy_mm_dd': [date(2024, 1, 1)] * 5,
        'revenue': [100, 200, 300, 400, 500]
    })

    ds = ArrowDataSource("test", data)
    analyzer = Analyzer()

    # Request metric on non-existent column
    key = ResultKey(date(2024, 1, 1), {})
    metrics_by_key = {key: [Sum("non_existent_column")]}

    # Should raise DQXError from SQL execution
    with pytest.raises(DQXError):
        analyzer.analyze_batch(ds, metrics_by_key)
```

### Task Group 6: Documentation

**Files to update:**
- Update `README.md` with batch analysis documentation

**Changes:**

Add a new section to README.md after the existing analysis examples:

```markdown
## Batch Analysis

For time-series analysis across multiple dates, use `analyze_batch` to process multiple dates in a single SQL query:

```python
from dqx import Analyzer, Sum, Average, ResultKey
from datetime import date

# Analyze a week of data in one query
analyzer = Analyzer()
metrics_by_key = {
    ResultKey(date(2024, 1, i), {}): [Sum("revenue"), Average("price")]
    for i in range(1, 8)
}

report = analyzer.analyze_batch(ds, metrics_by_key)

# Results are in the same format as sequential analysis
for i in range(1, 8):
    key = ResultKey(date(2024, 1, i), {})
    revenue = report[(Sum("revenue"), key)].value
    avg_price = report[(Average("price"), key)].value
    print(f"Jan {i}: Revenue=${revenue:,.2f}, Avg Price=${avg_price:.2f}")
```

### Benefits:
- **Performance**: Up to 50x faster for multi-date analysis
- **Flexibility**: Different metrics per date
- **Compatibility**: Same `AnalysisReport` format
- **Scalability**: Automatic batching for large date ranges
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
uv run mypy src/dqx/analyzer.py src/dqx/dialect.py src/dqx/models.py
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

## Changes from v2

1. **Batch size**: Reduced from 100 to 7 for optimal query performance
2. **Deduplication optimization**: Moved deduplication logic to `_analyze_batch_internal` and optimized to single pass
3. **Complete test implementations**: Added all test code for Task Groups 2, 3, and 4
4. **Input validation**: `analyze_batch_sql_ops` now expects already-deduplicated ops
