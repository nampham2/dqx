"""Test dialect integration with Analyzer and DataSources."""

import datetime
from unittest.mock import Mock

import numpy as np
import pyarrow as pa

from dqx import specs
from dqx.analyzer import Analyzer, analyze_sql_ops
from dqx.common import ResultKey
from dqx.dialect import DuckDBDialect
from dqx.extensions.pyarrow_ds import ArrowDataSource
from dqx.ops import Average, Maximum, Minimum, NullCount, NumRows, SqlOp, Sum


class MockPostgreSQLDialect:
    """Mock PostgreSQL dialect for testing."""
    
    name = "postgresql"
    
    def translate_sql_op(self, op: SqlOp) -> str:
        """Translate ops to PostgreSQL-compatible SQL."""
        match op:
            case NumRows():
                return f"COUNT(*)::FLOAT8 AS {op.sql_col}"
            case Average(column=col):
                return f"AVG({col})::FLOAT8 AS {op.sql_col}"
            case Sum(column=col):
                return f"SUM({col})::FLOAT8 AS {op.sql_col}"
            case Minimum(column=col):
                return f"MIN({col})::FLOAT8 AS {op.sql_col}"
            case Maximum(column=col):
                return f"MAX({col})::FLOAT8 AS {op.sql_col}"
            case NullCount(column=col):
                # PostgreSQL doesn't have COUNT_IF
                return f"COUNT(CASE WHEN {col} IS NULL THEN 1 END)::FLOAT8 AS {op.sql_col}"
            case _:
                raise ValueError(f"Unsupported op: {type(op).__name__}")
    
    def build_cte_query(self, cte_sql: str, select_expressions: list[str]) -> str:
        """Build PostgreSQL-compatible CTE query."""
        from dqx.dialect import build_cte_query
        return build_cte_query(cte_sql, select_expressions)


def test_datasource_with_duckdb_dialect() -> None:
    """Test that ArrowDataSource uses DuckDB dialect by default."""
    # Create test data
    table = pa.table({
        'value': [10, 20, 30, None, 50],
        'category': ['A', 'B', 'A', 'B', 'A']
    })
    
    # Create data source - should use DuckDB dialect by default
    ds = ArrowDataSource(table)
    
    assert ds.dialect.name == "duckdb"
    assert isinstance(ds.dialect, DuckDBDialect)


def test_datasource_with_custom_dialect() -> None:
    """Test that ArrowDataSource can use a custom dialect."""
    # Create test data
    table = pa.table({
        'value': [10, 20, 30, None, 50],
        'category': ['A', 'B', 'A', 'B', 'A']
    })
    
    # Create data source with custom dialect
    custom_dialect = MockPostgreSQLDialect()
    ds = ArrowDataSource(table, dialect=custom_dialect)
    
    assert ds.dialect.name == "postgresql"
    assert isinstance(ds.dialect, MockPostgreSQLDialect)


def test_analyzer_uses_dialect_for_sql_generation() -> None:
    """Test that analyzer uses the data source's dialect for SQL generation."""
    # Create test data
    table = pa.table({
        'value': [10, 20, 30, 40, 50],
        'category': ['A', 'B', 'A', 'B', 'A']
    })
    
    # Create metrics
    avg_metric = specs.Average("value")
    sum_metric = specs.Sum("value")
    
    # Test with DuckDB dialect (default)
    ds_duckdb = ArrowDataSource(table)
    analyzer = Analyzer()
    
    key = ResultKey(yyyy_mm_dd=datetime.date(2024, 1, 1), tags={})
    report = analyzer.analyze(ds_duckdb, [avg_metric, sum_metric], key)
    
    # Results should be computed correctly
    assert len(report) == 2
    assert report[(avg_metric, key)].value == 30.0
    assert report[(sum_metric, key)].value == 150.0


def test_analyze_sql_ops_with_different_dialects() -> None:
    """Test analyze_sql_ops function with different dialects."""
    # Create mock ops
    ops: list[SqlOp] = [NumRows(), Average("price"), NullCount("quantity")]
    
    # Create mock data source with DuckDB dialect
    ds_duckdb = Mock()
    ds_duckdb.dialect = DuckDBDialect()
    ds_duckdb.cte = "SELECT * FROM orders"
    
    # Mock the query result - column names from dialect don't have quotes in result keys
    mock_result = {}
    for op in ops:
        mock_result[op.sql_col] = np.array([100.0]) if isinstance(op, NumRows) else np.array([25.5]) if isinstance(op, Average) else np.array([5.0])
    
    ds_duckdb.query.return_value.fetchnumpy.return_value = mock_result
    
    # Run analysis
    analyze_sql_ops(ds_duckdb, ops)
    
    # Verify SQL was generated with DuckDB dialect
    call_args = ds_duckdb.query.call_args[0][0]
    # Remove extra spaces for easier assertion
    normalized_sql = ' '.join(call_args.split())
    assert "CAST(COUNT(*) AS DOUBLE)" in normalized_sql
    assert "CAST(AVG(price) AS DOUBLE)" in normalized_sql
    assert "COUNT_IF(quantity IS NULL)" in normalized_sql
    
    # Create mock data source with PostgreSQL dialect
    ds_postgres = Mock()
    ds_postgres.dialect = MockPostgreSQLDialect()
    ds_postgres.cte = "SELECT * FROM orders"
    ds_postgres.query.return_value.fetchnumpy.return_value = mock_result
    
    # Run analysis with PostgreSQL dialect - using same ops to keep same prefixes
    analyze_sql_ops(ds_postgres, ops)
    
    # Verify SQL was generated with PostgreSQL dialect
    call_args = ds_postgres.query.call_args[0][0]
    # Remove extra spaces for easier assertion
    normalized_sql = ' '.join(call_args.split())
    assert "COUNT(*)::FLOAT8" in normalized_sql
    assert "AVG(price)::FLOAT8" in normalized_sql
    assert "COUNT(CASE WHEN quantity IS NULL THEN 1 END)" in normalized_sql


def test_datasource_without_dialect_raises_error() -> None:
    """Test that analyzer raises error for data sources without dialect."""
    # Create mock data source without dialect attribute
    ds_legacy = Mock()
    del ds_legacy.dialect  # Ensure no dialect attribute
    ds_legacy.cte = "SELECT * FROM legacy_table"
    ds_legacy.name = "legacy_ds"
    
    # Create ops
    ops: list[SqlOp] = [NumRows(), Average("value")]
    
    # Should raise error when trying to analyze without dialect
    from dqx.common import DQXError
    import pytest
    
    with pytest.raises(DQXError, match="Data source legacy_ds must have a dialect to analyze SQL ops"):
        analyze_sql_ops(ds_legacy, ops)


def test_dialect_query_formatting() -> None:
    """Test the beautiful query formatting from dialects."""
    dialect = DuckDBDialect()
    
    cte_sql = "SELECT * FROM sales WHERE date > '2024-01-01'"
    expressions = [
        dialect.translate_sql_op(NumRows()),
        dialect.translate_sql_op(Average("price")),
        dialect.translate_sql_op(Sum("quantity")),
        dialect.translate_sql_op(NullCount("customer_id"))
    ]
    
    query = dialect.build_cte_query(cte_sql, expressions)
    
    # Check query structure
    assert "WITH source AS (" in query
    assert cte_sql in query
    assert "SELECT" in query
    assert "FROM source" in query
    
    # Check alignment - all commas should be aligned
    lines = query.split('\n')
    select_lines = [line for line in lines if line.strip().startswith(',') or line.strip().startswith('CAST')]
    
    # All continuation lines should start with comma at same position
    comma_positions = [line.find(',') for line in select_lines if ',' in line]
    if comma_positions:
        assert len(set(comma_positions)) == 1, "Commas should be aligned"
