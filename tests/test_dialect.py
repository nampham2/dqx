"""Test dialect implementation and integration."""

import datetime
from typing import Any
from unittest.mock import Mock

import numpy as np
import pyarrow as pa
import pytest

from dqx import specs
from dqx.analyzer import Analyzer, analyze_sql_ops
from dqx.common import DQXError, ResultKey
from dqx.dialect import (
    _DIALECT_REGISTRY,
    Dialect,
    DuckDBDialect,
    build_cte_query,
    get_dialect,
    register_dialect,
)
from dqx.extensions.pyarrow_ds import ArrowDataSource
from dqx.ops import Average, First, Maximum, Minimum, NegativeCount, NullCount, NumRows, SqlOp, Sum, Variance


class MockDialect:
    """Mock dialect for testing."""

    name = "mock"

    def translate_sql_op(self, op: SqlOp) -> str:
        """Translate ops to SQL."""
        return f"MOCK({op.column if hasattr(op, 'column') else '*'}) AS {op.sql_col}"

    def build_cte_query(self, cte_sql: str, select_expressions: list[str]) -> str:
        """Build CTE query."""
        return f"WITH source AS ({cte_sql}) SELECT {', '.join(select_expressions)} FROM source"


class NotADialect:
    """Class that doesn't implement Dialect protocol."""

    def some_method(self) -> str:
        return "not a dialect"


class MockPostgreSQLDialect:
    """Mock PostgreSQL dialect for integration testing."""

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
        return build_cte_query(cte_sql, select_expressions)


# =============================================================================
# Protocol Compliance Tests
# =============================================================================


class TestDialectProtocol:
    """Test Dialect protocol compliance."""

    def test_dialect_is_protocol(self) -> None:
        """Test that Dialect is a Protocol."""
        # Protocol is a special typing construct, not a regular class
        assert hasattr(Dialect, "_is_protocol")
        assert Dialect._is_protocol  # type: ignore[attr-defined]

    def test_mock_dialect_implements_protocol(self) -> None:
        """Test that MockDialect implements Dialect protocol."""
        dialect = MockDialect()

        # Check methods exist and are callable
        assert hasattr(dialect, "translate_sql_op")
        assert hasattr(dialect, "build_cte_query")
        assert callable(dialect.translate_sql_op)
        assert callable(dialect.build_cte_query)

        # Test method signatures by calling them
        op = NumRows()
        result = dialect.translate_sql_op(op)
        assert isinstance(result, str)

        cte_query = dialect.build_cte_query("SELECT *", ["col1", "col2"])
        assert isinstance(cte_query, str)

    def test_not_a_dialect_doesnt_implement_protocol(self) -> None:
        """Test that NotADialect doesn't implement Dialect protocol."""
        not_dialect = NotADialect()

        # Should not have required methods
        assert not hasattr(not_dialect, "translate_sql_op")
        assert not hasattr(not_dialect, "build_cte_query")

    def test_protocol_type_checking(self) -> None:
        """Test protocol type checking at runtime."""
        dialect = MockDialect()
        not_dialect = NotADialect()

        def requires_dialect(d: Any) -> bool:
            """Check if object implements Dialect protocol."""
            return (
                hasattr(d, "translate_sql_op")
                and hasattr(d, "build_cte_query")
                and callable(d.translate_sql_op)
                and callable(d.build_cte_query)
            )

        assert requires_dialect(dialect) is True
        assert requires_dialect(not_dialect) is False


# =============================================================================
# DuckDB Dialect Implementation Tests
# =============================================================================


class TestDuckDBDialect:
    """Test DuckDB dialect implementation."""

    def test_duckdb_implements_dialect(self) -> None:
        """Test that DuckDBDialect implements Dialect protocol."""
        dialect = DuckDBDialect()
        assert hasattr(dialect, "translate_sql_op")
        assert hasattr(dialect, "build_cte_query")
        assert hasattr(dialect, "name")
        assert dialect.name == "duckdb"

    def test_translate_num_rows(self) -> None:
        """Test translation of NumRows op."""
        dialect = DuckDBDialect()
        op = NumRows()
        sql = dialect.translate_sql_op(op)
        assert "CAST(COUNT(*) AS DOUBLE)" in sql
        assert f"AS '{op.sql_col}'" in sql

    def test_translate_average(self) -> None:
        """Test translation of Average op."""
        dialect = DuckDBDialect()
        op = Average("price")
        sql = dialect.translate_sql_op(op)
        assert "CAST(AVG(price) AS DOUBLE)" in sql
        assert f"AS '{op.sql_col}'" in sql

    def test_translate_sum(self) -> None:
        """Test translation of Sum op."""
        dialect = DuckDBDialect()
        op = Sum("quantity")
        sql = dialect.translate_sql_op(op)
        assert "CAST(SUM(quantity) AS DOUBLE)" in sql
        assert f"AS '{op.sql_col}'" in sql

    def test_translate_minimum(self) -> None:
        """Test translation of Minimum op."""
        dialect = DuckDBDialect()
        op = Minimum("temperature")
        sql = dialect.translate_sql_op(op)
        assert "CAST(MIN(temperature) AS DOUBLE)" in sql
        assert f"AS '{op.sql_col}'" in sql

    def test_translate_maximum(self) -> None:
        """Test translation of Maximum op."""
        dialect = DuckDBDialect()
        op = Maximum("score")
        sql = dialect.translate_sql_op(op)
        assert "CAST(MAX(score) AS DOUBLE)" in sql
        assert f"AS '{op.sql_col}'" in sql

    def test_translate_null_count(self) -> None:
        """Test translation of NullCount op."""
        dialect = DuckDBDialect()
        op = NullCount("email")
        sql = dialect.translate_sql_op(op)
        assert "CAST(COUNT_IF(email IS NULL) AS DOUBLE)" in sql
        assert f"AS '{op.sql_col}'" in sql

    def test_translate_variance(self) -> None:
        """Test translation of Variance op."""
        dialect = DuckDBDialect()
        op = Variance("sales")
        sql = dialect.translate_sql_op(op)
        assert "CAST(VARIANCE(sales) AS DOUBLE)" in sql
        assert f"AS '{op.sql_col}'" in sql

    def test_translate_first(self) -> None:
        """Test translation of First op."""
        dialect = DuckDBDialect()
        op = First("timestamp")
        sql = dialect.translate_sql_op(op)
        assert "CAST(FIRST(timestamp) AS DOUBLE)" in sql
        assert f"AS '{op.sql_col}'" in sql

    def test_translate_negative_count(self) -> None:
        """Test translation of NegativeCount op."""
        dialect = DuckDBDialect()
        op = NegativeCount("profit")
        sql = dialect.translate_sql_op(op)
        assert "CAST(COUNT_IF(profit < 0.0) AS DOUBLE)" in sql
        assert f"AS '{op.sql_col}'" in sql

    def test_translate_duplicate_count(self) -> None:
        """Test translation of DuplicateCount op."""
        from dqx import ops
        from dqx.dialect import DuckDBDialect

        dialect = DuckDBDialect()

        # Test single column
        op1 = ops.DuplicateCount(["user_id"])
        sql1 = dialect.translate_sql_op(op1)
        assert sql1 == f"CAST(COUNT(*) - COUNT(DISTINCT user_id) AS DOUBLE) AS '{op1.sql_col}'"

        # Test multiple columns
        op2 = ops.DuplicateCount(["user_id", "product_id", "date"])
        sql2 = dialect.translate_sql_op(op2)
        # Columns should be sorted: date, product_id, user_id
        assert sql2 == f"CAST(COUNT(*) - COUNT(DISTINCT (date, product_id, user_id)) AS DOUBLE) AS '{op2.sql_col}'"

    def test_translate_duplicate_count_with_duckdb_execution(self) -> None:
        """Test that the generated SQL actually works in DuckDB."""
        import duckdb

        from dqx import ops
        from dqx.dialect import DuckDBDialect

        dialect = DuckDBDialect()

        # Create test data
        conn = duckdb.connect(":memory:")
        conn.execute("""
            CREATE TABLE test_data AS
            SELECT * FROM (VALUES
                (1, 'A', 100),
                (1, 'A', 100),  -- Duplicate
                (2, 'B', 200),
                (2, 'B', 300),  -- Same user_id and name, different amount
                (3, 'C', 400)
            ) AS t(user_id, name, amount)
        """)

        # Test single column
        op1 = ops.DuplicateCount(["user_id"])
        sql1 = dialect.translate_sql_op(op1)

        # Execute the generated SQL
        row1 = conn.execute(f"SELECT {sql1} FROM test_data").fetchone()
        assert row1 is not None
        result1 = row1[0]
        assert result1 == 2.0  # 5 rows - 3 unique user_ids = 2 duplicates

        # Test multiple columns
        op2 = ops.DuplicateCount(["user_id", "name"])
        sql2 = dialect.translate_sql_op(op2)

        row2 = conn.execute(f"SELECT {sql2} FROM test_data").fetchone()
        assert row2 is not None
        result2 = row2[0]
        assert result2 == 2.0  # 5 rows - 3 unique (user_id, name) pairs = 2 duplicates

        # Test all columns
        op3 = ops.DuplicateCount(["user_id", "name", "amount"])
        sql3 = dialect.translate_sql_op(op3)

        row3 = conn.execute(f"SELECT {sql3} FROM test_data").fetchone()
        assert row3 is not None
        result3 = row3[0]
        assert result3 == 1.0  # 5 rows - 4 unique combinations = 1 duplicate

        # Test column order doesn't affect result
        op4 = ops.DuplicateCount(["name", "user_id"])  # Different order
        sql4 = dialect.translate_sql_op(op4)

        row4 = conn.execute(f"SELECT {sql4} FROM test_data").fetchone()
        assert row4 is not None
        result4 = row4[0]
        assert result4 == 2.0  # Same result as op2

        conn.close()

    def test_translate_unsupported_op(self) -> None:
        """Test translation of unsupported op."""
        dialect = DuckDBDialect()

        class UnsupportedOp(SqlOp):
            def __init__(self) -> None:
                super().__init__()
                self._value: float = 0.0

            @property
            def name(self) -> str:
                return "unsupported"

            @property
            def prefix(self) -> str:
                return "unsup"

            @property
            def sql_col(self) -> str:
                return f"_{self.prefix}_{self.name}()"

            @property
            def value(self) -> Any:
                return self._value

            def assign(self, src: Any) -> None:
                """Not implemented."""
                pass

            def clear(self) -> None:
                """Not implemented."""
                pass

        op = UnsupportedOp()
        with pytest.raises(ValueError, match="Unsupported SqlOp type: UnsupportedOp"):
            dialect.translate_sql_op(op)

    def test_build_cte_query_single_expression(self) -> None:
        """Test building CTE query with single expression."""
        dialect = DuckDBDialect()
        cte_sql = "SELECT * FROM sales"
        expressions = ["COUNT(*) AS row_count"]

        query = dialect.build_cte_query(cte_sql, expressions)

        assert "WITH source AS (" in query
        assert cte_sql in query
        assert "COUNT(*) AS row_count" in query
        assert "FROM source" in query

    def test_build_cte_query_empty_expressions_raises_error(self) -> None:
        """Test that build_cte_query raises ValueError with empty expressions."""
        dialect = DuckDBDialect()
        cte_sql = "SELECT * FROM sales"
        empty_expressions: list[str] = []

        with pytest.raises(ValueError, match="No SELECT expressions provided"):
            dialect.build_cte_query(cte_sql, empty_expressions)

    def test_build_cte_query_multiple_expressions(self) -> None:
        """Test building CTE query with multiple expressions."""
        dialect = DuckDBDialect()
        cte_sql = "SELECT * FROM products WHERE active = true"
        expressions = ["COUNT(*) AS total_count", "AVG(price) AS avg_price", "MAX(stock) AS max_stock"]

        query = dialect.build_cte_query(cte_sql, expressions)

        # Check structure
        assert "WITH source AS (" in query
        assert cte_sql in query
        assert "FROM source" in query

        # Check expressions are included
        assert "total_count" in query and "COUNT(*)" in query
        assert "avg_price" in query and "AVG(price)" in query
        assert "max_stock" in query and "MAX(stock)" in query

        # Check formatting - expressions should be on separate lines
        lines = query.split("\n")
        assert len(lines) > 3  # At least WITH, SELECT, expressions, FROM

    def test_build_cte_query_formatting(self) -> None:
        """Test CTE query formatting is consistent."""
        dialect = DuckDBDialect()
        cte_sql = "SELECT * FROM orders"
        expressions = ["COUNT(*)", "SUM(amount)", "AVG(quantity)"]

        query = dialect.build_cte_query(cte_sql, expressions)
        lines = query.split("\n")

        # Check indentation
        assert lines[0] == "WITH source AS ("
        assert lines[1].startswith("  ")  # CTE SQL should be indented
        assert lines[2] == ")"
        assert "SELECT" in lines[3]
        assert "FROM source" in lines[-1]

    def test_dialect_with_real_ops(self) -> None:
        """Test dialect with actual SqlOp instances."""
        dialect = DuckDBDialect()

        ops = [NumRows(), Average("revenue"), Sum("units"), Maximum("price"), Minimum("cost"), NullCount("customer_id")]

        expressions = [dialect.translate_sql_op(op) for op in ops]  # type: ignore[arg-type]

        # All expressions should be valid SQL
        assert all("AS" in expr for expr in expressions)
        assert all("CAST(" in expr for expr in expressions)
        assert all("AS DOUBLE)" in expr for expr in expressions)

        # Build complete query
        cte_sql = "SELECT * FROM transactions WHERE year = 2024"
        query = dialect.build_cte_query(cte_sql, expressions)

        # Should be valid SQL structure
        assert query.startswith("WITH source AS (")
        assert "FROM source" in query
        assert query.count("AS") >= len(ops) * 2  # Each op has AS in CAST and alias


# =============================================================================
# Dialect Registry Tests
# =============================================================================


class TestDialectRegistry:
    """Test dialect registry functionality."""

    def setup_method(self) -> None:
        """Store original registry state before each test."""
        self.original_registry = _DIALECT_REGISTRY.copy()

    def teardown_method(self) -> None:
        """Restore original registry state after each test."""
        _DIALECT_REGISTRY.clear()
        _DIALECT_REGISTRY.update(self.original_registry)

    def test_duckdb_dialect_is_registered_by_default(self) -> None:
        """Test that DuckDB dialect is registered on import."""
        assert "duckdb" in _DIALECT_REGISTRY
        assert _DIALECT_REGISTRY["duckdb"] is DuckDBDialect

    def test_register_dialect_success(self) -> None:
        """Test successful dialect registration."""
        register_dialect("test", MockDialect)
        assert "test" in _DIALECT_REGISTRY
        assert _DIALECT_REGISTRY["test"] is MockDialect

    def test_register_dialect_duplicate_fails(self) -> None:
        """Test that registering duplicate dialect fails."""
        register_dialect("custom", MockDialect)

        with pytest.raises(ValueError, match="Dialect 'custom' is already registered"):
            register_dialect("custom", DuckDBDialect)

    def test_get_dialect_success(self) -> None:
        """Test getting registered dialect."""
        # DuckDB should be pre-registered
        dialect = get_dialect("duckdb")
        assert isinstance(dialect, DuckDBDialect)

        # Register and get custom dialect
        register_dialect("mock", MockDialect)
        mock_dialect = get_dialect("mock")
        assert isinstance(mock_dialect, MockDialect)

    def test_get_dialect_not_found(self) -> None:
        """Test getting unregistered dialect."""
        with pytest.raises(DQXError) as exc_info:
            get_dialect("nonexistent")

        error_msg = str(exc_info.value)
        assert "Dialect 'nonexistent' not found in registry" in error_msg
        assert "Available dialects:" in error_msg
        assert "duckdb" in error_msg  # Should list available dialects

    def test_get_dialect_creates_new_instance(self) -> None:
        """Test that get_dialect returns new instance each time."""
        dialect1 = get_dialect("duckdb")
        dialect2 = get_dialect("duckdb")

        # Should be same class but different instances
        assert type(dialect1) is type(dialect2)
        assert dialect1 is not dialect2

    def test_registry_isolation(self) -> None:
        """Test that registry modifications are isolated."""
        # Register dialect
        register_dialect("isolated", MockDialect)
        assert "isolated" in _DIALECT_REGISTRY

        # Clear and restore
        _DIALECT_REGISTRY.clear()
        assert "isolated" not in _DIALECT_REGISTRY

        # Original duckdb registration should be restorable
        _DIALECT_REGISTRY.update(self.original_registry)
        assert "duckdb" in _DIALECT_REGISTRY


# =============================================================================
# DataSource Integration Tests
# =============================================================================


class TestDataSourceDialectIntegration:
    """Test dialect integration with data sources."""

    def test_datasource_with_duckdb_dialect(self) -> None:
        """Test that ArrowDataSource uses DuckDB dialect by default."""
        # Create test data
        table = pa.table({"value": [10, 20, 30, None, 50], "category": ["A", "B", "A", "B", "A"]})

        # Create data source - should use DuckDB dialect by default
        ds = ArrowDataSource(table)

        assert ds.dialect == "duckdb"


# =============================================================================
# Analyzer Integration Tests
# =============================================================================


class TestAnalyzerDialectIntegration:
    """Test dialect integration with analyzer."""

    def test_analyzer_uses_dialect_for_sql_generation(self) -> None:
        """Test that analyzer uses the data source's dialect for SQL generation."""
        # Create test data
        table = pa.table({"value": [10, 20, 30, 40, 50], "category": ["A", "B", "A", "B", "A"]})

        # Create metrics
        avg_metric = specs.Average("value")
        sum_metric = specs.Sum("value")

        # Test with DuckDB dialect (default)
        ds_duckdb = ArrowDataSource(table)
        analyzer = Analyzer()

        key = ResultKey(yyyy_mm_dd=datetime.date(2024, 1, 1), tags={})
        report = analyzer.analyze(ds_duckdb, [avg_metric, sum_metric], key)

        # Results should be computed correctly
        # Check that our specific metrics are in the report with correct values
        assert (avg_metric, key) in report
        assert (sum_metric, key) in report
        assert report[(avg_metric, key)].value == 30.0
        assert report[(sum_metric, key)].value == 150.0

    def test_analyze_sql_ops_with_different_dialects(self) -> None:
        """Test analyze_sql_ops function with different dialects."""
        # Create mock ops
        ops: list[SqlOp] = [NumRows(), Average("price"), NullCount("quantity")]

        # Create mock data source with DuckDB dialect
        ds_duckdb = Mock()
        ds_duckdb.dialect = "duckdb"
        ds_duckdb.cte = "SELECT * FROM orders"

        # Mock the query result - column names from dialect don't have quotes in result keys
        mock_result = {}
        for op in ops:
            mock_result[op.sql_col] = (
                np.array([100.0])
                if isinstance(op, NumRows)
                else np.array([25.5])
                if isinstance(op, Average)
                else np.array([5.0])
            )

        ds_duckdb.query.return_value.fetchnumpy.return_value = mock_result

        # Run analysis
        analyze_sql_ops(ds_duckdb, ops)

        # Verify SQL was generated with DuckDB dialect
        call_args = ds_duckdb.query.call_args[0][0]
        # Remove extra spaces for easier assertion
        normalized_sql = " ".join(call_args.split())
        assert "CAST(COUNT(*) AS DOUBLE)" in normalized_sql
        assert "CAST(AVG(price) AS DOUBLE)" in normalized_sql
        assert "COUNT_IF(quantity IS NULL)" in normalized_sql

        # Store original registry state
        original_registry = _DIALECT_REGISTRY.copy()

        try:
            # Register the mock dialect
            register_dialect("postgresql", MockPostgreSQLDialect)

            # Create mock data source with PostgreSQL dialect
            ds_postgres = Mock()
            ds_postgres.dialect = "postgresql"
            ds_postgres.cte = "SELECT * FROM orders"
            ds_postgres.query.return_value.fetchnumpy.return_value = mock_result

            # Run analysis with PostgreSQL dialect - using same ops to keep same prefixes
            analyze_sql_ops(ds_postgres, ops)

            # Verify SQL was generated with PostgreSQL dialect
            call_args = ds_postgres.query.call_args[0][0]
            # Remove extra spaces for easier assertion
            normalized_sql = " ".join(call_args.split())
            assert "COUNT(*)::FLOAT8" in normalized_sql
            assert "AVG(price)::FLOAT8" in normalized_sql
            assert "COUNT(CASE WHEN quantity IS NULL THEN 1 END)" in normalized_sql

        finally:
            # Restore original registry state
            _DIALECT_REGISTRY.clear()
            _DIALECT_REGISTRY.update(original_registry)


# =============================================================================
# Query Formatting Tests
# =============================================================================


class TestDialectQueryFormatting:
    """Test query formatting functionality."""

    def test_dialect_query_formatting(self) -> None:
        """Test the beautiful query formatting from dialects."""
        dialect = DuckDBDialect()

        cte_sql = "SELECT * FROM sales WHERE date > '2024-01-01'"
        expressions = [
            dialect.translate_sql_op(NumRows()),
            dialect.translate_sql_op(Average("price")),
            dialect.translate_sql_op(Sum("quantity")),
            dialect.translate_sql_op(NullCount("customer_id")),
        ]

        query = dialect.build_cte_query(cte_sql, expressions)

        # Check query structure
        assert "WITH source AS (" in query
        assert cte_sql in query
        assert "SELECT" in query
        assert "FROM source" in query

        # Check alignment - all commas should be aligned
        lines = query.split("\n")
        select_lines = [line for line in lines if line.strip().startswith(",") or line.strip().startswith("CAST")]

        # All continuation lines should start with comma at same position
        comma_positions = [line.find(",") for line in select_lines if "," in line]
        if comma_positions:
            assert len(set(comma_positions)) == 1, "Commas should be aligned"

    def test_build_cte_query_helper_function(self) -> None:
        """Test the build_cte_query helper function directly."""
        cte_sql = "SELECT id, name, price FROM products"
        expressions = ["COUNT(*) AS total", "AVG(price) AS average_price", "MAX(price) AS max_price"]

        query = build_cte_query(cte_sql, expressions)

        # Verify structure
        assert query.startswith("WITH source AS (")
        assert cte_sql in query
        assert "SELECT" in query
        assert "FROM source" in query

        # Verify all expressions are included (checking parts due to formatting)
        assert "COUNT(*)" in query and "AS total" in query
        assert "AVG(price)" in query and "AS average_price" in query
        assert "MAX(price)" in query and "AS max_price" in query
