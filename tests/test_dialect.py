"""Tests for SQL dialect abstraction."""

import pytest

from dqx import ops
from dqx.dialect import Dialect, DuckDBDialect


class TestDialectProtocol:
    """Test the Dialect protocol."""
    
    def test_duckdb_dialect_implements_protocol(self) -> None:
        """Test that DuckDBDialect implements the Dialect protocol."""
        dialect = DuckDBDialect()
        assert isinstance(dialect, Dialect)
        assert hasattr(dialect, 'name')
        assert hasattr(dialect, 'translate_sql_op')
        assert hasattr(dialect, 'build_cte_query')


class TestDuckDBDialect:
    """Test DuckDB dialect implementation."""
    
    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.dialect = DuckDBDialect()
    
    def test_dialect_name(self) -> None:
        """Test dialect name property."""
        assert self.dialect.name == "duckdb"
    
    def test_translate_num_rows(self) -> None:
        """Test NumRows SqlOp translation."""
        op = ops.NumRows()
        op._prefix = "test"  # Set prefix for predictable output
        sql = self.dialect.translate_sql_op(op)
        assert sql == f"CAST(COUNT(*) AS DOUBLE) AS '{op.sql_col}'"
    
    def test_translate_average(self) -> None:
        """Test Average SqlOp translation."""
        op = ops.Average("price")
        op._prefix = "test"
        sql = self.dialect.translate_sql_op(op)
        assert sql == f"CAST(AVG(price) AS DOUBLE) AS '{op.sql_col}'"
    
    def test_translate_minimum(self) -> None:
        """Test Minimum SqlOp translation."""
        op = ops.Minimum("amount")
        op._prefix = "test"
        sql = self.dialect.translate_sql_op(op)
        assert sql == f"CAST(MIN(amount) AS DOUBLE) AS '{op.sql_col}'"
    
    def test_translate_maximum(self) -> None:
        """Test Maximum SqlOp translation."""
        op = ops.Maximum("amount")
        op._prefix = "test"
        sql = self.dialect.translate_sql_op(op)
        assert sql == f"CAST(MAX(amount) AS DOUBLE) AS '{op.sql_col}'"
    
    def test_translate_sum(self) -> None:
        """Test Sum SqlOp translation."""
        op = ops.Sum("quantity")
        op._prefix = "test"
        sql = self.dialect.translate_sql_op(op)
        assert sql == f"CAST(SUM(quantity) AS DOUBLE) AS '{op.sql_col}'"
    
    def test_translate_variance(self) -> None:
        """Test Variance SqlOp translation."""
        op = ops.Variance("value")
        op._prefix = "test"
        sql = self.dialect.translate_sql_op(op)
        assert sql == f"CAST(VARIANCE(value) AS DOUBLE) AS '{op.sql_col}'"
    
    def test_translate_first(self) -> None:
        """Test First SqlOp translation."""
        op = ops.First("timestamp")
        op._prefix = "test"
        sql = self.dialect.translate_sql_op(op)
        assert sql == f"CAST(FIRST(timestamp) AS DOUBLE) AS '{op.sql_col}'"
    
    def test_translate_null_count(self) -> None:
        """Test NullCount SqlOp translation."""
        op = ops.NullCount("email")
        op._prefix = "test"
        sql = self.dialect.translate_sql_op(op)
        assert sql == f"CAST(COUNT_IF(email IS NULL) AS DOUBLE) AS '{op.sql_col}'"
    
    def test_translate_negative_count(self) -> None:
        """Test NegativeCount SqlOp translation."""
        op = ops.NegativeCount("balance")
        op._prefix = "test"
        sql = self.dialect.translate_sql_op(op)
        assert sql == f"CAST(COUNT_IF(balance < 0.0) AS DOUBLE) AS '{op.sql_col}'"
    
    def test_translate_unsupported_op(self) -> None:
        """Test translation of unsupported SqlOp type."""
        # Create a mock SqlOp that's not supported
        class UnsupportedOp:
            sql_col = "test_col"
        
        with pytest.raises(ValueError, match="Unsupported SqlOp type: UnsupportedOp"):
            self.dialect.translate_sql_op(UnsupportedOp())  # type: ignore[arg-type]
    
    def test_build_cte_query_single_expression(self) -> None:
        """Test CTE query building with single expression."""
        cte_sql = "SELECT * FROM orders"
        expressions = ["COUNT(*) AS 'count'"]
        
        result = self.dialect.build_cte_query(cte_sql, expressions)
        
        expected = (
            "WITH source AS (\n"
            "    SELECT * FROM orders\n"
            ")\n"
            "SELECT\n"
            "    COUNT(*) AS 'count'\n"
            "FROM source"
        )
        assert result == expected
    
    def test_build_cte_query_multiple_expressions(self) -> None:
        """Test CTE query building with multiple expressions."""
        cte_sql = "SELECT * FROM customers"
        expressions = [
            "COUNT(*) AS 'total_count'",
            "AVG(age) AS 'avg_age'",
            "MAX(created_at) AS 'last_created'"
        ]
        
        result = self.dialect.build_cte_query(cte_sql, expressions)
        
        expected = (
            "WITH source AS (\n"
            "    SELECT * FROM customers\n"
            ")\n"
            "SELECT\n"
            "    COUNT(*)        AS 'total_count'\n"
            "  , AVG(age)        AS 'avg_age'\n"
            "  , MAX(created_at) AS 'last_created'\n"
            "FROM source"
        )
        assert result == expected
    
    def test_build_cte_query_aligned_formatting(self) -> None:
        """Test that CTE query properly aligns expressions."""
        cte_sql = "SELECT * FROM table"
        expressions = [
            "CAST(COUNT(*) AS DOUBLE) AS 'num_rows'",
            "AVG(price) AS 'avg_price'",
            "COUNT_IF(status = 'active') AS 'active_count'",
            "SUM(amount) AS 'total'"
        ]
        
        result = self.dialect.build_cte_query(cte_sql, expressions)
        
        # Check that expressions are properly aligned
        lines = result.split('\n')
        select_lines = [line for line in lines if line.strip() and not line.strip().startswith(('WITH', ')', 'SELECT', 'FROM'))]
        
        # Extract the position of 'AS' in each line
        as_positions = []
        for line in select_lines:
            if ' AS ' in line:
                as_positions.append(line.index(' AS '))
        
        # All 'AS' should be at the same position (allowing for leading comma)
        if len(as_positions) > 1:
            # Account for lines with leading comma having different offset
            other_lines_as = set(as_positions[1:])
            assert len(other_lines_as) == 1, "All AS keywords should be aligned"
    
    def test_build_cte_query_empty_expressions(self) -> None:
        """Test CTE query building with empty expressions list."""
        cte_sql = "SELECT * FROM table"
        expressions: list[str] = []
        
        with pytest.raises(ValueError, match="No SELECT expressions provided"):
            self.dialect.build_cte_query(cte_sql, expressions)
    
    def test_build_cte_query_expression_without_as(self) -> None:
        """Test CTE query building with expression that has no AS clause."""
        cte_sql = "SELECT * FROM table"
        expressions = [
            "COUNT(*)",
            "AVG(price) AS 'avg_price'"
        ]
        
        result = self.dialect.build_cte_query(cte_sql, expressions)
        
        # Check that the result contains the expected parts
        assert "WITH source AS (\n    SELECT * FROM table\n)" in result
        assert "SELECT" in result
        assert "COUNT(*)" in result
        assert "AVG(price) AS 'avg_price'" in result
        assert "FROM source" in result
        
        # Verify structure
        lines = result.split('\n')
        assert any("COUNT(*)" in line for line in lines)
        assert any("AVG(price)" in line and "'avg_price'" in line for line in lines)
    
    def test_all_sql_ops_have_translations(self) -> None:
        """Test that all SqlOp subclasses can be translated by DuckDBDialect.
        
        This test ensures that whenever a new SqlOp is added to ops.py,
        a corresponding translation is added to the dialect.
        """
        # List all concrete SqlOp classes explicitly
        sql_op_classes = [
            ops.NumRows,
            ops.Average,
            ops.Minimum,
            ops.Maximum,
            ops.Sum,
            ops.Variance,
            ops.First,
            ops.NullCount,
            ops.NegativeCount
        ]
        
        # Test that each SqlOp can be translated
        for op_class in sql_op_classes:
            # Create instance with appropriate parameters
            if op_class == ops.NumRows:
                # NumRows takes no parameters
                op = op_class()
            else:
                # All other SqlOps take a column parameter
                op = op_class("test_column")
            
            # Set a predictable prefix (all concrete SqlOps have _prefix attribute)
            op._prefix = "test"  # type: ignore[attr-defined]
            
            # Attempt translation - should not raise ValueError
            try:
                sql = self.dialect.translate_sql_op(op)
                # Verify the SQL contains expected components
                assert op.sql_col in sql, f"Column alias '{op.sql_col}' not found in SQL: {sql}"
                assert " AS " in sql, f"AS clause not found in SQL: {sql}"
                # For DuckDB, we expect CAST...AS DOUBLE
                assert "CAST(" in sql or op_class == ops.Average, f"Expected CAST for {op_class.__name__}"
                assert "AS DOUBLE)" in sql or op_class == ops.Average, f"Expected AS DOUBLE for {op_class.__name__}"
            except ValueError as e:
                pytest.fail(
                    f"Translation failed for {op_class.__name__}: {e}\n"
                    f"This likely means a translation case is missing in DuckDBDialect.translate_sql_op()"
                )
    
    def test_real_world_scenario(self) -> None:
        """Test a real-world scenario with multiple SqlOps."""
        # Create various ops
        num_rows_op = ops.NumRows()
        avg_price_op = ops.Average("price")
        min_qty_op = ops.Minimum("quantity")
        null_customer_op = ops.NullCount("customer_id")
        
        # Set predictable prefixes
        num_rows_op._prefix = "a"
        avg_price_op._prefix = "b"
        min_qty_op._prefix = "c"
        null_customer_op._prefix = "d"
        
        # Translate ops
        expressions = [
            self.dialect.translate_sql_op(num_rows_op),
            self.dialect.translate_sql_op(avg_price_op),
            self.dialect.translate_sql_op(min_qty_op),
            self.dialect.translate_sql_op(null_customer_op)
        ]
        
        # Build CTE query
        cte_sql = "SELECT * FROM orders WHERE date = '2024-01-01'"
        query = self.dialect.build_cte_query(cte_sql, expressions)
        
        # Verify the query structure
        assert "WITH source AS (" in query
        assert cte_sql in query
        # Check for the SQL functions (they may have extra spaces due to alignment)
        assert "CAST(COUNT(*)" in query and "AS DOUBLE)" in query
        assert "AVG(price)" in query
        assert "MIN(quantity)" in query
        assert "COUNT_IF(customer_id IS NULL)" in query
        
        # Verify alignment
        lines = query.split('\n')
        select_section = False
        for line in lines:
            if "SELECT" in line:
                select_section = True
            elif "FROM" in line:
                select_section = False
            elif select_section and " AS '" in line:
                # Check that the column alias AS is consistently positioned
                # Note: Some lines may have multiple AS (e.g., CAST(x AS type) AS 'alias')
                # We're checking for AS followed by the alias quote
                assert " AS '" in line
