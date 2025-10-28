"""Test cases for UniqueCount SQL dialect implementation."""

from dqx import ops
from dqx.dialect import BigQueryDialect, DuckDBDialect


class TestUniqueCountSQLDialect:
    """Test UniqueCount SQL translation."""

    def test_unique_count_sql_duckdb(self) -> None:
        """Test UniqueCount SQL generation for DuckDB."""
        op = ops.UniqueCount("product_id")
        dialect = DuckDBDialect()

        sql = dialect.translate_sql_op(op)
        expected = f"CAST(COUNT(DISTINCT product_id) AS DOUBLE) AS '{op.sql_col}'"
        assert sql == expected

    def test_unique_count_sql_bigquery(self) -> None:
        """Test UniqueCount SQL generation for BigQuery."""
        op = ops.UniqueCount("customer_id")
        dialect = BigQueryDialect()

        sql = dialect.translate_sql_op(op)
        expected = f"CAST(COUNT(DISTINCT customer_id) AS FLOAT64) AS `{op.sql_col}`"
        assert sql == expected

    def test_unique_count_sql_with_special_chars(self) -> None:
        """Test UniqueCount SQL with column names containing special characters."""
        op = ops.UniqueCount("product-id")
        dialect = DuckDBDialect()

        sql = dialect.translate_sql_op(op)
        expected = f"CAST(COUNT(DISTINCT product-id) AS DOUBLE) AS '{op.sql_col}'"
        assert sql == expected

    def test_unique_count_repr(self) -> None:
        """Test UniqueCount string representation."""
        op = ops.UniqueCount("session_id")
        # The repr returns the name property
        assert repr(op) == "unique_count(session_id)"
        assert str(op) == "unique_count(session_id)"

    def test_unique_count_name(self) -> None:
        """Test UniqueCount name format."""
        op = ops.UniqueCount("visitor_id")
        assert op.name == "unique_count(visitor_id)"

    def test_unique_count_sql_consistency(self) -> None:
        """Test that UniqueCount SQL is consistent across similar operations."""
        op1 = ops.UniqueCount("product_id")
        op2 = ops.UniqueCount("user_id")

        dialect = DuckDBDialect()

        sql1 = dialect.translate_sql_op(op1)
        sql2 = dialect.translate_sql_op(op2)

        # Check SQL structure
        assert "COUNT(DISTINCT product_id)" in sql1
        assert "COUNT(DISTINCT user_id)" in sql2
        assert "CAST(" in sql1
        assert "AS DOUBLE)" in sql1
        assert sql1 != sql2  # Different columns produce different SQL

    def test_unique_count_sql_col_property(self) -> None:
        """Test that UniqueCount has proper sql_col property."""
        op = ops.UniqueCount("test_column")
        # sql_col should include the operation name
        assert "unique_count(test_column)" in op.sql_col
        # It should have a prefix for uniqueness
        assert op.sql_col.startswith("_")
        assert op.sql_col.endswith("_unique_count(test_column)")
