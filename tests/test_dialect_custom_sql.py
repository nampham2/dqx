"""Tests for dialect handling of CustomSQL operation."""

from datetime import date

import pytest

from dqx.dialect import BigQueryDialect, DuckDBDialect
from dqx.ops import CustomSQL


def test_dialect_custom_sql_basic() -> None:
    """Test dialect handling of basic CustomSQL."""
    dialect = DuckDBDialect()
    op = CustomSQL("COUNT(*)")

    sql = dialect.translate_sql_op(op)
    # DuckDB uses single quotes and CAST
    assert sql.startswith("CAST((COUNT(*)) AS DOUBLE) AS '")
    assert op.sql_col in sql


def test_dialect_custom_sql_complex() -> None:
    """Test dialect handling of complex CustomSQL."""
    dialect = DuckDBDialect()
    sql_expr = "AVG(CASE WHEN status = 'active' THEN value END)"
    op = CustomSQL(sql_expr)

    sql = dialect.translate_sql_op(op)
    assert sql.startswith(f"CAST(({sql_expr}) AS DOUBLE)")
    assert op.sql_col in sql


def test_dialect_custom_sql_with_cte_parameters() -> None:
    """Test dialect handling of CustomSQL with CTE parameters."""
    dialect = DuckDBDialect()
    cte_params = {"start_date": date(2024, 1, 1), "end_date": date(2024, 12, 31)}

    op = CustomSQL("COUNT(DISTINCT user_id)", cte_params)

    # CTE parameters should be available but not affect the SQL expression
    sql = dialect.translate_sql_op(op)
    assert sql.startswith("CAST((COUNT(DISTINCT user_id)) AS DOUBLE)")
    assert op.parameters == cte_params


def test_bigquery_dialect_custom_sql() -> None:
    """Test BigQuery dialect handling of CustomSQL."""
    dialect = BigQueryDialect()
    op = CustomSQL("PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY amount)")

    sql = dialect.translate_sql_op(op)
    # BigQuery uses backticks and FLOAT64
    assert sql.startswith("CAST((PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY amount)) AS FLOAT64) AS `")
    assert op.sql_col in sql


def test_dialect_custom_sql_various_expressions() -> None:
    """Test various SQL expressions with both dialects."""
    expressions = [
        "COUNT(DISTINCT user_id)",
        "SUM(amount) FILTER (WHERE status = 'active')",
        "ARRAY_AGG(DISTINCT category)",
        "STRING_AGG(name, ', ' ORDER BY id)",
    ]

    duckdb_dialect = DuckDBDialect()
    bigquery_dialect = BigQueryDialect()

    for expr in expressions:
        op = CustomSQL(expr)

        # DuckDB
        duckdb_sql = duckdb_dialect.translate_sql_op(op)
        assert f"({expr})" in duckdb_sql
        assert "AS DOUBLE" in duckdb_sql

        # BigQuery
        bigquery_sql = bigquery_dialect.translate_sql_op(op)
        assert f"({expr})" in bigquery_sql
        assert "AS FLOAT64" in bigquery_sql


@pytest.mark.parametrize("dialect_class", [DuckDBDialect, BigQueryDialect])
def test_dialect_custom_sql_with_params_grouping(dialect_class: type[DuckDBDialect] | type[BigQueryDialect]) -> None:
    """Test that CustomSQL operations group correctly by parameters."""
    dialect = dialect_class()

    ops1 = [
        CustomSQL("COUNT(*)", {"region": "US"}),
        CustomSQL("SUM(amount)", {"region": "US"}),
        CustomSQL("AVG(price)", {"region": "EU"}),
    ]

    # Test grouping method if available
    if hasattr(dialect, "_group_operations_by_parameters"):
        groups = dialect._group_operations_by_parameters(ops1)
        assert len(groups) == 2  # US and EU groups

        # Check US group
        us_key = (("region", "US"),)
        assert us_key in groups
        assert len(groups[us_key]) == 2

        # Check EU group
        eu_key = (("region", "EU"),)
        assert eu_key in groups
        assert len(groups[eu_key]) == 1
