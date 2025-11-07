"""Test batch query generation with parameters."""

import datetime
from typing import Type

import pytest

from dqx.common import ResultKey
from dqx.dialect import BatchCTEData, BigQueryDialect, DuckDBDialect
from dqx.ops import Average, CustomSQL, Sum


@pytest.mark.parametrize("dialect_class", [DuckDBDialect, BigQueryDialect])
def test_batch_query_groups_by_parameters(dialect_class: Type[DuckDBDialect] | Type[BigQueryDialect]) -> None:
    """Batch queries should group operations by parameters."""
    dialect = dialect_class()

    # Create test data with different parameter groups
    key = ResultKey(yyyy_mm_dd=datetime.date(2024, 1, 1), tags={})

    cte_data = [
        BatchCTEData(
            key=key,
            cte_sql="SELECT * FROM sales WHERE date = '2024-01-01'",
            ops=[
                Average("price", parameters={"region": "US"}),
                Sum("amount", parameters={"region": "US"}),
                Average("price", parameters={"region": "EU"}),
                CustomSQL("COUNT(DISTINCT user_id)"),
            ],
        )
    ]

    sql = dialect.build_cte_query(cte_data)

    # Should have multiple source CTEs for different parameter groups
    assert "source_2024_01_01_0_0" in sql  # First parameter group
    assert "source_2024_01_01_0_1" in sql  # Second parameter group
    assert "source_2024_01_01_0_2" in sql  # Third parameter group (empty params)

    # Should have corresponding metrics CTEs
    assert "metrics_2024_01_01_0_0" in sql
    assert "metrics_2024_01_01_0_1" in sql
    assert "metrics_2024_01_01_0_2" in sql


def test_parameter_aware_batch_sql_generation() -> None:
    """Test full SQL generation with parameter grouping."""
    dialect = DuckDBDialect()

    # Multiple dates with different operations
    cte_data = [
        BatchCTEData(
            key=ResultKey(yyyy_mm_dd=datetime.date(2024, 1, 1), tags={}),
            cte_sql="SELECT * FROM sales WHERE date = '2024-01-01'",
            ops=[
                Average("revenue", parameters={"segment": "enterprise"}),
                Sum("revenue", parameters={"segment": "enterprise"}),
                Average("revenue", parameters={"segment": "smb"}),
            ],
        ),
        BatchCTEData(
            key=ResultKey(yyyy_mm_dd=datetime.date(2024, 1, 2), tags={}),
            cte_sql="SELECT * FROM sales WHERE date = '2024-01-02'",
            ops=[
                CustomSQL("PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY amount)"),
            ],
        ),
    ]

    sql = dialect.build_cte_query(cte_data)

    # Verify structure
    assert sql.startswith("WITH")
    assert "UNION ALL" in sql
    assert "'2024-01-01' as date" in sql
    assert "'2024-01-02' as date" in sql

    # Verify array format for values
    assert "[{" in sql  # DuckDB array syntax
    assert "'key':" in sql
    assert "'value':" in sql
