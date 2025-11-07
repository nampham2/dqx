"""Test BigQuery dialect implementation."""

from typing import Any

from dqx.dialect import BigQueryDialect
from dqx.ops import Average, Maximum, Minimum, NumRows, Sum


class TestBigQueryDialect:
    """Test BigQuery dialect implementation."""

    def test_bigquery_dialect_name(self) -> None:
        """Test dialect name property."""
        dialect = BigQueryDialect()
        assert dialect.name == "bigquery"

    def test_translate_num_rows(self) -> None:
        """Test NumRows translation to BigQuery SQL."""
        dialect = BigQueryDialect()
        op = NumRows()
        sql = dialect.translate_sql_op(op)
        assert sql == f"CAST(COUNT(*) AS FLOAT64) AS `{op.sql_col}`"

    def test_translate_average(self) -> None:
        """Test Average translation to BigQuery SQL."""
        dialect = BigQueryDialect()
        op = Average("revenue")
        sql = dialect.translate_sql_op(op)
        assert sql == f"CAST(AVG(revenue) AS FLOAT64) AS `{op.sql_col}`"

    def test_translate_sum(self) -> None:
        """Test Sum translation to BigQuery SQL."""
        dialect = BigQueryDialect()
        op = Sum("quantity")
        sql = dialect.translate_sql_op(op)
        assert sql == f"CAST(SUM(quantity) AS FLOAT64) AS `{op.sql_col}`"

    def test_translate_minimum(self) -> None:
        """Test Minimum translation to BigQuery SQL."""
        dialect = BigQueryDialect()
        op = Minimum("price")
        sql = dialect.translate_sql_op(op)
        assert sql == f"CAST(MIN(price) AS FLOAT64) AS `{op.sql_col}`"

    def test_translate_maximum(self) -> None:
        """Test Maximum translation to BigQuery SQL."""
        dialect = BigQueryDialect()
        op = Maximum("score")
        sql = dialect.translate_sql_op(op)
        assert sql == f"CAST(MAX(score) AS FLOAT64) AS `{op.sql_col}`"

    def test_translate_variance(self) -> None:
        """Test Variance translation to BigQuery SQL."""
        from dqx.ops import Variance

        dialect = BigQueryDialect()
        op = Variance("sales")
        sql = dialect.translate_sql_op(op)
        assert sql == f"CAST(VAR_SAMP(sales) AS FLOAT64) AS `{op.sql_col}`"

    def test_translate_first(self) -> None:
        """Test First translation using MIN for deterministic results."""
        from dqx.ops import First

        dialect = BigQueryDialect()
        op = First("timestamp")
        sql = dialect.translate_sql_op(op)
        # Using MIN for deterministic "first" value
        assert sql == f"CAST(MIN(timestamp) AS FLOAT64) AS `{op.sql_col}`"

    def test_translate_null_count(self) -> None:
        """Test NullCount translation to BigQuery SQL."""
        from dqx.ops import NullCount

        dialect = BigQueryDialect()
        op = NullCount("email")
        sql = dialect.translate_sql_op(op)
        assert sql == f"CAST(COUNTIF(email IS NULL) AS FLOAT64) AS `{op.sql_col}`"

    def test_translate_negative_count(self) -> None:
        """Test NegativeCount translation to BigQuery SQL."""
        from dqx.ops import NegativeCount

        dialect = BigQueryDialect()
        op = NegativeCount("profit")
        sql = dialect.translate_sql_op(op)
        assert sql == f"CAST(COUNTIF(profit < 0) AS FLOAT64) AS `{op.sql_col}`"

    def test_translate_duplicate_count_single_column(self) -> None:
        """Test DuplicateCount translation for single column."""
        from dqx.ops import DuplicateCount

        dialect = BigQueryDialect()
        op = DuplicateCount(["user_id"])
        sql = dialect.translate_sql_op(op)
        assert sql == f"CAST(COUNT(*) - COUNT(DISTINCT user_id) AS FLOAT64) AS `{op.sql_col}`"

    def test_translate_duplicate_count_multiple_columns(self) -> None:
        """Test DuplicateCount translation for multiple columns."""
        from dqx.ops import DuplicateCount

        dialect = BigQueryDialect()
        op = DuplicateCount(["user_id", "product_id"])
        sql = dialect.translate_sql_op(op)
        # Columns should be sorted in DuplicateCount
        assert sql == f"CAST(COUNT(*) - COUNT(DISTINCT (product_id, user_id)) AS FLOAT64) AS `{op.sql_col}`"

    def test_translate_unsupported_op(self) -> None:
        """Test error handling for unsupported operations."""
        import pytest

        from dqx.ops import SqlOp

        dialect = BigQueryDialect()

        class UnsupportedOp(SqlOp):
            @property
            def name(self) -> str:
                return "unsupported"

            @property
            def prefix(self) -> str:
                return "unsup"

            @property
            def sql_col(self) -> str:
                return "unsup_col"

            @property
            def parameters(self) -> dict[str, Any]:
                return {}

            def value(self) -> float:
                return 0.0

            def assign(self, value: float) -> None:
                pass

            def clear(self) -> None:
                pass

        op = UnsupportedOp()
        with pytest.raises(ValueError, match="Unsupported SqlOp type: UnsupportedOp"):
            dialect.translate_sql_op(op)

    def test_build_batch_cte_query_empty(self) -> None:
        """Test error handling for empty CTE data."""
        import pytest

        dialect = BigQueryDialect()
        with pytest.raises(ValueError, match="No CTE data provided"):
            dialect.build_cte_query([])

    def test_build_batch_cte_query_single_date(self) -> None:
        """Test batch CTE query with single date."""
        from datetime import date

        from dqx.common import ResultKey
        from dqx.dialect import BatchCTEData
        from dqx.ops import Average, NumRows

        dialect = BigQueryDialect()
        key = ResultKey(yyyy_mm_dd=date(2024, 1, 1), tags={})
        ops: list[Any] = [NumRows(), Average("revenue")]
        cte_data = [BatchCTEData(key=key, cte_sql="SELECT * FROM sales", ops=ops)]

        query = dialect.build_cte_query(cte_data)

        # Should contain:
        # - WITH clause with source and metrics CTEs
        # - SELECT with date and array of STRUCTs
        assert "WITH" in query
        assert "source_2024_01_01_0" in query
        assert "metrics_2024_01_01_0" in query
        assert "'2024-01-01' as date" in query
        # Check for array format
        assert "[STRUCT(" in query
        assert f"'{ops[0].sql_col}' AS key" in query
        assert f"'{ops[1].sql_col}' AS key" in query
        assert f"`{ops[0].sql_col}` AS value" in query
        assert f"`{ops[1].sql_col}` AS value" in query

    def test_build_batch_cte_query_multiple_dates(self) -> None:
        """Test batch CTE query with multiple dates."""
        from datetime import date

        from dqx.common import ResultKey
        from dqx.dialect import BatchCTEData
        from dqx.ops import Average, Maximum, Minimum, NumRows

        dialect = BigQueryDialect()

        # Create data for two dates
        key1 = ResultKey(yyyy_mm_dd=date(2024, 1, 1), tags={})
        ops1: list[Any] = [NumRows(), Average("revenue")]
        cte_data1 = BatchCTEData(key=key1, cte_sql="SELECT * FROM sales WHERE date='2024-01-01'", ops=ops1)

        key2 = ResultKey(yyyy_mm_dd=date(2024, 1, 2), tags={})
        ops2: list[Any] = [Minimum("price"), Maximum("price")]
        cte_data2 = BatchCTEData(key=key2, cte_sql="SELECT * FROM sales WHERE date='2024-01-02'", ops=ops2)

        query = dialect.build_cte_query([cte_data1, cte_data2])

        # Should contain both dates
        assert "'2024-01-01' as date" in query
        assert "'2024-01-02' as date" in query
        assert "UNION ALL" in query
        # Check for multiple CTEs
        assert "source_2024_01_01_0" in query
        assert "source_2024_01_02_1" in query
        assert "metrics_2024_01_01_0" in query
        assert "metrics_2024_01_02_1" in query

    def test_build_batch_cte_query_no_ops(self) -> None:
        """Test batch CTE query when no ops provided (should not create metrics CTE)."""
        from datetime import date

        import pytest

        from dqx.common import ResultKey
        from dqx.dialect import BatchCTEData

        dialect = BigQueryDialect()
        key = ResultKey(yyyy_mm_dd=date(2024, 1, 1), tags={})
        cte_data = [BatchCTEData(key=key, cte_sql="SELECT * FROM sales", ops=[])]

        with pytest.raises(ValueError, match="No metrics to compute"):
            dialect.build_cte_query(cte_data)

    def test_register_bigquery_dialect(self) -> None:
        """Test BigQuery dialect registration."""
        from dqx.dialect import _DIALECT_REGISTRY, get_dialect

        # BigQuery should already be registered by the module
        assert "bigquery" in _DIALECT_REGISTRY

        # Get an instance
        dialect = get_dialect("bigquery")
        assert isinstance(dialect, BigQueryDialect)
        assert dialect.name == "bigquery"

    def test_register_duplicate_dialect_error(self, isolated_dialect_registry: dict[str, type]) -> None:
        """Test error when registering duplicate dialect."""
        import pytest

        from dqx.dialect import register_dialect

        # First registration should succeed
        register_dialect("test_dialect", BigQueryDialect)

        # Second registration with same name should fail
        with pytest.raises(ValueError, match="Dialect 'test_dialect' is already registered"):
            register_dialect("test_dialect", BigQueryDialect)

    def test_get_unregistered_dialect_error(self) -> None:
        """Test error when getting unregistered dialect."""
        import pytest

        from dqx.common import DQXError
        from dqx.dialect import get_dialect

        with pytest.raises(DQXError, match="Dialect 'nonexistent' not found in registry"):
            get_dialect("nonexistent")
