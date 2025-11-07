"""Tests for parameter grouping in dialect with freeze_for_hashing."""

import datetime
from typing import Any

from dqx.common import ResultKey
from dqx.dialect import BatchCTEData, DuckDBDialect
from dqx.ops import Average, CountValues, CustomSQL, NumRows, SqlOp, Sum


class TestDialectParameterGrouping:
    """Test dialect's ability to group operations by parameters."""

    def test_group_ops_by_parameters_simple(self) -> None:
        """Test grouping operations with simple parameters."""
        ops: list[SqlOp[Any]] = [
            NumRows(parameters={"region": "US"}),
            Average("price", parameters={"region": "US"}),
            Sum("quantity", parameters={"region": "EU"}),
            NumRows(parameters={"region": "EU"}),
        ]

        batch_data = BatchCTEData(key=ResultKey(datetime.date(2024, 1, 1), {}), cte_sql="SELECT * FROM sales", ops=ops)

        grouped = batch_data.group_ops_by_parameters()

        # Should have 2 groups: US and EU
        assert len(grouped) == 2

        # Check that each group has the right ops
        for param_tuple, ops_list in grouped.items():
            if param_tuple == (("region", "US"),):
                assert len(ops_list) == 2
                assert any(isinstance(op, NumRows) for op in ops_list)
                assert any(isinstance(op, Average) for op in ops_list)
            elif param_tuple == (("region", "EU"),):
                assert len(ops_list) == 2
                assert any(isinstance(op, Sum) for op in ops_list)
                assert any(isinstance(op, NumRows) for op in ops_list)

    def test_group_ops_by_parameters_with_lists(self) -> None:
        """Test grouping operations with list parameters."""
        ops: list[SqlOp[Any]] = [
            Average("price", parameters={"categories": ["A", "B"]}),
            Sum("quantity", parameters={"categories": ["A", "B"]}),
            NumRows(parameters={"categories": ["C", "D"]}),
        ]

        batch_data = BatchCTEData(
            key=ResultKey(datetime.date(2024, 1, 1), {}), cte_sql="SELECT * FROM products", ops=ops
        )

        grouped = batch_data.group_ops_by_parameters()

        # Should have 2 groups
        assert len(grouped) == 2

        # Find the group with ["A", "B"]
        ab_group = None
        cd_group = None
        for param_tuple, ops_list in grouped.items():
            if param_tuple == (("categories", ("A", "B")),):
                ab_group = ops_list
            elif param_tuple == (("categories", ("C", "D")),):
                cd_group = ops_list

        assert ab_group is not None
        assert len(ab_group) == 2
        assert any(isinstance(op, Average) for op in ab_group)
        assert any(isinstance(op, Sum) for op in ab_group)

        assert cd_group is not None
        assert len(cd_group) == 1
        assert isinstance(cd_group[0], NumRows)

    def test_group_ops_empty_parameters(self) -> None:
        """Test grouping operations with empty and None parameters."""
        ops: list[SqlOp[Any]] = [
            NumRows(),  # No parameters
            Average("price"),  # No parameters
            Sum("quantity", parameters={}),  # Empty dict
            NumRows(parameters=None),  # Explicitly None
        ]

        batch_data = BatchCTEData(key=ResultKey(datetime.date(2024, 1, 1), {}), cte_sql="SELECT * FROM data", ops=ops)

        grouped = batch_data.group_ops_by_parameters()

        # All should be in one group (empty parameters)
        assert len(grouped) == 1
        assert () in grouped
        assert len(grouped[()]) == 4

    def test_group_ops_complex_parameters(self) -> None:
        """Test grouping with complex nested parameters."""
        ops: list[SqlOp[Any]] = [
            CustomSQL("COUNT(*)", parameters={"filter": {"region": "US", "active": True}}),
            Average("price", parameters={"filter": {"active": True, "region": "US"}}),  # Same as above, different order
            Sum("total", parameters={"filter": {"region": "EU", "active": False}}),
        ]

        batch_data = BatchCTEData(
            key=ResultKey(datetime.date(2024, 1, 1), {}), cte_sql="SELECT * FROM complex", ops=ops
        )

        grouped = batch_data.group_ops_by_parameters()

        # Should have 2 groups
        assert len(grouped) == 2

        # First two ops should be grouped together despite dict order difference
        for param_tuple, ops_list in grouped.items():
            if len(ops_list) == 2:
                assert any(isinstance(op, CustomSQL) for op in ops_list)
                assert any(isinstance(op, Average) for op in ops_list)
            elif len(ops_list) == 1:
                assert isinstance(ops_list[0], Sum)

    def test_deterministic_grouping(self) -> None:
        """Test that grouping is deterministic across multiple runs."""
        ops: list[SqlOp[Any]] = [
            NumRows(parameters={"b": 2, "a": 1}),
            Average("col", parameters={"a": 1, "b": 2}),
            Sum("col", parameters={"x": [3, 2, 1]}),
            CustomSQL("MAX(col)", parameters={"x": [1, 2, 3]}),  # Different list order
        ]

        batch_data = BatchCTEData(key=ResultKey(datetime.date(2024, 1, 1), {}), cte_sql="SELECT * FROM test", ops=ops)

        # Run grouping multiple times
        results = []
        for _ in range(5):
            grouped = batch_data.group_ops_by_parameters()
            # Convert to a hashable representation for comparison
            result_repr = []
            for param_tuple, ops_list in grouped.items():
                ops_types = sorted(type(op).__name__ for op in ops_list)
                result_repr.append((param_tuple, tuple(ops_types)))
            results.append(tuple(sorted(result_repr)))

        # All results should be identical
        assert all(r == results[0] for r in results[1:])

    def test_dialect_batch_cte_query_with_parameters(self) -> None:
        """Test that dialect generates correct SQL with parameter grouping."""
        dialect = DuckDBDialect()

        ops: list[SqlOp[Any]] = [
            NumRows(parameters={"region": "US"}),
            Average("price", parameters={"region": "US"}),
            Sum("quantity", parameters={"region": "EU"}),
        ]

        batch_data = BatchCTEData(key=ResultKey(datetime.date(2024, 1, 1), {}), cte_sql="SELECT * FROM sales", ops=ops)

        # Create mock data source
        class MockDataSource:
            dialect = "duckdb"

            @property
            def name(self) -> str:
                return "test_datasource"

            @property
            def skip_dates(self) -> set[datetime.date]:
                return set()

            def cte(self, date: datetime.date, params: dict[str, str] | None = None) -> str:
                if params and params.get("region") == "US":
                    return "SELECT * FROM sales WHERE region = 'US'"
                elif params and params.get("region") == "EU":
                    return "SELECT * FROM sales WHERE region = 'EU'"
                return "SELECT * FROM sales"

            def query(self, query: str) -> Any:
                # Mock implementation
                return None

        sql = dialect.build_cte_query([batch_data], MockDataSource())

        # Should have two source CTEs (one for each parameter set)
        assert "source_2024_01_01_0_0" in sql
        assert "source_2024_01_01_0_1" in sql

        # Should have two metrics CTEs
        assert "metrics_2024_01_01_0_0" in sql
        assert "metrics_2024_01_01_0_1" in sql

        # Check that region filtering is applied
        assert "WHERE region = 'US'" in sql
        assert "WHERE region = 'EU'" in sql

    def test_count_values_with_parameters(self) -> None:
        """Test CountValues operations with parameters group correctly."""
        ops: list[SqlOp[Any]] = [
            CountValues("status", "active", parameters={"dept": "sales"}),
            CountValues("type", ["A", "B"], parameters={"dept": "sales"}),
            CountValues("status", "inactive", parameters={"dept": "eng"}),
        ]

        batch_data = BatchCTEData(
            key=ResultKey(datetime.date(2024, 1, 1), {}), cte_sql="SELECT * FROM records", ops=ops
        )

        grouped = batch_data.group_ops_by_parameters()

        # Should have 2 groups
        assert len(grouped) == 2

        for param_tuple, ops_list in grouped.items():
            if param_tuple == (("dept", "sales"),):
                assert len(ops_list) == 2
            elif param_tuple == (("dept", "eng"),):
                assert len(ops_list) == 1

    def test_parameter_grouping_preserves_order(self) -> None:
        """Test that ops within a parameter group maintain their relative order."""
        ops: list[SqlOp[Any]] = [
            NumRows(parameters={"x": 1}),  # 0
            Average("a", parameters={"x": 2}),  # 1
            Sum("b", parameters={"x": 1}),  # 2
            CustomSQL("MIN(c)", parameters={"x": 2}),  # 3
            CountValues("d", 5, parameters={"x": 1}),  # 4
        ]

        batch_data = BatchCTEData(key=ResultKey(datetime.date(2024, 1, 1), {}), cte_sql="SELECT * FROM test", ops=ops)

        grouped = batch_data.group_ops_by_parameters()

        for param_tuple, ops_list in grouped.items():
            if param_tuple == (("x", 1),):
                # Should have NumRows, Sum, CountValues in that order
                assert len(ops_list) == 3
                assert isinstance(ops_list[0], NumRows)
                assert isinstance(ops_list[1], Sum)
                assert isinstance(ops_list[2], CountValues)
            elif param_tuple == (("x", 2),):
                # Should have Average, CustomSQL in that order
                assert len(ops_list) == 2
                assert isinstance(ops_list[0], Average)
                assert isinstance(ops_list[1], CustomSQL)
