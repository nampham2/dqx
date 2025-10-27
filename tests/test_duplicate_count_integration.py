"""Integration tests for DuplicateCount functionality."""

from typing import Any
from unittest.mock import Mock

import pytest
from returns.result import Success

from dqx import ops, specs, states
from dqx.analyzer import Analyzer
from dqx.datasource import DuckRelationDataSource
from dqx.dialect import DuckDBDialect
from dqx.orm.repositories import MetricDB
from dqx.provider import MetricProvider


class TestDuplicateCountIntegration:
    """Test DuplicateCount integration across the entire system."""

    def test_duplicate_count_end_to_end_flow(self) -> None:
        """Test the complete flow from provider to spec to op to state."""
        # 1. Create provider with mock DB
        mock_db = Mock(spec=MetricDB)
        provider = MetricProvider(mock_db)

        # 2. Create a duplicate_count symbol
        columns = ["user_id", "session_id"]
        symbol = provider.duplicate_count(columns)

        # 3. Verify symbol registration
        registered_metric = provider.get_symbol(symbol)
        assert registered_metric.name == "duplicate_count(session_id,user_id)"
        assert isinstance(registered_metric.metric_spec, specs.DuplicateCount)

        # 4. Get the spec and verify its properties
        spec = registered_metric.metric_spec
        assert spec.metric_type == "DuplicateCount"
        assert spec.name == "duplicate_count(session_id,user_id)"
        assert spec.parameters == {"columns": ["session_id", "user_id"]}

        # 5. Get the analyzers (ops) from the spec
        analyzers = spec.analyzers
        assert len(analyzers) == 1
        assert isinstance(analyzers[0], ops.DuplicateCount)

        # 6. Test the state creation
        # Simulate op execution
        analyzers[0].assign(42.0)
        state = spec.state()
        assert isinstance(state, states.DuplicateCount)
        assert state.value == 42.0

        # 7. Test serialization/deserialization
        serialized = state.serialize()
        deserialized = spec.deserialize(serialized)
        assert isinstance(deserialized, states.DuplicateCount)
        assert deserialized.value == 42.0

    def test_duplicate_count_with_analyzer(self) -> None:
        """Test DuplicateCount with the Analyzer and DataSource."""
        import datetime

        import duckdb

        from dqx.common import ResultKey

        # Create test data
        conn = duckdb.connect(":memory:")
        conn.execute("""
            CREATE TABLE test_data AS
            SELECT * FROM (VALUES
                (1, 101),
                (1, 101),  -- Duplicate
                (2, 102),
                (3, 103),
                (2, 102)   -- Another duplicate
            ) AS t(product_id, user_id)
        """)

        # Create a DuckDB relation and data source
        relation = conn.sql("SELECT * FROM test_data")
        data_source = DuckRelationDataSource(relation, "test_data")

        # Create analyzer
        analyzer = Analyzer()

        # Create DuplicateCount spec
        dc_spec = specs.DuplicateCount(["product_id", "user_id"])

        # Analyze with the spec
        key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})
        report = analyzer.analyze(data_source, {key: [dc_spec]})

        # Get the metric from the report
        metric = report[(dc_spec, key)]

        # Verify the value
        # 5 rows - 3 unique combinations = 2 duplicates
        assert metric.value == 2.0

    def test_duplicate_count_in_verification_suite(self) -> None:
        """Test using DuplicateCount in a verification suite context."""
        from dqx.api import VerificationSuite, check

        # Create mock provider
        mock_db = Mock(spec=MetricDB)

        # Define check function
        @check(name="no_duplicate_orders")
        def check_no_duplicates(mp: MetricProvider, ctx: Any) -> None:
            duplicate_count = mp.duplicate_count(["order_id"])

            # Mock the evaluation
            mp.index[duplicate_count].fn = lambda k: Success(0.0)

            ctx.assert_that(duplicate_count).where(name="No duplicate orders").is_eq(0.0)

        # Create suite
        suite = VerificationSuite(checks=[check_no_duplicates], db=mock_db, name="test_suite")

        # Run suite which will build graph internally
        import datetime

        from dqx.common import ResultKey

        key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})

        import pyarrow as pa

        from dqx.datasource import DuckRelationDataSource

        data = pa.table({"order_id": [1, 2, 3]})
        suite.run([DuckRelationDataSource.from_arrow(data, "data")], key)

        # Access graph after running
        graph = suite.graph

        # Verify graph structure
        # Root -> Check -> Assertion
        assert len(list(graph.checks())) == 1
        assert len(list(graph.assertions())) == 1

        # Verify assertion details
        assertion = list(graph.assertions())[0]
        assert assertion.name == "No duplicate orders"

    def test_duplicate_count_multiple_columns_order(self) -> None:
        """Test that column order is handled consistently."""
        # Create specs with different column orders
        dc1 = specs.DuplicateCount(["col1", "col2", "col3"])
        dc2 = specs.DuplicateCount(["col3", "col1", "col2"])
        dc3 = specs.DuplicateCount(["col2", "col3", "col1"])

        # All should have the same name (sorted)
        assert dc1.name == dc2.name == dc3.name
        assert dc1.name == "duplicate_count(col1,col2,col3)"

        # All should be equal
        assert dc1 == dc2 == dc3

        # All should have the same hash
        assert hash(dc1) == hash(dc2) == hash(dc3)

    def test_duplicate_count_sql_execution_mock(self) -> None:
        """Test the full SQL execution flow with mocked database."""
        import duckdb

        # Create in-memory DuckDB connection
        conn = duckdb.connect(":memory:")

        # Create test table
        conn.execute("""
            CREATE TABLE orders AS
            SELECT * FROM (VALUES
                (1, 101, '2024-01-01'),
                (2, 101, '2024-01-02'),
                (3, 102, '2024-01-01'),
                (1, 101, '2024-01-01'),  -- Duplicate order
                (4, 103, '2024-01-03')
            ) AS t(order_id, customer_id, order_date)
        """)

        # Create dialect and op
        dialect = DuckDBDialect()
        op = ops.DuplicateCount(["order_id"])

        # Generate SQL
        sql = dialect.translate_sql_op(op)
        full_query = f"SELECT {sql} FROM orders"

        # Execute and verify
        result = conn.execute(full_query).fetchone()
        assert result is not None
        assert result[0] == 1.0  # One duplicate order_id

        # Test with multiple columns
        op2 = ops.DuplicateCount(["order_id", "customer_id", "order_date"])
        sql2 = dialect.translate_sql_op(op2)
        full_query2 = f"SELECT {sql2} FROM orders"

        result2 = conn.execute(full_query2).fetchone()
        assert result2 is not None
        assert result2[0] == 1.0  # One complete duplicate row

    def test_duplicate_count_error_handling(self) -> None:
        """Test error handling for DuplicateCount."""
        # Test empty columns
        with pytest.raises(ValueError, match="At least one column must be specified"):
            specs.DuplicateCount([])

        # Test op without value assignment
        op = ops.DuplicateCount(["col1"])
        with pytest.raises(Exception, match="has not been collected yet"):
            op.value()

        # Test invalid column types (at op level)
        with pytest.raises(ValueError, match="at least one column"):
            ops.DuplicateCount([])
