import datetime

import pyarrow as pa
import pytest

from dqx.common import SqlDataSource
from dqx.datasource import ArrowDataSource, DuckRelationDataSource


def test_arrow_datasource_init(commerce_data_c1: pa.Table) -> None:
    """Test ArrowDataSource initialization with PyArrow Table."""
    ds = ArrowDataSource(commerce_data_c1, "test_data")

    assert ds.name == "test_data"
    assert ds.dialect == "duckdb"
    assert hasattr(ds, "_table")
    assert hasattr(ds, "_table_name")
    assert len(ds._table_name) == 7  # random prefix is _ + 6 characters


def test_arrow_datasource_cte(commerce_data_c1: pa.Table) -> None:
    """Test ArrowDataSource cte method."""
    ds = ArrowDataSource(commerce_data_c1, "test_data")

    nominal_date = datetime.date(2024, 1, 1)
    cte = ds.cte(nominal_date)
    assert cte.startswith("SELECT * FROM ")
    assert ds._table_name in cte


def test_arrow_datasource_query(commerce_data_c1: pa.Table) -> None:
    """Test ArrowDataSource query method."""
    ds = ArrowDataSource(commerce_data_c1, "test_data")

    # Test a simple query
    query = f"SELECT COUNT(*) as count FROM {ds._table_name}"
    result = ds.query(query)

    # Verify return type is pa.Table
    assert isinstance(result, pa.Table)
    # Fetch the result to verify it works
    count_result = result.to_pylist()
    assert count_result[0]["count"] == 1000  # commerce_data_c1 has 1000 rows


def test_arrow_datasource_protocol_compliance(commerce_data_c1: pa.Table) -> None:
    """Test ArrowDataSource implements SqlDataSource protocol."""
    ds = ArrowDataSource(commerce_data_c1, "commerce_data")

    # Should be recognized as a SqlDataSource
    assert isinstance(ds, SqlDataSource)
    assert ds.name == "commerce_data"
    assert ds.dialect == "duckdb"

    # Test that name is read-only
    with pytest.raises(AttributeError):
        ds.name = "new_name"  # type: ignore[misc]


class TestArrowDataSource:
    """Test ArrowDataSource functionality."""

    def test_arrow_datasource_schema(self, commerce_data_c1: pa.Table) -> None:
        """Test ArrowDataSource schema property."""
        ds = ArrowDataSource(commerce_data_c1, "test_data")

        # Verify schema is returned
        schema = ds.schema
        assert isinstance(schema, pa.Schema)

        # Verify schema matches the original data
        assert schema == commerce_data_c1.schema

        # Verify schema has expected fields from commerce_data_c1
        assert "name" in schema.names
        assert "quantity" in schema.names
        assert "price" in schema.names

    def test_arrow_datasource_skip_dates(self, commerce_data_c1: pa.Table) -> None:
        """Test ArrowDataSource skip_dates property."""
        skip_dates = {datetime.date(2024, 1, 1), datetime.date(2024, 1, 2)}
        ds = ArrowDataSource(commerce_data_c1, "test_data", skip_dates=skip_dates)

        # Verify skip_dates is returned correctly
        assert ds.skip_dates == skip_dates

        # Verify skip_dates is read-only
        with pytest.raises(AttributeError):
            ds.skip_dates = set()  # type: ignore[misc]

    def test_arrow_datasource_skip_dates_default(self, commerce_data_c1: pa.Table) -> None:
        """Test ArrowDataSource skip_dates defaults to empty set."""
        ds = ArrowDataSource(commerce_data_c1, "test_data")

        # Verify skip_dates defaults to empty set
        assert ds.skip_dates == set()

    def test_arrow_datasource_query_with_filter(self, commerce_data_c1: pa.Table) -> None:
        """Test ArrowDataSource query with WHERE clause."""
        ds = ArrowDataSource(commerce_data_c1, "test_data")

        # Query with filter
        query = f"SELECT * FROM {ds._table_name} WHERE quantity > 50"
        result = ds.query(query)

        # Verify return type
        assert isinstance(result, pa.Table)

        # Verify filter works
        quantities = result.column("quantity").to_pylist()
        assert all(q > 50 for q in quantities)

    def test_arrow_datasource_cte_ignores_parameters(self, commerce_data_c1: pa.Table) -> None:
        """Test ArrowDataSource cte ignores date and parameters."""
        ds = ArrowDataSource(commerce_data_c1, "test_data")

        # CTE should be the same regardless of date or parameters
        cte1 = ds.cte(datetime.date(2024, 1, 1))
        cte2 = ds.cte(datetime.date(2024, 12, 31))
        cte3 = ds.cte(datetime.date(2024, 6, 15), {"param": "value"})

        assert cte1 == cte2 == cte3
        assert cte1 == f"SELECT * FROM {ds._table_name}"


class TestArrowDataSourceRecordBatch:
    """Test ArrowDataSource with PyArrow RecordBatch input."""

    def test_arrow_datasource_from_recordbatch(self) -> None:
        """Test ArrowDataSource initialization with RecordBatch."""
        # Create a RecordBatch
        batch = pa.record_batch(
            [pa.array([1, 2, 3, 4]), pa.array(["a", "b", "c", "d"]), pa.array([10.5, 20.3, 30.1, 40.7])],
            names=["id", "category", "value"],
        )

        ds = ArrowDataSource(batch, "test_batch")

        assert ds.name == "test_batch"
        assert ds.dialect == "duckdb"
        assert isinstance(ds, SqlDataSource)

    def test_arrow_datasource_recordbatch_schema(self) -> None:
        """Test ArrowDataSource schema property with RecordBatch."""
        batch = pa.record_batch([pa.array([1, 2, 3]), pa.array(["x", "y", "z"])], names=["num", "letter"])

        ds = ArrowDataSource(batch, "test_batch")

        schema = ds.schema
        assert isinstance(schema, pa.Schema)
        assert schema == batch.schema
        assert "num" in schema.names
        assert "letter" in schema.names

    def test_arrow_datasource_recordbatch_query(self) -> None:
        """Test ArrowDataSource query with RecordBatch input."""
        batch = pa.record_batch(
            [pa.array([1, 2, 3, 4, 5]), pa.array([100, 200, 300, 400, 500])], names=["id", "amount"]
        )

        ds = ArrowDataSource(batch, "test_batch")

        # Query the data
        query = f"SELECT SUM(amount) as total FROM {ds._table_name}"
        result = ds.query(query)

        assert isinstance(result, pa.Table)
        total_result = result.to_pylist()
        assert total_result[0]["total"] == 1500

    def test_arrow_datasource_recordbatch_with_skip_dates(self) -> None:
        """Test ArrowDataSource with RecordBatch and skip_dates."""
        batch = pa.record_batch([pa.array([1, 2, 3]), pa.array(["a", "b", "c"])], names=["id", "value"])

        skip_dates = {datetime.date(2024, 1, 1), datetime.date(2024, 1, 2)}
        ds = ArrowDataSource(batch, "test_batch", skip_dates=skip_dates)

        assert ds.skip_dates == skip_dates
        assert ds.name == "test_batch"


class TestArrowDataSourceIntegration:
    """Integration tests for ArrowDataSource."""

    def test_arrow_datasource_comparison_with_duck_relation(self, commerce_data_c1: pa.Table) -> None:
        """Test that ArrowDataSource produces same results as DuckRelationDataSource."""
        # Create both datasources from same table
        arrow_ds = ArrowDataSource(commerce_data_c1, "arrow_data")
        duck_ds = DuckRelationDataSource.from_arrow(commerce_data_c1, "duck_data")

        # Execute same query on both
        query_arrow = f"SELECT COUNT(*) as count FROM {arrow_ds._table_name}"
        query_duck = f"SELECT COUNT(*) as count FROM {duck_ds._table_name}"

        result_arrow = arrow_ds.query(query_arrow)
        result_duck = duck_ds.query(query_duck)

        # Results should be identical
        assert result_arrow.to_pylist() == result_duck.to_pylist()

    def test_arrow_datasource_complex_query(self, commerce_data_c1: pa.Table) -> None:
        """Test ArrowDataSource with complex SQL query."""
        ds = ArrowDataSource(commerce_data_c1, "commerce")

        # Complex query with aggregation and filtering
        query = f"""
        SELECT
            COUNT(*) as total_rows,
            AVG(quantity) as avg_quantity,
            SUM(price) as total_price
        FROM {ds._table_name}
        WHERE quantity > 0
        """

        result = ds.query(query)
        assert isinstance(result, pa.Table)

        data = result.to_pylist()[0]
        assert data["total_rows"] == 1000
        assert data["avg_quantity"] > 0
        assert data["total_price"] > 0

    def test_arrow_datasource_empty_table(self) -> None:
        """Test ArrowDataSource with empty table."""
        # Create empty table
        empty_table = pa.table({"id": pa.array([], type=pa.int64()), "value": pa.array([], type=pa.string())})

        ds = ArrowDataSource(empty_table, "empty_data")

        # Schema should still be accessible
        assert ds.schema == empty_table.schema

        # Query should work but return no rows
        query = f"SELECT COUNT(*) as count FROM {ds._table_name}"
        result = ds.query(query)
        assert result.to_pylist()[0]["count"] == 0
