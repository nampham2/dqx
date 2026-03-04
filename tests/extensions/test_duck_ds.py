import datetime

import duckdb
import pyarrow as pa
import pytest

from dqx.common import SqlDataSource
from dqx.datasource import DuckRelationDataSource


def test_duck_relation_datasource_init(commerce_data_c1: pa.Table) -> None:
    """Test DuckRelationDataSource initialization."""
    # Create a duckdb relation from the arrow table
    relation = duckdb.arrow(commerce_data_c1)

    ds = DuckRelationDataSource(relation, "test_data")
    assert ds.name == "test_data"
    assert ds.dialect == "duckdb"
    assert hasattr(ds, "_relation")
    assert hasattr(ds, "_table_name")
    assert len(ds._table_name) == 7  # random prefix is _ + 6 characters


def test_duck_relation_datasource_cte(commerce_data_c1: pa.Table) -> None:
    """Test DuckRelationDataSource cte method."""
    relation = duckdb.arrow(commerce_data_c1)
    ds = DuckRelationDataSource(relation, "test_data")

    nominal_date = datetime.date(2024, 1, 1)
    cte = ds.cte(nominal_date)
    assert cte.startswith("SELECT * FROM ")
    assert ds._table_name in cte


def test_duck_relation_datasource_query(commerce_data_c1: pa.Table) -> None:
    """Test DuckRelationDataSource query method."""
    relation = duckdb.arrow(commerce_data_c1)
    ds = DuckRelationDataSource(relation, "test_data")

    # Test a simple query
    query = f"SELECT COUNT(*) as count FROM {ds._table_name}"
    result = ds.query(query)

    # Verify return type is pa.Table
    assert isinstance(result, pa.Table)
    # Fetch the result to verify it works
    count_result = result.to_pylist()
    assert count_result[0]["count"] == 1000  # commerce_data_c1 has 1000 rows


def test_duck_relation_datasource_from_arrow(commerce_data_c1: pa.Table) -> None:
    """Test DuckRelationDataSource.from_arrow classmethod."""
    ds = DuckRelationDataSource.from_arrow(commerce_data_c1, "commerce_data")

    # Should return a DuckRelationDataSource instance
    assert isinstance(ds, DuckRelationDataSource)
    assert isinstance(ds, SqlDataSource)
    assert ds.name == "commerce_data"
    assert ds.dialect == "duckdb"

    # Test that name is read-only
    with pytest.raises(AttributeError):
        ds.name = "new_name"  # type: ignore[misc]


class TestDuckRelationDataSource:
    """Test DuckRelationDataSource functionality."""

    def test_duck_relation_datasource_schema(self, commerce_data_c1: pa.Table) -> None:
        """Test DuckRelationDataSource schema property."""
        relation = duckdb.arrow(commerce_data_c1)
        ds = DuckRelationDataSource(relation, "test_data")

        # Verify schema is returned
        schema = ds.schema
        assert isinstance(schema, pa.Schema)

        # Verify schema matches the original data
        assert schema == commerce_data_c1.schema

        # Verify schema has expected fields from commerce_data_c1
        assert "name" in schema.names
        assert "quantity" in schema.names
        assert "price" in schema.names
