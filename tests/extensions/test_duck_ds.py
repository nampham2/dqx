import duckdb
import pyarrow as pa

from dqx.common import SqlDataSource
from dqx.extensions.duck_ds import DuckRelationDataSource
from dqx.extensions.pyarrow_ds import ArrowDataSource


def test_duck_relation_datasource_init(commerce_data_c1: pa.Table) -> None:
    """Test DuckRelationDataSource initialization."""
    # Create a duckdb relation from the arrow table
    relation = duckdb.arrow(commerce_data_c1)
    
    ds = DuckRelationDataSource(relation)
    assert ds.name == "duckdb"
    assert ds.dialect == "duckdb"
    assert hasattr(ds, "_relation")
    assert hasattr(ds, "_table_name")
    assert len(ds._table_name) == 7  # random prefix is _ + 6 characters


def test_duck_relation_datasource_cte(commerce_data_c1: pa.Table) -> None:
    """Test DuckRelationDataSource cte property."""
    relation = duckdb.arrow(commerce_data_c1)
    ds = DuckRelationDataSource(relation)
    
    cte = ds.cte
    assert cte.startswith("SELECT * FROM ")
    assert ds._table_name in cte


def test_duck_relation_datasource_query(commerce_data_c1: pa.Table) -> None:
    """Test DuckRelationDataSource query method."""
    relation = duckdb.arrow(commerce_data_c1)
    ds = DuckRelationDataSource(relation)
    
    # Test a simple query
    query = f"SELECT COUNT(*) as count FROM {ds._table_name}"
    result = ds.query(query)
    
    assert isinstance(result, duckdb.DuckDBPyRelation)
    # Fetch the result to verify it works
    count_result = result.fetchall()
    assert count_result[0][0] == 1000  # commerce_data_c1 has 1000 rows


def test_duck_relation_datasource_from_arrow(commerce_data_c1: pa.Table) -> None:
    """Test DuckRelationDataSource.from_arrow classmethod."""
    ds = DuckRelationDataSource.from_arrow(commerce_data_c1)
    
    # Should return an ArrowDataSource instance
    assert isinstance(ds, ArrowDataSource)
    assert isinstance(ds, SqlDataSource)
