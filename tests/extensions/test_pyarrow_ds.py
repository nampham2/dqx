import duckdb
import pyarrow as pa

from dqx.common import SqlDataSource
from dqx.extensions.pyarrow_ds import ArrowDataSource


def test_pyarrow_ds(commerce_data_c1: pa.Table) -> None:
    ds = ArrowDataSource(commerce_data_c1)
    assert isinstance(ds, SqlDataSource)


def test_arrow_datasource_init(commerce_data_c1: pa.Table) -> None:
    """Test ArrowDataSource initialization."""
    ds = ArrowDataSource(commerce_data_c1)
    assert ds.name == "pyarrow"
    assert ds.dialect == "duckdb"
    assert hasattr(ds, "_table")
    assert hasattr(ds, "_table_name")
    assert len(ds._table_name) == 7  # random prefix is _ + 6 characters


def test_arrow_datasource_cte(commerce_data_c1: pa.Table) -> None:
    """Test ArrowDataSource cte property."""
    ds = ArrowDataSource(commerce_data_c1)

    cte = ds.cte
    assert cte == f"SELECT * FROM {ds._table_name}"


def test_arrow_datasource_query(commerce_data_c1: pa.Table) -> None:
    """Test ArrowDataSource query method."""
    ds = ArrowDataSource(commerce_data_c1)

    # Test a simple query
    query = f"SELECT COUNT(*) as count FROM {ds._table_name}"
    result = ds.query(query)

    assert isinstance(result, duckdb.DuckDBPyRelation)
    # Fetch the result to verify it works
    count_result = result.fetchall()
    assert count_result[0][0] == 1000  # commerce_data_c1 has 1000 rows

    # Test a more complex query
    query2 = f"SELECT name, price FROM {ds._table_name} WHERE price > 5000 LIMIT 5"
    result2 = ds.query(query2)
    rows = result2.fetchall()
    assert len(rows) <= 5
    for row in rows:
        assert row[1] > 5000  # price should be > 5000


def test_arrow_datasource_with_record_batch(commerce_data_c1: pa.Table) -> None:
    """Test ArrowDataSource with RecordBatch."""
    # Convert table to record batch
    batch = commerce_data_c1.to_batches()[0]

    ds = ArrowDataSource(batch)
    assert ds.name == "pyarrow"

    # Test query works with RecordBatch
    query = f"SELECT COUNT(*) as count FROM {ds._table_name}"
    result = ds.query(query)
    count_result = result.fetchall()
    assert count_result[0][0] == len(batch)
