import tempfile
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from dqx.common import SqlDataSource
from dqx.extensions.pyarrow_ds import ArrowBatchDataSource, ArrowDataSource


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


def test_arrow_batch_datasource_init(commerce_data_c1: pa.Table, commerce_data_c2: pa.Table) -> None:
    """Test ArrowBatchDataSource initialization."""
    batches = [commerce_data_c1, commerce_data_c2]

    batch_ds = ArrowBatchDataSource(batches)
    assert batch_ds.name == "pyarrow_batch"
    assert batch_ds.dialect == "duckdb"
    assert hasattr(batch_ds, "_batches")


def test_arrow_batch_datasource_arrow_ds(commerce_data_c1: pa.Table, commerce_data_c2: pa.Table) -> None:
    """Test ArrowBatchDataSource arrow_ds method."""
    batches = [commerce_data_c1, commerce_data_c2]
    batch_ds = ArrowBatchDataSource(batches)

    # Iterate through arrow_ds generator
    ds_list = list(batch_ds.arrow_ds())

    assert len(ds_list) == 2
    for ds in ds_list:
        assert isinstance(ds, ArrowDataSource)
        assert ds.name == "pyarrow"
        assert ds.dialect == "duckdb"


def test_arrow_batch_datasource_from_parquets(commerce_data_c1: pa.Table, commerce_data_c2: pa.Table) -> None:
    """Test ArrowBatchDataSource from_parquets classmethod."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test parquet files
        parquet1 = Path(tmpdir) / "test1.parquet"
        parquet2 = Path(tmpdir) / "test2.parquet"

        pq.write_table(commerce_data_c1, str(parquet1))
        pq.write_table(commerce_data_c2, str(parquet2))

        # Test from_parquets with default parameters
        batch_ds = ArrowBatchDataSource.from_parquets([str(parquet1), str(parquet2)])
        assert batch_ds.name == "pyarrow_batch"
        assert batch_ds.dialect == "duckdb"

        # Verify we can iterate through the batches
        count = 0
        for ds in batch_ds.arrow_ds():
            assert isinstance(ds, ArrowDataSource)
            count += 1
        assert count > 0  # Should have at least one batch

        # Test with custom batch_size
        batch_ds2 = ArrowBatchDataSource.from_parquets(
            [str(parquet1), str(parquet2)],
            batch_size=500,  # Each file has 1000 rows, so this should create multiple batches
        )

        # Count batches
        batch_count = sum(1 for _ in batch_ds2.arrow_ds())
        assert batch_count >= 4  # 2000 rows / 500 = 4 batches minimum


def test_arrow_batch_datasource_with_record_batches(commerce_data_c1: pa.Table) -> None:
    """Test ArrowBatchDataSource with RecordBatch objects."""
    # Convert table to multiple record batches
    batches = commerce_data_c1.to_batches(max_chunksize=500)

    batch_ds = ArrowBatchDataSource(batches)

    # Verify arrow_ds works with RecordBatch objects
    ds_count = 0
    for ds in batch_ds.arrow_ds():
        assert isinstance(ds, ArrowDataSource)
        ds_count += 1

    assert ds_count == len(batches)
