import pyarrow as pa

from dqx.common import DuckDataSource
from dqx.extensions.pyarrow_ds import ArrowDataSource


def test_pyarrow_ds(commerce_data: pa.Table) -> None:
    ds = ArrowDataSource(commerce_data)
    assert isinstance(ds, DuckDataSource)
