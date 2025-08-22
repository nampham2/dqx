import pyarrow as pa

from dqx.common import SqlDataSource
from dqx.extensions.pyarrow_ds import ArrowDataSource


def test_pyarrow_ds(commerce_data_c1: pa.Table) -> None:
    ds = ArrowDataSource(commerce_data_c1)
    assert isinstance(ds, SqlDataSource)
