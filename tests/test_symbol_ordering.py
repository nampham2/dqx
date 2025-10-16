"""Test natural ordering of symbols in collect_symbols method."""

from datetime import date

import pyarrow as pa

from dqx.api import Context, MetricProvider, VerificationSuite, check
from dqx.common import ResultKey
from dqx.extensions.pyarrow_ds import ArrowDataSource
from dqx.orm.repositories import InMemoryMetricDB


def test_collect_symbols_natural_ordering() -> None:
    """Test that symbols are sorted in natural numeric order."""
    db = InMemoryMetricDB()

    @check(name="Many Metrics", datasets=["test"])
    def many_metrics(mp: MetricProvider, ctx: Context) -> None:
        # Create metrics that will generate x_1 through x_15
        # Intentionally create them out of order to test sorting
        mp.sum("col_5", dataset="test")  # x_1
        mp.average("col_2", dataset="test")  # x_2
        mp.minimum("col_10", dataset="test")  # x_3
        mp.maximum("col_1", dataset="test")  # x_4
        mp.variance("col_8", dataset="test")  # x_5
        mp.sum("col_12", dataset="test")  # x_6
        mp.average("col_3", dataset="test")  # x_7
        mp.minimum("col_11", dataset="test")  # x_8
        mp.maximum("col_4", dataset="test")  # x_9
        mp.variance("col_7", dataset="test")  # x_10
        mp.sum("col_9", dataset="test")  # x_11
        mp.average("col_6", dataset="test")  # x_12
        mp.minimum("col_13", dataset="test")  # x_13
        mp.maximum("col_14", dataset="test")  # x_14
        mp.variance("col_15", dataset="test")  # x_15

    suite = VerificationSuite([many_metrics], db, "Test Suite")

    # Create test data with all columns (2 rows to satisfy variance calculation)
    data = {f"col_{i}": [float(i), float(i)] for i in range(1, 16)}
    table = pa.table(data)
    datasource = ArrowDataSource(table)

    key = ResultKey(yyyy_mm_dd=date.today(), tags={})
    suite.run({"test": datasource}, key)

    symbols = suite.collect_symbols()

    # Extract symbol names
    names = [s.name for s in symbols]

    # Should have all 15 symbols
    assert len(names) == 15

    # Verify natural ordering (x_1, x_2, ..., x_10, x_11, ..., x_15)
    expected = [f"x_{i}" for i in range(1, 16)]
    assert names == expected, f"Expected {expected}, but got {names}"

    # Specifically check the problematic transition from single to double digits
    x_9_index = names.index("x_9")
    x_10_index = names.index("x_10")
    assert x_10_index == x_9_index + 1, "x_10 should come immediately after x_9"


def test_collect_symbols_large_numbers() -> None:
    """Test natural ordering with larger numbers (x_99, x_100, x_101)."""
    db = InMemoryMetricDB()

    @check(name="Large Numbers", datasets=["test"])
    def large_numbers(mp: MetricProvider, ctx: Context) -> None:
        # Create 105 metrics to test x_1 through x_105
        for i in range(105):
            mp.sum(f"col_{i}", dataset="test")

    suite = VerificationSuite([large_numbers], db, "Test Suite")

    # Create test data (2 rows to satisfy variance calculation)
    data = {f"col_{i}": [float(i), float(i)] for i in range(105)}
    table = pa.table(data)
    datasource = ArrowDataSource(table)

    key = ResultKey(yyyy_mm_dd=date.today(), tags={})
    suite.run({"test": datasource}, key)

    symbols = suite.collect_symbols()
    names = [s.name for s in symbols]

    # Check specific transitions
    assert names.index("x_99") < names.index("x_100")
    assert names.index("x_100") < names.index("x_101")
    assert names.index("x_9") < names.index("x_10")
    assert names.index("x_10") < names.index("x_100")

    # Verify complete ordering
    expected = [f"x_{i}" for i in range(1, 106)]
    assert names == expected
