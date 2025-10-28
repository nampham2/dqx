"""API-level tests for UniqueCount functionality.

This module tests the UniqueCount feature through the high-level API,
ensuring it correctly counts distinct values in columns.
"""

import datetime

import pyarrow as pa

from dqx import specs
from dqx.api import Context, VerificationSuite, check
from dqx.common import ResultKey
from dqx.datasource import DuckRelationDataSource
from dqx.orm.repositories import InMemoryMetricDB
from dqx.provider import MetricProvider


def test_unique_count_basic() -> None:
    """Test UniqueCount through the check decorator API."""
    data = pa.table(
        {
            "category": ["A", "B", "A", "C", "B", "A", "D"],
            "user_id": [1, 2, 1, 3, 2, 4, 5],
        }
    )

    ds = DuckRelationDataSource.from_arrow(data, "data")

    @check(name="Unique Count Check")
    def unique_count_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.unique_count("category")).where(name="Number of unique categories should be 4").is_eq(4)

        ctx.assert_that(mp.unique_count("user_id")).where(name="Number of unique users should be 5").is_eq(5)

    db = InMemoryMetricDB()
    suite = VerificationSuite([unique_count_check], db, "Test Suite")
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})
    suite.run([ds], key)


def test_unique_count_with_nulls() -> None:
    """Test UniqueCount behavior with null values."""
    data = pa.table(
        {
            "product": ["A", None, "B", "A", None, "C", "B"],
            "score": [10, 20, None, 10, 30, None, 20],
        }
    )

    ds = DuckRelationDataSource.from_arrow(data, "data")

    @check(name="Null Handling Check")
    def null_handling_check(mp: MetricProvider, ctx: Context) -> None:
        # COUNT(DISTINCT) excludes nulls
        ctx.assert_that(mp.unique_count("product")).where(name="Unique products should be 3 (nulls excluded)").is_eq(3)

        ctx.assert_that(mp.unique_count("score")).where(name="Unique scores should be 3 (nulls excluded)").is_eq(3)

    db = InMemoryMetricDB()
    suite = VerificationSuite([null_handling_check], db, "Test Suite")
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})
    suite.run([ds], key)


def test_unique_count_edge_cases() -> None:
    """Test UniqueCount with edge cases."""
    # Empty column
    empty_data = pa.table({"col": pa.array([], type=pa.string())})

    # All nulls
    null_data = pa.table({"col": [None, None, None]})

    # All same value
    same_data = pa.table({"col": ["X", "X", "X", "X"]})

    @check(name="Empty Data Check")
    def empty_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.unique_count("col")).where(name="Empty column should have 0 unique values").is_eq(0)

    @check(name="All Nulls Check")
    def null_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.unique_count("col")).where(name="All nulls should have 0 unique values").is_eq(0)

    @check(name="Same Value Check")
    def same_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.unique_count("col")).where(name="All same value should have 1 unique value").is_eq(1)

    db = InMemoryMetricDB()
    suite = VerificationSuite([empty_check], db, "Test Suite")
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})
    suite.run([DuckRelationDataSource.from_arrow(empty_data, "empty")], key)

    suite = VerificationSuite([null_check], db, "Test Suite")
    suite.run([DuckRelationDataSource.from_arrow(null_data, "nulls")], key)

    suite = VerificationSuite([same_check], db, "Test Suite")
    suite.run([DuckRelationDataSource.from_arrow(same_data, "same")], key)


def test_unique_count_with_spec_directly() -> None:
    """Test using UniqueCount spec directly."""
    data = pa.table({"region": ["US", "EU", "US", "APAC", "EU", "US"]})
    ds = DuckRelationDataSource.from_arrow(data, "data")

    unique_count_spec = specs.UniqueCount("region")

    @check(name="Spec Direct Check")
    def spec_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.metric(unique_count_spec)).where(name="Unique regions should be 3").is_eq(3)

    db = InMemoryMetricDB()
    suite = VerificationSuite([spec_check], db, "Test Suite")
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})
    suite.run([ds], key)


def test_unique_count_various_types() -> None:
    """Test UniqueCount with different data types."""
    data = pa.table(
        {
            "strings": ["apple", "banana", "apple", "cherry", "banana"],
            "integers": [100, 200, 100, 300, 200],
            "floats": [1.5, 2.5, 1.5, 3.5, 2.5],
            "booleans": [True, False, True, False, True],
        }
    )

    ds = DuckRelationDataSource.from_arrow(data, "data")

    @check(name="Type Variety Check")
    def type_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.unique_count("strings")).where(name="Unique strings").is_eq(3)

        ctx.assert_that(mp.unique_count("integers")).where(name="Unique integers").is_eq(3)

        ctx.assert_that(mp.unique_count("floats")).where(name="Unique floats").is_eq(3)

        ctx.assert_that(mp.unique_count("booleans")).where(name="Unique booleans").is_eq(2)

    db = InMemoryMetricDB()
    suite = VerificationSuite([type_check], db, "Test Suite")
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})
    suite.run([ds], key)


def test_unique_count_symbol_info() -> None:
    """Test that UniqueCount symbols have correct metadata."""
    db = InMemoryMetricDB()
    provider = MetricProvider(db)

    symbol = provider.unique_count("customer_id")
    symbol_info = provider.get_symbol(symbol)

    assert symbol_info.name == "unique_count(customer_id)"
    assert isinstance(symbol_info.metric_spec, specs.UniqueCount)
    assert symbol_info.metric_spec.parameters["column"] == "customer_id"
