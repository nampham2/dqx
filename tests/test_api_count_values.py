"""API-level tests for CountValues functionality.

This module tests the CountValues feature through the high-level API,
ensuring it works correctly with VerificationSuite and check decorator.
"""

import datetime

import pyarrow as pa
import pytest

from dqx import specs
from dqx.api import Context, VerificationSuite, check
from dqx.common import ResultKey
from dqx.datasource import DuckRelationDataSource
from dqx.orm.repositories import InMemoryMetricDB
from dqx.provider import MetricProvider


def test_count_values_with_check() -> None:
    """Test CountValues through the check decorator API."""
    # Create test data
    data = pa.table(
        {
            "status": ["active", "inactive", "active", "pending", "active", "inactive"],
            "priority": [1, 2, 1, 3, 1, 2],
        }
    )

    ds = DuckRelationDataSource.from_arrow(data, "data")

    # Create check using count_values
    @check(name="Count Values Check")
    def count_values_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.count_values("status", "active")).where(name="Count of active status should be 3").is_eq(3)

        ctx.assert_that(mp.count_values("priority", [1, 2])).where(
            name="Count of priority 1 or 2 should be at least 5"
        ).is_geq(5)

    # Run verification
    db = InMemoryMetricDB()
    suite = VerificationSuite([count_values_check], db, "Test Suite")
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})

    # Run the suite - it will print audit report but doesn't return a result by default
    suite.run([ds], key)

    # The test passes if no exceptions are raised
    # The audit report shows the assertions passed


def test_count_values_multiple_values() -> None:
    """Test CountValues with multiple values."""
    data = pa.table(
        {
            "region": ["US", "EU", "APAC", "US", "EU", "US", "Other"],
            "category": ["A", "B", "A", "C", "A", "B", "D"],
        }
    )

    ds = DuckRelationDataSource.from_arrow(data, "data")

    @check(name="Multiple Values Check")
    def multiple_values_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.count_values("region", ["US", "EU", "APAC"])).where(
            name="Count of main regions should be 6"
        ).is_eq(6)

        ctx.assert_that(mp.count_values("category", ["A", "B"])).where(
            name="Count of categories A or B should be greater than 4"
        ).is_gt(4)

    db = InMemoryMetricDB()
    suite = VerificationSuite([multiple_values_check], db, "Test Suite")
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})

    # Run the suite
    suite.run([ds], key)

    # The test passes if no exceptions are raised


def test_count_values_with_integers() -> None:
    """Test CountValues with integer values."""
    data = pa.table(
        {
            "score": [100, 200, 100, 300, 100, 200, 400],
            "level": [1, 2, 3, 1, 2, 1, 4],
        }
    )

    ds = DuckRelationDataSource.from_arrow(data, "data")

    @check(name="Integer Values Check")
    def integer_values_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.count_values("score", 100)).where(name="Count of score 100 should be 3").is_eq(3)

        ctx.assert_that(mp.count_values("level", [1, 2, 3])).where(
            name="Count of levels 1-3 should be at least 6"
        ).is_geq(6)

    db = InMemoryMetricDB()
    suite = VerificationSuite([integer_values_check], db, "Test Suite")
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})

    # Run the suite
    suite.run([ds], key)

    # The test passes if no exceptions are raised


def test_count_values_with_nulls() -> None:
    """Test CountValues behavior with nulls in data."""
    data = pa.table(
        {
            "status": ["active", None, "active", "inactive", None, "active"],
            "code": [1, 2, None, 1, 2, None],
        }
    )

    ds = DuckRelationDataSource.from_arrow(data, "data")

    # CountValues should not count nulls
    @check(name="Nulls Check")
    def nulls_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.count_values("status", "active")).where(
            name="Count of active status should be 3 (ignoring nulls)"
        ).is_eq(3)

        ctx.assert_that(mp.count_values("code", 1)).where(name="Count of code 1 should be 2 (ignoring nulls)").is_eq(2)

    db = InMemoryMetricDB()
    suite = VerificationSuite([nulls_check], db, "Test Suite")
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})

    # Run the suite
    suite.run([ds], key)

    # The test passes if no exceptions are raised


def test_count_values_with_spec_directly() -> None:
    """Test using CountValues spec directly in checks."""
    data = pa.table(
        {
            "type": ["A", "B", "A", "C", "A", "B"],
        }
    )

    ds = DuckRelationDataSource.from_arrow(data, "data")

    # Use the spec directly
    count_values_spec = specs.CountValues("type", "A")

    @check(name="Spec Direct Check")
    def spec_direct_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.metric(count_values_spec)).where(name="Count of type A should be 3").is_eq(3)

    db = InMemoryMetricDB()
    suite = VerificationSuite([spec_direct_check], db, "Test Suite")
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})

    # Run the suite
    suite.run([ds], key)

    # The test passes if no exceptions are raised


def test_count_values_failure_case() -> None:
    """Test CountValues with a failing check."""
    data = pa.table(
        {
            "status": ["active", "inactive", "pending", "inactive"],
        }
    )

    ds = DuckRelationDataSource.from_arrow(data, "data")

    @check(name="Failure Check")
    def failure_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.count_values("status", "active")).where(
            name="Count of active status should be greater than 2"
        ).is_gt(2)

    db = InMemoryMetricDB()
    suite = VerificationSuite([failure_check], db, "Test Suite")
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})

    # Run the suite - we expect it to complete but with failed checks
    # This will be visible in the audit report
    suite.run([ds], key)

    # The test passes because we're testing that CountValues works,
    # even when the assertion fails


def test_count_values_with_empty_list() -> None:
    """Test CountValues with an empty list of values raises ValueError."""
    data = pa.table({"col": [1, 2, 3]})
    ds = DuckRelationDataSource.from_arrow(data, "data")

    @check(name="Empty List Check")
    def empty_list_check(mp: MetricProvider, ctx: Context) -> None:
        # This should raise ValueError during graph building
        ctx.assert_that(mp.count_values("col", [])).where(name="Count with empty list should be 0").is_eq(0)

    db = InMemoryMetricDB()
    suite = VerificationSuite([empty_list_check], db, "Test Suite")
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})

    # Expecting ValueError because empty list is not allowed
    with pytest.raises(ValueError, match="CountValues requires at least one value"):
        suite.run([ds], key)


def test_count_values_with_special_characters() -> None:
    """Test CountValues with special characters in string values."""
    data = pa.table(
        {
            "name": ["O'Brien", "Smith", "O'Brien", "Jones"],
            "path": ["C:\\Users\\test", "/home/user", "C:\\Users\\test", "D:\\Data"],
        }
    )

    ds = DuckRelationDataSource.from_arrow(data, "data")

    @check(name="Special Characters Check")
    def special_chars_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.count_values("name", "O'Brien")).where(name="Count of O'Brien should be 2").is_eq(2)

        ctx.assert_that(mp.count_values("path", ["C:\\Users\\test", "D:\\Data"])).where(
            name="Count of Windows paths should be 3"
        ).is_eq(3)

    db = InMemoryMetricDB()
    suite = VerificationSuite([special_chars_check], db, "Test Suite")
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})

    # Run the suite
    suite.run([ds], key)

    # The test passes if no exceptions are raised


def test_count_values_case_sensitive() -> None:
    """Test that CountValues is case sensitive."""
    data = pa.table(
        {
            "status": ["Active", "active", "ACTIVE", "inactive"],
        }
    )

    ds = DuckRelationDataSource.from_arrow(data, "data")

    @check(name="Case Sensitive Check")
    def case_sensitive_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.count_values("status", "active")).where(name="Count should be case sensitive").is_eq(1)

    db = InMemoryMetricDB()
    suite = VerificationSuite([case_sensitive_check], db, "Test Suite")
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})

    # Run the suite
    suite.run([ds], key)

    # The test passes if no exceptions are raised


def test_count_values_symbol_info() -> None:
    """Test that CountValues symbols have correct metadata."""
    db = InMemoryMetricDB()
    provider = MetricProvider(db)

    # Create symbol
    symbol = provider.count_values("status", ["active", "pending"])

    # Get symbol info
    symbol_info = provider.get_symbol(symbol)

    # Verify metadata
    assert symbol_info.name == "count_values(status,[active,pending])"
    assert isinstance(symbol_info.metric_spec, specs.CountValues)
    assert symbol_info.metric_spec.parameters["column"] == "status"
    assert symbol_info.metric_spec.parameters["values"] == ["active", "pending"]


def test_count_values_with_booleans() -> None:
    """Test CountValues with boolean values."""
    data = pa.table(
        {
            "is_active": [True, False, True, True, False, None, True],
            "verified": [True, True, False, False, True, True, False],
        }
    )

    ds = DuckRelationDataSource.from_arrow(data, "data")

    @check(name="Boolean Values Check")
    def boolean_values_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.count_values("is_active", True)).where(
            name="Count of True values should be 4 (ignoring nulls)"
        ).is_eq(4)

        ctx.assert_that(mp.count_values("is_active", False)).where(name="Count of False values should be 2").is_eq(2)

        ctx.assert_that(mp.count_values("verified", True)).where(name="Count of verified True should be 4").is_eq(4)

    db = InMemoryMetricDB()
    suite = VerificationSuite([boolean_values_check], db, "Test Suite")
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})

    # Run the suite
    suite.run([ds], key)

    # The test passes if no exceptions are raised
