# Capture log output for testing

from datetime import date

import pytest

from dqx.api import Context, VerificationSuite, check
from dqx.common import DQXError, ResultKey
from dqx.orm.repositories import InMemoryMetricDB
from dqx.provider import MetricProvider


def test_suite_validation_on_build_graph_success() -> None:
    """Test that validation runs during suite initialization with valid suite."""
    db = InMemoryMetricDB()

    @check(name="Valid Check 1")
    def check1(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.num_rows()).config(name="Has data").is_gt(0)

    @check(name="Valid Check 2")
    def check2(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.average("price")).config(name="Price check").is_positive()

    # Should not raise any errors during suite creation (graph is built in __init__)
    suite = VerificationSuite([check1, check2], db, "Valid Suite")
    assert suite is not None  # Suite created successfully


def test_suite_validation_on_build_graph_failure() -> None:
    """Test that validation fails with duplicate check names during suite creation."""
    db = InMemoryMetricDB()

    @check(name="Duplicate Name")
    def check1(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.num_rows()).config(name="Test").is_gt(0)

    @check(name="Duplicate Name")  # Same name!
    def check2(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.average("price")).config(name="Test").is_positive()

    # Should raise DQXError during suite creation (graph built in __init__)
    with pytest.raises(DQXError) as exc_info:
        VerificationSuite([check1, check2], db, "Invalid Suite")

    assert "validation failed" in str(exc_info.value).lower()
    assert "Duplicate check name" in str(exc_info.value)


def test_suite_validation_warnings_logged() -> None:
    """Test that validation warnings don't cause failure during suite creation."""
    db = InMemoryMetricDB()

    @check(name="Empty Check")
    def empty_check(mp: MetricProvider, ctx: Context) -> None:
        pass  # No assertions!

    # Should not raise error even with warnings (graph built in __init__)
    # The warning will be logged to stderr but won't cause failure
    suite = VerificationSuite([empty_check], db, "Test Suite")

    # If we got here without exception, the test passed
    # The warning was logged but didn't cause failure
    assert suite is not None


def test_validation_warnings_during_build_graph() -> None:
    """Test that validation warnings are logged but don't fail suite creation."""
    db = InMemoryMetricDB()

    @check(name="Empty Check")
    def empty_check(mp: MetricProvider, ctx: Context) -> None:
        pass  # No assertions!

    # Build graph (during __init__) should succeed despite warnings (warnings are only logged)
    # This won't raise an exception because warnings don't cause failure
    suite = VerificationSuite([empty_check], db, "Test Suite")

    # The test passes if no exception was raised
    # Warnings are logged but don't prevent execution
    assert suite is not None


def test_noop_assertion_always_succeeds() -> None:
    """Test that noop assertions always return SUCCESS."""
    import pyarrow as pa

    from dqx.datasource import DuckRelationDataSource

    db = InMemoryMetricDB()

    @check(name="Noop Test")
    def noop_test(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.average("value")).config(name="Collect average without validation").noop()

    suite = VerificationSuite([noop_test], db, "Noop Suite")

    # Create test data
    data = pa.table({"value": [1, 2, 3, 4, 5]})
    datasource = DuckRelationDataSource.from_arrow(data, "data")

    # Run suite
    key = ResultKey(yyyy_mm_dd=date.today(), tags={})
    suite.run([datasource], key)

    # Verify noop always succeeds
    results = suite.collect_results()
    assert len(results) == 1
    assert results[0].status == "PASSED"
