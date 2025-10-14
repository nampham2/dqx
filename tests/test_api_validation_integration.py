# Capture log output for testing

from datetime import date

import pytest

from dqx.api import Context, VerificationSuite, check
from dqx.common import DQXError, ResultKey
from dqx.orm.repositories import InMemoryMetricDB
from dqx.provider import MetricProvider


def test_suite_validation_on_collect_success() -> None:
    """Test that validation runs during collect with valid suite."""
    db = InMemoryMetricDB()

    @check(name="Valid Check 1")
    def check1(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.num_rows()).where(name="Has data").is_gt(0)

    @check(name="Valid Check 2")
    def check2(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.average("price")).where(name="Price check").is_positive()

    suite = VerificationSuite([check1, check2], db, "Valid Suite")

    # Should not raise any errors
    suite.collect(suite._context, ResultKey(date(2024, 1, 1), {"test": "true"}))


def test_suite_validation_on_collect_failure() -> None:
    """Test that validation fails with duplicate check names."""
    db = InMemoryMetricDB()

    @check(name="Duplicate Name")
    def check1(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.num_rows()).where(name="Test").is_gt(0)

    @check(name="Duplicate Name")  # Same name!
    def check2(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.average("price")).where(name="Test").is_positive()

    suite = VerificationSuite([check1, check2], db, "Invalid Suite")

    # Should raise DQXError with validation message
    with pytest.raises(DQXError) as exc_info:
        suite.collect(suite._context, ResultKey(date(2024, 1, 1), {"test": "true"}))

    assert "validation failed" in str(exc_info.value).lower()
    assert "Duplicate check name" in str(exc_info.value)


def test_suite_validation_warnings_logged() -> None:
    """Test that validation warnings don't cause failure."""
    db = InMemoryMetricDB()

    @check(name="Empty Check")
    def empty_check(mp: MetricProvider, ctx: Context) -> None:
        pass  # No assertions!

    suite = VerificationSuite([empty_check], db, "Test Suite")

    # Should not raise error even with warnings
    # The warning will be logged to stderr but won't cause failure
    suite.collect(suite._context, ResultKey(date(2024, 1, 1), {"test": "true"}))

    # If we got here without exception, the test passed
    # The warning was logged but didn't cause failure


def test_explicit_validate_method() -> None:
    """Test explicit validation method."""
    db = InMemoryMetricDB()

    @check(name="Empty Check")
    def empty_check(mp: MetricProvider, ctx: Context) -> None:
        pass  # No assertions!

    suite = VerificationSuite([empty_check], db, "Test Suite")

    # Call validate explicitly
    report = suite.validate()

    assert report.has_warnings()
    assert "Empty Check" in str(report)

    # Test structured output
    data = report.to_dict()
    assert data["summary"]["warning_count"] == 1
    assert data["warnings"][0]["rule"] == "empty_checks"
